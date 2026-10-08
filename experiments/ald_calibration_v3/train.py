from pathlib import Path
import sys,os,json,math,time,copy,argparse
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto');E=R/'experiments/ald_calibration_v3';U=R/'runs/ald_calibration_v3'
S=R/'experiments/prototype_sep_v1/c_new_anchor/src'
sys.path.insert(0,str(S))
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from model.model_seg_neg import network
from losses import loss as calibrated_loss
from kd_runtime import safe_path,atomic_json,now,digest

class Heads(torch.nn.Module):
    def __init__(self,model):
        super().__init__();self.seg=model.decoder.conv8;self.cls=model.classifier;self.aux=model.aux_classifier
    def forward(self,feature,pooled,aux):
        return self.seg(feature),self.cls(pooled.flatten(0,1)).flatten(1),self.aux(aux.flatten(0,1)).flatten(1)

def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');args=p.parse_args()
    rank=int(os.environ['LOCAL_RANK']);assert int(os.environ['WORLD_SIZE'])==8
    torch.cuda.set_device(rank);dist.init_process_group('nccl');torch.set_num_threads(1);torch.manual_seed(0)
    torch.set_float32_matmul_precision('highest');torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    run=safe_path(U/('smoke' if args.smoke else 'formal'));stage=run/'10-5/step2'
    study=json.loads((run/'study.json').read_text());cfg=study['config'];steps=study['updates']
    assert all(digest(E/k)==v for k,v in study['sources'].items())
    receipts=[json.loads(safe_path(U/f'cache_receipt_rank{i}.json').read_text()) for i in range(8)]
    names=[n for r in receipts for n in r['images']];assert len(names)==len(set(names))==2145
    assert all(r['passed'] for r in receipts)
    rec=receipts[rank];assert digest(safe_path(rec['cache']))==rec['cache_sha256']
    absence=torch.load(rec['cache'],map_location='cpu',weights_only=True,mmap=True)
    prev=json.loads(safe_path(R/'runs/ald_calibration_v1'/f'cache_receipt_rank{rank}.json').read_text())
    assert prev['cache_sha256']==rec['ALD_v1_cache_sha256']==digest(safe_path(prev['cache']))
    evidence=torch.load(prev['cache'],map_location='cpu',weights_only=True,mmap=True)
    assert evidence['names']==absence['names'];evidence['proposal']=absence['labels'];evidence['selected']=absence['background']|absence['new_teacher_old']
    prior=R/'runs/pair_preserving_kd_v1'/f'cache_rank{rank}.pth';assert digest(prior)==prev['prior_cache_sha256']
    features=torch.load(prior,map_location='cpu',weights_only=True,mmap=True)
    assert features['names']==evidence['names']==rec['images']
    data={k:v.cuda() for k,v in evidence.items() if torch.is_tensor(v)}
    for key in ['feature','old_logits','tags']:data[key]=features[key].cuda()
    n=len(data['feature']);source=safe_path(study['resume']);assert digest(source)==study['resume_sha256']
    saved=torch.load(source,map_location='cpu',weights_only=True,mmap=True);assert saved['iteration']==8300
    calibrator_path=safe_path(R/'runs/ald_calibration_v2/formal/10-5/step2/checkpoints/model_final.pth')
    assert digest(calibrator_path)=='0d6fcb1e8b83a3a79155bb480f33b11cf8895cef068968c42651d041ad93333c'
    calibrator=torch.load(calibrator_path,map_location='cpu',weights_only=True,mmap=True)
    saved={**saved,'model_state':dict(saved['model_state']),'optimizer_state':{**saved['optimizer_state'],'state':dict(saved['optimizer_state']['state'])}}
    for key,value in calibrator['model_state'].items():
        if key.startswith(('classifier.','aux_classifier.')):saved['model_state'][key]=value
    for ident in range(152,158):saved['optimizer_state']['state'][ident]=calibrator['optimizer_state']['state'][ident]
    model=network(backbone=cfg['backbone'],num_classes=21,classes_list=[11,5,5],pretrained=False,init_momentum=cfg['momentum'],aux_layer=cfg['aux_layer'])
    model.load_state_dict(saved['model_state'],strict=True);model.eval().requires_grad_(False)
    heads=model.decoder.conv8.cuda().requires_grad_(True);reference=copy.deepcopy(heads).eval().requires_grad_(False)
    names_train=[k for k,p in model.named_parameters() if p.requires_grad];assert len(names_train)==3,names_train
    groups=model.get_param_groups()
    opt=torch.optim.AdamW([{'params':g,'lr':cfg['lr']*(1 if i<2 else 10),'weight_decay':cfg['wt_decay']} for i,g in enumerate(groups)],lr=cfg['lr'],betas=cfg['betas'],weight_decay=cfg['wt_decay'])
    opt.load_state_dict(saved['optimizer_state'])
    start_steps={k:int(opt.state[p]['step']) for k,p in model.named_parameters() if p.requires_grad}
    assert set(start_steps.values())=={8300}
    ddp=DDP(heads,device_ids=[rank],broadcast_buffers=False)
    gen=torch.Generator(device='cuda').manual_seed(rank);order=torch.randperm(n,device='cuda',generator=gen);cursor=0
    started=time.monotonic();log=open(safe_path(run/'training.jsonl'),'x') if rank==0 else None
    for step in range(1,steps+1):
        if cursor+8>n:
            remainder=order[cursor:];order=torch.randperm(n,device='cuda',generator=gen);need=8-len(remainder);ids=torch.cat([remainder,order[:need]]);cursor=need
        else:ids=order[cursor:cursor+8];cursor+=8
        b={k:v[ids] for k,v in data.items()}
        with torch.no_grad():ref=reference(b['feature'])
        output=ddp(b['feature'])
        loss,ss=calibrated_loss(output,ref,b['proposal'],b['selected'],b['old_logits'],b['par'],b['cams'],b['trusted_old'])
        assert torch.isfinite(loss)
        opt.zero_grad(set_to_none=True);loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in heads.parameters())
        lr=cfg['lr']*(.5+.5*math.cos(math.pi*(step-1)/300))
        for g in opt.param_groups:g['lr']=lr
        opt.step()
        if rank==0:
            row={'update':step,'iteration':8300+step,'loss':float(loss.detach()),'lr':lr,**ss,'elapsed_seconds':time.monotonic()-started}
            log.write(json.dumps(row)+'\n');log.flush()
            if step%25==0 or step==steps:print(json.dumps(row),flush=True)
    if log:log.close()
    changed=[k for k,v in model.state_dict().items() if not torch.equal(v.cpu(),saved['model_state'][k])]
    assert set(changed)==set(names_train),changed
    actual=opt.state_dict()
    for group,g,oldg in zip(groups,actual['param_groups'],saved['optimizer_state']['param_groups']):
        for param,ident,oldident in zip(group,g['params'],oldg['params']):
            before=saved['optimizer_state']['state'].get(oldident);after=actual['state'].get(ident)
            if not param.requires_grad:
                assert (before is None)==(after is None)
                if before:
                    for k,v in before.items():assert torch.equal(v.cpu(),after[k].cpu()) if torch.is_tensor(v) else v==after[k]
                assert param.grad is None
    for k,p in model.named_parameters():
        if p.requires_grad:assert int(opt.state[p]['step'])==start_steps[k]+steps
    flat=torch.cat([p.detach().flatten() for p in heads.parameters()]);root=flat.clone();dist.broadcast(root,0);assert torch.equal(root,flat)
    dist.barrier()
    if rank==0:
        ck=safe_path(stage/'checkpoints/model_final.pth');ck.parent.mkdir(parents=True,exist_ok=True)
        result={**saved,'iteration':8300+steps,'model_state':{k:v.detach().cpu() for k,v in model.state_dict().items()},'optimizer_state':actual,'ALD_source_sha256':study['sources'],
            'ALD_metadata':{'frozen_evidence':True,'unknown_not_negative':True,'old_KD_explicit_trusted_gate':True,'past_pair_selector_not_applied':True,'confusion_unchanged':True,'classifier_inherited_from_ALD_v2':str(calibrator_path),'correction_domains':['background','new_teacher_old']}}
        torch.save(result,ck)
        atomic_json(run/'training_complete.json',{'status':'complete','utc':now(),'checkpoint':str(ck),'checkpoint_sha256':digest(ck),'steps':steps,
            'changed_model_tensors':changed,'all_frozen_tensors_exact':True,'frozen_optimizer_tensors_exact':True,'all_rank_parameters_exact':True,
            'initial_optimizer_steps':start_steps,'elapsed_seconds':time.monotonic()-started})
        if not args.smoke:atomic_json(run/f'eval_queue/step2_iter{8300+steps}.json',{'checkpoint':str(ck),'checkpoint_sha256':digest(ck),'config':cfg,'step':2,'iteration':8300+steps,'published_utc':now()})
    dist.barrier();dist.destroy_process_group()

if __name__=='__main__':main()
