"""Short head-only refinement of the actual best checkpoint, with cached weak evidence."""
from pathlib import Path
import sys,os,json,math,time,copy,argparse
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto');E=R/'experiments/pair_preserving_kd_v1';U=R/'runs/pair_preserving_kd_v1'
C=R/'runs/prototype_sep_v1/formal/c_new_anchor';S=R/'experiments/prototype_sep_v1/c_new_anchor/src'
sys.path.insert(0,str(S))
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from model.model_seg_neg import network
from model.online_directed_confusion import OnlineDirectedConfusion
from model.geometry_pair_selector import GeometryPairSelector
from model.pixel_kd import pixel_kd_loss
from pair_targets import correction_targets,background_directions
from pair_loss import pair_preserving_loss
from kd_runtime import safe_path,atomic_json,now,digest

def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');args=p.parse_args()
    rank=int(os.environ['LOCAL_RANK']);assert int(os.environ['WORLD_SIZE'])==8
    torch.cuda.set_device(rank);dist.init_process_group('nccl')
    torch.set_num_threads(1);torch.manual_seed(0)
    torch.set_float32_matmul_precision('highest');torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    run=safe_path(U/('smoke' if args.smoke else 'formal'));stage=safe_path(run/'10-5/step2')
    study=json.loads(safe_path(run/'study.json').read_text());cfg=study['config'];steps=study['updates']
    assert all(digest(E/k)==v for k,v in study['sources'].items())
    receipts=[json.loads(safe_path(U/f'cache_receipt_rank{i}.json').read_text()) for i in range(8)]
    names=[n for rec in receipts for n in rec['images']]
    assert len(names)==len(set(names))==2145 and all(rec['passed'] for rec in receipts)
    rec=receipts[rank];cache_path=safe_path(rec['cache']);assert digest(cache_path)==rec['cache_sha256']
    data=torch.load(cache_path,map_location='cpu',weights_only=True,mmap=True)
    assert data['names']==rec['images']
    data={k:v.cuda() for k,v in data.items() if torch.is_tensor(v)};n=len(data['feature'])
    source=safe_path(C/'10-5/step2/checkpoints/model_final.pth');assert digest(source)==study['resume_sha256']
    saved=torch.load(source,map_location='cpu',weights_only=True,mmap=True);assert saved['iteration']==8000
    model=network(backbone=cfg['backbone'],num_classes=21,classes_list=[11,5,5],pretrained=False,
        init_momentum=cfg['momentum'],aux_layer=cfg['aux_layer'])
    model.load_state_dict(saved['model_state'],strict=True);model.eval().requires_grad_(False)
    model.decoder.conv8.cuda().requires_grad_(True)
    frozen_head=copy.deepcopy(model.decoder.conv8).eval().requires_grad_(False)
    head_names=[name for name,p in model.named_parameters() if p.requires_grad]
    assert head_names==['decoder.conv8.0.weight','decoder.conv8.1.weight','decoder.conv8.2.weight']
    groups=model.get_param_groups()
    opt=torch.optim.AdamW([{'params':group,'lr':cfg['lr']*(1 if i<2 else 10),'weight_decay':cfg['wt_decay']}
        for i,group in enumerate(groups)],lr=cfg['lr'],betas=cfg['betas'],weight_decay=cfg['wt_decay'])
    opt.load_state_dict(saved['optimizer_state'])
    for p in model.decoder.conv8.parameters():assert int(opt.state[p]['step'])==8000
    head=DDP(model.decoder.conv8,device_ids=[rank],broadcast_buffers=False)
    obs=OnlineDirectedConfusion(21,momentum=cfg['confusion_momentum'],high_threshold=cfg['high_thre'],
        low_threshold=cfg['low_thre'],stage=2).cuda();obs.load_state_dict(saved['online_confusion_state'],strict=True)
    selector=GeometryPairSelector(21,old_classes=15,stage=2,refresh_interval=cfg['pair_refresh_interval'],
        min_row_images=cfg['pair_min_row_images'],min_pair_images=cfg['pair_min_pair_images'],min_rate=cfg['pair_min_rate'],
        min_updates=cfg['pair_min_updates'],ramp_updates=cfg['pair_ramp_updates'],max_stale_updates=cfg['pair_max_stale_updates']).cuda()
    selector.load_state_dict(saved['geometry_selector_state'],strict=True)
    gen=torch.Generator(device='cuda');gen.manual_seed(rank)
    order=torch.randperm(n,device='cuda',generator=gen);cursor=0;epoch=0
    totals={'updates':0,'new_pair_pixels':0,'bg_pair_pixels':0,'absent_pixels':0,'active_updates':0}
    started=time.monotonic();log=open(safe_path(run/'training.jsonl'),'x') if rank==0 else None
    for step in range(1,steps+1):
        if cursor+8>n:
            remaining=order[cursor:];order=torch.randperm(n,device='cuda',generator=gen);need=8-len(remaining)
            ids=torch.cat([remaining,order[:need]]);cursor=need;epoch+=1
        else:ids=order[cursor:cursor+8];cursor+=8
        b={k:v[ids] for k,v in data.items()};h,w=b['feature'].shape[-2:]
        boxes=torch.tensor([[0,h,0,w]]*len(ids))
        with torch.no_grad():
            reference=frozen_head(b['feature'])
            pair=selector.update(obs,8000+step);bg=background_directions(obs)
            z=correction_targets(reference,b['old_logits'],b['par'],b['cams'],boxes,b['tags'],pair,bg,cfg['kd_temperature'],cfg['high_thre'])
        output=head(b['feature'])
        protected,stats=pair_preserving_loss(output,z,cfg['kd_temperature'])
        kd,kd_stats=pixel_kd_loss(output,b['old_logits'],b['par'],b['cams'],boxes,cfg['kd_temperature'])
        loss=protected+.1*kd
        assert torch.isfinite(loss)
        opt.zero_grad(set_to_none=True);loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.decoder.conv8.parameters())
        lr=cfg['lr']*(.5+.5*math.cos(math.pi*(step-1)/300))
        for group in opt.param_groups:group['lr']=lr
        opt.step()
        obs.update(output.detach(),b['par'],b['cams'],boxes,synchronize=True)
        totals['updates']+=1;totals['active_updates']+=int(stats['active_pixels']>0)
        for key in ['new_pair_pixels','bg_pair_pixels','absent_pixels']:totals[key]+=stats[key]
        if rank==0:
            row={'update':step,'iteration':8000+step,'total_loss':float(loss.detach()),'old_conditional_kd':float(kd.detach()),
                 **stats,'lr':lr,'pair_targets':pair.tolist(),'background_targets':bg.nonzero().flatten().tolist(),
                 'elapsed_seconds':time.monotonic()-started,'totals':dict(totals)}
            log.write(json.dumps(row)+'\n');log.flush()
            if step%25==0 or step==steps:print(json.dumps(row),flush=True)
    if log:log.close()
    # Byte-exact frozen state check, including all buffers and existing prototypes.
    changed=[]
    for key,value in model.state_dict().items():
        if not torch.equal(value.detach().cpu(),saved['model_state'][key]):changed.append(key)
    assert sorted(changed)==sorted(head_names),changed
    frozen_optimizer_exact=True
    actual=opt.state_dict()
    for parameters,group,old_group in zip(groups,actual['param_groups'],saved['optimizer_state']['param_groups']):
        for parameter,ident,old_ident in zip(parameters,group['params'],old_group['params']):
            before=saved['optimizer_state']['state'].get(old_ident);after=actual['state'].get(ident)
            if not parameter.requires_grad:
                assert (before is None)==(after is None)
                if before is None:continue
                for key,value in before.items():
                    assert torch.equal(value.cpu(),after[key].cpu()) if torch.is_tensor(value) else value==after[key]
    for p in model.decoder.conv8.parameters():assert int(opt.state[p]['step'])==8000+steps
    for name,p in model.named_parameters():
        if name not in head_names:assert p.grad is None
    assert int(obs.updates)==6000+steps and int(obs.seen_images)==48000+steps*64
    # Actual DDP parameters must agree across all ranks before publication.
    flat=torch.cat([p.detach().flatten() for p in model.decoder.conv8.parameters()])
    root=flat.clone();dist.broadcast(root,0);assert torch.equal(flat,root)
    dist.barrier()
    if rank==0:
        ck=safe_path(stage/'checkpoints/model_final.pth');ck.parent.mkdir(parents=True,exist_ok=True)
        torch.save({'iteration':8000+steps,'model_state':{k:v.detach().cpu() for k,v in model.state_dict().items()},
            'optimizer_state':actual,'online_confusion_state':obs.state_dict(),'geometry_selector_state':selector.state_dict(),
            'refinement_totals':totals,'refinement_source_sha256':study['sources']},ck)
        sha=digest(ck)
        atomic_json(run/'training_complete.json',{'status':'complete','utc':now(),'checkpoint':str(ck),'checkpoint_sha256':sha,
            'steps':steps,'all_frozen_tensors_exact':True,'changed_model_tensors':changed,'head_optimizer_steps':8000+steps,
            'frozen_optimizer_tensors_exact':frozen_optimizer_exact,'all_rank_head_parameters_exact':True,
            'observer_updates':int(obs.updates),'observer_seen_images':int(obs.seen_images),'totals':totals,'elapsed_seconds':time.monotonic()-started})
        if not args.smoke:
            atomic_json(run/f'eval_queue/step2_iter{8000+steps}.json',{'checkpoint':str(ck),'checkpoint_sha256':sha,
                'config':cfg,'step':2,'iteration':8000+steps,'published_utc':now()})
    dist.barrier();dist.destroy_process_group()

if __name__=='__main__':main()
