from pathlib import Path
import sys,os,json,math,time,argparse
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto');E=R/'experiments/ald_calibration_v5';U=R/'runs/ald_calibration_v5';DATA=R/'runs/ald_calibration_v4'
S=R/'experiments/prototype_sep_v1/c_new_anchor/src';sys.path.insert(0,str(S))
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from residual import SemanticResidual
from losses import loss as calibrated_loss
from kd_runtime import safe_path,atomic_json,digest,now

def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');args=p.parse_args()
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);dist.init_process_group('nccl');assert dist.get_world_size()==8
    torch.set_num_threads(1);torch.manual_seed(0);torch.set_float32_matmul_precision('highest');torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True
    run=safe_path(U/('smoke' if args.smoke else 'formal'));study=json.loads((run/'study.json').read_text());cfg=study['config'];steps=study['updates']
    assert all(digest(E/k)==v for k,v in study['sources'].items())
    rec=json.loads(safe_path(DATA/f'cache_receipt_rank{rank}.json').read_text());assert rec['passed'] and digest(safe_path(rec['cache']))==rec['cache_sha256']
    proposal=torch.load(rec['cache'],map_location='cpu',weights_only=True,mmap=True)
    r1=json.loads(safe_path(R/'runs/ald_calibration_v1'/f'cache_receipt_rank{rank}.json').read_text());assert digest(safe_path(r1['cache']))==r1['cache_sha256']==rec['ALD_v1_cache_sha256']
    ev=torch.load(r1['cache'],map_location='cpu',weights_only=True,mmap=True)
    feature_path=safe_path(R/'runs/pair_preserving_kd_v1'/f'cache_rank{rank}.pth');assert digest(feature_path)==r1['prior_cache_sha256']
    features=torch.load(feature_path,map_location='cpu',weights_only=True,mmap=True);assert proposal['names']==ev['names']==features['names']
    source=safe_path(study['resume']);assert digest(source)==study['resume_sha256']
    saved=torch.load(source,map_location='cpu',weights_only=True,mmap=True);assert saved['iteration']==8300
    calibrator_path=safe_path(R/'runs/ald_calibration_v2/formal/10-5/step2/checkpoints/model_final.pth');assert digest(calibrator_path)=='0d6fcb1e8b83a3a79155bb480f33b11cf8895cef068968c42651d041ad93333c'
    cal=torch.load(calibrator_path,map_location='cpu',weights_only=True,mmap=True)
    weights=torch.cat([saved['model_state'][f'decoder.conv8.{i}.weight'] for i in range(3)]).cuda()
    model=SemanticResidual(weights).cuda()
    dist.broadcast(model.basis,0);dist.broadcast(model.reference_weights,0)
    assert float((weights.flatten(1).double()@model.basis.double()).abs().max())<1e-6
    with torch.no_grad():
        feat=features['feature'].cuda();reference=F.conv2d(feat,weights);projected=model.features(feat)
        assert torch.equal(model(reference,projected),reference);del feat
    data={'reference':reference,'features':projected,'proposal':proposal['labels'].cuda(),'selected':(proposal['background']|proposal['new_teacher_old']).cuda(),
        'old_logits':features['old_logits'].cuda(),'par':ev['par'].cuda(),'cams':ev['cams'].cuda(),'trusted_old':ev['trusted_old'].cuda()}
    opt=torch.optim.AdamW([model.delta],lr=cfg['lr'],betas=cfg['betas'],weight_decay=cfg['wt_decay'])
    ddp=DDP(model,device_ids=[rank],broadcast_buffers=False);n=len(reference);gen=torch.Generator(device='cuda').manual_seed(rank)
    order=torch.randperm(n,device='cuda',generator=gen);cursor=0;started=time.monotonic();log=open(safe_path(run/'training.jsonl'),'x') if rank==0 else None
    for step in range(1,steps+1):
        if cursor+8>n:
            rest=order[cursor:];order=torch.randperm(n,device='cuda',generator=gen);need=8-len(rest);ids=torch.cat([rest,order[:need]]);cursor=need
        else:ids=order[cursor:cursor+8];cursor+=8
        b={k:v[ids] for k,v in data.items()};output=ddp(b['reference'],b['features'])
        loss,stats=calibrated_loss(output,b['reference'],b['proposal'],b['selected'],b['old_logits'],b['par'],b['cams'],b['trusted_old'])
        assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();assert torch.isfinite(model.delta.grad).all()
        lr=cfg['lr']*(.5+.5*math.cos(math.pi*(step-1)/300));opt.param_groups[0]['lr']=lr;opt.step()
        if rank==0:
            row={'update':step,'iteration':8300+step,'loss':float(loss.detach()),'lr':lr,**stats,'elapsed_seconds':time.monotonic()-started}
            log.write(json.dumps(row)+'\n');log.flush()
            if step%25==0 or step==steps:print(json.dumps(row),flush=True)
    if log:log.close()
    assert int(opt.state[model.delta]['step'])==steps
    flat=model.delta.detach().flatten();root=flat.clone();dist.broadcast(root,0);assert torch.equal(root,flat)
    folded=model.folded_new_weight().detach();change=(folded-weights[16:]).flatten(1)
    error=float((change.double()@weights.flatten(1).double().T).abs().max());assert error<1e-6,error
    with torch.no_grad():
        feat=features['feature'][:8].cuda();direct=model(F.conv2d(feat,weights),model.features(feat))
        full=torch.cat([weights[:16],folded]);reconstructed=F.conv2d(feat,full)
        ferr=float((direct-reconstructed).abs().max());assert ferr<1e-4,ferr
    dist.barrier()
    if rank==0:
        model_state=dict(saved['model_state']);model_state['decoder.conv8.2.weight']=folded.cpu()
        for key,value in cal['model_state'].items():
            if key.startswith(('classifier.','aux_classifier.')):model_state[key]=value
        changed=[k for k,v in model_state.items() if not torch.equal(v,saved['model_state'][k])]
        assert len(changed)==7 and all(k=='decoder.conv8.2.weight' or k.startswith(('classifier.','aux_classifier.')) for k in changed)
        optimizer_state={**saved['optimizer_state'],'state':dict(saved['optimizer_state']['state'])}
        for ident in range(152,158):optimizer_state['state'][ident]=cal['optimizer_state']['state'][ident]
        # Native new-head Adam moments cannot represent the residual-coordinate
        # optimizer. Reset that one state on generic folded-head continuation;
        # preserve a complete residual optimizer separately for exact resume.
        del optimizer_state['state'][170]
        result={**saved,'model_state':model_state,'optimizer_state':optimizer_state,'iteration':8300+steps,
            'ALD_source_sha256':study['sources'],'ALD_residual_state':{k:v.cpu() for k,v in model.state_dict().items()},'ALD_residual_optimizer':opt.state_dict(),
            'ALD_metadata':{'classifiers_from_v2':True,'old16_logits_exact':True,'all21_semantic_directions_preserved':True,'folded_new_head_optimizer_reset':True,
                'exact_resume':'Use ALD_residual_state and ALD_residual_optimizer in residual coordinates; generic full-head continuation initializes only new-head Adam state.'}}
        stage=safe_path(run/'10-5/step2');ck=safe_path(stage/'checkpoints/model_final.pth');ck.parent.mkdir(parents=True,exist_ok=True);torch.save(result,ck)
        atomic_json(run/'training_complete.json',{'status':'complete','utc':now(),'checkpoint':str(ck),'checkpoint_sha256':digest(ck),'steps':steps,'changed_model_tensors':changed,
            'all_frozen_tensors_exact':True,'all_rank_parameters_exact':True,'old16_logits_exact':True,'semantic_constraint_max_error':error,'folded_forward_error':ferr,
            'residual_rank':model.rank,'residual_parameters':model.delta.numel(),'optimizer_semantics':result['ALD_metadata'],'elapsed_seconds':time.monotonic()-started})
        if not args.smoke:atomic_json(run/f'eval_queue/step2_iter{8300+steps}.json',{'checkpoint':str(ck),'checkpoint_sha256':digest(ck),'config':cfg,'step':2,'iteration':8300+steps,'published_utc':now()})
    dist.barrier();dist.destroy_process_group()

if __name__=='__main__':main()
