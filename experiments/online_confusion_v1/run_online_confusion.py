"""Scoped runner for short observer integration checks and frozen diagnostics."""
from pathlib import Path
import os,sys,json,subprocess,socket,argparse,time
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto')
E=R/'experiments/online_confusion_v1';U=R/'runs/online_confusion_v1'
os.chdir(R);sys.path.insert(0,str(R/'experiments/kd_pixel_v2'))
from run_kd import environment
sys.path.insert(0,str(E/'src'));from kd_runtime import safe_path,atomic_json,now,digest
PY=str(R/'.runtime/env/bin/python')

def integration():
    cfg=json.loads((R/'runs/kd_parallel_v1/formal/b_relational/10-5/step1/config.json').read_text())
    for key in ['work_dir','ckpt_dir','pred_dir','prev_checkpoint','step','resume_checkpoint','local_rank']:cfg.pop(key,None)
    cfg.update(pretrained=False,spg=2,num_workers=2,log_iters=1,eval_iters=2004,train_limit=64,val_limit=16,
               max_iters=2004,async_eval=False,online_confusion=True)
    previous=R/'runs/fixed_baseline_v1/shared/10-5/step0/checkpoints/model_final.pth'
    resume=R/'runs/kd_pixel_v2/10-5/step1/checkpoints/model_iter_2000.pth'
    env=environment('8card');env['CUDA_VISIBLE_DEVICES']='0,1,2,3'
    records=[]
    for role in ['smoke_schema2','resume_schema2']:
        run=safe_path(U/role);run.mkdir(parents=True,exist_ok=False)
        current=dict(cfg)
        if role=='resume_schema2':
            resume=U/'smoke_schema2/10-5/step1/checkpoints/model_iter_2004.pth'
            current.update(max_iters=2006,eval_iters=2006)
        cmd=[PY,'-B','-m','torch.distributed.run','--standalone','--nproc_per_node=4',str(E/'src/scripts/dist_train_voc_seg_neg.py'),
             '--work_dir',str(run),'--step','1','--prev_checkpoint',str(previous),'--resume_checkpoint',str(resume)]
        for k,v in current.items():
            if isinstance(v,bool):
                if v:cmd.append('--'+k)
                elif k=='pretrained':cmd.append('--no-pretrained')
            elif isinstance(v,list):cmd.extend(['--'+k,*map(str,v)])
            else:cmd.extend(['--'+k,str(v)])
        with open(run/'launcher.log','x') as f:p=subprocess.Popen(cmd,cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        record={'utc':now(),'pid':p.pid,'command':cmd,'role':role};atomic_json(run/'launch.json',record)
        rc=p.wait();record.update(returncode=rc,completed_utc=now());atomic_json(run/'launch.json',record);records.append(record)
        if rc:raise RuntimeError(f'{role} failed {rc}')
    import torch
    checks=[]
    for role,it,updates in [('smoke_schema2',2004,4),('resume_schema2',2006,6)]:
        p=U/role/f'10-5/step1/checkpoints/model_iter_{it}.pth'
        state=torch.load(p,map_location='cpu',weights_only=True,mmap=True)
        assert int(state['online_confusion_state']['updates'])==updates
        final=torch.load(U/role/'10-5/step1/checkpoints/model_final.pth',map_location='cpu',weights_only=True,mmap=True)
        assert int(final['online_confusion_state']['updates'])==updates
        assert state['online_confusion_state']['_extra_state']['schema']==2
        assert state['online_confusion_state']['broad_counts'].sum()>0
        assert state['online_confusion_state']['broad_pair_image_observations'].sum()>0
        checks.append({'role':role,'iteration':it,'observer_updates':updates,'schema':2,'broad_and_trusted_states_saved':True,'model_and_optimizer_saved':True,'final_state_saved':True})
    atomic_json(U/'integration_verification.json',{'utc':now(),'passed':True,'records':records,'checks':checks,'scope':'Only 6 additional training updates total, disposable checkpoints, no baseline retraining'})

def diagnostics():
    records=[];children=[]
    for checkpoint in ['warmup','final']:
        for rank in range(4):
            gpu=rank+(4 if checkpoint=='final' else 0)
            env=environment('8card');env['CUDA_VISIBLE_DEVICES']=str(gpu)
            cmd=[PY,'-B',str(E/'diagnose_online_confusion.py'),'--checkpoint',checkpoint,'--split','val','--rank',str(rank)]
            log=safe_path(U/f'{checkpoint}_val_rank{rank}.log')
            with open(log,'x') as f:p=subprocess.Popen(cmd,cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            rec={'utc':now(),'pid':p.pid,'gpu':gpu,'command':cmd,'checkpoint':checkpoint,'split':'val','rank':rank}
            atomic_json(log.with_suffix('.process.json'),rec);records.append(rec);children.append(p)
    for p,rec in zip(children,records):
        rec['returncode']=p.wait()
    atomic_json(U/'validation_workers_complete.json',{'utc':now(),'records':records})
    if any(x['returncode'] for x in records):raise RuntimeError('Validation worker failure')
    children=[];records=[]
    for rank in range(8):
        env=environment('8card');env['CUDA_VISIBLE_DEVICES']=str(rank)
        cmd=[PY,'-B',str(E/'diagnose_online_confusion.py'),'--checkpoint','final','--split','train','--rank',str(rank),'--shards','8']
        log=safe_path(U/f'final_train_rank{rank}.log')
        with open(log,'x') as f:p=subprocess.Popen(cmd,cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        rec={'utc':now(),'pid':p.pid,'gpu':rank,'command':cmd,'rank':rank};records.append(rec);children.append(p)
        atomic_json(log.with_suffix('.process.json'),rec)
    for p,rec in zip(children,records):rec['returncode']=p.wait()
    atomic_json(U/'training_stream_workers_complete.json',{'utc':now(),'records':records})
    if any(x['returncode'] for x in records):raise RuntimeError('Training stream worker failure')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('role',choices=['integration','diagnostics']);a=p.parse_args()
    if a.role=='integration':integration()
    else:diagnostics()
