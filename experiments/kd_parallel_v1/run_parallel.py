"""Two distinct KD mechanisms sharing a warmup checkpoint, on partitioned 8card GPUs."""
from pathlib import Path
import os,sys,json,subprocess,signal,socket,argparse,fcntl,ast
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto');E=Path(__file__).resolve().parent
sys.path.insert(0,str(R/'experiments/kd_pixel_v2'));from run_kd import environment
sys.path.insert(0,str(R/'experiments/kd_pixel_v2/src'));from kd_runtime import safe_path,atomic_json,now,digest
PY=R/'.runtime/env/bin/python'

def sources(arm):
 s=E/arm/'src';return {str(p.relative_to(s)):digest(p) for p in s.rglob('*.py') if '__pycache__' not in p.parts}

def train(arm,gpus,smoke):
 run=safe_path(R/'runs/kd_parallel_v1'/('smoke' if smoke else 'formal')/arm);run.mkdir(parents=True,exist_ok=True)
 lock=open(run/'train.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 cfg=json.loads((R/'runs/kd_pixel_v2/study.json').read_text())['common_config']
 cfg.update(spg=8//len(gpus),num_workers=2)
 assert cfg['spg']*len(gpus)==8
 origin=json.loads((R/'experiments/kd_pixel_v2/origin.json').read_text())
 previous=Path(origin['step0']);assert digest(previous)==origin['step0_sha256']
 resume=R/'runs/kd_pixel_v2/10-5/step1/checkpoints/model_iter_2000.pth'
 code=sources(arm)
 if (run/'study.json').exists():raise RuntimeError('Refuse duplicate training')
 atomic_json(run/'study.json',{'arm':arm,'created_utc':now(),'gpus':gpus,'common_config':cfg,'global_batch':8,
  'source_sha256':code,'runner_sha256':digest(Path(__file__)),'step0':origin,
  'shared_warmup_checkpoint':str(resume),'shared_warmup_sha256':digest(resume),'smoke':smoke,
  'resume_caveat':'student and optimizer restored at2000; data sampler restarts under new hardware grouping'})
 env=environment('8card');env['CUDA_VISIBLE_DEVICES']=','.join(map(str,gpus))
 for step in [1,2]:
  if sources(arm)!=code:raise RuntimeError('Frozen source changed')
  current=dict(cfg)
  if smoke:
   current.update(max_iters=2004 if step==1 else 4,log_iters=1,eval_iters=2004 if step==1 else 4,val_limit=16,train_limit=64)
   if step==2:current.update(warmup_iters=1,loss_warmup_iters=1)
  cmd=[str(PY),'-B','-m','torch.distributed.run','--standalone',f'--nproc_per_node={len(gpus)}',str(E/arm/'src/scripts/dist_train_voc_seg_neg.py'),'--work_dir',str(run),'--step',str(step),'--prev_checkpoint',str(previous)]
  if step==1:cmd.extend(['--resume_checkpoint',str(resume)])
  for k,v in current.items():
   if isinstance(v,bool):
    if v:cmd.append('--'+k)
    elif k=='pretrained':cmd.append('--no-pretrained')
   elif isinstance(v,list):cmd.extend(['--'+k,*map(str,v)])
   else:cmd.extend(['--'+k,str(v)])
  stage=safe_path(run/f'10-5/step{step}');stage.mkdir(parents=True,exist_ok=True)
  with open(stage/'launcher.log','x') as log:
   child=subprocess.Popen(cmd,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   def stop(signum,frame):os.killpg(child.pid,signal.SIGTERM);raise KeyboardInterrupt
   signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
   record={'pid':child.pid,'starttime':Path(f'/proc/{child.pid}/stat').read_text().split()[21],'command':cmd,'gpus':gpus,'host':socket.gethostname(),'utc':now()}
   atomic_json(stage/'launch.json',record)
   rc=child.wait()
  atomic_json(stage/'launch.json',{**record,'returncode':rc,'end_utc':now()})
  if rc:raise RuntimeError(f'{arm} step{step} failed: {rc}')
  previous=stage/'checkpoints/model_final.pth';assert previous.exists()
  atomic_json(stage/'training_complete.json',{'utc':now(),'checkpoint_sha256':digest(previous),'returncode':rc})
 atomic_json(run/'training_complete.json',{'utc':now(),'stages':2,'status':'training_complete'})

def evaluate(arm,gpus,smoke):
 run=safe_path(R/'runs/kd_parallel_v1'/('smoke' if smoke else 'formal')/arm);run.mkdir(parents=True,exist_ok=True)
 env=environment('8card');env['CUDA_VISIBLE_DEVICES']=','.join(map(str,gpus))
 # Two shards per arm bound inference interference when both training jobs run.
 children=[]
 for rank in range(2):
  cmd=[str(PY),'-B',str(E/arm/'src/evaluate_kd.py'),'--run',str(run),'--rank',str(rank),'--shards','2']
  with open(run/f'evaluator_rank{rank}.log','a') as log:
   p=subprocess.Popen(cmd,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  atomic_json(run/f'evaluator_rank{rank}.process.json',{'pid':p.pid,'starttime':Path(f'/proc/{p.pid}/stat').read_text().split()[21],'command':cmd,'utc':now()});children.append(p)
 codes=[p.wait() for p in children]
 atomic_json(run/'evaluation_workers_complete.json',{'returncodes':codes,'utc':now()})
 if any(codes):raise RuntimeError(codes)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('role',choices=['train','eval']);p.add_argument('arm',choices=['a_sigmoid','b_relational']);p.add_argument('--gpus',required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
 gpus=[int(x) for x in a.gpus.split(',')];assert gpus and len(set(gpus))==len(gpus) and all(0<=x<8 for x in gpus)
 if a.role=='train':train(a.arm,gpus,a.smoke)
 else:evaluate(a.arm,gpus,a.smoke)
