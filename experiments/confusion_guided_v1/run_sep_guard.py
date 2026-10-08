from pathlib import Path
import os,sys,json,subprocess,argparse,signal,fcntl,time
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto');E=R/'experiments/confusion_guided_v1';SRC=E/'c_newaware_sep/src'
os.chdir(R);sys.path.insert(0,str(R/'experiments/kd_pixel_v2'));from run_kd import environment
sys.path.insert(0,str(SRC));from kd_runtime import safe_path,atomic_json,now,digest
PY=R/'.runtime/env/bin/python'
def source_hashes():return {str(p.relative_to(SRC)):digest(p) for p in SRC.rglob('*.py') if '__pycache__' not in p.parts}
def train(smoke):
 run=safe_path(R/'runs/confusion_guided_v1'/('guard_smoke' if smoke else 'formal')/'c_newaware_sep');run.mkdir(parents=True,exist_ok=True)
 lock=open(run/'train.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (run/'study.json').exists():raise RuntimeError('Refuse duplicate')
 parent=R/'runs/confusion_guided_v1/formal/a_sep';teacher=parent/'10-5/step1/checkpoints/model_final.pth';resume=parent/'10-5/step2/checkpoints/model_iter_2000.pth'
 cfg=json.loads((parent/'10-5/step2/config.json').read_text())
 for key in ['local_rank','step','work_dir','prev_checkpoint','resume_checkpoint','ckpt_dir','pred_dir']:cfg.pop(key,None)
 cfg.update(spg=1,sep_new_cam_guard=True,pair_mode='sep')
 if smoke:cfg.update(max_iters=2004,eval_iters=2004,log_iters=1,val_limit=16,train_limit=64)
 code=source_hashes()
 atomic_json(run/'study.json',{'arm':'c_newaware_sep','utc':now(),'gpus':list(range(8)),'global_batch':8,'common_config':cfg,'source_sha256':code,'runner_sha256':digest(Path(__file__)),
  'inherited_stage1':str(parent/'10-5/step1'),'teacher':str(teacher),'teacher_sha256':digest(teacher),
  'shared_stage2_warmup':str(resume),'shared_stage2_warmup_sha256':digest(resume),'start_iteration':2000,'smoke':smoke,
  'protocol':str(E/'guard_protocol.json'),'scope':'Stage2 only targeted old SEP guard; no baseline/step0/step1/warmup re-training'})
 env=environment('8card');env['CUDA_VISIBLE_DEVICES']='0,1,2,3,4,5,6,7'
 cmd=[str(PY),'-B','-m','torch.distributed.run','--standalone','--nproc_per_node=8',str(SRC/'scripts/dist_train_voc_seg_neg.py'),
  '--work_dir',str(run),'--step','2','--prev_checkpoint',str(teacher),'--resume_checkpoint',str(resume)]
 for k,v in cfg.items():
  if v is None:continue
  if isinstance(v,bool):
   if v:cmd.append('--'+k)
   elif k=='pretrained':cmd.append('--no-pretrained')
  elif isinstance(v,list):cmd.extend(['--'+k,*map(str,v)])
  else:cmd.extend(['--'+k,str(v)])
 stage=run/'10-5/step2';stage.mkdir(parents=True,exist_ok=True)
 with open(stage/'launcher.log','x') as log:child=subprocess.Popen(cmd,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 def stop(signum,frame):os.killpg(child.pid,signal.SIGTERM);raise KeyboardInterrupt
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 record={'pid':child.pid,'starttime':Path(f'/proc/{child.pid}/stat').read_text().split()[21],'command':cmd,'gpus':list(range(8)),'utc':now()}
 atomic_json(stage/'launch.json',record);rc=child.wait();atomic_json(stage/'launch.json',{**record,'returncode':rc,'end_utc':now()})
 if rc:raise RuntimeError(rc)
 assert code==source_hashes()
 atomic_json(stage/'training_complete.json',{'utc':now(),'returncode':0,'checkpoint_sha256':digest(stage/'checkpoints/model_final.pth')})
 atomic_json(run/'training_complete.json',{'utc':now(),'stages':1,'status':'training_complete','inherited_stage1':str(parent/'10-5/step1')})
def evaluate(smoke):
 run=safe_path(R/'runs/confusion_guided_v1'/('guard_smoke' if smoke else 'formal')/'c_newaware_sep');run.mkdir(parents=True,exist_ok=True)
 env=environment('8card');env['CUDA_VISIBLE_DEVICES']='0,1,2,3,4,5,6,7';children=[]
 for rank in range(4):
  cmd=[str(PY),'-B',str(SRC/'evaluate_kd.py'),'--run',str(run),'--rank',str(rank),'--shards','4']
  with open(run/f'evaluator_rank{rank}.log','a') as log:p=subprocess.Popen(cmd,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  atomic_json(run/f'evaluator_rank{rank}.process.json',{'pid':p.pid,'starttime':Path(f'/proc/{p.pid}/stat').read_text().split()[21],'command':cmd,'utc':now()});children.append(p)
 failure=None
 while True:
  codes=[p.poll() for p in children]
  if any(code not in (None,0) for code in codes):failure=f'evaluator failure {codes}';break
  if all(code==0 for code in codes):break
  if (run/'training_failed.json').exists():failure='training failed';break
  time.sleep(2)
 if failure:
  for p in children:
   if p.poll() is None:
    try:os.killpg(p.pid,signal.SIGTERM)
    except ProcessLookupError:pass
  for p in children:
   try:p.wait(timeout=15)
   except subprocess.TimeoutExpired:
    try:os.killpg(p.pid,signal.SIGKILL)
    except ProcessLookupError:pass
    p.wait()
 atomic_json(run/'evaluation_workers_complete.json',{'utc':now(),'returncodes':[p.poll() for p in children],'status':'failed' if failure else 'complete','failure':failure})
 if failure:raise RuntimeError(failure)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('role',choices=['train','eval']);p.add_argument('--smoke',action='store_true');a=p.parse_args()
 if a.role=='train':
  try:train(a.smoke)
  except BaseException as exc:
   run=R/'runs/confusion_guided_v1'/('guard_smoke' if a.smoke else 'formal')/'c_newaware_sep';atomic_json(run/'training_failed.json',{'utc':now(),'error':repr(exc)});raise
 else:evaluate(a.smoke)
