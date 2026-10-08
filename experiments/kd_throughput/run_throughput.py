from pathlib import Path
import sys,os,json,subprocess,signal,time
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto');E=R/'experiments/kd_throughput';os.chdir(R)
sys.path.insert(0,str(R/'experiments/kd_pixel_v2'));from run_kd import environment
sys.path.insert(0,str(E/'src'));from kd_runtime import atomic_json,safe_path,now
U=R/'runs/kd_pixel_v2'
# Wait for the already-running common warmup checkpoint, then stop only our study.
while not (U/'evaluations/step1_iter2000/result.json').exists():
 d=json.loads((U/'coordinator_train.process.json').read_text());p=Path(f'/proc/{d["pid"]}/stat')
 if not p.exists() or p.read_text().split()[21]!=d['starttime']:raise RuntimeError('Training no longer live before checkpoint')
 time.sleep(5)
for f in [U/'coordinator_train.process.json',*U.glob('evaluator_rank*.process.json'),U/'coordinator_eval.process.json']:
 d=json.loads(f.read_text());p=Path(f'/proc/{d["pid"]}/cmdline')
 if p.exists():
  assert b'kd_pixel_v2' in p.read_bytes()
  if 'starttime' in d:assert Path(f'/proc/{d["pid"]}/stat').read_text().split()[21]==d['starttime']
  os.kill(d['pid'],signal.SIGTERM)
time.sleep(5)
atomic_json(U/'method_stopped.json',{'status':'reschedule_for_efficiency','utc':now(),'resume_checkpoint':str(U/'10-5/step1/checkpoints/model_iter_2000.pth'),'reason':'User requested larger per-GPU batch / parallel KD scheduling'})
base=json.loads((U/'study.json').read_text())['common_config']
results=[]
for gpus,spg in [(8,1),(4,2),(2,4),(1,8)]:
 run=safe_path(R/f'runs/kd_throughput/g{gpus}_b{spg}');run.mkdir(parents=True,exist_ok=False)
 cfg={**base,'spg':spg,'num_workers':2,'max_iters':60,'log_iters':10,'eval_iters':60,'warmup_iters':1,'loss_warmup_iters':1,'save_ckpt':False}
 cmd=[str(R/'.runtime/env/bin/python'),'-B','-m','torch.distributed.run','--standalone',f'--nproc_per_node={gpus}',str(E/'src/scripts/dist_train_voc_seg_neg.py'),'--work_dir',str(run),'--step','1','--prev_checkpoint',str(R/'runs/fixed_baseline_v1/shared/10-5/step0/checkpoints/model_final.pth')]
 for k,v in cfg.items():
  if isinstance(v,bool):
   if v:cmd.append('--'+k)
   elif k=='pretrained':cmd.append('--no-pretrained')
  elif isinstance(v,list):cmd.extend(['--'+k,*map(str,v)])
  else:cmd.extend(['--'+k,str(v)])
 env=environment('8card');env['CUDA_VISIBLE_DEVICES']=','.join(map(str,range(gpus)))
 with open(run/'launcher.log','x') as log:
  p=subprocess.Popen(cmd,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  atomic_json(run/'process.json',{'pid':p.pid,'starttime':Path(f'/proc/{p.pid}/stat').read_text().split()[21],'command':cmd,'utc':now()})
  rc=p.wait()
 if rc:raise RuntimeError(f'Benchmark failed {run}: {rc}')
 result=json.loads((run/'10-5/step1/throughput.json').read_text());results.append(result)
 atomic_json(R/'runs/kd_throughput/results.json',{'updated_utc':now(),'results':results,'complete':len(results)==4})
 print(json.dumps(result),flush=True)
