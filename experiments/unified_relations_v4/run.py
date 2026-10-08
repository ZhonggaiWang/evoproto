from common import *
import subprocess,json,argparse,time
import numpy as np

def merge(mode):
 parts=[json.loads((U/mode/f'eval_rank{i}.json').read_text()) for i in range(8)]
 names=sum([p['images'] for p in parts],[]);expected=256 if mode=='diagnose' else 16 if mode=='smoke' else 1449
 assert len(names)==len(set(names))==expected
 target=(R/'datasets/voc/incremental_split/val_10-5_step_3.txt').read_text().splitlines()[:expected]
 assert set(names)==set(target)
 results={}
 for key in parts[0]['histograms']:
  h=sum(np.array(p['histograms'][key]) for p in parts);den=h.sum(0)+h.sum(1)-h.diagonal()
  iou=np.divide(h.diagonal(),den,out=np.zeros(21,dtype=float),where=den>0)*100
  results[key]=dict(all_miou=float(iou.mean()),old15=float(iou[1:16].mean()),new5=float(iou[16:].mean()),class_iou=iou.tolist(),histogram=h.tolist())
 h=sum(np.array(p['anchor_gt']) for p in parts);c=sum(np.array(p['directed_weak_confusion']) for p in parts)
 def precision(part):
  total=h[:,part].sum();correct=sum(h[i,i] for i in range(21)[part])
  return float(correct/total) if total else None
 result=dict(results=results,images=expected,anchor_precision=dict(all=precision(slice(None)),foreground=precision(slice(1,None)),old=precision(slice(1,16)),new=precision(slice(16,None))),anchor_gt=h.tolist(),weak_confusion=c.tolist(),checkpoint_sha256=parts[0]['checkpoint_sha256'],utc=now())
 atomic_json(U/mode/'result.json',result)
 print(json.dumps(dict(mode=mode,results={k:{q:v for q,v in r.items() if q in ['all_miou','old15','new5']} for k,r in results.items()},anchor_precision=result['anchor_precision'])),flush=True)
 return result

def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['diagnose','smoke','formal']);p.add_argument('--worker',action='store_true');a=p.parse_args()
 run=safe_path(U/a.mode);run.mkdir(parents=True,exist_ok=True)
 if not a.worker:
  pre=json.loads((E/'preflight.json').read_text());assert pre['passed']
  assert all(digest(E/k)==v for k,v in pre['source_sha256'].items())
  if a.mode=='formal':
   assert json.loads((U/'smoke/status.json').read_text())['status']=='complete'
   assert all(json.loads((U/f'smoke/audit_rank{i}.json').read_text())['passed'] for i in range(8))
  with safe_path(run/'coordinator.log').open('x') as f:
   child=subprocess.Popen([str(R/'.runtime/env/bin/python'),'-B',str(E/'run.py'),a.mode,'--worker'],cwd=R,env=os.environ.copy(),stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  atomic_json(run/'process.json',dict(pid=child.pid,utc=now()));print(child.pid);return
 started=time.time()
 try:
  if a.mode!='diagnose':
   cmd=[str(R/'.runtime/env/bin/python'),'-B','-m','torch.distributed.run','--standalone','--nproc_per_node=8',str(E/'train.py'),'--mode',a.mode]
   with safe_path(run/'training.log').open('x') as f:r=subprocess.run(cmd,cwd=R,env=os.environ.copy(),stdout=f,stderr=subprocess.STDOUT)
   assert r.returncode==0,f'Training exit {r.returncode}'
  atomic_json(run/'status.json',dict(status='evaluating',utc=now()))
  workers=[]
  for rank in range(8):
   with safe_path(run/f'eval_rank{rank}.log').open('x') as f:
    c=subprocess.Popen([str(R/'.runtime/env/bin/python'),'-B',str(E/'evaluate.py'),'--mode',a.mode,'--rank',str(rank)],cwd=R,env=os.environ.copy(),stdout=f,stderr=subprocess.STDOUT)
   workers.append(c)
  codes=[c.wait() for c in workers];assert codes==[0]*8,codes
  result=merge(a.mode)
  atomic_json(run/'status.json',dict(status='complete',elapsed_seconds=time.time()-started,results={k:v['all_miou'] for k,v in result['results'].items()},utc=now()))
 except BaseException as e:
  atomic_json(run/'status.json',dict(status='failed',error=repr(e),utc=now()));raise

if __name__=='__main__':main()
