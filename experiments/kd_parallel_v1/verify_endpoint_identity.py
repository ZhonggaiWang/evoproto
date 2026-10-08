"""Prove that final models loaded by the next stage equal evaluated endpoints."""
from pathlib import Path
import sys,os,json
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto');os.chdir(R)
sys.path.insert(0,str(R/'experiments/kd_pixel_v2'))
from run_kd import environment
os.environ.update(environment('8card'))
sys.path.insert(0,str(R/'experiments/kd_pixel_v2/src'))
from kd_runtime import atomic_json,digest,now
import torch
torch.set_num_threads(2)

for arm in ['a_sigmoid','b_relational']:
 for step in [1,2]:
  run=R/f'runs/kd_parallel_v1/formal/{arm}';stage=run/f'10-5/step{step}'
  final=stage/'checkpoints/model_final.pth';evaluated=stage/'checkpoints/model_iter_8000.pth'
  result=run/f'evaluations/step{step}_iter8000/result.json'
  if not (final.exists() and evaluated.exists() and result.exists()):continue
  out=stage/'endpoint_identity.json';a_hash=digest(final);b_hash=digest(evaluated)
  if out.exists():
   cached=json.loads(out.read_text())
   if cached['final_sha256']==a_hash and cached['evaluated_sha256']==b_hash and cached['all_model_tensors_exactly_equal']:continue
  assert b_hash==json.loads(result.read_text())['checkpoint_sha256']
  a=torch.load(final,map_location='cpu',weights_only=True,mmap=True)['model_state']
  b=torch.load(evaluated,map_location='cpu',weights_only=True,mmap=True)['model_state']
  assert a.keys()==b.keys()
  mismatch=[name for name in a if not torch.equal(a[name],b[name])]
  if mismatch:raise RuntimeError(f'{arm} stage{step} final/evaluated mismatch: {mismatch}')
  atomic_json(out,{'utc':now(),'final_sha256':a_hash,'evaluated_sha256':b_hash,
    'all_model_tensors_exactly_equal':True,'verified_tensor_count':len(a),'arm':arm,'step':step})
  print(arm,step,'exactly equal',len(a),'tensors')
  del a,b
