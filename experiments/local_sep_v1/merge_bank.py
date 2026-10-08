from pathlib import Path
import os,sys,json
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto');E=R/'experiments/local_sep_v1';U=R/'runs/local_sep_v1'
sys.path.insert(0,str(R/'experiments/kd_pixel_v2'))
from run_kd import environment
os.environ.update(environment('8card'))
from kd_runtime import safe_path,atomic_json,digest,now
import torch
import torch.nn.functional as F
torch.set_num_threads(4)
parts=[];names=[]
for i in range(8):
 p=safe_path(U/f'bank_rank{i}.pth');r=json.loads((U/f'bank_receipt{i}.json').read_text());assert r['bank_sha256']==digest(p)
 v=torch.load(p,weights_only=True);parts.append(v);names+=v['names']
assert len(names)==len(set(names))==2145
x=torch.cat([p['vectors'] for p in parts]);y=torch.cat([p['classes'] for p in parts]);owners=sum([p['owners'] for p in parts],[])
centers=torch.zeros(21,4,512);available=torch.zeros(21,dtype=torch.bool);counts={}
for c in range(1,21):
 v=x[y==c];counts[c]=len(v)
 if len(v)<4:continue
 # Four deterministic spherical modes, seeded by mean/most-distant samples.
 center=F.normalize(v.mean(0),dim=0)[None]
 for k in range(3):
  index=(v@center.T).amax(1).argmin();center=torch.cat([center,v[index][None]])
 for _ in range(20):
  ids=(v@center.T).argmax(1)
  center=torch.stack([F.normalize(v[ids==j].mean(0),dim=0) if (ids==j).any() else center[j] for j in range(4)])
 centers[c]=center;available[c]=True
out=safe_path(E/'bank.pth');assert not out.exists()
torch.save({'centers':centers,'available':available,'counts':counts,'names':names},out)
atomic_json(U/'bank.json',{'images':2145,'unique_images':2145,'image_class_vectors':len(x),'class_image_counts':counts,'available':available.tolist(),'bank_sha256':digest(out),'utc':now(),'GT_role':'training weak evidence only; no validation-derived choices','algorithm':'four spherical modes per class; 20 deterministic clustering steps; class requires four distinct images; ALD+CAM+model agreement, eroded class interiors'})
print(json.dumps(counts))
