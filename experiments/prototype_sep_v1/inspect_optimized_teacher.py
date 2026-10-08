from pathlib import Path
import json, sys
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto')
O=R/'runs/kd_parallel_v1/formal/b_relational'
sys.path.insert(0,str(R/'experiments/kd_pixel_v2'))
from run_kd import environment
import os
os.environ.update(environment('8card'))
import torch
stage=O/'10-5/step1'
saved=torch.load(stage/'checkpoints/model_final.pth',map_location='cpu',weights_only=True,mmap=True)
out={'checkpoint_keys':list(saved),'saved_iteration':saved.get('iteration')}
for name in ['training_complete.json','endpoint_identity.json']:
    out[name]=json.loads((stage/name).read_text())
for name in ['eval_queue/step1_iter8000.json','evaluations/step1_iter8000/result.json']:
    d=json.loads((O/name).read_text())
    out[name]={k:d.get(k) for k in ['step','iteration','checkpoint','checkpoint_sha256','images','all_miou']}
print(json.dumps(out))
