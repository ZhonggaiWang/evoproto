from pathlib import Path
import sys,os,copy
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto');E=R/'experiments/ald_calibration_v7'
sys.path.insert(0,str(R/'experiments/prototype_sep_v1/c_new_anchor/src'))
import torch
import torch.distributed as dist
from losses import loss,absence_loss
from kd_runtime import atomic_json,digest
torch.manual_seed(42);torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');torch.backends.cudnn.allow_tf32=False
model=torch.nn.Sequential(torch.nn.Conv2d(7,12,3,padding=1,bias=False),torch.nn.ReLU(),torch.nn.Conv2d(12,21,1,bias=False))
model[2].requires_grad_(False);original=model[0].weight.detach().clone();n=24;x=torch.randn(n,7,3,3)
with torch.no_grad():reference=model(x)
labels=torch.randint(16,21,(n,3,3));selected=torch.rand(n,3,3)>.5;selected[7::8]=False
old=torch.randn(n,16,3,3);par=torch.full((n,3,3),255);cams=torch.rand(n,20,3,3);trusted=torch.rand(n,3,3)>.5;trusted[7::8]=False
tags=torch.randint(0,2,(n,5)).float()
def objective(ids):return loss(model(x[ids]),reference[ids],labels[ids],selected[ids],old[ids],par[ids],cams[ids],trusted[ids],tags[ids])[0]
v=objective(torch.arange(n));v.backward();eg=model[0].weight.grad.clone();expected=float(v);assert eg.abs().sum()>0 and model[2].weight.grad is None
rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);dist.init_process_group('nccl');assert dist.get_world_size()==8
x=x.cuda();reference=reference.cuda();labels=labels.cuda();selected=selected.cuda();old=old.cuda();par=par.cuda();cams=cams.cuda();trusted=trusted.cuda();tags=tags.cuda();model.cuda();model.zero_grad()
got=objective(torch.arange(rank,n,8,device='cuda'));got.backward();dist.all_reduce(model[0].weight.grad);model[0].weight.grad/=8
error=float((model[0].weight.grad.cpu()-eg).abs().max());ve=abs(float(got)-expected);assert error<5e-6 and ve<5e-6,(error,ve)
assert model[2].weight.grad is None
dist.barrier()
if rank==0:atomic_json(E/'preflight.json',{'passed':True,'gradient_error':error,'value_error':ve,
    'checks':['spatial_layer_real_gradient','frozen_output_weights_no_gradient','8rank_unequal_support_empty_rank_full_loss_gradient'],
    'sources':{n:digest(E/n) for n in ['losses.py','test_ald.py']}})
dist.destroy_process_group()
