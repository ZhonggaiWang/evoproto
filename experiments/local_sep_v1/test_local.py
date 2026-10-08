from pathlib import Path
import os,sys,json
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto');E=R/'experiments/local_sep_v1';S=E/'src'
sys.path.insert(0,str(R/'experiments/kd_pixel_v2'))
from run_kd import environment
os.environ.update(environment('8card'));sys.path.insert(0,str(S))
import torch
import torch.distributed as dist
from model.local_sep import separation_loss,retention_loss
from kd_runtime import atomic_json,digest
rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);dist.init_process_group('nccl');torch.set_num_threads(1)
dev='cuda';bank=torch.zeros(21,4,2,device=dev);bank[16,:,0]=1;bank[9,:,1]=1;bank.requires_grad_(True)
avail=torch.zeros(21,device=dev,dtype=torch.bool);avail[16]=True;avail[9]=True
pairs=torch.full((21,),-1,device=dev,dtype=torch.long);pairs[16]=9
y=torch.tensor([[[16,16,0,255]]],device=dev);valid=torch.ones_like(y,dtype=torch.bool);valid[:,:,-1]=False
cams=torch.zeros(1,20,1,4,device=dev);cams[:,15]=.9
f=torch.tensor([[[[.6,1.,.6,.6]],[[.8,0.,.8,.8]]]],device=dev,requires_grad=True)
loss,stats=separation_loss(f,y,cams,valid,pairs,bank,avail);loss.backward()
assert torch.isfinite(loss) and loss>0 and f.grad[:,:,:,0].abs().sum()>0
assert f.grad[:,:,:,1:].abs().sum()==0 and bank.grad is None
assert f.grad[0,0,0,0]<0 and f.grad[0,1,0,0]>0
scaled=f.detach()*3;ls,_=separation_loss(scaled,y,cams,valid,pairs,bank,avail);assert torch.allclose(loss,ls,atol=1e-5)
empty=torch.full_like(pairs,-1);f2=f.detach().clone().requires_grad_(True)
l,_=separation_loss(f2,y,cams,valid,empty,bank,avail);l.backward();assert l==0 and f2.grad.abs().sum()==0
# Different support per rank: compare all-reduced parameter gradient with
# analytical concatenated-batch reference, including ranks with no support.
param=torch.tensor(.6,device=dev,requires_grad=True)
inp=torch.stack([param,torch.tensor(.8,device=dev)]).reshape(1,2,1,1)
mask=torch.ones(1,1,1,device=dev,dtype=torch.bool)*(rank%2==0)
yy=y[:,:,:1];cc=cams[:,:,:,:1]
l,_=separation_loss(inp,yy,cc,mask,pairs,bank,avail);l.backward();g=param.grad.clone();dist.all_reduce(g);g/=dist.get_world_size()
pp=torch.tensor(.6,device=dev,requires_grad=True);gap=(pp-.8)/torch.sqrt(pp*pp+.64);expected=torch.relu((.05-gap)/.1).square();expected.backward()
assert torch.allclose(l,expected,atol=1e-5) and torch.allclose(g,pp.grad,atol=1e-4)
z=torch.randn(1,21,2,2,device=dev,requires_grad=True);ref=z.detach().clone().requires_grad_(True)
kl=retention_loss(z,ref,torch.ones(1,2,2,device=dev,dtype=torch.bool));kl.backward()
assert abs(float(kl))<1e-6 and z.grad.abs().max()<1e-6 and ref.grad is None
dist.barrier()
if rank==0:
 atomic_json(E/'preflight.json',{'passed':True,'ranks':8,'tests':['inactive_and_padding_zero_gradient','resolved_gap_zero_gradient','source_attraction_rival_repulsion','bank_detached','feature_scale_invariance','empty_support_finite_zero','unequal_DDP_support_matches_global_gradient','reference_KL_identity_and_detached'],
 'source_sha256':{str(p.relative_to(S)):digest(p) for p in S.rglob('*.py')},'test_sha256':digest(E/'test_local.py')})
 print('PASS 8 GPU local SEP invariants',flush=True)
dist.destroy_process_group()
