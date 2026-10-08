from pathlib import Path
import os,sys,json
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto');E=R/'experiments/ald_calibration_v9'
sys.path.insert(0,str(E/'src'))
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from model.ald import image_targets,fuse,gate_auxiliary,uncertain_loss,partial_set_loss,foreground_relation_loss
from model.pixel_kd import pixel_kd_loss
from kd_runtime import atomic_json,digest

from utils.optimizer import PolyWarmupAdamW
import io
probe=torch.nn.Parameter(torch.ones(1))
optimizer=PolyWarmupAdamW([probe],lr=2e-6,weight_decay=.01,betas=(.9,.999),warmup_iter=2000,max_iter=8600,warmup_ratio=1e-6,power=.9)
optimizer.global_step=8300;probe.grad=torch.ones_like(probe);optimizer.step()
assert type(optimizer.param_groups[0]['lr']) is float
buffer=io.BytesIO();torch.save(optimizer.state_dict(),buffer);buffer.seek(0)
assert torch.load(buffer,weights_only=True)['param_groups'][0]['lr']==2e-6

rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);dist.init_process_group('nccl')
torch.manual_seed(9);torch.set_num_threads(1)
torch.set_float32_matmul_precision('highest');torch.backends.cudnn.allow_tf32=False
device=torch.device('cuda',rank)
state=torch.full((1,15),-1,device=device,dtype=torch.long);state[0,0]=1;state[0,1]=0
old=torch.full((1,15),-2.,device=device);old[:,2]=4
teacher=torch.zeros(1,16,2,2,device=device);teacher[:,0]=2;teacher[0,1,0,0]=5;teacher[0,3,0,1]=5
ref=torch.zeros(1,21,2,2,device=device);ref[:,0]=2;ref[0,16,1,0]=5;ref[0,1,0,0]=6
tags=torch.ones(1,5,device=device)
target,allowed,conflict=image_targets(state,old,old,teacher,ref,tags)
assert target[0,0]==1 and target[0,1]==0 and 0<target[0,3]<.5
assert allowed[0,2] and not allowed[0,1]
old2=old.clone();old2[0,4]=5
target2,_,conflict2=image_targets(state,old2,old2,teacher,ref,tags)
assert target2[0,4]==.5 and conflict2[0,4]
par=torch.zeros(1,2,2,device=device,dtype=torch.long);box=[[0,2,0,2]]
fused=fuse(par.float(),teacher,ref,state,box)
assert fused['labels'].tolist()==[[[1,255],[255,0]]]
assert fused['trusted_old'].sum()==1 and fused['unknown'].sum()==2
par[0,0,0]=17;fused2=fuse(par,teacher,ref,state,[[0,1,0,2]])
assert fused2['labels'][0,0,0]==255 and not fused2['trusted_old'].any()
ref_new=ref.clone();ref_new[0,17,0,0]=8
assert fuse(par,teacher,ref_new,state,box)['labels'][0,0,0]==17
assert (fused2['labels'][:,1]==255).all() and not fused2['unknown'][:,1].any()
aux=gate_auxiliary(torch.tensor([[[1,3],[0,0]]],device=device).float(),state,ref)
assert aux.tolist()==[[[1,255],[255,0]]]

head=torch.nn.Conv2d(3,21,1,bias=False).to(device);ddp=DDP(head,device_ids=[rank])
gen=torch.Generator(device=device).manual_seed(rank+10)
x=torch.randn(1,3,2,2,device=device,generator=gen)
y=ddp(x);loss=uncertain_loss(y,ref,fused['unknown'],fused['valid'],state,tags)
loss.backward();actual=head.weight.grad.clone()
xs=[torch.empty_like(x) for _ in range(8)];dist.all_gather(xs,x)
weight=head.weight.detach().clone().requires_grad_();global_logits=F.conv2d(torch.cat(xs),weight)
allowed_all=torch.cat([torch.ones(1,1,device=device,dtype=torch.bool),state!=0,tags.bool()],1)
fg_allowed=allowed_all[:,1:]
lt=(ref[:,1:]/2).masked_fill(~fg_allowed[:,:,None,None],-1e4).log_softmax(1)
ls=(global_logits[:,1:]/2).masked_fill(~fg_allowed[:,:,None,None],-1e4).log_softmax(1)
per=(lt.exp()*(lt-ls)).sum(1)*4
per=per+torch.logsumexp(global_logits,1)-torch.logsumexp(global_logits.masked_fill(~allowed_all[:,:,None,None],-1e4),1)
expected_loss=(per*fused['unknown']).sum()/(8*fused['valid'].sum());expected_loss.backward()
assert torch.allclose(actual,weight.grad,atol=2e-7,rtol=2e-5),(actual-weight.grad).abs().max()
assert torch.allclose(loss,expected_loss,atol=2e-6)

student=torch.randn(1,21,2,2,device=device,requires_grad=True)
cams=torch.ones(1,20,2,2,device=device);cams[:,15:]=0
kd,_=pixel_kd_loss(student,teacher,torch.zeros_like(par),cams,box,2.,trusted_mask=torch.zeros_like(par,dtype=torch.bool))
kd.backward();assert kd==0 and student.grad.abs().sum()==0
rel_input=student.detach().clone().requires_grad_()
rel=foreground_relation_loss(rel_input,ref,fused['unknown'],fused['valid'],state,tags)
rel.backward()
assert rel_input.grad[:,0].abs().sum()==0
assert rel_input.grad[:,1:].sum(1).abs().max()<2e-7
shifted=rel_input.detach().clone();shifted[:,1:]+=3.
assert torch.allclose(rel,foreground_relation_loss(shifted,ref,fused['unknown'],fused['valid'],state,tags),atol=2e-6)
part_input=student.detach().clone().requires_grad_()
part=partial_set_loss(part_input,fused['unknown'],fused['valid'],state,tags);part.backward()
assert (part_input.grad[:,0]<=0).all() and (part_input.grad[:,2]>=0).all()
assert partial_set_loss(student,fused['unknown'],fused['valid'],torch.ones_like(state),tags).abs()<1e-6
z=uncertain_loss(student,ref,torch.zeros_like(par,dtype=torch.bool),fused['valid'],state,tags)
assert z==0 and torch.isfinite(z)
dist.barrier()
if rank==0:atomic_json(E/'preflight.json',{'passed':True,'ranks':8,'tests':['unknown_not_negative','neutral_conflict','no_old_refill','new_PAR_conflict_unknown_consensus_preserved','conditional_FG_BG_gradient_zero','conditional_FG_common_shift_invariance','partial_set_only_excludes_absence','padding_excluded','PTC_confusion_calibration','global_DDP_loss_and_gradient','KD_rejected_pixel_zero_gradient','zero_unknown_finite'],
    'source_sha256':{str(p.relative_to(E/'src')):digest(p) for p in (E/'src').rglob('*.py') if '__pycache__' not in p.parts},'test_sha256':digest(E/'test_ald.py')})
dist.destroy_process_group()
