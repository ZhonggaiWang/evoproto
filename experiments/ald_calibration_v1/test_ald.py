from pathlib import Path
import sys,os
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto');E=R/'experiments/ald_calibration_v1'
sys.path.insert(0,str(R/'experiments/prototype_sep_v1/c_new_anchor/src'))
import torch
import torch.distributed as dist
from evidence import calibrate_presence,fuse_supervision
from losses import classification_loss,segmentation_loss,balanced_mean
from kd_runtime import atomic_json,digest
torch.set_num_threads(1);torch.manual_seed(42)
torch.set_float32_matmul_precision('highest');torch.backends.cudnn.allow_tf32=False

# Unknown is not a negative: its logit has exactly zero gradient at reference.
reference=torch.randn(4,20);s=reference.clone().requires_grad_();state=torch.full((2,15),-1);tags=torch.ones(2,5)
l,_=classification_loss(s,reference,state,tags);l.backward()
assert s.grad[:,:15].abs().max()<1e-7 and s.grad[:,15:].abs().sum()>0
# No accepted pseudo pixel: no hard-label or old-teacher KD gradient.
ref=torch.randn(2,21,2,2);s=ref.clone().requires_grad_();labels=torch.full((2,2,2),255)
old=torch.randn(2,16,2,2);cam=torch.ones(2,20,2,2);trusted=torch.zeros(2,2,2,dtype=torch.bool)
l,stats=segmentation_loss(s,ref,labels,old,labels,cam,trusted);l.backward()
assert stats['KD_pixels']==0 and sum(stats['trusted_counts'])==0 and s.grad.abs().max()<1e-6
# A raw positive alone cannot enter trusted old supervision.
oc=torch.full((1,2,15),-2.);oc[:,:,0]=2;tv=torch.zeros(1,2,16,2,2);rv=torch.zeros(1,2,21,2,2)
tv[:,:,1]=2;rv[:,:,2]=2;cam=torch.zeros(1,20,2,2);cam[:,0]=1;new=torch.zeros(1,5)
p=calibrate_presence(oc,oc,tv,rv,cam,new)
assert p['state'][0,0]==-1 and not p['positive'][0,0]
f=fuse_supervision(p,tv,rv,torch.ones(1,2,2,dtype=torch.long),cam,new)
assert f['labels'].eq(255).all()
# Agreement and three-pixel support admit the positive; view conflict removes it.
rv[:,:,1]=3;p=calibrate_presence(oc,oc,tv,rv,cam,new);assert p['positive'][0,0]
rv[:,1,1]=0;p=calibrate_presence(oc,oc,tv,rv,cam,new);assert not p['positive'][0,0]

# Uneven global class support and a rank with no supervised pixels: compare
# DDP-equivalent gradients with one concatenated batch, not rank averages.
x=torch.randn(24,7);initial=torch.randn(7,4);label=torch.arange(24)%4;mask=torch.ones(24,dtype=torch.bool);mask[7::8]=False;mask[:4]=False
def objective(w,xx,yy,mm,rr,ss,tt):
    logits=xx@w
    value=torch.nn.functional.cross_entropy(logits,yy,reduction='none')
    a,_=balanced_mean(value,yy,mm,4)
    c,_=classification_loss((xx@w)[:,0,None].expand(-1,20),rr,ss,tt)
    return a+c
# Classification batches have pairs of views; use a separate 2-image/rank set.
cx=torch.randn(32,7);cref=torch.randn(32,20);states=torch.randint(-1,2,(16,15));nt=torch.randint(0,2,(16,5)).float();states[14:]=-1
def combined(w,ids,ci,si):
    out=x[ids]@w;a,_=balanced_mean(torch.nn.functional.cross_entropy(out,label[ids],reduction='none'),label[ids],mask[ids],4)
    c,_=classification_loss((cx[ci]@w)[:,0,None].expand(-1,20),cref[ci],states[si],nt[si])
    return a+c
w=initial.clone().requires_grad_();expected=combined(w,torch.arange(24),torch.arange(32),torch.arange(16));expected.backward();eg=w.grad.clone();ev=float(expected)
rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);dist.init_process_group('nccl');assert dist.get_world_size()==8
x=x.cuda();cx=cx.cuda();label=label.cuda();mask=mask.cuda();cref=cref.cuda();states=states.cuda();nt=nt.cuda()
w=initial.cuda().requires_grad_();got=combined(w,torch.arange(rank,24,8,device='cuda'),torch.arange(rank*4,rank*4+4,device='cuda'),torch.arange(rank*2,rank*2+2,device='cuda'))
got.backward();dist.all_reduce(w.grad);w.grad/=8
error=float((w.grad.cpu()-eg).abs().max());verror=abs(float(got)-ev)
assert error<4e-6 and verror<4e-6,(error,verror)
dist.barrier()
if rank==0:atomic_json(E/'preflight.json',{'passed':True,'gradient_error':error,'value_error':verror,
    'checks':['unknown_image_not_negative','empty_pixel_supervision_zero_gradient','KD_explicit_uncertainty_veto','raw_positive_insufficient','spatial_multiview_admission','view_conflict_veto','8rank_unequal_support_value_gradient'],
    'sources':{n:digest(E/n) for n in ['evidence.py','losses.py','test_ald.py']}})
dist.destroy_process_group()
