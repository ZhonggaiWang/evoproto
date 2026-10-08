from pathlib import Path
import sys,os
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto');E=R/'experiments/ald_calibration_v2'
sys.path.insert(0,str(R/'experiments/prototype_sep_v1/c_new_anchor/src'))
import torch
import torch.distributed as dist
from targets import calibrated_targets
from losses import segmentation_loss,classification_loss
from kd_runtime import atomic_json,digest
torch.manual_seed(42);torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');torch.backends.cudnn.allow_tf32=False
ref=torch.randn(2,21,2,2);absent=torch.zeros(2,15,dtype=torch.bool);absent[:,2]=True;tags=torch.ones(2,5);tags[:,0]=0
z=calibrated_targets(ref,absent,tags);assert z['target'][:,[3,16]].eq(0).all()
assert torch.allclose(z['target'].sum(1),torch.ones(2,2,2))
assert torch.allclose(z['target'][:,1]/z['target'][:,2],z['reference'][:,1]/z['reference'][:,2])
s=ref.clone().requires_grad_();kl=(z['target']*(z['target'].clamp_min(1e-30).log()-(s/2).log_softmax(1))).sum()*4;kl.backward();assert (s.grad[:,[3,16]]>0).all()
# An unknown class is not removed merely because it has spatial absence.
state=torch.full((1,15),-1);spatial=torch.ones(1,15,dtype=torch.bool);r=torch.zeros(2,20);r[:,2]=2
s=r.clone().requires_grad_();loss,stats=classification_loss(s,r,state,torch.ones(1,5),spatial);loss.backward()
assert stats['conflicts']==2 and (s.grad[:,2]>0).all() and s.grad[:,0].eq(0).all()
s=r.clone();s[:,2]=0;s.requires_grad_();q,_=classification_loss(s,s.detach(),state,torch.ones(1,5),spatial);q.backward();assert s.grad[:,:15].eq(0).all()
z=calibrated_targets(ref,torch.zeros_like(absent),torch.ones_like(tags));assert torch.allclose(z['target'],z['reference'])

# All objectives together, comparing one concatenated batch to eight ranks.
n=16;x=torch.randn(n,7,2,2);cx=torch.randn(n*2,7);initial=torch.randn(21,7)*.1
reference=torch.randn(n,21,2,2);old=torch.randn(n,16,2,2);labels=torch.full((n,2,2),255,dtype=torch.long);par=labels.clone()
cam=torch.rand(n,20,2,2);trusted=torch.rand(n,2,2)>.5;trusted[14:]=False
states=torch.randint(-1,2,(n,15));nt=torch.randint(0,2,(n,5)).float();spatial=torch.rand(n,15)>.5;cr=torch.randn(n*2,20)
def objective(w,ids,ci):
    output=torch.einsum('cf,nfhw->nchw',w,x[ids])
    a,_=segmentation_loss(output,reference[ids],labels[ids],old[ids],par[ids],cam[ids],trusted[ids],states[ids],nt[ids])
    c,_=classification_loss(cx[ci]@w[:20].T,cr[ci],states[ids],nt[ids],spatial[ids])
    return a+c
w=initial.clone().requires_grad_();v=objective(w,torch.arange(n),torch.arange(n*2));v.backward();expected=float(v);eg=w.grad.clone()
rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);dist.init_process_group('nccl');assert dist.get_world_size()==8
x=x.cuda();cx=cx.cuda();reference=reference.cuda();old=old.cuda();labels=labels.cuda();par=par.cuda();cam=cam.cuda();trusted=trusted.cuda();states=states.cuda();nt=nt.cuda();spatial=spatial.cuda();cr=cr.cuda()
w=initial.cuda().requires_grad_();got=objective(w,torch.arange(rank*2,rank*2+2,device='cuda'),torch.arange(rank*4,rank*4+4,device='cuda'));got.backward();dist.all_reduce(w.grad);w.grad/=8
err=float((w.grad.cpu()-eg).abs().max());ve=abs(float(got)-expected);assert err<5e-6 and ve<5e-6,(err,ve)
dist.barrier()
if rank==0:atomic_json(E/'preflight.json',{'passed':True,'gradient_error':err,'value_error':ve,'checks':['forbidden_mass_zero','remaining_ratios_preserved','no_forced_BG','veto_gradient_direction','unknown_not_hard_negative','conflict_neutral_target','8rank_full_loss_global_value_gradient'],
    'sources':{n:digest(E/n) for n in ['targets.py','losses.py','test_ald.py']}})
dist.destroy_process_group()
