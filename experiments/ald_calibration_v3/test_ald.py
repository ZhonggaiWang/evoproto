from pathlib import Path
import sys,os
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto');E=R/'experiments/ald_calibration_v3'
sys.path.insert(0,str(R/'experiments/prototype_sep_v1/c_new_anchor/src'))
import torch
import torch.distributed as dist
from proposals import proposals
from losses import targets,loss
from kd_runtime import atomic_json,digest
torch.manual_seed(42);torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');torch.backends.cudnn.allow_tf32=False
ref=torch.zeros(1,21,2,2);ref[:,3]=3;bg=torch.ones(1,2,2,dtype=torch.bool);z=targets(ref,torch.zeros_like(bg,dtype=torch.long),bg)
assert torch.allclose(z['target'].sum(1),torch.ones(1,2,2));assert z['target'].argmax(1).eq(0).all()
assert torch.equal(z['reference'][:,[0,3]].sum(1),z['target'][:,[0,3]].sum(1))
assert torch.equal(z['target'][:,[i for i in range(21) if i not in [0,3]]],z['reference'][:,[i for i in range(21) if i not in [0,3]]])
s=ref.clone().requires_grad_();q=z['target'];v=(q*(q.clamp_min(1e-30).log()-(s/2).log_softmax(1))).sum()*4;v.backward();assert s.grad[:,0].max()<0 and s.grad[:,3].min()>0
assert torch.equal(targets(ref,torch.zeros_like(bg,dtype=torch.long),~bg)['target'],z['reference'])
# An uncertain old-class CAM is a veto to BG; hard-label admission and BG
# rejection must not share a mask that erases uncertain foreground evidence.
p={'negative':torch.zeros(1,15,dtype=torch.bool),'positive':torch.zeros(1,15,dtype=torch.bool)}
tv=torch.zeros(1,2,16,2,2);tv[:,:,0]=1;rv=ref[:,None].expand(-1,2,-1,-1,-1);cam=torch.zeros(1,20,2,2);cam[:,0]=.8
pp=proposals(p,tv,rv,torch.zeros(1,2,2,dtype=torch.long),cam,torch.zeros(1,5));assert not pp['background'].any()
cam[:,0]=.1;pp=proposals(p,tv,rv,torch.zeros(1,2,2,dtype=torch.long),cam,torch.zeros(1,5));assert pp['background'].all()

n=24;x=torch.randn(n,7,2,2);initial=torch.randn(21,7)*.1;reference=torch.randn(n,21,2,2);old=torch.randn(n,16,2,2)
par=torch.full((n,2,2),255);cams=torch.rand(n,20,2,2);trusted=torch.rand(n,2,2)>.5;mask=torch.rand(n,2,2)>.5;proposal=torch.randint(16,21,(n,2,2));proposal[:4]=0;mask[7::8]=False;trusted[7::8]=False
def objective(w,ids):return loss(torch.einsum('cf,nfhw->nchw',w,x[ids]),reference[ids],proposal[ids],mask[ids],old[ids],par[ids],cams[ids],trusted[ids])[0]
w=initial.clone().requires_grad_();v=objective(w,torch.arange(n));v.backward();expected=float(v);eg=w.grad.clone()
rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);dist.init_process_group('nccl');assert dist.get_world_size()==8
x=x.cuda();reference=reference.cuda();old=old.cuda();par=par.cuda();cams=cams.cuda();trusted=trusted.cuda();mask=mask.cuda();proposal=proposal.cuda()
w=initial.cuda().requires_grad_();got=objective(w,torch.arange(rank,n,8,device='cuda'));got.backward();dist.all_reduce(w.grad);w.grad/=8
err=float((w.grad.cpu()-eg).abs().max());ve=abs(float(got)-expected);assert err<5e-6 and ve<5e-6,(err,ve)
dist.barrier()
if rank==0:atomic_json(E/'preflight.json',{'passed':True,'gradient_error':err,'value_error':ve,'checks':['mass_conservation','untouched_classes_exact','candidate_pair_correction_gradient','inactive_exact_reference','uncertain_CAM_veto','8rank_uneven_support_empty_rank_full_loss'],
    'sources':{n:digest(E/n) for n in ['proposals.py','losses.py','test_ald.py']}})
dist.destroy_process_group()
