from pathlib import Path
import sys,os,json
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto');E=R/'experiments/pair_preserving_kd_v1';C=R/'experiments/prototype_sep_v1/c_new_anchor/src'
sys.path.insert(0,str(C))
import torch
import torch.distributed as dist
from pair_targets import correction_targets
from pair_loss import hierarchical_kl,pair_preserving_loss
from kd_runtime import atomic_json,digest

torch.set_float32_matmul_precision('highest')
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False

torch.manual_seed(42)
ref=torch.full((1,21,2,2),-4.);ref[:,3]=3;ref[:,16]=-1
old=torch.full((1,16,2,2),-4.);old[:,3]=3
par=torch.full((1,2,2),16,dtype=torch.long);cam=torch.full((1,20,2,2),.1);cam[:,15]=.9
boxes=torch.tensor([[0,2,0,2]]);tags=torch.ones(1,5);pairs=torch.full((21,),-1,dtype=torch.long);pairs[16]=3
bg=torch.zeros(21,dtype=torch.bool)
z=correction_targets(ref,old,par,cam,boxes,tags,pairs,bg)
assert z['new_pair'].all() and not z['absent'].any()
other=[i for i in range(21) if i not in [3,16]]
assert torch.equal(z['reference'][:,other],z['target'][:,other])
assert torch.allclose(z['target'][:,[3,16]].sum(1),z['reference'][:,[3,16]].sum(1))
s=ref.clone().requires_grad_();val,parts=hierarchical_kl(s,z['target']);val.sum().backward()
assert s.grad[:,3].min()>0 and s.grad[:,16].max()<0 and s.grad[:,other].abs().max()<1e-6
full=(z['target']*(z['target'].clamp_min(1e-30).log()-(s/2).log_softmax(1))).sum(1)*4
assert torch.allclose(val,full,atol=2e-6)
oldbg=old.clone();oldbg[:,3]=-4;oldbg[:,0]=3
veto=correction_targets(ref,oldbg,par,cam,boxes,tags,pairs,bg);assert not veto['new_pair'].any()
ar=ref.clone();ar[:,3]=-4;ar[:,20]=3
a=correction_targets(ar,oldbg,par,cam,boxes,tags*0,pairs,bg)
assert a['absent'].all() and (a['target'][:,20]==0).all()
assert torch.allclose(a['target'][:,1]/a['target'][:,0],a['reference'][:,1]/a['reference'][:,0])
bgs=bg.clone();bgs[3]=True
b=correction_targets(ref,oldbg,par*0,cam*.1,boxes,tags,pairs,bgs)
assert b['bg_pair'].all() and b['target'].argmax(1).eq(0).all()
empty=correction_targets(ref,old,par,cam,torch.tensor([[0,0,0,0]]),tags,pairs,bg)
assert not empty['active'].any()

# Unequal per-rank support plus a completely empty rank, compared with the
# gradient/value of one concatenated batch before initializing distribution.
x=torch.randn(21,7,2,2);initial=torch.randn(21,7)*.1
reference=torch.einsum('cf,nfhw->nchw',initial,x)
q=(reference/2).softmax(1);target=q.clone();mask=torch.zeros(21,2,2,dtype=torch.bool)
mask[:7,0,0]=True;mask[7:13]=True
target[:,3]=torch.where(mask,q[:,16],q[:,3]);target[:,16]=torch.where(mask,q[:,3],q[:,16])
fake={'target':target,'reference':q,'valid':torch.ones_like(mask),'active':mask,'new_pair':mask,
      'bg_pair':torch.zeros_like(mask),'absent':torch.zeros_like(mask),'pseudo':torch.full_like(mask,16,dtype=torch.long),
      'reference_prediction':torch.full_like(mask,3,dtype=torch.long)}
w=initial.clone().requires_grad_();expected,_=pair_preserving_loss(torch.einsum('cf,nfhw->nchw',w,x),fake)
expected.backward();eg=w.grad.clone();ev=float(expected)
rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1))
assert world==8
torch.cuda.set_device(rank);dist.init_process_group('nccl')
indices=torch.arange(rank,21,7) if rank<7 else torch.empty(0,dtype=torch.long)
local={k:v[indices].cuda() for k,v in fake.items()};ww=initial.cuda().requires_grad_()
got,stats=pair_preserving_loss(torch.einsum('cf,nfhw->nchw',ww,x[indices].cuda()),local)
got.backward();dist.all_reduce(ww.grad);ww.grad/=world
err=float((ww.grad.cpu()-eg).abs().max());value_error=abs(float(got)-ev)
assert err<3e-6 and value_error<3e-6,(err,value_error)
dist.barrier()
if rank==0:
    atomic_json(E/'preflight.json',{'passed':True,'checks':['probability_sum','untouched_classes_exact',
        'selected_pair_sum','correction_gradient_locality_at_reference','hierarchy_full_KL_equivalence',
        'teacher_BG_new_veto','absent_class_conditional_target','background_pair_swap','empty_ROI',
        '8rank_unequal_support_empty_rank_global_value_gradient'],
        'gradient_max_error':err,'value_error':value_error,
        'sources':{name:digest(E/name) for name in ['pair_targets.py','pair_loss.py','test_pair.py']}})
    print(json.dumps({'passed':True,'gradient_max_error':err,'value_error':value_error}))
dist.destroy_process_group()
