from pathlib import Path
import sys,os
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto');E=R/'experiments/ald_calibration_v5'
sys.path.insert(0,str(R/'experiments/prototype_sep_v1/c_new_anchor/src'))
import torch
import torch.distributed as dist
import torch.nn.functional as F
from residual import SemanticResidual
from losses import loss
from kd_runtime import atomic_json,digest
torch.manual_seed(42);torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');torch.backends.cudnn.allow_tf32=False
weights=torch.randn(21,512,1,1)*.03;m=SemanticResidual(weights)
with torch.no_grad():m.delta.normal_(0,.001)
delta=m.folded_new_weight()-weights[16:];assert (delta.flatten(1).double()@weights.flatten(1).double().T).abs().max()<1e-6
# Every old logit is unchanged. Corrections vanish on all21 semantic vectors.
x=weights.flatten(1)[:,:,None,None];ref=F.conv2d(x,weights);out=m(ref,m.features(x));assert torch.equal(out[:,:16],ref[:,:16]);assert (out-ref).abs().max()<1e-6
saved={k:v.clone() for k,v in m.state_dict().items()};restored=SemanticResidual(weights);restored.load_state_dict(saved);assert torch.equal(restored.delta,m.delta)

n=24;features=torch.randn(n,512,2,2);reference=F.conv2d(features,weights);proj=m.features(features).detach()
labels=torch.randint(16,21,(n,2,2));selected=torch.rand(n,2,2)>.5;selected[7::8]=False
old=reference[:,:16].clone();par=torch.full((n,2,2),255);cams=torch.zeros(n,20,2,2);trusted=torch.zeros(n,2,2,dtype=torch.bool)
initial=m.delta.detach().clone()
def objective(module,idx):
    output=module(reference[idx],proj[idx]);return loss(output,reference[idx],labels[idx],selected[idx],old[idx],par[idx],cams[idx],trusted[idx])[0]
m.zero_grad();value=objective(m,torch.arange(n));value.backward();expected=float(value);eg=m.delta.grad.clone()
rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);dist.init_process_group('nccl');assert dist.get_world_size()==8
reference=reference.cuda();proj=proj.cuda();labels=labels.cuda();selected=selected.cuda();old=old.cuda();par=par.cuda();cams=cams.cuda();trusted=trusted.cuda();m.cuda();m.zero_grad()
got=objective(m,torch.arange(rank,n,8,device='cuda'));got.backward();dist.all_reduce(m.delta.grad);m.delta.grad/=8
error=float((m.delta.grad.cpu()-eg).abs().max());ve=abs(float(got)-expected);assert error<5e-6 and ve<5e-6,(error,ve)
dist.barrier()
if rank==0:atomic_json(E/'preflight.json',{'passed':True,'gradient_error':error,'value_error':ve,
    'checks':['old16_logits_exact','all21_semantic_directions_preserved','residual_checkpoint_roundtrip','fixed_eligibility_normalization','8rank_uneven_support_empty_rank_loss_gradient'],
    'sources':{n:digest(E/n) for n in ['residual.py','losses.py','test_ald.py']}})
dist.destroy_process_group()
