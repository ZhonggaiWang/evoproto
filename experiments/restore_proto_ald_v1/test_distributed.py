"""Two CPU ranks must reproduce ALD gradients for a concatenated batch."""
import json
from unittest.mock import patch
import torch
import torch.distributed as dist
import torch.nn.functional as F
from experiments.restore_proto_ald_v1.ald import uncertain_loss


def compute(section):
    torch.manual_seed(17)
    features=torch.randn(4,5,2,2)[section]
    weights=torch.randn(21,5,1,1,requires_grad=True)
    reference=torch.randn(4,21,2,2)[section].requires_grad_()
    states=torch.randint(-1,2,(4,15))[section]
    tags=torch.randint(0,2,(4,5))[section]
    valid=torch.tensor([[[1,1],[0,0]],[[1,1],[1,1]],[[1,0],[0,0]],[[1,1],[1,0]]],dtype=torch.bool)[section]
    unknown=valid.clone();unknown[:,0,0]=False
    loss=uncertain_loss(F.conv2d(features,weights),reference,unknown,valid,states,tags)
    loss.backward();assert reference.grad is None
    return loss.detach(),weights.grad


dist.init_process_group('gloo');rank=dist.get_rank();assert dist.get_world_size()==2
torch.set_num_threads(1)
loss,gradient=compute(slice(rank*2,rank*2+2));dist.all_reduce(gradient);gradient/=2
with patch('torch.distributed.is_initialized',return_value=False):
    expected_loss,expected_gradient=compute(slice(None))
assert torch.allclose(loss,expected_loss,atol=2e-6,rtol=2e-5)
assert torch.allclose(gradient,expected_gradient,atol=2e-6,rtol=2e-5)
if rank==0:print(json.dumps({'passed':True,'ranks':2,'gradient_max_abs_error':float((gradient-expected_gradient).abs().max()),'loss':float(loss)}))
dist.destroy_process_group()
