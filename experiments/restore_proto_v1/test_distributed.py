"""Two CPU ranks must match the concatenated-batch prototype gradient."""
import os,json
from unittest.mock import patch
import torch
import torch.distributed as dist
import torch.nn.functional as F
from experiments.restore_proto_v1.mechanism import ConfusionProto

def compute(sl):
    torch.manual_seed(13)
    f=torch.randn(4,5,2,2)[sl]
    p=torch.randn(4,5,requires_grad=True)
    teacher=torch.randn(4,3,2,2)[sl];teacher[:,1]=3
    main=torch.randn(4,4,2,2)[sl]
    z=torch.einsum('bdhw,kd->bkhw',F.normalize(f,dim=1),F.normalize(p,dim=1))/.1
    m=ConfusionProto(4,3,'cpu');m.ema[1,2]=1;m.ema[3,2]=1;m.mass[:]=2
    m.seen[1][2]={'a','b','c'};m.seen[3][2]={'a','b','c'}
    anchors=torch.tensor([1,1,3,3])[:,None,None].expand(4,2,2)[sl]
    cams=torch.ones(4,3,2,2)[sl]*.7;cams[:,2]=.1
    valid=torch.ones(4,2,2,dtype=torch.bool)[sl];pred=torch.ones_like(anchors)
    loss,_=m.loss(z,teacher,main,teacher,anchors,cams,valid,pred,anchors,p,p[:3].detach())
    loss.backward();return loss.detach(),p.grad

dist.init_process_group('gloo');rank=dist.get_rank()
loss,grad=compute(slice(rank*2,rank*2+2));dist.all_reduce(grad);grad/=2
with patch('torch.distributed.is_initialized',return_value=False):
    ref,expected=compute(slice(None))
assert torch.allclose(grad,expected,atol=2e-6,rtol=2e-5),(grad-expected).abs().max()
assert torch.allclose(loss,ref,atol=2e-6,rtol=2e-5),(loss,ref)
if rank==0:print(json.dumps({'distributed_gradient_matches_full_batch':True,'max_abs_error':float((grad-expected).abs().max()),'loss':float(loss)}))
dist.destroy_process_group()
