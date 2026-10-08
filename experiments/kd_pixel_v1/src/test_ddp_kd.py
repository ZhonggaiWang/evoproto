"""Check that 8-rank KD, BCE and PTC gradients equal a single global batch."""
import os,json
import torch
import torch.distributed as dist
from model.pixel_kd import pixel_kd_loss
from model.losses import get_seg_loss,get_masked_ptc_loss

rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE'])
torch.manual_seed(123)
student=torch.randn(world,4,3,3);teacher=torch.randn(world,3,3,3)*3
labels=torch.randint(0,4,(world,3,3));cam=torch.rand(world,3,3,3)*.5
boxes=torch.tensor([[0,3,0,3]]*world)
features=torch.randn(world,5,3,3);mask=torch.randint(0,2,(world,9,9));target=labels.clone();target[0,0,0]=255
weight=torch.ones(4)
p=torch.tensor(0.8,requires_grad=True)
reference=pixel_kd_loss(student*p,teacher,labels,cam,boxes,2)[0]+get_seg_loss(student*p,target,weight)+get_masked_ptc_loss(features*p+0.1,mask)
reference.backward();expected=p.grad.detach().clone()
dist.init_process_group('gloo')
p=torch.tensor(0.8,requires_grad=True)
ix=slice(rank,rank+1)
actual=pixel_kd_loss(student[ix]*p,teacher[ix],labels[ix],cam[ix],boxes[ix],2)[0]+get_seg_loss(student[ix]*p,target[ix],weight)+get_masked_ptc_loss(features[ix]*p+0.1,mask[ix])
actual.backward();grad=p.grad.detach().clone();dist.all_reduce(grad);grad/=world
torch.testing.assert_close(grad,expected,rtol=1e-5,atol=1e-6)
if rank==0:print(json.dumps({'world':world,'reference_gradient':float(expected),'distributed_gradient':float(grad),'passed':True}))
dist.destroy_process_group()
