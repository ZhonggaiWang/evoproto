"""Eight-rank pair KD parameter gradients versus one pooled global batch."""
import os,json
import torch
import torch.distributed as dist
from test_confusion_pair_ddp import batch
from model.pixel_kd import pixel_kd_loss

def pooled(device):
 parts=[batch(rank%4,device) for rank in range(8)]
 tensors=[torch.cat([p[index] for p in parts]) for index in range(4)]
 boxes=[box for p in parts for box in p[4]]
 evidence={key:torch.cat([p[5][key] for p in parts]) for key in parts[0][5]}
 return (*tensors,boxes,evidence)

def measure(data,device):
 features,teacher,anchors,cams,boxes,evidence=data
 weight=((torch.arange(18,device=device).float().reshape(6,3)-8)/7).requires_grad_()
 student=torch.einsum('cd,ndhw->nchw',weight,features)
 targets=torch.tensor([-1,2,1,1,2,-1],device=device)
 loss,stats=pixel_kd_loss(student,teacher,anchors,cams,boxes,temperature=2.,
                        pair_targets=targets,pair_evidence=evidence,pair_blend=.5)
 return loss.detach(),torch.autograd.grad(loss,weight)[0].detach(),stats

rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);device=torch.device('cuda',rank)
reference=measure(pooled(device),device)
dist.init_process_group('nccl');world=dist.get_world_size();assert world==8
actual=measure(batch(rank%4,device),device)
if rank%4==3:assert float(actual[1].abs().sum())==0
gradient=actual[1].clone();dist.all_reduce(gradient);gradient/=world
torch.testing.assert_close(actual[0],reference[0],atol=2e-6,rtol=2e-6)
torch.testing.assert_close(gradient,reference[1],atol=2e-6,rtol=2e-6)
for key in ['kd_pair_pixels','kd_pair_class_pixels','class_eligible_pixels','veto_new_pixels']:
 assert actual[2][key]==reference[2][key],key
assert actual[2]['kd_pair_pixels']>0
assert float(gradient[0].abs().sum())==0 and float(gradient[4:].abs().sum())==0
if rank==0:print(json.dumps({'test':'pair_KD_8rank_global_batch_parameter_gradient_equivalence',
 'passed':True,'ranks':8,'batch_sizes':[1,2,3,4,1,2,3,4],'zero_evidence_ranks':[3,7],
 'loss':float(reference[0]),'loss_max_error':float((actual[0]-reference[0]).abs()),
 'parameter_gradient_max_error':float((gradient-reference[1]).abs().max()),
 'pair_pixels':actual[2]['kd_pair_pixels'],'no_direct_bg_or_new_gradient':True}))
dist.destroy_process_group()
