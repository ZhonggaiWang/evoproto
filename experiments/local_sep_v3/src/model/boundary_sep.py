"""Bounded directed decision-boundary separation; no GT interface."""
import math
import torch
import torch.nn.functional as F
from model.pixel_kd import global_ratio,distributed_sum

@torch.no_grad()
def targets(labels,cams,aux_cams,valid,reference):
 size=reference.shape[-2:]
 y=F.interpolate(labels[:,None].float(),size,mode='nearest')[:,0].long()
 area=F.interpolate(valid[:,None].float(),size,mode='nearest')[:,0]>0
 ref=reference.detach();source=y.clamp(0,20)
 fg=ref.clone();fg[:,0]=-1e4;fg.scatter_(1,source[:,None],-1e4)
 rival=fg.argmax(1)
 gap=ref.gather(1,source[:,None])[:,0]-ref.gather(1,rival[:,None])[:,0]
 member=(ref[:,1:].topk(2,1).indices+1==source[:,None]).any(1)
 eligible=member&area&(y>=16)&(y<21)&(ref.argmax(1)>0)&(gap<math.log(2))
 for cam in [cams,aux_cams]:
  cam=F.interpolate(cam.detach(),size,mode='bilinear',align_corners=False)
  top=cam.max(1)
  eligible &= (top.indices+1==y)&(top.values>=.7)
 # Move at most log(2) from the frozen reference. No forced large flip.
 desired=torch.minimum(gap+math.log(2),torch.full_like(gap,math.log(2)))
 return {'source':source,'rival':rival,'eligible':eligible,'valid':area,'margin':desired,'reference_gap':gap}

class PairEvidence(torch.nn.Module):
 def __init__(self):
  super().__init__()
  for key in ['pixels','image_exposures','active_pixels']:
   self.register_buffer(key,torch.zeros(21,21,dtype=torch.float64))
  self.register_buffer('last_seen',torch.full((21,21),-1,dtype=torch.long))
  self.register_buffer('updates',torch.zeros((),dtype=torch.long))
 @torch.no_grad()
 def supported(self):
  return (self.image_exposures>=3)&(self.last_seen>=0)&(self.updates-1-self.last_seen<=50)
 @torch.no_grad()
 def update(self,t,student):
  ids=t['source']*21+t['rival'];mask=t['eligible']
  gap=student.detach().gather(1,t['source'][:,None])[:,0]-student.detach().gather(1,t['rival'][:,None])[:,0]
  active=mask&(gap<t['margin']);exposures=torch.zeros_like(self.pixels)
  for code,m in zip(ids,mask):exposures.flatten()[code[m].unique()]+=1
  pixels=torch.bincount(ids[mask],minlength=441).reshape(21,21).double()
  violations=torch.bincount(ids[active],minlength=441).reshape(21,21).double()
  packed=distributed_sum(torch.stack([pixels,exposures,violations]))
  self.pixels+=packed[0];self.image_exposures+=packed[1];self.active_pixels+=packed[2]
  self.last_seen[packed[0]>0]=self.updates;self.updates+=1
 def export(self):
  return {key:getattr(self,key).cpu().tolist() for key in ['pixels','image_exposures','active_pixels','last_seen','updates']}

def separation_loss(student,t,supported):
 mask=t['eligible']&supported[t['source'],t['rival']]
 gap=student.gather(1,t['source'][:,None])[:,0]-student.gather(1,t['rival'][:,None])[:,0]
 violation=F.relu(t['margin']-gap)/math.log(2)
 # A fixed reference cohort, not a denominator shrinking with current errors.
 # One pseudo-count per image bounds the influence of tiny supported islands.
 denom=t['eligible'].sum()+student.shape[0]
 loss=global_ratio((violation.square()*mask).sum(),denom)
 nums=distributed_sum(torch.stack([t['eligible'].sum(),mask.sum(),(mask&(violation>0)).sum()]).float())
 return loss,{'reference_cohort':int(nums[0]),'supported':int(nums[1]),'active':int(nums[2]),'loss':float(loss.detach())}
