"""Directed local feature separation. No GT input, frozen multimodal anchors."""
import torch
import torch.nn.functional as F
from model.pixel_kd import global_ratio,distributed_sum

@torch.no_grad()
def local_evidence(labels,cams,valid,pair_targets,size):
 y=F.interpolate(labels.detach()[:,None].float(),size,mode='nearest')[:,0].long()
 area=F.interpolate(valid.detach()[:,None].float(),size,mode='nearest')[:,0]>0
 cam=F.interpolate(cams.detach(),size,mode='bilinear',align_corners=False)
 top=cam.topk(2,1);rival=pair_targets[y.clamp(0,20)]
 mask=area&(y>0)&(y<21)&(rival>0)&(rival!=y)&(top.indices[:,0]+1==y)&(top.values[:,0]>=.7)
 return y,rival,mask,area

def separation_loss(features,labels,cams,valid,pair_targets,centers,available,margin=.05):
 """Stop after a 0.05 cosine gap. Background and unselected pixels get no SEP.

 Normalize by all eligible anchor pixels, never by remaining violations.
 Both positive and negative reference centers are frozen; old/new anchors
 share the same feature space. Matrix rates only select, never scale gradients.
 """
 y,rival,eligible,area=local_evidence(labels,cams,valid,pair_targets,features.shape[-2:])
 centers=centers.detach();available=available.detach()
 flat=F.normalize(features,dim=1).permute(0,2,3,1)
 sims=torch.einsum('nhwd,ckd->nhwck',flat,centers).amax(-1)
 positive=sims.gather(-1,y.clamp(0,20)[...,None])[...,0]
 negative=sims.gather(-1,rival.clamp(0,20)[...,None])[...,0]
 eligible=eligible&available[y.clamp(0,20)]&available[rival.clamp(0,20)]
 gap=positive-negative
 violation=F.relu((margin-gap)/.1)
 loss=global_ratio((violation.square()*eligible).sum(),eligible.sum())
 counts=distributed_sum(torch.stack([eligible.sum(),(eligible&(gap<margin)).sum(),area.sum()]).float())
 return loss,{'eligible':int(counts[0]),'active':int(counts[1]),'valid':int(counts[2]),'loss':float(loss.detach()),'margin':margin}

def retention_loss(student,reference,valid,temperature=2.):
 ref=reference.detach()/temperature
 logq=ref.log_softmax(1);q=logq.exp()
 per=(q*(logq-(student/temperature).log_softmax(1))).sum(1)*temperature**2
 area=F.interpolate(valid[:,None].float(),student.shape[-2:],mode='nearest')[:,0]
 return global_ratio((per*area).sum(),area.sum())
