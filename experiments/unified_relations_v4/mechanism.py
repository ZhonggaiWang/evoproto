"""ALD -> one directed evidence graph -> complementary KD / SEP.

KD preserves the collapsed pair mass and all other relations, including BG.
SEP corrects only the conditional order inside a supported confused pair.
No class-specific branches, extra teacher, margin schedule, or new parameters.
"""
import torch
import torch.nn.functional as F
import torch.distributed as dist

def global_mean(total,count):
 n=total.detach().clone();d=count.detach().clone();world=1
 if dist.is_initialized():
  dist.all_reduce(n);dist.all_reduce(d);world=dist.get_world_size()
 return (n+world*(total-total.detach()))/d.clamp_min(1)

@torch.no_grad()
def normalize_cam(cam,valid):
 cam=cam.relu();mask=valid[:,None]
 low=cam.masked_fill(~mask,torch.inf).flatten(2).amin(2)[:,:,None,None]
 high=cam.masked_fill(~mask,-torch.inf).flatten(2).amax(2)[:,:,None,None]
 return ((cam-low)/(high-low).clamp_min(1e-5)).clamp(0,1).masked_fill(~mask,0)

@torch.no_grad()
def calibrate(outputs,new_tags,valid,high=.7,low=.25):
 logits,main,aux,cls,cls_aux=outputs
 k=logits.shape[1];old=k-1-new_tags.shape[1]
 assert main.shape[1]==aux.shape[1]==k-1
 # Only current-class image tags are ground truth; no inferred old absences.
 allowed=torch.ones_like(cls,dtype=torch.bool);allowed[:,old:]=new_tags.bool()
 present=torch.cat([(cls[:,:old]>0)&(cls_aux[:,:old]>0),new_tags.bool()],1)
 m=normalize_cam(main,valid)*present[:,:,None,None]
 a=normalize_cam(aux,valid)*present[:,:,None,None]
 mv,mi=m.max(1);av,ai=a.max(1)
 labels=torch.full_like(mi,255)
 fg=valid&(mi==ai)&(mv>=high)&(av>=high)
 # Old labels have no current GT support: require dense/CAM agreement.
 # New labels have current image evidence, so they can challenge the teacher.
 fg &= ((mi+1)>old)|(logits.argmax(1)==mi+1)
 labels[fg]=mi[fg]+1
 bg=valid&(mv<low)&(av<low)&(logits.argmax(1)==0)
 labels[bg]=0
 return dict(labels=labels,valid=valid,allowed=allowed,foreground=fg,background=bg)

@torch.no_grad()
def calibrated_reference(logits,evidence):
 allowed=torch.cat([torch.ones_like(evidence['allowed'][:,:1]),evidence['allowed']],1)
 return logits.masked_fill(~allowed[:,:,None,None],-1e4).softmax(1)

class DirectedEvidence:
 """Distinct-image support, one matrix, synchronised across all workers."""
 def __init__(self,classes,images,device,min_images=3):
  self.seen=torch.zeros(classes*classes,images,dtype=torch.bool,device=device)
  self.k=classes;self.min_images=min_images
 def counts(self):return self.seen.sum(1).reshape(self.k,self.k)
 def supported(self):
  s=self.counts()>=self.min_images;s.fill_diagonal_(False);s[0]=False;return s
 @torch.no_grad()
 def update(self,evidence,reference,image_ids):
  source=evidence['labels'];rival=calibrated_reference(reference,evidence).argmax(1)
  edge=evidence['foreground']&(rival!=source)
  present=torch.zeros(len(image_ids),self.k*self.k,dtype=torch.bool,device=reference.device)
  for i in range(len(image_ids)):
   pairs=(source[i][edge[i]]*self.k+rival[i][edge[i]]).unique();present[i,pairs]=True
  if dist.is_initialized():
   packets=[torch.empty_like(present) for _ in range(dist.get_world_size())];ids=[torch.empty_like(image_ids) for _ in packets]
   dist.all_gather(packets,present);dist.all_gather(ids,image_ids);present=torch.cat(packets);image_ids=torch.cat(ids)
  for row,image_id in zip(present,image_ids):self.seen[:,image_id]|=row

def relation_loss(student,reference,evidence,supported):
 # One calibrated full distribution. Unselected BG/FG mass is retained.
 # Student is unmasked: absent-class probability is penalized by the same KL.
 logp=student.log_softmax(1)
 with torch.no_grad():
  q=calibrated_reference(reference.detach(),evidence)
  source=evidence['labels'].clamp(0,student.shape[1]-1)
  rival=q.argmax(1)
  selected=evidence['foreground']&(source!=rival)&supported[source,rival]
  valid=evidence['valid']
  selected &= valid
 logq=q.clamp_min(1e-12).log()
 full=(q*(logq-logp)).sum(1)
 def gather(v,c):return v.gather(1,c[:,None])[:,0]
 qa,qb=gather(q,source),gather(q,rival);mass=qa+qb
 lpa,lpb=gather(logp,source),gather(logp,rival)
 lpm=torch.logaddexp(lpa,lpb)
 ca=(qa/mass.clamp_min(1e-12)).clamp_min(1e-12);cb=(qb/mass.clamp_min(1e-12)).clamp_min(1e-12)
 original_cond=qa*(ca.log()-(lpa-lpm))+qb*(cb.log()-(lpb-lpm))
 kd=torch.where(selected,full-original_cond,full)
 # Minimum correction: remove the erroneous ordering, with no tuned margin.
 # When source already wins, SEP is exactly zero; KD keeps the pair collapsed.
 active=selected&(lpa<lpb)
 sep=mass*(-.6931471805599453-.5*(lpa+lpb-2*lpm))
 kd_loss=global_mean((kd*valid).sum(),valid.sum())
 sep_loss=global_mean((sep*active).sum(),valid.sum())
 return kd_loss,sep_loss,dict(selected=selected,active=active,kd_map=kd,sep_map=sep,q=q)
