"""Class-compatible, confidence-anchored foreground/background margin diffusion.

No fitted parameters, image tags, GT, or validation-specific class rules.
Foreground logits stay exact. Only uncertain foreground/background odds change.
"""
import math
import torch
import torch.nn.functional as F

def shifted(x,dy,dx,pad=8):
 h,w=x.shape[-2:];q=F.pad(x,(pad,pad,pad,pad),mode='replicate')
 return q[...,pad+dy:pad+dy+h,pad+dx:pad+dx+w]

@torch.no_grad()
def boundary_margin(image,logits,iterations=5):
 assert image.shape[0]==logits.shape[0]==1 and image.shape[-2:]==logits.shape[-2:]
 if iterations==0:return logits.clone()
 top,cls=logits[:,1:].max(1,keepdim=True);margin=top-logits[:,:1];limit=math.log(2)
 # +/- log(2) means a two-to-one pairwise odds anchor, not calibrated accuracy.
 field=margin.clamp(-limit,limit);uncertain=margin.abs()<limit
 offsets=[(d*y,d*x) for d in (1,2,4,8) for y,x in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]]
 diffs=torch.stack([(image-shifted(image,dy,dx)).square().mean(1,keepdim=True) for dy,dx in offsets])
 local_scale=diffs.mean(0).clamp_min(1e-6)
 weights=[]
 for i,(dy,dx) in enumerate(offsets):
  # A foreground donor must support the recipient's competing class.
  # Background donors assert exclusion only, never a different foreground label.
  compatible=(shifted(cls,dy,dx)==cls)|(shifted(margin,dy,dx)<=0)
  weights.append(torch.exp(-diffs[i]/local_scale)*compatible)
 denom=1+sum(weights)
 for _ in range(iterations):
  proposal=(field+sum(w*shifted(field,*offset) for w,offset in zip(weights,offsets)))/denom
  field=torch.where(uncertain,proposal,field)
 out=logits.clone();out[:,:1]=torch.where(uncertain,top-field,logits[:,:1])
 return out

def test():
 torch.manual_seed(0);z=torch.randn(1,21,13,17);im=torch.rand(1,3,13,17)
 out=boundary_margin(im,z);fg=z[:,1:].max(1,keepdim=True).values
 assert torch.equal(out[:,1:],z[:,1:])
 anchors=(fg-z[:,:1]).abs()>=math.log(2)
 assert torch.equal(out[:,:1][anchors],z[:,:1][anchors])
 pred=out.argmax(1);assert ((pred==0)|(pred==z[:,1:].argmax(1)+1)).all()
 assert torch.isfinite(out).all()
 assert torch.equal(boundary_margin(im,z,iterations=0),z)
 return ['foreground_logits_exact','confident_BG_FG_anchors_exact','no_new_foreground_class','finite','zero_iteration_identity']
