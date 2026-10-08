"""Local paired logit correction with conserved pair mass at the native grid."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def redistribute(logits,raw,limit=math.log(2)):
 top=logits[:,1:].topk(2,1).indices+1
 a,b=top[:,0:1],top[:,1:2]
 za,zb=logits.gather(1,a),logits.gather(1,b)
 delta=limit*torch.tanh(raw.gather(1,a)-raw.gather(1,b))
 # log((exp(za+delta)+exp(zb))/(exp(za)+exp(zb))).
 # Identical operations cancel exactly when delta is zero.
 shift=F.softplus(za-zb+delta)-F.softplus(za-zb)
 da,db=delta-shift,-shift
 permitted=((a>=16)|(b>=16))&(logits.argmax(1,keepdim=True)>0)
 # Preserve native-grid foreground/background decisions as well as BG mass.
 permitted &= torch.maximum(za+da,zb+db)>logits[:,0:1]
 da=da*permitted;db=db*permitted
 out=logits.clone();out.scatter_add_(1,a,da);out.scatter_add_(1,b,db)
 return out

class PairAdapter(nn.Module):
 def __init__(self,channels=512,hidden=64):
  super().__init__()
  self.spatial=nn.Conv2d(channels,hidden,3,padding=1)
  self.output=nn.Conv2d(hidden,21,1,bias=False)
  nn.init.zeros_(self.output.weight)
 def forward(self,features,logits):
  x=F.normalize(features,dim=1)*math.sqrt(features.shape[1])
  raw=self.output(F.relu(self.spatial(x)))
  return redistribute(logits,raw)
