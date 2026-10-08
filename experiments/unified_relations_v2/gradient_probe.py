from common import *
import json,random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from network import load_parent
from data import TrainImages
from mechanism import calibrate,ald_loss,relation_loss

torch.set_num_threads(1);torch.cuda.set_device(0);torch.manual_seed(0);np.random.seed(0);random.seed(0)
t=load_parent().cuda().eval().requires_grad_(False);s=load_parent().cuda().train()
records=[]
for n,(ids,x,tags,valid) in enumerate(DataLoader(TrainImages(),batch_size=4,shuffle=False,num_workers=0)):
 if n==4:break
 x=x.cuda();tags=tags.cuda();valid=F.avg_pool2d(valid.cuda()[:,None],16,16)[:,0]>.999
 with torch.no_grad():ref=t(x);e=calibrate(ref,tags,valid)
 out=s(x);a=ald_loss(out[0],e);kd,sep,_=relation_loss(out[0],ref[0],e,torch.zeros(21,21,device=x.device,dtype=torch.bool))
 cls=.5*(F.binary_cross_entropy_with_logits(out[3],e['image_target'])+F.binary_cross_entropy_with_logits(out[4],e['image_target']))
 params=[p for name,p in s.named_parameters() if name.startswith('encoder.') and p.requires_grad]
 gc=torch.autograd.grad(cls,params,retain_graph=True,allow_unused=True);gd=torch.autograd.grad(.1*(a+kd+sep),params,allow_unused=True)
 normc=sum((g*g).sum() for g in gc if g is not None).sqrt();normd=sum((g*g).sum() for g in gd if g is not None).sqrt()
 dot=sum((g*h).sum() for g,h in zip(gc,gd) if g is not None and h is not None)
 records.append(dict(batch=n,classification_norm=float(normc),dense_norm=float(normd),ratio=float(normc/normd.clamp_min(1e-12)),cosine=float(dot/(normc*normd).clamp_min(1e-12))))
atomic_json(U/'gradient_probe.json',dict(records=records,role='Training images only. Diagnostic of shared encoder gradients at the same parent initialization, no tuning or GT masks.',utc=now()))
print(json.dumps(records))
