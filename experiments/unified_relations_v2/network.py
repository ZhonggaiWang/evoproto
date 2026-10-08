"""One encoder, two existing classification views, one segmentation decoder.

Unused projection/prototype branches and the pair adapter are absent.
Parameter names of the remaining network match the original checkpoint.
"""
from common import *
import torch
from torch import nn
import torch.nn.functional as F
from model.backbone import vit_base_patch16_224

class Classifier(nn.ModuleList):
 def weight(self):return torch.cat([m.weight for m in self],0)
 def forward(self,x):return torch.cat([m(x) for m in self],1)

class Decoder(nn.Module):
 def __init__(self,classes):
  super().__init__();self.conv6=nn.Conv2d(768,512,3,padding=5,dilation=5,bias=False)
  self.conv7=nn.Conv2d(512,512,3,padding=5,dilation=5,bias=False)
  self.conv8=Classifier([nn.Conv2d(512,c,1,bias=False) for c in classes])
 def forward(self,x):return self.conv8(F.relu(self.conv7(F.relu(self.conv6(x)))))

class UnifiedNet(nn.Module):
 def __init__(self,classes=(11,5,5)):
  super().__init__();self.encoder=vit_base_patch16_224(pretrained=False,aux_layer=-3)
  self.encoder.head=nn.Identity();self.decoder=Decoder(classes)
  self.classifier=Classifier([nn.Conv2d(768,c-(i==0),1,bias=False) for i,c in enumerate(classes)])
  self.aux_classifier=Classifier([nn.Conv2d(768,c-(i==0),1,bias=False) for i,c in enumerate(classes)])
 def forward(self,x):
  _,features,aux=self.encoder.forward_features(x);h,w=x.shape[-2]//16,x.shape[-1]//16
  f=features.transpose(1,2).reshape(x.shape[0],768,h,w);a=aux.transpose(1,2).reshape_as(f)
  main_cam=F.conv2d(f,self.classifier.weight());aux_cam=F.conv2d(a,self.aux_classifier.weight())
  # Original image classifier pools features before applying class weights.
  cls=self.classifier(F.adaptive_max_pool2d(f,1)).flatten(1)
  cls_aux=self.aux_classifier(F.adaptive_max_pool2d(a,1)).flatten(1)
  return self.decoder(f),main_cam,aux_cam,cls,cls_aux

def load_parent():
 assert digest(PARENT)==PARENT_SHA
 net=UnifiedNet();state=torch.load(PARENT,map_location='cpu',weights_only=True,mmap=True)['model_state']
 keep={k:v for k,v in state.items() if k in net.state_dict()}
 removed=set(state)-set(keep)
 assert all(k.startswith(('proj_head.','proj_head_t.','decoder.class_prototypes.','encoder.head.')) for k in removed),removed
 net.load_state_dict(keep,strict=True)
 return net
