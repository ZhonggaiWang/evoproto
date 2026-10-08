from common import *
import argparse,json,time,math
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader,Subset
from network import UnifiedNet,load_parent
from data import config
from datasets.voc import VOC12SegDataset,class_list
from mechanism import calibrate,DirectedEvidence

def hist(gt,pred,mask=None):
 ok=(gt>=0)&(gt<21)
 if mask is not None:ok&=mask
 return np.bincount(gt[ok].astype(np.int64)*21+pred[ok],minlength=441).reshape(21,21)

def main():
 p=argparse.ArgumentParser();p.add_argument('--rank',type=int,required=True);p.add_argument('--mode',choices=['diagnose','smoke','formal'],required=True);a=p.parse_args()
 torch.set_num_threads(1);torch.cuda.set_device(a.rank);torch.set_float32_matmul_precision('high');torch.backends.cudnn.allow_tf32=True;torch.backends.cudnn.benchmark=False
 cfg=config();m=load_parent() if a.mode=='diagnose' else UnifiedNet()
 if a.mode!='diagnose':m.load_state_dict(torch.load(U/a.mode/'model_final.pth',map_location='cpu',weights_only=True,mmap=True)['model_state'],strict=True)
 m.cuda().eval();ds=VOC12SegDataset(root_dir=cfg['data_folder'],name_list_dir=cfg['list_folder'],split='val',stage='val',aug=False,ignore_index=255,num_classes=21,tasks='10-5',step=2);ds.label_dir=cfg['val_label_dir']
 if a.mode=='diagnose':ds.name_list=ds.name_list[:256]
 if a.mode=='smoke':ds.name_list=ds.name_list[:16]
 loader=DataLoader(Subset(ds,list(range(a.rank,len(ds),8))),batch_size=1,shuffle=False,num_workers=2)
 hists={k:np.zeros((21,21),np.int64) for k in ['square448','aspect672']};anchors=np.zeros((21,21),np.int64);confusion=np.zeros((21,21),np.int64);names=[]
 with torch.inference_mode():
  for nb,image,label,tags in loader:
   image=image.cuda();hw=image.shape[-2:];x=F.interpolate(image,size=(448,448),mode='bilinear',align_corners=False);out=m(x)
   preds={'square448':F.interpolate(out[0],size=hw,mode='bilinear',align_corners=False).argmax(1)[0].cpu().numpy()}
   if a.mode=='diagnose':
    # Diagnostic oracle role: permitted current-new image tags, GT mask only after evidence.
    e=calibrate(out,tags[:,15:].cuda().float(),torch.ones_like(out[0][:,0],dtype=torch.bool))
    weak=e['labels'][0].cpu().numpy();gt=F.interpolate(label[:,None].float(),size=weak.shape,mode='nearest')[0,0].numpy().astype(np.int64)
    anchors+=hist(gt,weak,weak<21)
    source=weak;pred=out[0].argmax(1)[0].cpu().numpy()
    confusion+=hist(source,pred,e['foreground'][0].cpu().numpy())
   else:
    scale=672/math.sqrt(hw[0]*hw[1]);size=tuple(max(16,round(v*scale/16)*16) for v in hw)
    z=m(F.interpolate(image,size=size,mode='bilinear',align_corners=False))[0]
    preds['aspect672']=F.interpolate(z,size=hw,mode='bilinear',align_corners=False).argmax(1)[0].cpu().numpy()
   gt=label[0].numpy()
   for key,pred in preds.items():hists[key]+=hist(gt,pred)
   names.append(str(nb[0]))
 atomic_json(U/a.mode/f'eval_rank{a.rank}.json',dict(images=names,histograms={k:v.tolist() for k,v in hists.items()},anchor_gt=anchors.tolist(),directed_weak_confusion=confusion.tolist(),checkpoint_sha256=PARENT_SHA if a.mode=='diagnose' else digest(U/a.mode/'model_final.pth'),utc=now()))
 print('DONE',a.mode,a.rank,flush=True)

if __name__=='__main__':main()
