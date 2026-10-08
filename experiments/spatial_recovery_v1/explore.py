"""Frozen-checkpoint spatial diagnosis. GT is used only after predictions."""
from pathlib import Path
import sys, os, json, math, time, argparse
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto')
E=R/'experiments/spatial_recovery_v1'; U=R/'runs/spatial_recovery_v1'
sys.path.insert(0,str(R/'experiments/kd_pixel_v2'))
from run_kd import environment
os.environ.update(environment('8card'))
from kd_runtime import safe_path,atomic_json,digest,now
S=R/'experiments/local_sep_v5/src';sys.path.insert(0,str(S))
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader,Subset
from scipy.ndimage import distance_transform_edt, label as components
from datasets import voc
from model.model_seg_neg import network

MODES=['square448','aspect448','square672']
def histogram(gt,pred,mask=None):
 valid=(gt>=0)&(gt<21)
 if mask is not None:valid &= mask
 return np.bincount(21*gt[valid].astype(np.int64)+pred[valid],minlength=441).reshape(21,21)

def regions(gt):
 valid=gt!=255;edge=np.zeros_like(valid)
 d=(gt[1:]!=gt[:-1])&valid[1:]&valid[:-1];edge[1:]|=d;edge[:-1]|=d
 d=(gt[:,1:]!=gt[:,:-1])&valid[:,1:]&valid[:,:-1];edge[:,1:]|=d;edge[:,:-1]|=d
 radius=max(2,round(.01*math.hypot(*gt.shape)))
 near=(distance_transform_edt(~edge)<=radius) if edge.any() else np.zeros_like(valid)
 small=np.zeros_like(valid);medium=np.zeros_like(valid);large=np.zeros_like(valid)
 for c in range(1,21):
  cc,n=components(gt==c);counts=np.bincount(cc.ravel());counts[0]=0
  frac=counts/valid.sum();small|=(cc>0)&(frac[cc]<.01)
  medium|=(cc>0)&(frac[cc]>=.01)&(frac[cc]<.1);large|=(cc>0)&(frac[cc]>=.1)
 return dict(boundary=near,interior=~near,small_foreground=small,medium_foreground=medium,large_foreground=large)

def dimensions(mode,h,w):
 if mode=='square448':return 448,448
 if mode=='square672':return 672,672
 scale=448/math.sqrt(h*w)
 return max(16,round(h*scale/16)*16),max(16,round(w*scale/16)*16)

def main():
 p=argparse.ArgumentParser();p.add_argument('--rank',type=int,required=True);p.add_argument('--limit',type=int,default=0);a=p.parse_args()
 torch.set_num_threads(1);torch.cuda.set_device(a.rank);torch.manual_seed(0)
 torch.set_float32_matmul_precision('high');torch.backends.cudnn.allow_tf32=True
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 cfg=json.loads((R/'runs/local_sep_v5/formal/study.json').read_text())['config']
 ck=R/'runs/local_sep_v5/formal/10-5/step2/checkpoints/model_final.pth'
 expected='0770ac07bcb343d510bbb6287526231352891e16f0c3e11e4b0d36a14fca4177'
 assert digest(ck)==expected
 m=network(backbone=cfg['backbone'],num_classes=21,classes_list=[11,5,5],pretrained=False,init_momentum=cfg['momentum'],aux_layer=cfg['aux_layer'])
 m.load_state_dict(torch.load(ck,map_location='cpu',weights_only=True,mmap=True)['model_state'],strict=True);m.cuda().eval()
 ds=voc.VOC12SegDataset(root_dir=cfg['data_folder'],name_list_dir=cfg['list_folder'],split='val',stage='val',aug=False,ignore_index=255,num_classes=21,tasks='10-5',step=2)
 ds.label_dir=cfg['val_label_dir']
 if a.limit:ds.name_list=ds.name_list[:a.limit]
 loader=DataLoader(Subset(ds,list(range(a.rank,len(ds),8))),batch_size=1,shuffle=False,num_workers=2)
 keys=['all','boundary','interior','small_foreground','medium_foreground','large_foreground']
 hist={mode:{k:np.zeros((21,21),np.int64) for k in keys} for mode in MODES}
 names=[];cache={};elapsed={mode:0. for mode in MODES};start=time.time()
 out=safe_path(U/('smoke' if a.limit else 'diagnosis'));out.mkdir(parents=True,exist_ok=True)
 with torch.inference_mode():
  for names_batch,images,labels,_ in loader:
   name=names_batch[0];images=images.cuda();hw=images.shape[-2:];assert tuple(hw)==tuple(labels.shape[-2:])
   # No label, image tag, or GT-derived quantity reaches prediction.
   outputs={};record={}
   for mode in MODES:
    inp=F.interpolate(images,size=dimensions(mode,*hw),mode='bilinear',align_corners=False)
    torch.cuda.synchronize();t=time.time();z=m(inp)[1];torch.cuda.synchronize();elapsed[mode]+=time.time()-t
    assert torch.isfinite(z).all()
    record[mode]=z[0].cpu();outputs[mode]=F.interpolate(z,size=hw,mode='bilinear',align_corners=False).argmax(1)[0].cpu().numpy()
   gt=labels[0].numpy();masks=regions(gt)
   for mode,pred in outputs.items():
    hist[mode]['all']+=histogram(gt,pred)
    for key,mask in masks.items():hist[mode][key]+=histogram(gt,pred,mask)
   cache[name]=record;names.append(name)
   if len(names)%25==0:print(a.rank,len(names),round(time.time()-start,1),flush=True)
 torch.save(cache,safe_path(out/f'logits_rank{a.rank}.pth'))
 atomic_json(out/f'rank{a.rank}.json',dict(rank=a.rank,images=names,histograms={m:{k:v.tolist() for k,v in h.items()} for m,h in hist.items()},forward_seconds=elapsed,wall_seconds=time.time()-start,checkpoint_sha256=expected,source_sha256=digest(Path(__file__)),utc=now(),gt_role='Diagnostics only, never model input'))
 print('DONE',a.rank,flush=True)

if __name__=='__main__':main()
