"""Offline old-class image-label diagnostics; annotations never enter prediction."""
import os,sys,json,argparse,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset,Subset
from datasets.voc import VOC12ClsDataset
from model.model_seg_neg import network
UUIDS='GPU-82e069d7-189e-06b0-faed-b08e1794a981,GPU-2d0a204c-101b-1149-06e1-e3528d9f38bf'
class Images(Dataset):
 def __init__(self,stage):
  self.ds=VOC12ClsDataset(root_dir='/data/zhonggai/coco/PascalVOC12',name_list_dir=str(ROOT/'datasets/voc'),split='train',stage='train',tasks='10-5',step=stage,aug=False)
 def __len__(self):return len(self.ds)
 def __getitem__(self,i):
  name,image,tags=self.ds[i]
  return name,F.interpolate(image[None],(448,448),mode='bilinear',align_corners=False)[0],torch.as_tensor(tags)
def metrics(pred,known,gt):
 pos=known&pred;neg=known&~pred
 tp=int((pos&gt).sum());fp=int((pos&~gt).sum());fn=int((~pos&gt).sum());tn=int((neg&~gt).sum());wrong_neg=int((neg&gt).sum())
 def div(a,b):return a/b if b else None
 return dict(tp=tp,fp=fp,fn_including_abstention=fn,precision=div(tp,tp+fp),recall=div(tp,int(gt.sum())),f1=div(2*tp,2*tp+fp+fn),coverage=float(known.mean()),selective_accuracy=div(tp+tn,int(known.sum())),negative_precision=div(tn,tn+wrong_neg),wrong_negative=wrong_neg,positive_count=int(pos.sum()),unknown_positive=int((~known&gt).sum()))
def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',type=int,choices=[1,2],required=True);p.add_argument('--teacher',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==UUIDS
 out=Path(a.output).resolve();assert out.is_relative_to(ROOT) and not out.exists()
 dist.init_process_group('nccl');rank=dist.get_rank();world=dist.get_world_size();torch.cuda.set_device(rank);torch.set_num_threads(4)
 torch.set_float32_matmul_precision('high');torch.backends.cudnn.allow_tf32=True
 classes=[11]+[5]*(a.stage-1);k=sum(classes)-1
 net=network('vit_base_patch16_224',num_classes=k+1,classes_list=classes,pretrained=False,init_momentum=.9,aux_layer=-3)
 raw=torch.load(a.teacher,map_location='cpu',weights_only=True);net.load_state_dict({n.removeprefix('module.'):v for n,v in raw['model_state'].items()},strict=True)
 del raw
 net.cuda().eval().requires_grad_(False);ds=Images(a.stage)
 loader=DataLoader(Subset(ds,range(rank,len(ds),world)),batch_size=4,num_workers=2,shuffle=False)
 rows=[];start=time.time()
 with torch.inference_mode():
  for names,x,annotations in loader:
   x=x.cuda();n=len(x)
   main,aux,*_=net(torch.cat([x,x.flip(-1)]),step0=True)
   probs=torch.stack([main[:n].sigmoid(),aux[:n].sigmoid(),main[n:].sigmoid(),aux[n:].sigmoid()],1).cpu().numpy()
   # Join annotations only after independent predictions have been produced.
   for name,prob,gt in zip(names,probs,annotations[:,:k].numpy()):rows.append((str(name),prob,gt))
   if len(rows)%200<4:print(json.dumps({'rank':rank,'images':len(rows),'seconds':time.time()-start}),flush=True)
 gathered=[None]*world;dist.all_gather_object(gathered,rows)
 if rank==0:
  rows=sorted([row for part in gathered for row in part],key=lambda z:z[0]);names=np.array([r[0] for r in rows]);prob=np.stack([r[1] for r in rows]);gt=np.stack([r[2] for r in rows]).astype(bool)
  assert len(names)==len(set(names))==len(ds)
  main=prob[:,0]>.5;aux=prob[:,1]>.5
  policies={'main_threshold_0.5':(main,np.ones_like(main)),'aux_threshold_0.5':(aux,np.ones_like(main)),'dual_head_consensus':(main&aux,main==aux),'mean_two_heads_0.5':(prob[:,:2].mean(1)>.5,np.ones_like(main)),'two_view_dual_head_consensus':((prob>.5).all(1),((prob>.5).all(1)|(prob<=.5).all(1)))}
  result={'stage':a.stage,'images':len(ds),'old_classes':k,'input':'square448 original+horizontal flip','teacher':str(Path(a.teacher).resolve()),'teacher_sha256':hashlib.sha256(Path(a.teacher).read_bytes()).hexdigest(),'predictions_use_gt':False,'annotation_use':'offline image-label diagnosis only, never read by training OLC','policies':{name:metrics(*value,gt) for name,value in policies.items()},'per_class':{name:[metrics(value[0][:,c],value[1][:,c],gt[:,c]) for c in range(k)] for name,value in policies.items()},'time':time.time()}
  out.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(out.with_suffix('.npz'),names=names,probabilities=prob,diagnostic_gt=gt);out.write_text(json.dumps(result,indent=2));print(json.dumps({key:value for key,value in result.items() if key!='per_class'}),flush=True)
 dist.destroy_process_group()
if __name__=='__main__':main()
