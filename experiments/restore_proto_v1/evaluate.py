"""Reproducible square448/aspect672 evaluation without image-tag oracle."""
import os,sys,json,math,argparse,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import torch
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader,Subset
from model.model_seg_neg import network
from datasets.voc import VOC12SegDataset,class_list


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--stage',type=int,required=True);p.add_argument('--output',required=True);p.add_argument('--square-only',action='store_true');p.add_argument('--expected-square',type=float);a=p.parse_args()
    output=Path(a.output).resolve();assert output.is_relative_to(ROOT)
    dist.init_process_group('nccl');rank=dist.get_rank();world=dist.get_world_size();torch.cuda.set_device(rank);torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high');torch.backends.cudnn.allow_tf32=True;torch.backends.cudnn.benchmark=False
    classes=[11]+[5]*a.stage;k=sum(classes)
    net=network('vit_base_patch16_224',num_classes=k,classes_list=classes,pretrained=False,init_momentum=.9,aux_layer=-3)
    raw=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    net.load_state_dict({key.removeprefix('module.'):v for key,v in raw['model_state'].items()},strict=True);del raw
    net.cuda().eval()
    ds=VOC12SegDataset(root_dir='/data/zhonggai/coco/PascalVOC12',name_list_dir=str(ROOT/'datasets/voc'),split='val',stage='val',aug=False,ignore_index=255,num_classes=21,tasks='10-5',step=a.stage)
    ds.label_dir='/data/zhonggai/coco/PascalVOC12/SegmentationClass'
    loader=DataLoader(Subset(ds,range(rank,len(ds),world)),batch_size=1,num_workers=2,shuffle=False)
    modes=['square448'] if a.square_only else ['square448','aspect672']
    matrices={m:torch.zeros(2,k,k,device='cuda',dtype=torch.float64) for m in modes}
    with torch.inference_mode():
        for _,image,label,_ in loader:
            image=image.cuda();label=label.cuda();h,w=image.shape[-2:]
            for mode in modes:
                scale=672/math.sqrt(h*w)
                size=(448,448) if mode=='square448' else tuple(max(16,round(x*scale/16)*16) for x in (h,w))
                _,_,main,_,_,proto,_=net(F.interpolate(image,size=size,mode='bilinear',align_corners=False),cam_grad=True)
                valid=(label>=0)&(label<k)
                for i,z in enumerate([main,proto]):
                    pred=F.interpolate(z,size=(h,w),mode='bilinear',align_corners=False).argmax(1)
                    matrices[mode][i]+=torch.bincount((label[valid]*k+pred[valid]).flatten(),minlength=k*k).reshape(k,k)
    result={'stage':a.stage,'images':len(ds),'checkpoint':str(Path(a.checkpoint).resolve()),'prediction_uses_gt_tags':False,'aspect_rule':'area approximately 672 squared; each dimension rounded to nearest multiple of16','results':{}}
    for mode,mat in matrices.items():
        dist.all_reduce(mat)
        result['results'][mode]={}
        for i,name in enumerate(['main','prototype']):
            hm=mat[i];den=hm.sum(0)+hm.sum(1)-hm.diag();iou=torch.where(den>0,100*hm.diag()/den,torch.nan)
            def avg(x):return float(x.nanmean()) if x.numel() else None
            old=10 if a.stage==1 else 15 if a.stage==2 else 10
            result['results'][mode][name]={'miou':avg(iou),'foreground':avg(iou[1:]),'previous_foreground':avg(iou[1:old+1]),'current_foreground':avg(iou[old+1:]) if a.stage else None,'class_iou':[float(x) if x.isfinite() else None for x in iou],'histogram':hm.cpu().tolist()}
    if rank==0:
        h=hashlib.sha256()
        with open(a.checkpoint,'rb') as f:
            for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
        result['checkpoint_sha256']=h.hexdigest()
        output.write_text(json.dumps(result,indent=2,allow_nan=False))
        print(json.dumps({m:{n:v['miou'] for n,v in r.items()} for m,r in result['results'].items()}),flush=True)
    score=result['results']['square448']['main']['miou']
    if a.expected_square is not None and abs(score-a.expected_square)>.2:raise RuntimeError(f'Initial checkpoint evaluation mismatch: {score} vs {a.expected_square}')
    dist.destroy_process_group()

if __name__=='__main__':main()
