"""Fixed-grid fusion probe; no changes to training or saved model weights.

Both heads were trained with sigmoid BCE. Prototype cosine logits must be
scaled by the training temperature .1 before their probabilities are mixed.
Endpoints alpha=0/1 recover the main/prototype prediction protocols.
"""
import os,sys,argparse,json,hashlib,math,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from model.model_seg_neg import network
from datasets.voc import VOC12SegDataset


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--stage',type=int,choices=[0,1,2],required=True);p.add_argument('--output',required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--threads',type=int,default=8);p.add_argument('--modes',nargs='+',choices=['square448','aspect672'],default=['square448','aspect672']);p.add_argument('--limit',type=int,default=0);p.add_argument('--expected-main',type=float);a=p.parse_args()
    out=Path(a.output).resolve();assert out.is_relative_to(ROOT);assert not out.exists();out.parent.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(a.threads);torch.set_float32_matmul_precision('high');torch.backends.cudnn.allow_tf32=True;torch.backends.cudnn.benchmark=False
    device=torch.device(a.device);classes=[11]+[5]*a.stage;k=sum(classes)
    net=network('vit_base_patch16_224',num_classes=k,classes_list=classes,pretrained=False,init_momentum=.9,aux_layer=-3)
    checkpoint=Path(a.checkpoint).resolve();h=hashlib.sha256()
    with checkpoint.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    net.load_state_dict({n.removeprefix('module.'):v for n,v in state['model_state'].items()},strict=True);del state;net.to(device).eval()
    ds=VOC12SegDataset(root_dir='/data/zhonggai/coco/PascalVOC12',name_list_dir=str(ROOT/'datasets/voc'),split='val',stage='val',aug=False,ignore_index=255,num_classes=21,tasks='10-5',step=a.stage)
    ds.label_dir='/data/zhonggai/coco/PascalVOC12/SegmentationClass'
    if a.limit:ds.name_list=ds.name_list[:a.limit]
    loader=DataLoader(ds,batch_size=1,num_workers=2,shuffle=False)
    alphas=[0.,.25,.5,.75,1.];hist={m:np.zeros((len(alphas),k,k),dtype=np.int64) for m in a.modes};per_image={m:[] for m in a.modes};names=[];started=time.time()
    with torch.inference_mode():
        for index,(name,image,label,_) in enumerate(loader):
            image=image.to(device);label=label[0].numpy();hgt,wdt=image.shape[-2:];valid=(label>=0)&(label<k)
            names.append(str(name[0]))
            for mode in a.modes:
                scale=672/math.sqrt(hgt*wdt);size=(448,448) if mode=='square448' else tuple(max(16,round(x*scale/16)*16) for x in (hgt,wdt))
                _,_,main,_,_,proto,_=net(F.interpolate(image,size=size,mode='bilinear',align_corners=False),cam_grad=True)
                main=F.interpolate(main,size=(hgt,wdt),mode='bilinear',align_corners=False)
                proto=F.interpolate(proto/.1,size=(hgt,wdt),mode='bilinear',align_corners=False)
                q_main=main.sigmoid();q_proto=proto.sigmoid();sample=[]
                for i,alpha in enumerate(alphas):
                    # Use logits for exact endpoints, avoiding sigmoid saturation ties.
                    logits=main if alpha==0 else proto if alpha==1 else (1-alpha)*q_main+alpha*q_proto
                    prediction=logits.argmax(1)[0].cpu().numpy()
                    hm=np.bincount(label[valid].astype(np.int64)*k+prediction[valid],minlength=k*k).reshape(k,k)
                    hist[mode][i]+=hm;sample.append(hm)
                per_image[mode].append(np.stack(sample))
            if (index+1)%50==0:print(json.dumps({'images':index+1,'total':len(ds),'elapsed_seconds':time.time()-started}),flush=True)
    result={'checkpoint':str(checkpoint),'checkpoint_sha256':h.hexdigest(),'stage':a.stage,'images':len(names),'device':str(device),'alphas':alphas,'fusion':'(1-alpha)*sigmoid(main_logits)+alpha*sigmoid(prototype_cosine/.1); interpolate logits before sigmoid','image_tags_used':False,'results':{},'elapsed_seconds':time.time()-started,'selection_note':'Diagnostic fixed grid; any selected alpha must be declared and applied consistently to comparative arms. Validation model selection is not independent test evidence.'}
    assert len(names)==len(set(names))==len(ds)
    for mode,mats in hist.items():
        result['results'][mode]={}
        for alpha,hm in zip(alphas,mats):
            den=hm.sum(0)+hm.sum(1)-hm.diagonal();iou=np.divide(hm.diagonal()*100.,den,out=np.full(k,np.nan),where=den>0);old=10 if a.stage<=1 else 15
            result['results'][mode][str(alpha)]={'miou':float(np.nanmean(iou)),'previous_foreground':float(np.nanmean(iou[1:old+1])),'current_foreground':float(np.nanmean(iou[old+1:])) if a.stage else None,'class_iou':[float(x) if np.isfinite(x) else None for x in iou],'histogram':hm.tolist()}
    if a.expected_main is not None:
        assert abs(result['results']['square448']['0.0']['miou']-a.expected_main)<.2,'CPU/GPU reference discrepancy'
    np.savez_compressed(out.with_suffix('.npz'),names=np.asarray(names),alphas=np.asarray(alphas),**{m:np.stack(v) for m,v in per_image.items()})
    out.write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps({m:{alpha:r['miou'] for alpha,r in rows.items()} for m,rows in result['results'].items()}),flush=True)

if __name__=='__main__':main()
