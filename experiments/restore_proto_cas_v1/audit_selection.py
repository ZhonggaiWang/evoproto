"""Offline validation diagnosis. Pixel masks are used only after selection.
Scores from this script never enter training or the confusion graph.
"""
import sys,os,json,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import torch.nn.functional as F
from model.model_seg_neg import network
from model.PAR import PAR
from datasets.voc import VOC12SegDataset
from utils.camutils import multi_scale_cam2,cam_to_label,refine_cams_with_bkg_v2
from utils import imutils
from experiments.restore_proto_v1.mechanism import ConfusionProto
from experiments.restore_proto_cas_v1.supervision import select_conflicts


def load(path,stage):
    net=network('vit_base_patch16_224',num_classes=11+5*stage,classes_list=[11]+[5]*stage,pretrained=False,init_momentum=.9,aux_layer=-3)
    state=torch.load(path,map_location='cpu',weights_only=True)['model_state']
    net.load_state_dict({k.removeprefix('module.'):v for k,v in state.items()},strict=True)
    return net.cuda().eval()


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',type=int,required=True);p.add_argument('--directory',required=True);p.add_argument('--output',required=True);p.add_argument('--limit',type=int,default=200);a=p.parse_args()
    torch.set_num_threads(4);torch.set_float32_matmul_precision('high');torch.backends.cudnn.allow_tf32=True
    d=Path(a.directory);out=Path(a.output);assert out.resolve().is_relative_to(ROOT);assert not out.exists()
    teacher=load(json.loads((d/'predecessor.json').read_text())['path'],a.stage-1)
    net=load(d/'checkpoints/model_final.pth',a.stage);k=11+5*a.stage;oc=k-5
    graphstate=json.loads((d/'confusion.json').read_text());rel=ConfusionProto(k,oc,'cuda')
    rel.ema=torch.tensor(graphstate['ema'],device='cuda');rel.mass=torch.tensor(graphstate['anchor_mass'],device='cuda')
    rel.seen=[[set(range(n)) for n in row] for row in graphstate['unique_image_support']]
    G,_=rel.graph();par=PAR(num_iter=10,dilations=[1,2,4,8,12,24]).cuda()
    ds=VOC12SegDataset(root_dir='/data/zhonggai/coco/PascalVOC12',name_list_dir=str(ROOT/'datasets/voc'),split='val',stage='val',aug=False,ignore_index=255,num_classes=21,tasks='10-5',step=a.stage)
    ds.label_dir='/data/zhonggai/coco/PascalVOC12/SegmentationClass'
    indices=np.linspace(0,len(ds)-1,min(a.limit,len(ds)),dtype=int)
    stats={m:dict(selected=0,gt_old_candidate=0,gt_new_candidate=0,gt_other=0,gt_unknown=0,valid_pixels=0) for m in ['local_pair','confusion_pair','old_to_new','prototype_pair','prototype_winner','directed_prototype','par_pair','par_directed','par_prototype','par_directed_prototype']}
    images=[]
    with torch.inference_mode():
        for number,index in enumerate(indices):
            name,im,gt,tags=ds[int(index)]
            x=F.interpolate(torch.as_tensor(im,device='cuda')[None],size=(448,448),mode='bilinear',align_corners=False)
            cls,_,old,_,_=teacher(x,step0=True)
            old=F.interpolate(old,size=(448,448),mode='bilinear',align_corners=False)
            # Only current-stage image labels; old labels are teacher predictions.
            tags=torch.cat([(cls>0).float(),torch.as_tensor(tags,device='cuda')[None,oc-1:k-1]],1)
            cams,aux=multi_scale_cam2(net,x,scales=[1.,.5,1.5]);box=torch.tensor([[0,448,0,448]])
            valid_cam,_=cam_to_label(cams,cls_label=tags,img_box=box,ignore_mid=True,bkg_thre=.5,high_thre=.7,low_thre=.25,ignore_index=255)
            labels=refine_cams_with_bkg_v2(par,imutils.denormalize_img2(x.clone()),cams=valid_cam,cls_labels=tags,high_thre=.7,low_thre=.25,ignore_index=255,img_box=box)
            local,aa,bb,_=select_conflicts(old,cams,aux,tags,labels,torch.ones_like(labels,dtype=torch.bool),G,oc,'local_pair')
            _,_,_,_,_,proto,_=net(x,cam_grad=True)
            proto=F.interpolate(proto,size=(448,448),mode='bilinear',align_corners=False)
            agrees=proto.gather(1,aa[:,None])[:,0]>proto.gather(1,bb[:,None])[:,0]
            directed=G[aa,bb]>0;both=directed|(G[bb,aa]>0)
            masks=dict(local_pair=local,confusion_pair=local&both,old_to_new=local&directed,
                       prototype_pair=local&both&agrees,prototype_winner=local&both&(proto.argmax(1)==aa),
                       directed_prototype=local&directed&agrees)
            par_new=labels.long().clamp(1,k-1)
            ca=cams*tags[:,:,None,None];cb=aux*tags[:,:,None,None]
            old_tag=tags.gather(1,(aa-1).clamp(0,oc-2).flatten(1)).reshape_as(aa)>0
            medium=(labels>=oc)&(labels<k)&(aa>0)&old_tag
            medium &= old.sigmoid().gather(1,aa[:,None])[:,0]>=.7
            medium &= ca.gather(1,(par_new-1)[:,None])[:,0]>=.25
            medium &= cb.gather(1,(par_new-1)[:,None])[:,0]>=.25
            directed2=G[aa,par_new]>0;both2=directed2|(G[par_new,aa]>0)
            agrees2=proto.gather(1,aa[:,None])[:,0]>proto.gather(1,par_new[:,None])[:,0]
            predictions={m:(mask,aa,bb) for m,mask in masks.items()}
            predictions.update({m:(mask,aa,par_new) for m,mask in dict(par_pair=medium&both2,par_directed=medium&directed2,par_prototype=medium&both2&agrees2,par_directed_prototype=medium&directed2&agrees2).items()})
            # Ground truth enters only the post-prediction metric calculation.
            target=F.interpolate(torch.as_tensor(gt,device='cuda')[None,None].float(),size=(448,448),mode='nearest')[:,0].long()
            valid=(target>=0)&(target<k)
            for m,(mask,aa,bb) in predictions.items():
                s=stats[m];s['selected']+=int(mask.sum());s['valid_pixels']+=int(valid.sum())
                s['gt_unknown']+=int((mask&~valid).sum())
                s['gt_old_candidate']+=int((mask&valid&(target==aa)).sum())
                s['gt_new_candidate']+=int((mask&valid&(target==bb)).sum())
                s['gt_other']+=int((mask&valid&(target!=aa)&(target!=bb)).sum())
            images.append(str(name))
            if (number+1)%25==0:print('audit',number+1,len(indices),stats,flush=True)
    for s in stats.values():
        den=s['selected']-s['gt_unknown'];s['candidate_coverage']=None if den==0 else (s['gt_old_candidate']+s['gt_new_candidate'])/den
        s['original_new_label_accuracy']=None if den==0 else s['gt_new_candidate']/den
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(dict(stage=a.stage,source_directory=str(d),images=images,selection='200 deterministic evenly spaced validation images; square448; offline only',pixel_gt_used_for_selection=False,statistics=stats),indent=2))
    print(json.dumps(stats),flush=True)
if __name__=='__main__':main()
