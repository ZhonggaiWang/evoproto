"""Build teacher-confusion evidence on training images; diagnose held-out pixels.
Train calibration uses VOC12ClsDataset and never opens a pixel mask.
Validation GT is only accessed for metrics after all candidate masks are computed.
"""
import os,sys,argparse,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
import torch.nn.functional as F
import numpy as np
from datasets.voc import VOC12ClsDataset,VOC12SegDataset
from model.model_seg_neg import network
from model.PAR import PAR
from utils import imutils
from utils.camutils import multi_scale_cam2,cam_to_label,refine_cams_with_bkg_v2
from experiments.restore_proto_seed_v1.mechanism import SeedCorrection,nearest

def load(path,stage):
    net=network('vit_base_patch16_224',num_classes=11+5*stage,classes_list=[11]+[5]*stage,pretrained=False,init_momentum=.9,aux_layer=-3)
    s=torch.load(path,map_location='cpu',weights_only=True)['model_state']
    net.load_state_dict({k.removeprefix('module.'):v for k,v in s.items()},strict=True)
    return net.cuda().eval()

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',type=int,required=True);p.add_argument('--directory',required=True);p.add_argument('--output',required=True);p.add_argument('--limit',type=int,default=200);a=p.parse_args()
    torch.set_num_threads(4);torch.set_float32_matmul_precision('high');torch.backends.cudnn.allow_tf32=True
    d=Path(a.directory);out=Path(a.output);assert out.resolve().is_relative_to(ROOT) and not out.exists()
    teacher=load(json.loads((d/'predecessor.json').read_text())['path'],a.stage-1)
    net=load(d/'checkpoints/model_final.pth',a.stage);k=11+5*a.stage;oc=k-5
    module=SeedCorrection(k,oc,'cuda');par=PAR(num_iter=10,dilations=[1,2,4,8,12,24]).cuda()
    common=dict(root_dir='/data/zhonggai/coco/PascalVOC12',name_list_dir=str(ROOT/'datasets/voc'),aug=False,ignore_index=255,num_classes=21,tasks='10-5',step=a.stage)
    train=VOC12ClsDataset(split='train',stage='train',**common);val=VOC12SegDataset(split='val',stage='val',**common)
    val.label_dir='/data/zhonggai/coco/PascalVOC12/SegmentationClass'
    assert not set(train.name_list)&set(val.name_list)
    @torch.no_grad()
    def infer(im,tags):
        x=F.interpolate(torch.as_tensor(im,device='cuda')[None],size=(448,448),mode='bilinear',align_corners=False)
        cls,_,old,features,_=teacher(x,step0=True)
        old=F.interpolate(old,size=(448,448),mode='bilinear',align_corners=False)
        tags=torch.cat([(cls>0).float(),torch.as_tensor(tags,device='cuda')[None,oc-1:k-1]],1)
        cams,aux=multi_scale_cam2(net,x,scales=[1.,.5,1.5]);box=torch.tensor([[0,448,0,448]])
        ca,_=cam_to_label(cams,cls_label=tags,img_box=box,ignore_mid=True,bkg_thre=.5,high_thre=.7,low_thre=.25,ignore_index=255)
        labels=refine_cams_with_bkg_v2(par,imutils.denormalize_img2(x.clone()),cams=ca,cls_labels=tags,high_thre=.7,low_thre=.25,ignore_index=255,img_box=box).long()
        valid=torch.ones_like(labels,dtype=torch.bool);mixed=old.argmax(1);new=(labels>=oc)&(labels<k);mixed[new]=labels[new]
        evidence=module.evidence(features,cams,aux,tags,old,labels,valid)
        return evidence,mixed,valid
    calibration=[]
    online=d/'teacher_confusion.json'
    if online.exists():
        saved=json.loads(online.read_text())
        module.count=torch.tensor(saved['count'],device='cuda');module.mass=torch.tensor(saved['mass'],device='cuda')
        module.seen=[[set(range(n)) for n in row] for row in saved['unique_image_support']]
        module.updates=saved['updates']
        graph_source='saved online training graph'
    else:
        graph_source='200 fixed calibration training images, no pixel masks'
        for number,idx in enumerate(np.linspace(0,len(train)-1,min(a.limit,len(train)),dtype=int)):
            name,im,tags=train[int(idx)];e,_,_=infer(im,tags);module.update(e[4],e[3],[str(name)]);calibration.append(str(name))
            if (number+1)%50==0:print('calibration',number+1,flush=True)
    calibration_state=module.state()
    stats={};records=[]
    for mode in ['correct','ignore','no_graph']:
        stats[mode]=dict(valid=0,new_pixels=0,old_pixels=0,old_on_new_before=0,old_on_new_after=0,new_on_old_before=0,new_on_old_after=0,changed=0,changed_valid=0,changed_to_true_new=0,changed_from_true_old=0,changed_other=0,correct_before=0,correct_after=0,seeds=0,correct_seeds=0)
    for number,idx in enumerate(np.linspace(0,len(val)-1,min(a.limit,len(val)),dtype=int)):
        name,im,gt,tags=val[int(idx)];e,before,valid=infer(im,tags)
        proposals={}
        for mode in stats:
            module.mode=mode;proposals[mode]=module.correct(e,before,valid)
        target=nearest(torch.as_tensor(gt,device='cuda')[None],(448,448)).long();known=(target>=0)&(target<k)
        true_new=known&(target>=oc);true_old=known&(target>0)&(target<oc)
        seed=e[4];ngt=nearest(target,seed.shape[-2:]).long();valid_seed=(seed>=oc)&(seed<k)&(ngt<k)
        row=dict(name=str(name),modes={})
        for mode,(after,changed,proposal,diag) in proposals.items():
            m=changed&known
            counts=dict(valid=int(known.sum()),new_pixels=int(true_new.sum()),old_pixels=int(true_old.sum()),
                        old_on_new_before=int((true_new&(before>0)&(before<oc)).sum()),old_on_new_after=int((true_new&(after>0)&(after<oc)).sum()),
                        new_on_old_before=int((true_old&(before>=oc)&(before<k)).sum()),new_on_old_after=int((true_old&(after>=oc)&(after<k)).sum()),
                        changed=int(changed.sum()),changed_valid=int(m.sum()),changed_to_true_new=int((m&(target==proposal)).sum()),
                        changed_from_true_old=int((m&true_old).sum()),changed_other=int((m&~true_old&(target!=proposal)).sum()),
                        correct_before=int((known&(before==target)).sum()),correct_after=int((known&(after==target)).sum()),
                        seeds=int(valid_seed.sum()),correct_seeds=int((valid_seed&(seed==ngt)).sum()))
            for key,value in counts.items():stats[mode][key]+=value
            row['modes'][mode]=counts
        records.append(row)
        if (number+1)%50==0:print('validation',number+1,stats,flush=True)
    for s in stats.values():
        s['proposal_precision']=None if not s['changed_valid'] else s['changed_to_true_new']/s['changed_valid']
        s['old_to_new_error_delta']=s['new_on_old_after']-s['new_on_old_before']
        s['novel_absorption_reduced']=s['old_on_new_before']-s['old_on_new_after']
        s['seed_precision']=s['correct_seeds']/max(s['seeds'],1)
    result=dict(stage=a.stage,source=str(d),calibration_train_names=calibration,calibration_state=calibration_state,graph_source=graph_source,statistics=stats,per_image=records,pixel_gt_used_in_selection=False,validation_adaptation=False)
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2));print(json.dumps(stats),flush=True)
if __name__=='__main__':main()
