"""Read-only gradient probe on real incremental training images and saved weights."""
from pathlib import Path
import os,sys,json,random
ROOT=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto')
EXP=ROOT/'experiments/kd_pixel_v2';SRC=EXP/'src'
sys.path.insert(0,str(EXP));from run_kd import environment
os.environ.update(environment('8card-diagnostic'))
os.environ['CUDA_VISIBLE_DEVICES']='7'
sys.path.insert(0,str(SRC))
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader,Subset
import tasks
from datasets import voc
from model.model_seg_neg import network
from model.pixel_kd import pixel_kd_loss
from model.losses import get_seg_loss
from model.PAR import PAR
from utils.camutils import multi_scale_cam2,cam_to_label,refine_cams_with_bkg_v2,get_mixed_label
from utils.imutils import denormalize_img2
from kd_runtime import atomic_json,digest,now

def main():
    torch.manual_seed(0);np.random.seed(0);random.seed(0)
    cfg=json.loads((ROOT/'runs/kd_pixel_v1/10-5/step1/config.json').read_text())
    checkpoint=ROOT/'runs/kd_pixel_v1/10-5/step1/checkpoints/model_iter_2000.pth'
    teacher_checkpoint=Path(cfg['prev_checkpoint'])
    def load(step,weights):
        classes=tasks.get_per_task_classes('voc','10-5',step)
        m=network(backbone=cfg['backbone'],num_classes=sum(classes),classes_list=classes,
                  pretrained=False,init_momentum=cfg['momentum'],aux_layer=cfg['aux_layer'])
        d=torch.load(weights,map_location='cpu',weights_only=True,mmap=True)
        m.load_state_dict(d['model_state'],strict=True);del d
        return m.cuda().eval()
    student=load(1,checkpoint);teacher=load(0,teacher_checkpoint)
    teacher.requires_grad_(False)
    ds=voc.VOC12ClsDataset(root_dir=cfg['data_folder'],name_list_dir=cfg['list_folder'],split='train',
            stage='train',aug=True,rescale_range=cfg['scales'],crop_size=448,img_fliplr=True,
            ignore_index=255,num_classes=21,tasks='10-5',step=1)
    indices=[0,len(ds)//3,2*len(ds)//3]
    loader=DataLoader(Subset(ds,indices),batch_size=1,num_workers=0)
    par=PAR(num_iter=10,dilations=[1,2,4,8,12,24]).cuda()
    selected={name:param for name,param in student.named_parameters()
              if name.startswith('decoder.conv8.') or name=='encoder.patch_embed.proj.weight' or name.startswith('decoder.class_prototypes.')}
    assert any(name.startswith('encoder.') for name in selected)
    records=[]
    for names,inputs,image_labels,img_box,crops in loader:
        inputs=inputs.cuda();image_labels=image_labels.cuda()[:,:15]
        with torch.no_grad():
            old_cls,_,old_segs,_,_=teacher(inputs,step0=True)
            merged_labels=torch.cat(((old_cls>0).long(),image_labels[:,-5:]),1)
            cams,_=multi_scale_cam2(student,inputs,scales=cfg['cam_scales'])
            valid_cam,_=cam_to_label(cams.detach(),cls_label=merged_labels,img_box=img_box,ignore_mid=True,
                  bkg_thre=.5,high_thre=.7,low_thre=.25,ignore_index=255)
            refined=refine_cams_with_bkg_v2(par,denormalize_img2(inputs.clone()),cams=valid_cam,cls_labels=merged_labels,
                   high_thre=.7,low_thre=.25,ignore_index=255,img_box=img_box)
            teacher_label=F.interpolate(old_segs,size=inputs.shape[-2:],mode='bilinear',align_corners=False).argmax(1)
            mixed=get_mixed_label(refined,teacher_label,15,5)
        _,segs,_,_,_,_=student(inputs,crops=[])
        kd,stats=pixel_kd_loss(segs,old_segs,refined,valid_cam,img_box,2.)
        segmentation=get_seg_loss(F.interpolate(segs,size=mixed.shape[-2:],mode='bilinear',align_corners=False),
                                  mixed.long(),torch.ones(16,device='cuda'),ignore_index=255)
        kg=torch.autograd.grad(kd,tuple(selected.values()),retain_graph=True,allow_unused=True)
        sg=torch.autograd.grad(segmentation,tuple(selected.values()),allow_unused=True)
        gradients={}
        for name,kgrad,sgrad in zip(selected,kg,sg):
            norm=0 if kgrad is None else float(kgrad.norm())
            record={'kd_gradient_norm':norm,'seg_gradient_norm':0 if sgrad is None else float(sgrad.norm())}
            if kgrad is not None and sgrad is not None and kgrad.norm()>0 and sgrad.norm()>0:
                record['kd_seg_cosine']=float(F.cosine_similarity(kgrad.flatten(),sgrad.flatten(),dim=0))
            gradients[name]=record
        assert all(p.grad is None and not p.requires_grad for p in teacher.parameters())
        assert gradients['decoder.conv8.1.weight']['kd_gradient_norm']==0
        
        assert all(v['kd_gradient_norm']==0 for k,v in gradients.items() if k.startswith('decoder.class_prototypes.'))
        assert all(torch.isfinite(g).all() for g in kg if g is not None)
        records.append({'image':names[0],'stats':stats,'gradients':gradients,'teacher_frozen_and_no_grad':True})
    output=ROOT/'runs/kd_pixel_v2/gradient_probe_warmup2000.json'
    atomic_json(output,{'created_utc':now(),'checkpoint':str(checkpoint),'checkpoint_sha256':digest(checkpoint),
       'teacher_checkpoint_sha256':digest(teacher_checkpoint),'pixel_kd_sha256':digest(SRC/'model/pixel_kd.py'),
       'scope':'Three actual augmented training images; eval-mode gradient probe; no optimizer update; no training pixel GT read',
       'sample_indices':indices,'results':records,'passed':True})
    print(json.dumps({'passed':True,'images':len(records),'output':str(output)}))

if __name__=='__main__':main()
