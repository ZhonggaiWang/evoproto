"""Actual parameter gradients on saved weights/training images; no optimizer/GT."""
from pathlib import Path
import sys,os,json,random
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto');E=R/'experiments/prototype_sep_v1';S=E/'a_geometry/src'
sys.path.insert(0,str(R/'experiments/kd_pixel_v2'))
from run_kd import environment
os.environ.update(environment('8card'));os.environ['CUDA_VISIBLE_DEVICES']='7'
sys.path.insert(0,str(S))
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader,Subset
import tasks
from datasets import voc
from model.model_seg_neg import network
from model.pixel_kd import pixel_kd_loss
from model.confusion_prototype_sep import confusion_prototype_sep_loss
from model.geometry_pair_selector import GeometryPairSelector
from model.online_directed_confusion import OnlineDirectedConfusion
from model.losses import get_seg_loss
from model.PAR import PAR
from utils.camutils import multi_scale_cam2,cam_to_label,refine_cams_with_bkg_v2,get_mixed_label
from utils.imutils import denormalize_img2
from kd_runtime import atomic_json,digest,now

def main():
    torch.manual_seed(0);np.random.seed(0);random.seed(0)
    torch.set_float32_matmul_precision('high');torch.backends.cudnn.allow_tf32=True
    base=R/'runs/kd_parallel_v1/formal/b_relational/10-5'
    cfg=json.loads((base/'step2/config.json').read_text())
    checkpoint=base/'step2/checkpoints/model_final.pth'
    teacher_checkpoint=base/'step1/checkpoints/model_final.pth'
    replay=R/'runs/confusion_guided_v1/formal/a_sep/10-5/step2/checkpoints/model_final.pth'
    obs=OnlineDirectedConfusion(21,stage=2).cuda()
    saved=torch.load(replay,map_location='cpu',weights_only=True,mmap=True)
    obs.load_state_dict(saved['online_confusion_state'],strict=True);del saved
    selector=GeometryPairSelector(21,old_classes=15,stage=2).cuda()
    targets=selector.update(obs,8001)
    def load(step,weights):
        classes=tasks.get_per_task_classes('voc','10-5',step)
        model=network(backbone=cfg['backbone'],num_classes=sum(classes),classes_list=classes,
            pretrained=False,init_momentum=cfg['momentum'],aux_layer=cfg['aux_layer'])
        state=torch.load(weights,map_location='cpu',weights_only=True,mmap=True)
        model.load_state_dict(state['model_state'],strict=True);del state
        return model.cuda().eval()
    student=load(2,checkpoint);teacher=load(1,teacher_checkpoint);teacher.requires_grad_(False)
    params={name:param for name,param in student.named_parameters() if param.requires_grad}
    ds=voc.VOC12ClsDataset(root_dir=cfg['data_folder'],name_list_dir=cfg['list_folder'],split='train',
        stage='train',aug=True,rescale_range=cfg['scales'],crop_size=448,img_fliplr=True,
        ignore_index=255,num_classes=21,tasks='10-5',step=2)
    indices=[0,len(ds)//4,len(ds)//2,3*len(ds)//4,len(ds)-1]
    loader=DataLoader(Subset(ds,indices),batch_size=1,num_workers=0)
    par=PAR(num_iter=10,dilations=[1,2,4,8,12,24]).cuda()
    records=[];any_sep=False;any_kd=False
    for names,inputs,image_labels,img_box,crops in loader:
        inputs=inputs.cuda();image_labels=image_labels.cuda()[:,:20]
        with torch.no_grad():
            old_cls,_,old_segs,_,_=teacher(inputs,step0=True)
            labels=torch.cat(((old_cls>0).long(),image_labels[:,-5:]),1)
            cams,_=multi_scale_cam2(student,inputs,scales=cfg['cam_scales'])
            valid_cam,_=cam_to_label(cams.detach(),cls_label=labels,img_box=img_box,ignore_mid=True,
                bkg_thre=cfg['bkg_thre'],high_thre=cfg['high_thre'],low_thre=cfg['low_thre'],ignore_index=255)
            refined=refine_cams_with_bkg_v2(par,denormalize_img2(inputs.clone()),cams=valid_cam,cls_labels=labels,
                high_thre=cfg['high_thre'],low_thre=cfg['low_thre'],ignore_index=255,img_box=img_box)
            old_label=F.interpolate(old_segs,size=inputs.shape[-2:],mode='bilinear',align_corners=False).argmax(1)
            mixed=get_mixed_label(refined,old_label,20,5)
        _,segs,_,_,proto_logits,prototypes=student(inputs,crops=[])
        sep,stats=confusion_prototype_sep_loss(prototypes,targets,15,margin=0)
        kd,kd_stats=pixel_kd_loss(segs,old_segs,refined,valid_cam,img_box,2.)
        proto_seg=get_seg_loss(F.interpolate(proto_logits/.1,size=mixed.shape[-2:],mode='bilinear',align_corners=False),
            mixed.long(),torch.ones(21,device='cuda'),ignore_index=255)
        losses={'sep':.1*sep,'kd':.1*kd,'proto_seg':.1*proto_seg}
        grads={}
        for name,loss in losses.items():
            gs=torch.autograd.grad(loss,tuple(params.values()),retain_graph=True,allow_unused=True)
            grads[name]={key:g for key,g in zip(params,gs) if g is not None}
        metrics={}
        for name,gs in grads.items():
            metrics[name]={key:float(g.norm()) for key,g in gs.items() if g.norm()>0}
            assert all(torch.isfinite(g).all() for g in gs.values())
        old_nonzero=[key for key,g in grads['sep'].items()
            if (not key.startswith('decoder.class_prototypes.2.')) and g.count_nonzero()>0]
        assert not old_nonzero,old_nonzero
        assert any(g.count_nonzero()>0 for g in grads['sep'].values()),'Selected SEP must have real gradient'
        assert all(g.count_nonzero()==0 for key,g in grads['kd'].items() if key.startswith('decoder.class_prototypes.'))
        any_sep=True;any_kd|=any(g.count_nonzero()>0 for g in grads['kd'].values())
        relationships={}
        for left,right in [('sep','kd'),('sep','proto_seg')]:
            keys=set(grads[left])&set(grads[right])
            keys=[key for key in keys if grads[left][key].count_nonzero()>0 and grads[right][key].count_nonzero()>0]
            if keys:
                dot=sum((grads[left][k]*grads[right][k]).sum() for k in keys)
                ln=sum(grads[left][k].square().sum() for k in keys).sqrt()
                rn=sum(grads[right][k].square().sum() for k in keys).sqrt()
                relationships[left+'_'+right]={'shared_nonzero_parameter_names':keys,'cosine':float(dot/(ln*rn)),
                    'left_norm_on_shared':float(ln),'right_norm_on_shared':float(rn)}
            else:relationships[left+'_'+right]={'shared_nonzero_parameter_names':[], 'cosine':None,
                'meaning':'No directly shared nonzero parameter gradients at this iteration; later training dynamics may interact'}
        assert all(p.grad is None and not p.requires_grad for p in teacher.parameters())
        assert all(p.grad is None for p in student.parameters())
        records.append({'image':names[0],'weighted_loss':{k:float(v) for k,v in losses.items()},
            'sep':stats,'kd':kd_stats,'nonzero_parameter_gradient_norms':metrics,'relationships':relationships})
        del grads,losses,segs,proto_logits,prototypes
    assert any_sep and any_kd
    output=R/'runs/prototype_sep_v1/gradient_probe.json'
    atomic_json(output,{'utc':now(),'passed':True,'scope':'Five actual augmented stage2 training images, saved optimizedKD weights, eval mode; no optimizer step and no pixel GT interface',
        'caveat':'Selector replay uses A final online observer on optimizedKD final model; diagnostic transfer, not a contemporaneous training trajectory',
        'student_checkpoint':str(checkpoint),'student_sha256':digest(checkpoint),
        'teacher_checkpoint':str(teacher_checkpoint),'teacher_sha256':digest(teacher_checkpoint),
        'observer_checkpoint':str(replay),'observer_checkpoint_sha256':digest(replay),
        'selector':selector.export(),'sample_indices':indices,'records':records})
    print(json.dumps({'passed':True,'output':str(output),'images':len(records)}))

if __name__=='__main__':main()
