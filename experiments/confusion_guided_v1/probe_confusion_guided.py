"""Real training-image gradient path checks; no optimizer and no pixel GT."""
from pathlib import Path
import os,sys,json
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto');E=R/'experiments/confusion_guided_v1'
os.chdir(R);sys.path.insert(0,str(R/'experiments/kd_pixel_v2'))
from run_kd import environment
os.environ.update(environment('8card'));os.environ['CUDA_VISIBLE_DEVICES']='0'
sys.path.insert(0,str(E/'a_sep/src'))
import numpy as np
import torch
import torch.nn.functional as F
import tasks
from model.model_seg_neg import network
from model.PAR import PAR
from model.online_directed_confusion import OnlineDirectedConfusion
from model.confusion_pair_losses import directed_pair_sep_loss
from model.pixel_kd import pixel_kd_loss
from datasets import voc
from utils.camutils import multi_scale_cam2,cam_to_label,refine_cams_with_bkg_v2
from utils.imutils import denormalize_img2
from kd_runtime import atomic_json,now,digest
torch.set_num_threads(1);torch.manual_seed(0);np.random.seed(0)
torch.set_float32_matmul_precision('high');torch.backends.cudnn.allow_tf32=True
cfg=json.loads((R/'runs/kd_parallel_v1/formal/b_relational/10-5/step1/config.json').read_text())
student_path=R/'runs/kd_pixel_v2/10-5/step1/checkpoints/model_iter_2000.pth'
teacher_path=R/'runs/fixed_baseline_v1/shared/10-5/step0/checkpoints/model_final.pth'
def load(step,path):
    classes=tasks.get_per_task_classes('voc','10-5',step)
    m=network(backbone=cfg['backbone'],num_classes=sum(classes),classes_list=classes,
        pretrained=False,init_momentum=cfg['momentum'],aux_layer=cfg['aux_layer'])
    m.load_state_dict(torch.load(path,map_location='cpu',weights_only=True,mmap=True)['model_state'])
    return m.cuda().eval()
student=load(1,student_path);teacher=load(0,teacher_path).requires_grad_(False)
ds=voc.VOC12ClsDataset(root_dir=cfg['data_folder'],name_list_dir=cfg['list_folder'],split='train',stage='train',aug=True,
    rescale_range=cfg['scales'],crop_size=448,img_fliplr=True,ignore_index=255,num_classes=21,tasks='10-5',step=1)
selected=[]
for c in range(11,16):
    i=next((i for i,name in enumerate(ds.name_list) if ds.label_list[name][c-1] and i not in selected),None)
    if i is not None:selected.append(i)
observer=OnlineDirectedConfusion(16,stage=1).cuda();par=PAR(num_iter=10,dilations=[1,2,4,8,12,24]).cuda();records=[]
for index in selected:
    name,image,labels,box,crops=ds[index]
    x=image[None].cuda();labels=torch.as_tensor(labels)[None,:15].cuda();box=[box]
    with torch.no_grad():
        old_cls,_,old_logits,_,_=teacher(x,step0=True)
        merged=torch.cat(((old_cls>0).long(),labels[:,-5:]),1)
        cams,_=multi_scale_cam2(student,x,scales=cfg['cam_scales'])
        valid_cam,_=cam_to_label(cams,cls_label=merged,img_box=box,ignore_mid=True,
            bkg_thre=.5,high_thre=.7,low_thre=.25,ignore_index=255)
        refined=refine_cams_with_bkg_v2(par,denormalize_img2(x.clone()),cams=valid_cam,cls_labels=merged,
            high_thre=.7,low_thre=.25,ignore_index=255,img_box=box)
    _,_,logits,_,_,type_logits,_=student(x,cam_grad=True)
    evidence=observer.update(logits,refined,valid_cam,box,synchronize=False)
    # Controlled competitors isolate gradient paths. They are NOT the formal
    # online selector and do not use GT or perform optimizer updates.
    targets=torch.full((16,),-1,dtype=torch.long,device='cuda')
    k=old_logits.shape[1]
    for c in range(1,16):targets[c]=1 if c!=1 else 2
    sep,sep_stats=directed_pair_sep_loss(logits,type_logits/.1,old_logits,evidence,targets)
    kd,kd_stats=pixel_kd_loss(logits,old_logits,refined,valid_cam,box,2,
        pair_targets=targets,pair_evidence=evidence,pair_blend=.5)
    params=[student.decoder.conv8[0].weight,student.decoder.conv8[-1].weight,
            student.decoder.class_prototypes[0].prototype,student.decoder.class_prototypes[-1].prototype,
            student.encoder.patch_embed.proj.weight]
    def norms(loss,retain):
        gs=torch.autograd.grad(loss,params,allow_unused=True,retain_graph=retain)
        return {n:float(g.norm()) if g is not None else 0 for n,g in zip(['old_main_head','new_main_head','old_prototypes','new_prototypes','encoder'],gs)}
    records.append({'image':name,'sep':sep_stats,'kd':kd_stats,'SEP_gradient_norms':norms(sep,True),'KD_gradient_norms':norms(kd,False)})
    print(name,records[-1]['SEP_gradient_norms'],records[-1]['KD_gradient_norms'],flush=True)
    del logits,type_logits,sep,kd
assert any(r['SEP_gradient_norms']['old_main_head']>0 for r in records)
assert any(r['SEP_gradient_norms']['new_main_head']>0 for r in records)
assert any(r['SEP_gradient_norms']['old_prototypes']>0 for r in records)
assert any(r['SEP_gradient_norms']['new_prototypes']>0 for r in records)
assert any(r['SEP_gradient_norms']['encoder']>0 for r in records)
assert all(r['KD_gradient_norms']['new_main_head']==0 for r in records)
assert all(r['KD_gradient_norms']['new_prototypes']==0 for r in records)
assert any(r['kd'].get('kd_pair_pixels',0)>0 for r in records)
atomic_json(E/'gradient_probe.json',{'utc':now(),'passed':True,'scope':'5 training images chosen by current-class image labels; controlled competitor IDs to isolate gradients; no optimizer/pixelGT, not efficacy evaluation',
    'student':str(student_path),'student_sha256':digest(student_path),'teacher':str(teacher_path),'teacher_sha256':digest(teacher_path),'records':records})
print('Actual gradient paths verified')
