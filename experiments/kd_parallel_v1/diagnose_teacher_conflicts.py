"""Held-out diagnostic only: labels score pseudo-target errors, never form targets."""
from pathlib import Path
import os,sys,json
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto');E=R/'experiments/kd_parallel_v1';os.chdir(R)
sys.path.insert(0,str(R/'experiments/kd_pixel_v2'));from run_kd import environment
os.environ.update(environment('8card'));os.environ['CUDA_VISIBLE_DEVICES']='7'
sys.path.insert(0,str(E/'b_relational/src'))
import torch
import torch.nn.functional as F
import tasks
from datasets import voc
from model.model_seg_neg import network
from model.PAR import PAR
from utils.camutils import multi_scale_cam2,cam_to_label,refine_cams_with_bkg_v2,get_mixed_label
from utils.imutils import denormalize_img2
from kd_runtime import atomic_json,digest,now
torch.set_num_threads(1);torch.set_float32_matmul_precision('high');torch.backends.cudnn.allow_tf32=True
torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True

U=R/'runs/kd_parallel_v1/formal/b_relational';cfg=json.loads((U/'10-5/step2/config.json').read_text())
student_path=U/'10-5/step2/checkpoints/model_iter_2000.pth';teacher_path=U/'10-5/step1/checkpoints/model_final.pth'
def load(step,path):
 classes=tasks.get_per_task_classes('voc','10-5',step)
 m=network(backbone=cfg['backbone'],num_classes=sum(classes),classes_list=classes,pretrained=False,init_momentum=cfg['momentum'],aux_layer=cfg['aux_layer'])
 m.load_state_dict(torch.load(path,map_location='cpu',weights_only=True,mmap=True)['model_state'])
 return m.cuda().eval().requires_grad_(False)
student=load(2,student_path);teacher=load(1,teacher_path)
ds=voc.VOC12SegDataset(root_dir=cfg['data_folder'],name_list_dir=cfg['list_folder'],split='val',stage='val',aug=False,ignore_index=255,num_classes=21,tasks='10-5',step=2)
ds.label_dir=cfg['val_label_dir'];selected=[]
for class_id in range(16,21):
 candidates=[i for i,name in enumerate(ds.name_list) if ds.label_list[name][class_id-1] and i not in selected]
 selected.extend(candidates[:4])
assert len(selected)==20 and len(set(selected))==20
par=PAR(num_iter=10,dilations=[1,2,4,8,12,24]).cuda();records=[]
with torch.inference_mode():
 for index in selected:
  name,im,gt,cls=ds[index]
  x=F.interpolate(torch.as_tensor(im).float()[None].cuda(),size=(448,448),mode='bilinear',align_corners=False)
  image_labels=torch.as_tensor(cls).float()[None].cuda()
  box=torch.tensor([[0,448,0,448]])
  old_cls,_,logits,_,_=teacher(x,step0=True)
  merged=torch.cat(((old_cls>0).long(),image_labels[:,-5:]),1)
  cams,_=multi_scale_cam2(student,x,scales=cfg['cam_scales'])
  valid_cam,_=cam_to_label(cams.detach(),cls_label=merged,img_box=box,ignore_mid=True,bkg_thre=.5,high_thre=.7,low_thre=.25,ignore_index=255)
  refined=refine_cams_with_bkg_v2(par,denormalize_img2(x.clone()),cams=valid_cam,cls_labels=merged,high_thre=.7,low_thre=.25,ignore_index=255,img_box=box)
  old_y=F.interpolate(logits,size=(448,448),mode='bilinear',align_corners=False).argmax(1)
  mixed=get_mixed_label(refined,old_y.clone(),20,5)
  new_seed=(refined>=16)&(refined<21)
  new_cam=valid_cam[:,15:].amax(1)
  old_cam=valid_cam[:,:15].gather(1,(old_y-1).clamp_min(0)[:,None])[:,0]
  old_cam=torch.where(old_y>0,old_cam,torch.zeros_like(old_cam))
  conflict=(~new_seed)&(new_cam>=.25)&(new_cam>old_cam)
  # GT is introduced only below, for read-only held-out scoring.
  y=F.interpolate(torch.as_tensor(gt).float()[None,None].cuda(),size=(448,448),mode='nearest')[:,0].long()
  valid=(y>=0)&(y<21);new=(y>=16)&valid;old=(y>0)&(y<16)
  def count(mask):return int(mask.sum())
  record={'image':name,'valid_pixels':count(valid),'new_gt_pixels':count(new),'old_gt_pixels':count(old),
   'mixed_wrong_pixels':count(valid&(mixed!=y)),'new_gt_given_old_or_bg_target':count(new&(mixed<16)),
   'new_gt_recovered_by_PAR':count(new&new_seed&(mixed==y)),
   'conflict_pixels':count(valid&conflict),'conflict_wrong_target_pixels':count(valid&conflict&(mixed!=y)),
   'conflict_teacher_fg_pixels':count(valid&conflict&(old_y>0)),
   'conflict_teacher_fg_wrong_pixels':count(valid&conflict&(old_y>0)&(mixed!=y)),
   'conflict_teacher_fg_new_gt_pixels':count(conflict&(old_y>0)&new),
   'conflict_teacher_bg_pixels':count(valid&conflict&(old_y==0)),
   'conflict_teacher_bg_wrong_pixels':count(valid&conflict&(old_y==0)&(mixed!=y)),
   'conflict_new_gt_old_or_bg_target':count(conflict&new&(mixed<16)),
   'conflict_correct_old_target':count(conflict&old&(mixed==y)),
   'conflict_correct_bg_target':count(conflict&(y==0)&(mixed==y))}
  records.append(record)
total={key:sum(x[key] for x in records) for key in records[0] if key!='image'}
rates={'new_gt_missing_from_new_targets_percent':100*total['new_gt_given_old_or_bg_target']/max(1,total['new_gt_pixels']),
 'conflict_wrong_target_percent':100*total['conflict_wrong_target_pixels']/max(1,total['conflict_pixels']),
 'fraction_of_missed_new_pixels_in_conflict_percent':100*total['conflict_new_gt_old_or_bg_target']/max(1,total['new_gt_given_old_or_bg_target']),
 'foreground_conflict_wrong_percent':100*total['conflict_teacher_fg_wrong_pixels']/max(1,total['conflict_teacher_fg_pixels']),
 'background_conflict_wrong_percent':100*total['conflict_teacher_bg_wrong_pixels']/max(1,total['conflict_teacher_bg_pixels'])}
out=R/'runs/kd_parallel_v1/heldout_teacher_conflict_breakdown.json'
atomic_json(out,{'utc':now(),'sampling':'first4 unique validation images containing each current new class16..20, by fixed list order','sample_indices':selected,
 'student_checkpoint':str(student_path),'student_sha256':digest(student_path),'teacher_checkpoint':str(teacher_path),'teacher_sha256':digest(teacher_path),
 'scope':'20 held-out images at448; inference only; no optimizer, no training modification; GT only scores masks, never builds them',
 'conflict_rule':'Outside PAR new seeds: max_new_CAM>=existing low threshold0.25 and max_new_CAM>teacher-predicted old-class CAM; BG has old support0',
 'totals':total,'rates_percent':rates,'per_image':records,
 'limitations':['small deliberately new-class-enriched sample','resized448 diagnostic, not benchmark mIoU','diagnostic rule has not been trained or validated as an improvement']})
print(json.dumps({'output':str(out),'rates_percent':rates,'totals':total}))
