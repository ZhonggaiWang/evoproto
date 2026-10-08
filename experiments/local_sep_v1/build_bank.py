"""One-pass weak-only training feature bank, never opens pixel annotations."""
from pathlib import Path
import os,sys,json,argparse
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto');E=R/'experiments/local_sep_v1';U=R/'runs/local_sep_v1';S=E/'src'
sys.path.insert(0,str(R/'experiments/kd_pixel_v2'))
from run_kd import environment
os.environ.update(environment('8card'));sys.path.insert(0,str(S))
import torch
import torch.nn.functional as F
from model.model_seg_neg import network
from model.ald import image_targets,fuse
from model.PAR import PAR
from datasets import voc
from utils.camutils import multi_scale_cam2,cam_to_label,refine_cams_with_bkg_v2
from utils.imutils import denormalize_img2
from kd_runtime import safe_path,atomic_json,digest,now

def main():
 p=argparse.ArgumentParser();p.add_argument('--rank',type=int,required=True);rank=p.parse_args().rank
 torch.cuda.set_device(rank);torch.set_num_threads(1);torch.set_float32_matmul_precision('high');torch.backends.cudnn.allow_tf32=True
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 study=json.loads((R/'runs/ald_calibration_v9/formal/study.json').read_text());cfg=study['config']
 best=R/'runs/ald_calibration_v9/formal/10-5/step2/checkpoints/model_final.pth'
 def load(path,classes):
  ck=torch.load(safe_path(path),map_location='cpu',weights_only=True,mmap=True)
  m=network(backbone=cfg['backbone'],num_classes=sum(classes),classes_list=classes,pretrained=False,init_momentum=cfg['momentum'],aux_layer=cfg['aux_layer'])
  m.load_state_dict(ck['model_state'],strict=True);return m.cuda().eval().requires_grad_(False)
 teacher=load(cfg['prev_checkpoint'],[11,5]);reference=load(cfg['ald_reference'],[11,5,5]);model=load(best,[11,5,5])
 par_model=PAR(num_iter=10,dilations=[1,2,4,8,12,24]).cuda()
 ev=torch.load(safe_path(cfg['ald_evidence']),map_location='cpu',weights_only=True);states={n:ev['state'][i] for i,n in enumerate(ev['names'])}
 ds=voc.VOC12ClsDataset(root_dir=cfg['data_folder'],name_list_dir=cfg['list_folder'],split='train',stage='train',aug=False,num_classes=21,tasks='10-5',step=2)
 assert set(map(str,ds.name_list))==set(states) and len(ds)==2145
 val=(Path(cfg['list_folder'])/'incremental_split/val_10-5_step_3.txt').read_text().splitlines();assert not set(states)&set(val)
 capture={}
 def hook(m,args):capture['feature']=args[0].detach()
 h=model.decoder.conv8.register_forward_pre_hook(hook)
 vectors=[];classes=[];owners=[];names=[];pixels=[]
 with torch.inference_mode():
  for index in range(rank,len(ds),8):
   name,img,tags=ds[index];name=str(name);names.append(name)
   x=F.interpolate(img.cuda()[None],(448,448),mode='bilinear',align_corners=False);box=[[0,448,0,448]]
   new=torch.as_tensor(tags,device='cuda')[None,15:].float();state=states[name][None].cuda()
   oc,oa,tl,_,_=teacher(x,step0=True);_,rl,_,_=reference(x)
   _,allowed,_=image_targets(state,oc,oa,tl,rl,new)
   cams,_=multi_scale_cam2(model,x,scales=cfg['cam_scales'])
   valid,_=cam_to_label(cams,cls_label=allowed,img_box=box,ignore_mid=True,bkg_thre=.5,high_thre=.7,low_thre=.25,ignore_index=255)
   par=refine_cams_with_bkg_v2(par_model,denormalize_img2(x.clone()),cams=valid,cls_labels=allowed,high_thre=.7,low_thre=.25,ignore_index=255,img_box=box)
   fused=fuse(par,tl,rl,state,box)
   _,logits,_,_=model(x);f=capture.pop('feature');size=f.shape[-2:]
   y=F.interpolate(fused['labels'][:,None].float(),size,mode='nearest')[:,0].long()
   cam=F.interpolate(valid,size,mode='bilinear',align_corners=False);top=cam.topk(2,1)
   trusted=(y>0)&(y<21)&(top.indices[:,0]+1==y)&(top.values[:,0]>=.7)&(logits.argmax(1)==y)
   # Erode one native cell: bank represents class interiors, not mixed boundaries.
   fn=F.normalize(f,dim=1)
   for c in range(1,21):
    mask=(trusted&(y==c));interior=F.avg_pool2d(mask[:,None].float(),3,1,1)[:,0]>.999
    if int(interior.sum())<3:continue
    vector=F.normalize(fn.permute(0,2,3,1)[interior].mean(0),dim=0)
    vectors.append(vector.cpu());classes.append(c);owners.append(name);pixels.append(int(interior.sum()))
   if len(names)%64==0:print(json.dumps({'rank':rank,'images':len(names),'vectors':len(vectors)}),flush=True)
 h.remove();out=safe_path(U/f'bank_rank{rank}.pth');assert not out.exists()
 torch.save({'vectors':torch.stack(vectors),'classes':torch.tensor(classes),'owners':owners,'pixels':torch.tensor(pixels),'names':names},out)
 atomic_json(U/f'bank_receipt{rank}.json',{'images':names,'bank_sha256':digest(out),'best_sha256':digest(best),'source_sha256':digest(E/'build_bank.py'),'evidence_sha256':digest(cfg['ald_evidence']),'GT_role':'No pixel GT or old image tags; training images only','utc':now()})
if __name__=='__main__':main()
