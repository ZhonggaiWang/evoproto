"""One-pass weak-only training feature bank, never opens pixel annotations."""
from pathlib import Path
import os,sys,json,argparse
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto');E=R/'experiments/local_sep_v5';U=R/'runs/local_sep_v5';S=E/'src'
sys.path.insert(0,str(R/'experiments/kd_pixel_v2'))
from run_kd import environment
os.environ.update(environment('8card'));sys.path.insert(0,str(S))
import torch
import torch.nn.functional as F
from model.model_seg_neg import network
from model.ald import image_targets,fuse
from model.boundary_sep import targets
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
  mm=m.load_state_dict(ck['model_state'],strict=False);assert not mm.unexpected_keys and all(k.startswith('decoder.local_sep.') for k in mm.missing_keys)
  return m.cuda().eval().requires_grad_(False)
 teacher=load(cfg['prev_checkpoint'],[11,5]);reference=load(cfg['ald_reference'],[11,5,5]);model=load(best,[11,5,5])
 par_model=PAR(num_iter=10,dilations=[1,2,4,8,12,24]).cuda()
 ev=torch.load(safe_path(cfg['ald_evidence']),map_location='cpu',weights_only=True);states={n:ev['state'][i] for i,n in enumerate(ev['names'])}
 ds=voc.VOC12ClsDataset(root_dir=cfg['data_folder'],name_list_dir=cfg['list_folder'],split='train',stage='train',aug=False,num_classes=21,tasks='10-5',step=2)
 assert set(map(str,ds.name_list))==set(states) and len(ds)==2145
 val=(Path(cfg['list_folder'])/'incremental_split/val_10-5_step_3.txt').read_text().splitlines();assert not set(states)&set(val)
 counts=torch.zeros(3,21,21,device='cuda',dtype=torch.long);names=[]
 with torch.inference_mode():
  for index in range(rank,len(ds),8):
   name,img,tags=ds[index];name=str(name);names.append(name)
   x=F.interpolate(img.cuda()[None],(448,448),mode='bilinear',align_corners=False);box=[[0,448,0,448]]
   new=torch.as_tensor(tags,device='cuda')[None,15:].float();state=states[name][None].cuda()
   oc,oa,tl,_,_=teacher(x,step0=True);_,rl,_,_=reference(x)
   _,allowed,_=image_targets(state,oc,oa,tl,rl,new)
   cams,auxcams=multi_scale_cam2(model,x,scales=cfg['cam_scales'])
   valid,_=cam_to_label(cams,cls_label=allowed,img_box=box,ignore_mid=True,bkg_thre=.5,high_thre=.7,low_thre=.25,ignore_index=255)
   par=refine_cams_with_bkg_v2(par_model,denormalize_img2(x.clone()),cams=valid,cls_labels=allowed,high_thre=.7,low_thre=.25,ignore_index=255,img_box=box)
   fused=fuse(par,tl,rl,state,box)
   _,logits,_,_=model(x)
   target=targets(fused['labels'],valid,auxcams*allowed[:,:,None,None],fused['valid'],logits,state=state,new_tags=new)
   ids=target['source']*21+target['rival']
   for k,mask in enumerate([target['eligible'],target['positive'],target['negative']]):
    counts[k].flatten()[ids[mask].unique()]+=1
   if len(names)%64==0:print(json.dumps({'rank':rank,'images':len(names)}),flush=True)
 atomic_json(U/f'support_rank{rank}.json',{'images':names,'unique_pair_images':counts[0].cpu().tolist(),
  'positive_unique_pair_images':counts[1].cpu().tolist(),'negative_unique_pair_images':counts[2].cpu().tolist(),
  'best_sha256':digest(best),'source_sha256':digest(E/'support.py'),'evidence_sha256':digest(cfg['ald_evidence']),
  'GT_role':'No pixel GT or old image tags; every one of2145 training images once, resize448, no augmentation; validation names excluded','utc':now()})
if __name__=='__main__':main()
