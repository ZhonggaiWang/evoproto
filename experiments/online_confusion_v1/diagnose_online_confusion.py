"""Frozen online stream diagnostic. Pixel GT enters only post-evidence scoring."""
from pathlib import Path
import os,sys,json,argparse,time
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto')
E=R/'experiments/online_confusion_v1';U=R/'runs/online_confusion_v1'
os.chdir(R);sys.path.insert(0,str(R/'experiments/kd_pixel_v2'))
from run_kd import environment
os.environ.update(environment('8card'))
sys.path.insert(0,str(E/'src'))
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
import tasks
from datasets import voc
from model.model_seg_neg import network
from model.PAR import PAR
from model.online_directed_confusion import OnlineDirectedConfusion,pair_counts
from utils.camutils import multi_scale_cam2,cam_to_label,refine_cams_with_bkg_v2
from utils.imutils import denormalize_img2
from kd_runtime import atomic_json,now,digest,safe_path


class ImageStream(Dataset):
    """Reads only images and existing image labels; no pixel annotation reads."""
    def __init__(self,cfg,split,rank,shards):
        path=Path(cfg['list_folder'])/'incremental_split'/f'{split}_10-5_step_3.txt'
        self.names=np.atleast_1d(np.loadtxt(path,dtype=str))
        self.indices=list(range(rank,len(self.names),shards))
        self.labels=voc.load_cls_label_list(cfg['list_folder'])
        self.root=Path(cfg['data_folder'])/'JPEGImages'
    def __len__(self):return len(self.indices)
    def __getitem__(self,i):
        index=self.indices[i];name=str(self.names[index])
        with Image.open(self.root/(name+'.jpg')) as image:
            array=np.array(image.convert('RGB'),copy=True)
        x=torch.from_numpy(array).permute(2,0,1).float()/255
        x=(x-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]
        x=F.interpolate(x[None],size=(448,448),mode='bilinear',align_corners=False)[0]
        return index,name,x,torch.as_tensor(self.labels[name]).float()


def main(args):
    torch.cuda.set_device(0);torch.set_num_threads(1)
    torch.set_float32_matmul_precision('high');torch.backends.cudnn.allow_tf32=True
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    run=R/'runs/kd_parallel_v1/formal/b_relational'
    cfg=json.loads((run/'10-5/step2/config.json').read_text())
    iteration=2000 if args.checkpoint=='warmup' else 8000
    checkpoint=run/f'10-5/step2/checkpoints/model_iter_{iteration}.pth'
    teacher_path=run/'10-5/step1/checkpoints/model_final.pth'
    def load(step,path):
        classes=tasks.get_per_task_classes('voc','10-5',step)
        m=network(backbone=cfg['backbone'],num_classes=sum(classes),classes_list=classes,
                  pretrained=False,init_momentum=cfg['momentum'],aux_layer=cfg['aux_layer'])
        m.load_state_dict(torch.load(path,map_location='cpu',weights_only=True,mmap=True)['model_state'])
        return m.cuda().eval().requires_grad_(False)
    student=load(2,checkpoint);teacher=load(1,teacher_path)
    observer=OnlineDirectedConfusion(21,stage=2).cuda()
    par=PAR(num_iter=10,dilations=[1,2,4,8,12,24]).cuda()
    ds=ImageStream(cfg,args.split,args.rank,args.shards)
    loader=DataLoader(ds,batch_size=args.batch,num_workers=2,shuffle=False,pin_memory=False)
    prefix=safe_path(U/f'{args.checkpoint}_{args.split}_rank{args.rank}')
    records=[];start=time.monotonic()
    stat_names=['estimated','accepted','broad','evaluated_estimated','evaluated_accepted','evaluated_broad','gt_all','gt_selected','gt_broad_selected','pseudo_gt','broad_pseudo_gt','anchor_correct',
                'gt_covered_unweighted','teacher_gt_all','teacher_pseudo','bin_pseudo_gt']
    lists={key:[] for key in stat_names};image_indices=[];image_names=[]
    with torch.inference_mode():
        for batch_num,(indices,names,images,image_labels) in enumerate(loader):
            x=images.cuda();image_labels=image_labels.cuda()
            box=torch.tensor([[0,448,0,448]]*len(x))
            old_cls,_,old_logits,_,_=teacher(x,step0=True)
            merged=torch.cat(((old_cls>0).long(),image_labels[:,-5:]),1)
            cams,_=multi_scale_cam2(student,x,scales=cfg['cam_scales'])
            valid_cam,_=cam_to_label(cams.detach(),cls_label=merged,img_box=box,ignore_mid=True,
                bkg_thre=cfg['bkg_thre'],high_thre=cfg['high_thre'],low_thre=cfg['low_thre'],ignore_index=255)
            pseudo=refine_cams_with_bkg_v2(par,denormalize_img2(x.clone()),cams=valid_cam,cls_labels=merged,
                high_thre=cfg['high_thre'],low_thre=cfg['low_thre'],ignore_index=255,img_box=box)
            _,_,logits,_,_=student(x,step0=True)
            evidence=observer.update(logits,pseudo,valid_cam,box,synchronize=False)
            old_prediction=F.interpolate(old_logits,size=(448,448),mode='bilinear',align_corners=False).argmax(1)
            # Everything used to model directed confusion has now been computed.
            # GT is read below ONLY for detached scoring, never fed to observer.
            for j,name in enumerate(names):
                a=evidence['anchors'][j];p=evidence['predictions'][j]
                w=evidence['weights'][j];mask=evidence['accepted'][j];broad=evidence['broad'][j]
                result={'estimated':pair_counts(a,p,21,weights=w),'accepted':pair_counts(a,p,21,mask=mask),
                        'broad':pair_counts(a,p,21,mask=broad)}
                if args.split=='val':
                    with Image.open(Path(cfg['val_label_dir'])/(name+'.png')) as f:
                        gt=torch.from_numpy(np.array(f,copy=True)).long()[None,None].cuda()
                    gt=F.interpolate(gt.float(),size=(448,448),mode='nearest')[0,0].long()
                    good=(gt>=0)&(gt<21)
                    result.update(evaluated_estimated=pair_counts(a,p,21,weights=w,mask=good),
                        evaluated_accepted=pair_counts(a,p,21,mask=mask&good),
                        evaluated_broad=pair_counts(a,p,21,mask=broad&good),
                        gt_all=pair_counts(gt,p,21),
                        gt_selected=pair_counts(gt,p,21,weights=w),
                        gt_broad_selected=pair_counts(gt,p,21,mask=broad),
                        pseudo_gt=pair_counts(a,gt,21,weights=w),
                        broad_pseudo_gt=pair_counts(a,gt,21,mask=broad),
                        anchor_correct=pair_counts(gt,p,21,weights=w,mask=(a==gt)),
                        gt_covered_unweighted=pair_counts(gt,p,21,mask=mask),
                        teacher_gt_all=pair_counts(gt,old_prediction[j],21),
                        teacher_pseudo=pair_counts(a,old_prediction[j],21,weights=w))
                    bins=[]
                    for lo,hi in [(0,.25),(.25,.5),(.5,.75),(.75,1.00001)]:
                        bins.append(pair_counts(a,gt,21,mask=broad&(evidence['cam_gap'][j]>=lo)&(evidence['cam_gap'][j]<hi)))
                    result['bin_pseudo_gt']=torch.stack(bins)
                for key,value in result.items():lists[key].append(value.cpu().numpy())
                image_indices.append(int(indices[j]));image_names.append(name)
            if batch_num%10==0:
                print(json.dumps({'checkpoint':args.checkpoint,'split':args.split,'rank':args.rank,'processed':len(image_names),'total':len(ds),'seconds':time.monotonic()-start}),flush=True)
                atomic_json(prefix.with_suffix('.progress.json'),{'utc':now(),'processed':len(image_names),'total':len(ds)})
    arrays={key:np.stack(items) for key,items in lists.items() if items}
    arrays.update(indices=np.asarray(image_indices),names=np.asarray(image_names))
    with open(prefix.with_suffix('.npz'),'xb') as f:np.savez_compressed(f,**arrays)
    atomic_json(prefix.with_suffix('.json'),{'utc':now(),'checkpoint':str(checkpoint),'checkpoint_sha256':digest(checkpoint),
        'teacher':str(teacher_path),'teacher_sha256':digest(teacher_path),'split':args.split,'shard':args.rank,'shards':args.shards,
        'images':len(image_names),'seconds':time.monotonic()-start,'batch':args.batch,'gt_usage':'scoring only after observer update' if args.split=='val' else 'no pixelGT read',
        'diagnostic_scope':'448 frozen inference stream; not augmented training and not benchmark mIoU','observer':observer.export()})
    print(json.dumps({'completed':str(prefix),'images':len(image_names),'seconds':time.monotonic()-start}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',choices=['warmup','final'],required=True)
    p.add_argument('--split',choices=['train','val'],required=True);p.add_argument('--rank',type=int,required=True)
    p.add_argument('--shards',type=int,default=4);p.add_argument('--batch',type=int,default=4)
    main(p.parse_args())
