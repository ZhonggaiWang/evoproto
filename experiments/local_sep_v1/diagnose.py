"""Full-validation ALD credibility audit; no optimizer and no training outputs."""
from pathlib import Path
import os,sys,json,argparse
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto');E=R/'experiments/local_sep_v1';U=R/'runs/local_sep_v1';S=E/'src'
sys.path.insert(0,str(R/'experiments/kd_pixel_v2'))
from run_kd import environment
os.environ.update(environment('8card'));sys.path.insert(0,str(S))
import torch
import torch.nn.functional as F
from model.model_seg_neg import network
from model.ald import fuse as previous_fuse
sys.path.insert(0,str(E))
from model.ald import image_targets,fuse,gate_auxiliary
from model.online_directed_confusion import build_confusion_evidence
from model.PAR import PAR
from datasets import voc
from utils.camutils import multi_scale_cam2,cam_to_label,refine_cams_with_bkg_v2
from utils.imutils import denormalize_img2
from kd_runtime import safe_path,atomic_json,digest,now
sys.path.insert(0,str(R/'experiments/ald_calibration_v1'))
from evidence import calibrate_presence

p=argparse.ArgumentParser();p.add_argument('--rank',type=int,required=True);args=p.parse_args();rank=args.rank
torch.cuda.set_device(rank);torch.set_num_threads(1);torch.set_float32_matmul_precision('high');torch.backends.cudnn.allow_tf32=True
torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
run=R/'runs/ald_calibration_v8/formal';study=json.loads(safe_path(run/'study.json').read_text());cfg=study['config']
assert json.loads((run/'status.json').read_text())['status']=='complete'
def load(path,classes):
    path=safe_path(path);ck=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    m=network(backbone=cfg['backbone'],num_classes=sum(classes),classes_list=classes,pretrained=False,init_momentum=cfg['momentum'],aux_layer=cfg['aux_layer'])
    m.load_state_dict(ck['model_state'],strict=True);return m.cuda().eval().requires_grad_(False)
candidate_path=R/'runs/ald_calibration_v9/formal/10-5/step2/checkpoints/model_final.pth'
teacher=load(cfg['prev_checkpoint'],[11,5]);reference=load(cfg['ald_reference'],[11,5,5]);candidate=load(candidate_path,[11,5,5])
assert digest(cfg['ald_reference'])==study['reference_sha256'] and digest(cfg['prev_checkpoint'])==study['teacher_sha256']
par_model=PAR(num_iter=10,dilations=[1,2,4,8,12,24]).cuda()
ds=voc.VOC12SegDataset(root_dir=cfg['data_folder'],name_list_dir=cfg['list_folder'],split='val',stage='val',aug=False,ignore_index=255,num_classes=21,tasks='10-5',step=2)
ds.label_dir=cfg['val_label_dir'];assert len(ds)==1449
train=torch.load(cfg['ald_evidence'],map_location='cpu',weights_only=True)
assert not set(train['names'])&set(map(str,ds.name_list))
pixel={key:torch.zeros(22,22,dtype=torch.long,device='cuda') for key in ['legacy_fused','previous_ALD_fused','ALD_fused','reference_unknown','confusion_anchor']}
classification={key:torch.zeros(20,2,2,dtype=torch.long,device='cuda') for key in ['reference','candidate']}
presence_hist={key:torch.zeros(15,2,2,dtype=torch.long,device='cuda') for key in ['raw_teacher','positive','negative','unknown']}
pairs=torch.zeros(21,21,dtype=torch.long,device='cuda');pair_correct=pairs.clone()
counts={key:0 for key in ['valid_pixels','unknown_valid','legacy_correct_removed','legacy_wrong_removed','unsupported_old_valid','unsupported_old_wrong','unknown_GT_allowed','unknown_GT_excluded']}
names=[]
def hist(gt,pred):
    good=(gt>=0)&(gt<21);g=torch.where(good,gt,21);p=torch.where((pred>=0)&(pred<21),pred,21)
    return torch.bincount((g*22+p).flatten(),minlength=484).reshape(22,22)
with torch.inference_mode():
    for index in range(rank,len(ds),8):
        name,img,gt,tags=ds[index];names.append(str(name))
        x=F.interpolate(torch.as_tensor(img,device='cuda').float()[None],(448,448),mode='bilinear',align_corners=False)
        new_tags=torch.as_tensor(tags,device='cuda')[None,15:].float();box=[[0,448,0,448]]
        x2=torch.cat([x,x.flip(-1)],0)
        oc,oa,tl,_,_=teacher(x2,step0=True);rc,rl,_,_=reference(x2)
        tv=torch.stack([tl[0],tl[1].flip(-1)],0)[None];rv=torch.stack([rl[0],rl[1].flip(-1)],0)[None]
        ref_cams,_=multi_scale_cam2(reference,x,scales=cfg['cam_scales'])
        presence=calibrate_presence(oc[None],oa[None],tv,rv,ref_cams,new_tags)
        _,allowed,_=image_targets(presence['state'],oc[:1],oa[:1],tl[:1],rl[:1],new_tags)
        cc,cl,_,_=candidate(x);cams,_=multi_scale_cam2(candidate,x,scales=cfg['cam_scales'])
        def refine(mask):
            valid,_=cam_to_label(cams,cls_label=mask,img_box=box,ignore_mid=True,bkg_thre=.5,high_thre=.7,low_thre=.25,ignore_index=255)
            labels=refine_cams_with_bkg_v2(par_model,denormalize_img2(x.clone()),cams=valid,cls_labels=mask,high_thre=.7,low_thre=.25,ignore_index=255,img_box=box)
            return valid,labels
        valid_cams,labels=refine(allowed)
        fused=fuse(labels,tl[:1],rl[:1],presence['state'],box)
        previous=previous_fuse(labels,tl[:1],rl[:1],presence['state'],box)
        _,legacy_par=refine(torch.cat([oc[:1]>0,new_tags.bool()],1))
        tp=F.interpolate(tl[:1],(448,448),mode='bilinear',align_corners=False).argmax(1)
        legacy=tp.clone();isnew=(legacy_par>=16)&(legacy_par<21);legacy[isnew]=legacy_par.long()[isnew]
        anchors=gate_auxiliary(labels,presence['state'],rl[:1],calibrate_new=False)
        evidence=build_confusion_evidence(cl,anchors,valid_cams,box)
        # All masks/targets above use weak labels and predictions only.
        # GT starts here and is used exclusively to audit those fixed choices.
        truth=torch.as_tensor(tags,device='cuda').long()
        for key,logits in [('reference',rc[:1]),('candidate',cc)]:
            pos=(logits[0]>0).long();code=torch.arange(20,device='cuda')*4+truth*2+pos
            classification[key]+=torch.bincount(code,minlength=80).reshape(20,2,2)
        for key,pos in [('raw_teacher',oc[:1]>0),('positive',presence['positive']),('negative',presence['negative']),('unknown',presence['unknown'])]:
            code=torch.arange(15,device='cuda')*4+truth[:15]*2+pos[0].long()
            presence_hist[key]+=torch.bincount(code,minlength=60).reshape(15,2,2)
        g=F.interpolate(torch.as_tensor(gt,device='cuda').float()[None,None],(448,448),mode='nearest')[:,0].long();good=(g>=0)&(g<21)
        unknown=fused['unknown'];rp=fused['reference_prediction']
        pixel['previous_ALD_fused']+=hist(g,previous['labels'])
        pixel['legacy_fused']+=hist(g,legacy);pixel['ALD_fused']+=hist(g,fused['labels'])
        pixel['reference_unknown']+=hist(g,torch.where(unknown,rp,255))
        accepted=evidence['accepted'];anchor=evidence['anchors'];pred=evidence['predictions']
        pixel['confusion_anchor']+=hist(g,torch.where(accepted,anchor,255))
        mask=accepted&good;ids=(anchor*21+pred)[mask]
        pairs+=torch.bincount(ids,minlength=441).reshape(21,21)
        pair_correct+=torch.bincount((anchor*21+pred)[mask&(anchor==g)],minlength=441).reshape(21,21)
        unsupported=(tp>0)&(presence['state'].gather(1,(tp-1).clamp(0,14).flatten(1)).reshape_as(tp)!=1)&good
        allowed_gt=torch.cat([torch.ones(1,1,device='cuda',dtype=torch.bool),presence['state']!=0,new_tags.bool()],1).gather(1,g.clamp(0,20).flatten(1)).reshape_as(g)
        masks={'unknown_GT_allowed':unknown&good&allowed_gt,'unknown_GT_excluded':unknown&good&~allowed_gt,'valid_pixels':good,'unknown_valid':unknown&good,'legacy_correct_removed':unknown&good&(legacy==g),
            'legacy_wrong_removed':unknown&good&(legacy!=g),'unsupported_old_valid':unsupported,'unsupported_old_wrong':unsupported&(tp!=g)}
        for key,mask in masks.items():counts[key]+=int(mask.sum())
        if len(names)%48==0:print(json.dumps({'rank':rank,'images':len(names)}),flush=True)
atomic_json(U/f'diagnostic_rank{rank}.json',{'rank':rank,'images':names,'counts':counts,'resolution':[448,448],
    'pixel_histograms':{k:v.cpu().tolist() for k,v in pixel.items()},'classification':{k:v.cpu().tolist() for k,v in classification.items()},
    'presence':{k:v.cpu().tolist() for k,v in presence_hist.items()},'accepted_directed_pairs':pairs.cpu().tolist(),'correct_source_directed_pairs':pair_correct.cpu().tolist(),
    'candidate_sha256':digest(candidate_path),'reference_sha256':study['reference_sha256'],'diagnostic_sha256':digest(E/'diagnose.py'),'ALD_sha256':digest(S/'model/ald.py'),
    'GT_role':'Only after all calibration targets and masks; no optimizer or training state writes','utc':now()})
