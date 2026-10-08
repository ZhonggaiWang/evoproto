"""Read-only full-validation calibration diagnosis in eight disjoint shards."""
from pathlib import Path
import os,sys,json
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto');E=R/'experiments/ald_calibration_v1';U=R/'runs/ald_calibration_v1'
BEST=R/'runs/pair_preserving_kd_v1/formal/10-5/step2/checkpoints/model_final.pth'
rank=int(os.environ['ALD_RANK'])
sys.path.insert(0,str(R/'experiments/hierarchy_kd_v2'))
import diagnose_new_transfer as base
from evidence import calibrate_presence,fuse_supervision

def diagnosis(ctx,protocol):
    torch,F,cfg,device=ctx['torch'],ctx['F'],ctx['cfg'],ctx['device']
    from kd_runtime import digest,safe_path
    previous=json.loads(safe_path(R/'runs/pair_preserving_kd_v1/analysis.json').read_text())
    assert digest(safe_path(BEST))==previous['checkpoint_sha256']
    best=torch.load(BEST,map_location='cpu',weights_only=True,mmap=True)
    # Parent and best differ only in this final head. Verify before using the
    # parent's unchanged encoder/classifier to recover the exact best model.
    for key,value in ctx['student'].state_dict().items():
        if not key.startswith('decoder.conv8.'):
            assert torch.equal(value.detach().cpu(),best['model_state'][key]),key
    heads=[best['model_state'][f'decoder.conv8.{i}.weight'].to(device) for i in range(3)]
    del best
    capture={}
    def hook(module,args):capture['feature']=args[0].detach()
    handle=ctx['student'].decoder.conv8.register_forward_pre_hook(hook)
    ds=ctx['voc'].VOC12SegDataset(root_dir=cfg['data_folder'],name_list_dir=cfg['list_folder'],split=cfg['val_set'],
        stage='val',aug=False,ignore_index=255,num_classes=21,tasks='10-5',step=2)
    ds.label_dir=cfg['val_label_dir'];assert len(ds)==1449
    classification={key:torch.zeros(15,2,2,dtype=torch.int64,device=device) for key in
        ['legacy','main_aux_positive','main_flip_positive','calibrated_positive','calibrated_negative','unknown']}
    pixel={key:torch.zeros(22,22,dtype=torch.int64,device=device) for key in ['teacher','legacy_mixed','calibrated_PAR','calibrated_fused','reference']}
    counts={key:0 for key in ['valid_pixels','legacy_correct_kept','legacy_wrong_removed','legacy_correct_removed',
        'changed_hard_labels','changed_labels_correct','changed_labels_wrong','old_unsupported_class_pixels','old_unsupported_class_wrong']}
    records=[];names=[]
    with torch.inference_mode():
        for index in range(rank,len(ds),8):
            name,img,gt,tags=ds[index];names.append(str(name))
            x=F.interpolate(torch.as_tensor(img,device=device).float()[None],(448,448),mode='bilinear',align_corners=False)
            tags=torch.as_tensor(tags,device=device)[None];new_tags=tags[:,15:]
            box=torch.tensor([[0,448,0,448]]);x2=torch.cat([x,x.flip(-1)],0)
            old_cls,old_aux,old_logits,_,_=ctx['teacher'](x2,step0=True)
            _=ctx['student'](x2,crops=[]);feat=capture.pop('feature')
            best_logits=torch.cat([F.conv2d(feat,w) for w in heads],1)
            tv=torch.stack([old_logits[0],old_logits[1].flip(-1)],0)[None]
            rv=torch.stack([best_logits[0],best_logits[1].flip(-1)],0)[None]
            cams,_=ctx['multi_scale_cam2'](ctx['student'],x,scales=cfg['cam_scales'])
            presence=calibrate_presence(old_cls[None],old_aux[None],tv,rv,cams,new_tags,cfg['high_thre'])
            def refine(allowed):
                valid,_=ctx['cam_to_label'](cams.detach(),cls_label=allowed,img_box=box,ignore_mid=True,
                    bkg_thre=cfg['bkg_thre'],high_thre=cfg['high_thre'],low_thre=cfg['low_thre'],ignore_index=255)
                par=ctx['refine_cams_with_bkg_v2'](ctx['par'],ctx['denormalize_img2'](x.clone()),cams=valid,
                    cls_labels=allowed,high_thre=cfg['high_thre'],low_thre=cfg['low_thre'],ignore_index=255,img_box=box)
                return valid,par
            legacy_allowed=torch.cat([presence['raw_positive'],new_tags.bool()],1)
            legacy_cam,legacy_par=refine(legacy_allowed)
            calibrated_allowed=torch.cat([presence['positive'],new_tags.bool()],1)
            calibrated_cam,calibrated_par=refine(calibrated_allowed)
            fused=fuse_supervision(presence,tv,rv,calibrated_par,calibrated_cam,new_tags,cfg['high_thre'],cfg['low_thre'])
            size=best_logits.shape[-2:]
            teacher_label=tv[:,0].argmax(1)
            legacy_par_native=F.interpolate(legacy_par[:,None].float(),size,mode='nearest')[:,0].long()
            legacy=teacher_label.clone();new=(legacy_par_native>=16)&(legacy_par_native<21);legacy[new]=legacy_par_native[new]
            preds={'teacher':teacher_label,'legacy_mixed':legacy,'calibrated_PAR':F.interpolate(calibrated_par[:,None].float(),size,mode='nearest')[:,0].long(),
                'calibrated_fused':fused['labels'],'reference':rv[:,0].argmax(1)}
            # Only now read old-class image truth and pixel GT, exclusively for reporting.
            old_truth=tags[:,:15].bool()
            proposed={'legacy':presence['raw_positive'],'main_aux_positive':(old_cls[:1]>0)&(old_aux[:1]>0),
                'main_flip_positive':((old_cls>0).all(0))[None],'calibrated_positive':presence['positive'],
                'calibrated_negative':presence['negative'],'unknown':presence['unknown']}
            for key,positive in proposed.items():
                for c in range(15):classification[key][c,int(old_truth[0,c]),int(positive[0,c])]+=1
            g=F.interpolate(torch.as_tensor(gt,device=device).float()[None,None],size,mode='nearest')[:,0].long()
            valid=(g>=0)&(g<21);gcol=torch.where(valid,g,21)
            for key,pred in preds.items():
                pred=torch.where((pred>=0)&(pred<21),pred,21)
                pixel[key]+=torch.bincount((gcol*22+pred).flatten(),minlength=484).reshape(22,22)
            accepted=fused['labels']<21;changed=accepted&(fused['labels']!=legacy)&valid
            unsupported=(teacher_label>0)&~presence['positive'].gather(1,(teacher_label-1).clamp(0,14).flatten(1)).reshape_as(g)&valid
            masks={'valid_pixels':valid,'legacy_correct_kept':accepted&(legacy==g)&valid,
                'legacy_wrong_removed':~accepted&(legacy!=g)&valid,'legacy_correct_removed':~accepted&(legacy==g)&valid,
                'changed_hard_labels':changed,'changed_labels_correct':changed&(fused['labels']==g),
                'changed_labels_wrong':changed&(fused['labels']!=g),'old_unsupported_class_pixels':unsupported,
                'old_unsupported_class_wrong':unsupported&(teacher_label!=g)}
            for key,mask in masks.items():counts[key]+=int(mask.sum())
            records.append({'name':str(name),'raw_old_positive':presence['raw_positive'].nonzero()[:,1].add(1).tolist(),
                'calibrated_old_positive':presence['positive'].nonzero()[:,1].add(1).tolist(),
                'calibrated_old_negative':presence['negative'].nonzero()[:,1].add(1).tolist(),
                'unknown_old':presence['unknown'].nonzero()[:,1].add(1).tolist(),'GT_old_presence_diagnostic_only':old_truth.nonzero()[:,1].add(1).tolist()})
            if len(names)%32==0:print(json.dumps({'rank':rank,'images':len(names)}),flush=True)
    handle.remove()
    return {'readiness_coverage':counts,'rank':rank,'shards':8,'images':names,'records':records,
        'image_classification':{k:v.cpu().tolist() for k,v in classification.items()},
        'pixel_confusions':{k:v.cpu().tolist() for k,v in pixel.items()},
        'best_reference_checkpoint':str(BEST),'best_reference_sha256':previous['checkpoint_sha256'],
        'ALD_source_sha256':digest(E/'evidence.py'),'GT_role_override':'Old image tags and pixel GT enter only after all states, PAR and fused labels are constructed. Diagnostics never become training labels.'}

base.OUTPUT=U/f'diagnostic_rank{rank}.json';base.old_background_mode=diagnosis
if __name__=='__main__':base.main()
