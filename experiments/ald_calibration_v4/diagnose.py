from pathlib import Path
import os,sys,json,copy
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto');E=R/'experiments/ald_calibration_v4';U=R/'runs/ald_calibration_v4'
BEST=R/'runs/pair_preserving_kd_v1/formal/10-5/step2/checkpoints/model_final.pth';CAL=R/'runs/ald_calibration_v2/formal/10-5/step2/checkpoints/model_final.pth'
rank=int(os.environ['ALD_RANK'])
sys.path.insert(0,str(R/'experiments/hierarchy_kd_v2'))
import diagnose_new_transfer as base
sys.path.insert(0,str(R/'experiments/ald_calibration_v1'))
from evidence import calibrate_presence
from proposals import proposals

def diagnosis(ctx,protocol):
    torch,F,cfg,device=ctx['torch'],ctx['F'],ctx['cfg'],ctx['device']
    from kd_runtime import digest,safe_path
    assert digest(safe_path(BEST))=='3f43207d665ce6001aa149867435364e13baca5d47a0e2274dfb9187975dd28d'
    assert digest(safe_path(CAL))=='0d6fcb1e8b83a3a79155bb480f33b11cf8895cef068968c42651d041ad93333c'
    best=torch.load(BEST,map_location='cpu',weights_only=True,mmap=True)
    calibrated=torch.load(CAL,map_location='cpu',weights_only=True,mmap=True)
    from model.model_seg_neg import network
    model=network(backbone=cfg['backbone'],num_classes=21,classes_list=[11,5,5],pretrained=False,init_momentum=cfg['momentum'],aux_layer=cfg['aux_layer']).to(device).eval().requires_grad_(False)
    state=dict(best['model_state'])
    for key,v in calibrated['model_state'].items():
        if key.startswith(('classifier.','aux_classifier.')):state[key]=v
        elif not key.startswith('decoder.conv8.'):assert torch.equal(v,best['model_state'][key])
    model.load_state_dict(state,strict=True)
    ds=ctx['voc'].VOC12SegDataset(root_dir=cfg['data_folder'],name_list_dir=cfg['list_folder'],split=cfg['val_set'],stage='val',aug=False,ignore_index=255,num_classes=21,tasks='10-5',step=2)
    ds.label_dir=cfg['val_label_dir'];assert len(ds)==1449
    keys=['background','old','new_teacher_old','new_teacher_BG']
    counts={key:{'selected':0,'correct':0,'reference_correct':0,'net_correct':0,'GT': [0]*21,'matrix':[[0]*21 for _ in range(21)]} for key in keys}
    names=[]
    prior_records={r['name']:r['calibrated_old_positive'] for j in range(8) for r in json.loads(safe_path(R/'runs/ald_calibration_v1'/f'diagnostic_rank{j}.json').read_text())['records']}
    with torch.inference_mode():
        for i in range(rank,len(ds),8):
            name,img,gt,tags=ds[i];names.append(str(name))
            x=F.interpolate(torch.as_tensor(img,device=device).float()[None],(448,448),mode='bilinear',align_corners=False)
            nt=torch.as_tensor(tags,device=device)[None,15:];xx=torch.cat([x,x.flip(-1)],0);box=torch.tensor([[0,448,0,448]])
            oc,oa,t,_,_=ctx['teacher'](xx,step0=True);cls,seg,_,aux=model(xx)
            tv=torch.stack([t[0],t[1].flip(-1)],0)[None];rv=torch.stack([seg[0],seg[1].flip(-1)],0)[None]
            cams,_=ctx['multi_scale_cam2'](model,x,scales=cfg['cam_scales'])
            presence=calibrate_presence(oc[None],oa[None],tv,rv,cams,nt)
            # Retain spatially confirmed positive classes; do not let the
            # classifier's reduced recall silently erase spatial evidence.
            allowed=torch.cat([presence['positive'],nt.bool()],1)
            valid,_=ctx['cam_to_label'](cams,cls_label=allowed,img_box=box,ignore_mid=True,bkg_thre=cfg['bkg_thre'],high_thre=cfg['high_thre'],low_thre=cfg['low_thre'],ignore_index=255)
            par=ctx['refine_cams_with_bkg_v2'](ctx['par'],ctx['denormalize_img2'](x.clone()),cams=valid,cls_labels=allowed,high_thre=cfg['high_thre'],low_thre=cfg['low_thre'],ignore_index=255,img_box=box)
            z=proposals(presence,tv,rv,par,cams,nt)
            _,_,single_teacher,_,_=ctx['teacher'](x,step0=True)
            tp=single_teacher.argmax(1)
            confirmed=torch.zeros(1,15,device=device,dtype=torch.bool)
            confirmed[0,[c-1 for c in prior_records[str(name)]]]=True
            supported=confirmed.gather(1,(tp-1).clamp(0,14).flatten(1)).reshape_as(tp)
            z['new_teacher_old']&=~supported

            g=F.interpolate(torch.as_tensor(gt,device=device).float()[None,None],seg.shape[-2:],mode='nearest')[:,0].long();validgt=(g>=0)&(g<21)
            for key in keys:
                mask=z[key]&validgt;out=z['labels'];ref=z['reference_prediction'];c=counts[key]
                selected=int(mask.sum());good=int((mask&(out==g)).sum());previous=int((mask&(ref==g)).sum())
                c['selected']+=selected;c['correct']+=good;c['reference_correct']+=previous;c['net_correct']+=good-previous
                hist=torch.bincount(g[mask],minlength=21).cpu().tolist();c['GT']=[a+b for a,b in zip(c['GT'],hist)]
                matrix=torch.bincount((out[mask]*21+ref[mask]),minlength=441).reshape(21,21).cpu().tolist();c['matrix']=[[a+b for a,b in zip(aa,bb)] for aa,bb in zip(c['matrix'],matrix)]
            if len(names)%48==0:print(json.dumps({'rank':rank,'images':len(names)}),flush=True)
    return {'readiness_coverage':{k:v['selected'] for k,v in counts.items()},'rank':rank,'images':names,'corrections':counts,
        'best_reference_sha256':digest(BEST),'calibrator_sha256':digest(CAL),'proposal_sha256':digest(E/'proposals.py'),
        'GT_role':'Only after proposal construction, image-old tags unused; diagnose proposed changes rather than precision on already-agreeing pixels.'}

base.OUTPUT=U/f'diagnostic_rank{rank}.json';base.old_background_mode=diagnosis
if __name__=='__main__':base.main()
