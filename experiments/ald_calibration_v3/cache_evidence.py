from pathlib import Path
import os,sys,json,copy
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto');E=R/'experiments/ald_calibration_v3';U=R/'runs/ald_calibration_v3'
BEST=R/'runs/pair_preserving_kd_v1/formal/10-5/step2/checkpoints/model_final.pth';CAL=R/'runs/ald_calibration_v2/formal/10-5/step2/checkpoints/model_final.pth'
rank=int(os.environ['ALD_RANK'])
sys.path.insert(0,str(R/'experiments/hierarchy_kd_v2'))
import diagnose_new_transfer as base
sys.path.insert(0,str(R/'experiments/ald_calibration_v1'))
from evidence import calibrate_presence
from proposals import proposals

def cache(ctx,protocol):
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
    prior=json.loads(safe_path(R/'runs/ald_calibration_v1'/f'cache_receipt_rank{rank}.json').read_text());assert prior['passed']
    ds=ctx['voc'].VOC12ClsDataset(root_dir=cfg['data_folder'],name_list_dir=cfg['list_folder'],split='train',stage='train',aug=False,ignore_index=255,num_classes=21,tasks='10-5',step=2)
    assert len(ds)==2145
    names=[];values={};counts={key:0 for key in ['background','old','new_teacher_old','new_teacher_BG']}
    with torch.inference_mode():
        for i in range(rank,len(ds),8):
            name,img,tags=ds[i];assert str(name)==prior['images'][len(names)]
            x=F.interpolate(torch.as_tensor(img,device=device).float()[None],(448,448),mode='bilinear',align_corners=False)
            nt=torch.as_tensor(tags,device=device)[None,15:];xx=torch.cat([x,x.flip(-1)],0);box=torch.tensor([[0,448,0,448]])
            oc,oa,t,_,_=ctx['teacher'](xx,step0=True);cls,seg,_,aux=model(xx)
            tv=torch.stack([t[0],t[1].flip(-1)],0)[None];rv=torch.stack([seg[0],seg[1].flip(-1)],0)[None]
            cams,_=ctx['multi_scale_cam2'](model,x,scales=cfg['cam_scales'])
            presence=calibrate_presence(oc[None],oa[None],tv,rv,cams,nt)
            allowed=torch.cat([presence['positive'],nt.bool()],1)
            valid,_=ctx['cam_to_label'](cams,cls_label=allowed,img_box=box,ignore_mid=True,bkg_thre=cfg['bkg_thre'],high_thre=cfg['high_thre'],low_thre=cfg['low_thre'],ignore_index=255)
            par=ctx['refine_cams_with_bkg_v2'](ctx['par'],ctx['denormalize_img2'](x.clone()),cams=valid,cls_labels=allowed,high_thre=cfg['high_thre'],low_thre=cfg['low_thre'],ignore_index=255,img_box=box)
            z=proposals(presence,tv,rv,par,cams,nt)
            for key in counts:counts[key]+=int(z[key].sum())
            for key,value in z.items():values.setdefault(key,[]).append(value[0].cpu().clone())
            names.append(str(name))
            if len(names)%64==0:print(json.dumps({'rank':rank,'images':len(names),'counts':counts}),flush=True)
    target=safe_path(U/f'cache_rank{rank}.pth');assert not target.exists()
    torch.save({'names':names,**{k:torch.stack(v) for k,v in values.items()}},target)
    return {'readiness_coverage':counts,'rank':rank,'images':names,'cache':str(target),'cache_sha256':digest(target),
        'best_reference_sha256':digest(BEST),'calibrator_sha256':digest(CAL),'proposal_sha256':digest(E/'proposals.py'),
        'ALD_v1_cache_sha256':prior['cache_sha256'],'semantics':'All 2145 train images, refreshed CAM/PAR using calibrated classifiers and best segmentation head; no old image labels or pixel GT.'}

base.OUTPUT=U/f'cache_receipt_rank{rank}.json';base.old_background_mode=cache
if __name__=='__main__':base.main()
