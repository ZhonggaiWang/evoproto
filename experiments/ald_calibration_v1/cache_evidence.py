"""Training-image weak evidence, with no access to segmentation annotations."""
from pathlib import Path
import os,sys,json
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto');E=R/'experiments/ald_calibration_v1';U=R/'runs/ald_calibration_v1'
BEST=R/'runs/pair_preserving_kd_v1/formal/10-5/step2/checkpoints/model_final.pth'
rank=int(os.environ['ALD_RANK'])
sys.path.insert(0,str(R/'experiments/hierarchy_kd_v2'))
import diagnose_new_transfer as base
from evidence import calibrate_presence,fuse_supervision

def cache(ctx,protocol):
    torch,F,cfg,device=ctx['torch'],ctx['F'],ctx['cfg'],ctx['device']
    from kd_runtime import digest,safe_path
    best=torch.load(safe_path(BEST),map_location='cpu',weights_only=True,mmap=True)
    assert digest(BEST)=='3f43207d665ce6001aa149867435364e13baca5d47a0e2274dfb9187975dd28d'
    for key,value in ctx['student'].state_dict().items():
        if not key.startswith('decoder.conv8.'):assert torch.equal(value.cpu(),best['model_state'][key]),key
    heads=[best['model_state'][f'decoder.conv8.{i}.weight'].to(device) for i in range(3)]
    capture={};handles=[]
    def hook(key):
        def get(module,args):capture[key]=args[0].detach()
        return get
    for key,module in [('feature',ctx['student'].decoder.conv8),('pooled',ctx['student'].classifier),('aux_pooled',ctx['student'].aux_classifier)]:
        handles.append(module.register_forward_pre_hook(hook(key)))
    prior=R/'runs/pair_preserving_kd_v1'
    receipt=json.loads(safe_path(prior/f'cache_receipt_rank{rank}.json').read_text())
    assert receipt['passed'] and digest(safe_path(receipt['cache']))==receipt['cache_sha256']
    oldcache=torch.load(receipt['cache'],map_location='cpu',weights_only=True,mmap=True)
    ds=ctx['voc'].VOC12ClsDataset(root_dir=cfg['data_folder'],name_list_dir=cfg['list_folder'],split='train',stage='train',tasks='10-5',step=2,aug=False,num_classes=21,ignore_index=255)
    assert len(ds)==2145
    names=[];values={};max_error=0.;counts={'positive':0,'negative':0,'unknown':0,'trusted_pixels':0,'all_pixels':0}
    with torch.no_grad():
        for index in range(rank,len(ds),8):
            name,img,tags=ds[index];j=len(names)
            assert str(name)==oldcache['names'][j]
            x=F.interpolate(torch.as_tensor(img,device=device).float()[None],(448,448),mode='bilinear',align_corners=False)
            # Only current-step image labels are consumed. Old tags are never used.
            new_tags=torch.as_tensor(tags,device=device)[None,15:]
            x2=torch.cat([x,x.flip(-1)],0);box=torch.tensor([[0,448,0,448]])
            old_cls,old_aux,old_logits,_,_=ctx['teacher'](x2,step0=True)
            _=ctx['student'](x2,crops=[])
            feature=capture['feature'];pooled=capture['pooled'].clone();aux_pooled=capture['aux_pooled'].clone()
            # Reconstruct the reused single-image path exactly. Batch-two
            # convolution kernels can differ numerically from batch-one.
            _=ctx['student'](x,crops=[])
            single=capture['feature']
            error=float((single[0].cpu()-oldcache['feature'][j]).abs().max());max_error=max(max_error,error)
            assert error<1e-5,error
            best_logits=torch.cat([F.conv2d(feature,w) for w in heads],1)
            tv=torch.stack([old_logits[0],old_logits[1].flip(-1)],0)[None]
            rv=torch.stack([best_logits[0],best_logits[1].flip(-1)],0)[None]
            cams,_=ctx['multi_scale_cam2'](ctx['student'],x,scales=cfg['cam_scales'])
            presence=calibrate_presence(old_cls[None],old_aux[None],tv,rv,cams,new_tags,cfg['high_thre'])
            allowed=torch.cat([presence['positive'],new_tags.bool()],1)
            valid,_=ctx['cam_to_label'](cams,cls_label=allowed,img_box=box,ignore_mid=True,bkg_thre=cfg['bkg_thre'],high_thre=cfg['high_thre'],low_thre=cfg['low_thre'],ignore_index=255)
            par=ctx['refine_cams_with_bkg_v2'](ctx['par'],ctx['denormalize_img2'](x.clone()),cams=valid,cls_labels=allowed,high_thre=cfg['high_thre'],low_thre=cfg['low_thre'],ignore_index=255,img_box=box)
            fused=fuse_supervision(presence,tv,rv,par,valid,new_tags,cfg['high_thre'],cfg['low_thre'])
            size=feature.shape[-2:]
            items={'pooled':pooled,'aux_pooled':aux_pooled,'state':presence['state'][0],
                'labels':fused['labels'][0],'trusted_old':fused['old'][0],
                'cams':F.interpolate(valid,size,mode='bilinear',align_corners=False)[0],
                'par':F.interpolate(par[:,None].float(),size,mode='nearest')[0,0].long()}
            for key,value in items.items():values.setdefault(key,[]).append(value.cpu().clone())
            for key in ['positive','negative','unknown']:counts[key]+=int(presence[key].sum())
            counts['trusted_pixels']+=int((fused['labels']<21).sum());counts['all_pixels']+=fused['labels'].numel()
            names.append(str(name))
            if len(names)%64==0:print(json.dumps({'rank':rank,'images':len(names),'counts':counts}),flush=True)
    for h in handles:h.remove()
    path=safe_path(U/f'cache_rank{rank}.pth');assert not path.exists()
    torch.save({'names':names,**{k:torch.stack(v) for k,v in values.items()}},path)
    return {'readiness_coverage':counts,'rank':rank,'images':names,'cache':str(path),'cache_sha256':digest(path),
        'prior_cache_sha256':receipt['cache_sha256'],'maximum_reused_feature_error':max_error,
        'best_reference_sha256':digest(BEST),'ALD_source_sha256':digest(E/'evidence.py'),
        'semantics':'Frozen 2-view ALD evidence on 2145 training images; old image tags and pixel GT not consumed; prior decoder feature cache reused. No online CAM refresh during head-only updates.'}

base.OUTPUT=U/f'cache_receipt_rank{rank}.json';base.old_background_mode=cache
if __name__=='__main__':base.main()
