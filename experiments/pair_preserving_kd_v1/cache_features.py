"""Frozen training images/features/weak evidence only; never reads pixel annotations."""
from pathlib import Path
import sys,os,json
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto');E=R/'experiments/pair_preserving_kd_v1';U=R/'runs/pair_preserving_kd_v1'
rank=int(os.environ['CACHE_RANK'])
sys.path.insert(0,str(R/'experiments/hierarchy_kd_v2'))
import diagnose_new_transfer as base

def cache(ctx,protocol):
    torch,F,cfg,device=ctx['torch'],ctx['F'],ctx['cfg'],ctx['device']
    from kd_runtime import safe_path,digest
    ds=ctx['voc'].VOC12ClsDataset(root_dir=cfg['data_folder'],name_list_dir=cfg['list_folder'],
        split='train',stage='train',tasks='10-5',step=2,aug=False,num_classes=21,ignore_index=cfg['ignore_index'])
    assert len(ds)==2145
    captured={}
    def hook(module,args):captured['feature']=args[0].detach()
    handle=ctx['student'].decoder.conv8.register_forward_pre_hook(hook)
    values={key:[] for key in ['feature','old_logits','cams','par','tags']};names=[];max_error=0.
    with torch.no_grad():
        for index in range(rank,len(ds),8):
            name,img,tags=ds[index]
            x=F.interpolate(torch.as_tensor(img,device=device).float()[None],(448,448),mode='bilinear',align_corners=False)
            tags=torch.as_tensor(tags,device=device)[None];boxes=torch.tensor([[0,448,0,448]])
            old,cam,par,_=protocol.weak_signals(ctx,x,tags,boxes)
            _,reference,_,_,_,_=ctx['student'](x,crops=[])
            feature=captured.pop('feature');size=reference.shape[-2:]
            reconstructed=ctx['student'].decoder.conv8(feature)
            err=float((reference-reconstructed).abs().max());max_error=max(max_error,err)
            assert err<1e-5
            values['feature'].append(feature[0].cpu().clone())
            values['old_logits'].append(F.interpolate(old,size,mode='bilinear',align_corners=False)[0].cpu())
            values['cams'].append(F.interpolate(cam,size,mode='bilinear',align_corners=False)[0].cpu())
            values['par'].append(F.interpolate(par[:,None].float(),size,mode='nearest')[0,0].long().cpu())
            values['tags'].append(tags[0,15:].cpu());names.append(str(name))
            if len(names)%64==0:print(json.dumps({'rank':rank,'cached':len(names)}),flush=True)
    handle.remove();target=safe_path(U/f'cache_rank{rank}.pth');assert not target.exists()
    torch.save({'names':names,**{k:torch.stack(v) for k,v in values.items()}},target)
    return {'readiness_coverage':{'cached_images':len(names),'max_reconstruction_error':max_error},
        'rank':rank,'shards':8,'images':names,'cache':str(target),'cache_sha256':digest(target),
        'train_image_list_sha256':digest(Path(ds.name_list_dir)),
        'cache_semantics':'Every stage2 training image once, fixed resize448, float32 frozen decoder features, frozen weak signals; no pixel GT; no augmentation search.'}

base.OUTPUT=U/f'cache_receipt_rank{rank}.json';base.old_background_mode=cache
if __name__=='__main__':base.main()
