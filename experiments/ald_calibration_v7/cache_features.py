from pathlib import Path
import os,sys,json
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto');E=R/'experiments/ald_calibration_v7';U=R/'runs/ald_calibration_v7'
rank=int(os.environ['ALD_RANK']);sys.path.insert(0,str(R/'experiments/hierarchy_kd_v2'))
import diagnose_new_transfer as base

def cache(ctx,protocol):
    torch,F,cfg,device=ctx['torch'],ctx['F'],ctx['cfg'],ctx['device']
    from kd_runtime import safe_path,digest
    first=json.loads(safe_path(R/'runs/pair_preserving_kd_v1'/f'cache_receipt_rank{rank}.json').read_text());assert first['passed']
    assert digest(safe_path(first['cache']))==first['cache_sha256']
    old=torch.load(first['cache'],map_location='cpu',weights_only=True,mmap=True)
    ds=ctx['voc'].VOC12ClsDataset(root_dir=cfg['data_folder'],name_list_dir=cfg['list_folder'],split='train',stage='train',aug=False,ignore_index=255,num_classes=21,tasks='10-5',step=2)
    captured={}
    def hook(name):
        def get(module,args):captured[name]=args[0].detach()
        return get
    handles=[ctx['student'].decoder.conv7.register_forward_pre_hook(hook('input')),ctx['student'].decoder.conv8.register_forward_pre_hook(hook('output'))]
    names=[];values=[];maximum=0.
    with torch.no_grad():
        for i in range(rank,len(ds),8):
            name,img,_=ds[i];assert str(name)==first['images'][len(names)]
            x=F.interpolate(torch.as_tensor(img,device=device).float()[None],(448,448),mode='bilinear',align_corners=False)
            _=ctx['student'](x,crops=[])
            error=float((captured['output'][0].cpu()-old['feature'][len(names)]).abs().max());maximum=max(maximum,error);assert error<1e-5
            values.append(captured['input'][0].cpu().clone());names.append(str(name))
            if len(names)%64==0:print(json.dumps({'rank':rank,'images':len(names)}),flush=True)
    for h in handles:h.remove()
    p=safe_path(U/f'feature_rank{rank}.pth');assert not p.exists();torch.save({'names':names,'input':torch.stack(values)},p)
    return {'readiness_coverage':{'images':len(names),'decoder_reconstruction_error':maximum},'rank':rank,'images':names,'cache':str(p),'cache_sha256':digest(p),'prior_cache_sha256':first['cache_sha256'],
        'semantics':'Frozen encoder and decoder conv6 features for all2145 training images. No CAM recomputation, GT pixels or old image labels.'}

base.OUTPUT=U/f'feature_receipt_rank{rank}.json';base.old_background_mode=cache
if __name__=='__main__':base.main()
