from pathlib import Path
import sys,os,json
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto');E=R/'experiments/ald_calibration_v2';U=R/'runs/ald_calibration_v2'
rank=int(os.environ['ALD_RANK'])
sys.path.insert(0,str(R/'experiments/hierarchy_kd_v2'))
import diagnose_new_transfer as base

def cache(ctx,protocol):
    torch,F,cfg,device=ctx['torch'],ctx['F'],ctx['cfg'],ctx['device']
    from kd_runtime import safe_path,digest
    bestpath=safe_path(R/'runs/pair_preserving_kd_v1/formal/10-5/step2/checkpoints/model_final.pth')
    assert digest(bestpath)=='3f43207d665ce6001aa149867435364e13baca5d47a0e2274dfb9187975dd28d'
    best=torch.load(bestpath,map_location='cpu',weights_only=True,mmap=True)
    for key,v in ctx['student'].state_dict().items():
        if not key.startswith('decoder.conv8.'):assert torch.equal(v.cpu(),best['model_state'][key])
    heads=[best['model_state'][f'decoder.conv8.{i}.weight'].to(device) for i in range(3)]
    capture={}
    def hook(module,args):capture['feature']=args[0].detach()
    handle=ctx['student'].decoder.conv8.register_forward_pre_hook(hook)
    prior=json.loads(safe_path(R/'runs/ald_calibration_v1'/f'cache_receipt_rank{rank}.json').read_text());assert prior['passed']
    ds=ctx['voc'].VOC12ClsDataset(root_dir=cfg['data_folder'],name_list_dir=cfg['list_folder'],split='train',stage='train',tasks='10-5',step=2,aug=False,num_classes=21,ignore_index=255)
    names=[];values=[];corrected_states=[]
    assert digest(safe_path(prior['cache']))==prior['cache_sha256']
    old=torch.load(prior['cache'],map_location='cpu',weights_only=True,mmap=True)
    with torch.inference_mode():
        for index in range(rank,len(ds),8):
            name,img,_=ds[index];assert str(name)==prior['images'][len(names)]
            x=F.interpolate(torch.as_tensor(img,device=device).float()[None],(448,448),mode='bilinear',align_corners=False)
            xx=torch.cat([x,x.flip(-1)],0)
            _,_,t,_,_=ctx['teacher'](xx,step0=True);_=ctx['student'](xx,crops=[])
            feature=capture.pop('feature');reference=torch.cat([F.conv2d(feature,w) for w in heads],1)
            tp=t.argmax(1);rp=reference.argmax(1)
            absent=torch.stack([~((tp==c).any()|(rp==c).any()) for c in range(1,16)])
            state=old['state'][len(names)].clone();assert not (absent.cpu()&(state==1)).any()
            # Spatial non-detection alone is not a negative image label.
            values.append(absent.cpu());corrected_states.append(state);names.append(str(name))
            if len(names)%64==0:print(json.dumps({'rank':rank,'images':len(names)}),flush=True)
    handle.remove();path=safe_path(U/f'cache_rank{rank}.pth');assert not path.exists()
    torch.save({'names':names,'spatial_absent':torch.stack(values),'state':torch.stack(corrected_states)},path)
    return {'readiness_coverage':{'images':len(names),'negative_states':int(torch.stack(values).sum())},'rank':rank,'images':names,'cache':str(path),'cache_sha256':digest(path),
        'ALD_v1_cache_sha256':prior['cache_sha256'],'best_reference_sha256':digest(bestpath),'semantics':'Both frozen segmentation models and both views predict no pixel of class c. This is an absence hypothesis, not a GT certainty. Only training images, no old tags or pixel labels.'}

base.OUTPUT=U/f'cache_receipt_rank{rank}.json';base.old_background_mode=cache
if __name__=='__main__':base.main()
