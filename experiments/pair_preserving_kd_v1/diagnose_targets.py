"""Eight disjoint frozen-model diagnostic shards; GT never enters target construction."""
from pathlib import Path
import sys,os,json
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto')
E=R/'experiments/pair_preserving_kd_v1'; U=R/'runs/pair_preserving_kd_v1'
rank=int(os.environ['DIAGNOSTIC_RANK'])
sys.path.insert(0,str(R/'experiments/hierarchy_kd_v2'))
import diagnose_new_transfer as base
from pair_targets import correction_targets,background_directions

def diagnose(ctx, protocol):
    torch,F,cfg,device=ctx['torch'],ctx['F'],ctx['cfg'],ctx['device']
    # The original observer is read from the same verified checkpoint, not updated.
    from model.online_directed_confusion import OnlineDirectedConfusion
    obs=OnlineDirectedConfusion(21,momentum=cfg['confusion_momentum'],
        high_threshold=cfg['high_thre'],low_threshold=cfg['low_thre'],stage=2).to(device)
    saved=torch.load(base.RUN/'10-5/step2/checkpoints/model_final.pth',map_location='cpu',weights_only=True,mmap=True)
    obs.load_state_dict(saved['online_confusion_state'],strict=True);del saved
    bg=background_directions(obs)
    ds=ctx['voc'].VOC12SegDataset(root_dir=cfg['data_folder'],name_list_dir=cfg['list_folder'],
        split=cfg['val_set'],stage='val',aug=False,ignore_index=cfg['ignore_index'],num_classes=21,tasks='10-5',step=2)
    ds.label_dir=cfg['val_label_dir'];assert len(ds)==1449
    hists={key:torch.zeros(21,21,dtype=torch.int64,device=device) for key in ['reference_full','reference_native','corrected_native']}
    stats={key:{'valid':0,'void':0,'correct':0,'incorrect':0,'true_old':0,'true_new':0,'true_BG':0,
                'weighted_correct':0.,'weighted_valid':0.} for key in ['new_pair','bg_pair','absent']}
    names=[];records=[]
    with torch.inference_mode():
        for index in range(rank,len(ds),8):
            name,img,gt,tags=ds[index];names.append(str(name))
            x=F.interpolate(torch.as_tensor(img,device=device).float()[None],(448,448),mode='bilinear',align_corners=False)
            tags=torch.as_tensor(tags,device=device)[None];boxes=torch.tensor([[0,448,0,448]])
            old,cam,par,_=protocol.weak_signals(ctx,x,tags,boxes)
            _,ref,_,_,_,_=ctx['student'](x,crops=[])
            z=correction_targets(ref,old,par,cam,boxes,tags[:,15:],ctx['targets'],bg,cfg['kd_temperature'],cfg['high_thre'])
            # Ground truth becomes visible only after fixed targets and masks exist.
            gt=torch.as_tensor(gt,device=device).long()[None]
            ng=F.interpolate(gt[:,None].float(),ref.shape[-2:],mode='nearest')[:,0].long()
            good=(ng>=0)&(ng<21)
            pred=z['reference_prediction'];target=z['target'].argmax(1)
            assert torch.allclose(z['target'].sum(1),torch.ones_like(z['weights']),atol=2e-6)
            for key,g,p in [('reference_native',ng,pred),('corrected_native',ng,target),
                            ('reference_full',gt,F.interpolate(ref,gt.shape[-2:],mode='bilinear',align_corners=False).argmax(1))]:
                v=(g>=0)&(g<21);hists[key]+=torch.bincount((g[v]*21+p[v]),minlength=441).reshape(21,21)
            for key in stats:
                mask=z[key];v=mask&good
                # Absent-class exclusion is valid for any GT other than rejected class;
                # its eventual replacement need not be correct, so record both separately.
                correct=(target==ng) if key!='absent' else (pred!=ng)
                st=stats[key];st['valid']+=int(v.sum());st['void']+=int((mask&~good).sum())
                st['correct']+=int((v&correct).sum());st['incorrect']+=int((v&~correct).sum())
                st['true_old']+=int((v&(ng>0)&(ng<16)).sum());st['true_new']+=int((v&(ng>=16)).sum());st['true_BG']+=int((v&(ng==0)).sum())
                st['weighted_correct']+=float((z['weights']*v*correct).double().sum())
                st['weighted_valid']+=float((z['weights']*v).double().sum())
            records.append({'name':str(name),'new_pair':int(z['new_pair'].sum()),'bg_pair':int(z['bg_pair'].sum()),'absent':int(z['absent'].sum())})
            if len(names)%32==0:print(json.dumps({'rank':rank,'images':len(names)}),flush=True)
    return {'readiness_coverage':stats,'images':names,'records':records,'rank':rank,'shards':8,
        'histograms':{k:v.cpu().tolist() for k,v in hists.items()},'background_targets':bg.nonzero().flatten().tolist(),
        'target_module_sha256':ctx['digest'](E/'pair_targets.py'),
        'counterfactual_limit':'Targets use existing weak image tags and CAM/PAR evidence; native-grid target substitution is only a diagnostic potential, not trained-model validation.'}

base.OUTPUT=U/f'readiness_rank{rank}.json'
base.old_background_mode=diagnose
if __name__=='__main__':base.main()
