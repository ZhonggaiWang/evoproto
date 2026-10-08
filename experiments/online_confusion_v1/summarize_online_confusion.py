from pathlib import Path
import sys,json
import numpy as np
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto');E=R/'experiments/online_confusion_v1';U=R/'runs/online_confusion_v1'
sys.path.insert(0,str(E));sys.path.insert(0,str(E/'src'))
from online_confusion_metrics import evaluate_matrices
from kd_runtime import atomic_json,now,digest
NAMES=['background','aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']

def pooled(checkpoint,split,shards):
    arrays=[];receipts=[]
    for rank in range(shards):
        prefix=U/f'{checkpoint}_{split}_rank{rank}'
        receipt=json.loads(prefix.with_suffix('.json').read_text());receipts.append(receipt)
        with np.load(prefix.with_suffix('.npz'),allow_pickle=False) as f:arrays.append({k:f[k] for k in f.files})
    assert len({r['checkpoint_sha256'] for r in receipts})==1
    assert len({r['teacher_sha256'] for r in receipts})==1
    combined={k:np.concatenate([a[k] for a in arrays]) for k in arrays[0]}
    order=np.argsort(combined['indices']);combined={k:v[order] for k,v in combined.items()}
    expected=np.atleast_1d(np.loadtxt(R/'datasets/voc/incremental_split'/f'{split}_10-5_step_3.txt',dtype=str))
    assert np.array_equal(combined['indices'],np.arange(len(expected)))
    assert np.array_equal(combined['names'],expected)
    totals={k:v.sum(0) for k,v in combined.items() if k not in ['indices','names']}
    return combined,totals,receipts

def normalized(x):return np.divide(x,x.sum(1)[:,None],out=np.zeros_like(x),where=x.sum(1)[:,None]>0)

def strongest_foreground(ep,gp,source):
    e=ep[source].copy();g=gp[source].copy();e[0]=g[0]=e[source]=g[source]=0
    if g.max()<=0:return None,None,None
    e_id=int(e.argmax()) if e.max()>0 else None;g_id=int(g.argmax())
    correct=bool(e_id is not None and g[e_id]==g.max())
    return e_id,g_id,correct

def bootstrap(arrays,samples=200):
    rng=np.random.default_rng(20261004);values=[];n=len(arrays['indices'])
    for _ in range(samples):
        inds=rng.integers(0,n,n)
        e=arrays['evaluated_estimated'][inds].sum(0);g=arrays['gt_all'][inds].sum(0)
        s=arrays['gt_selected'][inds].sum(0);a=arrays['pseudo_gt'][inds].sum(0)
        ep,gp,sp=normalized(e),normalized(g),normalized(s)
        eligible=(e.sum(1)>0)&(g.sum(1)>=100);eligible[0]=False
        eligible_s=(e.sum(1)>0)&(s.sum(1)>=100);eligible_s[0]=False
        tv=float(np.abs(ep-gp)[eligible].sum(1).mean()/2) if eligible.any() else None
        tvs=float(np.abs(ep-sp)[eligible_s].sum(1).mean()/2) if eligible_s.any() else None
        purity=float(np.diag(a)[1:].sum()/a[1:].sum()) if a[1:].sum()>0 else None
        vals=[strongest_foreground(ep,gp,i)[2] for i in range(1,21) if eligible[i]]
        vals=[v for v in vals if v is not None]
        values.append([tv,tvs,purity,float(np.mean(vals)) if vals else None])
    result={}
    for i,key in enumerate(['mean_foreground_TV_full','mean_foreground_TV_same_support','weighted_anchor_precision_foreground','strongest_foreground_target_accuracy']):
        v=[x[i] for x in values if x[i] is not None]
        result[key]={'percentile95_interval':np.percentile(v,[2.5,97.5]).tolist() if v else None,'bootstrap_samples':len(v)}
    return {'unit':'validation image','resamples':samples,'scope':'Uncertainty of frozen-checkpoint diagnostic sampling, not training-seed variability or a significance claim',**result}

def partitions(counts):
    p=normalized(counts);out={}
    for label,rows in [('old',range(1,16)),('new',range(16,21))]:
        vals=[]
        for i in rows:
            if counts[i].sum()==0:continue
            same_old=p[i,1:16].sum()-(p[i,i] if i<16 else 0)
            same_new=p[i,16:21].sum()-(p[i,i] if i>=16 else 0)
            vals.append([p[i,0],same_old,same_new,p[i,i]])
        out[label]={'supported_source_classes':len(vals),'macro_background':float(np.mean([v[0] for v in vals])) if vals else None,
            'macro_other_old':float(np.mean([v[1] for v in vals])) if vals else None,
            'macro_other_new':float(np.mean([v[2] for v in vals])) if vals else None,
            'macro_correct':float(np.mean([v[3] for v in vals])) if vals else None}
    return out

def evaluate_checkpoint(name):
    arrays,t,receipts=pooled(name,'val',4)
    metrics=evaluate_matrices(t['evaluated_estimated'],t['gt_all'],t['gt_selected'],t['pseudo_gt'])
    assert metrics['support_consistency']['same_support_total_mass_matches_estimate']
    assert metrics['support_consistency']['pseudo_gt_total_mass_matches_estimate']
    broad=evaluate_matrices(t['evaluated_broad'],t['gt_all'],t['gt_broad_selected'],t['broad_pseudo_gt'])
    ep,gp,sp=normalized(t['evaluated_estimated']),normalized(t['gt_all']),normalized(t['gt_selected'])
    bp=normalized(t['evaluated_broad'])
    cover=t['gt_covered_unweighted'].sum(1)/np.maximum(t['gt_all'].sum(1),1)
    rows=[]
    for i in range(21):
        e_id,g_id,correct=strongest_foreground(ep,gp,i)
        rows.append({'class_id':i,'class_name':NAMES[i],'gt_pixels':float(t['gt_all'][i].sum()),
            'accepted_gt_coverage':float(cover[i]),'anchor_precision':metrics['per_row'][i]['pseudo_anchor_precision'],
            'correct_anchor_weighted_mass_per_GT_pixel':float(t['pseudo_gt'][i,i]/max(t['gt_all'][i].sum(),1)),
            'anchor_image_observations':int((arrays['estimated'][:,i].sum(1)>0).sum()),
            'estimated_top_foreground_target':None if e_id is None else NAMES[e_id],
            'gt_top_foreground_target':None if g_id is None else NAMES[g_id],
            'top_foreground_target_correct':correct})
    bins=[]
    for i,(lo,hi) in enumerate([(0,.25),(.25,.5),(.5,.75),(.75,1)]):
        a=t['bin_pseudo_gt'][i]
        bins.append({'cam_gap_range':[lo,hi],'foreground_anchor_pixels':float(a[1:].sum()),
            'foreground_anchor_precision':float(np.diag(a)[1:].sum()/a[1:].sum()) if a[1:].sum() else None,
            'background_anchor_pixels':float(a[0].sum()),'background_anchor_precision':float(a[0,0]/a[0].sum()) if a[0].sum() else None})
    ranked=[]
    for i in range(1,21):
        for j in range(1,21):
            if i!=j:ranked.append({'source':NAMES[i],'target':NAMES[j],'estimate':float(ep[i,j]),'gt_full':float(gp[i,j]),'gt_same_support':float(sp[i,j]),'source_anchor_precision':rows[i]['anchor_precision'],'source_gt_coverage':rows[i]['accepted_gt_coverage']})
    ranked.sort(key=lambda x:x['estimate'],reverse=True)
    top_correct=[r['top_foreground_target_correct'] for r in rows[1:] if r['top_foreground_target_correct'] is not None]
    broad_correct=[strongest_foreground(bp,gp,i)[2] for i in range(1,21)]
    broad_correct=[v for v in broad_correct if v is not None]
    return {'images':len(arrays['indices']),'checkpoint':receipts[0]['checkpoint'],'checkpoint_sha256':receipts[0]['checkpoint_sha256'],
        'metrics':metrics,'broad_PAR_reference_metrics':broad,'perclass_coverage':rows,
        'strongest_foreground_target_accuracy':float(np.mean(top_correct)) if top_correct else None,
        'strongest_foreground_evaluable_classes':len(top_correct),'confidence_gap_bins':bins,
        'broad_strongest_foreground_target_accuracy':float(np.mean(broad_correct)) if broad_correct else None,
        'coverage_scope':'Accepted GT coverage counts any accepted anchor including wrong/background labels. Correct-anchor weighted mass per GT pixel is reported separately; neither is full-population accuracy.',
        'confidence_bin_scope':'Broad pre-mixing PAR anchors; foreground and background separated; gap is uncalibrated, includes rejected anchors',
        'estimated_top20_foreground_pairs':ranked[:20],'partitions':{'estimated':partitions(t['evaluated_estimated']),'gt_all':partitions(t['gt_all']),'gt_same_support':partitions(t['gt_selected'])},
        'bootstrap':bootstrap(arrays),'shards':[{k:r[k] for k in ['shard','images','seconds','batch','gt_usage']} for r in receipts],
        'pooled_matrix_policy':'SUM cumulative counts over all images; shard EMAs not merged',
        'GT_void_scoring_excluded_evidence_mass':float(t['estimated'].sum()-t['evaluated_estimated'].sum())}

def main():
    out={'utc':now(),'class_names':NAMES,'task':'VOC10-5 step2','scope':'Full validation, fixed448 frozen checkpoints; not benchmark mIoU, GT scoring only',
         'checkpoints':{name:evaluate_checkpoint(name) for name in ['warmup','final']},
         'protocol':json.loads((E/'protocol.json').read_text()),
         'integration_verification':json.loads((U/'integration_verification.json').read_text())}
    out['modeling_revision']={'decision':'Retain broad-PAR directed confusion as primary relation view and CAM-gated confusion as trusted evidence reference; track support/staleness for both.',
        'reason':'GT diagnosis shows evidence gating improves anchor purity but can hide actual difficult regions. Do not interpret gated rates as full-class confusion.',
        'GT_tuning':'No threshold grid or learned GT correction; diagnostic motivated preserving both already-computed views.',
        'implementation':'src/model/online_directed_confusion.py schema2; no training loss changes'}
    trainarrays,t,receipts=pooled('final','train',8)
    val=out['checkpoints']['final']['metrics']['matrices']
    transfer=evaluate_matrices(t['broad'],np.asarray(val['gt_all_counts']),np.asarray(val['gt_same_support_counts']))
    trusted_transfer=evaluate_matrices(t['estimated'],np.asarray(val['gt_all_counts']),np.asarray(val['gt_same_support_counts']))
    # Train vs val has different populations; never label this as same-support accuracy.
    transfer['summary'].pop('same_evidence_support',None)
    transfer['conditioning_scope']['full_population']='Training-stream pseudo anchor confusion versus heldout validation GT confusion. Includes domain and sample differences.'
    out['training_stream']={'images':len(trainarrays['indices']),'gt_read':False,'trusted_counts':t['estimated'].tolist(),
        'trusted_probabilities':normalized(t['estimated']).tolist(),'accepted_counts':t['accepted'].tolist(),'broad_counts':t['broad'].tolist(),
        'primary_relation_probabilities':normalized(t['broad']).tolist(),
        'heldout_transfer_full_population_summary':transfer['summary']['full_population'],
        'trusted_heldout_transfer_full_population_summary':trusted_transfer['summary']['full_population'],
        'caveat':'Frozen final weights and unaugmented448 images. This validates streaming estimation, not evolving-training EMA accuracy; train-vs-val comparison includes distribution shift.'}
    atomic_json(U/'gt_validation_report.json',out)
    summary={name:{'images':v['images'],'anchor_precision':v['metrics']['pseudo_anchor_precision'],
        'mean_class_gt_coverage':float(np.mean([r['accepted_gt_coverage'] for r in v['perclass_coverage'][1:]])),
        'gt_same':v['metrics']['summary']['same_evidence_support'],'gt_full':v['metrics']['summary']['full_population'],
        'strongest_foreground_target_accuracy':v['strongest_foreground_target_accuracy']} for name,v in out['checkpoints'].items()}
    atomic_json(U/'summary.json',{'utc':now(),'validation':summary,'training_images':len(trainarrays['indices'])})
    print(json.dumps({'report':str(U/'gt_validation_report.json'),'summary':summary,'training_images':len(trainarrays['indices'])},ensure_ascii=False))

if __name__=='__main__':main()
