"""Paired image bootstrap for completed arms; does not estimate training-seed variance."""
import argparse,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
FORMAL=ROOT/'runs/restore_proto_v1/formal'


def miou(hist):
    h=np.asarray(hist,dtype=np.float64)
    intersection=np.diagonal(h,axis1=-2,axis2=-1)
    union=h.sum(-2)+h.sum(-1)-intersection
    return np.nanmean(np.divide(intersection*100,union,out=np.full_like(intersection,np.nan),where=union>0),axis=-1)


def paired_bootstrap(reference,candidate,samples=1000,seed=20261009):
    assert reference.shape==candidate.shape and reference.ndim==3
    n=len(reference);rng=np.random.default_rng(seed);deltas=[]
    for _ in range(samples):
        indices=rng.integers(0,n,n)
        deltas.append(float(miou(candidate[indices].sum(0))-miou(reference[indices].sum(0))))
    observed=float(miou(candidate.sum(0))-miou(reference.sum(0)))
    return {'candidate_minus_reference_pp':observed,'paired_image_bootstrap_95_percentile_interval':np.quantile(deltas,[.025,.975]).tolist(),'bootstrap_samples':samples,'seed':seed,'images':n}


def read(path):return json.loads(path.read_text())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--samples',type=int,default=1000);parser.add_argument('--self-test',action='store_true');args=parser.parse_args()
    if args.self_test:
        good=np.repeat(np.eye(2,dtype=np.int64)[None]*100,4,axis=0);bad=good[:,:,::-1]
        assert miou(good.sum(0))==100 and miou(bad.sum(0))==0
        assert paired_bootstrap(good,good,30)['paired_image_bootstrap_95_percentile_interval']==[0.,0.]
        assert paired_bootstrap(bad,good,30)['paired_image_bootstrap_95_percentile_interval']==[100.,100.]
        print('PASS identical predictions give zero delta; uniformly perfect versus reversed predictions give exactly 100 points');return
    assert args.samples>=100
    arms={};waiting=[]
    for variant in ('full','without_confusion','without_proto'):
        directory=FORMAL/variant/'10-5/step2';path=directory/'fusion_evaluation.json';verification=directory/'fusion_verification.json'
        if not path.exists() or not verification.exists():waiting.append(variant);continue
        data=read(path);verify=read(verification);receipt=read(directory/'final_receipt.json');official=read(directory/'evaluation.json')
        assert data['checkpoint_sha256']==verify['checkpoint_sha256']==receipt['sha256']==official['checkpoint_sha256']
        assert official['histogram_label_dtype']=='int64' and verify['per_image_histograms_verified']
        assert not data['image_tags_used']
        with np.load(path.with_suffix('.npz'),allow_pickle=False) as arrays:
            assert len(arrays['names'])==len(set(arrays['names']))==1449
            alpha_index=list(arrays['alphas']).index(0.)
            heads={mode:arrays[mode][:,alpha_index].copy() for mode in ('square448','aspect672')}
            for mode,hist in heads.items():assert np.array_equal(hist.sum(0),data['results'][mode]['0.0']['histogram'])
            arms[variant]={'names':arrays['names'].copy(),'histograms':heads,'checkpoint_sha256':data['checkpoint_sha256']}
    comparisons={}
    if 'full' in arms:
        full=arms['full']
        for ablation in ('without_confusion','without_proto'):
            if ablation not in arms:continue
            base=arms[ablation];assert np.array_equal(full['names'],base['names'])
            comparisons['full_minus_'+ablation]={mode:paired_bootstrap(base['histograms'][mode],full['histograms'][mode],args.samples) for mode in ('square448','aspect672')}
    result={'status':'complete' if not waiting else 'waiting','waiting_for':waiting,'primary_head':'main; alpha=0','comparisons':comparisons,'checkpoint_sha256':{k:v['checkpoint_sha256'] for k,v in arms.items()},'limitations':'Single-seed, adaptively inspected validation set. Bootstrap resamples whole validation images, not pixels; it describes validation-sample uncertainty only, not training-seed variance or an independent test result. Causal claims require the matched protocol and should not be inferred from an interval alone.'}
    output=ROOT/'runs/restore_proto_v1/paired_comparisons.json';temp=output.with_suffix('.tmp');temp.write_text(json.dumps(result,indent=2));temp.replace(output)
    print(json.dumps(result))

if __name__=='__main__':main()
