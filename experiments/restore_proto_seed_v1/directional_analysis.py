"""Read-only final segmentation error analysis from saved per-image histograms."""
import json, hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'runs/restore_proto_seed_v1'
NAMES=['background','aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']
def summarize(h, boundary):
    assert h.shape==(21,21) and (h>=0).all()
    gt=h.sum(1); new_gt=int(gt[boundary:].sum()); old_gt=int(gt[1:boundary].sum())
    new_as_old=int(h[boundary:,1:boundary].sum());old_as_new=int(h[1:boundary,boundary:].sum())
    correct_new=int(h.diagonal()[boundary:].sum()); wrong_new=int(h[boundary:,boundary:].sum())-correct_new; new_bg=int(h[boundary:,0].sum())
    assert correct_new+wrong_new+new_bg+new_as_old==new_gt
    return dict(correct_new_pixels=correct_new,correct_new_rate=correct_new/max(new_gt,1),new_predicted_other_new_pixels=wrong_new,new_predicted_other_new_rate=wrong_new/max(new_gt,1),new_predicted_background_rate=new_bg/max(new_gt,1),old_foreground_ids=list(range(1,boundary)),new_ids=list(range(boundary,21)),new_gt_pixels=new_gt,old_gt_pixels=old_gt,new_predicted_old_pixels=new_as_old,new_predicted_old_rate=new_as_old/max(new_gt,1),old_predicted_new_pixels=old_as_new,old_predicted_new_rate=old_as_new/max(old_gt,1),new_predicted_background_pixels=int(h[boundary:,0].sum()),per_new_class={NAMES[c]:dict(gt_pixels=int(gt[c]),predicted_old_pixels=int(h[c,1:boundary].sum()),predicted_old_rate=float(h[c,1:boundary].sum()/max(gt[c],1)),predicted_background_pixels=int(h[c,0])) for c in range(boundary,21)})
def main():
    base=ROOT/'runs/restore_proto_v1/formal/full/10-5/step2'
    sources={'baseline':(base,ROOT/'runs/restore_proto_olc_v1/reference_full_gpu')}
    sources.update({a:(STUDY/'formal'/a/'10-5/step2',STUDY/'formal'/a/'10-5/step2') for a in ['correct','ignore','no_graph']})
    out=dict(status='waiting',matrix_orientation='rows=ground truth, columns=prediction; confirmed from evaluator bincount(gt*21+prediction)',primary_partition='Step2: old foreground1..15, new16..20',auxiliary_partition='Initial10 vs subsequently introduced10; descriptive only',prediction_uses_image_tags=False,arms={},missing=[])
    ref_names=None;ref_gt=None
    for arm,(model,metrics) in sources.items():
        jf=metrics/'fusion_evaluation.json';npf=metrics/'fusion_evaluation.npz'
        if not jf.exists() or not npf.exists():out['missing'].append(arm);continue
        j=json.loads(jf.read_text());receipt=json.loads((model/'final_receipt.json').read_text())
        assert j['checkpoint_sha256']==receipt['sha256'] and j['images']==1449 and j['stage']==2 and not j['image_tags_used'] and j['device']=='cuda'
        row=dict(checkpoint_sha256=receipt['sha256'],source=str(jf.relative_to(ROOT)),histogram_file_sha256=hashlib.sha256(npf.read_bytes()).hexdigest(),protocols={})
        with np.load(npf,allow_pickle=False) as data:
            names=data['names'];assert len(names)==len(set(names))==1449
            if ref_names is None:ref_names=names.copy()
            assert np.array_equal(names,ref_names)
            alphas=list(data['alphas'])
            for mode in ['square448','aspect672']:
                assert data[mode].shape==(1449,len(alphas),21,21)
                row['protocols'][mode]={}
                for alpha in [0.,.5]:
                    h=data[mode][:,alphas.index(alpha)].sum(0);v=j['results'][mode][str(alpha)]
                    assert np.array_equal(h,np.asarray(v['histogram']))
                    if ref_gt is None:ref_gt=h.sum(1).copy()
                    assert np.array_equal(h.sum(1),ref_gt)
                    den=h.sum(0)+h.sum(1)-h.diagonal();miou=float(np.mean(100*h.diagonal()/den));assert abs(miou-v['miou'])<1e-9
                    row['protocols'][mode][str(alpha)]=dict(miou=miou,step2_partition=summarize(h,16),initial_partition=summarize(h,11))
        out['arms'][arm]=row
    if not out['missing']:out['status']='complete'
    dest=STUDY/'control/directional_segmentation.json';dest.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'status':out['status'],'available':list(out['arms']),'missing':out['missing'],'output':str(dest)}))
if __name__=='__main__':main()
