from pathlib import Path
import sys,json,os
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto');E=R/'experiments/pair_preserving_kd_v1';U=R/'runs/pair_preserving_kd_v1'
sys.path.insert(0,str(R/'experiments/kd_pixel_v2'))
from run_kd import environment
os.environ.update(environment('8card'))
from kd_runtime import safe_path,atomic_json,digest
import numpy as np
parts=[json.loads(safe_path(U/f'readiness_rank{i}.json').read_text()) for i in range(8)]
names=[x for p in parts for x in p['images']];assert len(names)==len(set(names))==1449
assert all(p['passed'] and p['target_module_sha256']==digest(E/'pair_targets.py') for p in parts)
assert len({p['student_sha256'] for p in parts})==len({p['teacher_sha256'] for p in parts})==1
h={k:np.sum([p['histograms'][k] for p in parts],axis=0,dtype=np.int64) for k in parts[0]['histograms']}
ref=json.loads(safe_path(R/'runs/prototype_sep_v1/formal/c_new_anchor/evaluations/step2_iter8000/result.json').read_text())
assert np.array_equal(h['reference_full'],ref['histogram']), 'Reproduced frozen reference histogram mismatch'
def metrics(hist):
    den=hist.sum(0)+hist.sum(1)-hist.diagonal();iou=np.divide(hist.diagonal(),den,out=np.zeros(21),where=den>0)*100
    return {'all':float(iou.mean()),'old':float(iou[1:16].mean()),'new':float(iou[16:].mean()),'class_iou':iou.tolist(),
        'BG_to_new':int(hist[0,16:].sum()),'new_to_BG':int(hist[16:,0].sum()),'old_to_new':int(hist[1:16,16:].sum()),'new_to_old':int(hist[16:,1:16].sum())}
stats={}
for key in parts[0]['readiness_coverage']:
    st={k:sum(p['readiness_coverage'][key][k] for p in parts) for k in parts[0]['readiness_coverage'][key]}
    st['precision']=st['correct']/st['valid'] if st['valid'] else None
    st['weighted_precision']=st['weighted_correct']/st['weighted_valid'] if st['weighted_valid'] else None
    stats[key]=st
result={'passed':True,'images':1449,'no_duplicates':True,'frozen_reference_full_histogram_exact':True,
    'student_sha256':parts[0]['student_sha256'],'teacher_sha256':parts[0]['teacher_sha256'],
    'target_module_sha256':digest(E/'pair_targets.py'),'statistics':stats,'metrics':{k:metrics(v) for k,v in h.items()},
    'histograms':{k:v.tolist() for k,v in h.items()},'saved_pair_targets':parts[0]['saved_selector']['targets'],
    'background_targets':parts[0]['background_targets'],'state_unchanged_all_shards':all(all(p['state_checks'].values()) for p in parts),
    'limit':'Corrected native predictions are weak-label-assisted counterfactual targets, NOT a trained-model score. GT diagnoses fixed targets only.'}
atomic_json(U/'readiness.json',result)
print(json.dumps({k:v for k,v in result.items() if k!='histograms'},indent=2))
