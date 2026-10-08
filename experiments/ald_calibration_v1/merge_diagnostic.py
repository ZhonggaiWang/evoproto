from pathlib import Path
import sys,os,json
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto');E=R/'experiments/ald_calibration_v1';U=R/'runs/ald_calibration_v1'
sys.path.insert(0,str(R/'experiments/kd_pixel_v2'))
from run_kd import environment
os.environ.update(environment('8card'))
from kd_runtime import atomic_json,digest,safe_path
import numpy as np
parts=[json.loads(safe_path(U/f'diagnostic_rank{i}.json').read_text()) for i in range(8)]
names=[n for p in parts for n in p['images']];assert len(names)==len(set(names))==1449
assert all(p['passed'] and p['ALD_source_sha256']==digest(E/'evidence.py') and all(p['state_checks'].values()) for p in parts)
classification={}
for key in parts[0]['image_classification']:
    hist=np.sum([p['image_classification'][key] for p in parts],axis=0,dtype=np.int64);h=hist.sum(0)
    if key=='calibrated_negative':
        precision=h[0,1]/max(1,h[:,1].sum());recall=h[0,1]/max(1,h[0].sum())
    else:precision=h[1,1]/max(1,h[:,1].sum());recall=h[1,1]/max(1,h[1].sum())
    classification[key]={'histogram_GT_by_selected':h.tolist(),'precision':float(precision),'recall':float(recall),
        'selected_image_class_cases':int(h[:,1].sum()),'per_class_histograms':hist.tolist()}
pixels={}
for key in parts[0]['pixel_confusions']:
    h=np.sum([p['pixel_confusions'][key] for p in parts],axis=0,dtype=np.int64)
    valid=h[:21].sum();accepted=h[:21,:21].sum();diag=h.diagonal()[:21]
    pixels[key]={'histogram':h.tolist(),'accepted_valid_pixels':int(accepted),'coverage':float(accepted/valid),
        'precision':float(diag.sum()/max(1,accepted)),
        'old_label_precision':float(diag[1:16].sum()/max(1,h[:21,1:16].sum())),
        'new_label_precision':float(diag[16:21].sum()/max(1,h[:21,16:21].sum())),
        'BG_label_precision':float(h[0,0]/max(1,h[:21,0].sum()))}
counts={k:sum(p['readiness_coverage'][k] for p in parts) for k in parts[0]['readiness_coverage']}
report={'passed':True,'images':1449,'best_reference_sha256':parts[0]['best_reference_sha256'],
    'ALD_source_sha256':digest(E/'evidence.py'),'image_level':classification,'pixel_level':pixels,'counts':counts,
    'GT_role':'Diagnosis only; no GT or old image tags enter calibrate_presence, CAM allowance, PAR or fusion.',
    'limits':['Two-view fixed resize448 native-grid diagnosis, not final original-resolution evaluation',
        'Corroborating signals are correlated; agreement is not guaranteed correctness',
        'Three-state unknown is not a negative label; precision must be read together with coverage']}
atomic_json(U/'diagnostic.json',report)
print(json.dumps({'image_level':{k:{a:b for a,b in v.items() if a!='per_class_histograms'} for k,v in classification.items()},
    'pixel_level':{k:{a:b for a,b in v.items() if a!='histogram'} for k,v in pixels.items()},'counts':counts},indent=2))
