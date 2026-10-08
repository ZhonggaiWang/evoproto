from pathlib import Path
import sys,os,json
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto');E=R/'experiments/ald_calibration_v4';U=R/'runs/ald_calibration_v4'
sys.path.insert(0,str(R/'experiments/kd_pixel_v2'))
from run_kd import environment
os.environ.update(environment('8card'))
from kd_runtime import safe_path,atomic_json,digest
import numpy as np
records=[json.loads(safe_path(U/f'diagnostic_rank{i}.json').read_text()) for i in range(8)]
assert all(v['passed'] and v['proposal_sha256']==digest(E/'proposals.py') for v in records)
names=[n for v in records for n in v['images']];assert len(names)==len(set(names))==1449
results={}
for key in records[0]['corrections']:
    result={field:sum(v['corrections'][key][field] for v in records) for field in ['selected','correct','reference_correct','net_correct']}
    result['GT']=sum(np.array(v['corrections'][key]['GT']) for v in records).tolist()
    result['matrix']=sum(np.array(v['corrections'][key]['matrix']) for v in records).tolist()
    result['precision']=result['correct']/max(result['selected'],1);result['reference_precision']=result['reference_correct']/max(result['selected'],1)
    results[key]=result
atomic_json(U/'diagnostic.json',{'passed':True,'images':1449,'proposals':results,'proposal_sha256':digest(E/'proposals.py'),
    'scope':'Only changed native-grid predictions; no count inflation from already agreeing pixels; GT diagnostics only; not final mIoU.'})
print(json.dumps({k:{f:v[f] for f in ['selected','correct','reference_correct','net_correct','precision','reference_precision']} for k,v in results.items()},indent=2))
