"""Reuse frozen evidence; derive one reliability veto without new inference."""
from pathlib import Path
import sys,os,json
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto');E=R/'experiments/ald_calibration_v4';U=R/'runs/ald_calibration_v4'
sys.path.insert(0,str(R/'experiments/kd_pixel_v2'))
from run_kd import environment
os.environ.update(environment('8card'))
from kd_runtime import safe_path,digest,atomic_json
import torch
from admission import retain_unsupported_old
torch.set_num_threads(1)
for rank in range(8):
    prior=json.loads(safe_path(R/'runs/ald_calibration_v3'/f'cache_receipt_rank{rank}.json').read_text())
    assert prior['passed'] and digest(safe_path(prior['cache']))==prior['cache_sha256']
    z=torch.load(prior['cache'],map_location='cpu',weights_only=True,mmap=True)
    first=json.loads(safe_path(R/'runs/ald_calibration_v1'/f'cache_receipt_rank{rank}.json').read_text())
    assert first['passed'] and first['cache_sha256']==prior['ALD_v1_cache_sha256']==digest(safe_path(first['cache']))
    state=torch.load(first['cache'],map_location='cpu',weights_only=True,mmap=True)
    fp=safe_path(R/'runs/pair_preserving_kd_v1'/f'cache_rank{rank}.pth');assert digest(fp)==first['prior_cache_sha256']
    features=torch.load(fp,map_location='cpu',weights_only=True,mmap=True)
    assert z['names']==state['names']==features['names']
    z['new_teacher_old']=retain_unsupported_old(z['new_teacher_old'],features['old_logits'].argmax(1),state['state'])
    target=safe_path(U/f'cache_rank{rank}.pth');assert not target.exists();torch.save(z,target)
    counts={key:int(z[key].sum()) for key in ['background','old','new_teacher_old','new_teacher_BG']}
    atomic_json(U/f'cache_receipt_rank{rank}.json',{**prior,'cache':str(target),'cache_sha256':digest(target),'readiness_coverage':counts,
        'upstream_cache_sha256':prior['cache_sha256'],'admission_sha256':digest(E/'admission.py'),
        'semantics':'No new inference; same refreshed ALD-v3 CAM/PAR proposals, veto new-class takeover if old teacher class has confirmed old presence in frozen ALD-v1 evidence. No GT.'})
    print(rank,counts,flush=True)
