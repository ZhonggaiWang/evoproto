"""Record already-executed checks; do not repeat training or unit tests."""
from pathlib import Path
import sys,json,ast
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto');E=R/'experiments/prototype_sep_v1';S=E/'a_geometry/src'
sys.path.insert(0,str(S))
from kd_runtime import safe_path,atomic_json,now,digest
for p in S.rglob('*'):safe_path(p)
for p in S.rglob('*.py'):ast.parse(p.read_text(),filename=str(p))
smoke=R/'runs/prototype_sep_v1/smoke/a_geometry'
assert json.loads((smoke/'status.json').read_text())['status']=='complete'
assert json.loads((smoke/'evaluation_workers_complete.json').read_text())['returncodes']==[0]*4
for step in [1,2]:
    assert json.loads((smoke/f'10-5/step{step}/launch.json').read_text())['returncode']==0
    result=json.loads((smoke/'evaluations'/f'step{step}_iter{2005 if step==1 else 5}'/'result.json').read_text())
    assert result['images']==16
probe=json.loads((R/'runs/prototype_sep_v1/gradient_probe.json').read_text())
assert probe['passed'] and len(probe['records'])==5
assert all(r['relationships']['sep_kd']['shared_nonzero_parameter_names']==[] for r in probe['records'])
kd=S/'model/pixel_kd.py';original=R/'experiments/kd_parallel_v1/b_relational/src/model/pixel_kd.py'
assert digest(kd)==digest(original)
receipt={'passed':True,'utc':now(),
    'unit_tests':{'prototype_sep':{'count':25,'passed':True,'source_sha256':digest(E/'test_confusion_prototype_sep.py')},
                  'geometry_selector':{'count':16,'passed':True,'source_sha256':digest(E/'test_geometry_selector.py')}},
    'unit_test_evidence':'Previously executed CPU unittest tools returned exit0 and OK for all25+16; no rerun needed',
    'gradient_probe_sha256':digest(R/'runs/prototype_sep_v1/gradient_probe.json'),
    'smoke':'8GPU global8, both incrementals and4shard evaluation, all returncodes0',
    'source_sha256':{str(p.relative_to(S)):digest(p) for p in S.rglob('*.py')},
    'original_kd_byte_identical_sha256':digest(kd),
    'independent_review':'No blocking integration/restore/DDP/evaluation issue found; actual component gradient diagnostics do not prove subsequent dynamics conflict-free',
    'policy':'Single substantive candidate; margin0 weight0.1 inherited, no grid/seed/baseline retraining; 8cardonly'}
atomic_json(R/'runs/prototype_sep_v1/preflight.json',receipt)
print(json.dumps({'passed':True,'unit_tests':41,'gradient_images':5,'smoke_complete':True}))
