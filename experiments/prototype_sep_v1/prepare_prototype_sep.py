"""Create independent sources; never write through workspace links/mounts."""
from pathlib import Path
import sys, shutil, ast
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto')
BASE=R/'experiments/kd_parallel_v1/b_relational/src'
sys.path.insert(0,str(BASE))
from kd_runtime import safe_path, atomic_json, digest, now
E=safe_path(R/'experiments/prototype_sep_v1')
U=safe_path(R/'runs/prototype_sep_v1')
assert not E.exists() and not U.exists(), 'Refuse overwriting an existing study'
E.mkdir(); U.mkdir()
for arm in ['a_geometry']:
    S=safe_path(E/arm/'src')
    shutil.copytree(BASE,S,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for p in S.rglob('*'):
        safe_path(p)
        if p.suffix=='.py': ast.parse(p.read_text(),filename=str(p))
    for name in ['online_directed_confusion.py','directed_pair_selector.py']:
        source=R/'experiments/confusion_guided_v1/a_sep/src/model'/name
        shutil.copyfile(source,safe_path(S/'model'/name))
atomic_json(E/'origin.json',{
    'utc':now(),'optimized_kd_source':str(BASE),
    'optimized_kd_source_sha256':{str(p.relative_to(BASE)):digest(p) for p in BASE.rglob('*.py')},
    'scope':'Selective bounded prototype separation with old reference detached; existing conditional KD unchanged',
    'work_area':str(R.parents[1]),'host':'8card only',
    'baseline_retraining':False,'seed_search':False,
    'step0':str(R/'runs/fixed_baseline_v1/shared/10-5/step0/checkpoints/model_final.pth'),
    'step1_warmup':str(R/'runs/kd_pixel_v2/10-5/step1/checkpoints/model_iter_2000.pth')})
print(str(E))
