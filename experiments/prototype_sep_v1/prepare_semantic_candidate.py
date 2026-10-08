"""Prepare a separate conditional source copy; never edit the running candidate."""
from pathlib import Path
import sys,json,shutil,ast,subprocess
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto')
E=R/'experiments/prototype_sep_v1';A=E/'a_geometry/src';B=E/'b_semantic/src'
sys.path.insert(0,str(A))
from kd_runtime import safe_path,atomic_json,digest,now
study=json.loads((R/'runs/prototype_sep_v1/formal/a_geometry/study.json').read_text())
current={str(p.relative_to(A)):digest(p) for p in A.rglob('*.py') if '__pycache__' not in p.parts}
assert current==study['source_sha256'],'Primary source must remain frozen'
safe_path(B);assert not B.exists(),'Refuse duplicate candidate preparation'
for p in A.rglob('*'):safe_path(p)
shutil.copytree(A,B,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
for p in B.rglob('*'):safe_path(p)
shutil.copyfile(safe_path(E/'semantic_protected_sep.py'),safe_path(B/'model/semantic_protected_sep.py'))
sys.path.insert(0,str(E))
subprocess.run([sys.executable,'-B',str(E/'patch_semantic_candidate.py'),'--src',str(B)],cwd=E,check=True)
for p in B.rglob('*.py'):ast.parse(p.read_text(),filename=str(p))
assert digest(B/'model/pixel_kd.py')==digest(A/'model/pixel_kd.py')
atomic_json(E/'semantic_origin.json',{'created_utc':now(),'parent_source':str(A),
    'parent_source_sha256':current,'conditional_source':str(B),
    'conditional_source_sha256':{str(p.relative_to(B)):digest(p) for p in B.rglob('*.py')},
    'changed_files':[str(p.relative_to(B)) for p in B.rglob('*.py') if str(p.relative_to(B)) not in current or digest(p)!=current[str(p.relative_to(B))]],
    'KD_byte_identical':True,'primary_source_unchanged':True,'scope':'Prepared only; no training launch'})
print(json.dumps({'prepared':str(B),'KD_byte_identical':True}))
