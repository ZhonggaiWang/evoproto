"""Run meaningful CPU selector tests and publish a source-bound receipt."""
from pathlib import Path
import json
import os
import re
import subprocess
import sys

R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto')
E=R/'experiments/prototype_sep_v1'
S=E/'c_new_anchor/src'
sys.path.insert(0,str(S))
from kd_runtime import safe_path, atomic_json, digest, now
sys.path.insert(0,str(R/'experiments/kd_pixel_v2'))
from run_kd import environment
origin=json.loads(safe_path(E/'new_anchor_origin.json').read_text())
code={str(p.relative_to(S)):digest(safe_path(p)) for p in sorted(S.rglob('*.py')) if '__pycache__' not in p.parts}
assert code==origin['candidate_source_sha256']
env=environment('8card')
env['CUDA_VISIBLE_DEVICES']=''
env['PYTHONPATH']=str(S)
test=safe_path(E/'test_new_anchor_selector.py')
result=subprocess.run([sys.executable,'-B',str(test),'-v'],cwd=R,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
log=safe_path(E/'new_anchor_selector_tests.log')
assert not log.exists(), 'Refuse overwriting test receipt'
log.write_text(result.stdout)
count=int(re.search(r'Ran (\d+) tests',result.stdout).group(1))
passed=result.returncode==0 and count==10 and re.search(r'\nOK\s*$',result.stdout) is not None
assert {str(p.relative_to(S)):digest(safe_path(p)) for p in sorted(S.rglob('*.py')) if '__pycache__' not in p.parts}==code
atomic_json(E/'new_anchor_selector_tests.json',{'created_utc':now(),'passed':passed,
    'returncode':result.returncode,'count':count,'source':str(S),'source_sha256':code,
    'module_sha256':code['model/geometry_pair_selector.py'],'test_source':str(test),
    'test_source_sha256':digest(test),'log':str(log),'log_sha256':digest(log),
    'cpu_only':True,'scope':'New selector behavior only; B loss and DDP source are unchanged and retain their existing tests.'})
print(result.stdout)
assert passed
