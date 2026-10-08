from pathlib import Path
import sys,os,subprocess,json
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto');E=R/'experiments/hierarchy_kd_v3'
sys.path.insert(0,str(R/'experiments/kd_pixel_v2'))
from run_kd import environment
sys.path.insert(0,str(E/'src'))
from kd_runtime import safe_path
cmd=[str(R/'.runtime/env/bin/python'),'-B','-m','torch.distributed.run','--standalone','--nproc_per_node=8',str(E/'test_hierarchy.py')]
with open(safe_path(E/'preflight.log'),'x') as log:
    result=subprocess.run(cmd,cwd=R,env=environment('8card'),stdout=log,stderr=subprocess.STDOUT)
if result.returncode:
    print((E/'preflight.log').read_text()[-7000:]);raise SystemExit(result.returncode)
x=json.loads((E/'preflight.json').read_text());print(json.dumps({k:v for k,v in x.items() if k!='source_sha256'}))
