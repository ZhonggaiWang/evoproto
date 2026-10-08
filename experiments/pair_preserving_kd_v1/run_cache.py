from pathlib import Path
import sys,os,json,subprocess
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto');E=R/'experiments/pair_preserving_kd_v1';U=R/'runs/pair_preserving_kd_v1'
sys.path.insert(0,str(R/'experiments/kd_pixel_v2'))
from run_kd import environment
env=environment('8card');os.environ.update(env)
from kd_runtime import safe_path,atomic_json,now,digest
readiness=json.loads(safe_path(U/'readiness.json').read_text())
assert readiness['passed'] and readiness['statistics']['new_pair']['precision']>.9
assert readiness['target_module_sha256']==digest(E/'pair_targets.py')
assert not (U/'cache_processes.json').exists()
workers=[]
for rank in range(8):
    command=[str(R/'.runtime/env/bin/python'),'-B',str(E/'cache_features.py'),'--device',str(rank)]
    with open(safe_path(U/f'cache_rank{rank}.log'),'x') as log:
        p=subprocess.Popen(command,cwd=R,env={**env,'CACHE_RANK':str(rank)},stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    workers.append({'rank':rank,'pid':p.pid,'command':command})
atomic_json(U/'cache_processes.json',{'started_utc':now(),'workers':workers,
    'sources':{p.name:digest(p) for p in E.glob('*.py')}})
print(json.dumps(workers))
