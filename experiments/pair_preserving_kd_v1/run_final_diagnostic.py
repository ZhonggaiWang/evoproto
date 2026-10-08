from pathlib import Path
import sys,os,json,subprocess
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto');E=R/'experiments/pair_preserving_kd_v1';U=R/'runs/pair_preserving_kd_v1'
sys.path.insert(0,str(R/'experiments/kd_pixel_v2'))
from run_kd import environment
env=environment('8card');os.environ.update(env)
from kd_runtime import safe_path,atomic_json,now,digest
U=safe_path(U);U.mkdir(parents=True,exist_ok=True)
assert not (U/'final_diagnostic_processes.json').exists()
workers=[]
for rank in range(8):
    command=[str(R/'.runtime/env/bin/python'),'-B',str(E/'diagnose_final_targets.py'),'--device',str(rank)]
    with open(safe_path(U/f'final_diagnostic_rank{rank}.log'),'x') as log:
        p=subprocess.Popen(command,cwd=R,env={**env,'DIAGNOSTIC_RANK':str(rank)},stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    workers.append({'rank':rank,'pid':p.pid,'command':command})
atomic_json(U/'final_diagnostic_processes.json',{'started_utc':now(),'workers':workers,
    'sources':{p.name:digest(p) for p in E.glob('*.py')}})
print(json.dumps(workers))
