from pathlib import Path
import sys,os,json,subprocess
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto');E=R/'experiments/ald_calibration_v7';U=R/'runs/ald_calibration_v7/formal'
S=R/'experiments/prototype_sep_v1/c_new_anchor/src'
sys.path.insert(0,str(R/'experiments/kd_pixel_v2'))
from run_kd import environment
env=environment('8card');os.environ.update(env)
sys.path.insert(0,str(S))
from kd_runtime import safe_path,atomic_json,digest,now
done=json.loads(safe_path(U/'training_complete.json').read_text());assert done['status']=='complete'
assert done['steps']==300 and digest(safe_path(done['checkpoint']))==done['checkpoint_sha256']
assert not (U/'evaluation_processes.json').exists()
workers=[]
for rank in range(8):
    cmd=[str(R/'.runtime/env/bin/python'),'-B',str(E/'evaluate.py'),'--run',str(U),'--rank',str(rank),'--shards','8','--once']
    with open(safe_path(U/f'evaluator{rank}.log'),'x') as log:
        p=subprocess.Popen(cmd,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    workers.append({'rank':rank,'pid':p.pid,'command':cmd})
atomic_json(U/'evaluation_processes.json',{'utc':now(),'workers':workers,'evaluator_sha256':digest(E/'evaluate.py')});print(json.dumps(workers))
