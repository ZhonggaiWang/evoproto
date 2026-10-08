from pathlib import Path
import os,sys,json,subprocess,time,argparse
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto');E=R/'experiments/ald_calibration_v8';U=R/'runs/ald_calibration_v8';PY=R/'.runtime/env/bin/python'
sys.path.insert(0,str(R/'experiments/kd_pixel_v2'))
from run_kd import environment
env=environment('8card');os.environ.update(env)
from kd_runtime import safe_path,atomic_json,digest,now
p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');args=p.parse_args()
if not args.worker:
    assert not (U/'finish_process.json').exists()
    cmd=[str(PY),'-B',str(E/'finish.py'),'--worker']
    with open(safe_path(U/'finish.log'),'x') as log:c=subprocess.Popen(cmd,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    atomic_json(U/'finish_process.json',{'pid':c.pid,'command':cmd,'utc':now(),'diagnostic_sha256':digest(E/'diagnose.py'),'analysis_sha256':digest(E/'analyze.py')});print(c.pid);sys.exit()
record=json.loads((U/'finish_process.json').read_text())
assert digest(E/'diagnose.py')==record['diagnostic_sha256'] and digest(E/'analyze.py')==record['analysis_sha256']
started=time.monotonic()
try:
    while True:
        status=json.loads((U/'formal/status.json').read_text());assert status['status']!='failed',status
        if status['status']=='complete':break
        assert time.monotonic()-started<3600,'Formal job timeout'
        proc=json.loads((U/'formal/process.json').read_text());assert Path(f"/proc/{proc['pid']}/cmdline").exists(),'Formal coordinator exited early'
        time.sleep(10)
    atomic_json(U/'finish_status.json',{'status':'diagnosing','utc':now()})
    workers=[]
    for rank in range(8):
        cmd=[str(PY),'-B',str(E/'diagnose.py'),'--rank',str(rank)]
        with open(safe_path(U/f'diagnostic_rank{rank}.log'),'x') as log:c=subprocess.Popen(cmd,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT)
        workers.append(c)
    codes=[c.wait() for c in workers];assert codes==[0]*8,codes
    with open(safe_path(U/'analysis.log'),'x') as log:rc=subprocess.call([str(PY),'-B',str(E/'analyze.py')],cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT)
    assert rc==0,rc
    atomic_json(U/'finish_status.json',{'status':'complete','utc':now()})
except BaseException as ex:
    atomic_json(U/'finish_status.json',{'status':'failed','error':repr(ex),'utc':now()});raise
