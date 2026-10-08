from pathlib import Path
import os,sys,json,subprocess,time,argparse
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto');E=R/'experiments/ald_calibration_v9';U=R/'runs/ald_calibration_v9';PY=R/'.runtime/env/bin/python'
sys.path.insert(0,str(R/'experiments/kd_pixel_v2'))
from run_kd import environment
env=environment('8card');os.environ.update(env)
from kd_runtime import safe_path,atomic_json,digest,now
p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');args=p.parse_args()
if not args.worker:
    assert not (U/'finish_process.json').exists()
    cmd=[str(PY),'-B',str(E/'finish.py'),'--worker']
    with open(safe_path(U/'finish.log'),'x') as log:c=subprocess.Popen(cmd,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    atomic_json(U/'finish_process.json',{'pid':c.pid,'command':cmd,'utc':now(),'evaluator_sha256':digest(E/'evaluate.py'),'analysis_sha256':digest(E/'analyze.py')});print(c.pid);sys.exit()
record=json.loads((U/'finish_process.json').read_text());assert digest(E/'evaluate.py')==record['evaluator_sha256'] and digest(E/'analyze.py')==record['analysis_sha256']
started=time.monotonic()
try:
    while True:
        status=json.loads((U/'formal/status.json').read_text());assert status['status']!='failed',status
        if status['status']=='complete':break
        proc=json.loads((U/'formal/process.json').read_text());assert Path(f"/proc/{proc['pid']}/cmdline").exists() and time.monotonic()-started<1800
        time.sleep(10)
    atomic_json(U/'finish_status.json',{'status':'classification_audit','utc':now()})
    run=safe_path(U/'audit');done=json.loads((U/'formal/training_complete.json').read_text());job=json.loads((U/'formal/eval_queue/step2_iter8600.json').read_text())
    job.update(checkpoint=done['checkpoint'],checkpoint_sha256=done['checkpoint_sha256']);atomic_json(run/'eval_queue/step2_iter8600.json',job)
    workers=[]
    for rank in range(8):
        cmd=[str(PY),'-B',str(E/'evaluate.py'),'--run',str(run),'--rank',str(rank),'--shards','8','--once']
        with open(safe_path(run/f'rank{rank}.log'),'x') as log:c=subprocess.Popen(cmd,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT)
        workers.append(c)
    codes=[c.wait() for c in workers];assert codes==[0]*8,codes
    sys.path.insert(0,str(E));from evaluate import merge
    merge(job,run/'evaluations/step2_iter8600',8)
    with open(safe_path(U/'analysis.log'),'x') as log:rc=subprocess.call([str(PY),'-B',str(E/'analyze.py')],cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT)
    assert rc==0,rc
    atomic_json(U/'finish_status.json',{'status':'complete','utc':now()})
except BaseException as ex:
    atomic_json(U/'finish_status.json',{'status':'failed','error':repr(ex),'utc':now()});raise
