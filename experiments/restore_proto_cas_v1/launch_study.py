"""Bounded, one-shot experiment pipeline with preflight and cleanup on failure."""
import sys,json,time,subprocess,os,signal,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import experiments.restore_proto_v1.run as common
C=ROOT/'runs/restore_proto_cas_v1/control';common.CONTROL=C
E=dict(os.environ,CUDA_VISIBLE_DEVICES=common.UUIDS,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',NCCL_P2P_DISABLE='1',NCCL_IB_DISABLE='1')
child=None

def stop(*unused):
    if child is not None and child.poll() is None:
        os.killpg(child.pid,signal.SIGTERM)
        try:child.wait(timeout=30)
        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    raise KeyboardInterrupt

def run(cmd,log):
    global child
    common.reserve('training')
    with log.open('w') as f:
        child=subprocess.Popen(cmd,cwd=ROOT,env=E,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        code=child.wait()
    common.reserve('reserve')
    if code:raise RuntimeError(f'Command failed ({code}): {log}')

def state(status,**kw):common.write(C/'workflow_state.json',dict(status=status,pid=os.getpid(),time=time.time(),**kw))
def main():
    global child
    p=argparse.ArgumentParser();p.add_argument('--smoke-pid',type=int,required=True);a=p.parse_args()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    state('waiting_for_revised_smoke')
    while Path(f'/proc/{a.smoke_pid}').exists():time.sleep(2)
    smoke=json.loads((C/'runner_state.json').read_text());assert smoke['status']=='completed' and smoke['smoke']
    assert (C/'revised_cpu_tests.log').read_text().startswith('PASS:')
    state('revised_selection_audit')
    for stage in [1,2]:
        run([sys.executable,'-B',str(ROOT/'experiments/restore_proto_cas_v1/audit.py'),'--stage',str(stage),'--directory',str(ROOT/f'runs/restore_proto_v1/formal/full/10-5/step{stage}'),'--output',str(C/f'revised_baseline_step{stage}_audit.json')],C/f'revised_baseline_step{stage}_audit.log')
    state('active_gradient_probe')
    cmd=json.loads((ROOT/'runs/restore_proto_cas_v1/smoke/confusion_pair/10-5/step1/command.json').read_text())
    for key,value in [('--work_dir',str(ROOT/'runs/restore_proto_cas_v1/revised_probe/confusion_pair')),('--max_iters','80'),('--train_limit','128'),('--eval_iters','80'),('--log_iters','10')]:
        for i,x in enumerate(cmd):
            if x==key:cmd[i+1]=value
    run(cmd,C/'revised_active_probe.log')
    probe=ROOT/'runs/restore_proto_cas_v1/revised_probe/confusion_pair/10-5/step1'
    records=list(map(json.loads,(probe/'cas_metrics.jsonl').read_text().splitlines()))
    assert sum(r['selected_pixels'] for r in records)>0,'Revised selection was never active in probe'
    assert json.loads((probe/'metrics.jsonl').read_text().splitlines()[-1])['iteration']==80
    common.write(C/'preflight_passed.json',dict(status='passed',cpu=True,smoke_arms=['off','ignore','local_pair','confusion_pair'],active_probe_updates=80,selected_probe_pixels_sampled=sum(r['selected_pixels'] for r in records),time=time.time()))
    state('formal_training')
    cmd=[sys.executable,'-B',str(ROOT/'experiments/restore_proto_cas_v1/run.py'),'--arms','confusion_pair','ignore','local_pair']
    # Formal runner owns reservation transitions and final release.
    with (C/'formal_runner.log').open('w') as f:
        child=subprocess.Popen(cmd,cwd=ROOT,env=E,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        common.write(C/'formal_launch.json',dict(pid=child.pid,command=cmd,time=time.time()))
        code=child.wait()
    if code:raise RuntimeError(f'Formal runner failed: {code}')
    assert json.loads((ROOT/'runs/restore_proto_cas_v1/comparison.json').read_text())['status']=='complete'
    state('completed')

if __name__=='__main__':
    try:main()
    except BaseException as e:state('failed',error=repr(e));raise
    finally:common.write(C/'reservation_request.json',{'mode':'stop'})
