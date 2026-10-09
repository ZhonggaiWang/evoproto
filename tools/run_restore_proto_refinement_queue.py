"""Finite follow-on pair: wait for the original study, smoke, then matched refinement."""
import json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CONTROL=ROOT/'runs/restore_proto_v1/control'
child=None


def read(path):return json.loads(path.read_text())


def status(**data):
    data.update(time=time.time(),pid=os.getpid())
    p=CONTROL/'refinement_queue_state.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(data,indent=2));t.replace(p)
    print(json.dumps(data),flush=True)


def alive(pidfile,needle):
    try:return needle in (Path('/proc')/pidfile.read_text().strip()/'cmdline').read_bytes()
    except FileNotFoundError:return False


def stop(*unused):
    if child is not None and child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
    raise KeyboardInterrupt


def run(mode):
    global child
    command=[sys.executable,'-B',str(ROOT/'experiments/restore_proto_ald_v1/run.py'),'--mode',mode,'--arms','full_control','full_ald','--attempt','v1']
    if mode=='formal':command+=['--decision','Full prototype restoration: 68.7125 square448, 69.8902 aspect672; fixed equal fusion 70.0644. Test V9-style reliability calibration versus equal-budget continuation, retaining explicit prototypes and confusion.']
    status(state='running',mode=mode,command=command)
    with (CONTROL/f'refinement_{mode}.log').open('a') as stream:
        child=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=child.wait()
        except BaseException:
            if child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=45)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            raise
    if code:raise RuntimeError(f'Refinement {mode} failed ({code}); inspect refinement_{mode}.log')


def main():
    import fcntl
    lock=(CONTROL/'refinement_queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    (CONTROL/'refinement_queue.pid').write_text(str(os.getpid()))
    assert read(CONTROL/'ald_activation_review.json')['fusion_evaluation_verified']
    status(state='waiting_for_original_formal_queue')
    while alive(CONTROL/'formal.pid',b'experiments/restore_proto_v1/run.py'):time.sleep(30)
    assert read(CONTROL/'runner_state.json')['status']=='completed','Original study did not finish successfully'
    for mode in ('smoke','formal'):run(mode)
    status(state='matched_pair_completed',next='Review results before selecting ALD or launching its module ablations; reservation remains held.')


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        status(state='failed',error=repr(exc));raise
