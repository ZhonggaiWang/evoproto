"""Audit online prediction snapshots and evaluate the completed same-budget chain."""
import hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
CONTROL=ROOT/'runs/restore_proto_olc_v1/control'
FORMAL=ROOT/'runs/restore_proto_olc_v1/formal/olc/10-5'
from experiments.restore_proto_v1.run import reserve,UUIDS
child=None
reserved_for_evaluation=False

def write(path,value):
    assert path.resolve().is_relative_to(ROOT)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)

def stop(*unused):
    global child
    if child is not None and child.poll() is None:
        os.killpg(child.pid,signal.SIGTERM)
        try:child.wait(timeout=20)
        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    raise KeyboardInterrupt

def main():
    global child,reserved_for_evaluation
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='8')
    while True:
        state=json.loads((CONTROL/'runner_state.json').read_text())
        if state['status']=='failed':raise RuntimeError('Formal training failed: '+str(state))
        for stage in [1,2]:
            d=FORMAL/f'step{stage}';memory=d/'olc_memory_latest.pth'
            if memory.exists():
                try: iteration=torch.load(memory,map_location='cpu',weights_only=True)['updates']
                except (EOFError,RuntimeError,OSError):continue
                out=d/f'olc_label_audit_{iteration}.json'
                if not out.exists():
                    command=[sys.executable,'-B',str(ROOT/'tools/audit_olc_memory.py'),'--memory',str(memory),'--annotation-audit',str(ROOT/f'runs/online_ald_label_audit_v1/step{stage}.npz'),'--output',str(out)]
                    with (d/'olc_label_audit.log').open('a') as f:subprocess.run(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
                    print('LABEL_AUDIT',stage,iteration,flush=True)
            receipt=d/'final_receipt.json';audit=d/'olc_checkpoint_audit.json'
            if receipt.exists() and not audit.exists():
                r=json.loads(receipt.read_text());p=Path(r['path']);h=hashlib.sha256(p.read_bytes()).hexdigest()
                assert h==r['sha256']
                checkpoint=torch.load(p,map_location='cpu',weights_only=True)
                assert checkpoint['iteration']==8000 and checkpoint['olc']
                assert all(torch.isfinite(v).all() for v in checkpoint['model_state'].values())
                prototypes={n:list(v.shape) for n,v in checkpoint['model_state'].items() if 'class_prototypes' in n}
                assert len(prototypes)==stage+1
                write(audit,dict(checkpoint_sha256=h,iteration=8000,finite=True,olc=True,prototypes=prototypes,predecessor=json.loads((d/'predecessor.json').read_text())))
                del checkpoint
                print('CHECKPOINT_AUDIT',stage,flush=True)
        if state['status']=='completed':break
        time.sleep(30)
    # Wait for the formal runner's final reservation restoration to finish.
    runner=json.loads((CONTROL/'formal_launch.json').read_text())['pid']
    while True:
        try:running=b'restore_proto_olc_v1/run.py' in Path(f'/proc/{runner}/cmdline').read_bytes()
        except OSError:running=False
        if not running:break
        time.sleep(1)
    guard=json.loads((ROOT/'runs/restore_proto_v1/control/reservation_state.json').read_text())
    for line in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True).splitlines():
        uuid,pid=map(str.strip,line.split(','))
        if uuid in UUIDS.split(','):assert int(pid)==guard['pid'],line
    reserve('training');reserved_for_evaluation=True
    gpu_env=dict(env,CUDA_VISIBLE_DEVICES=UUIDS)
    jobs=[('reference',ROOT/'runs/restore_proto_v1/formal/full/10-5/step2/checkpoints/model_final.pth',ROOT/'runs/restore_proto_olc_v1/reference_full_gpu/fusion_evaluation.json'),
          ('olc',FORMAL/'step2/checkpoints/model_final.pth',FORMAL/'step2/fusion_evaluation.json')]
    for name,checkpoint,output in jobs:
        if output.exists():
            existing=json.loads(output.read_text())
            assert existing['device']=='cuda' and existing['checkpoint_sha256']==hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            continue
        output.parent.mkdir(parents=True,exist_ok=True)
        cmd=[sys.executable,'-B',str(ROOT/'tools/evaluate_restore_proto_fusion.py'),'--checkpoint',str(checkpoint),'--stage','2','--output',str(output),'--device','cuda','--threads','4']
        write(CONTROL/'postprocess_state.json',dict(status='paired_gpu_fusion_running',arm=name,pid=os.getpid(),time=time.time()))
        with output.with_suffix('.log').open('a') as f:
            child=subprocess.Popen(cmd,cwd=ROOT,env=gpu_env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            if child.wait()!=0:raise RuntimeError('Fusion evaluation failed: '+name)
    subprocess.run([sys.executable,'-B',str(ROOT/'tools/compare_olc_results.py')],cwd=ROOT,env=env,check=True)
    assert json.loads((ROOT/'runs/restore_proto_olc_v1/comparison.json').read_text())['status']=='complete'
    write(CONTROL/'postprocess_state.json',dict(status='completed',time=time.time()))
    print('OLC_POSTPROCESS_COMPLETED',flush=True)

if __name__=='__main__':
    torch.set_num_threads(2)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:main()
    except BaseException as e:
        write(CONTROL/'postprocess_state.json',dict(status='failed',error=repr(e),time=time.time()));raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        if reserved_for_evaluation:reserve('reserve')
