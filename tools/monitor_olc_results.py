"""Audit online prediction snapshots and evaluate the completed same-budget chain."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
CONTROL=ROOT/'runs/restore_proto_olc_v1/control'
FORMAL=ROOT/'runs/restore_proto_olc_v1/formal/olc/10-5'

def write(path,value):
    assert path.resolve().is_relative_to(ROOT)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)

def main():
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
    d=FORMAL/'step2';output=d/'fusion_evaluation.json'
    if not output.exists():
        cmd=[sys.executable,'-B',str(ROOT/'tools/evaluate_restore_proto_fusion.py'),'--checkpoint',str(d/'checkpoints/model_final.pth'),'--stage','2','--output',str(output),'--device','cpu','--threads','8']
        write(CONTROL/'postprocess_state.json',dict(status='fusion_running',pid=os.getpid(),time=time.time()))
        with (d/'fusion_evaluation.log').open('a') as f:subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    write(CONTROL/'postprocess_state.json',dict(status='completed',time=time.time()))
    print('OLC_POSTPROCESS_COMPLETED',flush=True)

if __name__=='__main__':
    torch.set_num_threads(2)
    try:main()
    except BaseException as e:
        write(CONTROL/'postprocess_state.json',dict(status='failed',error=repr(e),time=time.time()));raise
