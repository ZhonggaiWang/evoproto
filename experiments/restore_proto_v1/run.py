"""Run complete stage chains and restore GPU reservation between jobs/on failure."""
import argparse,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
CONTROL=ROOT/'runs/restore_proto_v1/control'
UUIDS='GPU-82e069d7-189e-06b0-faed-b08e1794a981,GPU-2d0a204c-101b-1149-06e1-e3528d9f38bf'
child=None

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def write(p,obj):
    assert p.resolve().is_relative_to(ROOT)
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(p)

def reserve(mode):
    state=json.loads((CONTROL/'reservation_state.json').read_text())
    if time.time()-state['time']>15:raise RuntimeError('GPU reservation heartbeat stale')
    write(CONTROL/'reservation_request.json',{'mode':mode})
    for _ in range(30):
        s=json.loads((CONTROL/'reservation_state.json').read_text())
        if s['mode']==mode and time.time()-s['time']<10:return
        time.sleep(1)
    raise RuntimeError('GPU reservation mode did not change')

def stop(*unused):
    if child is not None and child.poll() is None:
        os.killpg(child.pid,signal.SIGTERM)
        try:child.wait(timeout=30)
        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    raise KeyboardInterrupt

def main():
    global child
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--variants',nargs='+',default=['full','without_confusion','without_proto']);a=p.parse_args()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    out=ROOT/'runs/restore_proto_v1'/('smoke' if a.smoke else 'formal');out.mkdir(parents=True,exist_ok=True)
    source_files=sorted((ROOT/'experiments/restore_proto_v1').glob('*.py'))+list((ROOT/'model').rglob('*.py'))+list((ROOT/'utils').glob('*.py'))+list((ROOT/'datasets').glob('*.py'))
    source={str(x.relative_to(ROOT)):sha(x) for x in source_files}
    base=ROOT/'.runtime/restore_proto/shared_step0.pth'
    manifest={'source_sha256':source,'initial_checkpoint':str(base),'initial_sha256':sha(base),'python':sys.executable,'gpus':[5,6],'variants':a.variants,'iterations':4 if a.smoke else 8000,'batch_size':8,'seed':0,'data':'/data/zhonggai/coco/PascalVOC12','checkpoint_policy':'final_only; stage1 retained as necessary predecessor','step0':'Reused fixed_baseline_v1 shared final, 20000 updates; not retrained','ALD':'off; V9 final-stage reference dependency is excluded'}
    mf=out/'manifest.json'
    if mf.exists() and json.loads(mf.read_text())!=manifest:raise RuntimeError('Existing experiment provenance changed')
    write(mf,manifest)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=UUIDS,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',NCCL_P2P_DISABLE='1',NCCL_IB_DISABLE='1',TMPDIR=str(ROOT/'.runtime/tmp'))
    (ROOT/'.runtime/tmp').mkdir(exist_ok=True,parents=True)
    try:
        def evaluate_checkpoint(checkpoint, stage, output, expected=None):
            global child
            if output.exists(): return
            reserve('training')
            cmd=[sys.executable,'-B','-m','torch.distributed.run','--master_addr=127.0.0.1','--master_port=49365','--nnodes=1','--nproc_per_node=2',str(ROOT/'experiments/restore_proto_v1/evaluate.py'),'--checkpoint',str(checkpoint),'--stage',str(stage),'--output',str(output)]
            if expected is not None:cmd+=['--square-only','--expected-square',str(expected)]
            print('EVALUATE',stage,str(checkpoint),flush=True)
            with output.with_suffix('.log').open('a') as f:
                child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                code=child.wait()
            reserve('reserve')
            if code:raise RuntimeError('Evaluation failed: '+str(output))
        if not a.smoke:
            evaluate_checkpoint(base,0,out/'shared_step0_evaluation.json',83.556)
        for variant in a.variants:
            previous=base
            for stage in [1,2]:
                d=out/variant/'10-5'/f'step{stage}';d.mkdir(parents=True,exist_ok=True)
                final=d/'checkpoints/model_final.pth'
                inp={'path':str(previous),'sha256':sha(previous)}
                ip=d/'predecessor.json'
                if ip.exists() and json.loads(ip.read_text())!=inp:raise RuntimeError('Predecessor changed')
                write(ip,inp)
                if final.exists():
                    completed=json.loads((d/'metrics.jsonl').read_text().splitlines()[-1])
                    if completed['iteration']!=manifest['iterations']:raise RuntimeError('Final evaluation incomplete')
                    previous=final;continue
                cmd=[sys.executable,'-B','-m','torch.distributed.run','--master_addr=127.0.0.1','--master_port=49365','--nnodes=1','--nproc_per_node=2',str(ROOT/'experiments/restore_proto_v1/train.py'),'--variant',variant,'--step',str(stage),'--task','10-5','--work_dir',str(out/variant),'--max_iters',str(manifest['iterations']),'--lr','2e-5','--spg','4','--num_workers','4','--seed','0','--save_ckpt','--no-pretrained','--prev_checkpoint',str(previous),'--data_folder',manifest['data'],'--val_label_dir',manifest['data']+'/SegmentationClass','--seg_label_dir',manifest['data']+'/SegmentationClass','--w_proto_kd','0','--w_proto_sep','0','--log_iters','50','--eval_iters','2000']
                if a.smoke:cmd+=['--train_limit','32','--val_limit','8','--max_iters','4','--log_iters','1','--eval_iters','4','--warmup_iters','1','--loss_warmup_iters','1']
                write(d/'command.json',cmd)
                reserve('training')
                write(CONTROL/'runner_state.json',{'status':'running','variant':variant,'stage':stage,'smoke':a.smoke,'time':time.time()})
                print('START',variant,stage,flush=True)
                with (d/'launcher.log').open('a') as f:
                    child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                    code=child.wait()
                reserve('reserve')
                if code or not final.exists():raise RuntimeError(f'{variant} stage{stage} failed exit={code}; inspect {d}/launcher.log')
                write(d/'final_receipt.json',{'path':str(final),'sha256':sha(final),'time':time.time(),'source_sha256':source})
                previous=final
                print('DONE',variant,stage,flush=True)
                if not a.smoke and stage==2:evaluate_checkpoint(final,stage,d/'evaluation.json')
        write(CONTROL/'runner_state.json',{'status':'completed','smoke':a.smoke,'time':time.time()})
    except BaseException as exc:
        write(CONTROL/'runner_state.json',{'status':'failed','error':repr(exc),'time':time.time()})
        raise
    finally:reserve('reserve')

if __name__=='__main__':main()
