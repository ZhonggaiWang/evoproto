"""Same-budget OLC stage chains on the authorized pair of GPUs."""
import argparse, json, os, signal, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.restore_proto_v1.run import UUIDS, sha, write
import experiments.restore_proto_v1.run as common
common.CONTROL=ROOT/'runs/restore_proto_cas_v1/control'
reserve=common.reserve
CONTROL=ROOT/'runs/restore_proto_cas_v1/control'
child=None

def stop(*unused):
    if child is not None and child.poll() is None:
        os.killpg(child.pid,signal.SIGTERM)
        try: child.wait(timeout=20)
        except subprocess.TimeoutExpired: os.killpg(child.pid,signal.SIGKILL);child.wait()
    raise KeyboardInterrupt

def main():
    global child
    p=argparse.ArgumentParser()
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--arms',nargs='+',choices=['baseline','off','ignore','local_pair','confusion_pair'],default=['confusion_pair','ignore','local_pair'])
    a=p.parse_args();signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    CONTROL.mkdir(parents=True,exist_ok=True)
    out=ROOT/'runs/restore_proto_cas_v1'/('smoke' if a.smoke else 'formal')
    out.mkdir(parents=True,exist_ok=True)
    files=[]
    for relative in ['experiments/restore_proto_v1','experiments/restore_proto_cas_v1','model','utils','datasets']:
        files.extend((ROOT/relative).rglob('*.py'))
    files.extend([ROOT/'tasks.py']+list((ROOT/'tools').glob('*.py')))
    source={str(x.relative_to(ROOT)):sha(x) for x in sorted(files)}
    base=ROOT/'.runtime/restore_proto/shared_step0.pth'
    manifest=dict(source_sha256=source,initial_checkpoint=str(base),initial_sha256=sha(base),
                  iterations=4 if a.smoke else 8000,batch_size=8,seed=0,gpus=[5,6],arms=a.arms,
                  data='/data/zhonggai/coco/PascalVOC12',method="exact marginal BCE; graph-gated old/new conflicts",
                  protocol='Original 8000 updates per incremental stage; no extra refinement; old image annotations excluded',
                  checkpoint_policy='Only final per stage; protected baseline weights unchanged')
    mf=out/'manifest.json'
    if mf.exists() and json.loads(mf.read_text())!=manifest:raise RuntimeError('Source or protocol changed')
    write(mf,manifest)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=UUIDS,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',NCCL_P2P_DISABLE='1',NCCL_IB_DISABLE='1',TMPDIR=str(ROOT/'.runtime/tmp'))
    def execute(command,log):
        global child
        reserve('training')
        with log.open('a') as f:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            code=child.wait()
        reserve('reserve')
        if code:raise RuntimeError(f'Job failed exit={code}: {log}')
    launch=[sys.executable,'-B','-m','torch.distributed.run','--master_addr=127.0.0.1','--master_port=49375','--nproc_per_node=2']
    try:
        guard=json.loads((ROOT/'runs/restore_proto_cas_v1/control/reservation_state.json').read_text())
        assert time.time()-guard['time']<15
        for line in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True).splitlines():
            uuid,pid=map(str.strip,line.split(','))
            if uuid in UUIDS.split(','):assert int(pid)==guard['pid'],line
        for arm in a.arms:
            previous=base
            for stage in [1,2]:
                d=out/arm/'10-5'/f'step{stage}';d.mkdir(parents=True,exist_ok=True)
                final=d/'checkpoints/model_final.pth'
                predecessor={'path':str(previous),'sha256':sha(previous)}
                ip=d/'predecessor.json'
                if ip.exists() and json.loads(ip.read_text())!=predecessor:raise RuntimeError('Predecessor changed')
                write(ip,predecessor)
                if not final.exists():
                    if (d/'train.log').exists():raise RuntimeError('Partial run exists; inspect before restarting')
                    entry='restore_proto_v1' if arm=='baseline' else 'restore_proto_cas_v1'
                    cmd=launch+[str(ROOT/f'experiments/{entry}/train.py'),'--variant','full','--step',str(stage),'--task','10-5','--work_dir',str(out/arm),'--max_iters',str(manifest['iterations']),'--lr','2e-5','--spg','4','--num_workers','4','--seed','0','--save_ckpt','--no-pretrained','--prev_checkpoint',str(previous),'--data_folder',manifest['data'],'--val_label_dir',manifest['data']+'/SegmentationClass','--seg_label_dir',manifest['data']+'/SegmentationClass','--w_proto_kd','0','--w_proto_sep','0','--log_iters','50','--eval_iters','2000']
                    if arm!='baseline':cmd+=['--cas_mode',arm]
                    if a.smoke:cmd+=['--train_limit','32','--val_limit','8','--log_iters','1','--eval_iters','4','--warmup_iters','1','--loss_warmup_iters','1']
                    write(d/'command.json',cmd)
                    write(CONTROL/'runner_state.json',dict(status='running',arm=arm,stage=stage,smoke=a.smoke,time=time.time(),pid=os.getpid()))
                    print('START',arm,stage,flush=True);execute(cmd,d/'launcher.log')
                    if not final.exists():raise RuntimeError('Missing final checkpoint')
                    write(d/'final_receipt.json',dict(path=str(final),sha256=sha(final),source_sha256=source,time=time.time()))
                metrics=json.loads((d/'metrics.jsonl').read_text().splitlines()[-1])
                assert metrics['iteration']==manifest['iterations']
                previous=final;print('DONE',arm,stage,flush=True)
                if not a.smoke and not (d/'evaluation.json').exists():
                    execute(launch+[str(ROOT/'experiments/restore_proto_v1/evaluate.py'),'--checkpoint',str(final),'--stage',str(stage),'--output',str(d/'evaluation.json')],d/'evaluation.log')
                if not a.smoke:
                    execute([sys.executable,'-B',str(ROOT/'tools/audit_olc_prototypes.py'),'--stage-dir',str(d)],d/'prototype_audit.log')
                    execute([sys.executable,'-B',str(ROOT/'experiments/restore_proto_cas_v1/audit.py'),'--stage',str(stage),'--directory',str(d),'--output',str(d/'ambiguity_audit.json')],d/'ambiguity_audit.log')
                    if stage==2:
                        execute([sys.executable,'-B',str(ROOT/'tools/evaluate_restore_proto_fusion.py'),'--checkpoint',str(final),'--stage','2','--output',str(d/'fusion_evaluation.json'),'--device','cuda','--threads','4'],d/'fusion_evaluation.log')
        if not a.smoke:
            execute([sys.executable,'-B',str(ROOT/'experiments/restore_proto_cas_v1/compare.py')],CONTROL/'comparison.log')
        write(CONTROL/'runner_state.json',dict(status='completed',smoke=a.smoke,time=time.time()))
    except BaseException as exc:
        write(CONTROL/'runner_state.json',dict(status='failed',error=repr(exc),time=time.time()));raise
    finally:
        try: reserve('reserve')
        finally:
            if not a.smoke: write(CONTROL/'reservation_request.json',{'mode':'stop'})
if __name__=='__main__':main()
