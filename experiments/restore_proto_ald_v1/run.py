"""Explicitly invoked refinement queue; plans by default and never interrupts training."""
import argparse,json,os,signal,subprocess,sys,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.restore_proto_v1.run import reserve,UUIDS
CONTROL=ROOT/'runs/restore_proto_v1/control'
BASE=ROOT/'runs/restore_proto_v1/formal'
ARMS={'full_control':('full',False),'full_ald':('full',True),
      'without_confusion_ald':('without_confusion',True),'without_proto_ald':('without_proto',True)}
child=None


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda:stream.read(8*1024*1024),b''):h.update(data)
    return h.hexdigest()


def read(path):return json.loads(path.read_text())


def write(path,obj):
    assert path.resolve().is_relative_to(ROOT)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(obj,indent=2));temp.replace(path)


def source_hashes():
    files=list((ROOT/'experiments/restore_proto_ald_v1').glob('*.py'))
    for directory in ('experiments/restore_proto_v1','model','utils','datasets'):
        files.extend((ROOT/directory).rglob('*.py'))
    return {str(p.relative_to(ROOT)):digest(p) for p in sorted(set(files))}


def stop(*unused):
    if child is not None and child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
    raise KeyboardInterrupt


def execute(command,log):
    global child
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=UUIDS,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',NCCL_P2P_DISABLE='1',NCCL_IB_DISABLE='1',TMPDIR=str(ROOT/'.runtime/tmp'))
    reserve('training');write(log.with_suffix('.command.json'),command)
    with log.open('a') as stream:
        child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=child.wait()
        except BaseException:
            if child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            raise
    reserve('reserve')
    if code:raise RuntimeError(f'Job failed ({code}); inspect {log}')


def torchrun(script):
    return [sys.executable,'-B','-m','torch.distributed.run','--master_addr=127.0.0.1','--master_port=49365','--nnodes=1','--nproc_per_node=2',str(ROOT/script)]


def ensure_idle_formal_queue():
    pidfile=CONTROL/'formal.pid'
    if pidfile.exists():
        cmdline=Path('/proc')/pidfile.read_text().strip()/'cmdline'
        if cmdline.exists() and b'restore_proto_v1/run.py' in cmdline.read_bytes():
            raise RuntimeError('Formal restoration coordinator is still live; leave its GPU jobs running.')
    assert read(CONTROL/'runner_state.json')['status']=='completed'
    state=read(CONTROL/'reservation_state.json')
    assert state['mode']=='reserve' and time.time()-state['time']<10
    pid=state['pid'];assert b'gpu_reservation' in Path(f'/proc/{pid}/cmdline').read_bytes()
    # A live CUDA workload on either card must be the recorded reservation guard.
    lines=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True).splitlines()
    for line in lines:
        uuid,process=line.split(',')
        if uuid.strip() in UUIDS.split(','):assert int(process)==pid, 'Another process is using a target GPU: '+line


def main():
    import fcntl
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['plan','smoke','formal'],default='plan')
    parser.add_argument('--arms',nargs='+',choices=list(ARMS),default=['full_control','full_ald'])
    parser.add_argument('--attempt',default='v1');parser.add_argument('--decision',default='')
    args=parser.parse_args();assert args.attempt.replace('_','').replace('-','').isalnum()
    study=ROOT/'runs/restore_proto_ald_v1'/args.attempt
    if args.mode=='plan':
        print(json.dumps({'launches_jobs':False,'arms':{a:ARMS[a] for a in args.arms},'global_batch':8,'formal_updates':1200,'sample_exposures':9600,'fresh_optimizer':True,'output':str(study),'prerequisite':'Formal restoration queue finished; corrected evaluation and GPU smoke passed'},indent=2));return
    if args.mode=='formal' and not args.decision:parser.error('Record the evidence-based activation rationale with --decision')
    ensure_idle_formal_queue()
    study.mkdir(parents=True,exist_ok=True);lock=(study/'runner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    source=source_hashes();protocol={'source_sha256':source,'global_batch':8,'gpus':[5,6],'formal_updates':1200,'optimizer':'fresh AdamW','lr_encoder':2e-6,'lr_head':2e-5,'schedule':'cosine; no warmup','weight_policy':'only final and necessary parent dependencies'}
    manifest=study/'manifest.json'
    if manifest.exists():assert read(manifest)==protocol,'Candidate source/protocol changed; use a new attempt and repeat smoke'
    else:write(manifest,protocol)
    write(study/'runner.pid',os.getpid())
    try:
        if args.mode=='formal':
            full_eval=read(BASE/'full/10-5/step2/evaluation.json')
            assert full_eval['histogram_label_dtype']=='int64' and full_eval['images']==1449
            assert read(BASE/'full/10-5/step2/fusion_verification.json')['per_image_histograms_verified']
        for arm in args.arms:
            variant,ald=ARMS[arm];parent=BASE/variant/'10-5';teacher=parent/'step1/checkpoints/model_final.pth';start=parent/'step2/checkpoints/model_final.pth';graph=parent/'step2/confusion.json'
            for step,path in [(1,teacher),(2,start)]:
                receipt=read(parent/f'step{step}/final_receipt.json');assert digest(path)==receipt['sha256']
                assert read(parent/f'step{step}/config.json')['variant']==variant
            lineage={'teacher':str(teacher),'teacher_sha256':digest(teacher),'start':str(start),'start_sha256':digest(start),'confusion':str(graph),'confusion_sha256':digest(graph)}
            directory=study/args.mode/arm/'10-5/step2';directory.mkdir(parents=True,exist_ok=True)
            if args.mode=='formal':
                smoke=read(study/'smoke'/arm/'10-5/step2/completion.json')
                assert smoke['source_sha256']==source and smoke['lineage']==lineage
                assert smoke['gpu_preflight_passed']
            completion=directory/'completion.json'
            if completion.exists():
                saved=read(completion);assert saved['source_sha256']==source and saved['lineage']==lineage
                assert digest(directory/'checkpoints/model_final.pth')==saved['checkpoint_sha256'];continue
            write(study/'state.json',{'status':'running','mode':args.mode,'arm':arm,'time':time.time(),'pid':os.getpid()})
            evidence=study/args.mode/f'{variant}_image_evidence.json'
            if ald and not evidence.exists():
                command=torchrun('experiments/restore_proto_ald_v1/cache.py')+['--teacher',str(teacher),'--reference',str(start),'--output',str(evidence)]
                if args.mode=='smoke':command+=['--limit','32']
                execute(command,study/args.mode/f'{variant}_cache.log')
            iterations=4 if args.mode=='smoke' else 1200
            command=torchrun('experiments/restore_proto_ald_v1/train.py')+['--variant',variant,'--step','2','--spg','4','--max_iters',str(iterations),'--eval_iters',str(iterations),'--log_iters','1' if args.mode=='smoke' else '50','--lr','2e-6','--no-pretrained','--loss_warmup_iters','0','--w_proto_kd','0','--w_proto_sep','0','--save_ckpt','--start-checkpoint',str(start),'--prev_checkpoint',str(teacher),'--initial-confusion',str(graph),'--work_dir',str(study/args.mode/arm),'--data_folder','/data/zhonggai/coco/PascalVOC12','--seg_label_dir','/data/zhonggai/coco/PascalVOC12/SegmentationClass','--val_label_dir','/data/zhonggai/coco/PascalVOC12/SegmentationClass']
            if ald:command+=['--refine-ald','--image-evidence',str(evidence)]
            if args.mode=='smoke':command+=['--train_limit','32','--val_limit','8']
            execute(command,directory/'launcher.log')
            checks=[read(directory/f'refinement_preflight_rank{rank}.json') for rank in range(2)]
            assert all(c['passed'] and c['teacher_and_reference_frozen'] and c['world_size']==2 and c['batch_per_rank']==4 and c['variant']==variant and c['ald']==ald for c in checks)
            final=directory/'checkpoints/model_final.pth'
            import torch
            from experiments.restore_proto_ald_v1.cache import load
            torch.set_num_threads(4)
            raw=torch.load(final,map_location='cpu',weights_only=True)
            assert raw['iteration']==iterations and raw['variant']==variant and raw['refine_ald']==ald
            assert all(torch.isfinite(t).all().item() for t in raw['model_state'].values())
            net=load(final,2,'cpu');assert [tuple(p.shape) for p in net.decoder.class_prototypes.parameters()]==[(11,512),(5,512),(5,512)]
            del net,raw
            metrics=json.loads((directory/'metrics.jsonl').read_text().splitlines()[-1]);assert metrics['iteration']==iterations
            if args.mode=='formal':
                execute(torchrun('experiments/restore_proto_v1/evaluate.py')+['--checkpoint',str(final),'--stage','2','--output',str(directory/'evaluation.json'),'--expected-square',str(metrics['all_miou'])],directory/'evaluation.log')
                evaluation=read(directory/'evaluation.json');assert evaluation['histogram_label_dtype']=='int64' and evaluation['images']==1449
                assert evaluation['checkpoint_sha256']==digest(final)
            assert source_hashes()==source,'Candidate sources changed during execution'
            write(completion,{'status':'completed','mode':args.mode,'arm':arm,'source_sha256':source,'lineage':lineage,'checkpoint_sha256':digest(final),'checkpoint':str(final),'gpu_preflight_passed':True,'preflight':checks,'metrics':metrics,'activation_rationale':args.decision,'time':time.time()})
        write(study/'state.json',{'status':'completed','mode':args.mode,'arms':args.arms,'time':time.time(),'pid':os.getpid()})
    except BaseException as exc:
        write(study/'state.json',{'status':'failed','mode':args.mode,'error':repr(exc),'time':time.time(),'pid':os.getpid()});raise
    finally:reserve('reserve')


if __name__=='__main__':main()
