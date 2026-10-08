"""Own one two-stage KD trajectory on 8-card host; 8-card host also evaluates it."""
from pathlib import Path
import os,sys,json,argparse,subprocess,socket,fcntl,signal,ast
EXP=Path(__file__).resolve().parent
SRC=EXP/'src';ROOT=EXP.parents[1]
sys.path.insert(0,str(SRC))
from kd_runtime import safe_path,atomic_json,digest,now
PY=ROOT/'.runtime/env/bin/python'

def environment(role):
    env=os.environ.copy();runtime=ROOT/'.runtime/kd_pixel_v2'/role
    for key,sub in {'TMPDIR':'tmp','XDG_CACHE_HOME':'cache','HF_HOME':'cache/hf','TORCH_HOME':'cache/torch',
                    'MPLCONFIGDIR':'mpl','CUDA_CACHE_PATH':'cuda','TRITON_CACHE_DIR':'triton'}.items():
        p=safe_path(ROOT.parents[1]/('.kd8tmp' if role=='8card' else '.kd4tmp') if key=='TMPDIR' else runtime/sub)
        p.mkdir(parents=True,exist_ok=True);env[key]=str(p)
    env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
               MPLBACKEND='Agg',KD_HOST=role)
    return env

def frozen_sources():
    result={}
    for p in sorted(SRC.rglob('*.py')):
        if '__pycache__' in p.parts:continue
        ast.parse(p.read_text(),filename=str(p));result[str(p.relative_to(SRC))]=digest(p)
    result['../run_kd.py']=digest(Path(__file__))
    return result

def baseline_job(run,variant,step):
    stage=ROOT/f'runs/fixed_baseline_v1/{variant}/10-5/step{step}'
    config=json.loads((stage/'config.json').read_text());checkpoint=stage/'checkpoints/model_final.pth'
    path=run/'eval_queue'/f'baseline_{variant}_step{step}.json'
    if not path.exists():
        atomic_json(path,{'study':'historical_reference_validation','checkpoint':str(checkpoint),
            'checkpoint_sha256':digest(checkpoint),'config':config,'step':step,'iteration':8000,
            'reference_metrics':str(stage/'metrics.jsonl'),'published_utc':now()})

def admission(gpus,minimum):
    text=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    rows={int(line.split(',')[0]):[int(x.strip()) for x in line.split(',')[1:]] for line in text.splitlines()}
    for gpu in gpus:
        if rows[gpu][0]<minimum:raise RuntimeError(f'Insufficient free memory GPU{gpu}: {rows[gpu]}')
    return rows

def train(run,smoke):
    env=environment('8card');env['CUDA_VISIBLE_DEVICES']='0,1,2,3,4,5,6,7'
    origin=json.loads((EXP/'origin.json').read_text());previous=Path(origin['step0'])
    if digest(previous)!=origin['step0_sha256']:raise RuntimeError('step0 hash changed')
    config=json.loads((ROOT/'runs/fixed_baseline_v1/base/10-5/step1/config.json').read_text())
    for key in ['ckpt_dir','pred_dir','work_dir','prev_checkpoint','step','local_rank']:config.pop(key)
    config.update(spg=1,num_workers=1 if smoke else 2,w_proto_kd=0.,w_proto_sep=0.,w_pixel_kd=.1,kd_temperature=2.,async_eval=True)
    if smoke:config.update(max_iters=4,log_iters=1,eval_iters=4,warmup_iters=1,loss_warmup_iters=1,train_limit=64,val_limit=16)
    code=frozen_sources()
    manifest={'study':'Evidence-gated old-output Bernoulli KD','created_utc':now(),'host':socket.gethostname(),
              'source':str(SRC),'source_sha256':code,'step0':origin,'common_config':config,
              'global_batch':8,'train_gpus':list(range(8)),'eval_host_alias':'8card','smoke':smoke,
              'budget':'2 incremental stages only; frozen shared step0; no baseline training'}
    if (run/'study.json').exists():raise RuntimeError('This study already exists; refuse accidental rerun')
    atomic_json(run/'study.json',manifest)
    for step in (1,2):
        if frozen_sources()!=code:raise RuntimeError('Frozen KD source changed')
        resources=admission(range(8),24000)
        command=[str(PY),'-B','-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=8',
                 str(SRC/'scripts/dist_train_voc_seg_neg.py'),'--work_dir',str(run),'--step',str(step),'--prev_checkpoint',str(previous)]
        for key,value in config.items():
            if isinstance(value,bool):
                if value:command.append('--'+key)
                elif key=='pretrained':command.append('--no-pretrained')
            elif isinstance(value,list):command.extend(['--'+key,*map(str,value)])
            else:command.extend(['--'+key,str(value)])
        stage=safe_path(run/'10-5'/f'step{step}');stage.mkdir(parents=True,exist_ok=True)
        if (stage/'config.json').exists():raise RuntimeError(f'Stage already launched: {stage}')
        record={'step':step,'previous':str(previous),'previous_sha256':digest(previous),'command':command,
                'host':socket.gethostname(),'start_utc':now(),'resources':resources,'global_batch':8}
        atomic_json(stage/'launch.json',record)
        with open(safe_path(stage/'launcher.log'),'x') as log:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            def stop(signum,frame):
                os.killpg(child.pid,signal.SIGTERM);raise KeyboardInterrupt
            signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
            record.update(pid=child.pid,process_group=child.pid)
            atomic_json(stage/'launch.json',record)
            atomic_json(run/'training_status.json',{'status':'running','step':step,'pid':child.pid,'host':socket.gethostname(),'utc':now()})
            rc=child.wait()
        record.update(returncode=rc,end_utc=now());atomic_json(stage/'launch.json',record)
        if rc:
            atomic_json(run/'training_status.json',{'status':'failed','step':step,'returncode':rc,'utc':now()})
            raise RuntimeError(f'Training failed; see {stage}/launcher.log')
        previous=stage/'checkpoints/model_final.pth'
        if not previous.exists():raise RuntimeError('Missing final checkpoint')
        atomic_json(stage/'training_complete.json',{'returncode':rc,'checkpoint_sha256':digest(previous),'utc':now()})
    atomic_json(run/'training_complete.json',{'status':'training_complete','stages':2,'utc':now(),'evaluation_pending':True})
    atomic_json(run/'training_status.json',{'status':'training_complete','utc':now()})

def eval_workers(run):
    env=environment('4card');env['CUDA_VISIBLE_DEVICES']='0,1,2,3'
    admission(range(4),10000)
    children=[]
    for rank in range(4):
        log=open(safe_path(run/f'evaluator_rank{rank}.log'),'a')
        cmd=[str(PY),'-B',str(SRC/'evaluate_kd.py'),'--run',str(run),'--rank',str(rank)]
        child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        children.append(child);log.close()
        atomic_json(run/f'evaluator_rank{rank}.process.json',{'pid':child.pid,'host':socket.gethostname(),'command':cmd,'utc':now()})
    codes=[p.wait() for p in children]
    atomic_json(run/'evaluation_workers_complete.json',{'returncodes':codes,'utc':now()})
    if any(codes):raise RuntimeError(codes)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('role',choices=['train','eval','prepare']);parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    run=safe_path(ROOT/('runs/kd_pixel_v2_smoke' if args.smoke else 'runs/kd_pixel_v2'));run.mkdir(parents=True,exist_ok=True)
    lock=open(safe_path(run/f'{args.role}.lock'),'a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if args.role=='prepare':
        if not args.smoke:
            for variant in ['base','kd']:
                for step in [1,2]:baseline_job(run,variant,step)
        print(str(run))
    elif args.role=='train':train(run,args.smoke)
    else:eval_workers(run)

if __name__=='__main__':main()
