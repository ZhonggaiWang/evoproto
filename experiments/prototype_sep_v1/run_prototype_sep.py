"""One substantive geometry candidate, reused warmup, asynchronous full eval."""
from pathlib import Path
import sys,os,json,subprocess,signal,socket,fcntl,argparse,time
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto')
E=Path(__file__).resolve().parent; S=E/'a_geometry/src'
sys.path.insert(0,str(R/'experiments/kd_pixel_v2'))
from run_kd import environment
sys.path.insert(0,str(S))
from kd_runtime import safe_path,atomic_json,now,digest
PY=R/'.runtime/env/bin/python'

def sources():
    return {str(p.relative_to(S)):digest(p) for p in S.rglob('*.py') if '__pycache__' not in p.parts}

def main(smoke):
    run=safe_path(R/'runs/prototype_sep_v1'/('smoke' if smoke else 'formal')/'a_geometry')
    run.mkdir(parents=True,exist_ok=True)
    lock=open(safe_path(run/'coordinator.lock'),'a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (run/'study.json').exists(),'Refuse duplicate training'
    cfg=json.loads((R/'runs/kd_parallel_v1/formal/b_relational/study.json').read_text())['common_config']
    cfg.update(spg=1,num_workers=2,w_geometry_sep=.1,confusion_momentum=.98,
        pair_refresh_interval=50,pair_min_row_images=8,pair_min_pair_images=3,
        pair_min_rate=.01,pair_min_updates=100,pair_ramp_updates=200,pair_max_stale_updates=200)
    assert cfg['spg']*8==8 and cfg['w_proto_sep']==0 and cfg['w_proto_kd']==0
    previous=R/'runs/fixed_baseline_v1/shared/10-5/step0/checkpoints/model_final.pth'
    assert digest(previous)=='4f298e14721630cf3c66ba4f6af04bd198ebeaf77adc6b7b07ab8403ad0893b5'
    resume=R/'runs/kd_pixel_v2/10-5/step1/checkpoints/model_iter_2000.pth'
    assert digest(resume)=='4cdc0087e802524b14aa4c4a10a32dd4c20f7484f35525f249965b8d6aa5ca8f'
    code=sources();env=environment('8card');env['CUDA_VISIBLE_DEVICES']='0,1,2,3,4,5,6,7'
    atomic_json(run/'study.json',{'created_utc':now(),'arm':'a_geometry','smoke':smoke,
        'common_config':cfg,'train_gpus':list(range(8)),'batch_per_gpu':1,'global_batch':8,
        'evaluation':'4 sharded evaluators on same8card host; forward only while a job is queued',
        'source_sha256':code,'runner_sha256':digest(Path(__file__)),
        'step0':str(previous),'step0_sha256':digest(previous),
        'step1_warmup':str(resume),'step1_warmup_sha256':digest(resume),
        'mechanism':'supported hard prototype pairs involving new classes; old reference stop gradient within SEP; squared cosine hinge margin0',
        'resume_caveat':'student and optimizer restored at2000; original baseline4x2 versus candidate8x1 restarts sampler/augmentations; development validation, fixedseed0',
        'scope':'no seed search, no parameter grid, no baseline or step0 retraining; original optimized KD unchanged'})
    workers=[];train_child=None
    def stop(signum,frame):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        for rank in range(4):
            cmd=[str(PY),'-B',str(S/'evaluate_kd.py'),'--run',str(run),'--rank',str(rank),'--shards','4']
            with open(safe_path(run/f'evaluator_rank{rank}.log'),'x') as log:
                child=subprocess.Popen(cmd,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            record={'pid':child.pid,'starttime':Path(f'/proc/{child.pid}/stat').read_text().split()[21],
                'command':cmd,'host':socket.gethostname(),'utc':now()}
            atomic_json(run/f'evaluator_rank{rank}.process.json',record);workers.append((child,record))
        for step in [1,2]:
            assert sources()==code,'Frozen source changed'
            current=dict(cfg)
            if smoke:
                end=2005 if step==1 else 5
                current.update(max_iters=end,log_iters=1,eval_iters=end,val_limit=16,train_limit=64,
                    pair_min_updates=0,pair_ramp_updates=1,pair_min_row_images=1,
                    pair_min_pair_images=1,pair_min_rate=0.0,pair_refresh_interval=1)
                if step==2:current.update(warmup_iters=1,loss_warmup_iters=1)
            cmd=[str(PY),'-B','-m','torch.distributed.run','--standalone','--nproc_per_node=8',
                str(S/'scripts/dist_train_voc_seg_neg.py'),'--work_dir',str(run),
                '--step',str(step),'--prev_checkpoint',str(previous)]
            if step==1:cmd.extend(['--resume_checkpoint',str(resume)])
            for key,value in current.items():
                if isinstance(value,bool):
                    if value:cmd.append('--'+key)
                    elif key=='pretrained':cmd.append('--no-pretrained')
                elif isinstance(value,list):cmd.extend(['--'+key,*map(str,value)])
                else:cmd.extend(['--'+key,str(value)])
            stage=safe_path(run/f'10-5/step{step}');stage.mkdir(parents=True,exist_ok=True)
            with open(safe_path(stage/'launcher.log'),'x') as log:
                train_child=subprocess.Popen(cmd,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            record={'pid':train_child.pid,'starttime':Path(f'/proc/{train_child.pid}/stat').read_text().split()[21],
                'command':cmd,'host':socket.gethostname(),'start_utc':now(),
                'teacher':str(previous),'teacher_sha256':digest(previous)}
            atomic_json(stage/'launch.json',record)
            atomic_json(run/'status.json',{'status':'training','step':step,'utc':now()})
            while train_child.poll() is None:
                failed=[child.returncode for child,_ in workers if child.poll() is not None]
                assert not failed,('Evaluator exited before training completed',failed)
                time.sleep(2)
            rc=train_child.returncode
            atomic_json(stage/'launch.json',{**record,'returncode':rc,'end_utc':now()})
            if rc:raise RuntimeError(f'Training step{step} failed: {rc}')
            previous=stage/'checkpoints/model_final.pth';assert previous.exists()
            atomic_json(stage/'training_complete.json',{'returncode':rc,'utc':now(),'checkpoint_sha256':digest(previous)})
        atomic_json(run/'training_complete.json',{'status':'complete','stages':2,'utc':now()})
        atomic_json(run/'status.json',{'status':'finishing_evaluation','utc':now()})
        deadline=time.monotonic()+600
        while not all(child.poll() is not None for child,_ in workers):
            if time.monotonic()>deadline:raise RuntimeError('Evaluation timeout after training')
            if any(child.poll() not in (None,0) for child,_ in workers):raise RuntimeError('Evaluation failed')
            time.sleep(2)
        for rank,(child,record) in enumerate(workers):
            assert child.returncode==0
            atomic_json(run/f'evaluator_rank{rank}.process.json',{**record,'returncode':child.returncode,'end_utc':now()})
        queue=list((run/'eval_queue').glob('*.json'));assert queue
        assert all((run/'evaluations'/q.stem/'result.json').exists() for q in queue)
        atomic_json(run/'evaluation_workers_complete.json',{'returncodes':[c.returncode for c,_ in workers],'utc':now()})
        atomic_json(run/'status.json',{'status':'complete','utc':now()})
        print(json.dumps({'status':'complete','run':str(run)}),flush=True)
    except BaseException as exc:
        atomic_json(run/'failure.json',{'exception':repr(exc),'utc':now()})
        atomic_json(run/'status.json',{'status':'failed','utc':now()})
        for child in [train_child,*[c for c,_ in workers]]:
            if child is not None and child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');a=p.parse_args();main(a.smoke)
