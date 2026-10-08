"""One conditional correction: resume the owned stage2 warmup, no stage1 rerun."""
from pathlib import Path
import os,sys,json,subprocess,signal,socket,fcntl,argparse,time
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto')
E=Path(__file__).resolve().parent; S=E/'b_semantic/src'
PARENT=R/'runs/prototype_sep_v1/formal/a_geometry'
sys.path.insert(0,str(R/'experiments/kd_pixel_v2'))
from run_kd import environment
sys.path.insert(0,str(S))
from kd_runtime import safe_path,atomic_json,now,digest
PY=R/'.runtime/env/bin/python'

def sources():
    return {str(p.relative_to(S)):digest(p) for p in S.rglob('*.py') if '__pycache__' not in p.parts}

def main(smoke):
    assert json.loads((PARENT/'status.json').read_text())['status']=='complete','Primary must finish first'
    decision=json.loads((R/'runs/prototype_sep_v1/result_analysis.json').read_text())['recommendation']
    assert not decision['candidate_exceeds_required_endpoint'],'Do not launch an unnecessary correction'
    run=safe_path(R/'runs/prototype_sep_v1'/('smoke' if smoke else 'formal')/'b_semantic')
    run.mkdir(parents=True,exist_ok=True)
    lock=open(safe_path(run/'coordinator.lock'),'a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (run/'study.json').exists(),'Refuse duplicate training'
    cfg=json.loads((PARENT/'study.json').read_text())['common_config']
    previous=safe_path(PARENT/'10-5/step1/checkpoints/model_final.pth')
    resume=safe_path(PARENT/'10-5/step2/checkpoints/model_iter_2000.pth')
    assert previous.is_file() and resume.is_file()
    if not smoke:
        assert json.loads((R/'runs/prototype_sep_v1/smoke/b_semantic/status.json').read_text())['status']=='complete'
        assert json.loads((E/'semantic_preflight.json').read_text())['passed']
    code=sources();env=environment('8card');env['CUDA_VISIBLE_DEVICES']='0,1,2,3,4,5,6,7'
    atomic_json(run/'study.json',{'created_utc':now(),'arm':'b_semantic','smoke':smoke,
        'common_config':cfg,'train_gpus':list(range(8)),'batch_per_gpu':1,'global_batch':8,
        'source_sha256':code,'runner_sha256':digest(Path(__file__)),
        'inherited_stage1_run':str(PARENT),'teacher':str(previous),'teacher_sha256':digest(previous),
        'resume_checkpoint':str(resume),'resume_checkpoint_sha256':digest(resume),'resume_iteration':2000,
        'restored_state':['student','optimizer','online_confusion','geometry_selector'],
        'mechanism':'Same geometry candidate; replace only SEP conflicting new-row tangent gradients with semantic-orthogonal projection before DDP output wrapping',
        'scope':'One evidence-driven stage2 repair; reuse step0, completed stage1 and stage2 warmup; no seeds, grids or baseline reruns',
        'limits':'Current Euclidean component-gradient guard, not AdamW or IoU guarantee; sampler/augmentations restart on resume; development validation fixedseed0'})
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
        current=dict(cfg)
        if smoke:current.update(max_iters=2004,log_iters=1,eval_iters=2004,val_limit=16,train_limit=64)
        cmd=[str(PY),'-B','-m','torch.distributed.run','--standalone','--nproc_per_node=8',
            str(S/'scripts/dist_train_voc_seg_neg.py'),'--work_dir',str(run),'--step','2',
            '--prev_checkpoint',str(previous),'--resume_checkpoint',str(resume)]
        for key,value in current.items():
            if isinstance(value,bool):
                if value:cmd.append('--'+key)
                elif key=='pretrained':cmd.append('--no-pretrained')
            elif isinstance(value,list):cmd.extend(['--'+key,*map(str,value)])
            else:cmd.extend(['--'+key,str(value)])
        stage=safe_path(run/'10-5/step2');stage.mkdir(parents=True,exist_ok=True)
        with open(safe_path(stage/'launcher.log'),'x') as log:
            train_child=subprocess.Popen(cmd,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        record={'pid':train_child.pid,'starttime':Path(f'/proc/{train_child.pid}/stat').read_text().split()[21],
            'command':cmd,'host':socket.gethostname(),'start_utc':now(),'teacher':str(previous),'teacher_sha256':digest(previous)}
        atomic_json(stage/'launch.json',record)
        atomic_json(run/'status.json',{'status':'training','step':2,'utc':now()})
        while train_child.poll() is None:
            assert sources()==code,'Frozen source changed'
            failed=[child.returncode for child,_ in workers if child.poll() is not None]
            assert not failed,('Evaluator exited before training completed',failed)
            time.sleep(2)
        rc=train_child.returncode
        atomic_json(stage/'launch.json',{**record,'returncode':rc,'end_utc':now()})
        if rc:raise RuntimeError(f'Stage2 failed: {rc}')
        final=stage/'checkpoints/model_final.pth';assert final.exists()
        atomic_json(stage/'training_complete.json',{'returncode':rc,'utc':now(),'checkpoint_sha256':digest(final)})
        atomic_json(run/'training_complete.json',{'status':'complete','stages_trained':1,'inherited_stage1':str(PARENT),'utc':now()})
        atomic_json(run/'status.json',{'status':'finishing_evaluation','utc':now()})
        deadline=time.monotonic()+600
        while not all(child.poll() is not None for child,_ in workers):
            if time.monotonic()>deadline:raise RuntimeError('Evaluation timeout after training')
            if any(child.poll() not in (None,0) for child,_ in workers):raise RuntimeError('Evaluation failed')
            time.sleep(2)
        for rank,(child,record) in enumerate(workers):
            assert child.returncode==0
            atomic_json(run/f'evaluator_rank{rank}.process.json',{**record,'returncode':child.returncode,'end_utc':now()})
        queue=list((run/'eval_queue').glob('*.json'));assert len(queue)==(1 if smoke else 3)
        assert all((run/'evaluations'/q.stem/'result.json').exists() for q in queue)
        atomic_json(run/'evaluation_workers_complete.json',{'returncodes':[c.returncode for c,_ in workers],'utc':now()})
        atomic_json(run/'status.json',{'status':'complete','utc':now()})
        print(json.dumps({'status':'complete','run':str(run)}),flush=True)
    except BaseException as exc:
        atomic_json(run/'failure.json',{'exception':repr(exc),'utc':now()})
        atomic_json(run/'status.json',{'status':'failed','utc':now()})
        for child in [train_child,*[c for c,_ in workers]]:
            if child is not None and child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');a=p.parse_args();main(a.smoke)
