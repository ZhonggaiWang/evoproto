"""One stage2 continuation, frozen teacher/warmup, eight GPU training and four evaluators."""
from pathlib import Path
import sys,os,json,subprocess,time,signal,fcntl,argparse
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto')
E=R/'experiments/hierarchy_kd_v2'; U=R/'runs/hierarchy_kd_v2'; S=E/'src'
PARENT_E=R/'experiments/prototype_sep_v1'
PARENT=R/'runs/prototype_sep_v1/formal/c_new_anchor'
PY=R/'.runtime/env/bin/python'
sys.path.insert(0,str(PARENT_E))
import run_new_anchor_sep as prior
prior.S=S
services=prior.initialize()
environment,safe_path,atomic_json,now,digest=services
require=prior.require

def source_hashes():
    return prior.sources(services,S)

def main(smoke=False,launch=False):
    run=safe_path(U/('smoke' if smoke else 'formal')/'h2')
    if launch:
        require(not (run/'coordinator.process.json').exists(),'Duplicate launch')
        run.mkdir(parents=True,exist_ok=True)
        with open(safe_path(run/'coordinator.log'),'x') as log:
            cmd=[str(PY),'-B',str(E/'run_hierarchy.py')]+(['--smoke'] if smoke else [])
            child=subprocess.Popen(cmd,cwd=R,env=environment('8card'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        atomic_json(run/'coordinator.process.json',{'pid':child.pid,'command':cmd,'utc':now()})
        print(json.dumps({'run':str(run),'pid':child.pid,'smoke':smoke}));return
    run.mkdir(parents=True,exist_ok=True)
    lock=open(safe_path(run/'coordinator.lock'),'a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    require(not (run/'study.json').exists(),'Duplicate study')
    code=source_hashes(); runner_sha=digest(Path(__file__))
    origin=json.loads(safe_path(E/'origin.json').read_text())
    require(code==origin['source_sha256'],'Source freeze differs from origin')
    parentstudy=json.loads((PARENT/'study.json').read_text())
    require(prior.sources(services,PARENT_E/'c_new_anchor/src')==parentstudy['source_sha256'],'Parent source changed')
    preflight=json.loads(safe_path(E/'preflight.json').read_text())
    require(preflight['passed'] and preflight['source_sha256']==code,'Need exact-source new-loss tests')
    readiness=json.loads(safe_path(R/'runs/hierarchy_kd_v1/readiness_extended.json').read_text())
    require(readiness['passed'],'Readiness diagnostic required')
    cfg=dict(parentstudy['common_config'])
    cfg.update(w_mass_kd=.1,w_background_rejection=.1)
    if smoke:cfg.update(prior.SMOKE_OVERRIDES)
    inherited=prior.restoration(services)
    if not smoke:
        sm=U/'smoke/h2'
        require(json.loads((sm/'status.json').read_text())['status']=='complete','Real smoke must complete')
        smstudy=json.loads((sm/'study.json').read_text())
        require(smstudy['source_sha256']==code and smstudy['runner_sha256']==runner_sha,'Smoke freeze mismatch')
        rows=[json.loads(line) for line in (sm/'10-5/step2/hierarchy_metrics.jsonl').read_text().splitlines()]
        require(any(x['mass_pixels']>0 for x in rows) and any(x['background_pair_pixels']+x['absent_new_pixels']>0 for x in rows),'New objectives inactive in smoke')
    command=prior.command_for(run,cfg)
    atomic_json(run/'study.json',{'arm':'h2','created_utc':now(),'smoke':smoke,'source':str(S),
        'source_sha256':code,'runner_sha256':runner_sha,'common_config':cfg,'teacher':str(prior.TEACHER),
        'teacher_sha256':inherited['teacher_sha256'],'resume_checkpoint':str(prior.RESUME),
        'resume_checkpoint_sha256':inherited['resume_checkpoint_sha256'],'restoration':inherited,
        'new_stage1_training':False,'baseline_retrained':False,'step0_retrained':False,
        'global_batch':8,'train_gpus':list(range(8)),'resume_iteration':2000,
        'remaining_updates':cfg['max_iters']-2000,'routing':'past loss-evidence observer top3 old->BG+new and top3 BG->FG; rates rank only',
        'mechanism':'unchanged conditional KD and C SEP; loss-evidence-aligned routing, teacher old mass floor, trusted BG/absent-new rejection scaled by square-root ROI coverage',
        'GT_role':'development readiness diagnostics only; no pixel GT loss/gating input',
        'reference_miou':69.2772822002147,'limitations':'fixed seed adaptive development, no significance or independent-test claim'})
    env=environment('8card');env['CUDA_VISIBLE_DEVICES']='0,1,2,3,4,5,6,7'
    workers=[];train=None;stage=safe_path(run/'10-5/step2');stage.mkdir(parents=True,exist_ok=True)
    def frozen():
        require(source_hashes()==code and digest(Path(__file__))==runner_sha,'Active source changed')
    try:
        for rank in range(4):
            cmd=[str(PY),'-B',str(S/'evaluate_kd.py'),'--run',str(run),'--rank',str(rank),'--shards','4']
            with open(safe_path(run/f'evaluator_rank{rank}.log'),'x') as log:
                p=subprocess.Popen(cmd,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            workers.append(p)
            atomic_json(run/f'evaluator_rank{rank}.process.json',{'pid':p.pid,'command':cmd})
        with open(safe_path(stage/'launcher.log'),'x') as log:
            train=subprocess.Popen(command,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        atomic_json(stage/'launch.json',{'pid':train.pid,'command':command,'utc':now()})
        atomic_json(run/'status.json',{'status':'training','utc':now()})
        while train.poll() is None:
            frozen()
            require(all(p.poll() is None for p in workers),'Evaluator stopped during training')
            require(not (run/'evaluation_failed.json').exists(),'Evaluation failed')
            time.sleep(5)
        require(train.returncode==0,'Training failed: '+str(train.returncode));frozen()
        final=stage/'checkpoints/model_final.pth'
        atomic_json(stage/'training_complete.json',{'returncode':0,'checkpoint_sha256':digest(final),'utc':now()})
        atomic_json(run/'training_complete.json',{'status':'complete','stages_trained':1,'utc':now()})
        atomic_json(run/'status.json',{'status':'finishing_evaluation','utc':now()})
        deadline=time.monotonic()+600
        while any(p.poll() is None for p in workers):
            require(time.monotonic()<deadline and all(p.poll() in (None,0) for p in workers),'Evaluation failed or timed out')
            time.sleep(5)
        require(all(p.returncode==0 for p in workers),'Evaluator exit failed')
        prior.evaluation_complete(run,smoke,services);frozen()
        atomic_json(run/'evaluation_workers_complete.json',{'returncodes':[0]*4})
        atomic_json(run/'status.json',{'status':'complete','utc':now()})
        print(json.dumps({'status':'complete','run':str(run)}),flush=True)
    except BaseException as exc:
        prior.stop_owned([train,*workers])
        atomic_json(run/'status.json',{'status':'failed','exception':repr(exc),'utc':now()})
        raise
    finally:lock.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--launch',action='store_true')
    args=p.parse_args();main(args.smoke,args.launch)
