"""One stage2 continuation, frozen teacher/warmup, eight GPU training and four evaluators."""
from pathlib import Path
import sys,os,json,subprocess,time,signal,fcntl,argparse
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto')
E=R/'experiments/hierarchy_kd_v3'; U=R/'runs/hierarchy_kd_v3'; S=E/'src'
PARENT_E=R/'experiments/prototype_sep_v1'
PARENT=R/'runs/prototype_sep_v1/formal/c_new_anchor'
PY=R/'.runtime/env/bin/python'
sys.path.insert(0,str(PARENT_E))
import run_new_anchor_sep as prior
prior.S=S
H2=R/'runs/hierarchy_kd_v2/formal/h2'
prior.RESUME=H2/'10-5/step2/checkpoints/model_iter_4000.pth'
services=prior.initialize()
environment,safe_path,atomic_json,now,digest=services
require=prior.require

def source_hashes():
    return prior.sources(services,S)

def restoration():
    require(json.loads((H2/'status.json').read_text())['status']=='complete','H2 must finish first')
    endpoint=json.loads((H2/'evaluations/step2_iter8000/result.json').read_text())
    require(endpoint['all_miou']<69.2772822002147,'No failed H2 endpoint to motivate revision')
    parent=json.loads((H2/'study.json').read_text())
    require(prior.sources(services,Path(parent['source']))==parent['source_sha256'],'H2 source changed')
    job=json.loads((H2/'eval_queue/step2_iter4000.json').read_text())
    result=json.loads((H2/'evaluations/step2_iter4000/result.json').read_text())
    sha=digest(safe_path(prior.RESUME));teacher_sha=digest(safe_path(prior.TEACHER))
    require(sha==job['checkpoint_sha256']==result['checkpoint_sha256'] and result['images']==1449,'Invalid4000 checkpoint')
    require(teacher_sha==parent['teacher_sha256'],'Teacher changed')
    import torch
    saved=torch.load(prior.RESUME,map_location='cpu',weights_only=True,mmap=True)
    require(saved['iteration']==4000 and int(saved['online_confusion_state']['updates'])==2000
            and int(saved['hierarchy_observer_state']['updates'])==2000,'Incomplete inherited states')
    require(all(int(v['step'])==4000 for v in saved['optimizer_state']['state'].values() if 'step' in v),'Optimizer iteration mismatch')
    return {'iteration':4000,'teacher_sha256':teacher_sha,'resume_checkpoint_sha256':sha,
        'model_tensors':len(saved['model_state']),'optimizer_state_entries':len(saved['optimizer_state']['state']),
        'restored_states':['model','optimizer','online_confusion','geometry_selector','hierarchy_observer','hierarchy_totals'],
        'inherited_h2_run':str(H2),'inherited_h2_updates':2000,'shared_OPT_warmup_updates':2000}

def evaluation_complete(run,smoke):
    expected=['step2_iter4008'] if smoke else ['step2_iter6000','step2_iter8000']
    require(sorted(p.stem for p in (run/'eval_queue').glob('*.json'))==expected,'Unexpected evaluation queue')
    for name in expected:
        job=json.loads((run/'eval_queue'/f'{name}.json').read_text())
        folder=run/'evaluations'/name;result=json.loads((folder/'result.json').read_text())
        parts=[json.loads((folder/f'rank{i}.json').read_text()) for i in range(4)]
        names=[n for p in parts for n in p['images']]
        require(len(names)==len(set(names))==result['images']==(16 if smoke else 1449),'Incomplete validation')
        require(all(p['checkpoint_sha256']==job['checkpoint_sha256']==result['checkpoint_sha256'] for p in parts),'Shard checkpoint mismatch')
        merged=[[sum(p['histogram'][i][j] for p in parts) for j in range(21)] for i in range(21)]
        require(merged==result['histogram'],'Histogram merge mismatch')

def main(smoke=False,launch=False):
    run=safe_path(U/('smoke' if smoke else 'formal')/'h3')
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
    transfer_readiness=json.loads(safe_path(R/'runs/hierarchy_kd_v2/new_transfer_readiness.json').read_text())
    evidence=transfer_readiness['new_complement_GT']['transfer_deficit']['weights']
    require(transfer_readiness['passed'] and evidence['correct_new_source']/(sum(evidence.values())-evidence['void'])>.9,
            'Transfer direction lacks the observed development evidence')
    cfg=dict(parentstudy['common_config'])
    cfg.update(w_mass_kd=0.,w_background_rejection=.1,w_transfer_kd=.1,transfer_start_iteration=4000)
    if smoke:cfg.update(max_iters=4008,log_iters=1,eval_iters=4008,val_limit=16,train_limit=64,transfer_start_iteration=3700)
    inherited=restoration()
    if not smoke:
        sm=U/'smoke/h3'
        require(json.loads((sm/'status.json').read_text())['status']=='complete','Real smoke must complete')
        smstudy=json.loads((sm/'study.json').read_text())
        require(smstudy['source_sha256']==code and smstudy['runner_sha256']==runner_sha,'Smoke freeze mismatch')
        rows=[json.loads(line) for line in (sm/'10-5/step2/hierarchy_metrics.jsonl').read_text().splitlines()]
        require(any(x['transfer_pixels']>0 and x['transfer_ramp']>0 for x in rows),'Transfer objective inactive in smoke')
    command=prior.command_for(run,cfg)
    atomic_json(run/'study.json',{'arm':'h3','created_utc':now(),'smoke':smoke,'source':str(S),
        'source_sha256':code,'runner_sha256':runner_sha,'common_config':cfg,'teacher':str(prior.TEACHER),
        'teacher_sha256':inherited['teacher_sha256'],'resume_checkpoint':str(prior.RESUME),
        'resume_checkpoint_sha256':inherited['resume_checkpoint_sha256'],'restoration':inherited,
        'new_stage1_training':False,'baseline_retrained':False,'step0_retrained':False,
        'global_batch':8,'train_gpus':list(range(8)),'resume_iteration':4000,
        'remaining_updates':cfg['max_iters']-4000,'routing':'past trusted new-source geometry directions; teacher must predict the selected old competitor; rates rank only',
        'mechanism':'unchanged conditional KD and C SEP; unreliable old-mass floor disabled; foreground probability transferred from confused teacher old class to independently supported new class; inherited coverage-scaled background rejection',
        'lineage':'OPT warmup0..2000 + H2 updates2001..4000 + H3 updates4001..8000; adaptive curriculum, no baseline or step0 retraining',
        'transfer_evidence':str(R/'runs/hierarchy_kd_v2/new_transfer_readiness.json'),
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
        evaluation_complete(run,smoke);frozen()
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
