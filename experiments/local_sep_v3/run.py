from pathlib import Path
import sys,os,json,subprocess,argparse,time,ast
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto');E=R/'experiments/local_sep_v3';U=R/'runs/local_sep_v3';S=E/'src'
sys.path.insert(0,str(R/'experiments/kd_pixel_v2'))
from run_kd import environment
env=environment('8card');os.environ.update(env)
from kd_runtime import safe_path,atomic_json,digest,now
PY=R/'.runtime/env/bin/python'
OPT=R/'runs/kd_parallel_v1/formal/b_relational'
RESUME=R/'runs/ald_calibration_v9/formal/10-5/step2/checkpoints/model_final.pth'
TEACHER=OPT/'10-5/step1/checkpoints/model_final.pth'
BEST=R/'runs/pair_preserving_kd_v1/formal/10-5/step2/checkpoints/model_final.pth'

def sources():return {str(p.relative_to(S)):digest(p) for p in S.rglob('*.py') if '__pycache__' not in p.parts}

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['test','smoke','formal']);p.add_argument('--worker',action='store_true');a=p.parse_args()
    origin=json.loads(safe_path(E/'origin.json').read_text());assert sources()==origin['source_sha256']
    assert digest(S/'model/ald.py')==origin['ald_sha256']
    if a.mode=='test':
        cmd=[str(PY),'-B','-m','torch.distributed.run','--standalone','--nproc_per_node=8',str(E/'test_local.py')]
        with open(safe_path(U/'preflight.log'),'x') as log:r=subprocess.run(cmd,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT)
        print(json.dumps({'returncode':r.returncode,'log':str(U/'preflight.log')}));sys.exit(r.returncode)
    run=safe_path(U/a.mode);stage=safe_path(run/'10-5/step2')
    if not a.worker:
        readiness=json.loads(safe_path(U/'readiness.json').read_text());assert readiness['passed'] and readiness['ALD_sha256']==digest(S/'model/ald.py')
        pre=json.loads(safe_path(E/'preflight.json').read_text());assert pre['passed'] and pre['source_sha256']==sources()
        assert pre['test_sha256']==digest(E/'test_local.py')
        if a.mode=='formal':
            smoke=json.loads(safe_path(U/'smoke/status.json').read_text());assert smoke['status']=='complete'
            st=json.loads(safe_path(U/'smoke/study.json').read_text());assert st['source_sha256']==sources() and st['runner_sha256']==digest(E/'run.py')
            assert all(json.loads(safe_path(U/'smoke/10-5/step2'/f'ald_smoke_rank{i}.json').read_text())['passed'] for i in range(8))
        assert digest(safe_path(RESUME))=='8330ec6a7ae80aaa224c0c63f282a3e12462a38ac25e6ff448a3416d788822c8'
        assert digest(safe_path(TEACHER))=='6d3d54769b6a54e1700c7cd352b24e4bb95f4ccf398bc200b74f60795780ab33'
        assert digest(safe_path(BEST))=='3f43207d665ce6001aa149867435364e13baca5d47a0e2274dfb9187975dd28d'
        import torch
        ck=torch.load(RESUME,map_location='cpu',weights_only=True,mmap=True)
        assert ck['iteration']==8600 and len(ck['optimizer_state']['state'])==163
        assert set(int(v['step']) for v in ck['optimizer_state']['state'].values())=={8300,8600}
        assert ck.get('online_confusion_state') is not None  # Archived for provenance; new source resets it.
        del ck
        cfg=json.loads(safe_path(R/'runs/prototype_sep_v1/formal/c_new_anchor/10-5/step2/config.json').read_text())
        for k in ['local_rank','ckpt_dir','pred_dir']:cfg.pop(k,None)
        cfg.update(work_dir=str(run),spg=4,num_workers=4,pretrained=False,ald_stride=1,
            ald_reference=str(BEST),ald_evidence=str(R/'experiments/ald_calibration_v8/image_evidence.pth'),ald_smoke=a.mode=='smoke',
            log_iters=4 if a.mode=='smoke' else 100,max_iters=8608 if a.mode=='smoke' else 8900,
            eval_iters=8608 if a.mode=='smoke' else 8900,val_limit=16 if a.mode=='smoke' else 0,
            train_limit=0,resume_checkpoint=str(RESUME),prev_checkpoint=str(TEACHER),async_eval=True,lr=1e-3,w_geometry_sep=0.)
        # Minimum evidence timing scales by exposure; selectors still count batches.
        cfg.update(pair_min_updates=25,pair_ramp_updates=50,pair_max_stale_updates=50,pair_refresh_interval=48)
        assert not (run/'study.json').exists();stage.mkdir(parents=True,exist_ok=True)
        atomic_json(run/'study.json',{'utc':now(),'source_sha256':sources(),'runner_sha256':digest(E/'run.py'),
            'config':cfg,'origin':origin,'resume_checkpoint':str(RESUME),'resume_sha256':digest(RESUME),
            'reference_checkpoint':str(BEST),'reference_sha256':digest(BEST),'teacher_sha256':digest(TEACHER),'SEP_reference':str(RESUME),'boundary_source_sha256':digest(E/'boundary_sep.py'),
            'global_batch':32,'actual_updates':8 if a.mode=='smoke' else 300,
            'remaining_sample_exposures':256 if a.mode=='smoke' else 9600,
            'scheduler':'300 new local pair-adapter updates from best8600, cosine adapterLR1e-3 with fresh Adam. Every original model tensor and original Adam state retained unchanged. Data sampler RNG restarted.',
            'ALD':'Confirmed old/image evidence plus reference agreement, current-new PAR/reference agreement and teacher/reference BG agreement form hard labels. Other valid pixels remain unknown. Unknowns receive allowed-set absence constraints and foreground-conditional relations, never a prescribed foreground mass. New PTC conflicts ignored; online confusion new anchors remain CAM/PAR independent of reference agreement. Existing explicit supported-old conditional KD retained.',
            'confusion':'Inherited observer under identical anchors, fresh current-new-source directed selector. Rates select only. Local directed evidence selects supported actual competitors from a frozen hard cohort, main/aux CAM consensus, capped log-odds increase log2; SEPweight .1, reference retention weight1. A new local pair adapter is the only trainable module; native FG pair mass and BG decision preserved. Counts are image exposures.',
            'protocol':'Only current-new image tags enter training; GT diagnostic only. Fixed-seed adaptive additional training, not statistical significance or isolated causality.',
            'readiness':readiness})
        cmd=[str(PY),'-B',str(E/'run.py'),a.mode,'--worker']
        with open(safe_path(run/'coordinator.log'),'x') as log:child=subprocess.Popen(cmd,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        atomic_json(run/'process.json',{'pid':child.pid,'starttime':Path(f'/proc/{child.pid}/stat').read_text().split()[21],'command':cmd,'utc':now()})
        print(json.dumps({'pid':child.pid,'run':str(run)}));return
    study=json.loads(safe_path(run/'study.json').read_text());assert study['source_sha256']==sources() and study['runner_sha256']==digest(E/'run.py')
    cfg=study['config'];script=S/'scripts/dist_train_voc_seg_neg.py'
    opts=set()
    for node in ast.walk(ast.parse(script.read_text())):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='add_argument':
            opts.update(arg.value for arg in node.args if isinstance(arg,ast.Constant) and isinstance(arg.value,str))
    cmd=[str(PY),'-B','-m','torch.distributed.run','--standalone','--nproc_per_node=8',str(script)]
    for k,v in cfg.items():
        assert '--'+k in opts,k
        if isinstance(v,bool):
            if v:cmd.append('--'+k)
            elif k=='pretrained':cmd.append('--no-pretrained')
        elif isinstance(v,(list,tuple)):cmd+=['--'+k,*map(str,v)]
        elif v is not None:cmd+=['--'+k,str(v)]
    started=time.monotonic();atomic_json(run/'status.json',{'status':'training','utc':now()})
    try:
        with open(safe_path(stage/'launcher.log'),'x') as log:
            child=subprocess.Popen(cmd,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT)
            atomic_json(stage/'launch.json',{'pid':child.pid,'command':cmd,'utc':now()})
            rc=child.wait()
        assert rc==0,f'Training exit {rc}'
        assert sources()==study['source_sha256']
        final=safe_path(stage/'checkpoints/model_final.pth');assert final.is_file()
        atomic_json(run/'training_complete.json',{'status':'complete','checkpoint':str(final),'checkpoint_sha256':digest(final),'actual_updates':study['actual_updates'],'elapsed_seconds':time.monotonic()-started,'utc':now()})
        atomic_json(run/'status.json',{'status':'evaluating','utc':now()})
        workers=[]
        for rank in range(8):
            cmd=[str(PY),'-B',str(S/'evaluate_kd.py'),'--run',str(run),'--rank',str(rank),'--shards','8','--once']
            with open(safe_path(run/f'evaluator{rank}.log'),'x') as log:w=subprocess.Popen(cmd,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT)
            workers.append(w)
        codes=[w.wait() for w in workers];assert codes==[0]*8,codes
        jobs=sorted((run/'eval_queue').glob('*.json'));assert len(jobs)==1
        # Rank0 can finish its once-pass before another shard is published.
        # Merge again after every worker has exited successfully.
        sys.path.insert(0,str(S))
        from evaluate_kd import merge
        for job in jobs:merge(json.loads(job.read_text()),run/'evaluations'/job.stem,8)
        results=[]
        for job in jobs:
            item=json.loads(job.read_text());folder=run/'evaluations'/job.stem
            result=json.loads((folder/'result.json').read_text())
            parts=[json.loads((folder/f'rank{i}.json').read_text()) for i in range(8)]
            names=[n for part in parts for n in part['images']]
            assert len(names)==len(set(names))==result['images']==(16 if a.mode=='smoke' else 1449)
            assert result['checkpoint_sha256']==item['checkpoint_sha256']
            results.append({'iteration':result['iteration'],'all_miou':result['all_miou'],'checkpoint_sha256':result['checkpoint_sha256']})
        atomic_json(run/'status.json',{'status':'complete','results':results,'elapsed_seconds':time.monotonic()-started,'utc':now()})
        print(json.dumps(results),flush=True)
    except BaseException as ex:
        atomic_json(run/'status.json',{'status':'failed','error':repr(ex),'utc':now()});raise

if __name__=='__main__':main()
