from pathlib import Path
import sys,os,json,subprocess,argparse
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto');E=R/'experiments/pair_preserving_kd_v1';U=R/'runs/pair_preserving_kd_v1'
C=R/'runs/prototype_sep_v1/formal/c_new_anchor';S=R/'experiments/prototype_sep_v1/c_new_anchor/src'
sys.path.insert(0,str(R/'experiments/kd_pixel_v2'))
from run_kd import environment
env=environment('8card');os.environ.update(env)
from kd_runtime import safe_path,atomic_json,digest,now

def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');args=p.parse_args()
    run=safe_path(U/('smoke' if args.smoke else 'formal'));run.mkdir(parents=True,exist_ok=True)
    assert not (run/'study.json').exists(),'Do not duplicate an existing refinement'
    sources={name:digest(E/name) for name in ['pair_targets.py','pair_loss.py','train_head.py','run_refinement.py']}
    pf=json.loads(safe_path(E/'preflight.json').read_text());assert pf['passed']
    assert all(digest(E/k)==v for k,v in pf['sources'].items())
    parent=json.loads(safe_path(C/'study.json').read_text())
    assert {str(p.relative_to(S)):digest(p) for p in S.rglob('*.py') if '__pycache__' not in p.parts}==parent['source_sha256']
    readiness=json.loads(safe_path(U/'readiness.json').read_text());assert readiness['passed']
    receipts=[json.loads(safe_path(U/f'cache_receipt_rank{i}.json').read_text()) for i in range(8)]
    assert all(rec['passed'] and rec['student_sha256']==readiness['student_sha256'] for rec in receipts)
    if not args.smoke:
        smoke=json.loads(safe_path(U/'smoke/training_complete.json').read_text())
        assert smoke['status']=='complete' and smoke['all_frozen_tensors_exact'] and smoke['totals']['active_updates']>0
        assert json.loads(safe_path(U/'smoke/study.json').read_text())['sources']==sources
    cfg=json.loads(safe_path(C/'10-5/step2/config.json').read_text());steps=8 if args.smoke else 300
    cfg.update(max_iters=8000+steps,spg=8,work_dir=str(run),ckpt_dir=str(run/'10-5/step2/checkpoints'),pred_dir=str(run/'10-5/step2/predictions'))
    atomic_json(run/'study.json',{'created_utc':now(),'sources':sources,'parent_source':str(S),
        'parent_source_sha256':parent['source_sha256'],'resume':str(C/'10-5/step2/checkpoints/model_final.pth'),
        'resume_sha256':readiness['student_sha256'],'teacher_sha256':readiness['teacher_sha256'],
        'smoke':args.smoke,'updates':steps,'config':cfg,'global_batch':64,'gpus':list(range(8)),
        'head_only':True,'optimizer_restored':True,'retention_weight':1.,'correction_weight':.1,'old_KD_weight':.1,
        'lr_schedule':'original backbone LR 2e-5, cosine to zero over 300 steps; 1/10 original head LR',
        'lineage':'best C8000 + additional head-only updates; NOT equal training budget; no baseline or step0 retraining',
        'evidence':'frozen C CAM/PAR and old optimized teacher; online directed ranking only, binary correction gates',
        'data':'all2145 stage2 training images, fixed resize448 cachedfloat32, no pixel GT or validation data in optimization'})
    stage=safe_path(run/'10-5/step2');stage.mkdir(parents=True,exist_ok=True);atomic_json(stage/'config.json',cfg)
    command=[str(R/'.runtime/env/bin/python'),'-B','-m','torch.distributed.run','--standalone','--nproc_per_node=8',str(E/'train_head.py')]+(['--smoke'] if args.smoke else [])
    with open(safe_path(run/'launcher.log'),'x') as log:
        child=subprocess.Popen(command,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    atomic_json(run/'process.json',{'pid':child.pid,'command':command,'utc':now()});print(json.dumps({'pid':child.pid,'run':str(run)}))

if __name__=='__main__':main()
