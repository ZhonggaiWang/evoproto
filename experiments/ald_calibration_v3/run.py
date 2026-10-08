from pathlib import Path
import sys,os,json,subprocess,argparse
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto');E=R/'experiments/ald_calibration_v3';U=R/'runs/ald_calibration_v3'
S=R/'experiments/prototype_sep_v1/c_new_anchor/src';C=R/'runs/prototype_sep_v1/formal/c_new_anchor'
sys.path.insert(0,str(R/'experiments/kd_pixel_v2'))
from run_kd import environment
env=environment('8card');os.environ.update(env)
from kd_runtime import safe_path,atomic_json,digest,now

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['test','smoke','formal','eval']);args=p.parse_args()
    if args.mode=='test':
        cmd=[str(R/'.runtime/env/bin/python'),'-B','-m','torch.distributed.run','--standalone','--nproc_per_node=8',str(E/'test_ald.py')]
        with open(safe_path(U/'preflight.log'),'x') as log:
            child=subprocess.Popen(cmd,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        atomic_json(U/'preflight_process.json',{'pid':child.pid,'command':cmd});print(child.pid);return
    run=safe_path(U/('smoke' if args.mode=='smoke' else 'formal'))
    if args.mode=='eval':
        done=json.loads(safe_path(run/'training_complete.json').read_text());assert done['status']=='complete' and done['steps']==300
        assert digest(safe_path(done['checkpoint']))==done['checkpoint_sha256']
        assert not (run/'evaluation_processes.json').exists();workers=[]
        for rank in range(8):
            cmd=[str(R/'.runtime/env/bin/python'),'-B',str(S/'evaluate_kd.py'),'--run',str(run),'--rank',str(rank),'--shards','8','--once']
            with open(safe_path(run/f'evaluator{rank}.log'),'x') as log:
                child=subprocess.Popen(cmd,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            workers.append({'rank':rank,'pid':child.pid,'command':cmd})
        atomic_json(run/'evaluation_processes.json',{'utc':now(),'workers':workers});print(json.dumps(workers));return
    sources={n:digest(E/n) for n in ['proposals.py','losses.py','train.py','run.py','cache_evidence.py']}
    pre=json.loads(safe_path(E/'preflight.json').read_text());assert pre['passed'] and all(digest(E/k)==v for k,v in pre['sources'].items())
    diag=json.loads(safe_path(U/'diagnostic.json').read_text());assert diag['passed'] and diag['proposals']['new_teacher_old']['precision']>=.8 and diag['proposals']['new_teacher_old']['net_correct']>500
    receipts=[json.loads(safe_path(U/f'cache_receipt_rank{i}.json').read_text()) for i in range(8)]
    assert all(r['passed'] for r in receipts)
    names=[n for r in receipts for n in r['images']];assert len(names)==len(set(names))==2145
    validation=[n for i in range(8) for n in json.loads((U/f'diagnostic_rank{i}.json').read_text())['images']]
    assert not set(names)&set(validation)
    if args.mode=='formal':
        smoke=json.loads(safe_path(U/'smoke/training_complete.json').read_text());assert smoke['status']=='complete' and smoke['all_frozen_tensors_exact'] and smoke['all_rank_parameters_exact']
        assert json.loads((U/'smoke/study.json').read_text())['sources']==sources
    run.mkdir(parents=True,exist_ok=True);assert not (run/'study.json').exists()
    parent=json.loads(safe_path(C/'study.json').read_text());assert {str(p.relative_to(S)):digest(p) for p in S.rglob('*.py') if '__pycache__' not in p.parts}==parent['source_sha256']
    source=R/'runs/pair_preserving_kd_v1/formal/10-5/step2/checkpoints/model_final.pth';sha=digest(safe_path(source));assert sha=='3f43207d665ce6001aa149867435364e13baca5d47a0e2274dfb9187975dd28d'
    cfg=json.loads(safe_path(C/'10-5/step2/config.json').read_text());steps=8 if args.mode=='smoke' else 300
    cfg.update(ald=True,ald_mode='evidence_calibrated_partial_targets',max_iters=8300+steps,spg=8,work_dir=str(run),ckpt_dir=str(run/'10-5/step2/checkpoints'),pred_dir=str(run/'10-5/step2/predictions'))
    atomic_json(run/'study.json',{'created_utc':now(),'sources':sources,'resume':str(source),'resume_sha256':sha,'parent_source_sha256':parent['source_sha256'],
        'updates':steps,'config':cfg,'global_batch':64,'gpus':list(range(8)),'optimizer_restored':True,
        'weights':{'full_reference_KL':1,'calibrated_pair_mass_swap':.1,'trusted_old_KD':.1,'classifier_updates':0},
        'training':'Frozen encoder and decoder body; train three segmentation output heads; carry calibrated classifier heads and optimizer states from ALD v2 unchanged. 300 updates, LR2e-5 cosine to zero. No GT pixels or old-class tags in optimizer.',
        'calibration':'Refresh CAM/PAR with ALD-v2 calibrated classifiers. Use refreshed new-CAM/PAR disagreements only when teacher is old foreground; teacher-BG/new corrections rejected by GT diagnosis. Also allow sparse 2-view BG + low-CAM evidence. Swap candidate/winner mass, preserve all other classes; veto conflicting old KD. GT remains diagnostic only.',
        'confusion':'Inherited observer and selector preserved for provenance but not applied or updated. The previously unreliable remaining pair cannot drive this ALD update.',
        'data':'2145 unique training images; 1449 disjoint validation images only used for diagnosis and final evaluation',
        'comparison':'Additional training budget; no repeated baseline or step0; no seed selection or parameter sweep'})
    stage=safe_path(run/'10-5/step2');stage.mkdir(parents=True,exist_ok=True);atomic_json(stage/'config.json',cfg)
    cmd=[str(R/'.runtime/env/bin/python'),'-B','-m','torch.distributed.run','--standalone','--nproc_per_node=8',str(E/'train.py')]+(['--smoke'] if args.mode=='smoke' else [])
    with open(safe_path(run/'launcher.log'),'x') as log:child=subprocess.Popen(cmd,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    atomic_json(run/'process.json',{'pid':child.pid,'command':cmd,'utc':now()});print(json.dumps({'pid':child.pid,'run':str(run)}))

if __name__=='__main__':main()
