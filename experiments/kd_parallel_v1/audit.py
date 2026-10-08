"""Completion evidence, with missing endpoint work explicitly marked incomplete."""
from pathlib import Path
import json,sys,os
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto');E=R/'experiments/kd_parallel_v1';U=R/'runs/kd_parallel_v1'
sys.path.insert(0,str(R/'experiments/kd_pixel_v2/src'))
from kd_runtime import atomic_json,safe_path,digest,now

def main():
 os.chdir(R)
 origin=json.loads((R/'experiments/kd_pixel_v2/origin.json').read_text())
 step0_ok=digest(Path(origin['step0']))==origin['step0_sha256']
 archive_ok=digest(R/'runs/fixed_baseline_v1/source_snapshot.tar.gz')==origin['source_archive_sha256']
 arms={}
 for arm in ['a_sigmoid','b_relational']:
  run=U/'formal'/arm;manifest=json.loads((run/'study.json').read_text());S=E/arm/'src'
  changed=[name for name,sha in origin['original_source_hashes'].items() if (S/name).exists() and digest(S/name)!=sha]
  expected={'continual/Trainer.py','model/losses.py','model/model_seg_neg.py','scripts/dist_train_voc_seg_neg.py'}
  source_ok=all(digest(S/name)==sha for name,sha in manifest['source_sha256'].items())
  for p in [run,E/arm,*S.rglob('*')]:safe_path(p)
  stages=[]
  for step in [1,2]:
   stage=run/f'10-5/step{step}';launch=stage/'launch.json';result=run/f'evaluations/step{step}_iter8000/result.json'
   info={'step':step,'launched':launch.exists(),'final_evaluated':result.exists(),'training_exit0':False,'final_checkpoint_verified':False,'expected_teacher_verified':False}
   if launch.exists():
    record=json.loads(launch.read_text());info['training_exit0']=record.get('returncode')==0
    cfg=json.loads((stage/'config.json').read_text())
    info['config']={k:cfg[k] for k in ['step','seed','max_iters','spg','w_proto_kd','w_proto_sep','w_pixel_kd','kd_temperature','ald','confusion_reweight']}
    info['unchanged_global_batch']=cfg['spg']*len(manifest['gpus'])==8
    base_cfg=json.loads((R/f'runs/fixed_baseline_v1/base/10-5/step{step}/config.json').read_text())
    common_keys=['seed','max_iters','lr','betas','power','scales','crop_size','backbone','w_seg','w_proto_seg','loss_warmup_iters','warmup_iters','cam_scales','high_thre','low_thre','bkg_thre','wt_decay','optimizer','train_limit','val_limit']
    info['common_hyperparameter_mismatches']={k:[base_cfg.get(k),cfg.get(k)] for k in common_keys if base_cfg.get(k)!=cfg.get(k)}
    info['common_hyperparameters_verified']=not info['common_hyperparameter_mismatches']
    expected_teacher=Path(origin['step0']) if step==1 else run/'10-5/step1/checkpoints/model_final.pth'
    info['expected_teacher_verified']=Path(cfg['prev_checkpoint'])==expected_teacher and str(expected_teacher) in (stage/'train.log').read_text()
    if step==1:
     info['common_warmup_restored']='Resume student AND optimizer at iteration 2000' in (stage/'train.log').read_text()
     info['common_warmup_hash_valid']=digest(Path(manifest['shared_warmup_checkpoint']))==manifest['shared_warmup_sha256']
    complete=stage/'training_complete.json'
    if complete.exists():
     d=json.loads(complete.read_text());info['final_checkpoint_verified']=digest(stage/'checkpoints/model_final.pth')==d['checkpoint_sha256']
    if result.exists():
     d=json.loads(result.read_text());q=json.loads((run/f'eval_queue/step{step}_iter8000.json').read_text())
     hist=d['histogram'];n=16 if step==1 else 21
     info['histogram_shape_valid']=len(hist)==n and all(len(row)==n for row in hist)
     baseline_hist=json.loads((R/f'runs/kd_pixel_v1/evaluations/baseline_base_step{step}/result.json').read_text())['histogram']
     info['validation_GT_row_totals_match_reference']=[sum(row) for row in hist]==[sum(row) for row in baseline_hist]
     info['evaluation_images']=d['images'];info['all_validation_images']=d['images']==(1240 if step==1 else 1449)
     info['evaluated_checkpoint_verified']=digest(Path(q['checkpoint']))==q['checkpoint_sha256']==d['checkpoint_sha256']
     info['all_miou']=d['all_miou']
     identity=stage/'endpoint_identity.json'
     if identity.exists() and complete.exists():
      receipt=json.loads(identity.read_text())
      info['final_matches_evaluated_model']=receipt['all_model_tensors_exactly_equal'] and receipt['final_sha256']==json.loads(complete.read_text())['checkpoint_sha256'] and receipt['evaluated_sha256']==d['checkpoint_sha256']
    kd=stage/'kd_metrics.jsonl'
    if kd.exists():
     rows=[json.loads(x) for x in kd.read_text().splitlines()];rows=[x for x in rows if x['active']]
     if rows:
      total=sum(x['valid_pixels'] for x in rows)
      info['kd_coverage']={'logged_active_batches':len(rows),'nonzero_batches':sum(x['kd_nonzero_pixels']>0 for x in rows),
       'nonzero_pixel_fraction':sum(x['kd_nonzero_pixels'] for x in rows)/total,
       'eligible_old_class_ids':[i for i in range(1,len(rows[0]['class_eligible_pixels'])) if any(x['class_eligible_pixels'][i]>0 for x in rows)],
       'note':'Eligibility counts are region evidence, not per-class gradient norms; real gradient probes are recorded separately'}
   required=['training_exit0','final_evaluated','final_checkpoint_verified','expected_teacher_verified','all_validation_images','evaluated_checkpoint_verified','histogram_shape_valid','unchanged_global_batch','common_hyperparameters_verified','final_matches_evaluated_model','validation_GT_row_totals_match_reference']
   if step==1:required+=['common_warmup_restored','common_warmup_hash_valid']
   info['stage_complete']=all(info.get(k) is True for k in required)
   stages.append(info)
  arms[arm]={'frozen_source_verified':source_ok,'modified_baseline_files':changed,'only_expected_baseline_files_modified':set(changed)<=expected,
    'baseline_dataset_and_augmentation_files_unchanged':not any(x.startswith(('datasets/','utils/imutils.py','utils/camutils.py')) for x in changed),
    'stages':stages,'complete':source_ok and all(x['stage_complete'] for x in stages)}
 result={'utc':now(),'shared_step0_unchanged':step0_ok,'archived_baseline_source_unchanged':archive_ok,'arms':arms,
  'all_training_and_endpoint_requirements_complete':step0_ok and archive_ok and all(a['complete'] for a in arms.values()),
  'method_verification':str(E/'verification.json'),'cam_equivalence':str(E/'cam_equivalence.json'),
  'goal_amendment':str(U/'goal_amendment.json'),'resource_policy':'8card only; no 4card work',
  'interpretation':'This audit verifies experiment completeness and integrity, not that an accuracy improvement has been achieved.'}
 atomic_json(U/'completion_audit.json',result)
 print(json.dumps(result))

if __name__=='__main__':main()
