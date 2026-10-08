from pathlib import Path
import sys,os,json
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto');E=R/'experiments/local_sep_v4';U=R/'runs/local_sep_v4';RUN=U/'formal';S=E/'src'
sys.path.insert(0,str(R/'experiments/kd_pixel_v2'))
from run_kd import environment
os.environ.update(environment('8card'))
from kd_runtime import safe_path,atomic_json,digest,now
import torch,numpy as np
read=lambda p:json.loads(safe_path(p).read_text())
study=read(RUN/'study.json');done=read(RUN/'training_complete.json');assert read(RUN/'status.json')['status']=='complete'
assert {str(p.relative_to(S)):digest(p) for p in S.rglob('*.py') if '__pycache__' not in p.parts}==study['source_sha256']
assert digest(E/'run.py')==study['runner_sha256']
result=read(RUN/'evaluations/step2_iter9200/result.json');assert result['images']==1449
job=read(RUN/'eval_queue/step2_iter9200.json');assert digest(job['checkpoint'])==result['checkpoint_sha256']==job['checkpoint_sha256']
final=torch.load(done['checkpoint'],map_location='cpu',weights_only=True,mmap=True)
evaluated=torch.load(job['checkpoint'],map_location='cpu',weights_only=True,mmap=True)
parent=torch.load(study['resume_checkpoint'],map_location='cpu',weights_only=True,mmap=True)
original=torch.load(R/'runs/ald_calibration_v9/formal/10-5/step2/checkpoints/model_final.pth',map_location='cpu',weights_only=True,mmap=True)
assert final['iteration']==evaluated['iteration']==9200 and digest(done['checkpoint'])==done['checkpoint_sha256']
assert all(torch.equal(v,evaluated['model_state'][k]) for k,v in final['model_state'].items())
changed=[k for k,v in original['model_state'].items() if not torch.equal(v,final['model_state'][k])]
assert changed==[]
added=sorted(set(final['model_state'])-set(original['model_state']))
assert len(added)==3 and all(k.startswith('decoder.local_sep.') for k in added)
advanced=list(final['optimizer_state']['state'])
assert len(advanced)==3 and all(int(v['step'])==600 for v in final['optimizer_state']['state'].values())
for k,v in original['optimizer_state']['state'].items():
 for key,value in v.items():
  assert torch.equal(value,final['base_optimizer_state']['state'][k][key])
assert final['base_optimizer_state']['param_groups']==original['optimizer_state']['param_groups']
assert int(final['online_confusion_state']['updates'])-int(parent['online_confusion_state']['updates'])==300
assert digest(study['resume_checkpoint'])==study['resume_sha256']=='c4686142579257698e75e2e7ac15d5744a829826ac05ddf2a8032b18e86aab89'
parts=[read(RUN/'evaluations/step2_iter9200'/f'rank{i}.json') for i in range(8)]
names=sum([x['images'] for x in parts],[]);assert len(names)==len(set(names))==1449
def metrics(r):
 h=np.asarray(r['histogram']);d=h.diagonal();old=slice(1,16);new=slice(16,21)
 return {'all_miou':r['all_miou'],'old_miou':r['previous_foreground_miou'],'new_miou':r['current_foreground_miou'],
 'old_precision':float(d[old].sum()/h[:,old].sum()*100),'old_recall':float(d[old].sum()/h[old].sum()*100),
 'new_precision':float(d[new].sum()/h[:,new].sum()*100),'new_recall':float(d[new].sum()/h[new].sum()*100),
 'BG_to_new':int(h[0,new].sum()),'new_to_BG':int(h[new,0].sum()),'BG_to_old':int(h[0,old].sum()),'old_to_BG':int(h[old,0].sum()),
 'old_to_new':int(h[old,new].sum()),'new_to_old':int(h[new,old].sum()),'class_iou':r['class_iou']}
reference=read(R/'runs/ald_calibration_v9/formal/evaluations/step2_iter8600/result.json');m=metrics(result);r=metrics(reference)
logs=[json.loads(x) for x in (RUN/'10-5/step2/local_sep_metrics.jsonl').read_text().splitlines()]
assert int(final['local_sep_state']['updates'])==300
assert int(final['local_sep_state']['pixels'].sum())==logs[-1]['cumulative']['reference_cohort']
analysis={'status':'complete_endpoint_analysis','utc':now(),'candidate':m,'reference':r,'delta':{k:m[k]-r[k] for k in m if k!='class_iou'},
 'class_iou_delta':{k:m['class_iou'][k]-r['class_iou'][k] for k in m['class_iou']},
 'checkpoint':done['checkpoint'],'checkpoint_sha256':done['checkpoint_sha256'],'evaluated_checkpoint_sha256':result['checkpoint_sha256'],
 'actual_updates':300,'total_adapter_updates':600,'global_batch':32,'training_sample_exposures':9600,'training_seconds':done['elapsed_seconds'],
 'changed_model_tensors':changed,'added_adapter_tensors':added,'advanced_Adam_states':advanced,'readiness':read(U/'readiness.json'),'local_sep_logs':logs,
 'verified':{'full1449_unique':True,'final_equals_evaluated_model':True,'best_reference_unchanged':True,'every_original_model_and_optimizer_tensor_unchanged':True,'image_classifier_function_unchanged':True,'online_confusion_history_preserved':True},
 'target_exceeded':m['all_miou']>r['all_miou'],
 'limits':['Fixed-seed adaptive development, not statistical significance or isolated SEP attribution.','Inherited ALD/KD supervision plus reference retention accompany SEP.','No feature bank used. New-source weak anchors supply positive SEP; current-new image absence supplies reverse negative exclusion. Fixed original ALD-v9 reference, no GT class whitelist; validation GT audits only.','A new paired adapter redistributes two foreground logits. Native pair mass, nonpair logits and BG decisions are preserved; interpolation may change these guarantees at original GT resolution.','Initial cohort accuracy is at native28 grid before online support gating; endpoint metrics use original GT resolution. Training evidence counts repeated image exposures, not independent unique images.']}
atomic_json(U/'analysis.json',analysis)
print(json.dumps({k:analysis[k] for k in ['candidate','delta','class_iou_delta','target_exceeded']},indent=2))
