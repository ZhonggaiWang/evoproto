from pathlib import Path
import sys,os,json
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto');E=R/'experiments/ald_calibration_v9';U=R/'runs/ald_calibration_v9';RUN=U/'formal';S=E/'src'
sys.path.insert(0,str(R/'experiments/kd_pixel_v2'))
from run_kd import environment
os.environ.update(environment('8card'))
from kd_runtime import safe_path,atomic_json,digest,now
import torch,numpy as np
read=lambda p:json.loads(safe_path(p).read_text())
study=read(RUN/'study.json');done=read(RUN/'training_complete.json');assert read(RUN/'status.json')['status']=='complete'
assert {str(p.relative_to(S)):digest(p) for p in S.rglob('*.py') if '__pycache__' not in p.parts}==study['source_sha256']
assert digest(E/'run.py')==study['runner_sha256']
result=read(RUN/'evaluations/step2_iter8600/result.json');assert result['images']==1449
job=read(RUN/'eval_queue/step2_iter8600.json');assert digest(job['checkpoint'])==result['checkpoint_sha256']==job['checkpoint_sha256']
final=torch.load(done['checkpoint'],map_location='cpu',weights_only=True,mmap=True)
evaluated=torch.load(job['checkpoint'],map_location='cpu',weights_only=True,mmap=True)
parent=torch.load(study['resume_checkpoint'],map_location='cpu',weights_only=True,mmap=True)
assert final['iteration']==evaluated['iteration']==8600 and digest(done['checkpoint'])==done['checkpoint_sha256']
assert all(torch.equal(v,evaluated['model_state'][k]) for k,v in final['model_state'].items())
changed=[k for k,v in final['model_state'].items() if not torch.equal(v,parent['model_state'][k])]
for k,v in parent['optimizer_state']['state'].items():assert int(final['optimizer_state']['state'][k]['step'])==int(v['step'])+300
assert int(final['online_confusion_state']['updates'])==300 and study['config']['w_geometry_sep']==0
assert digest(study['reference_checkpoint'])==study['reference_sha256']=='3f43207d665ce6001aa149867435364e13baca5d47a0e2274dfb9187975dd28d'
audit=read(U/'audit/evaluations/step2_iter8600/result.json');assert audit['images']==1449 and audit['histogram']==result['histogram']
parts=[read(U/'audit/evaluations/step2_iter8600'/f'rank{i}.json') for i in range(8)];names=[n for d in parts for n in d['images']]
assert len(names)==len(set(names))==1449 and all(p['checkpoint_sha256']==done['checkpoint_sha256'] for p in parts)
classification={}
for key in ['reference','candidate']:
    h=np.sum([np.asarray(p['classification'][key]) for p in parts],axis=0);classification[key]={}
    for name,sl in [('old',slice(0,15)),('new',slice(15,20))]:
        tn,fp,fn,tp=h[sl].sum(0).flatten();classification[key][name]={'precision':float(tp/max(tp+fp,1)*100),'recall':float(tp/max(tp+fn,1)*100),'TP':int(tp),'FP':int(fp),'FN':int(fn),'TN':int(tn)}
def metrics(r):
    h=np.asarray(r['histogram']);d=h.diagonal();old=slice(1,16);new=slice(16,21)
    return {'all_miou':r['all_miou'],'old_miou':r['previous_foreground_miou'],'new_miou':r['current_foreground_miou'],
        'old_precision':float(d[old].sum()/h[:,old].sum()*100),'old_recall':float(d[old].sum()/h[old].sum()*100),
        'new_precision':float(d[new].sum()/h[:,new].sum()*100),'new_recall':float(d[new].sum()/h[new].sum()*100),
        'BG_to_new':int(h[0,new].sum()),'new_to_BG':int(h[new,0].sum()),'BG_to_old':int(h[0,old].sum()),'old_to_BG':int(h[old,0].sum()),
        'old_to_new':int(h[old,new].sum()),'new_to_old':int(h[new,old].sum()),'class_iou':r['class_iou']}
reference=read(R/'runs/pair_preserving_kd_v1/formal/evaluations/step2_iter8300/result.json');m=metrics(result);r=metrics(reference)
pre=read(E/'preflight.json');ready=read(U/'readiness.json');assert pre['passed'] and ready['passed']
assert ready['ALD_sha256']==digest(E/'ald.py') and pre['source_sha256']==study['source_sha256']
analysis={'status':'complete_endpoint_analysis','utc':now(),'candidate':m,'reference':r,'delta':{k:m[k]-r[k] for k in m if k!='class_iou'},
    'class_iou_delta':{k:m['class_iou'][k]-r['class_iou'][k] for k in m['class_iou']},'classification':classification,
    'checkpoint':done['checkpoint'],'checkpoint_sha256':done['checkpoint_sha256'],'evaluated_checkpoint_sha256':result['checkpoint_sha256'],
    'actual_updates':300,'global_batch':32,'training_sample_exposures':9600,'training_seconds':done['elapsed_seconds'],
    'changed_model_tensors':changed,'readiness_fixed_best_checkpoint':ready,'preflight':pre,
    'verified':{'full1449_unique':True,'final_equals_evaluated_model':True,'classification_audit_histogram_matches_endpoint':True,'best_reference_unchanged':True,'Adam_parameterwise_states_restored_and_advanced300':True,'fresh_confusion300updates':True,'geometry_SEP_not_strengthened':True},
    'target_exceeded':m['all_miou']>r['all_miou'],
    'limits':['Fixed-seed adaptive development, not statistical significance or an isolated ALD causal comparison.',
              '300 additional full-model updates; encoder2e-6/head2e-5 cosine; earlier SEP weights inherited but extra geometry SEP disabled.',
              'Current-best reference is frozen; old image states frozen, augmented predictions and CAM/PAR online. GT diagnostic only.',
              'Foreground-conditional term has zero BG-logit gradient and common-FG-shift invariance. Shared-parameter updates do not guarantee exact foreground probability preservation elsewhere.',
              'Readiness pseudo-label accuracy was measured before optimization on the fixed best model; it is not the final candidate pseudo-label accuracy.',
              'Confusion counts image exposures, not independent unique images; new row anchors deliberately remain independent CAM/PAR signals.']}
atomic_json(U/'analysis.json',analysis);print(json.dumps({k:analysis[k] for k in ['candidate','delta','classification','target_exceeded']},indent=2))
