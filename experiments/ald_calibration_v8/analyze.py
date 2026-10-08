from pathlib import Path
import os,sys,json
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto');E=R/'experiments/ald_calibration_v8';U=R/'runs/ald_calibration_v8';RUN=U/'formal';S=E/'src'
sys.path.insert(0,str(R/'experiments/kd_pixel_v2'))
from run_kd import environment
os.environ.update(environment('8card'))
from kd_runtime import safe_path,atomic_json,digest,now
import torch,numpy as np
read=lambda p:json.loads(safe_path(p).read_text())
study=read(RUN/'study.json');done=read(RUN/'training_complete.json');assert read(RUN/'status.json')['status']=='complete'
assert {str(p.relative_to(S)):digest(p) for p in S.rglob('*.py') if '__pycache__' not in p.parts}==study['source_sha256']
assert digest(E/'run.py')==study['runner_sha256']
results=[read(RUN/f'evaluations/step2_iter{i}/result.json') for i in [4000,6000,8000]]
assert all(r['images']==1449 for r in results)
for result in results:
    folder=RUN/f"evaluations/step2_iter{result['iteration']}"
    parts=[read(folder/f'rank{i}.json') for i in range(8)];names=[n for p in parts for n in p['images']]
    assert len(names)==len(set(names))==1449 and all(p['checkpoint_sha256']==result['checkpoint_sha256'] for p in parts)
    job=read(RUN/f"eval_queue/step2_iter{result['iteration']}.json");assert digest(job['checkpoint'])==job['checkpoint_sha256']==result['checkpoint_sha256']
final=torch.load(safe_path(done['checkpoint']),map_location='cpu',weights_only=True,mmap=True)
evaluated=torch.load(safe_path(RUN/'10-5/step2/checkpoints/model_iter_8000.pth'),map_location='cpu',weights_only=True,mmap=True)
warmup=torch.load(study['resume_checkpoint'],map_location='cpu',weights_only=True,mmap=True)
assert final['iteration']==evaluated['iteration']==8000 and digest(done['checkpoint'])==done['checkpoint_sha256']
assert all(torch.equal(v,evaluated['model_state'][k]) for k,v in final['model_state'].items())
changed=[k for k,v in final['model_state'].items() if not torch.equal(v,warmup['model_state'][k])]
steps={str(k):int(v['step']) for k,v in final['optimizer_state']['state'].items()}
assert all(v==3500 for v in steps.values()),set(steps.values())
assert digest(study['reference_checkpoint'])==study['reference_sha256']=='3f43207d665ce6001aa149867435364e13baca5d47a0e2274dfb9187975dd28d'
assert int(final['online_confusion_state']['updates'])==1500
def metrics(r):
    h=np.asarray(r['histogram']);d=h.diagonal();old=slice(1,16);new=slice(16,21)
    return {'all_miou':r['all_miou'],'old_miou':r['previous_foreground_miou'],'new_miou':r['current_foreground_miou'],
        'old_precision':float(d[old].sum()/h[:,old].sum()*100),'old_recall':float(d[old].sum()/h[old].sum()*100),
        'new_precision':float(d[new].sum()/h[:,new].sum()*100),'new_recall':float(d[new].sum()/h[new].sum()*100),
        'BG_to_new':int(h[0,new].sum()),'new_to_BG':int(h[new,0].sum()),'BG_to_old':int(h[0,old].sum()),'old_to_BG':int(h[old,0].sum()),
        'old_to_new':int(h[old,new].sum()),'new_to_old':int(h[new,old].sum()),'class_iou':r['class_iou']}
reference=read(R/'runs/pair_preserving_kd_v1/formal/evaluations/step2_iter8300/result.json')
m=metrics(results[-1]);ref=metrics(reference)
diagnostics=[read(U/f'diagnostic_rank{i}.json') for i in range(8)];names=[n for d in diagnostics for n in d['images']]
assert len(names)==len(set(names))==1449
assert all(d['candidate_sha256']==done['checkpoint_sha256'] and d['diagnostic_sha256']==digest(E/'diagnose.py') for d in diagnostics)
classification={}
for model in ['reference','candidate']:
    h=np.sum([np.asarray(d['classification'][model]) for d in diagnostics],axis=0);classification[model]={}
    for name,sl in [('old',slice(0,15)),('new',slice(15,20))]:
        tn,fp,fn,tp=h[sl].sum(0).flatten();classification[model][name]={'precision':float(tp/max(tp+fp,1)*100),'recall':float(tp/max(tp+fn,1)*100),'TP':int(tp),'FP':int(fp),'FN':int(fn),'TN':int(tn)}
pixel={}
for key in diagnostics[0]['pixel_histograms']:
    h=np.sum([np.asarray(d['pixel_histograms'][key]) for d in diagnostics],axis=0);valid=h[:21].sum();accepted=h[:21,:21].sum()
    pixel[key]={'correct':int(h[:21,:21].diagonal().sum()),'accepted':int(accepted),'valid_GT':int(valid),
        'precision':float(h[:21,:21].diagonal().sum()/max(accepted,1)*100),'coverage':float(accepted/max(valid,1)*100),
        'old_label_precision':float(h.diagonal()[1:16].sum()/max(h[:21,1:16].sum(),1)*100),
        'new_label_precision':float(h.diagonal()[16:21].sum()/max(h[:21,16:21].sum(),1)*100)}
counts={k:sum(d['counts'][k] for d in diagnostics) for k in diagnostics[0]['counts']}
pairs=np.sum([np.asarray(d['accepted_directed_pairs']) for d in diagnostics],axis=0)
correct=np.sum([np.asarray(d['correct_source_directed_pairs']) for d in diagnostics],axis=0)
selector=read(RUN/'10-5/step2/geometry_metrics.jsonl') if False else json.loads((RUN/'10-5/step2/geometry_metrics.jsonl').read_text().splitlines()[-1])['selector']
pair_audit=[]
for c in range(1,21):
    for k in range(1,21):
        if c!=k and pairs[c,k]>0:pair_audit.append({'source':c,'competitor':k,'pixels':int(pairs[c,k]),'source_correct':int(correct[c,k]),'source_precision':float(correct[c,k]/pairs[c,k]*100)})
pair_audit.sort(key=lambda x:x['pixels'],reverse=True)
analysis={'status':'complete_endpoint_analysis','utc':now(),'candidate':m,'reference':ref,'delta':{k:m[k]-ref[k] for k in m if k!='class_iou'},
    'class_iou_delta':{k:m['class_iou'][k]-ref['class_iou'][k] for k in m['class_iou']},'trajectory':[metrics(r) for r in results],
    'checkpoint':done['checkpoint'],'checkpoint_sha256':done['checkpoint_sha256'],'evaluated_checkpoint_sha256':results[-1]['checkpoint_sha256'],
    'classification':classification,'pseudo_supervision_448_grid':pixel,'diagnostic_counts':counts,'directed_pair_GT_audit':pair_audit,'final_selector':selector,
    'actual_optimizer_updates':1500,'sample_exposures':48000,'optimizer_steps':sorted(set(steps.values())),
    'changed_model_tensor_count':len(changed),'changed_model_tensors':changed,'training_seconds':done['elapsed_seconds'],
    'verified':{'complete_1449_unique_validation':True,'final_and_evaluated_model_tensors_exact':True,'best_reference_unchanged':True,'optimizer_restored_from_2000':True,'online_confusion_1500_updates':True,'eight_rank_smoke_passed':True},
    'target_exceeded':m['all_miou']>ref['all_miou'],
    'limits':['Fixed-seed adaptive development; not a significance test or isolated ALD causal comparison.',
              'Uses the previous best current-stage student as a frozen weak-evidence/soft-label reference; this is bootstrapped self-training, not just previous-stage inference.',
              'Global batch32 with1500updates; original LR progresses by sample exposures, Adam updates are fewer. Sampler RNG restarted.',
              'Old image state is frozen; crop predictions and student CAM/PAR are online. Unknown old classes get soft classifier targets.',
              'GT used exclusively for validation diagnostics after masks and targets; no old image tags or pixel GT consumed by training.',
              'Online confusion counts repeated image exposures, not independent unique images; GT pair accuracy audited separately.']}
atomic_json(U/'analysis.json',analysis)
print(json.dumps({k:analysis[k] for k in ['candidate','delta','classification','pseudo_supervision_448_grid','target_exceeded']},indent=2))
