"""Full endpoint verification and error-cost comparison; no peak selection."""
from pathlib import Path
import sys,json,math
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto');E=R/'experiments/hierarchy_kd_v1';U=R/'runs/hierarchy_kd_v1'
sys.path.insert(0,str(R/'experiments/kd_pixel_v2'))
from run_kd import environment
import os
os.environ.update(environment('8card'));sys.path.insert(0,str(E/'src'))
from kd_runtime import safe_path,digest,atomic_json,now
import torch
torch.set_num_threads(1)
run=U/'formal/h1';stage=run/'10-5/step2'
read=lambda p:json.loads(safe_path(p).read_text())
assert read(run/'status.json')['status']=='complete'
study=read(run/'study.json');src=Path(study['source'])
assert {str(p.relative_to(src)):digest(p) for p in src.rglob('*.py')}==study['source_sha256']
assert read(E/'preflight.json')['source_sha256']==study['source_sha256']
assert read(run/'evaluation_workers_complete.json')['returncodes']==[0]*4
paths={'candidate':run,'C_new_anchor':R/'runs/prototype_sep_v1/formal/c_new_anchor',
       'optimized_KD':R/'runs/kd_parallel_v1/formal/b_relational'}
results={k:read(p/'evaluations/step2_iter8000/result.json') for k,p in paths.items()}
for name,result in results.items():
 assert result['images']==1449 and result['step']==2 and result['iteration']==8000
 assert [sum(row) for row in result['histogram']]==[sum(row) for row in results['C_new_anchor']['histogram']]
job=read(run/'eval_queue/step2_iter8000.json');assert digest(Path(job['checkpoint']))==job['checkpoint_sha256']==results['candidate']['checkpoint_sha256']
final=torch.load(stage/'checkpoints/model_final.pth',map_location='cpu',weights_only=True,mmap=True)
evaluated=torch.load(job['checkpoint'],map_location='cpu',weights_only=True,mmap=True)
def exact(a,b):
 if torch.is_tensor(a):return torch.equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(exact(a[k],b[k]) for k in a)
 if isinstance(a,(tuple,list)):return len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
 return a==b
assert exact(final,evaluated),'Final/evaluated states differ'
assert final['iteration']==8000 and int(final['online_confusion_state']['updates'])==6000
assert int(final['online_confusion_state']['seen_images'])==48000
assert all(torch.isfinite(t).all() for t in final['model_state'].values() if torch.is_tensor(t))
assert all(int(v['step'])==8000 for v in final['optimizer_state']['state'].values() if 'step' in v)
def metrics(r):
 h=torch.tensor(r['histogram'],dtype=torch.float64);tp=h.diag();row=h.sum(1);col=h.sum(0)
 iou=100*tp/(row+col-tp);assert abs(float(iou.mean())-r['all_miou'])<1e-8
 out={'all_miou':r['all_miou'],'old_miou':float(iou[1:16].mean()),'new_miou':float(iou[16:].mean()),'class_iou':iou.tolist(),
 'BG_to_old_pixels':int(h[0,1:16].sum()),'BG_to_new_pixels':int(h[0,16:].sum()),
 'old_to_BG_pixels':int(h[1:16,0].sum()),'new_to_BG_pixels':int(h[16:,0].sum()),
 'old_to_new_pixels':int(h[1:16,16:].sum()),'new_to_old_pixels':int(h[16:,1:16].sum())}
 for tag,slice_ in [('old',slice(1,16)),('new',slice(16,21))]:
  a=tp[slice_].sum();b=col[slice_].sum();c=row[slice_].sum()
  out.update({tag+'_precision':float(100*a/b),tag+'_recall':float(100*a/c),tag+'_TP':int(a),tag+'_FP':int(b-a)})
 return out
summary={k:metrics(r) for k,r in results.items()}
deltas={name:{k:summary['candidate'][k]-ref[k] for k in ref if k!='class_iou'} for name,ref in summary.items() if name!='candidate'}
logs=[json.loads(line) for line in (stage/'hierarchy_metrics.jsonl').read_text().splitlines()]
assert len(logs)==120 and logs[-1]['iteration']==8000
assert all(all(math.isfinite(v) for v in [x['mass_kd'],x['background_rejection'],x['ramp']]) for x in logs)
report={'status':'complete_endpoint_analysis','created_utc':now(),'run':str(run),'study':study,
 'verified':{'full1449_endpoint':True,'same_GT_denominators':True,'final_and_evaluated_all_states_exact':True,
 'restored_optimizer_completed_8000':True,'fresh_observer_updates':6000,'new_training_steps':6000,
 'old_conditional_KD_unchanged':study['source_sha256']['model/pixel_kd.py']=='1c539c53d0d53a7eec1cdf7b9917bfdaf7db49ce7f0a721054772e3f900ff5c8'},
 'final_checkpoint':str(stage/'checkpoints/model_final.pth'),'final_checkpoint_sha256':digest(stage/'checkpoints/model_final.pth'),
 'metrics':summary,'deltas':deltas,
 'sampled_training_logs':{'every_updates':50,'records':len(logs),'mass_active_records':sum(x['mass_kd']>0 for x in logs),
 'rejection_active_records':sum(x['background_rejection']>0 for x in logs),
 'mass_pixels':sum(x['mass_pixels'] for x in logs),'BG_pair_pixels':sum(x['background_pair_pixels'] for x in logs),
 'absent_new_pixels':sum(x['absent_new_pixels'] for x in logs),'last':logs[-1]},
 'results':results,'limits':['Single fixed seed adaptive development; not statistical significance or independent validation.',
 'GT gate diagnosis used a fixed128 native-grid prefix, not the full-resolution1449 endpoint.',
 'Selection rates rank only, thresholds and weights were fixed before formal training.']}
atomic_json(U/'analysis.json',report)
print(json.dumps({k:report[k] for k in ['status','metrics','deltas','sampled_training_logs']},allow_nan=False))
