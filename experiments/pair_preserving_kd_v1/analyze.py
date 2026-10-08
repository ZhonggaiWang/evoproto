from pathlib import Path
import sys,os,json
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto');E=R/'experiments/pair_preserving_kd_v1';U=R/'runs/pair_preserving_kd_v1';RUN=U/'formal'
S=R/'experiments/prototype_sep_v1/c_new_anchor/src';C=R/'runs/prototype_sep_v1/formal/c_new_anchor'
sys.path.insert(0,str(R/'experiments/kd_pixel_v2'))
from run_kd import environment
os.environ.update(environment('8card'));sys.path.insert(0,str(S))
from kd_runtime import safe_path,atomic_json,digest,now
import torch,numpy as np
from evaluate_kd import merge
read=lambda p:json.loads(safe_path(p).read_text())
study=read(RUN/'study.json');receipt=read(RUN/'training_complete.json')
job=read(RUN/'eval_queue/step2_iter8300.json');parts=RUN/'evaluations/step2_iter8300'
assert all((parts/f'rank{i}.json').exists() for i in range(8))
merge(job,parts,8);result=read(parts/'result.json');reference=read(C/'evaluations/step2_iter8000/result.json')
assert result['images']==1449 and result['iteration']==8300 and receipt['steps']==300
assert digest(safe_path(job['checkpoint']))==job['checkpoint_sha256']==receipt['checkpoint_sha256']==result['checkpoint_sha256']
assert all(digest(E/k)==v for k,v in study['sources'].items())
assert {str(p.relative_to(S)):digest(p) for p in S.rglob('*.py') if '__pycache__' not in p.parts}==study['parent_source_sha256']
preflight=read(E/'preflight.json');assert preflight['passed'] and all(digest(E/k)==v for k,v in preflight['sources'].items())
assert digest(safe_path(study['resume']))==study['resume_sha256']
saved=torch.load(safe_path(job['checkpoint']),map_location='cpu',weights_only=True,mmap=True)
parent=torch.load(safe_path(study['resume']),map_location='cpu',weights_only=True,mmap=True)
assert saved['iteration']==8300 and saved['refinement_source_sha256']==study['sources']
changed=[k for k in saved['model_state'] if not torch.equal(saved['model_state'][k],parent['model_state'][k])]
assert set(changed)=={'decoder.conv8.0.weight','decoder.conv8.1.weight','decoder.conv8.2.weight'}
assert saved['model_state'].keys()==parent['model_state'].keys()
optimizer_changed=[]
for k,old in parent['optimizer_state']['state'].items():
    new=saved['optimizer_state']['state'][k]
    different=any(not torch.equal(v,new[n]) if torch.is_tensor(v) else v!=new[n] for n,v in old.items())
    if different:
        optimizer_changed.append(k);assert int(new['step'])==8300
    else:assert int(new['step'])==8000
assert len(optimizer_changed)==3
assert int(saved['online_confusion_state']['updates'])==6300
assert int(saved['online_confusion_state']['seen_images'])==67200
logs=[json.loads(line) for line in safe_path(RUN/'training.jsonl').read_text().splitlines()]
assert [v['update'] for v in logs]==list(range(1,301))
assert saved['refinement_totals']==receipt['totals']==logs[-1]['totals']
for key in ['new_pair_pixels','bg_pair_pixels','absent_pixels']:assert sum(r[key] for r in logs)==receipt['totals'][key]
assert all(np.isfinite(v['total_loss']) for v in logs)
def metrics(v):
    h=np.array(v['histogram']);old=slice(1,16);new=slice(16,21);diag=h.diagonal()
    return {'all_miou':v['all_miou'],'old_miou':v['previous_foreground_miou'],'new_miou':v['current_foreground_miou'],
        'old_precision':float(diag[old].sum()/h[:,old].sum()*100),'old_recall':float(diag[old].sum()/h[old].sum()*100),
        'new_precision':float(diag[new].sum()/h[:,new].sum()*100),'new_recall':float(diag[new].sum()/h[new].sum()*100),
        'BG_to_new':int(h[0,new].sum()),'new_to_BG':int(h[new,0].sum()),'BG_to_old':int(h[0,old].sum()),
        'old_to_BG':int(h[old,0].sum()),'old_to_new':int(h[old,new].sum()),'new_to_old':int(h[new,old].sum()),'class_iou':v['class_iou']}
m=metrics(result);ref=metrics(reference)
analysis={'status':'complete_endpoint_analysis','utc':now(),'candidate':m,'reference':ref,
    'delta':{k:m[k]-ref[k] for k in m if k!='class_iou'},'checkpoint':job['checkpoint'],'checkpoint_sha256':job['checkpoint_sha256'],
    'verified':{'full1449_unique':True,'evaluation_checkpoint_exact':True,'frozen_model_except_head_exact':True,
        'only_three_head_optimizer_states_changed':True,'optimizer_steps8300':True,'all300updates_accounted':True,
        'saved_observer_updates6300_images67200':True,'parent_and_new_sources_frozen':True,'8rank_preflight_passed':True},
    'changed_model_tensors':changed,'changed_optimizer_state_ids':optimizer_changed,'totals':receipt['totals'],
    'head_update_seconds':receipt['elapsed_seconds'],'saved_pair_targets':saved['geometry_selector_state']['targets'].tolist(),
    'extra_budget':{'head_only_updates':300,'global_batch':64,'new_training_image_exposures':19200,'cached_training_images':2145},
    'GT_role':'Fixed-gate diagnosis only; cache/optimization use images and current-new image tags; evaluated model uses no tags or GT.',
    'limits':['Single-seed adaptive development','300 additional updates, not a same-budget comparison','Fixed-view head refinement; no fresh backbone/SEP optimization','Direct BG-pair term inactive if cumulative count zero; absent-new correction preserves relative probabilities of other classes']}
atomic_json(U/'analysis.json',analysis);atomic_json(RUN/'status.json',{'status':'complete','utc':now()})
print(json.dumps(analysis,indent=2))
