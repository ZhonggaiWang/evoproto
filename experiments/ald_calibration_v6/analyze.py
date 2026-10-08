from pathlib import Path
import sys,os,json
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto');E=R/'experiments/ald_calibration_v6';U=R/'runs/ald_calibration_v6';RUN=U/'formal'
S=R/'experiments/prototype_sep_v1/c_new_anchor/src'
sys.path.insert(0,str(R/'experiments/kd_pixel_v2'))
from run_kd import environment
os.environ.update(environment('8card'));sys.path.insert(0,str(S))
from kd_runtime import safe_path,atomic_json,digest,now
import torch,numpy as np
from evaluate_kd import merge
read=lambda p:json.loads(safe_path(p).read_text())
study=read(RUN/'study.json');receipt=read(RUN/'training_complete.json');job=read(RUN/'eval_queue/step2_iter8600.json');parts=RUN/'evaluations/step2_iter8600'
assert all((parts/f'rank{i}.json').exists() for i in range(8))
merge(job,parts,8);result=read(parts/'result.json')
reference=read(R/'runs/pair_preserving_kd_v1/formal/evaluations/step2_iter8300/result.json')
assert result['images']==1449 and result['iteration']==8600 and receipt['steps']==300
assert digest(safe_path(job['checkpoint']))==job['checkpoint_sha256']==receipt['checkpoint_sha256']==result['checkpoint_sha256']
assert all(digest(E/k)==v for k,v in study['sources'].items())
assert {str(p.relative_to(S)):digest(p) for p in S.rglob('*.py') if '__pycache__' not in p.parts}==study['parent_source_sha256']
assert digest(E/'evaluate.py')==read(RUN/'evaluation_processes.json')['evaluator_sha256']
pre=read(E/'preflight.json');assert pre['passed'] and all(digest(E/k)==v for k,v in pre['sources'].items())
saved=torch.load(safe_path(job['checkpoint']),map_location='cpu',weights_only=True,mmap=True);parent=torch.load(safe_path(study['resume']),map_location='cpu',weights_only=True,mmap=True)
changed=[k for k,v in saved['model_state'].items() if not torch.equal(v,parent['model_state'][k])]
assert len(changed)==7 and set(changed)==set(receipt['changed_model_tensors']) and 'decoder.conv8.2.weight' in changed
assert all(k.startswith(('decoder.conv8.','classifier.','aux_classifier.')) for k in changed)
opt_changed=[]
for i,before in parent['optimizer_state']['state'].items():
    if i==170:
        assert i not in saved['optimizer_state']['state'];continue
    after=saved['optimizer_state']['state'][i]
    diff=any(not torch.equal(v,after[k]) if torch.is_tensor(v) else v!=after[k] for k,v in before.items())
    if diff:opt_changed.append(i);assert int(after['step'])==int(before['step'])+300
assert len(opt_changed)==6 and set(opt_changed)==set(range(152,158))
assert torch.equal(saved['model_state']['decoder.conv8.0.weight'],parent['model_state']['decoder.conv8.0.weight'])
assert torch.equal(saved['model_state']['decoder.conv8.1.weight'],parent['model_state']['decoder.conv8.1.weight'])
weights=torch.cat([parent['model_state'][f'decoder.conv8.{i}.weight'] for i in range(3)]).flatten(1).double()
change=(saved['model_state']['decoder.conv8.2.weight']-parent['model_state']['decoder.conv8.2.weight']).flatten(1).double()
semantic_error=float((change@weights.T).abs().max());assert semantic_error<1e-6
assert int(saved['ALD_residual_optimizer']['state'][0]['step'])==300
for section in ['online_confusion_state','geometry_selector_state']:
    for k,v in parent[section].items():assert torch.equal(v,saved[section][k]) if torch.is_tensor(v) else v==saved[section][k]
logs=[json.loads(s) for s in (RUN/'training.jsonl').read_text().splitlines()]
assert [v['update'] for v in logs]==list(range(1,301)) and all(np.isfinite(v['loss']) for v in logs)
def metrics(v):
    h=np.array(v['histogram']);old=slice(1,16);new=slice(16,21);d=h.diagonal()
    return {'all_miou':v['all_miou'],'old_miou':v['previous_foreground_miou'],'new_miou':v['current_foreground_miou'],
        'old_precision':float(d[old].sum()/h[:,old].sum()*100),'old_recall':float(d[old].sum()/h[old].sum()*100),
        'new_precision':float(d[new].sum()/h[:,new].sum()*100),'new_recall':float(d[new].sum()/h[new].sum()*100),
        'BG_to_new':int(h[0,new].sum()),'new_to_BG':int(h[new,0].sum()),'BG_to_old':int(h[0,old].sum()),'old_to_BG':int(h[old,0].sum()),
        'old_to_new':int(h[old,new].sum()),'new_to_old':int(h[new,old].sum()),'class_iou':v['class_iou']}
m=metrics(result);ref=metrics(reference);classification={}
records=[read(parts/f'rank{i}.json') for i in range(8)]
for key in ['reference','candidate']:
    h=np.sum([np.array(v['classification'][key]) for v in records],axis=0)
    classification[key]={}
    for name,sl in [('old',slice(0,15)),('new',slice(15,20))]:
        hh=h[sl].sum(0);tn,fp,fn,tp=hh.flatten();classification[key][name]={'precision':float(tp/max(tp+fp,1)*100),'recall':float(tp/max(tp+fn,1)*100),'TP':int(tp),'FP':int(fp),'FN':int(fn),'TN':int(tn)}
analysis={'status':'complete_endpoint_analysis','utc':now(),'candidate':m,'reference':ref,'delta':{k:m[k]-ref[k] for k in m if k!='class_iou'},
    'checkpoint':job['checkpoint'],'checkpoint_sha256':job['checkpoint_sha256'],'classification':classification,
    'verified':{'full1449_unique':True,'checkpoint_exact':True,'frozen_old16_and_body_exact':True,'semantic_direction_preservation':True,'old_optimizer_states_exact_except_inherited_classifiers':True,'residual_optimizer_saved_and_folded_head_state_reset':True,'all300updates_accounted':True,'8rank_gradient_checks':True,'past_confusion_selector_not_reused':True},
    'semantic_constraint_max_error':semantic_error,'changed_model_tensors':changed,'changed_optimizer_ids':opt_changed,'preflight':pre,'head_update_seconds':receipt['elapsed_seconds'],
    'limits':['Additional300 residual-coordinate updates and inherited300 classifier updates; fixed-seed adaptive development','Frozen evidence during head calibration; no claim of online feedback','Pixel GT only diagnostics and endpoint; current-new image absence supplies partial-label negative supervision; no training pixel GT or old image tags','Inherited confusion state preserved for provenance, not an updated confusion estimate','ALD-v2 classifiers inherited; original16 logits exact but new classes can still compete; new residual constrained orthogonally to all21 classifier directions. Gate593/649 correct in validation is not a guarantee of generalization.']}
atomic_json(U/'analysis.json',analysis);atomic_json(RUN/'status.json',{'status':'complete','utc':now()});print(json.dumps(analysis,indent=2))
