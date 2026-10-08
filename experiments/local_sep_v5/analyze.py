from pathlib import Path
import sys,os,json
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto');E=R/'experiments/local_sep_v5';U=R/'runs/local_sep_v5';S=E/'src';RUN=U/'formal'
sys.path.insert(0,str(R/'experiments/kd_pixel_v2'))
from run_kd import environment
os.environ.update(environment('8card'))
from kd_runtime import safe_path,atomic_json,digest,now
import torch,numpy as np
read=lambda p:json.loads(safe_path(p).read_text())
assert read(RUN/'status.json')['status']=='complete'
study=read(RUN/'study.json');done=read(RUN/'training_complete.json');result=read(RUN/'evaluations/step2_iter8900/result.json')
assert study['source_sha256']=={str(p.relative_to(S)):digest(p) for p in S.rglob('*.py')}
assert result['checkpoint_sha256']==digest(done['checkpoint'])==done['checkpoint_sha256']
final=torch.load(done['checkpoint'],map_location='cpu',weights_only=True,mmap=True)
parent=torch.load(study['parent_checkpoint'],map_location='cpu',weights_only=True,mmap=True)
original=torch.load(R/'runs/ald_calibration_v9/formal/10-5/step2/checkpoints/model_final.pth',map_location='cpu',weights_only=True,mmap=True)
assert all(torch.equal(v,final['model_state'][k]) for k,v in parent['model_state'].items())
assert all(torch.equal(v,final['model_state'][k]) for k,v in original['model_state'].items())
assert set(final['model_state'])-set(parent['model_state'])=={'decoder.local_sep.routing'}
assert digest(study['parent_checkpoint'])==study['parent_sha256']
for k,v in parent['optimizer_state']['state'].items():
 for key,value in v.items():assert torch.equal(value,final['optimizer_state']['state'][k][key])
for k,v in original['optimizer_state']['state'].items():
 for key,value in v.items():assert torch.equal(value,final['base_optimizer_state']['state'][k][key])
route=read(U/'routing.json');counts=torch.tensor(route['unique_pair_images']);mask=(counts>=3)&(counts.T>=3);mask.fill_diagonal_(False);mask[0]=False;mask[:,0]=False
assert torch.equal(mask,final['model_state']['decoder.local_sep.routing'])
parts=[read(RUN/'evaluations/step2_iter8900'/f'rank{i}.json') for i in range(8)]
names=sum([x['images'] for x in parts],[]);assert len(names)==len(set(names))==1449
def metrics(r):
 h=np.asarray(r['histogram']);d=h.diagonal();old=slice(1,16);new=slice(16,21)
 return {'all_miou':r['all_miou'],'old_miou':r['previous_foreground_miou'],'new_miou':r['current_foreground_miou'],
 'old_precision':float(d[old].sum()/h[:,old].sum()*100),'old_recall':float(d[old].sum()/h[old].sum()*100),
 'new_precision':float(d[new].sum()/h[:,new].sum()*100),'new_recall':float(d[new].sum()/h[new].sum()*100),
 'BG_to_new':int(h[0,new].sum()),'new_to_BG':int(h[new,0].sum()),'BG_to_old':int(h[0,old].sum()),'old_to_BG':int(h[old,0].sum()),
 'old_to_new':int(h[old,new].sum()),'new_to_old':int(h[new,old].sum()),'class_iou':r['class_iou']}
m=metrics(result);r=metrics(read(R/'runs/ald_calibration_v9/formal/evaluations/step2_iter8600/result.json'))
analysis={'status':'complete_endpoint_analysis','utc':now(),'candidate':m,'reference':r,'delta':{k:m[k]-r[k] for k in m if k!='class_iou'},
 'class_iou_delta':{k:m['class_iou'][k]-r['class_iou'][k] for k in m['class_iou']},'checkpoint':done['checkpoint'],'checkpoint_sha256':done['checkpoint_sha256'],
 'actual_updates':0,'total_adapter_updates':300,'global_batch':32,'adapter_training_sample_exposures':9600,
 'changed_original_model_tensors':[],'routing_pairs':route['pairs'],'routing_unique_training_images':2145,
 'verified':{'full1449_unique':True,'all_v3_learned_tensors_unchanged':True,'all_ALDv9_original_tensors_unchanged':True,'original_and_adapter_Adam_unchanged':True,'routing_derived_only_from_distinct_training_support':True,'image_classifier_function_unchanged':True},
 'target_exceeded':m['all_miou']>r['all_miou'],
 'limits':['Fixed-seed adaptive development; numerical comparison, not statistical significance or isolated causal attribution.','Route threshold3 was fixed from the existing support policy; no per-class GT whitelist. Only chair-sofa currently has bidirectional coverage.','Class-pair mass/nonpair/BG guarantees hold at native prediction grid; interpolation can change original-resolution boundary decisions.','Current-stage training data lack sufficient reverse examples for several other pairs; those routes remain at the original model.','The adopted adapter is v3, with300 updates; v4 negative-loss continuation was rejected. V5 applies support gating without additional training.']}
atomic_json(U/'analysis.json',analysis)
print(json.dumps({k:analysis[k] for k in ['candidate','delta','class_iou_delta','routing_pairs','target_exceeded']},indent=2))
