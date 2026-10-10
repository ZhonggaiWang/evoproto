"""Predeclared same-budget comparison; cached references need no retraining."""
import sys,json,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.restore_proto_v1.run import sha,write
S=ROOT/'runs/restore_proto_simplekd_v1';F=S/'formal';C=S/'control'
def read(p):return json.loads(p.read_text())
def miou(h):
 d=h.sum(0)+h.sum(1)-h.diagonal();return np.nanmean(np.divide(100*h.diagonal(),d,out=np.full(len(d),np.nan),where=d>0))
def main():
 manifest=read(F/'manifest.json')
 for p,h in manifest['source_sha256'].items():
  if sha(ROOT/p)!=h and p=='experiments/restore_proto_simplekd_v1/finish.py':
   repair=read(C/'analysis_helper_repair.json')
   assert repair['original_sha256']==h and sha(ROOT/p)==repair['repaired_sha256']
  else:assert sha(ROOT/p)==h,p
 stages={}
 previous=manifest['initial_sha256']
 for stage in [1,2]:
  d=F/'no_graph/10-5'/f'step{stage}';receipt=read(d/'final_receipt.json');parent=read(d/'predecessor.json')
  assert sha(Path(receipt['path']))==receipt['sha256'] and parent['sha256']==previous and sha(Path(parent['path']))==previous
  curve=[json.loads(x) for x in (d/'metrics.jsonl').read_text().splitlines()];assert curve[-1]['iteration']==8000
  cfg=read(d/'config.json');assert cfg['spg']==4 and cfg['seed_mode']=='no_graph'
  audit=read(d/'prototype_direction_audit.json');assert audit['direction_changed_beyond_decay']
  relation=[json.loads(x) for x in (d/'relation_metrics.jsonl').read_text().splitlines()]
  assert any(x.get('kd_novel_veto_pixels',0)>0 for x in relation),'No expanded-region KD veto observed'
  stages[str(stage)]={'receipt':receipt,'predecessor':parent,'curve':curve,'prototype_audit':audit,'sampled_extra_novel_veto_pixels':sum(x.get('kd_novel_veto_pixels',0) for x in relation),'pseudo_label_audit':read(d/'seed_audit.json')}
  previous=receipt['sha256']
 protected=read(ROOT/'runs/restore_proto_seed_v1/control/final_weight_retention.json')['protected_weights']
 for p,h in protected.items():assert sha(ROOT/p)==h,p
 refs={'baseline':ROOT/'runs/restore_proto_v1/formal/full/10-5/step2','original_seed_kd':ROOT/'runs/restore_proto_seed_v1/formal/no_graph/10-5/step2','simple_seed_kd':F/'no_graph/10-5/step2'}
 evals={a:read(d/'fusion_evaluation.json') for a,d in refs.items()};arrays={a:np.load(d/'fusion_evaluation.npz') for a,d in refs.items()}
 names=arrays['baseline']['names'];assert len(names)==1449
 for a,z in arrays.items():assert np.array_equal(names,z['names']) and np.array_equal(z['alphas'],arrays['baseline']['alphas']) and not evals[a]['image_tags_used']
 comparisons={}
 for mode in ['square448','aspect672']:
  for alpha in [0.,.5]:
   idx=list(arrays['baseline']['alphas']).index(alpha);mats={a:z[mode][:,idx] for a,z in arrays.items()}
   for a,h in mats.items():assert np.array_equal(h.sum(2),mats['baseline'].sum(2)) and abs(miou(h.sum(0))-evals[a]['results'][mode][str(alpha)]['miou'])<1e-8
   rows={}
   for ref in ['baseline','original_seed_kd']:
    delta=miou(mats['simple_seed_kd'].sum(0))-miou(mats[ref].sum(0));rng=np.random.default_rng(20261010);boot=[]
    for _ in range(1000):
     ids=rng.integers(0,len(names),len(names));boot.append(miou(mats['simple_seed_kd'][ids].sum(0))-miou(mats[ref][ids].sum(0)))
    rows[ref]={'delta':float(delta),'image_bootstrap_ci95':np.percentile(boot,[2.5,97.5]).tolist()}
   comparisons[f'{mode}_alpha{alpha}']=rows
 result={'status':'complete','time':time.time(),'method':'seed-region KD veto; no reliability score','stages':stages,'fusion_evaluations':evals,'comparisons':comparisons,'protected_weights_verified':protected,'limitations':'One training seed; image bootstrap does not measure training-seed variability; validation not independent test; final weights only retained pending review.'}
 write(S/'comparison.json',result);write(ROOT/'experiments/restore_proto_simplekd_v1/results.json',result)
 print(json.dumps({a:e['results']['aspect672']['0.5']['miou'] for a,e in evals.items()}),flush=True)
if __name__=='__main__':main()
