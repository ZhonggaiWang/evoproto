from pathlib import Path
import sys,os,json,zipfile
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto');E=R/'experiments/ald_calibration_v3';U=R/'runs/ald_calibration_v3'
sys.path.insert(0,str(R/'experiments/kd_pixel_v2'))
from run_kd import environment
os.environ.update(environment('8card'))
from kd_runtime import safe_path,atomic_json,digest,now
import torch
torch.set_num_threads(1)
read=lambda p:json.loads(safe_path(p).read_text())
a=read(U/'analysis.json');earlier=[read(R/f'runs/ald_calibration_v{i}/analysis.json') for i in [1,2]]
assert all(a['verified'].values())
candidate=torch.load(safe_path(a['checkpoint']),map_location='cpu',weights_only=True,mmap=True)
calibrator=torch.load(safe_path(earlier[1]['checkpoint']),map_location='cpu',weights_only=True,mmap=True)
assert digest(safe_path(a['checkpoint']))==a['checkpoint_sha256']
for key,value in calibrator['model_state'].items():
    if key.startswith(('classifier.','aux_classifier.')):assert torch.equal(value,candidate['model_state'][key])
for i in range(152,158):
    for key,value in calibrator['optimizer_state']['state'][i].items():
        assert torch.equal(value,candidate['optimizer_state']['state'][i][key]) if torch.is_tensor(value) else value==candidate['optimizer_state']['state'][i][key]
assert a['classification']==earlier[1]['classification']
logs=[json.loads(s) for s in (U/'formal/training.jsonl').read_text().splitlines()]
totals=[sum(v['correction_counts'][c] for v in logs) for c in range(42)]
assert sum(totals)==sum(v['correction_pixels'] for v in logs)>0
receipts=[read(U/f'cache_receipt_rank{i}.json') for i in range(8)]
counts={key:sum(v['readiness_coverage'][key] for v in receipts) for key in receipts[0]['readiness_coverage']}
assert all(v['passed'] and v['proposal_sha256']==digest(E/'proposals.py') and digest(safe_path(v['cache']))==v['cache_sha256'] for v in receipts)
names=[n for v in receipts for n in v['images']];assert len(names)==len(set(names))==2145
done=read(U/'formal/training_complete.json');assert done['steps']==300 and len(done['changed_model_tensors'])==3
goal_met=a['candidate']['all_miou']>a['reference']['all_miou']
audit={'utc':now(),'goal_numeric_target_met':goal_met,'full1449_endpoint_verified':True,'frozen_encoder_decoder_body_prototypes_verified':True,
    'classifier_and_optimizer_from_v2_exact':True,'all300updates_accounted':True,'training_unique_images':2145,'training_proposal_counts':counts,
    'correction_exposures_by_group':totals,'actual_foreground_correction_exposures':sum(totals[16:21]),'actual_background_correction_exposures':sum(totals[21:]),
    'GT_only_diagnostic_and_evaluation':True,'checkpoint':a['checkpoint'],'checkpoint_sha256':a['checkpoint_sha256'],
    'cost_review':a['delta'],'classification_review':a['classification'],
    'limitations':['Adaptive development on the same validation set; no held-out or multi-seed claim','Main model inference uses no image labels or pixel GT','Additional head-only refinement, not a same-budget comparison',
        'ALD evidence refreshed once after classifier calibration; not per-step online refresh','Historical confusion observer and selector retained only for provenance; not updated or used to choose pairs in this phase',
        'Spatial absence alone gave84.35% precision on newly selected classifier-positive cases and was rejected as hard negative supervision']}
atomic_json(U/'completion_audit.json',audit)
if goal_met:atomic_json(U/'best_candidate.json',{'utc':now(),'checkpoint':a['checkpoint'],'checkpoint_sha256':a['checkpoint_sha256'],'metrics':a['candidate'],'method':'three-state image calibration, uncertainty correction, refreshed weak evidence and pair-mass-preserving pseudo targets','resume_entry':str(E/'run.py')})
c=a['candidate'];r=a['reference'];d=a['delta'];cl=a['classification']
report=f'''ALD redesign — verified endpoint

Previous best: {r['all_miou']:.6f} mIoU
ALD v1, trusted hard labels: {earlier[0]['candidate']['all_miou']:.6f}
ALD v2, global partial-label projection: {earlier[1]['candidate']['all_miou']:.6f}
ALD v3, refreshed evidence and local soft correction: {c['all_miou']:.6f} ({d['all_miou']:+.6f})
Old15 mIoU: {r['old_miou']:.6f} -> {c['old_miou']:.6f}
New5 mIoU: {r['new_miou']:.6f} -> {c['new_miou']:.6f}

The old classifier does not directly define hard pseudo labels. Spatial and two-view evidence produce present/absent/unknown states. Unknown labels are not negatives. Classifier-only positive conflicts receive an uncertainty target rather than an invented absence label. Calibrated classifier heads refresh CAM/PAR.

Only refreshed new-class CAM/PAR corrections supported by old-teacher foreground enter local soft correction; old-teacher-background/new-label disagreements were rejected after negative net GT correction. Background correction has a separate low-CAM veto that keeps unknown old CAMs visible. Candidate/reference winner probabilities are exchanged without disturbing other classes or pair mass. Conflicting old KD is vetoed; remaining predictions retain the best reference.

Image old-class precision: {cl['reference']['old']['precision']:.4f}% -> {cl['candidate']['old']['precision']:.4f}%
Image old-class recall: {cl['reference']['old']['recall']:.4f}% -> {cl['candidate']['old']['recall']:.4f}%
Image old-class FP: {cl['reference']['old']['FP']} -> {cl['candidate']['old']['FP']}

Segmentation old precision delta: {d['old_precision']:+.4f} pp; recall: {d['old_recall']:+.4f} pp
Segmentation new precision delta: {d['new_precision']:+.4f} pp; recall: {d['new_recall']:+.4f} pp
BG->new delta: {d['BG_to_new']:+d}; old->new: {d['old_to_new']:+d}; new->old: {d['new_to_old']:+d}

Training used all2145 unique train images and current-new image labels. Evaluation used all1449 validation images at original GT resolution under the existing resize448 protocol. No GT pixel labels or old image tags enter optimization, and inference uses no tags. Eight GPUs on 8-card machine only. Step0 and baseline were not retrained. Encoder, decoder body and SEP prototypes remain unchanged. Classifier calibration and final segmentation refinement each used300 cached-feature head updates; earlier rejected development runs are retained.

This is adaptive fixed-seed validation evidence, not a significance or same-budget claim. The calibrated class heads lose some image-level recall. Evidence refresh is staged, not per-step online. Historical confusion state is not claimed as newly updated.

Checkpoint: {a['checkpoint']}
SHA256: {a['checkpoint_sha256']}
Numeric goal met: {goal_met}
'''
safe_path(U/'result.txt').write_text(report,encoding='utf-8')
bundle=safe_path(U/'implementation.zip')
with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as z:
    for i in [1,2,3]:
        ee=safe_path(R/f'experiments/ald_calibration_v{i}');uu=safe_path(R/f'runs/ald_calibration_v{i}')
        for p in sorted(ee.glob('*.py')):z.write(safe_path(p),f'v{i}/src/{p.name}')
        for p in sorted(uu.rglob('*.json')):z.write(safe_path(p),f'v{i}/records/{p.relative_to(uu)}')
        for p in sorted(uu.rglob('training.jsonl')):z.write(safe_path(p),f'v{i}/records/{p.relative_to(uu)}')
    z.writestr('result.txt',report)
artifacts=['analysis.json','diagnostic.json','completion_audit.json','result.txt','implementation.zip']+(['best_candidate.json'] if goal_met else [])
atomic_json(U/'export_manifest.json',{'utc':now(),'artifacts':{name:{'sha256':digest(U/name),'bytes':(U/name).stat().st_size} for name in artifacts},'checkpoint_remains_on_8card':a['checkpoint']})
print(json.dumps({'goal_numeric_target_met':goal_met,'metrics':c,'delta':d,'audit':str(U/'completion_audit.json'),'exports':artifacts}))
