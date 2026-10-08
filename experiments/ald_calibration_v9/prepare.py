from pathlib import Path
import sys,os,json,ast
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto');E=R/'experiments/ald_calibration_v9';S=E/'src';P=R/'experiments/ald_calibration_v8/src'
sys.path.insert(0,str(R/'experiments/kd_pixel_v2'))
from run_kd import environment
os.environ.update(environment('8card'))
from kd_runtime import safe_path,atomic_json,digest,now
parent=json.loads(safe_path(R/'runs/ald_calibration_v8/formal/study.json').read_text())
hashes={str(p.relative_to(P)):digest(p) for p in P.rglob('*.py') if '__pycache__' not in p.parts};assert hashes==parent['source_sha256']
assert not S.exists()
for p in P.rglob('*'):
    safe_path(p)
    if p.is_file() and '__pycache__' not in p.parts:
        q=safe_path(S/p.relative_to(P));q.parent.mkdir(parents=True,exist_ok=True)
        with q.open('xb') as out:out.write(p.read_bytes())
safe_path(S/'model/ald.py').write_bytes(safe_path(E/'ald.py').read_bytes())
def change(t,a,b):
    assert t.count(a)==1,(a[:100],t.count(a))
    return t.replace(a,b)
p=S/'continual/Trainer.py';t=p.read_text()
t=change(t,"            if saved.get('online_confusion_state') is not None:","            if False:  # Fresh observer: calibrated pseudo-anchor semantics changed.")
t=change(t,"            if saved.get('geometry_selector_state') is not None:","            if False:  # Do not re-use the stale prior pair selector.")
t=change(t,'        optim.max_iter = 8000  # Smoke uses the same LR trajectory as formal.','        optim.max_iter = 8600  # Smoke uses the same 300-update cosine as formal.')
t=change(t,"                # Iteration tracks sample exposure in legacy global-batch-8 units.\n                # Adam advances once per actual global-batch-32 update.","                # Actual ALD refinement updates; resume8300, finish8600.\n                # Restored parameter-wise Adam counters may differ at entry.")
t=change(t,'ald_anchors = gate_auxiliary(refined_pseudo_label, ald_state, reference_logits)','ald_anchors = gate_auxiliary(refined_pseudo_label, ald_state, reference_logits, calibrate_new=False)')
ast.parse(t);safe_path(p).write_text(t)
p=S/'utils/optimizer.py';t=p.read_text();a=t.index('    def step(self, closure=None):',t.index('class PolyWarmupAdamW'))
b=t.index('\nclass PolyWarmupSGD',a)
t=t[:a]+'''    def step(self, closure=None):
        # Short ALD refinement, preserving inherited Adam moments per parameter.
        phase = min(max((self.global_step - 8300) / 300, 0.), 1.)
        multiplier = float(.5 + .5 * np.cos(np.pi * phase))
        for group, initial in zip(self.param_groups, self.__init_lr):
            group['lr'] = initial * multiplier
        torch.optim.AdamW.step(self, closure)
        self.global_step += 1

''' + t[b:]
ast.parse(t);safe_path(p).write_text(t)
p=S/'scripts/dist_train_voc_seg_neg.py';t=p.read_text()
t=change(t,"    if args.ald_stride != args.spg or (args.max_iters-2000) % args.ald_stride:\n        parser.error('ALD stride must match batch per GPU and divide remaining exposure steps')",
    "    if args.ald_stride != 1 or not 8300 < args.max_iters <= 8600:\n        parser.error('ALD refinement resumes8300 with one index per actual update')")
ast.parse(t);safe_path(p).write_text(t)
atomic_json(E/'origin.json',{'utc':now(),'parent_source_sha256':hashes,'parent_source':str(P),
    'source_sha256':{str(p.relative_to(S)):digest(p) for p in S.rglob('*.py') if '__pycache__' not in p.parts},
    'ald_sha256':digest(E/'ald.py'),'changed_semantics':'Consensus old/new/BG hard labels; disagreements partial-set image absence constraints plus conditional foreground relations. PTC calibrated for new conflicts. Confusion new rows retain independent CAM/PAR anchors, reported separately; extra SEP disabled for this ALD refinement, inherited SEP-trained weights retained. Fresh observer/selector. Resume best8300 and Adam moments, 300updates, cosine encoder2e-6/head2e-5.'})
print('Prepared v9 source')
