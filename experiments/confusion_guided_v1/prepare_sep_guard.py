from pathlib import Path
import sys,shutil,ast
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto');E=R/'experiments/confusion_guided_v1'
sys.path.insert(0,str(E/'a_sep/src'))
from kd_runtime import safe_path,atomic_json,now,digest
source=E/'a_sep/src';target=safe_path(E/'c_newaware_sep/src')
assert not target.exists()
shutil.copytree(source,target)
shutil.copyfile(source/'model/confusion_pair_losses.py',E/'sep_legacy_reference.py')
trainer=target/'continual/Trainer.py';s=trainer.read_text()
anchor='teacher_native_logits, pair_evidence, pair_targets)'
assert s.count(anchor)==1
s=s.replace(anchor,'teacher_native_logits, pair_evidence, pair_targets,\n                        valid_cams=valid_cam, new_cam_guard=args.sep_new_cam_guard)')
ast.parse(s);trainer.write_text(s)
parser=target/'scripts/dist_train_voc_seg_neg.py';s=parser.read_text()
anchor='parser.add_argument("--pair_mode",'
assert s.count(anchor)==1
pos=s.index(anchor)
s=s[:pos]+'parser.add_argument("--sep_new_cam_guard", action="store_true")\n'+s[pos:]
ast.parse(s);parser.write_text(s)
atomic_json(E/'guard_protocol.json',{'utc':now(),'arm':'c_newaware_sep','hypothesis':'Old teacher agreement can endorse novel-class pixels; soften old SEP with existing KD new-CAM protection',
 'training':'Reuse armA stage1 final teacher and armA stage2 iter2000 full student/optimizer/observer/selector state; only stage2 remaining6000 updates',
 'change':'Only old SEP numerator gets (1-max_new_CAM)^2; new anchors/counts/classbalance/selector/ramp/KD/lambda unchanged',
 'stage1_policy':'Inherited tested plain directed SEP+legacy KD; no rerun',
 'evidence':str(R/'runs/confusion_guided_v1/diagnostics/sep_new_conflict/merged.json'),
 'causal_limit':'Resume restarts sampler/augmentation under8GPU instead of4; descriptive fixedseed comparison, not exact RNG continuation'})
print('Prepared isolated guard source; guarded loss pending deployment')
