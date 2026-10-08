from pathlib import Path
import sys,os,json,ast
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto');E=R/'experiments/local_sep_v1';S=E/'src'
sys.path.insert(0,str(R/'experiments/kd_pixel_v2'))
from run_kd import environment
os.environ.update(environment('8card'))
from kd_runtime import safe_path,atomic_json,digest,now
def change(t,a,b):
 assert t.count(a)==1,(a[:100],t.count(a));return t.replace(a,b)
safe_path(S/'model/local_sep.py').write_bytes(safe_path(E/'local_sep.py').read_bytes())
p=S/'model/decoder/conv_head.py';t=p.read_text()
t=change(t,'def forward(self, x, cal_sim=False):','def forward(self, x, cal_sim=False, return_features=False):')
t=change(t,'        if not cal_sim:\n            return out, sim_logits, prototypes','        if return_features:\n            return out, sim_logits, prototypes, x\n        if not cal_sim:\n            return out, sim_logits, prototypes')
safe_path(p).write_text(t)
p=S/'model/model_seg_neg.py';t=p.read_text()
t=change(t,'prototype_objective=None):','prototype_objective=None, local_objective=None):')
t=change(t,'        seg, type_seg, prototypes = self.decoder(_x4, cal_sim=cal_sim)',
 '''        if local_objective is None:
            seg, type_seg, prototypes = self.decoder(_x4, cal_sim=cal_sim)
            local_outputs = None
        else:
            seg, type_seg, prototypes, decoder_features = self.decoder(_x4, cal_sim=cal_sim, return_features=True)
            local_outputs = local_objective(decoder_features, seg, _x4)''')
t=change(t,'            if prototype_objective is not None:\n                return cls_x4, seg, _x4, cls_aux, type_seg, prototypes, prototype_outputs',
 '''            if local_objective is not None:
                return cls_x4, seg, _x4, cls_aux, type_seg, prototypes, prototype_outputs, local_outputs
            if prototype_objective is not None:
                return cls_x4, seg, _x4, cls_aux, type_seg, prototypes, prototype_outputs''')
safe_path(p).write_text(t)
p=S/'model/geometry_pair_selector.py';t=p.read_text().replace('SCHEMA = 4','SCHEMA = 5').replace('classes[:, None] > self.old_classes','classes[:, None] > 0').replace('current_new_only','all_foreground_with_ALD_local_anchor_gate').replace('current new foreground source','foreground source')
safe_path(p).write_text(t)
p=S/'continual/Trainer.py';t=p.read_text()
t=change(t,'            if False:  # Fresh observer: calibrated pseudo-anchor semantics changed.',"            if saved.get('online_confusion_state') is not None:  # Same ALD anchor semantics; preserve online history.")
t=change(t,'        optim.max_iter = 8600  # Smoke uses the same 300-update cosine as formal.',
 '''        # Only the two spatial decoder convolutions learn. Encoder, classifier,
        # old/new segmentation heads and prototype anchors remain fixed.
        import copy
        self.sep_reference = copy.deepcopy(model.decoder).eval().requires_grad_(False)
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name in {'decoder.conv6.weight', 'decoder.conv7.weight'})
        bank = torch.load(safe_path(Path(__file__).resolve().parents[2]/'bank.pth'), weights_only=True, map_location=device)
        self.sep_centers, self.sep_available = bank['centers'], bank['available']
        optim.max_iter = 8900''')
a=t.index('                    if n_iter < args.loss_warmup_iters:',t.index('                def prototype_objective'))
b=t.index("                    geometry_record['semantic_protection']",a)
t=t[:a]+'''                    protected = native_type_seg.sum() * 0.
                    protection = {'enabled': False, 'reason': 'fixed_prototypes_local_feature_SEP',
                                  'effective_geometry_weight': 0., 'semantic_gradient_used': False}
'''+t[b:]
t=change(t,'''                cls, segs, fmap, cls_aux, type_seg, new_prototypes, prototype_outputs = model(
                    inputs, crops=roi_crops, n_iter=n_iter, prototype_objective=prototype_objective)''',
 '''                from model.local_sep import separation_loss, retention_loss
                def local_objective(features, logits, encoder_features):
                    with torch.no_grad():
                        fixed_logits = self.sep_reference(encoder_features.detach())[0]
                    separation, record = separation_loss(features, mixed_pseudo_label,
                        valid_cam, ald_fused['valid'], pair_targets, self.sep_centers, self.sep_available)
                    retention = retention_loss(logits, fixed_logits, ald_fused['valid'])
                    record.update(retention=float(retention.detach()))
                    return separation, retention, record
                cls, segs, fmap, cls_aux, type_seg, new_prototypes, prototype_outputs, local_outputs = model(
                    inputs, crops=roi_crops, n_iter=n_iter, prototype_objective=prototype_objective,
                    local_objective=local_objective)
                local_sep, local_retention, local_record = local_outputs''')
t=change(t,' + protected_sep + args.w_seg * ald_soft',' + protected_sep + args.w_seg * ald_soft + .1 * local_sep + local_retention')
t=change(t,"                    ald_record.update(iteration=self.current_iteration, global_batch=args.spg*dist.get_world_size())",
 '''                    local_record.update(iteration=self.current_iteration, selected_pairs=pair_targets.tolist())
                    with open(osp.join(args.work_dir,'local_sep_metrics.jsonl'),'a') as handle:
                        handle.write(json.dumps(local_record,allow_nan=False)+'\\n')
                    ald_record.update(iteration=self.current_iteration, global_batch=args.spg*dist.get_world_size())''')
safe_path(p).write_text(t)
p=S/'utils/optimizer.py';t=p.read_text().replace('(self.global_step - 8300) / 300','(self.global_step - 8600) / 300');safe_path(p).write_text(t)
p=S/'scripts/dist_train_voc_seg_neg.py';t=p.read_text().replace('8300 < args.max_iters <= 8600','8600 < args.max_iters <= 8900').replace('resumes8300','resumes8600');safe_path(p).write_text(t)
for p in S.rglob('*.py'):ast.parse(p.read_text())
atomic_json(E/'origin.json',{'utc':now(),'source_sha256':{str(p.relative_to(S)):digest(p) for p in S.rglob('*.py')},'ald_sha256':digest(S/'model/ald.py'),'bank_sha256':digest(E/'bank.pth'),'mechanism':'Frozen 4-mode weak training anchors; directed pair local feature hinge; fixed eligibility normalization and stopping at cosine gap .05. Train spatial conv6/conv7 only, fixed full-reference KL plus inherited ALD/old KD. Preserve observer history, fresh all-FG selector. GT diagnostic only.'})
print('Prepared source')
