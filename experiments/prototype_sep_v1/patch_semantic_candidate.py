"""Patch an already copied, idle b_semantic/src; never edit primary a_geometry.

The module callback computes the two prototype components at the original p,
before DDP's output _DDPSink. Default model returns, state-dict keys, parameter
groups, KD bytes, masks, selector rules and semantic coefficient are retained.
Only weighted prototype-SEP gradients are replaced. This script does not copy
checkpoints, launch training, upload anything or choose whether to run B.
"""
from pathlib import Path
import argparse
import ast
import hashlib
import os
import sys


ROOT = Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto')
EXPERIMENT = ROOT/'experiments/prototype_sep_v1'
PRIMARY = EXPERIMENT/'a_geometry/src'
CANDIDATE = EXPERIMENT/'b_semantic/src'


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError(f'Expected exactly one frozen-source marker: {old[:100]!r}')
    return text.replace(old, new)


def patch_model(text):
    signature = '    def forward(self, x, cam_only=False, cam_grad=False, crops=None, n_iter=None, step0=False, cal_sim=False):'
    text = replace_once(text, signature,
        '    def forward(self, x, cam_only=False, cam_grad=False, crops=None, n_iter=None, step0=False, cal_sim=False, prototype_objective=None):')
    entry = '        cls_token, _x, x_aux = self.encoder.forward_features(x)'
    text = replace_once(text, entry, '''        if prototype_objective is not None and (cam_only or cam_grad or step0 or crops is None):
            raise ValueError('prototype_objective is only valid for the full incremental training forward')

'''+entry)
    decoder = '        seg, type_seg, prototypes = self.decoder(_x4, cal_sim=cal_sim)'
    text = replace_once(text, decoder, decoder+'''
        # Compute partial prototype gradients before DDP wraps sibling outputs.
        # This callable introduces no parameters or state-dict entries.
        prototype_outputs = (prototype_objective(type_seg, prototypes)
                             if prototype_objective is not None else None)
''')
    output = '            return cls_x4, seg, _x4, cls_aux, type_seg, prototypes'
    text = replace_once(text, output, '''            if prototype_objective is not None:
                return cls_x4, seg, _x4, cls_aux, type_seg, prototypes, prototype_outputs
'''+output)
    ast.parse(text)
    return text


def patch_trainer(text):
    text = replace_once(text,
        'from model.confusion_prototype_sep import confusion_prototype_sep_loss',
        'from model.confusion_prototype_sep import confusion_prototype_sep_loss\nfrom model.semantic_protected_sep import semantic_protected_sep')
    branch = '        else:\n            model_old = self.model_old.to(device)'
    location = text.index(branch)
    prefix, tail = text[:location], text[location:]
    old_teacher_labels = '''                old_segs = F.interpolate(old_segs, size=inputs.shape[-2:], mode='bilinear', align_corners=False)
                old_pixel_label = torch.argmax(old_segs, dim=1)
'''
    tail = replace_once(tail, old_teacher_labels, '')
    mask_start = tail.index('                # seg_loss & reg_loss\n')
    mask_end = tail.index('                # Select from past globally synchronized state.', mask_start)
    mask_code = tail[mask_start:mask_end]
    tail = tail[:mask_start]+tail[mask_end:]
    original_selection = '''                # Select from past globally synchronized state. Rates only rank
                # supported pairs; they are never interpreted as exact strengths.
                pair_targets = self.pair_selector.update(self.confusion,n_iter+1)
                pair_ramp = self.pair_selector.ramp(self.confusion)
                self.confusion.update(segs,refined_pseudo_label,valid_cam,img_box)
                geometry_sep, geometry_stats = confusion_prototype_sep_loss(
                    new_prototypes,pair_targets,self.old_classes,margin=args.proto_margin)
'''
    tail = replace_once(tail, original_selection,
        '                self.confusion.update(segs,refined_pseudo_label,valid_cam,img_box)\n')
    mixed_code = '''                mixed_pseudo_label = get_mixed_label(refined_pseudo_label, old_pixel_label,
                                                     self.total_classes, self.new_classes)
'''
    tail = replace_once(tail, mixed_code, '')
    forward = '                cls, segs, fmap, cls_aux, type_seg, new_prototypes = model(inputs, crops=roi_crops, n_iter=n_iter)\n'
    before_forward = '''                # Only deterministic, segmentation-independent label operations
                # move before the main forward. ROI random crops stay above.
'''+old_teacher_labels+mask_code+'''                # Past selector state, before observing this batch.
                pair_targets = self.pair_selector.update(self.confusion,n_iter+1)
                pair_ramp = self.pair_selector.ramp(self.confusion)
'''+mixed_code+'''
                def prototype_objective(native_type_seg, current_prototypes):
                    semantic_logits = F.interpolate(native_type_seg / 0.1,
                        size=mixed_pseudo_label.shape[1:], mode='bilinear', align_corners=False)
                    semantic_loss = get_seg_loss(semantic_logits, mixed_pseudo_label.long(),
                        class_weight=self.class_weight, ignore_index=args.ignore_index)
                    geometry_loss, geometry_record = confusion_prototype_sep_loss(
                        current_prototypes, pair_targets, self.old_classes, margin=args.proto_margin)
                    if n_iter < args.loss_warmup_iters:
                        protected = (current_prototypes * 0.0).sum()
                        protection = {'enabled': False, 'reason': 'loss_warmup',
                                      'effective_geometry_weight': 0.0,
                                      'semantic_gradient_used': False}
                    else:
                        effective_weight = args.w_geometry_sep * pair_ramp
                        protected, protection = semantic_protected_sep(current_prototypes,
                            effective_weight * geometry_loss,
                            args.w_proto_seg * semantic_loss, self.old_classes)
                        protection.update(enabled=True, effective_geometry_weight=effective_weight)
                    geometry_record['semantic_protection'] = protection
                    return semantic_loss, protected, geometry_loss, geometry_record

                cls, segs, fmap, cls_aux, type_seg, new_prototypes, prototype_outputs = model(
                    inputs, crops=roi_crops, n_iter=n_iter, prototype_objective=prototype_objective)
                type_seg_loss, protected_sep, geometry_sep, geometry_stats = prototype_outputs
'''
    tail = replace_once(tail, forward, before_forward)
    tail = replace_once(tail, '                T = 0.1\n                type_seg = type_seg / T\n', '')
    tail = replace_once(tail,
        "                type_seg = F.interpolate(type_seg, size=refined_pseudo_label.shape[1:], mode='bilinear', align_corners=False)\n", '')
    tail = replace_once(tail, '''                type_seg_loss = get_seg_loss(type_seg, mixed_pseudo_label.type(torch.long), class_weight=self.class_weight,
                                                  ignore_index=args.ignore_index)
''', '')
    tail = replace_once(tail, '+ pixel_kd + geometry_sep)', '+ pixel_kd + geometry_sep + protected_sep)')
    tail = replace_once(tail, '+ args.w_geometry_sep * pair_ramp * geometry_sep\n', '+ protected_sep\n')
    log_marker = '''                    with open(osp.join(args.work_dir,'geometry_metrics.jsonl'),'a') as handle:
                        handle.write(json.dumps(geometry_stats,allow_nan=False)+'\\n')
'''
    tail = replace_once(tail, log_marker, log_marker+'''                    protection_record = dict(geometry_stats['semantic_protection'])
                    protection_record.update(step=self.step, iteration=n_iter+1,
                        active=n_iter>=args.loss_warmup_iters,
                        weight=args.w_geometry_sep, ramp=pair_ramp)
                    with open(osp.join(args.work_dir,'semantic_guard_metrics.jsonl'),'a') as handle:
                        handle.write(json.dumps(protection_record,allow_nan=False)+'\\n')
''')
    result = prefix+tail
    ast.parse(result)
    return result


def helper_rng_audit(cam_text, par_text):
    """Inspect the actual moved functions and their refinement/PAR callees."""
    cam_functions = {'cam_to_label', 'refine_cams_with_bkg_v2', '_refine_cams',
                     'get_mixed_label', 'filter_cam_pesudo_label'}
    nodes = [node for node in ast.parse(cam_text).body
             if isinstance(node, ast.FunctionDef) and node.name in cam_functions]
    if {node.name for node in nodes} != cam_functions:
        raise RuntimeError('Unexpected CAM/PAR helper source')
    nodes += list(ast.parse(par_text).body)
    bad = []
    for root in nodes:
        for node in ast.walk(root):
            if not isinstance(node, ast.Call):
                continue
            name = ast.unparse(node.func)
            parts = name.lower().split('.')
            if any(part.startswith(('rand', 'dropout')) or part in {'bernoulli', 'multinomial', 'normal'}
                   for part in parts):
                bad.append(name)
    if bad:
        raise RuntimeError(f'Moving helpers would change RNG ordering: {bad}')
    return {'moved_helpers_checked': sorted(cam_functions), 'PAR_checked': True,
        'rng_calls_found': [],
        'ordering': 'Existing CAM calls and crop_from_roi_neg(randperm) stay before relocated deterministic helpers and the main stochastic forward.'}


def _atomic_source(path, text, safe_path):
    path = safe_path(path)
    temporary = safe_path(path.with_name(path.name+f'.{os.getpid()}.semantic.tmp'))
    with temporary.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--src', type=Path, default=CANDIDATE,
                        help='Must be the isolated b_semantic/src copy, never a_geometry')
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(PRIMARY))
    from kd_runtime import safe_path, atomic_json, digest, now
    import json
    candidate = safe_path(args.src)
    if candidate.resolve() != CANDIDATE.resolve() or candidate.resolve() == PRIMARY.resolve():
        raise RuntimeError('Only the isolated b_semantic/src can be patched')
    receipt_path = safe_path(candidate.parent/'semantic_integration.json')
    if receipt_path.exists():
        raise RuntimeError('Refusing duplicate semantic candidate patch')
    study = json.loads((ROOT/'runs/prototype_sep_v1/formal/a_geometry/study.json').read_text(encoding='utf-8'))
    for relative, expected in study['source_sha256'].items():
        if digest(PRIMARY/relative) != expected or digest(candidate/relative) != expected:
            raise RuntimeError(f'Expected an unchanged complete copy of frozen primary source: {relative}')
    kd_relative = 'model/pixel_kd.py'
    optimized_kd = ROOT/'experiments/kd_parallel_v1/b_relational/src/model/pixel_kd.py'
    kd_sha = digest(candidate/kd_relative)
    if kd_sha != digest(optimized_kd):
        raise RuntimeError('Candidate KD must remain byte-identical to optimized KD')
    model_path = safe_path(candidate/'model/model_seg_neg.py')
    trainer_path = safe_path(candidate/'continual/Trainer.py')
    original_model, original_trainer = model_path.read_text(encoding='utf-8'), trainer_path.read_text(encoding='utf-8')
    changed_model, changed_trainer = patch_model(original_model), patch_trainer(original_trainer)
    module_source = Path(__file__).resolve().parent/'semantic_protected_sep.py'
    module_text = module_source.read_text(encoding='utf-8')
    ast.parse(module_text)
    rng_audit = helper_rng_audit((PRIMARY/'utils/camutils.py').read_text(encoding='utf-8'),
                                (PRIMARY/'model/PAR.py').read_text(encoding='utf-8'))
    # Parse every edit before any mutation, then write only isolated B files.
    _atomic_source(model_path, changed_model, safe_path)
    _atomic_source(trainer_path, changed_trainer, safe_path)
    _atomic_source(candidate/'model/semantic_protected_sep.py', module_text, safe_path)
    if digest(candidate/kd_relative) != kd_sha:
        raise RuntimeError('KD unexpectedly changed during patch')
    atomic_json(receipt_path, {'schema': 1, 'utc': now(), 'candidate': str(candidate),
        'primary_frozen_source': str(PRIMARY), 'primary_source_sha256': study['source_sha256'],
        'changed_source_sha256': {str(path.relative_to(candidate)): digest(path)
            for path in [model_path, trainer_path, candidate/'model/semantic_protected_sep.py']},
        'patch_script_sha256': digest(Path(__file__)), 'optimized_kd_sha256_unchanged': kd_sha,
        'model_state_dict_and_optimizer': 'No modules/parameters added; get_param_groups and constructors unchanged.',
        'DDP': 'find_unused_parameters=True retained; prototype callback runs on original nonleaf p before _DDPSink.',
        'objective': 'Original semantic component once; already-weighted protected SEP once; warmup SEP0, no norm cap or parameter grid.',
        'rng_audit': rng_audit,
        'logs': {'original_geometry': 'geometry_metrics.jsonl with nested semantic_protection',
                 'semantic_guard': 'semantic_guard_metrics.jsonl, original log cadence, rank0 only'},
        'scope': 'Prepared isolated candidate only. No training, checkpoints, teachers, optimizer state or primary source modified.'})
    print(json.dumps({'patched': True, 'src': str(candidate), 'receipt': str(receipt_path)}))


if __name__ == '__main__':
    main()
