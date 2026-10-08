"""Read-only same-candidate endpoint GT diagnostic for c_new_anchor.

Only the completed stage2/8000 endpoint is supported. Reuse the fixed128
validation-prefix protocol and deterministic weak_signals from the frozen
prototype diagnostic; do not run gradient diagnostics, update online states,
or feed pixel GT to any inference/selector/loss operation. Four successful full
evaluators and the independent complete candidate audit must precede this run.
The unique output is U/new_anchor_endpoint_gt_diagnostic.json.
"""
from pathlib import Path
import argparse
import json
import os
import random
import sys

sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

ROOT = Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto')
EXPERIMENT = ROOT/'experiments/prototype_sep_v1'
UNIT = ROOT/'runs/prototype_sep_v1'
SRC = EXPERIMENT/'c_new_anchor/src'
RUN = UNIT/'formal/c_new_anchor'
OPT = ROOT/'runs/kd_parallel_v1/formal/b_relational'
OUTPUT = UNIT/'new_anchor_endpoint_gt_diagnostic.json'
AUDIT_FLAGS = ['all_training_and_endpoint_requirements_complete',
               'all_semantic_sep_requirements_complete',
               'all_best_pipeline_requirements_complete',
               'all_new_anchor_requirements_complete']
PROTOCOL_SHA = '743677a64fa835a91ff4652beab3c4485a5d18004457c9e7788289dcc68ea2b0'


def valid_gt_fractions(raw):
    """Add valid-GT fractions; original all-pixel fractions retain void pixels."""
    for views in raw['selected_directions'].values():
        for scopes in views.values():
            for value in scopes.values():
                valid = value['pixels']-value['gt_counts']['gt_void']
                weighted_valid = value['evidence_weight']-value['gt_evidence_weight']['gt_void']
                value['valid_gt_pixels'] = valid
                value['gt_source_fraction_valid_gt'] = value['gt_counts']['gt_source']/valid if valid else None
                value['gt_competitor_fraction_valid_gt'] = value['gt_counts']['gt_competitor']/valid if valid else None
                value['weighted_valid_gt_evidence'] = weighted_valid
                value['weighted_gt_source_fraction_valid_gt'] = value['gt_evidence_weight']['gt_source']/weighted_valid if weighted_valid > 0 else None
                value['weighted_gt_competitor_fraction_valid_gt'] = value['gt_evidence_weight']['gt_competitor']/weighted_valid if weighted_valid > 0 else None
    return raw


def anchor_group_summaries(result):
    summaries = {}
    for view, matrix in result['anchor_vs_gt_counts'].items():
        summaries[view] = {}
        for group, ids in [('old_foreground', list(range(1, 16))),
                           ('current_new_foreground', list(range(16, 21)))]:
            valid = sum(sum(matrix[i][:21]) for i in ids)
            correct = sum(matrix[i][i] for i in ids)
            void = sum(matrix[i][21] for i in ids)
            summaries[view][group] = {'source_class_ids': ids, 'valid_gt_pixels': valid,
                'gt_anchor_correct': correct, 'gt_anchor_correct_fraction_valid_gt': correct/valid if valid else None,
                'gt_void_pixels': void,
                'subset': 'All fixed-evidence anchor rows in this group; not the selected-disagreement subset.'}
        for scope in ['anchor_row', 'observed_direction']:
            selected = [scopes[scope] for scopes in result['selected_directions'][view].values()]
            valid = sum(value['valid_gt_pixels'] for value in selected)
            correct = sum(value['gt_counts']['gt_source'] for value in selected)
            competitor = sum(value['gt_counts']['gt_competitor'] for value in selected)
            summaries[view]['selected_new_source_'+scope] = {
                'valid_gt_pixels': valid, 'gt_source_pixels': correct, 'gt_competitor_pixels': competitor,
                'gt_source_fraction_valid_gt': correct/valid if valid else None,
                'gt_competitor_fraction_valid_gt': competitor/valid if valid else None,
                'selected_directions': len(selected),
                'subset': 'Final cached new-source selected directions only; each source has at most one selected partner.'}
    return {'views': summaries,
        'scope': 'Fixed128 validation-prefix, resize448 and CAM/PAR evidence; descriptive subset correctness only, not all-image population precision, calibration or learning weights.',
        'comparison_caveat': 'All-anchor-row summaries and selected-disagreement summaries condition on different pixels and must not be interchanged.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', type=int, default=0,
        help='Local CUDA ordinal within externally supplied CUDA_VISIBLE_DEVICES')
    args = parser.parse_args()
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT/'experiments/kd_pixel_v2'))
    from run_kd import environment
    os.environ.update(environment('8card'))
    sys.path.insert(0, str(SRC))
    sys.path.insert(0, str(EXPERIMENT))
    from kd_runtime import safe_path, atomic_json, digest, now
    output = safe_path(OUTPUT)
    if output.exists():
        raise RuntimeError(f'Refusing to replace an existing diagnostic: {output}')
    stage = safe_path(RUN/'10-5/step2')
    study = json.loads(safe_path(RUN/'study.json').read_text())
    cfg = json.loads(safe_path(stage/'config.json').read_text())
    audit = json.loads(safe_path(RUN/'completion_audit.json').read_text())
    if (study.get('arm') != 'c_new_anchor' or study.get('smoke') is not False or
            study.get('source') != str(SRC) or study.get('selector_schema') != 4 or
            study.get('source_anchor_policy') != 'current_new_only' or
            study.get('source_domain') != 'new_foreground_rows_only'):
        raise ValueError('Require the actual formal schema4 current-new-source candidate')
    if (json.loads(safe_path(RUN/'status.json').read_text()).get('status') != 'complete' or
            json.loads(safe_path(RUN/'evaluation_workers_complete.json').read_text()).get('returncodes') != [0]*4 or
            audit.get('arm') != 'c_new_anchor' or audit.get('status') != 'complete' or
            not all(audit.get(key) is True for key in AUDIT_FLAGS)):
        raise RuntimeError('Formal complete endpoint, four full evaluators and complete independent audit are required')
    actual = {str(path.relative_to(SRC)): digest(safe_path(path)) for path in SRC.rglob('*.py')
              if '__pycache__' not in path.parts}
    if not actual or actual != study['source_sha256']:
        raise RuntimeError('Full candidate source differs from its freeze')
    protocol_path = safe_path(EXPERIMENT/'diagnose_prototype_endpoint.py')
    if digest(protocol_path) != PROTOCOL_SHA:
        raise RuntimeError('Original fixed128 GT diagnostic protocol changed')
    import diagnose_prototype_endpoint as diagnostic
    if (cfg.get('step') != 2 or cfg.get('task') != '10-5' or cfg.get('crop_size') != 448
            or cfg.get('max_iters') != 8000 or cfg.get('proto_margin') != 0.
            or any(cfg.get(key) != .1 for key in ['w_geometry_sep', 'w_pixel_kd', 'w_proto_seg'])):
        raise ValueError('Require actual completed VOC10-5 stage2 margin0/.1 protocol')
    student_path = safe_path(stage/'checkpoints/model_final.pth')
    teacher_path = safe_path(Path(study['teacher']))
    expected_teacher = safe_path(OPT/'10-5/step1/checkpoints/model_final.pth')
    if (teacher_path != expected_teacher or Path(cfg['prev_checkpoint']) != teacher_path or
            study.get('inherited_stage1_run') != str(OPT) or
            digest(teacher_path) != study['teacher_sha256']):
        raise ValueError('Teacher must be the study frozen optimizedKD stage1 final, never an A/B teacher')
    teacher_receipt = json.loads(safe_path(OPT/'10-5/step1/training_complete.json').read_text())
    final_receipt = json.loads(safe_path(stage/'training_complete.json').read_text())
    if (teacher_receipt.get('returncode') != 0 or teacher_receipt.get('checkpoint_sha256') != digest(teacher_path)
            or final_receipt.get('returncode') != 0 or final_receipt.get('checkpoint_sha256') != digest(student_path)):
        raise RuntimeError('Student or teacher identity differs from successful training receipt')
    import numpy as np
    import torch
    import torch.nn.functional as F
    import tasks
    from datasets import voc
    from model.model_seg_neg import network
    from model.confusion_prototype_sep import confusion_prototype_sep_loss
    from model.online_directed_confusion import OnlineDirectedConfusion, build_confusion_evidence
    from model.geometry_pair_selector import GeometryPairSelector
    from model.PAR import PAR
    from utils.camutils import multi_scale_cam2, cam_to_label, refine_cams_with_bkg_v2, get_mixed_label
    from utils.imutils import denormalize_img2
    if not torch.cuda.is_available() or not 0 <= args.device < torch.cuda.device_count():
        raise ValueError('Expected an available explicitly selected CUDA device')
    torch.cuda.set_device(args.device)
    device = torch.device('cuda', args.device)
    torch.set_num_threads(1)
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    def load_model(step, saved):
        class_list = tasks.get_per_task_classes('voc', '10-5', step)
        model = network(backbone=cfg['backbone'], num_classes=sum(class_list), classes_list=class_list,
            pretrained=False, init_momentum=cfg['momentum'], aux_layer=cfg['aux_layer'])
        model.load_state_dict({k.removeprefix('module.'): v for k, v in saved['model_state'].items()}, strict=True)
        return model.to(device).eval().requires_grad_(False)
    saved = torch.load(student_path, map_location='cpu', weights_only=True, mmap=True)
    if saved.get('iteration') != 8000:
        raise ValueError('Final student must be the predefined stage2/8000 endpoint')
    classes = sum(tasks.get_per_task_classes('voc', '10-5', 2))
    old = sum(tasks.get_per_task_classes('voc', '10-5', 1))-1
    if (classes, old) != (21, 15):
        raise ValueError('Require21 total classes and15 old foreground classes')
    student = load_model(2, saved)
    observer = OnlineDirectedConfusion(classes, momentum=cfg['confusion_momentum'],
        high_threshold=cfg['high_thre'], low_threshold=cfg['low_thre'], stage=2).to(device)
    observer.load_state_dict(saved['online_confusion_state'], strict=True)
    selector = GeometryPairSelector(classes, old_classes=old, stage=2,
        refresh_interval=cfg['pair_refresh_interval'], min_row_images=cfg['pair_min_row_images'],
        min_pair_images=cfg['pair_min_pair_images'], min_rate=cfg['pair_min_rate'],
        min_updates=cfg['pair_min_updates'], ramp_updates=cfg['pair_ramp_updates'],
        max_stale_updates=cfg['pair_max_stale_updates']).to(device)
    selector.load_state_dict(saved['geometry_selector_state'], strict=True)
    del saved
    targets = selector.targets.detach().clone()
    exported = selector.export()
    if (exported.get('schema') != 4 or exported.get('source_anchor_policy') != 'current_new_only'
            or targets.shape != (21,) or bool(((targets < -1) | (targets >= 21)).any())
            or bool((targets[:16] != -1).any())
            or any(j == 0 or i == j for i, j in enumerate(targets.cpu().tolist()) if j >= 0)
            or int(observer.updates) != 6000 or int(observer.seen_images) != 48000):
        raise ValueError('Invalid fresh6000-update saved schema4 new-source selector/observer endpoint')
    saved_teacher = torch.load(teacher_path, map_location='cpu', weights_only=True, mmap=True)
    # The archived final teacher intentionally contains model_state only.
    # Confirm its actual tensors against the published stage1/8000 checkpoint
    # rather than requiring a metadata field absent from the older format.
    teacher_evaluated_path = safe_path(OPT/'10-5/step1/checkpoints/model_iter_8000.pth')
    teacher_job = json.loads(safe_path(OPT/'eval_queue/step1_iter8000.json').read_text())
    teacher_result = json.loads(safe_path(OPT/'evaluations/step1_iter8000/result.json').read_text())
    teacher_identity = json.loads(safe_path(OPT/'10-5/step1/endpoint_identity.json').read_text())
    teacher_evaluated_sha = digest(teacher_evaluated_path)
    if (saved_teacher.get('iteration') not in (None, 8000)
            or teacher_job.get('step') != 1 or teacher_result.get('step') != 1
            or teacher_job.get('iteration') != 8000 or teacher_result.get('iteration') != 8000
            or teacher_job.get('checkpoint') != str(teacher_evaluated_path)
            or teacher_job.get('checkpoint_sha256') != teacher_evaluated_sha
            or teacher_result.get('checkpoint_sha256') != teacher_evaluated_sha
            or teacher_identity.get('final_sha256') != digest(teacher_path)
            or teacher_identity.get('evaluated_sha256') != teacher_evaluated_sha
            or teacher_identity.get('all_model_tensors_exactly_equal') is not True):
        raise ValueError('Original teacher must match its published stage1/8000 identity')
    evaluated_teacher = torch.load(teacher_evaluated_path, map_location='cpu', weights_only=True, mmap=True)
    actual_teacher_tensors = saved_teacher['model_state']
    evaluated_teacher_tensors = evaluated_teacher['model_state']
    if (teacher_identity.get('verified_tensor_count') != len(actual_teacher_tensors)
            or actual_teacher_tensors.keys() != evaluated_teacher_tensors.keys()
            or any(not torch.equal(value, evaluated_teacher_tensors[name])
                   for name, value in actual_teacher_tensors.items())):
        raise ValueError('Archived final teacher tensors differ from evaluated stage1/8000')
    teacher_format_verified = {'checkpoint_keys': list(saved_teacher),
        'saved_iteration': saved_teacher.get('iteration'), 'endpoint_iteration': 8000,
        'evaluated_checkpoint': str(teacher_evaluated_path),
        'evaluated_sha256': teacher_evaluated_sha,
        'all_model_tensors_exactly_equal': True, 'tensor_count': len(actual_teacher_tensors)}
    del evaluated_teacher, actual_teacher_tensors, evaluated_teacher_tensors
    teacher = load_model(1, saved_teacher)
    del saved_teacher
    versions = {key: {name: p._version for name, p in model.named_parameters()}
                for key, model in [('student', student), ('teacher', teacher)]}
    buffers = {key: {name: value.detach().cpu().clone() for name, value in model.named_buffers()}
               for key, model in [('student', student), ('teacher', teacher)]}
    before_observer = diagnostic.freeze_state(observer, torch)
    before_selector = diagnostic.freeze_state(selector, torch)
    ctx = dict(torch=torch, F=F, np=np, cfg=cfg, device=device, student=student, teacher=teacher,
        step=2, classes=classes, old_classes=old, new_ids=list(range(16, 21)),
        voc=voc, par=PAR(num_iter=10, dilations=[1, 2, 4, 8, 12, 24]).to(device).eval(),
        targets=targets, ramp=selector.ramp(observer), digest=digest,
        sep_loss=confusion_prototype_sep_loss, build_confusion_evidence=build_confusion_evidence,
        multi_scale_cam2=multi_scale_cam2, cam_to_label=cam_to_label,
        refine_cams_with_bkg_v2=refine_cams_with_bkg_v2, get_mixed_label=get_mixed_label,
        denormalize_img2=denormalize_img2)
    result = valid_gt_fractions(diagnostic.gt_mode(ctx))
    if any(int(direction.split('->')[0]) <= old for view in result['selected_directions'].values() for direction in view):
        raise RuntimeError('GT diagnostic unexpectedly included a forbidden old/BG selected source')
    for key, model in [('student', student), ('teacher', teacher)]:
        if any(p._version != versions[key][name] for name, p in model.named_parameters()):
            raise RuntimeError(f'Diagnostic changed a {key} parameter')
        if any(not torch.equal(value.detach().cpu(), buffers[key][name]) for name, value in model.named_buffers()):
            raise RuntimeError(f'Diagnostic changed a {key} buffer')
        if any(p.grad is not None or p.requires_grad for p in model.parameters()):
            raise RuntimeError(f'Unexpected gradients or trainable {key}')
    if (not diagnostic.tensor_state_equal(before_observer, observer.state_dict(), torch)
            or not diagnostic.tensor_state_equal(before_selector, selector.state_dict(), torch)):
        raise RuntimeError('Diagnostic changed the saved observer or selector')
    atomic_json(output, {'schema': 1, 'created_utc': now(), 'mode': 'gt', 'passed': True,
        'candidate': 'c_new_anchor', 'stage': 2, 'iteration': 8000, 'final_checkpoint': True,
        'class_names': diagnostic.CLASSES, 'model_class_ids': list(range(21)),
        'old_foreground_classes': 15, 'new_foreground_class_ids': list(range(16, 21)),
        'student_checkpoint': str(student_path), 'student_sha256': digest(student_path),
        'teacher_checkpoint': str(teacher_path), 'teacher_sha256': digest(teacher_path),
        'teacher_lineage': 'Archived optimizedKD stage1 final from this study; no A/B teacher replay.',
        'teacher_checkpoint_format_verified': teacher_format_verified,
        'observer_and_selector_checkpoint': str(student_path),
        'saved_observer_updates': int(observer.updates), 'saved_observer_seen_images': int(observer.seen_images),
        'saved_selector': exported, 'source_anchor_policy': 'current_new_only',
        'saved_selector_usage': 'Exact candidate final cached targets; no update/refresh/reselection or cross-candidate replay.',
        'geometry_ramp': ctx['ramp'], 'source_sha256': study['source_sha256'],
        'diagnostic_script_sha256': digest(Path(__file__)), 'reused_GT_protocol_sha256': PROTOCOL_SHA,
        'completion_audit_sha256': digest(RUN/'completion_audit.json'),
        'state_checks': {'student_parameter_versions_unchanged': True, 'student_buffers_unchanged': True,
            'teacher_parameter_versions_unchanged': True, 'teacher_buffers_unchanged': True,
            'observer_unchanged': True, 'selector_unchanged': True,
            'student_frozen_no_gradients': True, 'teacher_frozen_no_gradients': True},
        'anchor_group_summary': anchor_group_summaries(result),
        'GT_role': 'GT partitions evidence only after predictions/anchors are fixed; never feeds selector, supervision or loss strengths. Full1449 evaluation is the effect criterion.',
        **result})
    print(json.dumps({'passed': True, 'mode': 'gt', 'output': str(output)}), flush=True)


if __name__ == '__main__':
    main()
