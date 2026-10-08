"""Read-only hierarchical KD and trusted-background readiness. GT diagnoses fixed gates only."""
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
SRC = ROOT/'experiments/hierarchy_kd_v2/src'
RUN = ROOT/'runs/hierarchy_kd_v2/formal/h2'
OPT = ROOT/'runs/kd_parallel_v1/formal/b_relational'
OUTPUT = ROOT/'runs/hierarchy_kd_v2/old_agreement_readiness.json'
AUDIT_FLAGS = ['all_training_and_endpoint_requirements_complete',
               'all_semantic_sep_requirements_complete',
               'all_best_pipeline_requirements_complete',
               'all_new_anchor_requirements_complete']
PROTOCOL_SHA = '743677a64fa835a91ff4652beab3c4485a5d18004457c9e7788289dcc68ea2b0'


GT_GROUPS = ('correct_old_source', 'other_old_foreground', 'current_new_foreground',
             'background', 'void')
SUBSETS = ('teacher_foreground_student_background', 'current_KD_eligible_student_background',
           'current_KD_nonzero_student_background', 'teacher_PAR_agreement_student_background',
           'trusted_strongest_CAM_intersection_student_background',
           'trusted_intersection_new_CAM_veto_student_background',
           'strict_candidate_nonzero_student_background', 'current_KD_nonzero_all_predictions',
           'strict_candidate_nonzero_all_predictions', 'production_mass_all_predictions', 'production_mass_student_teacher_agree')


def blank_statistics():
    return {'pixels': 0, 'weight_sum': 0., 'gt_counts': {key: 0 for key in GT_GROUPS},
            'gt_weight_sums': {key: 0. for key in GT_GROUPS}}


def finish_statistics(raw):
    valid = raw['pixels'] - raw['gt_counts']['void']
    weighted_valid = raw['weight_sum'] - raw['gt_weight_sums']['void']
    return {**raw, 'valid_gt_pixels': valid, 'weighted_valid_gt_mass': weighted_valid,
            'GT_source_precision_valid_gt': raw['gt_counts']['correct_old_source']/valid if valid else None,
            'weighted_GT_source_precision_valid_gt': raw['gt_weight_sums']['correct_old_source']/weighted_valid
                if weighted_valid > 0 else None}


def old_background_mode(ctx, diagnostic):
    torch, F, cfg, device = ctx['torch'], ctx['F'], ctx['cfg'], ctx['device']
    old, classes = ctx['old_classes'], ctx['classes']
    mass_dirs,bg_dirs=ctx['select_mass_directions'](ctx['observer'],old)
    ds = ctx['voc'].VOC12SegDataset(root_dir=cfg['data_folder'], name_list_dir=cfg['list_folder'],
        split=cfg['val_set'], stage='val', aug=False, ignore_index=cfg['ignore_index'],
        num_classes=classes, tasks='10-5', step=2)
    ds.label_dir = cfg['val_label_dir']
    if len(ds) < 128:
        raise ValueError('Validation list requires at least128 images')
    indices = list(range(128))
    names = [str(ds.name_list[index]) for index in indices]
    import hashlib
    prefix_sha = hashlib.sha256('\n'.join(names).encode()).hexdigest()
    prior_path = UNIT/'endpoint_gt_diagnostic.json'
    if prior_path.is_file():
        previous = json.loads(prior_path.read_text())
        if previous.get('validation_prefix128_sha256') != prefix_sha:
            raise RuntimeError('This diagnostic must use the same existing128 validation prefix')
    totals = {key: blank_statistics() for key in SUBSETS}
    by_teacher_class = {str(i): {key: blank_statistics() for key in SUBSETS} for i in range(1, old+1)}
    # Full native-grid GT-row histograms. The extra GT-void row is separately retained.
    prediction_gt = torch.zeros(22, classes, dtype=torch.int64, device=device)
    teacher_gt = torch.zeros(22, old+1, dtype=torch.int64, device=device)
    old_background_teacher = torch.zeros(old+1, old+1, dtype=torch.int64, device=device)
    native_shapes, records = set(), []
    bg_totals = {}
    with torch.inference_mode():
        for index in indices:
            name, image, pixel_gt, image_tags = ds[index]
            inputs = F.interpolate(torch.as_tensor(image, device=device).float()[None],
                                   size=(448, 448), mode='bilinear', align_corners=False)
            tags = torch.as_tensor(image_tags, device=device)[None]
            boxes = torch.tensor([[0, 448, 0, 448]])
            teacher_logits, valid_cams, refined, _ = diagnostic.weak_signals(ctx, inputs, tags, boxes)
            _, student_logits, _, _, _, _ = ctx['student'](inputs, crops=[])
            if not torch.isfinite(student_logits).all() or not torch.isfinite(teacher_logits).all():
                raise FloatingPointError('Nonfinite inference logits')
            size = student_logits.shape[-2:]
            native_shapes.add(tuple(size))
            teacher_native = F.interpolate(teacher_logits.detach(), size=size, mode='bilinear', align_corners=False)
            teacher_prediction = teacher_native.argmax(1)
            student_prediction = student_logits.argmax(1)
            par_native = F.interpolate(refined[:, None].float(), size=size, mode='nearest')[:, 0].long()
            cams_native = F.interpolate(valid_cams.detach(), size=size, mode='bilinear', align_corners=False).clamp(0, 1)
            valid_full = torch.zeros_like(refined, dtype=torch.float)
            for item, coordinates in enumerate(boxes):
                top, bottom, left, right = [int(value) for value in coordinates]
                valid_full[item, top:bottom, left:right] = 1
            valid = F.interpolate(valid_full[:, None], size=size, mode='nearest')[:, 0] > 0
            native_boxes = torch.tensor([[0, size[0], 0, size[1]]])
            evidence = ctx['build_confusion_evidence'](student_logits, par_native, cams_native, native_boxes,
                high_threshold=cfg['high_thre'], low_threshold=cfg['low_thre'])
            # Exact existing optimized-KD eligibility/weight construction. No KL or gradients.
            teacher_foreground = (teacher_prediction > 0) & (teacher_prediction <= old)
            new_PAR_region = (par_native >= old+1) & (par_native < classes)
            new_CAM = cams_native[:, old:].amax(1)
            old_support = cams_native[:, :old].gather(1,
                (teacher_prediction-1).clamp_min(0)[:, None])[:, 0]
            confidence = teacher_native.sigmoid().gather(1, teacher_prediction[:, None])[:, 0]
            reliability = (2*confidence-1).clamp_min(0).square()
            eligible = valid & ~new_PAR_region & teacher_foreground & (old_support >= .25)
            current_weights = eligible * reliability * old_support * (1-new_CAM).square()
            nonzero = current_weights > 0
            PAR_agrees = eligible & (par_native == teacher_prediction)
            trusted = PAR_agrees & evidence['accepted'] & (par_native > 0) & (par_native <= old)
            # Additional strict intersection: use the existing high CAM threshold as a
            # strong-new-evidence veto, retaining the existing continuous discount otherwise.
            trusted_new_veto = trusted & (new_CAM < cfg['high_thre'])
            strict = trusted_new_veto & nonzero
            student_BG = student_prediction == 0
            masks = {
                'teacher_foreground_student_background': valid & teacher_foreground & student_BG,
                'current_KD_eligible_student_background': eligible & student_BG,
                'current_KD_nonzero_student_background': nonzero & student_BG,
                'teacher_PAR_agreement_student_background': PAR_agrees & student_BG,
                'trusted_strongest_CAM_intersection_student_background': trusted & student_BG,
                'trusted_intersection_new_CAM_veto_student_background': trusted_new_veto & student_BG,
                'strict_candidate_nonzero_student_background': strict & student_BG,
                'current_KD_nonzero_all_predictions': nonzero,
                'strict_candidate_nonzero_all_predictions': strict,
                'production_mass_all_predictions': strict & mass_dirs[teacher_prediction] &
                    (ctx['group_log_prob'](student_logits,old,cfg['kd_temperature'])[:,0] <
                     ctx['group_log_prob'](teacher_native,old,cfg['kd_temperature'])[:,0]),
            }
            masks['production_mass_student_teacher_agree']=masks['production_mass_all_predictions'] & (student_prediction==teacher_prediction)
            bg_base = valid & (par_native == 0) & (cams_native.amax(1) < cfg['low_thre'])
            bg_gates = {'PAR_low_CAM': bg_base,
                        'PAR_low_CAM_teacher_BG': bg_base & (teacher_prediction == 0),
                        'PAR_BG_teacher_BG': valid & (par_native == 0) & (teacher_prediction == 0),
                        'PAR_BG_teacher_BG_no_strong_new': valid & (par_native == 0) & (teacher_prediction == 0) & (new_CAM < cfg['high_thre']),
                        'teacher_BG_no_strong_CAM': valid & (teacher_prediction == 0) & (cams_native.amax(1) < cfg['high_thre']),
                        'PAR_BG': valid & (par_native == 0)}
            # Only current-new image-level presence is supplied as supervision.
            new_present = tags[:, -5:].gather(1, (student_prediction-old-1).clamp(0,4).flatten(1)).reshape_as(student_prediction)
            absent_new = valid & (student_prediction > old) & (new_present == 0)
            bg_gates['image_absent_new'] = absent_new
            bg_gates['image_absent_new_teacher_BG'] = absent_new & (teacher_prediction == 0)
            bg_gates['image_absent_new_teacher_BG_PAR_BG'] = absent_new & (teacher_prediction == 0) & (par_native == 0)
            production_absent = absent_new & (teacher_prediction==0) & bg_dirs[student_prediction]
            production_bg = (valid & (par_native==0) & (teacher_prediction==0) & (new_CAM<cfg['high_thre'])
                             & (student_prediction>0) & bg_dirs[student_prediction] & ~production_absent)
            bg_gates['production_absent_complement']=production_absent
            bg_gates['production_background_pair']=production_bg
            _,_,loss_stats=ctx['hierarchical_kd_loss'](student_logits,teacher_logits,refined,valid_cams,boxes,
                tags[:,-5:],mass_dirs,bg_dirs,cfg['kd_temperature'],cfg['high_thre'])
            assert loss_stats['mass_pixels']==int(masks['production_mass_all_predictions'].sum())
            assert loss_stats['background_pair_pixels']==int(production_bg.sum())
            assert loss_stats['absent_new_pixels']==int(production_absent.sum())
            # Gates above contain predictions, weak labels and CAM/PAR only.
            # Pixel GT first enters here to partition fixed masks; never gate construction.
            gt = F.interpolate(torch.as_tensor(pixel_gt, device=device).float()[None, None],
                               size=size, mode='nearest')[:, 0].long()
            valid_gt = (gt >= 0) & (gt < classes)
            gt_column = torch.where(valid_gt, gt, torch.full_like(gt, 21))
            categories = {
                'correct_old_source': valid_gt & (gt == teacher_prediction) & teacher_foreground,
                'other_old_foreground': (gt > 0) & (gt <= old) & (gt != teacher_prediction),
                'current_new_foreground': (gt > old) & (gt < classes),
                'background': gt == 0,
                'void': ~valid_gt,
            }
            prediction_gt += torch.bincount((gt_column[valid]*classes+student_prediction[valid]).flatten(),
                                             minlength=22*classes).reshape(22, classes)
            teacher_gt += torch.bincount((gt_column[valid]*(old+1)+teacher_prediction[valid]).flatten(),
                                          minlength=22*(old+1)).reshape(22, old+1)
            for gate_name, gate_mask in bg_gates.items():
                for subset_name, subset_mask in [('all', valid), ('student_foreground', ~student_BG),
                                                 ('student_new', student_prediction > old)]:
                    key = gate_name + '/' + subset_name
                    matrix = torch.bincount((gt_column[gate_mask & subset_mask]*classes +
                        student_prediction[gate_mask & subset_mask]).flatten(), minlength=22*classes).reshape(22, classes)
                    bg_totals[key] = bg_totals.get(key, torch.zeros_like(matrix)) + matrix
            true_old_BG = valid & student_BG & (gt > 0) & (gt <= old)
            old_background_teacher += torch.bincount(
                (gt[true_old_BG]*(old+1)+teacher_prediction[true_old_BG]).flatten(),
                minlength=(old+1)*(old+1)).reshape(old+1, old+1)
            image_record = {'index': index, 'image': str(name), 'native_grid': list(size), 'subsets': {}}
            for key, mask in masks.items():
                local = blank_statistics()
                local['pixels'] = int(mask.sum())
                local['weight_sum'] = float(current_weights[mask].double().sum())
                for category, category_mask in categories.items():
                    selected = mask & category_mask
                    local['gt_counts'][category] = int(selected.sum())
                    local['gt_weight_sums'][category] = float(current_weights[selected].double().sum())
                if sum(local['gt_counts'].values()) != local['pixels']:
                    raise RuntimeError('GT partitions failed to cover a fixed candidate mask')
                for target in (totals[key],):
                    target['pixels'] += local['pixels']
                    target['weight_sum'] += local['weight_sum']
                    for category in GT_GROUPS:
                        target['gt_counts'][category] += local['gt_counts'][category]
                        target['gt_weight_sums'][category] += local['gt_weight_sums'][category]
                for i in range(1, old+1):
                    selected_class = mask & (teacher_prediction == i)
                    target = by_teacher_class[str(i)][key]
                    target['pixels'] += int(selected_class.sum())
                    target['weight_sum'] += float(current_weights[selected_class].double().sum())
                    for category, category_mask in categories.items():
                        selected = selected_class & category_mask
                        target['gt_counts'][category] += int(selected.sum())
                        target['gt_weight_sums'][category] += float(current_weights[selected].double().sum())
                image_record['subsets'][key] = finish_statistics(local)
            records.append(image_record)
            if (index+1) % 8 == 0:
                print(json.dumps({'mode': 'old_background_kd_readiness', 'images_completed': index+1}), flush=True)
    hist, teacher_hist, misses = (value.cpu().tolist() for value in
                                  (prediction_gt, teacher_gt, old_background_teacher))
    per_GT_old = []
    for i in range(1, old+1):
        pixels, missed = sum(hist[i]), sum(misses[i])
        recoverable = misses[i][i]
        per_GT_old.append({'class_id': i, 'class_name': diagnostic.CLASSES[i],
            'valid_GT_source_pixels': pixels, 'student_correct_source_pixels': hist[i][i],
            'student_source_to_background_pixels': missed,
            'student_source_to_background_fraction_of_full_GT_source_row': missed/pixels if pixels else None,
            'teacher_correct_source_among_student_background': recoverable,
            'teacher_correct_source_fraction_of_student_background': recoverable/missed if missed else None,
            'teacher_also_background_among_student_background': misses[i][0],
            'teacher_wrong_other_old_among_student_background': missed-recoverable-misses[i][0],
            'teacher_label_counts_among_student_background': misses[i]})
    total_missed = sum(row['student_source_to_background_pixels'] for row in per_GT_old)
    recoverable = sum(row['teacher_correct_source_among_student_background'] for row in per_GT_old)
    current_correct = totals['current_KD_nonzero_student_background']['gt_counts']['correct_old_source']
    strict_correct = totals['strict_candidate_nonzero_student_background']['gt_counts']['correct_old_source']
    coverage = {'GT_old_to_student_background_pixels': total_missed,
        'teacher_correct_same_old_pixels_within_those_misses': recoverable,
        'teacher_correct_fraction_of_GT_old_background_misses': recoverable/total_missed if total_missed else None,
        'current_KD_nonzero_correct_old_BG_pixels': current_correct,
        'strict_candidate_correct_old_BG_pixels': strict_correct,
        'current_gate_coverage_of_teacher_recoverable_misses': current_correct/recoverable if recoverable else None,
        'strict_gate_coverage_of_teacher_recoverable_misses': strict_correct/recoverable if recoverable else None,
        'strict_gate_coverage_of_all_GT_old_background_misses': strict_correct/total_missed if total_missed else None}
    return {'sample_indices': indices, 'sample_names': names,
        'validation_prefix128_sha256': prefix_sha, 'native_student_logit_grids': [list(x) for x in sorted(native_shapes)],
        'sampling': 'Same fixed first128 images of stage2 validation, no augmentation, input resize448. Counts at native student grid; nearest-neighbor GT.',
        'student_prediction_vs_GT_counts': hist, 'teacher_prediction_vs_GT_counts': teacher_hist,
        'histogram_GT_rows': list(range(21))+['void'],
        'student_histogram_columns': list(range(classes)), 'teacher_histogram_columns': list(range(old+1)),
        'old_GT_background_miss_teacher_distribution': per_GT_old,
        'gate_subsets': {key: finish_statistics(raw) for key, raw in totals.items()},
        'per_teacher_old_source_class': {key: {name: finish_statistics(raw) for name, raw in values.items()}
                                        for key, values in by_teacher_class.items()},
        'readiness_coverage': coverage, 'records': records,
        'production_directions':{'old_to_BG_or_new':mass_dirs.nonzero().flatten().tolist(),'BG_to_FG':bg_dirs.nonzero().flatten().tolist()},
        'background_gates': {key: {'GT_rows_student_columns': matrix.cpu().tolist(),
            'valid_pixels': int(matrix[:21].sum()), 'true_background_pixels': int(matrix[0].sum()),
            'true_old_pixels': int(matrix[1:old+1].sum()), 'true_new_pixels': int(matrix[old+1:21].sum()),
            'background_precision': float(matrix[0].sum()/matrix[:21].sum().clamp_min(1)),
            'covered_BG_to_foreground': int(matrix[0,1:].sum()),
            'coverage_BG_to_foreground': float(matrix[0,1:].sum()/prediction_gt[0,1:].sum().clamp_min(1)),
            'covered_BG_to_new': int(matrix[0,old+1:].sum()),
            'coverage_BG_to_new': float(matrix[0,old+1:].sum()/prediction_gt[0,old+1:].sum().clamp_min(1))}
            for key,matrix in bg_totals.items()},
        'definitions': {
            'GT_old_to_BG': 'Original GT classi in1..15 and current main-head native argmax0; teacher correctness is measured afterward.',
            'current_KD_eligible': 'Exact optimizedKD mask: valid ROI, PAR notnew, teacher foreground and same-old-CAM support>=0.25.',
            'current_KD_nonzero': 'Current eligible mask plus positive existing reliability*oldCAM*(1-maxNewCAM)^2 weight.',
            'strict_intersection': 'Current eligible AND PAR==teacher classi AND accepted strongestCAM matches PARi at existing high threshold with positive top2 gap.',
            'strong_new_CAM_veto': 'Additional diagnostic intersection maxNewCAM<existing high threshold; reported separately from the original optimizedKD mask.',
            'strict_candidate': 'Strict intersection plus strong-new-CAM veto and positive original KD weight. This diagnostic does not update the saved training state.',
            'weights': 'Only existing optimizedKD evidence weights are summed for descriptive overlap. This is not the complete sqrt-class-count-normalized KD objective or measured gradients.',
            'GT_source_precision': 'GT==teacher-sourcei divided by all nonvoid fixed-mask pixels, including otherold/new/BG; void excluded and separately counted.',
            'class_denominators': 'GT-old miss fractions use each complete native-grid GTi row. Gate precision uses each teacher-source candidate subset. Neither is an offdiagonal-renormalized population accuracy.'},
        'limitations': ['Fixed128 prefix and native resize448/logit-grid alignment are not the full1449 original-GT endpoint or population performance.',
            'Teacher-correct coverage is a diagnostic upper bound on covered student-only misses, not recoverable-IoU or guaranteed training improvement.',
            'Teacher/PAR/CAM gates are constructed without pixel GT. Their measured correctness here cannot become exact learning weights.',
            'The existing mixed-pseudo-label BCE already supervises foreground/background. A gap in conditional old-class KD does not prove missing background supervision causes these errors.',
            'Current eligible versus strict counts condition on different masks. Higher precision with tiny coverage is not enough to justify training.',
            'Evidence weights are uncalibrated and this single-image diagnostic does not reproduce global8 sqrt-class balancing or AdamW updates.',
            'Student-background candidates whose GT is new/background/otherold expose possible costs of old-foreground preservation; they are reported rather than discarded using GT.',
            'These same development-validation images previously informed source restriction. This is not independent validation or causal evidence.']}


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
    audit = {}
    if (study.get('arm') != 'h2' or study.get('smoke') is not False or study.get('source') != str(SRC)
            or json.loads(safe_path(RUN/'status.json').read_text()).get('status') != 'complete'):
        raise RuntimeError('Need complete frozen h1 endpoint')
    endpoint = json.loads(safe_path(RUN/'evaluations/step2_iter8000/result.json').read_text())
    reference = json.loads(safe_path(OPT/'evaluations/step2_iter8000/result.json').read_text())
    if (endpoint.get('step') != 2 or reference.get('step') != 2
            or endpoint.get('iteration') != 8000 or reference.get('iteration') != 8000
            or endpoint.get('images') != 1449 or reference.get('images') != 1449):
        raise RuntimeError('Conditional readiness diagnostic requires both complete full1449 endpoints')
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
    from model.hierarchical_kd import select_mass_directions, hierarchical_kd_loss, group_log_prob
    from model.hierarchy_observer import HierarchyObserver
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
    hierarchy_observer=HierarchyObserver(classes,old,momentum=cfg['confusion_momentum'],
        high_threshold=cfg['high_thre'],low_threshold=cfg['low_thre'],stage=2).to(device)
    hierarchy_observer.load_state_dict(saved['hierarchy_observer_state'],strict=True)
    before_hierarchy=diagnostic.freeze_state(hierarchy_observer,torch)
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
    ctx.update(observer=hierarchy_observer,select_mass_directions=select_mass_directions,hierarchical_kd_loss=hierarchical_kd_loss,group_log_prob=group_log_prob)
    result = old_background_mode(ctx, diagnostic)
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

    assert diagnostic.tensor_state_equal(before_hierarchy,hierarchy_observer.state_dict(),torch)
    atomic_json(output, {'hierarchy_observer_unchanged':True,'hierarchy_observer':hierarchy_observer.export(),'schema': 1, 'created_utc': now(), 'mode': 'old_background_kd_readiness', 'passed': True,
        'candidate': 'h2', 'stage': 2, 'iteration': 8000, 'final_checkpoint': True,
        'class_names': diagnostic.CLASSES, 'old_foreground_classes': 15, 'current_new_class_ids': list(range(16, 21)),
        'student_checkpoint': str(student_path), 'student_sha256': digest(student_path),
        'teacher_checkpoint': str(teacher_path), 'teacher_sha256': digest(teacher_path),
        'teacher_lineage': 'Actual study best optimizedKD stage1 final; no A/B teacher replay.',
        'complete_full1449_effect_gate': {'candidate_all_miou': endpoint['all_miou'],
            'optimized_KD_all_miou': reference['all_miou'], 'candidate_exceeds_optimized_KD': endpoint['all_miou'] > reference['all_miou'],
            'diagnostic_mIoU_not_used_as_final_endpoint': True},
        'observer_and_selector_checkpoint': str(student_path),
        'saved_observer_updates': int(observer.updates), 'saved_observer_seen_images': int(observer.seen_images),
        'saved_selector': exported, 'saved_online_old_background_relations': [
            {'source_class_id': i, 'source_class_name': diagnostic.CLASSES[i],
             'broad_row_total': float(observer.broad_counts[i].sum()),
             'broad_source_to_BG_count': float(observer.broad_counts[i, 0]),
             'broad_source_to_BG_fullrow_fraction': float(observer.broad_counts[i, 0]/observer.broad_counts[i].sum())
                 if float(observer.broad_counts[i].sum()) > 0 else None,
             'trusted_source_to_BG_count': float(observer.accepted_counts[i, 0]),
             'trusted_source_to_BG_image_exposures': float(observer.pair_image_observations[i, 0]),
             'broad_source_to_BG_image_exposures': float(observer.broad_pair_image_observations[i, 0])}
            for i in range(1, 16)],
        'saved_online_relation_usage': 'Original pseudo-PAR view retained for context; production directions use the separately saved loss-evidence-aligned observer. Rates rank directions only.',
        'source_sha256': study['source_sha256'], 'diagnostic_script_sha256': digest(Path(__file__)),
        'reused_weak_signal_protocol_sha256': PROTOCOL_SHA,
        'completion_analysis_sha256': digest(ROOT/'runs/hierarchy_kd_v2/analysis.json'),
        'state_checks': {'student_parameter_versions_unchanged': True, 'student_buffers_unchanged': True,
            'teacher_parameter_versions_unchanged': True, 'teacher_buffers_unchanged': True,
            'observer_unchanged': True, 'selector_unchanged': True,
            'student_frozen_no_gradients': True, 'teacher_frozen_no_gradients': True},
        'GT_role': 'Pixel GT partitions already fixed predictions and gates only. No GT source enters model, weak labels, acceptance, selection, weights or training. Current-new image tags follow the existing weak supervision protocol.',
        **result})
    print(json.dumps({'passed': True, 'mode': 'old_background_kd_readiness', 'output': str(output),
                      'coverage': result['readiness_coverage']}, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()

