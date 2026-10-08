"""Conditional held-out SEP audit; no optimizer, training or target changes.

Prepared for explicit invocation only. Fixed validation-list prefix and saved
checkpoint selectors define inference. Pixel GT is introduced after masks,
targets, evidence, logits and hypothetical factors have been constructed.
Shards emit additive raw statistics; this 128-image audit is not benchmark
accuracy and cannot establish that changing the loss would improve training.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
import random
import socket
import sys


ROOT = Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto')
EXPERIMENT = ROOT/'experiments/confusion_guided_v1'
RUN = ROOT/'runs/confusion_guided_v1/formal/a_sep'
SRC = EXPERIMENT/'a_sep/src'
BUCKET_EDGES = [0., .25, .5, .7, 1.]
CATEGORIES = ['all_old_eligible', 'correct_old_anchor', 'wrong_new_gt',
              'wrong_other_old_gt', 'wrong_background_gt', 'void_gt',
              'direct_new_target', 'old_eligible_with_new_competitor']
MEASURES = ['count', 'removed_pixel_equivalents', 'evidence_weight',
            'removed_evidence_weight', 'main_pair_push', 'removed_main_pair_push',
            'prototype_pair_push', 'removed_prototype_pair_push']


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for data in iter(lambda: handle.read(8*1024*1024), b''):
            h.update(data)
    return h.hexdigest()


def empty_statistics(classes):
    return {category: {measure: [[0.]*classes for _ in range(4)]
                       for measure in MEASURES} for category in CATEGORIES}


def sum_statistics(destination, source):
    for category in CATEGORIES:
        for measure in MEASURES:
            for bucket in range(4):
                row = destination[category][measure][bucket]
                for c, value in enumerate(source[category][measure][bucket]):
                    row[c] += value


def ratios_from_statistics(statistics):
    """Derived convenience fields; retain raw arrays for exact shard merging."""
    result = {}
    for category, measures in statistics.items():
        totals = {name: sum(sum(bucket) for bucket in array)
                  for name, array in measures.items()}
        fractions = {}
        for original, removed in [('count', 'removed_pixel_equivalents'),
                                  ('evidence_weight', 'removed_evidence_weight'),
                                  ('main_pair_push', 'removed_main_pair_push'),
                                  ('prototype_pair_push', 'removed_prototype_pair_push')]:
            fractions[original+'_fraction_removed'] = totals[removed]/totals[original] if totals[original] > 0 else None
        result[category] = {'totals': totals, 'hypothetical_removed_fractions': fractions}
    return result


def construct_sep_signals(student_logits, prototype_logits_scaled, teacher_logits,
                          refined_labels, valid_cam, boxes, targets, cfg, torch, F,
                          build_confusion_evidence):
    """No pixel-GT input exists in this function. Replicates actual SEP masks."""
    n, classes = student_logits.shape[:2]
    old_channels = teacher_logits.shape[1]
    size = student_logits.shape[-2:]
    if n != 1:
        raise ValueError('This fixed-image audit processes one image at a time')
    evidence = build_confusion_evidence(student_logits, refined_labels, valid_cam, boxes,
        high_threshold=cfg['high_thre'], low_threshold=cfg['low_thre'])
    anchors, weights, accepted = [F.interpolate(
        evidence[name][:, None].float(), size=size, mode='nearest')[:, 0]
        for name in ['anchors', 'weights', 'accepted']]
    anchors = anchors.long()
    accepted = accepted.bool()
    valid = F.interpolate(evidence['valid'][:, None].float(), size=size, mode='nearest')[:, 0].bool()
    anchor_safe = anchors.clamp(0, classes-1)
    partner = targets[anchor_safe]
    partner_safe = partner.clamp(0, classes-1)
    candidate = ((anchors > 0) & (anchors < classes) & accepted & (weights > 0)
                 & (partner > 0) & (partner < classes) & (partner != anchors) & valid)
    teacher = F.interpolate(teacher_logits.detach(), size=size, mode='bilinear', align_corners=False)
    teacher_prediction = teacher.argmax(1)
    old_anchor = anchors < old_channels
    teacher_veto = candidate & old_anchor & (teacher_prediction != anchors)
    eligible = candidate & ~teacher_veto
    old_eligible = eligible & old_anchor
    if prototype_logits_scaled.shape[-2:] != size:
        prototype_logits_scaled = F.interpolate(prototype_logits_scaled, size=size,
                                                 mode='bilinear', align_corners=False)
    main_margin = student_logits.gather(1, anchor_safe[:, None])[:, 0] - student_logits.gather(
        1, partner_safe[:, None])[:, 0]
    prototype_margin = prototype_logits_scaled.gather(1, anchor_safe[:, None])[:, 0] - prototype_logits_scaled.gather(
        1, partner_safe[:, None])[:, 0]
    cams_native = F.interpolate(valid_cam.detach(), size=size, mode='bilinear', align_corners=False).clamp(0, 1)
    max_new_cam = cams_native[:, old_channels-1:].amax(1)
    factor = (1-max_new_cam).square()
    # These are local derivative magnitudes w.r.t. competitor logits before
    # class normalization, external SEP coefficient, and the 0.5 branch mean.
    main_push = weights*torch.sigmoid(1-main_margin)
    prototype_push = weights*torch.sigmoid(1-prototype_margin)
    class_counts = torch.zeros(classes, dtype=torch.float64, device=anchors.device).scatter_add_(
        0, anchor_safe.flatten(), eligible.double().flatten())
    return {'anchors': anchors, 'partner': partner, 'candidate': candidate,
            'teacher_veto': teacher_veto, 'eligible': eligible, 'old_eligible': old_eligible,
            'weights': weights, 'max_new_cam': max_new_cam,
            'hypothetical_old_factor': factor, 'main_push': main_push,
            'prototype_push': prototype_push, 'class_eligible_counts_original': class_counts,
            'old_channels': old_channels, 'classes': classes, 'native_size': list(size)}


def score_after_forward(signals, pixel_gt, torch, F):
    """GT partitions already constructed masks; it never changes eligibility."""
    anchors, partner, old = signals['anchors'], signals['partner'], signals['old_eligible']
    classes, old_channels = signals['classes'], signals['old_channels']
    gt = torch.as_tensor(pixel_gt, device=anchors.device).float()[None, None]
    gt = F.interpolate(gt, size=(448, 448), mode='nearest')
    gt = F.interpolate(gt, size=signals['native_size'], mode='nearest')[:, 0].long()
    valid_gt = (gt >= 0) & (gt < classes)
    new_gt = valid_gt & (gt >= old_channels)
    other_old = (gt > 0) & (gt < old_channels) & (gt != anchors)
    category_masks = {
        'all_old_eligible': old,
        'correct_old_anchor': old & (gt == anchors),
        'wrong_new_gt': old & new_gt,
        'wrong_other_old_gt': old & other_old,
        'wrong_background_gt': old & (gt == 0),
        'void_gt': old & ~valid_gt,
        'direct_new_target': old & new_gt & (gt == partner),
        'old_eligible_with_new_competitor': old & (partner >= old_channels) & (partner < classes)}
    boundaries = torch.tensor(BUCKET_EDGES[1:-1], device=anchors.device)
    buckets = torch.bucketize(signals['max_new_cam'].contiguous(), boundaries, right=True)
    removed = 1-signals['hypothetical_old_factor']
    values = {'count': torch.ones_like(removed), 'removed_pixel_equivalents': removed,
        'evidence_weight': signals['weights'], 'removed_evidence_weight': signals['weights']*removed,
        'main_pair_push': signals['main_push'], 'removed_main_pair_push': signals['main_push']*removed,
        'prototype_pair_push': signals['prototype_push'], 'removed_prototype_pair_push': signals['prototype_push']*removed}
    result = empty_statistics(classes)
    for category, mask in category_masks.items():
        for bucket in range(4):
            selected = mask & (buckets == bucket)
            groups = anchors[selected]
            for name, value in values.items():
                amounts = torch.zeros(classes, dtype=torch.float64, device=anchors.device).scatter_add_(
                    0, groups.flatten(), value[selected].double().flatten())
                result[category][name][bucket] = amounts.cpu().tolist()
    partition = sum(int(category_masks[key].sum()) for key in
                    ['correct_old_anchor', 'wrong_new_gt', 'wrong_other_old_gt', 'wrong_background_gt', 'void_gt'])
    if partition != int(old.sum()):
        raise RuntimeError('GT diagnostic partition failed')
    metadata = {'candidate_pixels': int(signals['candidate'].sum()),
                'teacher_veto_old_pixels': int(signals['teacher_veto'].sum()),
                'eligible_old_pixels': int(old.sum()),
                'eligible_new_pixels': int((signals['eligible'] & ~old).sum()),
                'class_eligible_counts_original': signals['class_eligible_counts_original'].cpu().tolist(),
                'gt_partition_verified': True}
    return result, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rank', type=int, required=True)
    parser.add_argument('--shards', type=int, default=8)
    parser.add_argument('--checkpoint', choices=['2000', '8000'], nargs='+', default=['2000', '8000'])
    parser.add_argument('--checkpoint-path', default=None,
                        help='Optional exact path for one selected checkpoint label')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', type=int, default=None,
                        help='CUDA local ordinal; default rank, or0 when only one GPU is visible')
    args = parser.parse_args()
    if not 0 <= args.rank < args.shards or args.shards < 1:
        parser.error('Expected 0 <= rank < shards')
    if len(set(args.checkpoint)) != len(args.checkpoint):
        parser.error('Duplicate checkpoint labels are not allowed')
    if args.checkpoint_path and len(args.checkpoint) != 1:
        parser.error('An exact checkpoint path requires one checkpoint label')
    os.chdir(ROOT)
    sys.dont_write_bytecode = True
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    sys.path.insert(0, str(ROOT/'experiments/kd_pixel_v2'))
    from run_kd import environment
    os.environ.update(environment('8card'))
    sys.path.insert(0, str(SRC))
    import numpy as np
    import torch
    import torch.nn.functional as F
    import tasks
    from datasets import voc
    from model.model_seg_neg import network
    from model.PAR import PAR
    from model.online_directed_confusion import build_confusion_evidence
    from utils.camutils import multi_scale_cam2, cam_to_label, refine_cams_with_bkg_v2
    from utils.imutils import denormalize_img2
    from kd_runtime import safe_path, atomic_json
    output = safe_path(Path(args.output))
    if output.exists():
        raise RuntimeError(f'Refusing to replace an existing diagnostic: {output}')
    cfg = json.loads((RUN/'10-5/step2/config.json').read_text())
    if cfg['step'] != 2 or cfg['task'] != '10-5' or cfg['crop_size'] != 448:
        raise ValueError('This audit requires the stage2 VOC10-5 protocol at448')
    for key, name in [('model/model_seg_neg.py', 'network'),
                      ('model/online_directed_confusion.py', 'evidence'),
                      ('utils/camutils.py', 'CAM/PAR')]:
        expected = json.loads((RUN/'study.json').read_text())['source_sha256'][key]
        if digest(SRC/key) != expected:
            raise RuntimeError(f'Frozen {name} source changed')
    device_index = args.device if args.device is not None else (0 if torch.cuda.device_count() == 1 else args.rank)
    torch.cuda.set_device(device_index)
    device = torch.device('cuda', device_index)
    torch.set_num_threads(1)
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    teacher_path = safe_path(RUN/'10-5/step1/checkpoints/model_final.pth')
    if Path(cfg['prev_checkpoint']).resolve() != teacher_path.resolve():
        raise ValueError('Stage2 teacher must be armA stage1 final')
    def load_model(step, saved):
        per_task = tasks.get_per_task_classes('voc', cfg['task'], step)
        model = network(backbone=cfg['backbone'], num_classes=sum(per_task), classes_list=per_task,
                        pretrained=False, init_momentum=cfg['momentum'], aux_layer=cfg['aux_layer'])
        state = {key.removeprefix('module.'): value for key, value in saved['model_state'].items()}
        model.load_state_dict(state, strict=True)
        return model.to(device).eval().requires_grad_(False)
    teacher_saved = torch.load(teacher_path, map_location='cpu', weights_only=True, mmap=True)
    teacher = load_model(1, teacher_saved)
    del teacher_saved
    dataset = voc.VOC12SegDataset(root_dir=cfg['data_folder'], name_list_dir=cfg['list_folder'],
        split=cfg['val_set'], stage='val', aug=False, ignore_index=cfg['ignore_index'],
        num_classes=cfg['num_classes'], tasks=cfg['task'], step=2)
    dataset.label_dir = cfg['val_label_dir']
    if len(dataset) < 128:
        raise ValueError('Validation list has fewer than128images')
    fixed_indices = list(range(128))
    selected_indices = fixed_indices[args.rank::args.shards]
    expected_names = [str(dataset.name_list[i]) for i in fixed_indices]
    prefix_hash = hashlib.sha256('\n'.join(expected_names).encode()).hexdigest()
    par = PAR(num_iter=10, dilations=[1, 2, 4, 8, 12, 24]).to(device)
    records = []
    for label in args.checkpoint:
        path = safe_path(Path(args.checkpoint_path) if args.checkpoint_path else
                         RUN/f'10-5/step2/checkpoints/model_iter_{label}.pth')
        saved = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        selector = saved.get('pair_selector_state')
        if selector is None:
            raise ValueError('Checkpoint must contain its saved pair selector')
        metadata = selector.get('_extra_state', {})
        if metadata.get('stage') != 2 or metadata.get('classes') != 21:
            raise ValueError('Checkpoint selector stage/classes mismatch')
        if 'iteration' in saved and int(saved['iteration']) != int(label):
            raise ValueError('Checkpoint iteration does not match requested audit label')
        targets = selector['targets'].detach().to(device=device, dtype=torch.long)
        if targets.shape != (21,) or ((targets < -1) | (targets >= 21)).any():
            raise ValueError('Invalid saved selector targets')
        student = load_model(2, saved)
        selector_record = {'config': metadata, 'targets': targets.cpu().tolist(),
            'selected_rates_diagnostic_only': selector['selected_rates'].cpu().tolist(),
            'last_refresh_iteration': int(selector['last_refresh_iteration'])}
        observer_updates = int(saved['online_confusion_state']['updates']) if saved.get('online_confusion_state') else None
        del saved
        totals = empty_statistics(21)
        class_counts = [0.]*21
        per_image = []
        with torch.inference_mode():
            for index in selected_indices:
                name, image, pixel_gt, image_tags = dataset[index]
                inputs = F.interpolate(torch.as_tensor(image).float()[None].to(device),
                                       size=(448, 448), mode='bilinear', align_corners=False)
                tags = torch.as_tensor(image_tags)[None].to(device)
                boxes = torch.tensor([[0, 448, 0, 448]])
                old_cls, _, teacher_logits, _, _ = teacher(inputs, step0=True)
                # Only image-level tags of current classes are supplied as weak
                # supervision; previous-class image tags come from the teacher.
                weak_labels = torch.cat(((old_cls > 0).long(), tags[:, -5:]), 1)
                cams, _ = multi_scale_cam2(student, inputs, scales=cfg['cam_scales'])
                valid_cam, _ = cam_to_label(cams.detach(), cls_label=weak_labels, img_box=boxes,
                    ignore_mid=True, bkg_thre=cfg['bkg_thre'], high_thre=cfg['high_thre'],
                    low_thre=cfg['low_thre'], ignore_index=cfg['ignore_index'])
                refined = refine_cams_with_bkg_v2(par, denormalize_img2(inputs.clone()),
                    cams=valid_cam, cls_labels=weak_labels, high_thre=cfg['high_thre'],
                    low_thre=cfg['low_thre'], ignore_index=cfg['ignore_index'], img_box=boxes)
                _, _, main_logits, _, _, prototype_logits, _ = student(inputs, cam_grad=True)
                signals = construct_sep_signals(main_logits, prototype_logits/.1, teacher_logits,
                    refined, valid_cam, boxes, targets, cfg, torch, F, build_confusion_evidence)
                # Pixel GT enters only after every inference signal and gate.
                stats, image_meta = score_after_forward(signals, pixel_gt, torch, F)
                sum_statistics(totals, stats)
                for c, count in enumerate(image_meta['class_eligible_counts_original']):
                    class_counts[c] += count
                per_image.append({'index': index, 'image': str(name), **image_meta,
                                  'category_totals': ratios_from_statistics(stats)})
        records.append({'checkpoint_label': label, 'checkpoint': str(path),
            'checkpoint_sha256': digest(path), 'saved_observer_updates': observer_updates,
            'saved_selector': selector_record, 'images': [row['image'] for row in per_image],
            'indices': selected_indices, 'class_eligible_counts_original': class_counts,
            'raw_category_bucket_class_statistics': totals,
            'category_totals': ratios_from_statistics(totals), 'per_image': per_image})
        del student
        torch.cuda.empty_cache()
    report = {'schema': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'host': socket.gethostname(), 'rank': args.rank, 'shards': args.shards,
        'arm': 'a_sep', 'stage': 2, 'teacher': str(teacher_path), 'teacher_sha256': digest(teacher_path),
        'sampling': 'Same fixed first128validation-list images at both checkpoints; noaugmentation, resize448; rank-strided subset.',
        'validation_prefix128_sha256': prefix_hash, 'validation_prefix128_names': expected_names,
        'selected_indices': selected_indices, 'results': records,
        'cam_buckets': {'edges': BUCKET_EDGES, 'intervals': ['[0,.25)', '[.25,.5)', '[.5,.7)', '[.7,1]']},
        'scope': 'Conditional held-out mechanism audit; nooptimizer, no backward, no training or selector update. GT only partitions fixed masks afterforward.',
        'hypothesis': 'For old-anchor SEP only, multiply numerator by (1-max_new_CAM)^2; all original eligible classcounts and denominators remain unchanged. New-anchor SEP is unchanged.',
        'push_definition': 'evidence_weight*sigmoid(1-(anchor_logit-competitor_logit)); derivative magnitude w.r.t. competitor output before sqrtclass normalization, 0.5 branch averaging and0.1*ramp coefficient.',
        'normalization': 'Original eligible pixel counts retain both old and new anchors and teacher veto policy. Raw bucket/class numerators allow additive shard merging; no hypothetical confidence is substituted into denominators.',
        'merge_contract': 'Require matching teacherSHA, checkpointSHA, prefix128SHA, selector, labels andshards; each rank once; indices/names must partition the128prefix exactly. Addraw stats/classcounts then recompute ratios; never average shard ratios.',
        'limitations': ['128 fixed-order images may be class-imbalanced; not full-validation accuracy.',
            'Native28 nearest-GT diagnostic and evalmode differ from original-GT-space benchmark and augmented training batches.',
            'Saved selector differs across checkpoints; changes in diagnostics need not come from weights alone.',
            'Local logit push is not parameter-gradient norm or full training-gradient conflict.',
            'Potential suppression selectivity is observational; it does not establish benefit from changing or retraining the loss.',
            'Void-GT pixels remain visible as a separate category and never alter constructed eligibility.',
            'Selected confusion rates are diagnostic only and never used as precise learning weights.']}
    atomic_json(output, report)
    print(json.dumps({'output': str(output), 'rank': args.rank,
                      'images_per_checkpoint': len(selected_indices), 'checkpoint_labels': args.checkpoint}))


if __name__ == '__main__':
    main()
