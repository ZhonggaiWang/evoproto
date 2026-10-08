"""Read-only endpoint diagnostics for the one formal prototype-SEP candidate.

Deploy explicitly and run with --mode gradient or --mode gt. Default is the
complete stage2 endpoint; --step1 --iteration4000 audits a published midpoint.
No optimizer, observer/selector update, or training occurs.
The gradient mode never loads pixel annotations. GT only scores fixed evidence
in the GT mode; it does not enter model inference or prototype separation.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import random
import sys


ROOT = Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto')
SRC = ROOT/'experiments/prototype_sep_v1/a_geometry/src'
RUN = ROOT/'runs/prototype_sep_v1/formal/a_geometry'
CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
           'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
PAIR_CATEGORIES = ['gt_source', 'gt_competitor', 'gt_other_foreground',
                   'gt_background', 'gt_void']


def stratified_indices(names, label_list, new_ids, count=32, seed=0):
    """Use image multilabels only: six distinct slots/new class plus two controls.

    Multilabel overlap is retained and reported rather than treated as mutually
    exclusive classes. The fixed list order and local RNG define the sample.
    """
    if count != 32 or len(names) < count:
        raise ValueError('Expected32 images from a list with at least32 entries')
    pools = {class_id: [] for class_id in new_ids}
    controls, tags = [], []
    for index, name in enumerate(names):
        values = list(label_list[str(name)])
        if len(values) != 20 or any(not math.isfinite(float(v)) for v in values):
            raise ValueError('Need finite20-entry foreground image multilabels')
        current = [class_id for class_id in new_ids if values[class_id-1] > 0]
        tags.append(current)
        if not current:
            controls.append(index)
        for class_id in current:
            pools[class_id].append(index)
    rng = random.Random(seed)
    for pool in [*pools.values(), controls]:
        rng.shuffle(pool)
    indices, assigned, chosen = [], [], set()
    # Round robin avoids placing each class in a separate eight-image group.
    for slot in range(6):
        for class_id in new_ids:
            available = next((i for i in pools[class_id] if i not in chosen), None)
            if available is None:
                raise ValueError(f'Insufficient distinct samples for new class{class_id}')
            indices.append(available)
            assigned.append(class_id)
            chosen.add(available)
    selected_controls = [i for i in controls if i not in chosen][:2]
    for index in selected_controls:
        indices.append(index)
        assigned.append(None)
        chosen.add(index)
    fallback = [i for i in range(len(names)) if i not in chosen]
    rng.shuffle(fallback)
    for index in fallback[:count-len(indices)]:
        indices.append(index)
        assigned.append(None)
    if len(indices) != count or len(set(indices)) != count:
        raise RuntimeError('Sampling must yield exactly32 distinct indices')
    return indices, {
        'method': 'Six assigned slots per new foreground class, round robin; two no-new controls if available, otherwise unselected-list fallback.',
        'seed': seed, 'assigned_strata': assigned,
        'new_image_labels_by_sample': [tags[i] for i in indices],
        'selected_no_new_controls': selected_controls,
        'available_image_counts': {str(c): len(pools[c]) for c in new_ids},
        'sample_image_counts': {str(c): sum(c in tags[i] for i in indices) for c in new_ids},
        'caveat': 'Deliberately covers new classes; not an estimate under the original training image distribution. A multilabel image contributes to every present class.'}


def tensor_state_equal(left, right, torch):
    if left.keys() != right.keys():
        return False
    return all(torch.equal(left[k], right[k].detach().cpu()) if isinstance(left[k], torch.Tensor)
               else left[k] == right[k] for k in left)


def freeze_state(module, torch):
    return {k: v.detach().cpu().clone() if isinstance(v, torch.Tensor) else v
            for k, v in module.state_dict().items()}


def summarize(values):
    values = [float(v) for v in values if v is not None]
    if not values:
        return {'count': 0, 'minimum': None, 'median': None, 'mean': None, 'maximum': None}
    values.sort()
    middle = len(values)//2
    median = values[middle] if len(values)%2 else (values[middle-1]+values[middle])/2
    return {'count': len(values), 'minimum': values[0], 'median': median,
            'mean': sum(values)/len(values), 'maximum': values[-1]}


def vector_relationship(left, right, torch):
    left_norm, right_norm = float(left.norm()), float(right.norm())
    dot = float((left*right).sum())
    return {'left_norm': left_norm, 'right_norm': right_norm, 'dot': dot,
            'cosine': dot/(left_norm*right_norm) if left_norm > 0 and right_norm > 0 else None,
            'left_to_right_norm_ratio': left_norm/right_norm if right_norm > 0 else None,
            'semantic_dot_with_two_component_sum': right_norm**2+dot,
            'opposing_semantic_direction_fraction': -dot/right_norm**2 if right_norm > 0 and dot < 0 else 0.,
            'two_component_sum_reverses_semantic_direction': right_norm > 0 and right_norm**2+dot < 0}


def weak_signals(ctx, inputs, image_tags, boxes):
    """This function has no pixel-GT argument and updates no saved state."""
    torch, F, cfg = ctx['torch'], ctx['F'], ctx['cfg']
    with torch.no_grad():
        old_cls, _, teacher_logits, _, _ = ctx['teacher'](inputs, step0=True)
        new_tags = image_tags[:, ctx['old_classes']:ctx['classes']-1]
        weak_labels = torch.cat(((old_cls > 0).long(), new_tags), dim=1)
        cams, _ = ctx['multi_scale_cam2'](ctx['student'], inputs, scales=cfg['cam_scales'])
        valid_cams, _ = ctx['cam_to_label'](cams.detach(), cls_label=weak_labels,
            img_box=boxes, ignore_mid=True, bkg_thre=cfg['bkg_thre'],
            high_thre=cfg['high_thre'], low_thre=cfg['low_thre'],
            ignore_index=cfg['ignore_index'])
        refined = ctx['refine_cams_with_bkg_v2'](ctx['par'], ctx['denormalize_img2'](inputs.clone()),
            cams=valid_cams, cls_labels=weak_labels, high_thre=cfg['high_thre'],
            low_thre=cfg['low_thre'], ignore_index=cfg['ignore_index'], img_box=boxes)
        old_labels = F.interpolate(teacher_logits, size=inputs.shape[-2:], mode='bilinear',
                                  align_corners=False).argmax(1)
        mixed = ctx['get_mixed_label'](refined, old_labels, ctx['classes']-1, len(ctx['new_ids']))
    return teacher_logits, valid_cams, refined, mixed


def gradient_mode(ctx):
    torch, F, cfg, device = ctx['torch'], ctx['F'], ctx['cfg'], ctx['device']
    new_ids, new_start = ctx['new_ids'], ctx['old_classes']+1
    ds = ctx['voc'].VOC12ClsDataset(root_dir=cfg['data_folder'], name_list_dir=cfg['list_folder'],
        split='train', stage='train', tasks='10-5', step=ctx['step'], aug=True,
        rescale_range=cfg['scales'], crop_size=448, img_fliplr=True,
        ignore_index=cfg['ignore_index'], num_classes=ctx['classes'])
    indices, sampling = stratified_indices(ds.name_list, ds.label_list, new_ids)
    sampling['train_list_path'] = str(ds.name_list_dir)
    sampling['train_list_sha256'] = ctx['digest'](Path(ds.name_list_dir))
    sampling['image_multilabel_sha256'] = ctx['digest'](Path(cfg['list_folder'])/'cls_labels_onehot.npy')
    sampling['augmentation_seed_rule'] = '100000+dataset_index, reset Python/NumPy/Torch before each image'
    params = {name: p for name, p in ctx['student'].named_parameters() if p.requires_grad}
    new_key = f"decoder.class_prototypes.{ctx['step']}.prototype"
    if new_key not in params or tuple(params[new_key].shape[:1]) != (5,):
        raise ValueError('Expected the actual five-row new prototype parameter group')
    records, new_gradients = [], []
    for index in indices:
        seed = 100000+index
        random.seed(seed)
        ctx['np'].random.seed(seed)
        torch.manual_seed(seed)
        name, image, image_tags, box, _ = ds[index]
        inputs = image[None].to(device)
        tags = torch.as_tensor(image_tags, device=device)[None]
        boxes = torch.as_tensor(box).reshape(1, 4)
        teacher_logits, valid_cams, refined, mixed = weak_signals(ctx, inputs, tags, boxes)
        _, segs, _, _, proto_logits, prototypes = ctx['student'](inputs, crops=[])
        sep, sep_stats = ctx['sep_loss'](prototypes, ctx['targets'], ctx['old_classes'], margin=cfg['proto_margin'])
        kd, kd_stats = ctx['pixel_kd_loss'](segs, teacher_logits, refined, valid_cams,
                                          boxes, cfg['kd_temperature'])
        proto_seg = ctx['get_seg_loss'](F.interpolate(proto_logits/.1, size=mixed.shape[-2:],
            mode='bilinear', align_corners=False), mixed.long(),
            torch.ones(ctx['classes'], device=device), ignore_index=cfg['ignore_index'])
        losses = {'sep': .1*ctx['ramp']*sep, 'kd': .1*kd, 'proto_seg': .1*proto_seg}
        gradients = {}
        for component, loss in losses.items():
            raw = torch.autograd.grad(loss, tuple(params.values()), retain_graph=True, allow_unused=True)
            gradients[component] = {name: g for name, g in zip(params, raw) if g is not None}
            if any(not torch.isfinite(g).all() for g in gradients[component].values()):
                raise FloatingPointError('Nonfinite component parameter gradient')
        bad_sep = [k for k, g in gradients['sep'].items() if k != new_key and g.count_nonzero() > 0]
        bad_kd = [k for k, g in gradients['kd'].items()
                  if k.startswith('decoder.class_prototypes.') and g.count_nonzero() > 0]
        if bad_sep or bad_kd:
            raise RuntimeError(f'Unexpected gradient paths SEP={bad_sep}, KD={bad_kd}')
        current_new = {k: gradients[k].get(new_key, torch.zeros_like(params[new_key])).detach().cpu().clone()
                       for k in losses}
        relationship = vector_relationship(current_new['sep'], current_new['proto_seg'], torch)
        per_class = {str(c): vector_relationship(current_new['sep'][c-new_start], current_new['proto_seg'][c-new_start], torch)
                     for c in new_ids}
        new_gradients.append(current_new)
        nonzero_names = {k: sorted(n for n, g in values.items() if g.count_nonzero() > 0)
                         for k, values in gradients.items()}
        shared = sorted(set(nonzero_names['sep']) & set(nonzero_names['kd']))
        if shared:
            raise RuntimeError(f'SEP and KD unexpectedly share direct parameter support: {shared}')
        records.append({'index': index, 'image': str(name), 'augmentation_seed': seed,
            'image_new_labels': [c for c in new_ids if float(tags[0, c-1]) > 0],
            'mixed_valid_pixels': int((mixed != cfg['ignore_index']).sum()),
            'mixed_pseudo_new_pixels': {str(c): int((mixed == c).sum()) for c in new_ids},
            'weighted_component_loss': {k: float(v.detach()) for k, v in losses.items()},
            'sep': sep_stats, 'kd': kd_stats,
            'nonzero_parameter_gradient_norms': {k: {n: float(g.norm()) for n, g in values.items()
                                                    if g.count_nonzero() > 0}
                                                 for k, values in gradients.items()},
            'new_prototype_sep_vs_proto_seg': relationship,
            'new_prototype_per_class': per_class,
            'sep_kd_shared_nonzero_parameters': shared,
            'gradient_checks': {'sep_background_old_encoder_main_head_zero': True,
                                'kd_all_prototype_parameters_zero': True}})
        print(json.dumps({'mode': 'gradient', 'images_completed': len(records), 'image': str(name)}), flush=True)
        del gradients, losses, raw, segs, proto_logits, prototypes, inputs
    group_records = []
    for start in range(0, len(records), 8):
        subset = new_gradients[start:start+8]
        valid_counts = [record['mixed_valid_pixels'] for record in records[start:start+8]]
        total_valid = sum(valid_counts)
        pooled_semantic = sum((row['proto_seg']*count for row, count in zip(subset, valid_counts)),
                              torch.zeros_like(subset[0]['proto_seg']))/max(total_valid, 1)
        averaged = {'sep': torch.stack([row['sep'] for row in subset]).mean(0),
                    'proto_seg': pooled_semantic}
        group_records.append({'indices': indices[start:start+8],
            'mixed_valid_pixels_by_image': valid_counts, 'pooled_mixed_valid_pixels': total_valid,
            'sep_vs_proto_seg': vector_relationship(averaged['sep'], averaged['proto_seg'], torch),
            'per_new_class': {str(c): vector_relationship(averaged['sep'][c-new_start], averaged['proto_seg'][c-new_start], torch)
                              for c in new_ids},
            'aggregation': 'Prototype segmentation: sum(valid_pixels_i*single_image_gradient_i)/sum(valid_pixels_i), matching global valid-pixel normalization conditional on these fixed masks/forwards; all-zero support gives zero. SEP: mean of identical replica gradients, never summed eight times.'})
    return {'sampling': sampling, 'sample_indices': indices, 'records': records,
        'selected_pair_endpoint_geometry': records[0]['sep'],
        'summary': {'sep_nonzero_images': sum(bool(r['nonzero_parameter_gradient_norms']['sep']) for r in records),
            'kd_nonzero_images': sum(bool(r['nonzero_parameter_gradient_norms']['kd']) for r in records),
            'semantic_nonzero_images': sum(r['new_prototype_sep_vs_proto_seg']['right_norm'] > 0 for r in records),
            'negative_cosine_images': sum(r['new_prototype_sep_vs_proto_seg']['cosine'] is not None and
                                           r['new_prototype_sep_vs_proto_seg']['cosine'] < 0 for r in records),
            'new_prototype_cosine': summarize(r['new_prototype_sep_vs_proto_seg']['cosine'] for r in records),
            'sep_to_semantic_norm_ratio': summarize(r['new_prototype_sep_vs_proto_seg']['left_to_right_norm_ratio'] for r in records),
            'per_new_class': {str(c): {'cosine': summarize(r['new_prototype_per_class'][str(c)]['cosine'] for r in records),
                                      'sep_to_semantic_norm_ratio': summarize(r['new_prototype_per_class'][str(c)]['left_to_right_norm_ratio'] for r in records)}
                             for c in new_ids}},
        'eight_image_semantic_gradient_means': group_records,
        'eight_image_semantic_gradient_definition': 'Pooled-valid-pixel conditional parameter gradients, not unweighted image means; legacy field name retained for readability.',
        'scope': '32 fixed stratified actual augmented weakly labeled training images. Three weighted component gradients in eval mode, only trainable parameters; no optimizer, backward accumulation, or pixel-GT access.',
        'limitations': [
            'This is a component diagnostic, not the full training objective or an optimizer update. Main segmentation, image classification and affinity losses are not differentiated here.',
            'Single-image KD uses local class-count normalization. Actual8x1 DDP KD pools class counts across eight ranks; these individual KD magnitudes do not reproduce its global8 update.',
            'Eight-image pooled-valid-pixel semantic gradients and replica-mean SEP gradients match their respective normalizations conditional on these eval-mode forwards and fixed masks; they are not actual sampled training batches or a complete DDP gradient probe.',
            'The selected prototype SEP depends only on saved parameters and targets, so its gradient repeats across images.32 repetitions are not32 independent observations of geometric efficacy.',
            'Eval mode, fixed augmentations, deliberate new-class coverage and the endpoint differ from train mode and the full trajectory.',
            'No direct SEP/KD parameter overlap does not exclude later interaction through changed prototypes, semantic feature learning and future teachers.',
            'Gradient cosine describes a current Euclidean component direction; it does not establish AdamW update conflict or final IoU causality.',
            'The same development validation informed previous candidate changes; this diagnostic is not an independent efficacy test.']}


def blank_pair_statistics():
    return {scope: {'pixels': 0, 'evidence_weight': 0.,
                    'gt_counts': {c: 0 for c in PAIR_CATEGORIES},
                    'gt_evidence_weight': {c: 0. for c in PAIR_CATEGORIES}}
            for scope in ['anchor_row', 'observed_direction']}


def derive_pair_statistics(raw):
    out = {}
    for scope, values in raw.items():
        out[scope] = {**values,
            'gt_source_fraction_all_pixels': values['gt_counts']['gt_source']/values['pixels'] if values['pixels'] else None,
            'gt_competitor_fraction_all_pixels': values['gt_counts']['gt_competitor']/values['pixels'] if values['pixels'] else None,
            'weighted_gt_source_fraction': values['gt_evidence_weight']['gt_source']/values['evidence_weight'] if values['evidence_weight'] else None,
            'weighted_gt_competitor_fraction': values['gt_evidence_weight']['gt_competitor']/values['evidence_weight'] if values['evidence_weight'] else None}
    return out


def gt_mode(ctx):
    torch, F, cfg, device = ctx['torch'], ctx['F'], ctx['cfg'], ctx['device']
    classes, old_classes = ctx['classes'], ctx['old_classes']
    ds = ctx['voc'].VOC12SegDataset(root_dir=cfg['data_folder'], name_list_dir=cfg['list_folder'],
        split=cfg['val_set'], stage='val', aug=False, ignore_index=cfg['ignore_index'],
        num_classes=classes, tasks='10-5', step=ctx['step'])
    ds.label_dir = cfg['val_label_dir']
    if len(ds) < 128:
        raise ValueError('Validation list requires at least128 images')
    indices = list(range(128))
    names = [str(ds.name_list[i]) for i in indices]
    selected = [(i, int(j)) for i, j in enumerate(ctx['targets'].cpu().tolist())
                if i > 0 and j > 0 and i != j and (i > old_classes or j > old_classes)]
    raw_pairs = {view: {f'{i}->{j}': blank_pair_statistics() for i, j in selected}
                 for view in ['broad', 'trusted']}
    matrices = {view: torch.zeros(classes, 22, dtype=torch.int64, device=device)
                for view in ['broad', 'trusted']}
    weighted_matrices = {view: torch.zeros(classes, 22, dtype=torch.float64, device=device)
                        for view in ['broad', 'trusted']}
    prediction_gt = torch.zeros(21, classes, dtype=torch.int64, device=device)
    records, geometry = [], None
    with torch.inference_mode():
        for index in indices:
            name, image, pixel_gt, image_tags = ds[index]
            inputs = F.interpolate(torch.as_tensor(image, device=device).float()[None],
                                   size=(448, 448), mode='bilinear', align_corners=False)
            tags = torch.as_tensor(image_tags, device=device)[None]
            boxes = torch.tensor([[0, 448, 0, 448]])
            _, valid_cams, refined, _ = weak_signals(ctx, inputs, tags, boxes)
            _, logits, _, _, _, prototypes = ctx['student'](inputs, crops=[])
            evidence = ctx['build_confusion_evidence'](logits, refined, valid_cams, boxes,
                high_threshold=cfg['high_thre'], low_threshold=cfg['low_thre'])
            if geometry is None:
                _, geometry = ctx['sep_loss'](prototypes, ctx['targets'], old_classes, margin=cfg['proto_margin'])
            # Pixel GT is used only now, after all predictions/evidence were fixed.
            gt = F.interpolate(torch.as_tensor(pixel_gt, device=device).float()[None, None],
                               size=evidence['anchors'].shape[-2:], mode='nearest')[:, 0].long()
            valid_gt = (gt >= 0) & (gt < 21)
            gt_column = torch.where(valid_gt, gt, torch.full_like(gt, 21))
            anchors, predictions = evidence['anchors'], evidence['predictions']
            masks = {'broad': evidence['broad'], 'trusted': evidence['accepted']}
            image_record = {'index': index, 'image': str(name), 'views': {}}
            valid_prediction = evidence['valid'] & valid_gt
            prediction_gt += torch.bincount((gt[valid_prediction]*classes+predictions[valid_prediction]).flatten(),
                                             minlength=21*classes).reshape(21, classes)
            for view, mask in masks.items():
                weights = evidence['weights'] if view == 'trusted' else torch.ones_like(evidence['weights'])
                ids = (anchors[mask]*22+gt_column[mask]).flatten()
                matrices[view] += torch.bincount(ids, minlength=classes*22).reshape(classes, 22)
                weighted_matrices[view] += torch.bincount(ids, weights=weights[mask].double().flatten(),
                                                         minlength=classes*22).reshape(classes, 22)
                correct = mask & valid_gt & (anchors == gt)
                image_record['views'][view] = {'pixels': int(mask.sum()), 'gt_anchor_correct': int(correct.sum()),
                    'gt_void': int((mask & ~valid_gt).sum()), 'selected_directions': {}}
                for source, competitor in selected:
                    row = mask & (anchors == source)
                    direction = row & (predictions == competitor)
                    categories = {'gt_source': gt == source, 'gt_competitor': gt == competitor,
                        'gt_other_foreground': valid_gt & (gt > 0) & (gt != source) & (gt != competitor),
                        'gt_background': gt == 0, 'gt_void': ~valid_gt}
                    key = f'{source}->{competitor}'
                    per_image = blank_pair_statistics()
                    for scope, selected_mask in [('anchor_row', row), ('observed_direction', direction)]:
                        value = per_image[scope]
                        value['pixels'] = int(selected_mask.sum())
                        value['evidence_weight'] = float(weights[selected_mask].double().sum())
                        for category, category_mask in categories.items():
                            partition = selected_mask & category_mask
                            value['gt_counts'][category] = int(partition.sum())
                            value['gt_evidence_weight'][category] = float(weights[partition].double().sum())
                        if sum(value['gt_counts'].values()) != value['pixels']:
                            raise RuntimeError('GT categories do not partition the fixed selected direction')
                        total = raw_pairs[view][key][scope]
                        total['pixels'] += value['pixels']
                        total['evidence_weight'] += value['evidence_weight']
                        for category in PAIR_CATEGORIES:
                            total['gt_counts'][category] += value['gt_counts'][category]
                            total['gt_evidence_weight'][category] += value['gt_evidence_weight'][category]
                    image_record['views'][view]['selected_directions'][key] = derive_pair_statistics(per_image)
            records.append(image_record)
            if (index+1)%8 == 0:
                print(json.dumps({'mode': 'gt', 'images_completed': index+1}), flush=True)
    return {'sample_indices': indices, 'sample_names': names,
        'validation_prefix128_sha256': hashlib.sha256('\n'.join(names).encode()).hexdigest(),
        'sampling': 'Fixed first128 images of this stage validation list; stage2 reproduces the previous anchor-audit prefix. No augmentation, input resize448.',
        'selected_pair_endpoint_geometry': geometry,
        'anchor_row_class_ids': list(range(classes)),
        'anchor_gt_column_ids': list(range(21))+['void'],
        'anchor_vs_gt_counts': {k: v.cpu().tolist() for k, v in matrices.items()},
        'anchor_vs_gt_evidence_weight': {k: v.cpu().tolist() for k, v in weighted_matrices.items()},
        'selected_directions': {view: {key: derive_pair_statistics(value) for key, value in pairs.items()}
                                for view, pairs in raw_pairs.items()},
        'prediction_vs_gt_counts': prediction_gt.cpu().tolist(),
        'prediction_matrix_orientation': 'row=GT, column=main segmentation prediction, fixed resize448 diagnostic only',
        'records': records,
        'scope': 'Fixed128-image development-validation anchor audit. GT partitions saved-selector directions and fixed broad/trusted evidence; no observer/selector update or backward.',
        'definitions': {'anchor_row': 'All evidence pixels whose pre-teacher-mixing PAR anchor equals the selected source.',
            'observed_direction': 'Subset of that row whose current full main-head argmax equals the selected competitor.',
            'trusted': 'Matching strongest CAM>=high threshold; positive CAM-strength times top-two-gap evidence weight.',
            'broad': 'All valid pre-mixing PAR class anchors; count weights1.',
            'gt_source': 'Anchor label agrees with GT.', 'gt_competitor': 'GT is the selected competing class, so the source anchor is wrong.',
            'weight_fractions': 'Uncalibrated CAM evidence-weight fractions; broad weights are ordinary pixel counts.'},
        'limitations': ['This method separates global prototype vectors; none of these pixels receives a direct SEP logit push. These are anchor reliability/selection diagnostics, not local SEP-force attribution.',
            'Trusted support reduces some noisy anchors but is selection-biased and does not certify the correctness of every supported relation.',
            'Endpoint selection was learned from augmented training exposures; validation evidence need not reproduce its broad EMA ranking or exposure counts.',
            'The128-image prefix is neither class-balanced nor full benchmark evaluation; resize448 GT alignment differs from original-image evaluation.',
            'These same development-validation images previously helped select a candidate correction. This is not a new independent test or a significance estimate.',
            'Geometry loss is active whenever a selected cosine exceeds the margin, even if its classes are absent from an individual image. Anchor pixel coverage is not the loss denominator or exact prototype-learning strength.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['gradient', 'gt'], default='gradient')
    parser.add_argument('--step', type=int, choices=[1, 2], default=2)
    parser.add_argument('--step1', dest='step', action='store_const', const=1)
    parser.add_argument('--iteration', type=int, default=None,
                        help='Published same-stage checkpoint iteration; omit for model_final at8000')
    parser.add_argument('--device', type=int, default=0, help='Local CUDA ordinal within externally supplied CUDA_VISIBLE_DEVICES')
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT/'experiments/kd_pixel_v2'))
    from run_kd import environment
    os.environ.update(environment('8card'))
    sys.path.insert(0, str(SRC))
    from kd_runtime import safe_path, atomic_json, digest, now
    is_final = args.iteration is None
    iteration = 8000 if is_final else args.iteration
    if not 1 <= iteration <= 8000:
        parser.error('Iteration must lie in1..8000')
    stage = RUN/f'10-5/step{args.step}'
    label = 'endpoint' if is_final and args.step == 2 else f'stage{args.step}_iter{iteration}'
    output = safe_path(ROOT/f'runs/prototype_sep_v1/{label}_{args.mode}_diagnostic.json')
    if output.exists():
        raise RuntimeError(f'Refusing to replace an existing diagnostic: {output}')
    study = json.loads((RUN/'study.json').read_text())
    if study.get('smoke') is not False or study.get('arm') != 'a_geometry':
        raise ValueError('Require the actual formal prototype geometry candidate')
    if is_final and args.step == 2:
        if json.loads((RUN/'status.json').read_text()).get('status') != 'complete':
            raise RuntimeError('Formal training/evaluation must be complete before endpoint diagnostics')
        if json.loads((RUN/'evaluation_workers_complete.json').read_text()).get('returncodes') != [0, 0, 0, 0]:
            raise RuntimeError('All four full-evaluation workers must have succeeded')
    for relative, expected in study['source_sha256'].items():
        if digest(SRC/relative) != expected:
            raise RuntimeError(f'Frozen candidate source changed: {relative}')
    cfg = json.loads((stage/'config.json').read_text())
    if cfg['step'] != args.step or cfg['task'] != '10-5' or cfg['crop_size'] != 448 or cfg['max_iters'] != 8000:
        raise ValueError('Require actual formal VOC10-5 protocol at448')
    for key in ['w_geometry_sep', 'w_pixel_kd', 'w_proto_seg']:
        if float(cfg[key]) != .1:
            raise ValueError(f'Expected actual component coefficient0.1: {key}')
    if float(cfg['proto_margin']) != 0.:
        raise ValueError('This endpoint protocol requires the formal margin0 candidate')
    student_path = safe_path(stage/'checkpoints'/('model_final.pth' if is_final else f'model_iter_{iteration}.pth'))
    teacher_path = safe_path(Path(cfg['prev_checkpoint']))
    if args.step == 1:
        if teacher_path.resolve() != Path(study['step0']).resolve() or digest(teacher_path) != study['step0_sha256']:
            raise ValueError('Stage1 must use the frozen shared step0 teacher')
    else:
        expected_teacher = RUN/'10-5/step1/checkpoints/model_final.pth'
        if teacher_path.resolve() != expected_teacher.resolve():
            raise ValueError('Stage2 must use this candidate own stage1-final teacher')
        receipt = json.loads((RUN/'10-5/step1/training_complete.json').read_text())
        if receipt.get('returncode') != 0 or receipt['checkpoint_sha256'] != digest(teacher_path):
            raise RuntimeError('Incomplete/changed own stage1-final teacher')
    if is_final:
        receipt = json.loads((stage/'training_complete.json').read_text())
        if receipt.get('returncode') != 0 or receipt['checkpoint_sha256'] != digest(student_path):
            raise RuntimeError('Incomplete/changed final student checkpoint')
    else:
        receipt = json.loads((RUN/f'eval_queue/step{args.step}_iter{iteration}.json').read_text())
        if receipt['checkpoint_sha256'] != digest(student_path):
            raise RuntimeError('Checkpoint differs from its completed evaluation publication')
    import numpy as np
    import torch
    import torch.nn.functional as F
    import tasks
    from datasets import voc
    from model.model_seg_neg import network
    from model.pixel_kd import pixel_kd_loss
    from model.confusion_prototype_sep import confusion_prototype_sep_loss
    from model.online_directed_confusion import OnlineDirectedConfusion, build_confusion_evidence
    from model.geometry_pair_selector import GeometryPairSelector
    from model.losses import get_seg_loss
    from model.PAR import PAR
    from utils.camutils import multi_scale_cam2, cam_to_label, refine_cams_with_bkg_v2, get_mixed_label
    from utils.imutils import denormalize_img2
    if not torch.cuda.is_available() or not 0 <= args.device < torch.cuda.device_count():
        raise ValueError('Expected an available, explicitly selected CUDA device')
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
        classes = tasks.get_per_task_classes('voc', '10-5', step)
        model = network(backbone=cfg['backbone'], num_classes=sum(classes), classes_list=classes,
            pretrained=False, init_momentum=cfg['momentum'], aux_layer=cfg['aux_layer'])
        model.load_state_dict({k.removeprefix('module.'): v for k, v in saved['model_state'].items()}, strict=True)
        return model.to(device).eval()
    saved = torch.load(student_path, map_location='cpu', weights_only=True, mmap=True)
    if int(saved['iteration']) != iteration:
        raise ValueError('Student checkpoint must match the requested iteration')
    task_classes = tasks.get_per_task_classes('voc', '10-5', args.step)
    classes = sum(task_classes)
    old_classes = sum(tasks.get_per_task_classes('voc', '10-5', args.step-1))-1
    new_ids = list(range(old_classes+1, classes))
    if len(new_ids) != 5:
        raise ValueError('This stratified diagnostic expects five new classes')
    student = load_model(args.step, saved)
    observer = OnlineDirectedConfusion(classes, momentum=cfg['confusion_momentum'],
        high_threshold=cfg['high_thre'], low_threshold=cfg['low_thre'], stage=args.step).to(device)
    observer.load_state_dict(saved['online_confusion_state'], strict=True)
    selector = GeometryPairSelector(classes, old_classes=old_classes, stage=args.step,
        refresh_interval=cfg['pair_refresh_interval'], min_row_images=cfg['pair_min_row_images'],
        min_pair_images=cfg['pair_min_pair_images'], min_rate=cfg['pair_min_rate'],
        min_updates=cfg['pair_min_updates'], ramp_updates=cfg['pair_ramp_updates'],
        max_stale_updates=cfg['pair_max_stale_updates']).to(device)
    selector.load_state_dict(saved['geometry_selector_state'], strict=True)
    del saved
    targets = selector.targets.detach().clone()
    ramp = selector.ramp(observer)
    if targets.shape != (classes,) or ((targets < -1) | (targets >= classes)).any():
        raise ValueError('Invalid exact saved endpoint selector targets')
    saved_teacher = torch.load(teacher_path, map_location='cpu', weights_only=True, mmap=True)
    teacher = load_model(args.step-1, saved_teacher).requires_grad_(False)
    del saved_teacher
    if args.mode == 'gt':
        student.requires_grad_(False)
    state_versions = {name: p._version for name, p in student.named_parameters()}
    buffers = {name: value.detach().cpu().clone() for name, value in student.named_buffers()}
    observer_before, selector_before = freeze_state(observer, torch), freeze_state(selector, torch)
    ctx = dict(torch=torch, F=F, np=np, cfg=cfg, device=device, student=student, teacher=teacher,
        step=args.step, classes=classes, old_classes=old_classes, new_ids=new_ids,
        voc=voc, par=PAR(num_iter=10, dilations=[1, 2, 4, 8, 12, 24]).to(device).eval(),
        targets=targets, ramp=ramp, digest=digest, sep_loss=confusion_prototype_sep_loss,
        pixel_kd_loss=pixel_kd_loss, get_seg_loss=get_seg_loss,
        build_confusion_evidence=build_confusion_evidence, multi_scale_cam2=multi_scale_cam2,
        cam_to_label=cam_to_label, refine_cams_with_bkg_v2=refine_cams_with_bkg_v2,
        get_mixed_label=get_mixed_label, denormalize_img2=denormalize_img2)
    result = gradient_mode(ctx) if args.mode == 'gradient' else gt_mode(ctx)
    if any(p._version != state_versions[name] for name, p in student.named_parameters()):
        raise RuntimeError('Diagnostic changed a student parameter')
    if any(not torch.equal(value.detach().cpu(), buffers[name]) for name, value in student.named_buffers()):
        raise RuntimeError('Diagnostic changed a student buffer')
    if not tensor_state_equal(observer_before, observer.state_dict(), torch) or not tensor_state_equal(selector_before, selector.state_dict(), torch):
        raise RuntimeError('Diagnostic changed the saved observer or selector')
    if any(p.grad is not None for p in student.parameters()) or any(p.grad is not None or p.requires_grad for p in teacher.parameters()):
        raise RuntimeError('Unexpected accumulated gradients or unfrozen teacher')
    atomic_json(output, {'schema': 1, 'created_utc': now(), 'mode': args.mode, 'passed': True,
        'candidate': 'a_geometry', 'stage': args.step, 'iteration': iteration, 'final_checkpoint': is_final,
        'class_names': CLASSES, 'model_class_ids': list(range(classes)),
        'old_foreground_classes': old_classes, 'new_foreground_class_ids': new_ids,
        'student_checkpoint': str(student_path), 'student_sha256': digest(student_path),
        'teacher_checkpoint': str(teacher_path), 'teacher_sha256': digest(teacher_path),
        'observer_and_selector_checkpoint': str(student_path),
        'saved_observer_updates': int(observer.updates), 'saved_selector': selector.export(),
        'saved_selector_usage': 'Exact cached targets at the requested same-stage checkpoint; no update/refresh/reselection, no cross-candidate replay.',
        'midpoint_caveat': None if is_final else 'Mechanism inspection at an intermediate checkpoint; cannot replace the predefined complete8000 endpoint or justify midpoint cherry-picking.',
        'geometry_ramp': ramp, 'component_weights': {'sep': .1*ramp, 'kd': .1, 'proto_seg': .1},
        'source_sha256': study['source_sha256'], 'diagnostic_script_sha256': digest(Path(__file__)),
        'state_checks': {'student_parameter_versions_unchanged': True, 'student_buffers_unchanged': True,
                         'observer_unchanged': True, 'selector_unchanged': True,
                         'student_no_accumulated_gradients': True, 'teacher_frozen_no_gradients': True},
        **result})
    print(json.dumps({'passed': True, 'mode': args.mode, 'output': str(output)}), flush=True)


if __name__ == '__main__':
    main()
