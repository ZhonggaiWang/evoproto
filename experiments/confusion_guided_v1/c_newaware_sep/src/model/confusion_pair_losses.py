"""Directed pixel separation selected by detached online confusion evidence.

An old anchor must agree with the frozen teacher's argmax; a current-new
anchor does not need teacher agreement. This is supervised logit separation
on trusted PAR/CAM anchors, not direct prototype cosine separation. Pixel GT
is neither accepted nor consumed.
"""
import torch
import torch.nn.functional as F

from model.pixel_kd import distributed_sum, world_size


def directed_pair_sep_loss(student_logits, prototype_logits_scaled,
                           teacher_native_logits, evidence, targets,
                           new_cam_guard=False, valid_cams=None):
    """Return equal main/prototype pair loss under one external coefficient.

    ``targets[i]`` is one foreground partner for anchor i, or -1 when missing.
    The prototype input is already in the same training-logit scale as its
    existing segmentation loss (including the caller's temperature scaling).
    Confidence affects numerators only. Eligible counts supply the existing
    square-root class balance and are synchronized across DDP ranks.
    Optional new-CAM protection attenuates only old-anchor evidence weights;
    every gate, class count and denominator stays identical. Disabled guard
    executes the original numerical expressions and returns original stats.
    """
    student = student_logits
    prototype = prototype_logits_scaled
    if student.ndim != 4 or prototype.ndim != 4 or teacher_native_logits.ndim != 4:
        raise ValueError('Expected NCHW main, prototype and teacher logits')
    n, c = student.shape[:2]
    k = teacher_native_logits.shape[1]
    if prototype.shape[:2] != (n, c) or teacher_native_logits.shape[0] != n or not 2 <= k < c:
        raise ValueError('Prediction batch/classes differ or teacher has no added classes')
    size = student.shape[-2:]
    if prototype.shape[-2:] != size:
        prototype = F.interpolate(prototype, size=size, mode='bilinear', align_corners=False)
    with torch.no_grad():
        if any(key not in evidence for key in ('anchors', 'weights', 'accepted')):
            raise ValueError('Pair evidence requires anchors, weights and accepted')
        values = [evidence[key].detach().to(student.device) for key in ('anchors', 'weights', 'accepted')]
        if any(value.ndim != 3 or value.shape[0] != n for value in values):
            raise ValueError('Pair evidence must be batched NHW')
        if any(value.shape != values[0].shape for value in values):
            raise ValueError('Pair evidence shapes differ')
        if not torch.isfinite(values[1]).all() or (values[1] < 0).any():
            raise ValueError('Pair evidence weights must be finite and nonnegative')
        anchors, weights, accepted = [F.interpolate(
            value[:, None].float(), size=size, mode='nearest')[:, 0] for value in values]
        anchors = anchors.long()
        accepted = accepted.bool()
        target_ids = torch.as_tensor(targets, device=student.device).detach()
        if target_ids.ndim != 1 or target_ids.numel() != c:
            raise ValueError('Pair targets must have one entry per student class')
        if target_ids.dtype.is_floating_point or target_ids.dtype == torch.bool:
            raise ValueError('Pair targets must be integer class IDs, with -1 missing')
        target_ids = target_ids.long()
        if ((target_ids < -1) | (target_ids >= c)).any():
            raise ValueError('Pair target is outside the student class range')
        anchor_safe = anchors.clamp(0, c-1)
        partner = target_ids[anchor_safe]
        partner_safe = partner.clamp(0, c-1)
        candidate = ((anchors > 0) & (anchors < c) & accepted & (weights > 0)
                     & (partner > 0) & (partner < c) & (partner != anchors))
        if 'valid' in evidence:
            candidate &= F.interpolate(evidence['valid'].detach().to(student.device)[:, None].float(),
                                       size=size, mode='nearest')[:, 0].bool()
        teacher = F.interpolate(teacher_native_logits.detach(), size=size,
                                mode='bilinear', align_corners=False)
        teacher_prediction = teacher.argmax(1)
        old = anchors < k
        veto = candidate & old & (teacher_prediction != anchors)
        eligible = candidate & ~veto
        weights = weights * eligible
        if new_cam_guard:
            if valid_cams is None or valid_cams.ndim != 4 or valid_cams.shape[:2] != (n, c-1):
                raise ValueError('New-CAM guard requires N(C-1)HW valid CAMs')
            detached_cams = valid_cams.detach().to(student.device)
            if not torch.isfinite(detached_cams).all():
                raise ValueError('New-CAM guard requires finite CAMs')
            detached_cams = F.interpolate(detached_cams, size=size, mode='bilinear',
                                          align_corners=False).clamp(0, 1)
            max_new_cam = detached_cams[:, k-1:].amax(1)
            guard_factor = torch.where(old, (1-max_new_cam).square(), torch.ones_like(max_new_cam))
            weights_before_guard = weights
            weights = weights * guard_factor
    main_margin = student.gather(1, anchor_safe[:, None])[:, 0] - student.gather(
        1, partner_safe[:, None])[:, 0]
    prototype_margin = prototype.gather(1, anchor_safe[:, None])[:, 0] - prototype.gather(
        1, partner_safe[:, None])[:, 0]
    main_pixel = F.softplus(1-main_margin)
    prototype_pixel = F.softplus(1-prototype_margin)
    group = anchor_safe.flatten()
    main_numerator = torch.zeros(c, device=student.device).scatter_add_(
        0, group, (main_pixel * weights).flatten())
    prototype_numerator = torch.zeros(c, device=student.device).scatter_add_(
        0, group, (prototype_pixel * weights).flatten())
    local_count = torch.zeros(c, device=student.device).scatter_add_(
        0, group, eligible.float().flatten())
    counts = distributed_sum(local_count)
    inv_sqrt = torch.where(counts > 0, counts.clamp_min(1).rsqrt(), 0.)
    denominator = counts.sqrt().sum().clamp_min(1)
    global_main = distributed_sum(main_numerator)
    global_prototype = distributed_sum(prototype_numerator)
    reported_main = (global_main * inv_sqrt).sum()/denominator
    reported_prototype = (global_prototype * inv_sqrt).sum()/denominator
    reported = .5 * (reported_main + reported_prototype)
    local = world_size() * .5 * ((main_numerator + prototype_numerator) * inv_sqrt).sum()/denominator
    loss = reported + (local-local.detach())
    old_mask = eligible & old
    new_mask = eligible & ~old
    packed = distributed_sum(torch.stack((
        candidate.sum().float(), veto.sum().float(), eligible.sum().float(), weights.sum(),
        old_mask.sum().float(), new_mask.sum().float(),
        (main_margin.detach()*old_mask).sum(), (main_margin.detach()*new_mask).sum(),
        (prototype_margin.detach()*old_mask).sum(), (prototype_margin.detach()*new_mask).sum(),
        ((main_margin.detach() <= 0) & old_mask).sum().float(),
        ((main_margin.detach() <= 0) & new_mask).sum().float(),
        ((prototype_margin.detach() <= 0) & old_mask).sum().float(),
        ((prototype_margin.detach() <= 0) & new_mask).sum().float(),
        ((main_margin.detach() < 1) & old_mask).sum().float(),
        ((main_margin.detach() < 1) & new_mask).sum().float(),
        ((prototype_margin.detach() < 1) & old_mask).sum().float(),
        ((prototype_margin.detach() < 1) & new_mask).sum().float())))
    old_den, new_den = packed[4].clamp_min(1), packed[5].clamp_min(1)
    stats = {'pair_sep': float(reported), 'pair_sep_main': float(reported_main),
             'pair_sep_proto': float(reported_prototype),
             'sep_candidate_pixels': int(packed[0]), 'sep_teacher_veto_pixels': int(packed[1]),
             'sep_nonzero_pixels': int(packed[2]), 'sep_weight_sum': float(packed[3]),
             'sep_old_pixels': int(packed[4]), 'sep_new_pixels': int(packed[5]),
             'sep_class_eligible_pixels': counts.tolist(),
             'sep_class_loss': (.5*(global_main+global_prototype)/counts.clamp_min(1)).tolist(),
             'sep_class_main_loss': (global_main/counts.clamp_min(1)).tolist(),
             'sep_class_proto_loss': (global_prototype/counts.clamp_min(1)).tolist(),
             'sep_old_main_margin': float(packed[6]/old_den),
             'sep_new_main_margin': float(packed[7]/new_den),
             'sep_old_proto_margin': float(packed[8]/old_den),
             'sep_new_proto_margin': float(packed[9]/new_den),
             'sep_old_main_error_rate': float(packed[10]/old_den),
             'sep_new_main_error_rate': float(packed[11]/new_den),
             'sep_old_proto_error_rate': float(packed[12]/old_den),
             'sep_new_proto_error_rate': float(packed[13]/new_den),
             'sep_old_main_margin_violation_rate': float(packed[14]/old_den),
             'sep_new_main_margin_violation_rate': float(packed[15]/new_den),
             'sep_old_proto_margin_violation_rate': float(packed[16]/old_den),
             'sep_new_proto_margin_violation_rate': float(packed[17]/new_den),
             'objective': 'trusted_directed_pair_logit_margin'}
    if new_cam_guard:
        # These are detached native-logit supervision masses, not parameter
        # gradient norms, probabilities of correctness, or GT measurements.
        removed_weights = weights_before_guard-weights
        main_push = (1-main_margin.detach()).sigmoid()
        proto_push = (1-prototype_margin.detach()).sigmoid()
        guard_packed = distributed_sum(torch.stack((
            (weights_before_guard*old_mask).sum(), (weights*old_mask).sum(),
            (removed_weights*old_mask).sum(), (max_new_cam*old_mask).sum(),
            (guard_factor*old_mask).sum(),
            (weights_before_guard*new_mask).sum(), (weights*new_mask).sum(),
            (removed_weights*main_push).sum(), (removed_weights*proto_push).sum(),
            (weights_before_guard*main_push).sum(), (weights*main_push).sum(),
            (weights_before_guard*proto_push).sum(), (weights*proto_push).sum(),
            (weights > 0).sum().float(), ((weights > 0) & old_mask).sum().float(),
            ((weights > 0) & new_mask).sum().float(),
            ((guard_factor < 1) & old_mask).sum().float(),
            ((guard_factor == 0) & old_mask).sum().float())))
        stats.update({
            'new_cam_guard': True,
            'guard_old_weight_sum_before': float(guard_packed[0]),
            'guard_old_weight_sum_after': float(guard_packed[1]),
            'guard_old_removed_weight_sum': float(guard_packed[2]),
            'guard_old_mean_new_cam': float(guard_packed[3]/old_den),
            'guard_old_mean_factor': float(guard_packed[4]/old_den),
            'guard_new_weight_sum_before': float(guard_packed[5]),
            'guard_new_weight_sum_after': float(guard_packed[6]),
            'guard_removed_main_push': float(guard_packed[7]),
            'guard_removed_proto_push': float(guard_packed[8]),
            'guard_total_before_main_push': float(guard_packed[9]),
            'guard_total_after_main_push': float(guard_packed[10]),
            'guard_total_before_proto_push': float(guard_packed[11]),
            'guard_total_after_proto_push': float(guard_packed[12]),
            'guard_effective_pixels': int(guard_packed[13]),
            'guard_effective_old_pixels': int(guard_packed[14]),
            'guard_effective_new_pixels': int(guard_packed[15]),
            'guard_attenuated_old_pixels': int(guard_packed[16]),
            'guard_zero_weight_old_pixels': int(guard_packed[17]),
            'guard_definition': 'Only old-anchor numerator weights multiply (1-max_current_new_CAM)^2; unchanged eligible counts/class balance; new-anchor weights unchanged.',
            'guard_push_definition': 'Sum evidence_weight*sigmoid(1-native_pair_margin), main/prototype separately; before external lambda, class balance and 0.5 branch averaging; not parameter-gradient norms.'})
    return loss, stats
