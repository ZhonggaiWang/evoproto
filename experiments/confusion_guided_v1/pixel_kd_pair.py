"""Evidence-gated old-class KD with optional directed-pair distillation.

The disabled pair branch preserves the frozen relational-KD calculation.
Pair selection is detached and never consumes pixel ground truth.
"""
import torch
import torch.distributed as dist
import torch.nn.functional as F


def distributed_sum(value):
    result = value.detach().clone()
    if dist.is_initialized():
        dist.all_reduce(result)
    return result


def world_size():
    return dist.get_world_size() if dist.is_initialized() else 1


def global_ratio(numerator, denominator, offset=0.0):
    den = (distributed_sum(denominator) + offset).clamp_min(1.0)
    num = distributed_sum(numerator)
    return (num + world_size() * (numerator - numerator.detach())) / den


def _native_pair_evidence(evidence, size, batch, device):
    required = ('anchors', 'weights', 'accepted')
    if any(key not in evidence for key in required):
        raise ValueError('Pair evidence requires anchors, weights and accepted')
    values = [evidence[key].detach().to(device=device) for key in required]
    if any(value.ndim != 3 or value.shape[0] != batch for value in values):
        raise ValueError('Pair evidence must be batched NHW')
    if any(value.shape != values[0].shape for value in values):
        raise ValueError('Pair evidence shapes differ')
    if not torch.isfinite(values[1]).all() or (values[1] < 0).any():
        raise ValueError('Pair evidence weights must be finite and nonnegative')
    anchors, weights, accepted = [
        F.interpolate(value[:, None].float(), size=size, mode='nearest')[:, 0]
        for value in values
    ]
    return anchors.long(), weights, accepted.bool()


def _pair_targets(targets, classes, device):
    targets = torch.as_tensor(targets, device=device).detach()
    if targets.ndim != 1 or targets.numel() != classes:
        raise ValueError('Pair targets must have one entry per student class')
    if targets.dtype.is_floating_point or targets.dtype == torch.bool:
        raise ValueError('Pair targets must be integer class IDs, with -1 missing')
    targets = targets.long()
    if ((targets < -1) | (targets >= classes)).any():
        raise ValueError('Pair target is outside the student class range')
    return targets


def pixel_kd_loss(student, teacher, cam_labels, valid_cams, img_box, temperature=2.0,
                  pair_targets=None, pair_evidence=None, pair_blend=0.0):
    """Preserve conditional old-class KL; optionally emphasize selected pairs.

    A usable pair requires an old teacher winner, matching trusted PAR anchor,
    an old-foreground partner, and teacher binary confidence at least 0.75.
    Existing valid-region, new-region, CAM and reliability gates remain intact.
    Its Bernoulli KL mixes into the same per-pixel loss and the same class
    denominator. Background/new outputs are never directly distilled.
    """
    size, k = student.shape[-2:], teacher.shape[1]
    if temperature <= 0 or k >= student.shape[1] or k < 2:
        raise ValueError('Added student classes and positive temperature required')
    pair_enabled = pair_targets is not None and pair_evidence is not None and pair_blend != 0
    if pair_enabled and not 0 < float(pair_blend) <= 0.5:
        raise ValueError('Active pair blend must lie in (0, 0.5]')
    with torch.no_grad():
        teacher = F.interpolate(teacher.detach(), size=size, mode='bilinear', align_corners=False)
        prediction = teacher.argmax(1)
        valid = torch.zeros_like(cam_labels, dtype=torch.float)
        for i, coordinates in enumerate(img_box):
            top, bottom, left, right = [int(x) for x in coordinates]
            valid[i, top:bottom, left:right] = 1
        valid = F.interpolate(valid[:, None], size=size, mode='nearest')[:, 0] > 0
        cam_y = F.interpolate(cam_labels[:, None].float(), size=size, mode='nearest')[:, 0].long()
        new_region = (cam_y >= k) & (cam_y < student.shape[1])
        cams = F.interpolate(valid_cams.detach(), size=size, mode='bilinear', align_corners=False).clamp(0, 1)
        new_cam = cams[:, k-1:].amax(1)
        old_support = cams[:, :k-1].gather(1, (prediction-1).clamp_min(0)[:, None])[:, 0]
        eligible = valid & ~new_region & (prediction > 0) & (old_support >= .25)
        winner_confidence = teacher.sigmoid().gather(1, prediction[:, None])[:, 0]
        reliability = (2 * winner_confidence - 1).clamp_min(0).square()
        weights = eligible * reliability * old_support * (1-new_cam).square()
        scaled_teacher = teacher[:, 1:k] / temperature
        target = scaled_teacher.softmax(1)
        log_target = scaled_teacher.log_softmax(1)
    log_student = (student[:, 1:k] / temperature).log_softmax(1)
    per_pixel = (target * (log_target-log_student)).sum(1) * temperature ** 2
    if pair_enabled:
        with torch.no_grad():
            targets = _pair_targets(pair_targets, student.shape[1], student.device)
            anchors, evidence_weights, accepted = _native_pair_evidence(
                pair_evidence, size, student.shape[0], student.device)
            partner = targets[prediction]
            partner_safe = partner.clamp(1, k-1)
            teacher_margin = teacher.gather(1, prediction[:, None])[:, 0] - teacher.gather(
                1, partner_safe[:, None])[:, 0]
            teacher_difference = teacher_margin / temperature
            pair_probability = teacher_difference.sigmoid()
            usable = (eligible & accepted & (evidence_weights > 0) & (anchors == prediction)
                      & (partner > 0) & (partner < k) & (partner != prediction)
                      & (pair_probability >= .75))
            if 'valid' in pair_evidence:
                evidence_valid = F.interpolate(
                    pair_evidence['valid'].detach().to(student.device)[:, None].float(),
                    size=size, mode='nearest')[:, 0].bool()
                usable &= evidence_valid
            pair_log_target = F.logsigmoid(teacher_difference)
            pair_log_complement = F.logsigmoid(-teacher_difference)
        student_difference = (student.gather(1, prediction[:, None])[:, 0] - student.gather(
            1, partner_safe[:, None])[:, 0]) / temperature
        pair_per_pixel = (pair_probability * (pair_log_target - F.logsigmoid(student_difference))
                          + (1-pair_probability) * (pair_log_complement - F.logsigmoid(-student_difference)))
        pair_per_pixel = pair_per_pixel * temperature ** 2
        blend = float(pair_blend) * usable.to(per_pixel.dtype)
        per_pixel = (1-blend) * per_pixel + blend * pair_per_pixel
    group = prediction.flatten()
    numerator = torch.zeros(k, device=student.device).scatter_add_(0, group, (per_pixel * weights).flatten())
    count = torch.zeros(k, device=student.device).scatter_add_(0, group, eligible.float().flatten())
    counts = distributed_sum(count)
    # Weighted class means with class mass sqrt(count), no per-class singularity.
    inv_sqrt = torch.where(counts > 0, counts.clamp_min(1).rsqrt(), 0.)
    denominator = counts.sqrt().sum().clamp_min(1)
    global_numerator = distributed_sum(numerator)
    reported = (global_numerator * inv_sqrt).sum() / denominator
    local = world_size() * (numerator * inv_sqrt).sum() / denominator
    loss = reported + (local - local.detach())
    packed = distributed_sum(torch.stack((valid.sum(), (valid & new_region).sum(), (weights > 0).sum(),
                                         weights.sum(), (valid * reliability).sum())).float())
    stats = {'pixel_kd': float(reported), 'valid_pixels': int(packed[0]),
             'veto_new_pixels': int(packed[1]), 'kd_nonzero_pixels': int(packed[2]),
             'weight_sum': float(packed[3]), 'mean_teacher_reliability': float(packed[4]/packed[0].clamp_min(1)),
             'class_eligible_pixels': counts.tolist(),
             'class_kd': (global_numerator/counts.clamp_min(1)).tolist(),
             'objective': 'supported_old_foreground_conditional_KL'}
    if pair_enabled:
        effective = usable & (weights > 0)
        pair_counts = distributed_sum(torch.zeros(k, device=student.device).scatter_add_(
            0, group, effective.float().flatten()))
        pair_packed = distributed_sum(torch.stack((
            usable.sum().float(), effective.sum().float(),
            (pair_probability * effective).sum(), (teacher_margin * effective).sum(),
            (pair_per_pixel.detach() * weights * usable).sum())))
        stats.update({'kd_pair_blend': float(pair_blend),
                      'kd_pair_eligible_pixels': int(pair_packed[0]),
                      'kd_pair_pixels': int(pair_packed[1]),
                      'kd_pair_classes': (pair_counts > 0).nonzero(as_tuple=False).flatten().tolist(),
                      'kd_pair_class_pixels': pair_counts.tolist(),
                      'kd_pair_mean_teacher_confidence': float(pair_packed[2]/pair_packed[1].clamp_min(1)),
                      'kd_pair_mean_teacher_margin': float(pair_packed[3]/pair_packed[1].clamp_min(1)),
                      'kd_pair_weighted_kl_sum': float(pair_packed[4])})
    return loss, stats
