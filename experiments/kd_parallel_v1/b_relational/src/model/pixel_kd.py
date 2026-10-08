"""Evidence-gated conditional old-class relational KD."""
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


def pixel_kd_loss(student, teacher, cam_labels, valid_cams, img_box, temperature=2.0):
    """Do not couple new/background outputs to old-output distillation.

    Teacher foreground predictions require matching old-class CAM support.
    All new-CAM evidence continuously discounts that support; refined new
    regions veto KD entirely. Square-root support weighting prevents a tiny
    class island receiving the same batch influence as a large object.
    No training pixel ground truth is used.
    """
    size, k = student.shape[-2:], teacher.shape[1]
    if temperature <= 0 or k >= student.shape[1] or k < 2:
        raise ValueError('Added student classes and positive temperature required')
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
    return loss, stats
