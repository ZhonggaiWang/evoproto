"""New-class-aware, class-balanced pixel KD. Training uses no pixel ground truth."""
import math
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
    """Correct gradient for DDP mean reduction and a global-batch denominator."""
    global_denominator = (distributed_sum(denominator) + offset).clamp_min(1.0)
    # Report the actual global objective, while retaining DDP-scaled gradients.
    global_numerator = distributed_sum(numerator)
    return (global_numerator + world_size() * (numerator - numerator.detach())) / global_denominator


def student_to_teacher_logprobs(student, teacher_classes, temperature):
    scaled = student / temperature
    background = torch.logsumexp(torch.cat((scaled[:, :1], scaled[:, teacher_classes:]), dim=1), dim=1, keepdim=True)
    return torch.cat((background, scaled[:, 1:teacher_classes]), dim=1).log_softmax(dim=1)


def pixel_kd_loss(student, teacher, cam_labels, valid_cams, img_box, temperature=2.0):
    """Return DDP-correct loss and detached global diagnostics.

    Veto PAR foreground labels belonging to current new classes. Strong new-CAM
    evidence elsewhere softly attenuates KD. Teacher entropy weights remaining
    pixels. Average within each teacher-predicted class, then across classes.
    Student background and new probabilities are aggregated to teacher BG.
    """
    if temperature <= 0 or teacher.shape[1] >= student.shape[1]:
        raise ValueError('Positive temperature and added student classes required')
    size, k = student.shape[-2:], teacher.shape[1]
    with torch.no_grad():
        teacher = F.interpolate(teacher.detach(), size=size, mode='bilinear', align_corners=False)
        p = teacher.softmax(1)
        prediction = p.argmax(1)
        reliability = (1.0 + (p * p.clamp_min(1e-8).log()).sum(1) / math.log(k)).clamp(0, 1)
        valid = torch.zeros_like(cam_labels, dtype=torch.float)
        for index, coordinates in enumerate(img_box):
            top, bottom, left, right = [int(x) for x in coordinates]
            valid[index, top:bottom, left:right] = 1
        valid = F.interpolate(valid[:, None], size=size, mode='nearest')[:, 0] > 0
        cam_y = F.interpolate(cam_labels[:, None].float(), size=size, mode='nearest')[:, 0].long()
        new_region = (cam_y >= k) & (cam_y < student.shape[1])
        new_cam = F.interpolate(valid_cams[:, k-1:].detach(), size=size, mode='bilinear', align_corners=False).amax(1).clamp(0, 1)
        # Existing CAM high threshold (0.7); no new threshold sweep.
        soft_veto = ((new_cam - 0.7) / 0.3).clamp(0, 1)
        eligible = valid & ~new_region
        weights = reliability * (1 - soft_veto) * eligible
        target = (teacher / temperature).softmax(1)
    log_student = student_to_teacher_logprobs(student, k, temperature)
    per_pixel = (target * (target.clamp_min(1e-8).log() - log_student)).sum(1) * temperature ** 2
    flat_class = prediction.flatten()
    numerator = torch.zeros(k, device=student.device).scatter_add_(0, flat_class, (per_pixel * weights).flatten())
    count = torch.zeros(k, device=student.device).scatter_add_(0, flat_class, eligible.float().flatten())
    global_count = distributed_sum(count)
    present = global_count > 0
    loss = (numerator / global_count.clamp_min(1) * present).sum() * world_size() / present.sum().clamp_min(1)
    packed = torch.stack((valid.sum(), (new_region & valid).sum(), (weights > 0).sum(),
                          weights.sum(), reliability.mul(valid).sum(), (eligible & (prediction == 0)).sum())).float()
    packed = distributed_sum(packed)
    global_numerator = distributed_sum(numerator)
    reported_loss = (global_numerator / global_count.clamp_min(1) * present).sum() / present.sum().clamp_min(1)
    loss = reported_loss + (loss - loss.detach())
    weighted_count = distributed_sum(torch.zeros(k, device=student.device).scatter_add_(0, flat_class, weights.flatten()))
    stats = {
        'pixel_kd':float((global_numerator / global_count.clamp_min(1) * present).sum() / present.sum().clamp_min(1)),
        'valid_pixels':int(packed[0]), 'veto_new_pixels':int(packed[1]),
        'kd_nonzero_pixels':int(packed[2]), 'weight_sum':float(packed[3]),
        'mean_teacher_reliability':float(packed[4]/packed[0].clamp_min(1)),
        'eligible_background_pixels':int(packed[5]),
        'class_eligible_pixels':global_count.tolist(), 'class_weight_sums':weighted_count.tolist(),
        'class_kd':(global_numerator/global_count.clamp_min(1)).tolist(),
    }
    return loss, stats
