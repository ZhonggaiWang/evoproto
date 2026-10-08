"""Early-stage ALD: calibrate weak image evidence before hard pseudo fusion.

Only current-new image tags are accepted. Cached old-class states come from
frozen weak evidence; unknown never means negative. A fixed previous best
student supplies soft targets, not ground truth, on discarded hard targets.
"""
import torch
import torch.nn.functional as F
from model.pixel_kd import global_ratio, distributed_sum


@torch.no_grad()
def image_targets(state, old_cls, old_aux, teacher, reference, new_tags):
    assert state.shape == old_cls.shape == old_aux.shape
    assert state.shape[1] == 15 and new_tags.shape[1] == 5
    tp, rp = teacher.argmax(1), reference.argmax(1)
    tpresent = torch.stack([(tp == c).flatten(1).any(1) for c in range(1, 16)], 1)
    rpresent = torch.stack([(rp == c).flatten(1).any(1) for c in range(1, 16)], 1)
    target = (old_cls.sigmoid() + old_aux.sigmoid()) * .5
    # A positive classifier with no dense evidence is ambiguous, not absent.
    conflict = (state < 0) & ((old_cls > 0) | (old_aux > 0)) & ~tpresent & ~rpresent
    target[conflict] = .5
    target = torch.where(state >= 0, state.clamp_min(0).float(), target)
    allowed = (state == 1) | ((state < 0) & ((old_cls > 0) | (old_aux > 0) | tpresent | rpresent))
    return torch.cat([target, new_tags.float()], 1), torch.cat([allowed, new_tags.bool()], 1), conflict


@torch.no_grad()
def fuse(par, teacher_logits, reference_logits, state, img_box):
    size = par.shape[-2:]
    tp = F.interpolate(teacher_logits.detach(), size, mode='bilinear', align_corners=False).argmax(1)
    rp = F.interpolate(reference_logits.detach(), size, mode='bilinear', align_corners=False).argmax(1)
    admitted = state.gather(1, (tp-1).clamp(0, 14).flatten(1)).reshape_as(tp) == 1
    old = (tp > 0) & (tp < 16) & admitted
    new = (par >= 16) & (par < 21)
    bg = (tp == 0) & (rp == 0)
    valid = torch.zeros_like(par, dtype=torch.bool)
    for i, box in enumerate(img_box):
        t, b, l, r = map(int, box)
        valid[i, t:b, l:r] = True
    out = torch.full_like(par, 255)
    out[old] = tp[old]
    out[bg] = 0
    out[new] = par[new]  # Retain the established current-new PAR supervision.
    out[~valid] = 255
    return {'labels': out, 'trusted_old': old & ~new & valid,
            'unknown': (out == 255) & valid, 'valid': valid,
            'reference_prediction': rp}


@torch.no_grad()
def gate_auxiliary(labels, state, reference_logits):
    """Calibrate PTC/confusion anchors, without current-student agreement gates."""
    out = labels.clone()
    old = (out > 0) & (out < 16)
    admitted = state.gather(1, (out-1).clamp(0, 14).flatten(1)).reshape_as(out) == 1
    rp = F.interpolate(reference_logits.detach(), out.shape[-2:], mode='bilinear', align_corners=False).argmax(1)
    out[old & ~admitted] = 255
    out[(out == 0) & (rp != 0)] = 255
    return out


def uncertain_loss(student, reference, unknown, valid, state, new_tags, temperature=2.):
    """Soft allowed-set target, normalized by valid area (not shrinking unknowns)."""
    size = student.shape[-2:]
    with torch.no_grad():
        ref = F.interpolate(reference.detach(), size, mode='bilinear', align_corners=False)
        allowed = torch.cat([torch.ones_like(state[:, :1], dtype=torch.bool), state != 0, new_tags.bool()], 1)
        target_logits = (ref / temperature).masked_fill(~allowed[:, :, None, None], -1e4)
        log_target = target_logits.log_softmax(1)
        target = log_target.exp()
        mask = F.interpolate(unknown[:, None].float(), size, mode='nearest')[:, 0]
        area = F.interpolate(valid[:, None].float(), size, mode='nearest')[:, 0]
    per_pixel = (target * (log_target - (student / temperature).log_softmax(1))).sum(1) * temperature**2
    return global_ratio((per_pixel * mask).sum(), area.sum())


def stats(fused, state, conflict, soft_loss):
    values = torch.stack([fused['valid'].sum(), fused['unknown'].sum(), fused['trusted_old'].sum(),
                          ((fused['labels'] >= 16) & (fused['labels'] < 21)).sum(),
                          (state == 1).sum(), (state == 0).sum(), (state < 0).sum(), conflict.sum()]).float()
    values = distributed_sum(values).tolist()
    return dict(zip(['valid_pixels', 'unknown_pixels', 'trusted_old_pixels', 'new_PAR_pixels',
                     'old_positive_image_exposures', 'old_negative_image_exposures',
                     'old_unknown_image_exposures', 'neutralized_conflict_exposures'], values),
                uncertain_soft_KL=float(soft_loss.detach()), calibration='frozen_weak_image_state_plus_augmented_dense_evidence')
