"""Eight-GPU guarded SEP equals one global batch for actual model gradients.

Run: torchrun --standalone --nproc-per-node=8 test_sep_new_guard_ddp.py
Reference loss/parameter gradients are computed before process-group init.
Unequal image counts, teacher vetoes, current-new anchors and a completely
zero-evidence rank exercise every global class-normalization path.
"""
import json
import math
import os

import torch
import torch.distributed as dist

from model.confusion_pair_losses import directed_pair_sep_loss


def batch(rank, device):
    n, c, k, h, w = rank % 3 + 1, 6, 4, 2, 4
    coordinates = torch.arange(n * 3 * h * w, device=device).reshape(n, 3, h, w)
    features = ((coordinates % 19).float() - 9.) / 8. + rank * .0625
    anchors = torch.tensor([[1, 2, 3, 4], [1, 2, 0, 255]], device=device)
    anchors = anchors[None].repeat(n, 1, 1)
    weights = torch.where(anchors == 1, .875, .625).float()
    accepted = (anchors > 0) & (anchors < c)
    valid = torch.ones_like(anchors, dtype=torch.bool)
    winners = anchors.clamp(0, k - 1).clone()
    winners[anchors >= k] = 1
    winners[anchors == 255] = 0
    teacher = torch.zeros(n, k, h, w, device=device)
    teacher.scatter_(1, winners[:, None], 8.)
    # A high-CAM old anchor still needs the original teacher agreement gate.
    teacher[:, 1, 0, 0] = 0.; teacher[:, 2, 0, 0] = 8.
    cams = torch.full((n, c - 1, h, w), .125, device=device)
    values = ((torch.arange(n * h * w, device=device).reshape(n, h, w) + rank) % 5).float() * .25
    cams[:, k - 1] = values
    cams[:, k] = (values + .25).clamp_max(1.)
    # New anchors are retained even under maximum new evidence.
    cams[:, k - 1, 0, 3] = 1.
    if rank == 2:
        valid[-1, 1] = False
    if rank == 7:
        accepted.zero_(); valid.zero_()
    evidence = {'anchors': anchors, 'weights': weights,
                'accepted': accepted, 'valid': valid}
    return features, teacher, evidence, cams


def combined(device):
    parts = [batch(rank, device) for rank in range(8)]
    features = torch.cat([part[0] for part in parts])
    teacher = torch.cat([part[1] for part in parts])
    evidence = {key: torch.cat([part[2][key] for part in parts]) for key in parts[0][2]}
    cams = torch.cat([part[3] for part in parts])
    return features, teacher, evidence, cams


def measure(data, device):
    features, teacher, evidence, cams = data
    initial = (torch.arange(18, device=device).float().reshape(6, 3) - 8.) / 7.
    main_weight = initial.clone().requires_grad_()
    proto_weight = (initial.flip(0) * .75).clone().requires_grad_()
    main = torch.einsum('cd,ndhw->nchw', main_weight, features)
    prototype = torch.einsum('cd,ndhw->nchw', proto_weight, features)
    targets = torch.tensor([-1, 2, 1, 1, 2, -1], device=device)
    loss, stats = directed_pair_sep_loss(main, prototype, teacher, evidence, targets,
                                        new_cam_guard=True, valid_cams=cams)
    grads = torch.autograd.grad(loss, (main_weight, proto_weight))
    return loss.detach(), [gradient.detach() for gradient in grads], stats


def main():
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    reference_loss, reference_grads, reference_stats = measure(combined(device), device)
    dist.init_process_group('nccl')
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == 8, 'This test explicitly requires eight ranks.'
    actual_loss, actual_grads, actual_stats = measure(batch(rank, device), device)
    torch.testing.assert_close(actual_loss, reference_loss, atol=2e-6, rtol=2e-6)
    errors = {'loss': float((actual_loss - reference_loss).abs())}
    for label, actual, expected in zip(['main_model_parameter_gradient', 'prototype_model_parameter_gradient'],
                                       actual_grads, reference_grads):
        if rank == 7:
            assert float(actual.abs().sum()) == 0., label + ' leaked onto empty rank'
        gradient = actual.clone()
        dist.all_reduce(gradient); gradient /= world
        torch.testing.assert_close(gradient, expected, atol=2e-6, rtol=2e-6)
        errors[label] = float((gradient - expected).abs().max())
    for key in ['sep_class_eligible_pixels', 'sep_teacher_veto_pixels', 'sep_nonzero_pixels',
                'sep_old_pixels', 'sep_new_pixels', 'guard_effective_pixels',
                'guard_effective_old_pixels', 'guard_effective_new_pixels',
                'guard_attenuated_old_pixels', 'guard_zero_weight_old_pixels']:
        assert actual_stats[key] == reference_stats[key], key
    for key in ['sep_weight_sum', 'guard_old_weight_sum_before', 'guard_old_weight_sum_after',
                'guard_old_removed_weight_sum', 'guard_new_weight_sum_before', 'guard_new_weight_sum_after',
                'guard_removed_main_push', 'guard_removed_proto_push',
                'guard_total_before_main_push', 'guard_total_after_main_push',
                'guard_total_before_proto_push', 'guard_total_after_proto_push']:
        assert math.isclose(actual_stats[key], reference_stats[key], abs_tol=2e-6, rel_tol=2e-6), key
    assert actual_stats['sep_old_pixels'] > 0 and actual_stats['sep_new_pixels'] > 0
    assert actual_stats['guard_old_removed_weight_sum'] > 0
    assert actual_stats['guard_new_weight_sum_before'] == actual_stats['guard_new_weight_sum_after']
    assert actual_stats['guard_effective_old_pixels'] < actual_stats['sep_old_pixels']
    if rank == 0:
        print(json.dumps({'test': 'sep_new_cam_guard_8rank_global_batch_parameter_gradient_equivalence',
                          'passed': True, 'ranks': world, 'zero_evidence_rank': 7,
                          'rank_batch_sizes': [rank % 3 + 1 for rank in range(8)],
                          'reference_loss': float(reference_loss), 'max_absolute_errors': errors,
                          'eligible_old_pixels': actual_stats['sep_old_pixels'],
                          'effective_old_pixels': actual_stats['guard_effective_old_pixels'],
                          'eligible_new_pixels': actual_stats['sep_new_pixels'],
                          'removed_old_weight': actual_stats['guard_old_removed_weight_sum']}))
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
