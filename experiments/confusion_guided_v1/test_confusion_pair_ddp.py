"""Four-rank SEP/hybrid-KD equivalence to one heterogeneous global batch.

Run: torchrun --standalone --nproc-per-node=4 test_confusion_pair_ddp.py
The final rank has zero eligible evidence. Model-parameter gradients are
explicitly averaged as DDP does; local batches contain 1, 2, 3 and 4 images.
"""
import json
import os

import torch
import torch.distributed as dist

try:
    from model.confusion_pair_losses import directed_pair_sep_loss
    from model.pixel_kd import pixel_kd_loss
except ModuleNotFoundError:
    from confusion_pair_losses import directed_pair_sep_loss
    from pixel_kd_pair import pixel_kd_loss


def batch(rank, device):
    n, classes, old_classes, h, w = rank + 1, 6, 4, 2, 4
    coordinates = torch.arange(n * 3 * h * w, device=device).reshape(n, 3, h, w)
    features = ((coordinates % 17).float() - 8.) / 8. + rank * .125
    anchors = torch.tensor([[1, 2, 3, 4], [1, 2, 0, 255]], device=device)
    anchors = anchors[None].repeat(n, 1, 1)
    weights = torch.where(anchors == 1, .875, .625).float()
    accepted = (anchors > 0) & (anchors < classes)
    valid = torch.ones_like(anchors, dtype=torch.bool)
    # Old-anchor contradiction at the first pixel, and a teacher margin too
    # weak for pair KD at the second pixel. They have different veto policies.
    winners = anchors.clamp(0, old_classes - 1).clone()
    winners[anchors >= old_classes] = 1
    winners[anchors == 255] = 0
    teacher = torch.zeros(n, old_classes, h, w, device=device)
    teacher.scatter_(1, winners[:, None], 8.)
    teacher[:, 1, 0, 0] = 0.; teacher[:, 2, 0, 0] = 8.
    teacher[:, 2, 0, 1] = 1.; teacher[:, 1, 0, 1] = .8
    cams = torch.full((n, classes - 1, h, w), .125, device=device)
    for old_id in range(1, old_classes):
        cams[:, old_id - 1] = torch.where(teacher.argmax(1) == old_id, .875, .125)
    boxes = [[0, h, 0, w] for _ in range(n)]
    if rank == 2:
        valid[-1, 1] = False
        boxes[-1] = [0, 1, 0, w]
    if rank == 3:
        # Still forwards actual images, but contributes no SEP/KD loss.
        accepted.zero_(); valid.zero_()
        boxes = [[0, 0, 0, 0] for _ in range(n)]
    evidence = {'anchors': anchors, 'weights': weights,
                'accepted': accepted, 'valid': valid}
    return features, teacher, anchors, cams, boxes, evidence


def concatenate(device):
    parts = [batch(rank, device) for rank in range(4)]
    features, teacher, anchors, cams = [torch.cat([p[index] for p in parts]) for index in range(4)]
    boxes = [box for p in parts for box in p[4]]
    evidence = {key: torch.cat([p[5][key] for p in parts]) for key in parts[0][5]}
    return features, teacher, anchors, cams, boxes, evidence


def measure(data, device):
    features, teacher, anchors, cams, boxes, evidence = data
    # Both branches depend on separate shared model parameters, not directly
    # on leaf per-image logits; this matches DDP's model-gradient semantics.
    initial = (torch.arange(18, device=device).float().reshape(6, 3) - 8.) / 7.
    main_weight = initial.clone().requires_grad_()
    prototype_weight = (initial.flip(0) * .75).clone().requires_grad_()
    student = torch.einsum('cd,ndhw->nchw', main_weight, features)
    prototype = torch.einsum('cd,ndhw->nchw', prototype_weight, features)
    targets = torch.tensor([-1, 2, 1, 1, 2, -1], device=device)
    sep, sep_stats = directed_pair_sep_loss(student, prototype, teacher, evidence, targets)
    kd, kd_stats = pixel_kd_loss(student, teacher, anchors, cams, boxes,
                                temperature=2., pair_targets=targets,
                                pair_evidence=evidence, pair_blend=.5)
    sep_grad = torch.autograd.grad(sep, (main_weight, prototype_weight), retain_graph=True)
    kd_grad = torch.autograd.grad(kd, main_weight)[0]
    return {'sep_value': sep.detach(), 'kd_value': kd.detach(),
            'sep_main_gradient': sep_grad[0].detach(),
            'sep_proto_gradient': sep_grad[1].detach(),
            'kd_main_gradient': kd_grad.detach(),
            'sep_stats': sep_stats, 'kd_stats': kd_stats}


def main():
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    # Calculate the true single-process global-batch reference BEFORE any
    # process group exists. This prevents accidental double synchronization.
    reference = measure(concatenate(device), device)
    dist.init_process_group('nccl')
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == 4, 'This test is an explicit four-rank contract.'
    actual = measure(batch(rank, device), device)
    errors = {}
    for key in ['sep_value', 'kd_value']:
        torch.testing.assert_close(actual[key], reference[key], atol=2e-6, rtol=2e-6)
        errors[key] = float((actual[key] - reference[key]).abs().max())
    for key in ['sep_main_gradient', 'sep_proto_gradient', 'kd_main_gradient']:
        gradient = actual[key].clone()
        if rank == 3:
            assert float(gradient.abs().sum()) == 0., key + ' leaks onto zero-evidence rank'
        dist.all_reduce(gradient)
        gradient /= world
        torch.testing.assert_close(gradient, reference[key], atol=2e-6, rtol=2e-6)
        errors[key] = float((gradient - reference[key]).abs().max())
    for key in ['sep_nonzero_pixels', 'sep_old_pixels', 'sep_new_pixels',
                'sep_class_eligible_pixels', 'sep_teacher_veto_pixels']:
        assert actual['sep_stats'][key] == reference['sep_stats'][key], key
    for key in ['kd_pair_pixels', 'kd_pair_classes', 'kd_pair_class_pixels',
                'class_eligible_pixels', 'veto_new_pixels']:
        assert actual['kd_stats'][key] == reference['kd_stats'][key], key
    assert actual['sep_stats']['sep_old_pixels'] > 0
    assert actual['sep_stats']['sep_new_pixels'] > 0
    assert actual['kd_stats']['kd_pair_pixels'] > 0
    if rank == 0:
        print(json.dumps({'test': 'directed_pair_sep_and_hybrid_kd_4rank_equivalence',
                          'ranks': world, 'passed': True, 'rank_batch_sizes': [1, 2, 3, 4],
                          'zero_evidence_rank': 3, 'max_absolute_errors': errors,
                          'sep_global_value': float(reference['sep_value']),
                          'kd_global_value': float(reference['kd_value']),
                          'sep_old_pixels': reference['sep_stats']['sep_old_pixels'],
                          'sep_new_pixels': reference['sep_stats']['sep_new_pixels'],
                          'kd_pair_pixels': reference['kd_stats']['kd_pair_pixels']}))
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
