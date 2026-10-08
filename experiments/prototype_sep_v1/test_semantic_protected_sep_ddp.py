"""Actual DDP sibling-output and pooled-semantic prototype protection tests.

CPU: torchrun --standalone --nproc_per_node=2 test_semantic_protected_sep_ddp.py
8 GPU: torchrun --standalone --nproc_per_node=8 test_semantic_protected_sep_ddp.py --backend nccl

All cases retain find_unused_parameters=True and a real unused projector.
SEM is made from RETURNED logits, not newly made from returned prototypes.
The passing integration constructs the protected scalar INSIDE forward before
DDP's sibling-output wrapping. The unsafe external integration must explicitly
reject an unreachable nonzero semantic derivative rather than call it zero.
"""
import argparse
import datetime
import json
import os
import sys

os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    from model.semantic_protected_sep import semantic_protected_sep
except ModuleNotFoundError:
    from semantic_protected_sep import semantic_protected_sep
try:
    from model.confusion_prototype_sep import confusion_prototype_sep_loss
except ModuleNotFoundError:
    from confusion_prototype_sep import confusion_prototype_sep_loss
try:
    from model.pixel_kd import global_ratio
except ModuleNotFoundError:
    # This is the exact global_ratio contract in the frozen optimized KD SRC.
    # The test concerns its supplied semantic gradient, not its implementation.
    def global_ratio(numerator, denominator, offset=0.):
        num, den = numerator.detach().clone(), denominator.detach().clone()
        world = dist.get_world_size() if dist.is_initialized() else 1
        if dist.is_initialized():
            dist.all_reduce(num); dist.all_reduce(den)
        return (num+world*(numerator-numerator.detach()))/(den+offset).clamp_min(1.)


def initial_raw(device):
    return torch.tensor([[0., 0., 1.], [1., -1., .2], [.5, 1., 1.],
                         [2., 0., 0.], [0., 7., 0.]], dtype=torch.float64, device=device)


def batch(rank, world, case, device):
    count = rank+2
    index = torch.arange(count, dtype=torch.float64, device=device)
    features = torch.stack([.3+rank*.04+index*.02, 1.2+rank*.03+index*.01,
                            .15+index*.01], dim=1)
    # SEM target is old foreground1; the two new prototypes receive a known
    # positive BCE derivative. SEP new3 opposes it, new4 agrees with it.
    labels = torch.ones(count, dtype=torch.long, device=device)
    mask = torch.arange(count, device=device) < rank+1
    if case == 'uneven_with_zero_rank' and rank == world-1:
        mask.zero_()
    elif case == 'all_zero_semantic':
        mask.zero_()
    return features, labels, mask


def semantic_loss(logits, labels, mask):
    targets = F.one_hot(labels, num_classes=5).to(logits.dtype)
    terms = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    # Exactly the formal get_seg_loss: summed class BCE / global valid pixels,
    # with global_ratio's W-scaled local numerator gradient for DDP averaging.
    return .1*global_ratio((terms*mask[:, None]).sum(), mask.sum().to(logits.dtype))


def geometry_loss(p):
    targets = torch.tensor([-1, 3, 4, -1, -1], device=p.device)
    return .1*confusion_prototype_sep_loss(p, targets, 2, margin=0.)[0]


class TinyPrototypeModel(nn.Module):
    def __init__(self, device, protection_inside):
        super().__init__()
        self.raw = nn.Parameter(initial_raw(device))
        self.feature = nn.Parameter(torch.tensor([.2, .3, .1], dtype=torch.float64, device=device))
        # A legitimate unused branch makes disabling find_unused an invalid
        # way to escape the sibling-output issue in this model.
        self.unused_projector = nn.Linear(3, 2, dtype=torch.float64, device=device)
        self.protection_inside = protection_inside
        self.last_stats = None

    def forward(self, features, labels, mask):
        p = F.normalize(self.raw, dim=1)
        logits = (features+self.feature[None])@p.T
        if self.protection_inside:
            geometry = geometry_loss(p)
            semantic = semantic_loss(logits, labels, mask)
            protected, self.last_stats = semantic_protected_sep(p, geometry, semantic, 2)
            return logits, p, protected
        return logits, p


def reference(world, case, device):
    """Independent central projection with the true normalize Jacobian."""
    assert not dist.is_initialized(), 'Reference must not use distributed loss collectives.'
    parts = [batch(rank, world, case, device) for rank in range(world)]
    features, labels, mask = [torch.cat([part[k] for part in parts]) for k in range(3)]
    raw = initial_raw(device).requires_grad_()
    feature = torch.tensor([.2, .3, .1], dtype=torch.float64, device=device, requires_grad=True)
    p = F.normalize(raw, dim=1)
    logits = (features+feature[None])@p.T
    semantic, geometry = semantic_loss(logits, labels, mask), geometry_loss(p)
    gg = torch.autograd.grad(geometry, p, retain_graph=True)[0]
    gs = torch.autograd.grad(semantic, p, retain_graph=True)[0]
    def tangent(gradient):
        return gradient-(gradient*p.detach()).sum(1, keepdim=True)*p.detach()
    h, s = tangent(gg), tangent(gs)
    h[:3] = 0.
    dot, square = (h*s).sum(1), s.square().sum(1)
    conflict = (torch.arange(5, device=device) > 2) & (dot < 0)
    protected = h.clone()
    for row in range(3, 5):
        if conflict[row]:
            protected[row] -= dot[row]/square[row]*s[row]
    protected_raw = torch.autograd.grad((p*protected.detach()).sum(), raw, retain_graph=True)[0]
    semantic_raw, semantic_feature = torch.autograd.grad(semantic, (raw, feature), retain_graph=True)
    raw_dots = (protected_raw*semantic_raw).sum(1)
    assert bool((raw_dots[3:] >= -2e-13).all())
    if case != 'all_zero_semantic':
        assert bool(conflict[3]) and not bool(conflict[4]), 'Test must exercise both projection and no-change.'
        assert float(protected_raw[3].norm()) > 0., 'Projection must retain an orthogonal SEP component.'
    return {'value': (geometry+semantic).detach(), 'geometry_value': geometry.detach(),
        'raw': (semantic_raw+protected_raw).detach(), 'feature': semantic_feature.detach(),
        'protected_partial': protected.detach(), 'semantic_partial': s.detach(),
        'protected_raw': protected_raw.detach(), 'valid_pixels': int(mask.sum()),
        'conflict_rows': [int(row) for row in conflict.nonzero().flatten()]}


def install_hook_counts(model):
    counts = {name: 0 for name, _ in model.named_parameters()}
    handles = []
    for name, parameter in model.named_parameters():
        def hook(gradient, name=name):
            counts[name] += 1
            return gradient
        handles.append(parameter.register_hook(hook))
    return counts, handles


def outside_sibling_test(device, rank, world):
    model = TinyPrototypeModel(device, protection_inside=False)
    hooks, handles = install_hook_counts(model)
    ddp = DDP(model, device_ids=[device.index] if device.type == 'cuda' else None,
              find_unused_parameters=True)
    data = batch(rank, world, 'uneven_all_supported', device)
    logits, returned_p = ddp(*data)
    semantic, geometry = semantic_loss(logits, data[1], data[2]), geometry_loss(returned_p)
    partial = torch.autograd.grad(semantic, returned_p, allow_unused=True, retain_graph=True)[0]
    disconnected = partial is None
    rejected = False
    if disconnected:
        try:
            semantic_protected_sep(returned_p, geometry, semantic, 2)
        except (RuntimeError, ValueError):
            rejected = True
        assert rejected, 'Nonzero SEM sibling output must not silently become zero protection.'
    else:
        # A future DDP version might preserve the connection. It is still
        # required to expose a genuinely nonzero semantic partial derivative.
        assert int(partial.count_nonzero()) > 0
        semantic_protected_sep(returned_p, geometry, semantic, 2)
    assert all(value == 0 for value in hooks.values()), 'Internal component grad touched leaf hooks.'
    assert all(parameter.grad is None for parameter in model.parameters())
    # Finish this reducer iteration normally even when the proposed unsafe
    # integration was rejected; do not leave a pending DDP forward behind.
    (geometry+semantic).backward()
    assert hooks['raw'] == hooks['feature'] == 1
    assert all(value == 0 for name, value in hooks.items() if name.startswith('unused_projector.'))
    for handle in handles:
        handle.remove()
    return {'find_unused_parameters': True, 'semantic_to_returned_p_disconnected': disconnected,
            'unsafe_external_helper_explicitly_rejected': rejected,
            'baseline_final_parameter_hooks_once': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['gloo', 'nccl'], default='gloo')
    args = parser.parse_args()
    world, local_rank = int(os.environ['WORLD_SIZE']), int(os.environ['LOCAL_RANK'])
    assert world in (2, 8), 'Run this deterministic contract with two or eight ranks.'
    if args.backend == 'nccl':
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
    else:
        device = torch.device('cpu')
    torch.set_num_threads(1)
    cases = ['uneven_all_supported', 'uneven_with_zero_rank', 'all_zero_semantic']
    references = {case: reference(world, case, device) for case in cases}
    dist.init_process_group(args.backend, timeout=datetime.timedelta(seconds=90))
    rank = dist.get_rank()
    results = []
    try:
        sibling = outside_sibling_test(device, rank, world)
        dist.barrier()
        model = TinyPrototypeModel(device, protection_inside=True)
        hooks, handles = install_hook_counts(model)
        ddp = DDP(model, device_ids=[device.index] if device.type == 'cuda' else None,
                  find_unused_parameters=True)
        for case in cases:
            ddp.zero_grad(set_to_none=True)
            for name in hooks:
                hooks[name] = 0
            data, expected = batch(rank, world, case, device), references[case]
            logits, returned_p, protected = ddp(*data)
            assert all(value == 0 for value in hooks.values()), 'Inside-forward partial grad reached leaf hooks.'
            assert all(parameter.grad is None for parameter in model.parameters())
            semantic = semantic_loss(logits, data[1], data[2])
            torch.testing.assert_close(protected.detach(), expected['geometry_value'], atol=0, rtol=0)
            torch.testing.assert_close((semantic+protected).detach(), expected['value'], atol=5e-12, rtol=5e-12)
            assert model.last_stats['world_size'] == world
            assert model.last_stats['background_old_sep_gradient_zero'] is True
            if case != 'all_zero_semantic':
                assert model.last_stats['conflicting_new_rows'] == 1
            (semantic+protected).backward()
            errors = {}
            for name in ['raw', 'feature']:
                actual = getattr(model, name).grad
                torch.testing.assert_close(actual, expected[name], atol=5e-11, rtol=5e-11)
                errors[name] = float((actual-expected[name]).abs().max())
                assert hooks[name] == 1, f'{name}: expected exactly one final leaf/reducer hook.'
            for name, parameter in model.named_parameters():
                if name.startswith('unused_projector.'):
                    assert parameter.grad is None and hooks[name] == 0
            json.dumps(model.last_stats, allow_nan=False)
            results.append({'case': case, 'valid_pixels': expected['valid_pixels'],
                'rank_valid_pixels': [int(batch(r, world, case, device)[2].sum()) for r in range(world)],
                'conflicting_rows': expected['conflict_rows'], 'max_absolute_errors': errors,
                'inside_forward_partial_leaf_hooks_zero': True, 'final_leaf_hooks_once': True,
                'unused_projector_grad_none': True})
            dist.barrier()
        for handle in handles:
            handle.remove()
        if rank == 0:
            print(json.dumps({'passed': True, 'test': 'semantic_protected_sep_actual_DDP_pooled_ratio_and_sibling_outputs',
                'backend': args.backend, 'ranks': world, 'sibling_output_probe': sibling,
                'safe_integration': 'inside forward before DDP _DDPSink', 'cases': results}))
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
