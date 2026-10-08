"""A conditional prototype-SEP surrogate protecting the current semantic direction.

Mathematical contract
---------------------
Let q_i be a trainable raw prototype with r_i=||q_i|| greater than the
normalizer's epsilon, and p_i=q_i/r_i be the unit, NONLEAF prototype passed to
this function. The geometry and semantic losses already contain their actual
external coefficients. Their partial gradients at p are g_G and g_S. Define
the rowwise tangent operator T_i(v)=v-p_i*(p_i dot v). We use

    h_i = T_i(g_G_i)
    s_i = mean_over_DDP_ranks T_i(g_S_i).

This is the actual DDP mean of the supplied semantic component. If its loss
uses global_ratio, each local derivative is W*local_numerator/global_count;
SUM/W therefore gives the pooled semantic derivative. For a rank-local ratio,
the same operation instead gives the actual DDP mean of those local ratios.
The API does not guess the loss normalization or introduce a new denominator.
The geometry gradient must be identical across ranks: synchronized student
prototypes, globally identical selected pairs and identical coefficient.

For EACH new prototype row separately, set d_i=h_i dot s_i and

    h'_i = h_i - (d_i / ||s_i||^2)*s_i       if d_i < 0,
    h'_i = h_i                              otherwise.

BG and old rows of h' are zero. We use the true squared norm, never add an
epsilon to that denominator: adding epsilon would leave a negative residual.
If a conflicting semantic norm is below representable float64 normal range,
the entire SEP row is set to zero. No extra norm cap, hyperparameter or grid
is introduced. Projection arithmetic is float64, then the returned gradient
is cast to p's dtype; floating roundoff is disclosed by diagnostics.

The raw-normalization Jacobian is J_i=T_i/r_i. In exact arithmetic, h' and s
are tangent, so the actual raw prototype component gradients satisfy

    (J_i h'_i) dot (J_i s_i) = (h'_i dot s_i)/r_i^2 >= 0.

Different nonzero raw row norms contribute only positive 1/r_i^2 factors;
they do not reverse this inequality. This proof requires true normalization
above its epsilon. Unit-output validation alone cannot prove that the caller's
raw norm was above epsilon; that is a model-side precondition.

Let linear=sum(p*h'.detach()). The returned scalar is

    geometry.detach() + (linear-linear.detach()).

It has the SAME forward value as the weighted geometry loss, but its gradient
is h', not the original geometry gradient. It is a stop-gradient surrogate,
not a new differentiable scalar objective and not a higher-order derivative.
The ordinary semantic loss must still be added by the caller. Only SEP's old
and BG contributions are suppressed; ordinary semantic gradients remain.

DDP integration precondition
----------------------------
Both losses must have the exact supplied nonleaf p in their ancestor graph.
DDP(find_unused_parameters=True) can wrap separately returned logits and p in
_DDPSink. The wrapped p can then be a sibling of the wrapped logits, so the
semantic loss may no longer reach the returned p. A nonzero semantic loss with
an unreachable p raises explicitly; it is NOT silently treated as zero support.
The safe integration location is before such output wrapping (e.g. inside a
model forward that computes the losses), subject to an actual DDP test. Do not
disable find_unused_parameters without auditing every trainable branch.

autograd.grad stops at this nonleaf p, uses retain_graph=True/create_graph=False,
and must not reach leaf DDP reducer hooks. No p.grad/parameter.grad is populated.
If geometry has no new-row tangent gradient, a graph-connected surrogate with
zero gradient is returned without requiring semantic input or communication;
that branch must agree across replicas. Otherwise EVERY rank participates in
the semantic all-reduce, including ranks with zero semantic support.

Limits
------
This is current Euclidean semantic-direction protection, not a guarantee for
AdamW momentum/preconditioning, encoder dynamics, later teachers, convergence
or IoU. It neither reads pixel GT nor changes pair selection. Current primary
training is deliberately unchanged; this module is only a prepared contingency.
"""
from numbers import Integral
import math

import torch
import torch.distributed as dist


def _validate_loss(loss, prototypes, name, optional=False):
    if loss is None and optional:
        return
    if (not isinstance(loss, torch.Tensor) or loss.ndim != 0 or
            not loss.dtype.is_floating_point or loss.device != prototypes.device):
        raise ValueError(f'{name} must be a floating scalar on the prototype device')
    if not bool(torch.isfinite(loss.detach())):
        raise ValueError(f'{name} must be finite')


def _partial_gradient(loss, prototypes, name):
    if not loss.requires_grad:
        if float(loss.detach()) == 0.:
            return torch.zeros_like(prototypes), False
        raise ValueError(f'Nonzero {name} must have a gradient graph to the supplied nonleaf prototypes')
    gradient = torch.autograd.grad(loss, prototypes, retain_graph=True,
                                   create_graph=False, allow_unused=True)[0]
    if gradient is None:
        if float(loss.detach()) == 0.:
            return torch.zeros_like(prototypes), False
        raise RuntimeError(
            f'Nonzero {name} cannot reach the supplied nonleaf prototypes. '
            'Check sibling DDP _DDPSink output wrapping; compute this helper '
            'before that wrapping rather than silently accepting zero support.')
    if not bool(torch.isfinite(gradient).all()):
        raise FloatingPointError(f'Nonfinite partial gradient of {name}')
    return gradient.detach(), True


def _surrogate(prototypes, weighted_geometry_loss, corrected):
    # Accumulate the linear expression in float64, subtract BEFORE adding the
    # original value, and restore the scalar dtype. No cancellation of the
    # forward geometry value, and no origin-geometry backward path remains.
    linear = (prototypes.to(torch.float64)*corrected.detach().to(torch.float64)).sum()
    zero_value = (linear-linear.detach()).to(weighted_geometry_loss.dtype)
    return weighted_geometry_loss.detach()+zero_value


def _report(prototypes, geometry, before, corrected, semantic, old_classes,
            connected_geometry, connected_semantic, semantic_used, world,
            conflict, tiny_conflict):
    # Report the effective tangent after casting the actual surrogate gradient
    # to the prototype dtype, rather than idealized pre-cast arithmetic only.
    unit = prototypes.detach().to(torch.float64)
    unit = unit/unit.norm(dim=1, keepdim=True)
    effective = corrected.to(torch.float64)
    effective = effective-(effective*unit).sum(1, keepdim=True)*unit
    new = torch.arange(prototypes.shape[0], device=prototypes.device) > old_classes
    old = ~new
    before_norm = before.norm(dim=1)
    after_norm = effective.norm(dim=1)
    semantic_norm = semantic.norm(dim=1) if semantic is not None else None
    dot_before = (before*semantic).sum(1) if semantic is not None else None
    dot_after = (effective*semantic).sum(1) if semantic is not None else None
    tolerance = (64*torch.finfo(prototypes.dtype).eps*after_norm*semantic_norm
                 if semantic is not None else None)
    zero = torch.zeros_like(before_norm)
    unit_norms = prototypes.detach().to(torch.float64).norm(dim=1)
    removed_norm = (before-effective).norm(dim=1)
    row_tensor = torch.stack([before_norm, after_norm,
        semantic_norm if semantic is not None else zero,
        dot_before if semantic is not None else zero,
        dot_after if semantic is not None else zero,
        tolerance if semantic is not None else zero,
        conflict.to(torch.float64), tiny_conflict.to(torch.float64),
        unit_norms, removed_norm], dim=1)
    # One device->CPU transfer for all row fields plus scalar fields. Avoid a
    # separate CUDA synchronization for each class/statistic on every rank.
    packed = torch.cat([row_tensor.flatten(), geometry.detach().to(torch.float64).reshape(1),
        (corrected[old] != 0).sum().to(torch.float64).reshape(1)]).detach().cpu().tolist()
    rows = [packed[index*10:(index+1)*10] for index in range(prototypes.shape[0])]
    entries = []
    for index, row in enumerate(rows):
        b, a, raw_s, raw_db, raw_da, tol, was_conflict, tiny, _, _ = row
        s = raw_s if semantic is not None else None
        db = raw_db if semantic is not None else None
        da = raw_da if semantic is not None else None
        if semantic is not None and index > old_classes and da < -tol:
            raise FloatingPointError('Projection retained a negative tangent dot beyond dtype roundoff')
        entries.append({'class_id': index, 'is_new': index > old_classes,
            'geometry_tangent_norm_before': b, 'geometry_tangent_norm_after': a,
            'semantic_tangent_norm': s, 'dot_before': db, 'dot_after': da,
            'cosine_before': db/(b*s) if s is not None and b > 0 and s > 0 else None,
            'cosine_after': da/(a*s) if s is not None and a > 0 and s > 0 else None,
            'conflict_projected': bool(was_conflict),
            'small_norm_conflict_zeroed': bool(tiny),
            'roundoff_dot_tolerance': tol if semantic is not None else None})
    new_rows = rows[old_classes+1:]
    return {'weighted_geometry_forward_value': packed[-2],
        'forward_value_preserved': True,
        'old_classes': old_classes, 'classes_including_background': prototypes.shape[0],
        'world_size': world, 'semantic_gradient_used': semantic_used,
        'geometry_gradient_connected': connected_geometry,
        'local_semantic_gradient_connected': connected_semantic,
        'semantic_gradient_definition': 'DDP SUM/world of the supplied weighted semantic partial gradient, projected per prototype row; pooled only when the supplied loss uses global_ratio.',
        'geometry_replica_contract': 'Globally identical prototypes, targets and coefficient; SEP partial gradient is replicated, not all-reduced or multiplied by world size.',
        'active_geometry_new_rows': sum(row[0] > 0 for row in new_rows),
        'conflicting_new_rows': sum(bool(row[6]) for row in new_rows),
        'small_norm_conflict_zeroed_rows': sum(bool(row[7]) for row in new_rows),
        'zero_semantic_new_rows': sum(row[2] == 0 for row in new_rows) if semantic is not None else None,
        'geometry_tangent_norm_before': math.hypot(*(row[0] for row in new_rows)),
        'geometry_tangent_norm_after': math.hypot(*(row[1] for row in new_rows)),
        'removed_geometry_tangent_norm': math.hypot(*(row[9] for row in new_rows)),
        'semantic_tangent_norm': math.hypot(*(row[2] for row in new_rows)) if semantic is not None else None,
        'new_tangent_dot_sum_before': sum(row[3] for row in new_rows) if semantic is not None else None,
        'new_tangent_dot_sum_after': sum(row[4] for row in new_rows) if semantic is not None else None,
        'minimum_new_tangent_dot_before': min(row[3] for row in new_rows) if semantic is not None and new_rows else None,
        'minimum_new_tangent_dot_after': min(row[4] for row in new_rows) if semantic is not None and new_rows else None,
        'background_old_sep_gradient_zero': packed[-1] == 0,
        'prototype_unit_norm_min': min(row[8] for row in rows),
        'prototype_unit_norm_max': max(row[8] for row in rows),
        'per_class': entries,
        'gradient_scope': 'Only SEP new-prototype tangent gradients are changed. Semantic loss is still added normally; BG/old semantic gradients are not frozen.',
        'surrogate_definition': 'geometry.detach() + (sum(prototypes*corrected.detach()) - sum(prototypes*corrected.detach()).detach())',
        'no_additive_denominator_epsilon': True, 'no_norm_cap': True,
        'higher_order_gradients': False,
        'raw_gradient_proof': 'For q norm r above normalizer epsilon, J=(I-ppT)/r; each corrected SEP raw-row dot with DDP semantic raw-row gradient equals corrected_tangent_dot/r^2>=0 in exact arithmetic.',
        'limitations': ['Forward geometry value is preserved; its original gradient is deliberately replaced by a detached tangent surrogate.',
            'The raw-gradient inequality is an exact-arithmetic statement; diagnostics allow dtype roundoff.',
            'No guarantee for AdamW preconditioning/momentum, feature learning, later teachers, or final IoU.',
            'The supplied prototypes must be semantic-loss ancestors before any sibling DDP output wrapping.',
            'Zero-geometry early return requires the same geometry decision on all replicas. Zero semantic ranks must still join the common all-reduce.']}


def semantic_protected_sep(prototypes, weighted_geometry_loss, weighted_semantic_loss, old_classes):
    """Return a forward-identical weighted SEP surrogate and JSON diagnostics.

    Inputs: unit-normalized nonleaf CxD prototypes (BG0), scalar already-weighted
    losses, and old foreground count. SEM may be None only when SEP has zero
    new-row tangent gradient. This function never calls backward or writes
    any .grad field, and does not change prototypes or training state.
    """
    if (not isinstance(prototypes, torch.Tensor) or prototypes.ndim != 2 or
            not prototypes.dtype.is_floating_point or min(prototypes.shape) < 1 or
            prototypes.shape[0] < 2):
        raise ValueError('Expected floating CxD prototypes with BG and foreground')
    if not prototypes.requires_grad or prototypes.is_leaf or prototypes.grad_fn is None:
        raise ValueError('Prototypes must be a differentiable normalized NONLEAF network output')
    if (isinstance(old_classes, bool) or not isinstance(old_classes, Integral) or
            not 0 <= old_classes < prototypes.shape[0]):
        raise ValueError('old_classes must be a foreground count in[0,C-1]')
    old_classes = int(old_classes)
    p = prototypes.detach().to(torch.float64)
    norms = p.norm(dim=1, keepdim=True)
    unit_tolerance = max(1e-6, 4*torch.finfo(prototypes.dtype).eps)
    if (not bool(torch.isfinite(p).all()) or not bool(torch.isfinite(norms).all()) or
            bool((norms-1).abs().gt(unit_tolerance).any())):
        raise ValueError('Every supplied prototype row must have finite unit norm')
    _validate_loss(weighted_geometry_loss, prototypes, 'weighted geometry loss')
    # SEM is optional only in the zero-new-geometry branch; validate it later
    # so an absent/non-differentiable semantic objective is not required there.
    unit = p/norms
    def tangent(gradient):
        gradient = gradient.to(torch.float64)
        return gradient-(gradient*unit).sum(1, keepdim=True)*unit
    geometry_partial, geometry_connected = _partial_gradient(weighted_geometry_loss, prototypes,
                                                              'weighted geometry loss')
    before = tangent(geometry_partial)
    before[:old_classes+1] = 0.
    new = torch.arange(prototypes.shape[0], device=prototypes.device) > old_classes
    conflict = torch.zeros(prototypes.shape[0], dtype=torch.bool, device=prototypes.device)
    tiny_conflict = torch.zeros_like(conflict)
    world = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    if not bool(before[new].count_nonzero()):
        corrected = torch.zeros_like(prototypes)
        result = _surrogate(prototypes, weighted_geometry_loss, corrected)
        return result, _report(prototypes, weighted_geometry_loss, before, corrected, None,
            old_classes, geometry_connected, None, False, world, conflict, tiny_conflict)
    _validate_loss(weighted_semantic_loss, prototypes, 'weighted semantic loss')
    semantic_partial, semantic_connected = _partial_gradient(weighted_semantic_loss, prototypes,
                                                              'weighted semantic loss')
    semantic = tangent(semantic_partial)
    if world > 1:
        # No rank-local support branch: zero/unused local semantic gradients
        # participate exactly like supported gradients.
        dist.all_reduce(semantic, op=dist.ReduceOp.SUM)
        semantic /= world
    if not bool(torch.isfinite(semantic).all()):
        raise FloatingPointError('Nonfinite DDP semantic tangent gradient')
    dot = (before*semantic).sum(1)
    square_norm = semantic.square().sum(1)
    if not bool(torch.isfinite(dot).all()) or not bool(torch.isfinite(square_norm).all()):
        raise FloatingPointError('Projection inner products overflowed')
    conflict = new & (dot < 0)
    tiny_conflict = conflict & (square_norm <= torch.finfo(torch.float64).tiny)
    projected = before.clone()
    projected[tiny_conflict] = 0.
    normal = conflict & ~tiny_conflict
    if bool(normal.any()):
        projected[normal] -= (dot[normal]/square_norm[normal])[:, None]*semantic[normal]
    projected[~new] = 0.
    corrected = projected.to(prototypes.dtype).detach()
    if not bool(torch.isfinite(corrected).all()):
        raise FloatingPointError('Corrected SEP gradient is nonfinite after dtype conversion')
    result = _surrogate(prototypes, weighted_geometry_loss, corrected)
    stats = _report(prototypes, weighted_geometry_loss, before, corrected, semantic,
        old_classes, geometry_connected, semantic_connected, True, world, conflict, tiny_conflict)
    return result, stats
