"""Confusion-selected separation of current student foreground prototypes.

Rates select directions outside this loss and never weight its terms. An
unordered old/new pair moves only its new prototype; an unordered new/new
pair moves both. Old/old pairs are left to the existing preservation losses.
Current detached old prototypes share the student's coordinate system;
frozen teacher prototype vectors need not share it under logit-only KD.
"""
import math
from numbers import Integral, Real

import torch
import torch.nn.functional as F


def confusion_prototype_sep_loss(prototypes, pair_targets, old_classes: int, margin=0.0):
    """Return mean squared cosine hinge and JSON-serializable diagnostics.

    ``prototypes`` is floating CxD with background at index0. Old foreground
    IDs are1..old_classes; the remaining foreground IDs are current-new.
    ``pair_targets[i]`` is the selected directed competitor for i, or -1.
    BG/self/old-old directions are skipped. Reverse directions yield only
    one term. All valid selected pairs, including already-separated pairs,
    share the denominator; cosine<=margin gives exactly zero loss/gradient.

    Gradients to old prototypes are stopped only within this SEP objective.
    The caller's existing losses may continue updating those prototypes.
    No valid pairs returns graph-connected zero over the full matrix for DDP.
    """
    if not isinstance(prototypes, torch.Tensor) or prototypes.ndim != 2:
        raise ValueError('Prototypes must be a floating CxD tensor')
    classes, dimensions = prototypes.shape
    if classes < 2 or dimensions < 1 or not prototypes.dtype.is_floating_point:
        raise ValueError('Need at least background+foreground and a positive embedding dimension')
    if not torch.isfinite(prototypes).all():
        raise ValueError('Every prototype must be finite')
    if isinstance(old_classes, bool) or not isinstance(old_classes, Integral) or not 0 <= old_classes < classes:
        raise ValueError('old_classes must be an integer foreground count in [0,C-1]')
    old_classes = int(old_classes)
    if isinstance(margin, bool) or not isinstance(margin, Real) or not math.isfinite(float(margin)) or not -1 <= margin <= 1:
        raise ValueError('Cosine margin must be finite and lie in [-1,1]')
    margin = float(margin)
    targets = torch.as_tensor(pair_targets, device=prototypes.device).detach()
    if targets.ndim != 1 or targets.numel() != classes:
        raise ValueError('Pair targets require one entry per prototype class')
    if targets.dtype.is_floating_point or targets.dtype.is_complex or targets.dtype == torch.bool:
        raise ValueError('Pair targets must contain integer class IDs or -1')
    targets = targets.long()
    if ((targets < -1) | (targets >= classes)).any():
        raise ValueError('Pair target class is outside [-1,C-1]')
    target_ids = targets.cpu().tolist()
    skipped = {'missing': 0, 'background': 0, 'self': 0, 'old_old': 0}
    selected = {}
    for anchor, competitor in enumerate(target_ids):
        if competitor < 0:
            skipped['missing'] += 1
        elif anchor == 0 or competitor == 0:
            skipped['background'] += 1
        elif anchor == competitor:
            skipped['self'] += 1
        elif anchor <= old_classes and competitor <= old_classes:
            skipped['old_old'] += 1
        else:
            pair = tuple(sorted((anchor, competitor)))
            selected.setdefault(pair, []).append([anchor, competitor])
    pairs = sorted(selected)
    involved = sorted({index for pair in pairs for index in pair})
    if involved:
        norms = prototypes.detach()[involved].norm(dim=1)
        if not torch.isfinite(norms).all() or (norms == 0).any():
            raise ValueError('A participating foreground prototype has zero or nonfinite norm')
    terms, entries = [], []
    old_new, new_new = 0, 0
    for first, second in pairs:
        first_is_old = first <= old_classes
        left = prototypes[first].detach() if first_is_old else prototypes[first]
        right = prototypes[second]  # sorted old/new pairs always put new on the right
        cosine = F.cosine_similarity(left, right, dim=0, eps=1e-12).clamp(-1, 1)
        term = F.relu(cosine-margin).square()
        terms.append(term)
        kind = 'old_new' if first_is_old else 'new_new'
        old_new += int(first_is_old)
        new_new += int(not first_is_old)
        entries.append({'class_ids': [first, second], 'pair_type': kind,
                        'selected_directions': selected[(first, second)],
                        'cosine_similarity': float(cosine.detach()), 'margin': margin,
                        'active': bool((cosine.detach() > margin).item()),
                        'pair_loss': float(term.detach()),
                        'old_endpoint_detached': first if first_is_old else None,
                        'gradient_class_ids': [second] if first_is_old else [first, second]})
    # Multiply before summing so even large finite unused vectors cannot
    # overflow a graph-connected zero through sum(prototypes)*0.
    loss = torch.stack(terms).mean() if terms else (prototypes*0.0).sum()
    stats = {'prototype_sep': float(loss.detach()), 'margin': margin,
             'old_classes': old_classes, 'total_classes_including_background': classes,
             'pair_count': len(pairs), 'active_pair_count': sum(entry['active'] for entry in entries),
             'old_new_pair_count': old_new, 'new_new_pair_count': new_new,
             'selected_valid_directions': sum(len(origins) for origins in selected.values()),
             'duplicate_reverse_directions_removed': sum(len(origins)-1 for origins in selected.values()),
             'skipped_missing_directions': skipped['missing'],
             'skipped_background_directions': skipped['background'],
             'skipped_self_directions': skipped['self'],
             'skipped_old_old_directions': skipped['old_old'], 'pairs': entries,
             'objective': 'mean_over_all_selected_unique_pairs_of_relu_cosine_minus_margin_squared',
             'old_reference': 'current_student_prototype_detached_in_SEP_only',
             'rate_usage': 'Directions select unique hard pairs; confusion rates are never loss weights.',
             'endpoint_diagnostic_meaning': 'gradient_class_ids describe allowed differentiable endpoints, not measured nonzero gradients; inactive hinge terms supply zero gradient.',
             'gradient_scope': 'Old/new SEP detaches old endpoint; new/new updates both; background and old/old pairs excluded. Existing losses may still update old prototypes.'}
    return loss, stats
