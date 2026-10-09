"""Class-ambiguity supervision. Incremental training evidence only, no GT masks.
The marginal BCE is -log(P(exactly class a) + P(exactly class b)).
Unlike multi-hot BCE it does not require both candidates to be positive.
"""
import torch
import torch.nn.functional as F


@torch.no_grad()
def select_conflicts(teacher, cams, aux, tags, par, valid, graph, old_count, mode):
    """teacher and CAMs at image resolution; old_count includes background."""
    old = teacher.argmax(1)
    k = cams.shape[1] + 1
    ca = cams.detach() * tags[:, :, None, None]
    cb = aux.detach() * tags[:, :, None, None]
    av, ai = ca.max(1); bv, bi = cb.max(1)
    new = ai + 1
    teacher_conf = teacher.detach().sigmoid().gather(1, old[:, None])[:, 0]
    old_tag = tags.gather(1, (old - 1).clamp(0, old_count - 2).flatten(1)).reshape_as(old) > 0
    local = valid & (old > 0) & old_tag & (teacher_conf >= .7)
    local &= (new >= old_count) & (av >= .7) & (bv >= .7) & (ai == bi) & (par == new)
    linked = (graph[old, new] > 0) | (graph[new, old] > 0)
    selected = local if mode == 'local_pair' else local & linked
    if mode == 'off': selected = torch.zeros_like(selected)
    return selected, old, new, local


def ambiguity_loss(logits, labels, selected, old, new, mode, class_weight, ignore_index=255):
    """Preserve original local-rank valid-pixel denominator for every arm.
    Ignore removes the same selected numerators; candidate mode replaces them.
    All class weights must be one for the exact marginal likelihood.
    """
    if not torch.all(class_weight == 1):
        raise ValueError('Marginal BCE requires unit class weights')
    keep = labels != ignore_index
    selected = selected & keep
    safe = labels.clamp_max(logits.shape[1] - 1)
    target = F.one_hot(safe, logits.shape[1]).permute(0, 3, 1, 2).float()
    per = F.binary_cross_entropy_with_logits(logits, target, reduction='none').sum(1)
    if mode == 'ignore':
        per = per.masked_fill(selected, 0)
    elif mode in ('confusion_pair', 'local_pair'):
        za = logits.gather(1, old[:, None])[:, 0]
        zb = logits.gather(1, new[:, None])[:, 0]
        marginal = F.softplus(logits).sum(1) - torch.logaddexp(za, zb)
        per = torch.where(selected, marginal, per)
    elif mode != 'off':
        raise ValueError(mode)
    return (per * keep).sum() / keep.sum().clamp_min(1)


def resize_mask(mask, size):
    return F.interpolate(mask[:, None].float(), size=size, mode='nearest')[:, 0].bool()
