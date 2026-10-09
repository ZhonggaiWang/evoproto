"""Online old-class image-label correction; no annotations enter the memory."""
import torch
import torch.distributed as dist
import torch.nn.functional as F


class OldLabelMemory:
    def __init__(self, old_classes, momentum=0.5):
        if old_classes < 1 or not 0 <= momentum < 1:
            raise ValueError('Invalid OLC configuration')
        self.old_classes = old_classes
        self.momentum = momentum
        self.bank = {}
        self.visits = {}
        self.updates = 0

    @torch.no_grad()
    def update(self, names, main_logits, auxiliary_logits):
        if main_logits.shape != auxiliary_logits.shape or main_logits.shape != (len(names), self.old_classes):
            raise ValueError('OLC expects matching [batch, old classes] logits')
        probabilities = torch.stack([main_logits.sigmoid(), auxiliary_logits.sigmoid()], 1).detach().cpu()
        if not torch.isfinite(probabilities).all():
            raise ValueError('Nonfinite OLC evidence')
        local = [(str(n), p) for n, p in zip(names, probabilities)]
        if dist.is_initialized():
            gathered = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, local)
            local = [row for group in gathered for row in group]
        # Average repeated image IDs in the global batch before one EMA update.
        groups = {}
        for name, p in local:
            groups.setdefault(name, []).append(p)
        for name in sorted(groups):
            current = torch.stack(groups[name]).mean(0)
            if name in self.bank:
                current = self.momentum * self.bank[name] + (1 - self.momentum) * current
            self.bank[name] = current
            self.visits[name] = self.visits.get(name, 0) + 1
        self.updates += 1
        evidence = torch.stack([self.bank[str(n)] for n in names]).to(main_logits.device)
        positive = (evidence > 0.5).all(1)
        negative = (evidence <= 0.5).all(1)
        return positive, positive | negative

    def state_dict(self):
        names = sorted(self.bank)
        return dict(names=names, probabilities=torch.stack([self.bank[n] for n in names]),
                    visits=torch.tensor([self.visits[n] for n in names]),
                    updates=self.updates, momentum=self.momentum, old_classes=self.old_classes)


def masked_classification_loss(logits, targets, known):
    """Unknown entries have zero gradient; retain the baseline B*K denominator."""
    if logits.shape != targets.shape or logits.shape != known.shape:
        raise ValueError('Classification shapes must match')
    return (F.binary_cross_entropy_with_logits(logits, targets.float(), reduction='none') * known).mean()


def mask_untrusted_old(labels, positive, ignore_index=255):
    """Reject unsupported old-class IDs without turning them into background."""
    k = positive.shape[1]
    old = (labels > 0) & (labels <= k)
    accepted = positive.gather(1, (labels.long() - 1).clamp(0, k - 1).flatten(1)).reshape(labels.shape)
    return labels.masked_fill(old & ~accepted, ignore_index)
