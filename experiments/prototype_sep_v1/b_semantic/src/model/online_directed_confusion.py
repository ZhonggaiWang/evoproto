"""Detached online directed confusion. This module has no pixel-GT interface."""
import torch
import torch.nn.functional as F
import torch.distributed as dist


@torch.no_grad()
def build_confusion_evidence(logits, pseudo_labels, valid_cams, img_box,
                             high_threshold=0.7, low_threshold=0.25):
    """Row anchors are pre-teacher-mixing PAR labels; columns use all outputs.

    Segmentation predictions never gate acceptance. Foreground evidence must
    have matching strongest CAM and existing high-threshold support. Weight
    is CAM winner strength times its gap to the strongest alternative CAM.
    Background has a separate low-CAM criterion. These are evidence scores,
    not calibrated probabilities of label correctness.
    """
    labels = pseudo_labels.detach().long()
    if labels.ndim != 3 or logits.ndim != 4 or valid_cams.ndim != 4:
        raise ValueError('Expected labels NHW and logits/CAM NCHW')
    n, c = logits.shape[:2]
    if c < 2 or labels.shape[0] != n or valid_cams.shape[:2] != (n, c-1):
        raise ValueError('Class/channel or batch mismatch')
    if not 0 <= low_threshold < high_threshold <= 1:
        raise ValueError('Invalid evidence thresholds')
    if not torch.isfinite(logits).all() or not torch.isfinite(valid_cams).all():
        raise ValueError('Nonfinite prediction or CAM')
    size = labels.shape[-2:]
    predictions = F.interpolate(logits.detach(), size=size, mode='bilinear',
                                align_corners=False).argmax(1)
    cams = F.interpolate(valid_cams.detach(), size=size, mode='bilinear',
                         align_corners=False).clamp(0, 1)
    top = cams.topk(min(2, c-1), dim=1)
    winner = top.values[:, 0]
    runner = top.values[:, 1] if c > 2 else torch.zeros_like(winner)
    cam_class = top.indices[:, 0] + 1
    valid = torch.zeros_like(labels, dtype=torch.bool)
    if len(img_box) != n:
        raise ValueError('Expected one valid image box per sample')
    for i, coordinates in enumerate(img_box):
        t, b, l, r = [int(x) for x in coordinates]
        if not (0 <= t <= b <= size[0] and 0 <= l <= r <= size[1]):
            raise ValueError('Invalid image box')
        valid[i, t:b, l:r] = True
    broad = valid & (labels >= 0) & (labels < c)
    foreground = broad & (labels > 0) & (cam_class == labels) & (winner >= high_threshold)
    background = broad & (labels == 0) & (winner < low_threshold)
    weights = torch.where(foreground, winner * (winner-runner).clamp_min(0),
                          torch.where(background, 1-winner, torch.zeros_like(winner)))
    accepted = weights > 0
    return {'anchors': labels, 'predictions': predictions, 'weights': weights,
            'accepted': accepted, 'broad': broad, 'valid': valid,
            'cam_strength': winner, 'cam_gap': (winner-runner).clamp_min(0)}


@torch.no_grad()
def pair_counts(rows, columns, classes, weights=None, mask=None):
    if rows.shape != columns.shape or (mask is not None and mask.shape != rows.shape):
        raise ValueError('Count shapes differ')
    good = (rows >= 0) & (rows < classes) & (columns >= 0) & (columns < classes)
    if mask is not None:
        good &= mask
    ids = (rows[good].long()*classes + columns[good].long()).flatten()
    w = None if weights is None else weights[good].double().flatten()
    return torch.bincount(ids, weights=w, minlength=classes*classes).reshape(classes, classes).double()


class OnlineDirectedConfusion(torch.nn.Module):
    """Globally pooled count EMA, serializable and detached from training loss.

    Missing rows retain their previous EMA and expose staleness. Unsupported
    rows are unknown, never fabricated identity/zero-error estimates. Stage
    changes require a fresh estimator with the appropriate class mapping.
    """
    SCHEMA = 2

    def __init__(self, classes, momentum=0.98, high_threshold=0.7,
                 low_threshold=0.25, stage=1):
        super().__init__()
        if classes < 2 or not 0 <= momentum < 1:
            raise ValueError('Invalid classes or momentum')
        if not 0 <= low_threshold < high_threshold <= 1:
            raise ValueError('Invalid evidence thresholds')
        self.classes, self.momentum = int(classes), float(momentum)
        self.high_threshold, self.low_threshold = float(high_threshold), float(low_threshold)
        self.stage = int(stage)
        for name in ['counts', 'accepted_counts', 'broad_counts', 'ema_counts', 'broad_ema_counts',
                     'pair_image_observations', 'broad_pair_image_observations']:
            self.register_buffer(name, torch.zeros(classes, classes, dtype=torch.float64))
        for name in ['image_observations', 'ema_image_observations', 'broad_image_observations']:
            self.register_buffer(name, torch.zeros(classes, dtype=torch.float64))
        self.register_buffer('last_seen_update', torch.full((classes,), -1, dtype=torch.long))
        self.register_buffer('broad_last_seen_update', torch.full((classes,), -1, dtype=torch.long))
        self.register_buffer('updates', torch.zeros((), dtype=torch.long))
        self.register_buffer('seen_images', torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def update(self, logits, pseudo_labels, valid_cams, img_box, synchronize=True):
        if logits.shape[1] != self.classes:
            raise ValueError('Estimator classes differ from prediction head')
        evidence = build_confusion_evidence(logits, pseudo_labels, valid_cams, img_box,
                                             self.high_threshold, self.low_threshold)
        a, p = evidence['anchors'], evidence['predictions']
        weighted = pair_counts(a, p, self.classes, weights=evidence['weights'])
        accepted = pair_counts(a, p, self.classes, mask=evidence['accepted'])
        broad = pair_counts(a, p, self.classes, mask=evidence['broad'])
        observations = torch.zeros(self.classes, dtype=torch.float64, device=a.device)
        broad_observations = torch.zeros_like(observations)
        pair_observations = torch.zeros_like(weighted)
        broad_pair_observations = torch.zeros_like(weighted)
        for image_a, image_p, image_mask, broad_mask in zip(a, p, evidence['accepted'], evidence['broad']):
            observations[image_a[image_mask].unique()] += 1
            broad_observations[image_a[broad_mask].unique()] += 1
            for mask, target in [(image_mask,pair_observations),(broad_mask,broad_pair_observations)]:
                pairs=(image_a[mask]*self.classes+image_p[mask]).unique()
                target.view(-1)[pairs] += 1
        packed = torch.cat([weighted.flatten(), accepted.flatten(), broad.flatten(),
                            pair_observations.flatten(),broad_pair_observations.flatten(),
                            observations,broad_observations, weighted.new_tensor([len(a)])])
        if synchronize and dist.is_initialized():
            dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        m = self.classes**2
        weighted, accepted, broad = [packed[j*m:(j+1)*m].reshape(self.classes, self.classes) for j in range(3)]
        pair_observations=packed[3*m:4*m].reshape(self.classes,self.classes)
        broad_pair_observations=packed[4*m:5*m].reshape(self.classes,self.classes)
        observations=packed[5*m:5*m+self.classes]
        broad_observations=packed[5*m+self.classes:5*m+2*self.classes]
        images=packed[-1]
        present = weighted.sum(1) > 0
        initialized = self.ema_counts.sum(1) > 0
        first = present & ~initialized
        repeated = present & initialized
        self.ema_counts[first] = weighted[first]
        self.ema_counts[repeated] = self.momentum*self.ema_counts[repeated] + (1-self.momentum)*weighted[repeated]
        self.ema_image_observations[first] = observations[first]
        self.ema_image_observations[repeated] = self.momentum*self.ema_image_observations[repeated] + (1-self.momentum)*observations[repeated]
        self.counts += weighted; self.accepted_counts += accepted; self.broad_counts += broad
        self.image_observations += observations
        self.broad_image_observations += broad_observations
        self.pair_image_observations += pair_observations
        self.broad_pair_image_observations += broad_pair_observations
        broad_present=broad.sum(1)>0
        broad_initialized=self.broad_ema_counts.sum(1)>0
        first_broad=broad_present & ~broad_initialized
        repeated_broad=broad_present & broad_initialized
        self.broad_ema_counts[first_broad]=broad[first_broad]
        self.broad_ema_counts[repeated_broad]=self.momentum*self.broad_ema_counts[repeated_broad]+(1-self.momentum)*broad[repeated_broad]
        self.broad_last_seen_update[broad_present]=self.updates
        self.last_seen_update[present] = self.updates
        self.updates += 1; self.seen_images += images.long()
        return evidence

    def get_extra_state(self):
        return {'schema': self.SCHEMA, 'classes': self.classes, 'momentum': self.momentum,
                'high_threshold': self.high_threshold, 'low_threshold': self.low_threshold,
                'stage': self.stage, 'class_ids': list(range(self.classes))}

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError('Confusion state/configuration or stage mismatch')

    @torch.no_grad()
    def export(self):
        def probabilities(counts):
            mass = counts.sum(1)
            return [(row/mass[i]).tolist() if mass[i] > 0 else None for i, row in enumerate(counts)]
        last = self.last_seen_update.cpu()
        stale = [int(self.updates)-1-int(x) if x >= 0 else None for x in last]
        broad_last=self.broad_last_seen_update.cpu()
        broad_stale=[int(self.updates)-1-int(x) if x>=0 else None for x in broad_last]
        return {**self.get_extra_state(), 'updates': int(self.updates), 'seen_images': int(self.seen_images),
                'definition': 'row=pre-mixing PAR pseudo anchor; column=current full-head segmentation argmax',
                'evidence_policy': 'Matching strongest CAM>=high; weighted CAM strength*top2 gap. Background PAR label0 and maxCAM<low; weight1-maxCAM. No prediction agreement filter.',
                'counts': self.counts.cpu().tolist(), 'accepted_counts': self.accepted_counts.cpu().tolist(),
                'broad_counts': self.broad_counts.cpu().tolist(), 'ema_counts': self.ema_counts.cpu().tolist(),
                'probabilities': probabilities(self.counts.cpu()), 'ema_probabilities': probabilities(self.ema_counts.cpu()),
                'broad_probabilities':probabilities(self.broad_counts.cpu()),
                'broad_ema_probabilities':probabilities(self.broad_ema_counts.cpu()),
                'primary_relation_probabilities':probabilities(self.broad_counts.cpu()),
                'primary_relation_ema_probabilities':probabilities(self.broad_ema_counts.cpu()),
                'trusted_probabilities':probabilities(self.counts.cpu()),
                'trusted_ema_probabilities':probabilities(self.ema_counts.cpu()),
                'relation_views':{'primary':'broad pre-mixing PAR confusion: retains hard regions but has noisier anchors',
                                  'trusted_reference':'CAM-gated weighted confusion: more precise anchors but selection-biased',
                                  'usage':'Keep both directed views and evidence support; neither is a calibrated full-population GT confusion matrix. Disagreement is a caution signal, not an instruction to amplify a pair.'},
                'image_observations': self.image_observations.cpu().tolist(),
                'broad_image_observations':self.broad_image_observations.cpu().tolist(),
                'pair_image_observations':self.pair_image_observations.cpu().tolist(),
                'broad_pair_image_observations':self.broad_pair_image_observations.cpu().tolist(),
                'ema_image_observations': self.ema_image_observations.cpu().tolist(),
                'staleness_updates': stale,
                'broad_staleness_updates':broad_stale,
                'image_observation_caveat': 'Repeated training images are exposures, not independent observations; CAM weights are uncalibrated evidence scores.'}
