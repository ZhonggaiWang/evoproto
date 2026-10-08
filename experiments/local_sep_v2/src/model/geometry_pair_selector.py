"""Rank supported foreground competitors from current new-class anchors only."""
from types import SimpleNamespace
import torch
from model.directed_pair_selector import DirectedPairSelector


class GeometryPairSelector(DirectedPairSelector):
    SCHEMA = 4

    def __init__(self, classes, old_classes, **kwargs):
        if isinstance(old_classes, bool) or int(old_classes) != old_classes or not 1 <= old_classes < classes - 1:
            raise ValueError('old_classes must leave at least one new foreground class')
        self.old_classes = int(old_classes)
        super().__init__(classes, **kwargs)

    def get_extra_state(self):
        return {**super().get_extra_state(), 'old_classes': self.old_classes,
                'domain': 'current new foreground source to any other foreground competitor',
                'source_anchor_policy': 'current_new_only',
                'evidence': 'broad EMA ranking plus trusted CAM pair exposure support'}

    @torch.no_grad()
    def update(self, observer, iteration):
        # Source eligibility is fixed by the incremental class boundary. Keep
        # the original broad row mass, including BG/self, for normalization.
        classes = torch.arange(self.classes, device=observer.broad_pair_image_observations.device)
        domain = (classes[:, None] > self.old_classes).expand(self.classes, self.classes)
        trusted = observer.pair_image_observations >= self.min_pair_images
        trusted_age = int(observer.updates) - 1 - observer.last_seen_update
        trusted_fresh = ((observer.last_seen_update >= 0) & (trusted_age >= 0)
                         & (trusted_age <= self.max_stale_updates))
        pair_support = observer.broad_pair_image_observations.clone()
        pair_support.masked_fill_(~(domain & trusted & trusted_fresh[:, None]), -1)
        proxy = SimpleNamespace(classes=observer.classes, stage=observer.stage, updates=observer.updates,
            broad_last_seen_update=observer.broad_last_seen_update,
            broad_image_observations=observer.broad_image_observations,
            broad_ema_counts=observer.broad_ema_counts,
            broad_pair_image_observations=pair_support)
        super().update(proxy, iteration)
        # Invalid source/support is withdrawn immediately even when cached
        # ranking is not yet due for refresh. Parent handles BG/self targets.
        rows = torch.arange(self.classes, device=self.targets.device)
        valid = self.targets >= 0
        candidate = self.targets.clamp_min(0)
        keep = ((domain & trusted & trusted_fresh[:, None]).to(self.targets.device)[rows, candidate]
                & valid & (candidate > 0) & (candidate != rows))
        self.targets.masked_fill_(~keep, -1)
        self.selected_rates.masked_fill_(~keep, 0.)
        return self.targets.detach().clone()

    @torch.no_grad()
    def export(self):
        return {**super().export(),
            'source_anchor_policy': 'current_new_only',
            'definition': 'past full broad EMA current-new-source row -> highest supported other foreground competitor; trusted CAM exposure required',
            'rates_usage': 'ranking only; full original row normalization; no precise loss weighting'}
