"""Select supported directed foreground competitors from past online counts.

Rates rank candidate pairs; they never become calibrated loss weights. Call
``update`` before observing the current training batch. Image observations
are training exposures, including repeated visits, rather than independent
examples. The selected direction is anchor class -> competing output class.
"""
import math

import torch


class DirectedPairSelector(torch.nn.Module):
    SCHEMA = 1

    def __init__(self, classes, stage=1, refresh_interval=50,
                 min_row_images=8, min_pair_images=3, min_rate=.01,
                 min_updates=100, ramp_updates=200, max_stale_updates=200):
        super().__init__()
        integers = {'classes': classes, 'stage': stage,
                    'refresh_interval': refresh_interval,
                    'min_row_images': min_row_images,
                    'min_pair_images': min_pair_images,
                    'min_updates': min_updates, 'ramp_updates': ramp_updates,
                    'max_stale_updates': max_stale_updates}
        for name, value in integers.items():
            if isinstance(value, bool) or int(value) != value:
                raise ValueError(f'{name} must be an integer')
            setattr(self, name, int(value))
        if self.classes < 2 or self.stage < 1:
            raise ValueError('At least two classes and a positive incremental stage required')
        if self.refresh_interval < 1 or self.ramp_updates < 1:
            raise ValueError('Positive refresh and ramp intervals required')
        if min(self.min_row_images, self.min_pair_images, self.min_updates,
               self.max_stale_updates) < 0:
            raise ValueError('Support, warmup and staleness limits must be nonnegative')
        if not math.isfinite(float(min_rate)) or not 0 <= min_rate <= 1:
            raise ValueError('min_rate must be finite and in [0, 1]')
        self.min_rate = float(min_rate)
        self.register_buffer('targets', torch.full((self.classes,), -1, dtype=torch.long))
        self.register_buffer('selected_rates', torch.zeros(self.classes, dtype=torch.float64))
        self.register_buffer('last_refresh_iteration', torch.full((), -1, dtype=torch.long))

    def get_extra_state(self):
        return {'schema': self.SCHEMA, 'classes': self.classes, 'stage': self.stage,
                'refresh_interval': self.refresh_interval,
                'min_row_images': self.min_row_images,
                'min_pair_images': self.min_pair_images,
                'min_rate': self.min_rate, 'min_updates': self.min_updates,
                'ramp_updates': self.ramp_updates,
                'max_stale_updates': self.max_stale_updates,
                'class_ids': list(range(self.classes))}

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError('Directed pair selector state/configuration or stage mismatch')

    def _observer_updates(self, observer):
        if observer.classes != self.classes or observer.stage != self.stage:
            raise ValueError('Observer/selector classes or stage mismatch')
        updates = int(observer.updates)
        if updates < 0:
            raise ValueError('Observer update count must be nonnegative')
        return updates

    def ramp(self, observer):
        updates = self._observer_updates(observer)
        return min(1.0, max(0.0, (updates-self.min_updates)/self.ramp_updates))

    @torch.no_grad()
    def update(self, observer, iteration):
        """Return a detached copy of the cached competitor for each class.

        Unknown/background rows use -1. Refreshes select at most one target
        from *past* EMA rows after sufficient observation warmup. Between
        refreshes, stale cached rows are immediately withdrawn, while ranking
        changes wait for the next refresh. EMA mass includes background and
        the diagonal when normalizing a row; those columns are not candidates.
        """
        updates = self._observer_updates(observer)
        if isinstance(iteration, bool) or int(iteration) != iteration or iteration < 0:
            raise ValueError('Training iteration must be a nonnegative integer')
        iteration = int(iteration)
        last_refresh = int(self.last_refresh_iteration)
        if last_refresh > iteration:
            raise ValueError('Training iteration precedes restored selector state')
        device = self.targets.device
        last_seen = observer.broad_last_seen_update.detach().to(device)
        row_images = observer.broad_image_observations.detach().to(device)
        if last_seen.shape != (self.classes,) or row_images.shape != (self.classes,):
            raise ValueError('Observer row support shape mismatch')
        age = updates - 1 - last_seen
        fresh = (last_seen >= 0) & (age >= 0) & (age <= self.max_stale_updates)
        supported_rows = fresh & (row_images >= self.min_row_images)
        supported_rows[0] = False
        if updates < self.min_updates:
            self.targets.fill_(-1)
            self.selected_rates.zero_()
            return self.targets.detach().clone()

        # Never retain stale directions simply because ranking refresh is due later.
        self.targets.masked_fill_(~supported_rows, -1)
        self.selected_rates.masked_fill_(~supported_rows, 0.)
        if last_refresh >= 0 and iteration-last_refresh < self.refresh_interval:
            return self.targets.detach().clone()

        counts = observer.broad_ema_counts.detach().to(device=device, dtype=torch.float64)
        pair_images = observer.broad_pair_image_observations.detach().to(device)
        if counts.shape != (self.classes, self.classes) or pair_images.shape != counts.shape:
            raise ValueError('Observer pair support shape mismatch')
        if not torch.isfinite(counts).all() or (counts < 0).any():
            raise ValueError('Observer EMA counts must be finite and nonnegative')
        mass = counts.sum(1)
        rates = counts / mass.clamp_min(torch.finfo(counts.dtype).tiny)[:, None]
        eligible = ((pair_images >= self.min_pair_images) & (rates >= self.min_rate)
                    & (counts > 0) & supported_rows[:, None] & (mass > 0)[:, None])
        eligible[:, 0] = False
        eligible.fill_diagonal_(False)
        ranked = rates.masked_fill(~eligible, -1.)
        best_rate, best_target = ranked.max(1)  # max returns the lowest index on ties.
        known = best_rate >= 0
        self.targets.copy_(torch.where(known, best_target, torch.full_like(best_target, -1)))
        self.selected_rates.copy_(torch.where(known, best_rate, torch.zeros_like(best_rate)))
        self.last_refresh_iteration.fill_(iteration)
        return self.targets.detach().clone()

    @torch.no_grad()
    def export(self):
        return {**self.get_extra_state(), 'targets': self.targets.cpu().tolist(),
                'selected_rates': self.selected_rates.cpu().tolist(),
                'last_refresh_iteration': int(self.last_refresh_iteration),
                'definition': 'past broad EMA row i -> highest supported foreground competitor j',
                'rates_usage': 'ranking and diagnostics only; not calibrated loss strengths'}
