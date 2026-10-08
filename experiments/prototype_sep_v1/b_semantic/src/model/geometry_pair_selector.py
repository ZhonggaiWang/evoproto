"""Past broad relations ranked only among supported geometry candidates."""
from types import SimpleNamespace
import torch
from model.directed_pair_selector import DirectedPairSelector


class GeometryPairSelector(DirectedPairSelector):
    SCHEMA=3

    def __init__(self, classes, old_classes, **kwargs):
        if isinstance(old_classes,bool) or int(old_classes)!=old_classes or not 1<=old_classes<classes-1:
            raise ValueError('old_classes must leave at least one new foreground class')
        self.old_classes=int(old_classes)
        super().__init__(classes,**kwargs)

    def get_extra_state(self):
        return {**super().get_extra_state(),'old_classes':self.old_classes,
                'domain':'foreground old-new and new-new unordered geometry',
                'evidence':'broad EMA ranking plus trusted CAM pair exposure support'}

    @torch.no_grad()
    def update(self,observer,iteration):
        # Candidate support is screened without editing the original broad row
        # mass, which includes background, diagonal and old-old observations.
        classes=torch.arange(self.classes,device=observer.broad_pair_image_observations.device)
        domain=(classes[:,None]>self.old_classes)|(classes[None,:]>self.old_classes)
        trusted=observer.pair_image_observations>=self.min_pair_images
        trusted_age=int(observer.updates)-1-observer.last_seen_update
        trusted_fresh=(observer.last_seen_update>=0)&(trusted_age>=0)&(trusted_age<=self.max_stale_updates)
        pair_support=observer.broad_pair_image_observations.clone()
        pair_support.masked_fill_(~(domain & trusted & trusted_fresh[:,None]),-1)
        proxy=SimpleNamespace(classes=observer.classes,stage=observer.stage,updates=observer.updates,
            broad_last_seen_update=observer.broad_last_seen_update,
            broad_image_observations=observer.broad_image_observations,
            broad_ema_counts=observer.broad_ema_counts,
            broad_pair_image_observations=pair_support)
        targets=super().update(proxy,iteration)
        # Parent refresh caching retains ranking, but withdraw support that has
        # gone stale immediately, even before the next ranking refresh.
        rows=torch.arange(self.classes,device=self.targets.device)
        valid=self.targets>=0
        candidate=self.targets.clamp_min(0)
        keep=(domain & trusted & trusted_fresh[:,None]).to(self.targets.device)[rows,candidate]&valid
        self.targets.masked_fill_(~keep,-1)
        self.selected_rates.masked_fill_(~keep,0.)
        return self.targets.detach().clone()

    @torch.no_grad()
    def export(self):
        return {**super().export(),
            'definition':'past full broad EMA row -> highest supported foreground competitor involving a new class; trusted CAM exposure required',
            'rates_usage':'ranking only; full original row normalization; no precise loss weighting'}
