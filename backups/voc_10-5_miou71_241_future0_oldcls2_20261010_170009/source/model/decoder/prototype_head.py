"""Classify ViT patch features directly with learnable class prototypes."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv_head import IncrementalPrototypes


class PrototypeHead(nn.Module):
    def __init__(self, in_planes, classes_list, temperature=0.01):
        super().__init__()
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Prototype temperature must be finite and positive.")
        self.embed_dim = in_planes
        self.classes_list = list(classes_list)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / temperature)))
        self.class_prototypes = IncrementalPrototypes(self.classes_list, self.embed_dim)

    @property
    def temperature(self):
        return self.logit_scale.detach().neg().exp().item()

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        legacy_key = prefix + '_temperature'
        scale_key = prefix + 'logit_scale'
        if legacy_key in state_dict:
            temperature = state_dict.pop(legacy_key)
            if not torch.isfinite(temperature).all() or not (temperature > 0).all():
                error_msgs.append('Legacy prototype temperature must be finite and positive.')
            elif scale_key not in state_dict:
                state_dict[scale_key] = -temperature.log()
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    def forward(self, x, cal_sim=False):
        # x is only a spatial reshape of the ViT patch tokens; no projection or
        # convolution changes their feature space.
        features = F.normalize(x, dim=1)
        prototypes = torch.cat(self.class_prototypes(), dim=0)
        normalized_prototypes = F.normalize(prototypes, dim=1)
        cosine = torch.einsum("bdhw,cd->bchw", features, normalized_prototypes)
        # Preserve the decoder/network return interface. The second output is
        # raw cosine for diagnostics. Apply the learned scale exactly once;
        # Trainers pass these logits directly to CE without another division.
        return cosine * self.logit_scale.exp(), cosine, prototypes
