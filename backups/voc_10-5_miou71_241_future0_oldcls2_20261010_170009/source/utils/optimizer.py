import torch

import numpy as np


def warmup_argument(optimizer_name, base_lr, warmup_lr):
    """Convert the CLI's absolute starting LR to the scheduler's argument.

    PolyWarmupAdamW expects a multiplier and preserves parameter-group LR
    ratios. CosWarmupAdamW's legacy argument is an absolute starting LR.
    """
    if not np.isfinite(base_lr) or base_lr <= 0:
        raise ValueError('Base learning rate must be finite and positive.')
    if not np.isfinite(warmup_lr) or not 0 <= warmup_lr <= base_lr:
        raise ValueError('Warmup learning rate must be finite and between zero and the base LR.')
    return warmup_lr / base_lr if optimizer_name == 'PolyWarmupAdamW' else warmup_lr


def prototype_parameter_groups(model, param_groups, lr, weight_decay, layer_decay=1.0):
    """Keep heads at 10x LR and optionally decay ViT LR towards lower layers."""
    if not np.isfinite(layer_decay) or not 0 < layer_decay <= 1:
        raise ValueError('Layer decay must be finite and in (0, 1].')
    encoder = getattr(model, 'encoder', None)
    if encoder is not None and hasattr(encoder, 'blocks') and layer_decay < 1:
        depth = len(encoder.blocks)
        layers = {}
        for name, parameter in encoder.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith(('patch_embed.', 'pos_embed', 'cls_token')):
                layer = 0
            elif name.startswith('blocks.'):
                layer = int(name.split('.')[1]) + 1
            else:
                layer = depth + 1
            layers.setdefault(layer, []).append(parameter)
        # Highest backbone LR comes first so existing LR logging stays useful.
        groups = [{'params': layers[layer], 'lr': lr * layer_decay ** (depth + 1 - layer),
                   'weight_decay': weight_decay, 'name': f'backbone_layer_{layer}'}
                  for layer in sorted(layers, reverse=True)]
    else:
        groups = [{'params': parameters, 'lr': lr, 'weight_decay': weight_decay,
                   'name': f'backbone_{index}'} for index, parameters in enumerate(param_groups[:2])]
    groups.append({'params': param_groups[2], 'lr': lr * 10,
                   'weight_decay': weight_decay, 'name': 'cam_classifiers'})
    scale = getattr(getattr(model, 'decoder', None), 'logit_scale', None)
    prototypes = [parameter for parameter in param_groups[3] if parameter is not scale]
    groups.append({'params': prototypes, 'lr': lr * 10,
                   'weight_decay': weight_decay, 'name': 'prototypes'})
    if scale is not None:
        groups.append({'params': [scale], 'lr': lr * 10,
                       'weight_decay': 0.0, 'name': 'prototype_logit_scale'})
    return groups

class CosWarmupAdamW(torch.optim.AdamW):

    def __init__(self, params, lr, weight_decay, betas, warmup_iter=None, max_iter=None, warmup_ratio=None, power=None, **kwargs):
        super().__init__(params, lr=lr, betas=betas, weight_decay=weight_decay, eps=1e-8,)

        self.global_step = 0
        self.warmup_iter = float(warmup_iter)
        self.warmup_ratio = warmup_ratio
        self.max_iter = float(max_iter)
        self.power = power

        self.__init_lr = [group['lr'] for group in self.param_groups]

    def step(self, closure=None):
        ## adjust lr
        if self.global_step < self.warmup_iter:

            lr_mult = self.global_step / self.warmup_iter
            lr_add = (1 - self.global_step / self.warmup_iter) * self.warmup_ratio
            for i in range(len(self.param_groups)):
                self.param_groups[i]['lr'] = self.__init_lr[i] * lr_mult + lr_add

        elif self.global_step < self.max_iter: 

            lr_mult = np.cos((self.global_step - self.warmup_iter) / (self.max_iter - self.warmup_iter) * np.pi) * 0.5 + 0.5
            for i in range(len(self.param_groups)):
                self.param_groups[i]['lr'] = self.__init_lr[i] * lr_mult

        # step
        super().step(closure)

        self.global_step += 1

class PolyWarmupAdamW(torch.optim.AdamW):

    def __init__(self, params, lr, weight_decay, betas, warmup_iter=None, max_iter=None, warmup_ratio=None, power=None, **kwargs):
        super().__init__(params, lr=lr, betas=betas, weight_decay=weight_decay, eps=1e-8,)

        self.global_step = 0
        self.warmup_iter = warmup_iter
        self.warmup_ratio = warmup_ratio
        self.max_iter = max_iter
        self.power = power

        self.__init_lr = [group['lr'] for group in self.param_groups]

    def step(self, closure=None):
        ## adjust lr
        if self.global_step < self.warmup_iter:

            # Include both endpoints: update 1 uses the warmup LR and the final
            # warmup update uses the reference base LR exactly.
            progress = 1.0 if self.warmup_iter <= 1 else self.global_step / (self.warmup_iter - 1)
            lr_mult = self.warmup_ratio + progress * (1 - self.warmup_ratio)
            for i in range(len(self.param_groups)):
                self.param_groups[i]['lr'] = self.__init_lr[i] * lr_mult

        elif self.global_step < self.max_iter: 

            # Start polynomial decay at the peak LR when warmup finishes.
            # The remaining updates form the decay interval, avoiding a jump
            # down just as incremental pixel/prototype losses become active.
            decay_progress = (self.global_step - self.warmup_iter) / (self.max_iter - self.warmup_iter)
            lr_mult = (1 - decay_progress) ** self.power
            for i in range(len(self.param_groups)):
                self.param_groups[i]['lr'] = self.__init_lr[i] * lr_mult

        # step
        super().step(closure)

        self.global_step += 1

class PolyWarmupSGD(torch.optim.SGD):

    def __init__(self, params, lr, weight_decay, warmup_iter=None, max_iter=None, warmup_ratio=None, power=None, **kwargs):
        super().__init__(params, lr=lr, momentum=0.9, weight_decay=weight_decay,)

        self.global_step = 0
        self.warmup_iter = warmup_iter
        self.warmup_lr = warmup_ratio
        self.max_iter = max_iter
        self.power = power

        self.__init_lr = [group['lr'] for group in self.param_groups]

    def step(self, closure=None):
        ## adjust lr
        if self.global_step < self.warmup_iter:

            lr_mult = (1 - self.global_step / self.warmup_iter) ** self.power
            for i in range(len(self.param_groups)):
                self.param_groups[i]['lr'] = self.__init_lr[i] * lr_mult * 10

        elif self.global_step < self.max_iter: 

            lr_mult = (1 - (self.global_step - self.warmup_iter) / (self.max_iter - self.warmup_iter)) ** self.power
            for i in range(len(self.param_groups)):
                self.param_groups[i]['lr'] = self.__init_lr[i] * lr_mult

        # step
        super().step(closure)

        self.global_step += 1
