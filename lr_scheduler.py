# --------------------------------------------------------
# Reversible Column Networks
# Copyright (c) 2022 Megvii Inc.
# Licensed under The Apache License 2.0 [see LICENSE for details]
# Written by Yuxuan Cai
# --------------------------------------------------------

import torch
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.step_lr import StepLRScheduler
from timm.scheduler.scheduler import Scheduler


def build_scheduler(config, optimizer=None):

    lr_scheduler = None
    if config.TRAIN.LR_SCHEDULER.NAME == 'cosine':
        lr_scheduler = CosLRScheduler()
    elif config.TRAIN.LR_SCHEDULER.NAME == 'multistep':
        lr_scheduler = StepLRScheduler()
    else:
        raise NotImplementedError(f"Unkown lr scheduler: {config.TRAIN.LR_SCHEDULER.NAME}")

    return lr_scheduler

import math


class CosLRScheduler():
    def __init__(self) -> None:
        pass

    def step_update(self, optimizer, epoch, config):
        """Decay the learning rate with half-cycle cosine after warmup"""
        if epoch < config.TRAIN.WARMUP_EPOCHS:
            lr = (config.TRAIN.BASE_LR-config.TRAIN.WARMUP_LR) * epoch / config.TRAIN.WARMUP_EPOCHS + config.TRAIN.WARMUP_LR
        else:
            lr = config.TRAIN.MIN_LR + (config.TRAIN.BASE_LR - config.TRAIN.MIN_LR) * 0.5 * \
                (1. + math.cos(math.pi * (epoch - config.TRAIN.WARMUP_EPOCHS ) / (config.TRAIN.EPOCHS  - config.TRAIN.WARMUP_EPOCHS )))
        for param_group in optimizer.param_groups:
            if "lr_scale" in param_group:
                param_group["lr"] = lr * param_group["lr_scale"]
            else:
                param_group["lr"] = lr
        return lr

class LinearLRScheduler(Scheduler):
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 t_initial: int,
                 lr_min_rate: float,
                 warmup_t=0,
                 warmup_lr_init=0.,
                 t_in_epochs=True,
                 noise_range_t=None,
                 noise_pct=0.67,
                 noise_std=1.0,
                 noise_seed=42,
                 initialize=True,
                 ) -> None:
        super().__init__(
            optimizer, param_group_field="lr",
            noise_range_t=noise_range_t, noise_pct=noise_pct, noise_std=noise_std, noise_seed=noise_seed,
            initialize=initialize)

        self.t_initial = t_initial
        self.lr_min_rate = lr_min_rate
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.t_in_epochs = t_in_epochs
        if self.warmup_t:
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in self.base_values]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]

    def _get_lr(self, t):
        if t < self.warmup_t:
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            t = t - self.warmup_t
            total_t = self.t_initial - self.warmup_t
            lrs = [v - ((v - v * self.lr_min_rate) * (t / total_t)) for v in self.base_values]
        return lrs

    def get_epoch_values(self, epoch: int):
        if self.t_in_epochs:
            return self._get_lr(epoch)
        else:
            return None

    def get_update_values(self, num_updates: int):
        if not self.t_in_epochs:
            return self._get_lr(num_updates)
        else:
            return None
