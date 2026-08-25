"""
Warmup + cosine learning-rate schedule (stepped once per iteration).

From lidar-bonnetal / SalsaNext, as vendored by RangeViT
(github.com/valeoai/rangevit).
"""

import torch.optim.lr_scheduler as toptim


class WarmupCosineLR(toptim._LRScheduler):
    """Linearly warm the learning rate up, then anneal it with a cosine."""

    def __init__(self, optimizer, lr, warmup_steps, momentum, max_steps):
        self.optimizer = optimizer
        self.lr = lr
        self.warmup_steps = max(int(warmup_steps), 1)
        self.momentum = momentum

        self.cosine_scheduler = toptim.CosineAnnealingLR(self.optimizer, T_max=max(int(max_steps), 1))
        self.initial_scheduler = toptim.CyclicLR(
            self.optimizer,
            base_lr=0,
            max_lr=self.lr,
            step_size_up=self.warmup_steps,
            step_size_down=self.warmup_steps,
            cycle_momentum=False,
            base_momentum=self.momentum,
            max_momentum=self.momentum)

        self.last_epoch = -1
        self.finished = False
        super().__init__(optimizer)

    def step(self, epoch=None):
        if self.finished or self.initial_scheduler.last_epoch >= self.warmup_steps:
            if not self.finished:
                self.base_lrs = [self.lr for _ in self.base_lrs]
                self.finished = True
            return self.cosine_scheduler.step(epoch)
        return self.initial_scheduler.step(epoch)
