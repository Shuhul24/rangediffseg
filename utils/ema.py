"""
Exponential moving average of the trained weights.

Standard practice in the DiT codebase this trunk comes from, and close to free:
it mainly buys stability. The per-class IoU of the tail classes swings a lot
from epoch to epoch, because each of those classes is decided by a handful of
pixels per batch; the EMA weights average that noise out instead of leaving the
"best checkpoint" selection to pick a lucky epoch.

Only the *trainable* parameters are shadowed. With the default configuration
9.4 M of the 684 M parameters are trained, so a full `deepcopy` of the model
would spend 2.7 GB shadowing a frozen trunk that never moves.
"""

from contextlib import contextmanager

import torch


class ModelEMA:
    def __init__(self, model, decay=0.999, warmup_steps=1000):
        """
        Args:
            decay        : EMA decay at steady state
            warmup_steps : ramp the decay in over this many steps, so the average
                           is not anchored to the random initialisation
        """
        self.decay = decay
        self.warmup_steps = max(int(warmup_steps), 1)
        self.steps = 0
        self.shadow = {name: p.detach().clone().float()
                       for name, p in model.named_parameters() if p.requires_grad}
        # BatchNorm running stats live in the stem and the decoder and are
        # copied rather than averaged: they already carry their own momentum,
        # and averaging on top of it makes them lag the weights they belong to.
        self.buffers = {name: b.detach().clone()
                        for name, b in model.named_buffers()}

    def _current_decay(self):
        if self.steps >= self.warmup_steps:
            return self.decay
        return min(self.decay, (1 + self.steps) / (10 + self.steps))

    @torch.no_grad()
    def update(self, model):
        self.steps += 1
        d = self._current_decay()
        for name, p in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(d).add_(p.detach().float(), alpha=1.0 - d)
        for name, b in model.named_buffers():
            if name in self.buffers:
                self.buffers[name].copy_(b.detach())

    @contextmanager
    def average_parameters(self, model):
        """Temporarily swap the EMA weights into `model` (validation, saving)."""
        backup = {}
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name in self.shadow:
                    backup[name] = p.detach().clone()
                    p.copy_(self.shadow[name].to(p.dtype))
            buf_backup = {}
            for name, b in model.named_buffers():
                if name in self.buffers:
                    buf_backup[name] = b.detach().clone()
                    b.copy_(self.buffers[name])
        try:
            yield model
        finally:
            with torch.no_grad():
                for name, p in model.named_parameters():
                    if name in backup:
                        p.copy_(backup[name])
                for name, b in model.named_buffers():
                    if name in buf_backup:
                        b.copy_(buf_backup[name])

    def state_dict(self):
        return {'decay': self.decay, 'steps': self.steps,
                'shadow': self.shadow, 'buffers': self.buffers}

    def load_state_dict(self, state):
        self.decay = state.get('decay', self.decay)
        self.steps = state.get('steps', 0)
        for name, t in state.get('shadow', {}).items():
            if name in self.shadow:
                self.shadow[name].copy_(t)
        for name, t in state.get('buffers', {}).items():
            if name in self.buffers:
                self.buffers[name].copy_(t)

    def to(self, device):
        self.shadow = {k: v.to(device) for k, v in self.shadow.items()}
        self.buffers = {k: v.to(device) for k, v in self.buffers.items()}
        return self
