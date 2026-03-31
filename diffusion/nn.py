"""
Core neural-network primitives shared by the DDPM U-Net.
Adapted from SegDiff (github.com/tomeramit/SegDiff).
"""

import math
import torch
import torch.nn as nn


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000):
    """Sinusoidal timestep embeddings (same as transformer positional encoding)."""
    half  = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    args  = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # (B, dim)


def zero_module(module: nn.Module) -> nn.Module:
    """Zero-initialise all parameters (used for final residual projections)."""
    for p in module.parameters():
        p.detach().zero_()
    return module


def normalization(channels: int) -> nn.GroupNorm:
    """GroupNorm with 32 groups (requires channels divisible by 32)."""
    return nn.GroupNorm(32, channels)
