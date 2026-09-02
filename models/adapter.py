"""
Range-view spatial-prior adapter for a frozen DiT trunk.

Motivation
----------
DiT-XL/2 was trained to denoise SD-VAE latents of 256x256 natural images. A
range view is none of those things: it is 64 x 2048, strongly anisotropic,
cyclic in azimuth, and its channels are metric (range, x, y, z, intensity)
rather than photometric. In the RangeViT-style recipe the only LiDAR-specific
signal the trunk ever receives is the stem output at its input -- the 28 frozen
blocks then run with no further access to range-image structure.

`ViT-Adapter` (Chen et al., ICLR 2023) fixes exactly this for plain ViTs on
dense prediction: a parallel convolutional branch builds a multi-scale spatial
prior, and cross-attention injects it back into the frozen transformer at
several depths. `BALViT` (arXiv 2503.03299) applies the same idea to LiDAR by
injecting a second view into a frozen image backbone, and `RangeSAM`
(arXiv 2509.15886) shows that respecting the horizontal anisotropy of range
images matters when re-using a 2-D foundation model.

This module is the DiT counterpart:

    range image -> SpatialPriorModule -> multi-scale prior tokens
    DiT block k -> SpatialInjector(query = tokens, key/value = prior) -> block k+1

Every injector ends in a zero-initialised gate, so an adapted model starts
numerically identical to the un-adapted one and learns how much prior to admit.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvBNAct(nn.Sequential):
    def __init__(self, cin, cout, k=3, stride=1, padding=1):
        super().__init__(
            nn.Conv2d(cin, cout, k, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(cout),
            nn.GELU())


class SpatialPriorModule(nn.Module):
    """
    Convolutional pyramid over the range image.

    Strides are anisotropic (larger along azimuth than along the 64 scan lines)
    so that each level keeps the wide, short aspect ratio of a range view
    instead of collapsing the 64 rows first.
    """

    def __init__(self, in_channels=5, base_channels=64, d_model=1152,
                 strides=((2, 4), (2, 2), (1, 2))):
        super().__init__()
        self.strides = tuple(tuple(s) for s in strides)

        self.stem = nn.Sequential(
            _ConvBNAct(in_channels, base_channels, k=3, stride=1, padding=1),
            _ConvBNAct(base_channels, base_channels, k=3, stride=1, padding=1))

        levels, projs = [], []
        c = base_channels
        for stride in self.strides:
            cout = min(c * 2, 4 * base_channels)
            levels.append(nn.Sequential(
                _ConvBNAct(c, cout, k=3, stride=stride, padding=1),
                _ConvBNAct(cout, cout, k=3, stride=1, padding=1)))
            projs.append(nn.Conv2d(cout, d_model, kernel_size=1))
            c = cout

        self.levels = nn.ModuleList(levels)
        self.projs = nn.ModuleList(projs)
        self.norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, im):
        """(B, C, H, W) -> (B, N_prior, d_model) tokens from all pyramid levels."""
        x = self.stem(im)
        tokens = []
        for level, proj in zip(self.levels, self.projs):
            x = level(x)
            t = proj(x).flatten(2).transpose(1, 2)     # (B, h*w, d_model)
            tokens.append(t)
        return self.norm(torch.cat(tokens, dim=1))


class SpatialInjector(nn.Module):
    """
    One cross-attention injection point.

    DiT tokens are the queries, the spatial prior supplies keys and values, and
    a zero-initialised per-channel gate scales the result before it is added
    back to the residual stream.
    """

    def __init__(self, d_model, n_heads=8, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        self.norm_q = nn.LayerNorm(d_model, eps=1e-6)
        self.norm_kv = nn.LayerNorm(d_model, eps=1e-6)
        self.q = nn.Linear(d_model, d_model, bias=True)
        self.kv = nn.Linear(d_model, 2 * d_model, bias=True)
        self.proj = nn.Linear(d_model, d_model, bias=True)
        self.drop = nn.Dropout(dropout)

        # adaLN-Zero style: start as a no-op so the frozen trunk is undisturbed.
        self.gamma = nn.Parameter(torch.zeros(d_model))

    def forward(self, x, prior):
        B, N, D = x.shape
        M = prior.shape[1]

        q = self.q(self.norm_q(x)).reshape(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        kv = self.kv(self.norm_kv(prior)).reshape(B, M, 2, self.n_heads, self.head_dim)
        k, v = kv.permute(2, 0, 3, 1, 4)

        out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        out = out.transpose(1, 2).reshape(B, N, D)
        return x + self.gamma * self.drop(self.proj(out))
