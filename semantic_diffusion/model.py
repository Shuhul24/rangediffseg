"""Geometry-conditioned clean-label DiT; no independent segmentation branch.

State: full-resolution centered one-hot semantic codes. Strided patchification
packs each patch, rather than nearest-neighbor downsampling labels. Conditioning
is cached once per scan window. No circular padding on partial panoramas.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


def conv(cin, cout, stride=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride=stride, padding=1),
                         nn.GroupNorm(8, cout), nn.SiLU())


def time_embedding(t, dim):
    f = torch.exp(-math.log(10000) * torch.arange(dim // 2, device=t.device) / (dim // 2))
    a = t[:, None] * 1000 * f[None]
    return torch.cat([a.cos(), a.sin()], -1)


def alpha_bar(t):
    # Exact endpoints: pure Gaussian at t=1; clean code at t=0.
    return torch.cos(t.clamp(0, 1) * math.pi / 2).square().clamp(0, 1)


class Block(nn.Module):
    def __init__(self, dim, heads, window=8):
        super().__init__()
        self.window = window
        self.norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm_cross = nn.LayerNorm(dim)
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        self.mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        nn.init.zeros_(self.mod[-1].weight)
        nn.init.zeros_(self.mod[-1].bias)
        self.local = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)

    def forward(self, x, t, context):
        b, d, h, w = x.shape
        shift, scale, g1, g2, g3, g4 = self.mod(t).chunk(6, -1)
        tokens = x.flatten(2).transpose(1, 2)
        q = self.norm(tokens) * (1 + scale[:, None]) + shift[:, None]
        q = q.transpose(1, 2).reshape(b, d, h, w)
        wh, ww = min(self.window, h), min(self.window, w)
        ph, pw = (-h) % wh, (-w) % ww
        q = F.pad(q, (0, pw, 0, ph))
        hp, wp = h + ph, w + pw
        q = q.reshape(b, d, hp // wh, wh, wp // ww, ww).permute(0, 2, 4, 3, 5, 1)
        q = q.reshape(-1, wh * ww, d)
        valid = F.pad(torch.ones(b, 1, h, w, device=x.device), (0, pw, 0, ph))
        padding = valid.reshape(b, 1, hp // wh, wh, wp // ww, ww).permute(0, 2, 4, 3, 5, 1)
        padding = padding.reshape(-1, wh * ww).eq(0)
        a = self.self_attn(q, q, q, key_padding_mask=padding, need_weights=False)[0]
        a = a.reshape(b, hp // wh, wp // ww, wh, ww, d).permute(0, 5, 1, 3, 2, 4)
        a = a.reshape(b, d, hp, wp)[:, :, :h, :w].flatten(2).transpose(1, 2)
        tokens = tokens + g1[:, None] * a
        a = self.cross_attn(self.norm_cross(tokens), context, context, need_weights=False)[0]
        tokens = tokens + g2[:, None] * a
        tokens = tokens + g3[:, None] * self.mlp(self.norm_mlp(tokens))
        x = tokens.transpose(1, 2).reshape(b, d, h, w)
        return x + g4[:, :, None, None] * self.local(x)


class SemanticDiT(nn.Module):
    def __init__(self, classes=19, dim=256, depth=6, heads=8, patch=(2, 8),
                 code_scale=0.5, mode='diffusion'):
        super().__init__()
        if mode not in ('diffusion', 'deterministic'):
            raise ValueError(mode)
        self.classes, self.patch, self.code_scale, self.mode = classes, tuple(patch), code_scale, mode
        self.stem = nn.Sequential(conv(6, 32), conv(32, 64))
        self.geometry = nn.Conv2d(64, dim, self.patch, stride=self.patch)
        self.multiscale = nn.Sequential(conv(dim, dim), conv(dim, dim))
        self.position = nn.Conv2d(4, dim, 1)
        self.state = nn.Conv2d(classes * 2, dim, self.patch, stride=self.patch)
        self.time = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.unpatch = nn.ConvTranspose2d(dim, 64, self.patch, stride=self.patch)
        self.detail_gate = nn.Conv2d(64, 64, 1)
        self.decoder = nn.Sequential(conv(128, 64), nn.Conv2d(64, classes, 1))

    def encode(self, x, valid):
        if x.shape[-2] % self.patch[0] or x.shape[-1] % self.patch[1]:
            raise ValueError('Input dimensions must be divisible by patch size')
        fine = self.stem(torch.cat([x, valid[:, None].to(x.dtype)], 1))
        condition = self.geometry(fine)
        b, _, h, w = condition.shape
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, h, device=x.device),
                                torch.linspace(-1, 1, w, device=x.device), indexing='ij')
        pos = torch.stack([yy, xx, torch.sin(math.pi * xx), torch.cos(math.pi * xx)])[None]
        condition = condition + self.position(pos.to(condition.dtype))
        # Aligned features in queries and pooled multi-scale features in keys/values.
        global_features = self.multiscale(F.avg_pool2d(condition, (2, 4), ceil_mode=True))
        return fine, condition, global_features.flatten(2).transpose(1, 2)

    def codes(self, probabilities):
        return self.code_scale * (probabilities - 1.0 / self.classes)

    def label_codes(self, labels):
        # Dataset labels are 1..K; 0 is ignored, never a semantic class.
        p = F.one_hot((labels - 1).clamp(0, self.classes - 1), self.classes).permute(0, 3, 1, 2).float()
        return self.codes(p) * labels.gt(0)[:, None]

    def corrupt(self, clean, t, noise):
        a = alpha_bar(t)[:, None, None, None]
        return a.sqrt() * clean + (1 - a).sqrt() * noise

    def denoise(self, z, t, condition, self_condition=None):
        fine, aligned, context = condition
        sc = torch.zeros_like(z) if self_condition is None else self_condition
        x = self.state(torch.cat([z, sc], 1)) + aligned
        te = self.time(time_embedding(t, aligned.shape[1]).to(x.dtype))
        for block in self.blocks:
            x = block(x, te, context)
        up = self.unpatch(x)
        # Fine measurement detail is gated by denoiser output, not a second classifier.
        detail = fine * torch.sigmoid(self.detail_gate(up))
        return self.decoder(torch.cat([up, detail], 1))

    @torch.no_grad()
    def sample(self, x, valid, steps=3, seed=0):
        if steps < 1:
            raise ValueError('steps must be positive')
        condition = self.encode(x, valid)
        shape = (x.shape[0], self.classes, *x.shape[-2:])
        generator = torch.Generator(device=x.device).manual_seed(seed)
        z = torch.randn(shape, generator=generator, device=x.device) * valid[:, None]
        if self.mode == 'deterministic':
            z.zero_()
        sc = None
        for i in range(steps):
            t = torch.full((x.shape[0],), 1 - i / steps, device=x.device)
            if self.mode == 'deterministic':
                t.zero_()
            logits = self.denoise(z, t, condition, sc)
            clean = self.codes(logits.float().softmax(1)) * valid[:, None]
            sc = clean
            if self.mode == 'deterministic':
                z = clean
            elif i + 1 < steps:
                a = alpha_bar(t)[:, None, None, None]
                an = alpha_bar(torch.full_like(t, 1 - (i + 1) / steps))[:, None, None, None]
                eps = (z - a.sqrt() * clean) / (1 - a).sqrt().clamp_min(1e-6)
                z = (an.sqrt() * clean + (1 - an).sqrt() * eps) * valid[:, None]
        return logits
