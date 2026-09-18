"""
Compact conditional Diffusion Transformer over the semantic latent.

Each block is

    adaLN(timestep) -> self-attention on S_t
    adaLN(timestep) -> cross-attention, S_t queries / C keys+values
    adaLN(timestep) -> MLP

with zero-initialised adaLN gates (DiT's adaLN-Zero), so the block starts as the
identity and the model begins exactly where a no-DiT model would.

The RangeFormer condition C enters through cross-attention rather than being
concatenated onto S_t, so the conditioning path is explicit and cannot be
ignored by the network. Keys/values are spatially reduced by `cond_sr_ratio`
to keep the cross-attention affordable on a wide range-view grid.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def timestep_embedding(t, dim, max_period=10000):
    """Sinusoidal timestep embedding, as in DDPM/DiT."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) *
                      torch.arange(half, dtype=torch.float32, device=t.device) / half)
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden, freq_dim=256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(nn.Linear(freq_dim, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden))

    def forward(self, t):
        return self.mlp(timestep_embedding(t, self.freq_dim))


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class _MHA(nn.Module):
    """Multi-head attention; `kv` None means self-attention."""

    def __init__(self, dim, num_heads, dropout=0.0, kv_dim=None):
        super().__init__()
        assert dim % num_heads == 0
        self.h, self.hd = num_heads, dim // num_heads
        self.scale = self.hd ** -0.5
        kv_dim = kv_dim or dim
        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(kv_dim, dim * 2, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.drop = dropout

    def forward(self, x, kv=None):
        B, N, C = x.shape
        src = x if kv is None else kv
        q = self.q(x).reshape(B, N, self.h, self.hd).transpose(1, 2)
        k, v = self.kv(src).reshape(B, src.shape[1], 2, self.h, self.hd).permute(2, 0, 3, 1, 4)
        o = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.drop if self.training else 0.0, scale=self.scale)
        return self.proj(o.transpose(1, 2).reshape(B, N, C))


class DiTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0, cond_dim=None,
                 timestep_conditioning=True):
        super().__init__()
        self.timestep_conditioning = timestep_conditioning
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = _MHA(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.cross_attn = _MHA(dim, num_heads, dropout, kv_dim=cond_dim or dim)
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(approximate='tanh'),
                                 nn.Dropout(dropout), nn.Linear(hidden, dim))
        # 9 = (shift, scale, gate) x (self-attn, cross-attn, mlp)
        if timestep_conditioning:
            self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 9 * dim, bias=True))
            nn.init.constant_(self.adaLN[-1].weight, 0)
            nn.init.constant_(self.adaLN[-1].bias, 0)
        else:
            # Ablation B: identical block, modulation independent of t.
            self.mod = nn.Parameter(torch.zeros(9 * dim))

    def forward(self, x, c_emb, cond_tokens):
        mod = (self.adaLN(c_emb) if self.timestep_conditioning
               else self.mod.unsqueeze(0).expand(x.shape[0], -1))
        (sh1, sc1, g1, sh2, sc2, g2, sh3, sc3, g3) = mod.chunk(9, dim=1)
        x = x + g1.unsqueeze(1) * self.self_attn(modulate(self.norm1(x), sh1, sc1))
        x = x + g2.unsqueeze(1) * self.cross_attn(modulate(self.norm2(x), sh2, sc2), cond_tokens)
        return x + g3.unsqueeze(1) * self.mlp(modulate(self.norm3(x), sh3, sc3))


class ConditionalDiT(nn.Module):
    """
    Predicts epsilon on a semantic latent [B, D, h, w] given timestep t and the
    RangeFormer condition C [B, Cc, h, w].

    `use_diffusion=False` (ablation C) keeps the identical module and parameter
    count but is driven with t=0 and no noise by the caller, so B/C/D stay
    comparable in capacity.
    """

    def __init__(self, dim=256, depth=3, num_heads=8, mlp_ratio=4.0, dropout=0.0,
                 cond_dim=256, cond_sr_ratio=2, grad_checkpointing=False,
                 timestep_conditioning=True):
        super().__init__()
        self.dim = dim
        self.timestep_conditioning = timestep_conditioning
        self.grad_checkpointing = grad_checkpointing
        self.t_embedder = TimestepEmbedder(dim)

        self.cond_sr_ratio = cond_sr_ratio
        if cond_sr_ratio > 1:
            self.cond_sr = nn.Conv2d(cond_dim, cond_dim, cond_sr_ratio, cond_sr_ratio)
        self.cond_norm = nn.LayerNorm(cond_dim)

        self.pos_embed = nn.Parameter(torch.zeros(1, 1, dim))  # broadcast, resized below
        self.in_proj = nn.Conv2d(dim, dim, 1)
        self.blocks = nn.ModuleList([
            DiTBlock(dim, num_heads, mlp_ratio, dropout, cond_dim,
                     timestep_conditioning) for _ in range(depth)])
        self.norm_out = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_out = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim, bias=True))
        self.out_proj = nn.Linear(dim, dim)
        nn.init.constant_(self.adaLN_out[-1].weight, 0)
        nn.init.constant_(self.adaLN_out[-1].bias, 0)
        if not timestep_conditioning:
            self.mod_out = nn.Parameter(torch.zeros(2 * dim))
        # eps-prediction head starts at zero, as DiT does for its output layer.
        nn.init.constant_(self.out_proj.weight, 0)
        nn.init.constant_(self.out_proj.bias, 0)

    def _cond_tokens(self, cond):
        if self.cond_sr_ratio > 1:
            cond = self.cond_sr(cond)
        B, C, h, w = cond.shape
        return self.cond_norm(cond.flatten(2).transpose(1, 2))

    def forward(self, s_t, t, cond):
        """s_t: [B, D, h, w], t: [B], cond: [B, Cc, h, w] -> eps_hat [B, D, h, w]"""
        B, D, h, w = s_t.shape
        x = self.in_proj(s_t).flatten(2).transpose(1, 2)          # [B, h*w, D]
        c_emb = self.t_embedder(t)
        cond_tokens = self._cond_tokens(cond)

        for blk in self.blocks:
            if self.grad_checkpointing and self.training:
                x = checkpoint(blk, x, c_emb, cond_tokens, use_reentrant=False)
            else:
                x = blk(x, c_emb, cond_tokens)

        mod_out = (self.adaLN_out(c_emb) if self.timestep_conditioning
                   else self.mod_out.unsqueeze(0).expand(B, -1))
        shift, scale = mod_out.chunk(2, dim=1)
        x = self.out_proj(modulate(self.norm_out(x), shift, scale))
        return x.transpose(1, 2).reshape(B, D, h, w)
