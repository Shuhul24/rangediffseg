"""
Two-stage RangeFormer-style range-view encoder.

Adapted from the MixVisionTransformer of
github.com/nschi00/rangeview-rgb-lidar-fusion (modules/network/mix_transformer.py),
keeping only the LiDAR range-view pieces: overlapping patch embedding,
spatial-reduction attention (SRA) and the depthwise-conv FFN. Nothing from the
RGB, Mask2Former or fusion paths is imported.

Two deviations from the reference, both deliberate:
  * only two stages are kept (the reference builds four);
  * `circular` padding along width. A range image wraps in azimuth, and the
    reference does not model that. It is off by default so the encoder matches
    the reference exactly unless asked otherwise.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _trunc_normal_(t, std=0.02):
    nn.init.trunc_normal_(t, std=std, a=-2 * std, b=2 * std)


class DWConv(nn.Module):
    """Depthwise conv inside the FFN, as in the reference Mlp."""

    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        return self.dwconv(x).flatten(2).transpose(1, 2)


class Mlp(nn.Module):
    def __init__(self, dim, hidden, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.dwconv = DWConv(hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.drop(self.act(self.dwconv(self.fc1(x), H, W)))
        return self.drop(self.fc2(x))


class SRAttention(nn.Module):
    """Spatial-reduction self-attention: keys/values are pooled by `sr_ratio`."""

    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0.0,
                 proj_drop=0.0, sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f'dim {dim} not divisible by {num_heads} heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
        else:
            x_ = x
        kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0, scale=self.scale)
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(out))


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.0, attn_drop=0.0, sr_ratio=1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SRAttention(dim, num_heads, attn_drop=attn_drop,
                                proj_drop=drop, sr_ratio=sr_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x, H, W):
        x = x + self.attn(self.norm1(x), H, W)
        return x + self.mlp(self.norm2(x), H, W)


class OverlapPatchEmbed(nn.Module):
    """
    Overlapping patch embedding (reference: OverlapPatchEmbed, kernel 3 stride 2).

    With `circular=True` the horizontal padding wraps, so azimuth continuity is
    preserved; vertical padding stays zero because scan lines do not wrap.
    """

    def __init__(self, in_chans, embed_dim, patch_size=3, stride=2, circular=False):
        super().__init__()
        self.pad = patch_size // 2
        self.circular = circular
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=stride, padding=0 if circular else self.pad)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        if self.circular and self.pad > 0:
            x = F.pad(x, (self.pad, self.pad, 0, 0), mode='circular')
            x = F.pad(x, (0, 0, self.pad, self.pad), mode='constant', value=0.0)
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        return self.norm(x), H, W


class EncoderStage(nn.Module):
    """Overlapping patch embed (stride 2) followed by `depth` transformer blocks."""

    def __init__(self, in_chans, dim, depth, num_heads, sr_ratio,
                 mlp_ratio=4.0, drop=0.0, attn_drop=0.0, circular=False):
        super().__init__()
        self.patch_embed = OverlapPatchEmbed(in_chans, dim, 3, 2, circular)
        self.blocks = nn.ModuleList([
            Block(dim, num_heads, mlp_ratio, drop, attn_drop, sr_ratio)
            for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.out_channels = dim
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            _trunc_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels // m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        """[B, Cin, H, W] -> [B, dim, H/2, W/2]"""
        tok, H, W = self.patch_embed(x)
        for blk in self.blocks:
            tok = blk(tok, H, W)
        tok = self.norm(tok)
        B, _, C = tok.shape
        return tok.transpose(1, 2).reshape(B, C, H, W)
