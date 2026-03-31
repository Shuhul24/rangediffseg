"""
Timestep-conditioned U-Net backbone for DDPM range-view segmentation.
Adapted from SegDiff (github.com/tomeramit/SegDiff).

Input : cat([noisy_seg (num_classes, H, W), range_img (img_ch, H, W)])
Output: pred_x0 (num_classes, H, W)  — predicted clean segmentation in [-1,1]

Architecture for default ch_mults=(1,2,4,4), base_ch=64, proj 64×512:
  Encoder: 64 → 128 → 256 → 256  (3 stride-2 downsamples)
  Bottleneck: 256 ch, spatial 8×64, with self-attention
  Decoder: mirrors encoder with skip connections + bilinear upsamples
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .nn import timestep_embedding, normalization, zero_module


class ResBlock(nn.Module):
    """Residual block with scale-shift timestep conditioning."""

    def __init__(self, in_ch, out_ch, t_dim, dropout=0.0):
        super().__init__()
        self.norm1  = normalization(in_ch)
        self.conv1  = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.t_proj = nn.Linear(t_dim, out_ch * 2)          # → scale, shift
        self.norm2  = normalization(out_ch)
        self.drop   = nn.Dropout(dropout)
        self.conv2  = zero_module(nn.Conv2d(out_ch, out_ch, 3, padding=1))
        self.skip   = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, te):
        h               = self.conv1(F.silu(self.norm1(x)))
        scale, shift    = self.t_proj(F.silu(te)).chunk(2, dim=1)
        h               = self.norm2(h) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h               = self.conv2(self.drop(F.silu(h)))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """Multi-head self-attention (applied at the bottleneck)."""

    def __init__(self, ch, num_heads=8):
        super().__init__()
        assert ch % num_heads == 0
        self.norm    = normalization(ch)
        self.qkv     = nn.Conv1d(ch, ch * 3, 1)
        self.proj    = zero_module(nn.Conv1d(ch, ch, 1))
        self.heads   = num_heads
        self.head_ch = ch // num_heads

    def forward(self, x):
        B, C, H, W = x.shape
        h   = self.norm(x).view(B, C, -1)                    # (B, C, N)
        qkv = self.qkv(h).view(B, self.heads, self.head_ch * 3, -1)
        q, k, v = qkv.chunk(3, dim=2)                        # (B, heads, head_ch, N)
        scale = self.head_ch ** -0.5
        attn  = torch.softmax(torch.einsum('bhcn,bhcm->bhnm', q, k) * scale, dim=-1)
        h     = torch.einsum('bhnm,bhcm->bhcn', attn, v).reshape(B, C, -1)
        return x + self.proj(h).view(B, C, H, W)


class UNet(nn.Module):
    def __init__(self, img_channels=5, num_classes=20, base_ch=64,
                 ch_mults=(1, 2, 4, 4), num_res_blocks=2,
                 use_mid_attn=True, dropout=0.1):
        """
        Args:
            img_channels  : channels in the conditioning range image (5)
            num_classes   : segmentation classes, also output channels (20)
            base_ch       : channel width at the first encoder level (must be ÷32)
            ch_mults      : channel multipliers per encoder level
            num_res_blocks: residual blocks per level (decoder has +1 for skips)
            use_mid_attn  : self-attention at the bottleneck
            dropout       : dropout inside residual blocks
        """
        super().__init__()
        self.base_ch = base_ch
        t_dim  = base_ch * 4
        in_ch  = img_channels + num_classes

        # Timestep MLP: sinusoidal → base_ch → t_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(base_ch, t_dim), nn.SiLU(),
            nn.Linear(t_dim, t_dim),
        )

        self.in_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        # ---- Encoder ----
        # enc       : list of modules (ResBlock or stride-2 Conv)
        # enc_is_res: parallel bool list
        # skip_ch   : channel count at each skip output (used to size decoder)
        self.enc        = nn.ModuleList()
        self.enc_is_res = []
        skip_ch         = [base_ch]          # from in_conv
        ch              = base_ch

        for lvl, m in enumerate(ch_mults):
            out = base_ch * m
            for _ in range(num_res_blocks):
                self.enc.append(ResBlock(ch, out, t_dim, dropout))
                self.enc_is_res.append(True)
                ch = out; skip_ch.append(ch)
            if lvl < len(ch_mults) - 1:                        # downsample between levels
                self.enc.append(nn.Conv2d(ch, ch, 4, stride=2, padding=1))
                self.enc_is_res.append(False)
                skip_ch.append(ch)

        # ---- Bottleneck ----
        self.mid_res1 = ResBlock(ch, ch, t_dim, dropout)
        self.mid_attn = AttentionBlock(ch) if use_mid_attn else nn.Identity()
        self.mid_res2 = ResBlock(ch, ch, t_dim, dropout)

        # ---- Decoder (mirrors encoder, consumes skip_ch in reverse) ----
        self.dec        = nn.ModuleList()
        self.dec_is_res = []

        for lvl, m in reversed(list(enumerate(ch_mults))):
            out = base_ch * m
            for _ in range(num_res_blocks + 1):    # +1 to absorb one extra skip
                sc = skip_ch.pop()
                self.dec.append(ResBlock(ch + sc, out, t_dim, dropout))
                self.dec_is_res.append(True)
                ch = out
            if lvl > 0:                            # upsample between levels
                self.dec.append(nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(ch, ch, 3, padding=1),
                ))
                self.dec_is_res.append(False)

        self.out = nn.Sequential(
            normalization(ch), nn.SiLU(),
            zero_module(nn.Conv2d(ch, num_classes, 3, padding=1)),
            nn.Tanh(),          # output in [-1, 1] (matches one-hot target range)
        )

    def forward(self, x_t, t, cond):
        """
        Args:
            x_t  : (B, num_classes, H, W) noisy segmentation map
            t    : (B,) diffusion timestep indices
            cond : (B, img_channels, H, W) range-view image features
        Returns:
            (B, num_classes, H, W) predicted clean segmentation (x_0)
        """
        te = self.time_mlp(timestep_embedding(t, self.base_ch))
        h  = self.in_conv(torch.cat([x_t, cond], dim=1))

        skips = [h]
        for mod, is_res in zip(self.enc, self.enc_is_res):
            h = mod(h, te) if is_res else mod(h)
            skips.append(h)

        h = self.mid_res1(h, te)
        h = self.mid_attn(h)
        h = self.mid_res2(h, te)

        for mod, is_res in zip(self.dec, self.dec_is_res):
            if is_res:
                h = mod(torch.cat([h, skips.pop()], dim=1), te)
            else:
                h = mod(h)

        return self.out(h)
