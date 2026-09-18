"""
Two-stage lightweight decoder, plus the optional DiT-gated skip.

The decoder is fed only by the diffusion-refined latent S0_hat. There is no
clean C -> decoder path, so the final semantic loss must backpropagate through
the DiT; the diffusion bottleneck cannot be bypassed.

The single optional skip from F1 is gated by S0_hat rather than concatenated
raw, so even the high-resolution detail path is modulated by the diffusion
output.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvBNAct(nn.Sequential):
    def __init__(self, cin, cout, k=3):
        super().__init__(nn.Conv2d(cin, cout, k, padding=k // 2, bias=False),
                         nn.BatchNorm2d(cout), nn.LeakyReLU(inplace=True))


class UpStage(nn.Module):
    """Bilinear upsample x2 followed by a 3x3 conv."""

    def __init__(self, cin, cout):
        super().__init__()
        self.conv = _ConvBNAct(cin, cout, 3)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=False)
        return self.conv(x)


class GatedSkip(nn.Module):
    """
    gate = sigmoid(g(upsample(S0_hat)));  F1_gated = F1 * gate
    fused = Fusion([decoder_feature, F1_gated])
    """

    def __init__(self, dec_dim, skip_dim, latent_dim):
        super().__init__()
        self.gate = nn.Sequential(nn.Conv2d(latent_dim, skip_dim, 1), nn.Sigmoid())
        self.fuse = _ConvBNAct(dec_dim + skip_dim, dec_dim, 3)

    def forward(self, dec_feat, f1, s0_hat):
        s_up = F.interpolate(s0_hat, size=f1.shape[-2:], mode='bilinear', align_corners=False)
        f1_gated = f1 * self.gate(s_up)
        return self.fuse(torch.cat([dec_feat, f1_gated], dim=1))


class Decoder(nn.Module):
    """S0_hat [B, D, h, w] -> logits [B, n_cls, H, W] in exactly two up stages."""

    def __init__(self, latent_dim, dims=(128, 128), n_classes=20,
                 skip_dim=None, use_gated_skip=False):
        super().__init__()
        self.stage1 = UpStage(latent_dim, dims[0])
        self.stage2 = UpStage(dims[0], dims[1])
        self.use_gated_skip = use_gated_skip
        if use_gated_skip:
            assert skip_dim is not None, 'gated skip needs the F1 channel count'
            self.skip = GatedSkip(dims[1], skip_dim, latent_dim)
        self.head = nn.Conv2d(dims[1], n_classes, kernel_size=1)

    def forward(self, s0_hat, f1=None, out_hw=None):
        x = self.stage2(self.stage1(s0_hat))
        if self.use_gated_skip:
            assert f1 is not None, 'use_gated_skip is on but F1 was not provided'
            x = self.skip(x, f1, s0_hat)
        logits = self.head(x)
        if out_hw is not None and logits.shape[-2:] != tuple(out_hw):
            logits = F.interpolate(logits, size=tuple(out_hw),
                                   mode='bilinear', align_corners=False)
        return logits
