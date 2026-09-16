"""
Range-image VAE, following RangeLDM (Hu et al., ECCV 2024).

RangeLDM compresses range images with a Stable-Diffusion-style KL autoencoder
(`sgm.modules.diffusionmodules.model.Encoder/Decoder`) before running a
diffusion model in the latent space. Its config for KITTI360 is:

    z_channels 4, double_z, ch 64, ch_mult [1,2,4], num_res_blocks 2,
    attn_type none, act silu, circular True, DiagonalGaussianRegularizer

The number that matters for us is `z_channels: 4`. DiT-XL/2 was trained on
4-channel SD-VAE latents, and its `x_embedder` is Conv2d(4, 1152, k=2, s=2).
Our segmentation model currently throws that layer away, because a raw range
image is neither RGB nor a VAE latent. Put this autoencoder in front and the
trunk's input becomes a 4-channel latent again -- its native distribution --
which is also what makes a diffusion loss on the trunk well-posed.

Deviations from the reference config, and why:
  * in/out channels 3 rather than 2, carrying [range, intensity, mask].
    x/y/z are deterministic given range and the projection geometry, so they
    are not encoded; the explicit mask avoids overloading a sentinel value for
    empty pixels, which are ~80% of a range image.
  * No LPIPS or adversarial term. Both are perceptual losses designed for
    natural images; this first model is trained on reconstruction + KL so the
    feasibility question (can a 4x downsample reconstruct thin structure?) is
    answered without the extra machinery.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def circular_pad(x, pad):
    """Pad circularly along width (azimuth wraps) and by replication in height."""
    if pad == 0:
        return x
    x = F.pad(x, (pad, pad, 0, 0), mode='circular')
    return F.pad(x, (0, 0, pad, pad), mode='replicate')


class CircConv2d(nn.Conv2d):
    """3x3 conv whose horizontal padding wraps around the azimuth axis."""

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__(in_ch, out_ch, kernel_size, stride=stride, padding=0)
        self.pad_amt = padding

    def forward(self, x):
        return super().forward(circular_pad(x, self.pad_amt))


def normalize(in_channels):
    return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class ResnetBlock(nn.Module):
    def __init__(self, in_ch, out_ch=None, dropout=0.0):
        super().__init__()
        out_ch = in_ch if out_ch is None else out_ch
        self.in_channels, self.out_channels = in_ch, out_ch

        self.norm1 = normalize(in_ch)
        self.conv1 = CircConv2d(in_ch, out_ch)
        self.norm2 = normalize(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = CircConv2d(out_ch, out_ch)
        self.nin_shortcut = (nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else None)

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        if self.nin_shortcut is not None:
            x = self.nin_shortcut(x)
        return x + h


class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=0)

    def forward(self, x):
        x = F.pad(x, (0, 1, 0, 0), mode='circular')
        x = F.pad(x, (0, 0, 0, 1), mode='replicate')
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = CircConv2d(ch, ch)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2.0, mode='nearest'))


class Encoder(nn.Module):
    def __init__(self, in_channels=3, ch=64, ch_mult=(1, 2, 4), num_res_blocks=2,
                 z_channels=4, double_z=True, dropout=0.0):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks

        self.conv_in = CircConv2d(in_channels, ch)
        in_ch_mult = (1,) + tuple(ch_mult)

        self.down = nn.ModuleList()
        for i in range(self.num_resolutions):
            block = nn.ModuleList()
            block_in, block_out = ch * in_ch_mult[i], ch * ch_mult[i]
            for _ in range(num_res_blocks):
                block.append(ResnetBlock(block_in, block_out, dropout))
                block_in = block_out
            stage = nn.Module()
            stage.block = block
            stage.downsample = Downsample(block_in) if i != self.num_resolutions - 1 else None
            self.down.append(stage)

        self.mid_block_1 = ResnetBlock(block_in, block_in, dropout)
        self.mid_block_2 = ResnetBlock(block_in, block_in, dropout)
        self.norm_out = normalize(block_in)
        self.conv_out = CircConv2d(block_in, 2 * z_channels if double_z else z_channels)

    def forward(self, x):
        h = self.conv_in(x)
        for stage in self.down:
            for blk in stage.block:
                h = blk(h)
            if stage.downsample is not None:
                h = stage.downsample(h)
        h = self.mid_block_2(self.mid_block_1(h))
        return self.conv_out(F.silu(self.norm_out(h)))


class Decoder(nn.Module):
    def __init__(self, out_ch=3, ch=64, ch_mult=(1, 2, 4), num_res_blocks=2,
                 z_channels=4, dropout=0.0):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        block_in = ch * ch_mult[-1]

        self.conv_in = CircConv2d(z_channels, block_in)
        self.mid_block_1 = ResnetBlock(block_in, block_in, dropout)
        self.mid_block_2 = ResnetBlock(block_in, block_in, dropout)

        self.up = nn.ModuleList()
        for i in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            block_out = ch * ch_mult[i]
            for _ in range(num_res_blocks + 1):
                block.append(ResnetBlock(block_in, block_out, dropout))
                block_in = block_out
            stage = nn.Module()
            stage.block = block
            stage.upsample = Upsample(block_in) if i != 0 else None
            self.up.insert(0, stage)

        self.norm_out = normalize(block_in)
        self.conv_out = CircConv2d(block_in, out_ch)

    def forward(self, z):
        h = self.mid_block_2(self.mid_block_1(self.conv_in(z)))
        for stage in reversed(self.up):
            for blk in stage.block:
                h = blk(h)
            if stage.upsample is not None:
                h = stage.upsample(h)
        return self.conv_out(F.silu(self.norm_out(h)))


class DiagonalGaussian:
    """The DiagonalGaussianRegularizer of the reference config."""

    def __init__(self, parameters):
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.std = torch.exp(0.5 * self.logvar)

    def sample(self):
        return self.mean + self.std * torch.randn_like(self.mean)

    def kl(self):
        return 0.5 * torch.sum(self.mean.pow(2) + self.logvar.exp() - 1.0 - self.logvar,
                               dim=[1, 2, 3]).mean()


class RangeVAE(nn.Module):
    """Range-image KL autoencoder. 4x spatial downsampling, 4-channel latent."""

    def __init__(self, in_channels=3, out_ch=3, ch=64, ch_mult=(1, 2, 4),
                 num_res_blocks=2, z_channels=4, dropout=0.0):
        super().__init__()
        self.z_channels = z_channels
        self.downsample_factor = 2 ** (len(ch_mult) - 1)
        self.encoder = Encoder(in_channels, ch, ch_mult, num_res_blocks,
                               z_channels, double_z=True, dropout=dropout)
        self.decoder = Decoder(out_ch, ch, ch_mult, num_res_blocks, z_channels, dropout)
        # 1x1 convs between the encoder/decoder and the latent, as in SD's VAE.
        self.quant_conv = nn.Conv2d(2 * z_channels, 2 * z_channels, 1)
        self.post_quant_conv = nn.Conv2d(z_channels, z_channels, 1)

    def encode(self, x):
        return DiagonalGaussian(self.quant_conv(self.encoder(x)))

    def decode(self, z):
        return self.decoder(self.post_quant_conv(z))

    def forward(self, x, sample_posterior=True):
        posterior = self.encode(x)
        z = posterior.sample() if sample_posterior else posterior.mean
        return self.decode(z), posterior
