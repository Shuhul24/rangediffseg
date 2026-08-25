"""
Segmentation decoders that turn DiT patch tokens back into a full-resolution
range-view label map.

Both decoders come from RangeViT (github.com/valeoai/rangevit, Apache-2.0);
the linear one originates from Segmenter (github.com/rstrudel/segmenter). They
replace `final_layer.linear` of DiT, which projects tokens back to noise/latent
patches. `einops` is inlined here as plain tensor ops to keep the dependency
list short.
"""

import torch
import torch.nn as nn

from .model_utils import get_grid_size_2d, init_weights


def tokens_to_map(x, grid_h, grid_w):
    """(B, grid_h * grid_w, D) -> (B, D, grid_h, grid_w)."""
    B, N, D = x.shape
    assert N == grid_h * grid_w, f'{N} tokens do not fit a {grid_h}x{grid_w} grid'
    return x.view(B, grid_h, grid_w, D).permute(0, 3, 1, 2).contiguous()


class AnisotropicPixelShuffle(nn.Module):
    """
    Rearrange 'b (c s0 s1) h w -> b c (h s0) (w s1)', i.e. a pixel shuffle with
    a different upsampling factor along height and width (2x8 for range views).
    """

    def __init__(self, scale_factor):
        super().__init__()
        self.s0, self.s1 = scale_factor

    def forward(self, x):
        B, C, H, W = x.shape
        c = C // (self.s0 * self.s1)
        x = x.view(B, c, self.s0, self.s1, H, W)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        return x.view(B, c, H * self.s0, W * self.s1)


class DecoderLinear(nn.Module):
    """Per-token linear classifier (from Segmenter)."""

    def __init__(self, n_cls, patch_size, d_encoder, patch_stride=None):
        super().__init__()
        self.d_encoder = d_encoder
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.n_cls = n_cls

        self.head = nn.Linear(self.d_encoder, n_cls)
        self.apply(init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return set()

    def forward(self, x, im_size, skip=None):
        H, W = im_size
        GS_H, GS_W = get_grid_size_2d(H, W, self.patch_size, self.patch_stride)
        x = self.head(x)
        return tokens_to_map(x, GS_H, GS_W)


class UpConvBlock(nn.Module):
    """One anisotropic upsampling stage with an optional stem skip connection."""

    def __init__(self, in_filters, out_filters, dropout_rate, scale_factor=(2, 8),
                 drop_out=False, skip_filters=0):
        super().__init__()
        self.in_filters = in_filters
        self.out_filters = out_filters
        self.skip_filters = skip_filters

        if isinstance(scale_factor, int):
            scale_factor = (scale_factor, scale_factor)
        assert len(scale_factor) == 2
        self.scale_factor = tuple(scale_factor)

        upsample_layers = [
            nn.Conv2d(in_filters, out_filters * self.scale_factor[0] * self.scale_factor[1],
                      kernel_size=(1, 1)),
            AnisotropicPixelShuffle(self.scale_factor),
        ]
        if drop_out:
            upsample_layers.append(nn.Dropout2d(p=dropout_rate))
        self.conv_upsample = nn.Sequential(*upsample_layers)

        self.conv1 = nn.Sequential(
            nn.Conv2d(out_filters + skip_filters, out_filters, (3, 3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(out_filters))

        output_layers = [
            nn.Conv2d(out_filters, out_filters, kernel_size=(1, 1)),
            nn.LeakyReLU(),
            nn.BatchNorm2d(out_filters),
        ]
        if drop_out:
            output_layers.append(nn.Dropout2d(p=dropout_rate))
        self.conv_output = nn.Sequential(*output_layers)

    def forward(self, x, skip=None):
        x_up = self.conv_upsample(x)

        if self.skip_filters > 0:
            assert skip is not None, 'the stem skip connection is missing'
            assert skip.shape[1] == self.skip_filters
            x_up = torch.cat((x_up, skip), dim=1)

        return self.conv_output(self.conv1(x_up))


class DecoderUpConv(nn.Module):
    """Convolutional decoder: token grid -> full-resolution class logits."""

    def __init__(self, n_cls, patch_size, d_encoder, d_decoder, scale_factor=(2, 8),
                 patch_stride=None, dropout_rate=0.2, drop_out=False, skip_filters=0):
        super().__init__()
        self.d_encoder = d_encoder
        self.d_decoder = d_decoder
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.n_cls = n_cls

        self.up_conv_block = UpConvBlock(
            d_encoder, d_decoder,
            dropout_rate=dropout_rate,
            scale_factor=scale_factor,
            drop_out=drop_out,
            skip_filters=skip_filters)

        self.head = nn.Conv2d(d_decoder, n_cls, kernel_size=(1, 1))
        self.apply(init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return set()

    def forward(self, x, im_size, skip=None, return_features=False):
        H, W = im_size
        GS_H, GS_W = get_grid_size_2d(H, W, self.patch_size, self.patch_stride)
        x = tokens_to_map(x, GS_H, GS_W)
        x = self.up_conv_block(x, skip)
        return x if return_features else self.head(x)
