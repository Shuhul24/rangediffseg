"""
Range-image stems: patchification + patch encoding.

The convolutional stem is the one introduced by RangeViT
(github.com/valeoai/rangevit, Apache-2.0), itself built from the SalsaNext
residual blocks (github.com/TiagoCortinhal/SalsaNext, MIT). It replaces the
2x2 latent patch embedding of DiT (`x_embedder`), which cannot be reused
because range images are neither 3-channel RGB nor VAE latents, and because we
patchify with a strongly anisotropic 2x8 patch.
"""

import torch
import torch.nn as nn

from .model_utils import get_grid_size_1d, get_grid_size_2d


class PatchEmbedding(nn.Module):
    """Plain linear patch embedding (used when `conv_stem: none`)."""

    def __init__(self, image_size, patch_size, patch_stride, embed_dim, channels):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        if patch_stride is None:
            patch_stride = patch_size
        elif isinstance(patch_stride, int):
            patch_stride = (patch_stride, patch_stride)
        patch_size, patch_stride = tuple(patch_size), tuple(patch_stride)

        if image_size[0] % patch_size[0] != 0 or image_size[1] % patch_size[1] != 0:
            raise ValueError('image dimensions must be divisible by the patch size')

        self.image_size = image_size
        self.grid_size = (get_grid_size_1d(image_size[0], patch_size[0], patch_stride[0]),
                          get_grid_size_1d(image_size[1], patch_size[1], patch_stride[1]))
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.proj = nn.Conv2d(channels, embed_dim, kernel_size=patch_size, stride=patch_stride)

    def get_grid_size(self, H, W):
        return get_grid_size_2d(H, W, self.patch_size, self.patch_stride)

    def forward(self, im):
        x = self.proj(im).flatten(2).transpose(1, 2)   # (B, N, D)
        return x, None


class ConvStem(nn.Module):
    """
    Convolutional stem: three residual context blocks at full range-image
    resolution, then average pooling + 1x1 convolution down to the token grid.

    Returns both the token sequence and the full-resolution feature map, which
    the up-conv decoder consumes as a skip connection.
    """

    def __init__(self,
                 in_channels=5,
                 base_channels=32,
                 img_size=(64, 384),
                 patch_stride=(2, 8),
                 embed_dim=1152,
                 flatten=True,
                 hidden_dim=None):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = 2 * base_channels

        self.base_channels = base_channels
        self.hidden_dim = hidden_dim
        self.dropout_ratio = 0.2

        self.conv_block = nn.Sequential(
            ResContextBlock(in_channels, base_channels),
            ResContextBlock(base_channels, base_channels),
            ResContextBlock(base_channels, base_channels),
            ResBlock(base_channels, hidden_dim, self.dropout_ratio, pooling=False, drop_out=False))

        assert patch_stride[0] % 2 == 0
        assert patch_stride[1] % 2 == 0
        kernel_size = (patch_stride[0] + 1, patch_stride[1] + 1)
        padding = (patch_stride[0] // 2, patch_stride[1] // 2)
        self.proj_block = nn.Sequential(
            nn.AvgPool2d(kernel_size=kernel_size, stride=tuple(patch_stride), padding=padding),
            nn.Conv2d(hidden_dim, embed_dim, kernel_size=1))

        self.patch_stride = tuple(patch_stride)
        self.patch_size = tuple(patch_stride)
        self.grid_size = (img_size[0] // patch_stride[0], img_size[1] // patch_stride[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

    def get_grid_size(self, H, W):
        return get_grid_size_2d(H, W, self.patch_size, self.patch_stride)

    def forward(self, x):
        x_base = self.conv_block(x)          # (B, hidden_dim, H, W)
        x = self.proj_block(x_base)          # (B, embed_dim, H/ps_h, W/ps_w)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        return x, x_base


class ResContextBlock(nn.Module):
    # From T. Cortinhal et al. -- github.com/TiagoCortinhal/SalsaNext
    def __init__(self, in_filters, out_filters):
        super().__init__()
        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=(1, 1), stride=1)
        self.act1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(out_filters, out_filters, (3, 3), padding=1)
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)

        self.conv3 = nn.Conv2d(out_filters, out_filters, (3, 3), dilation=2, padding=2)
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(out_filters)

    def forward(self, x):
        shortcut = self.act1(self.conv1(x))
        resA = self.bn1(self.act2(self.conv2(shortcut)))
        resA = self.bn2(self.act3(self.conv3(resA)))
        return shortcut + resA


class ResBlock(nn.Module):
    # From T. Cortinhal et al. -- github.com/TiagoCortinhal/SalsaNext
    def __init__(self, in_filters, out_filters, dropout_rate, kernel_size=(3, 3), stride=1,
                 pooling=True, drop_out=True):
        super().__init__()
        self.pooling = pooling
        self.drop_out = drop_out

        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=(1, 1), stride=stride)
        self.act1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(in_filters, out_filters, kernel_size=(3, 3), padding=1)
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)

        self.conv3 = nn.Conv2d(out_filters, out_filters, kernel_size=(3, 3), dilation=2, padding=2)
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(out_filters)

        self.conv4 = nn.Conv2d(out_filters, out_filters, kernel_size=(2, 2), dilation=2, padding=1)
        self.act4 = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm2d(out_filters)

        self.conv5 = nn.Conv2d(out_filters * 3, out_filters, kernel_size=(1, 1))
        self.act5 = nn.LeakyReLU()
        self.bn4 = nn.BatchNorm2d(out_filters)

        self.dropout = nn.Dropout2d(p=dropout_rate)
        if pooling:
            self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=2, padding=1)

    def forward(self, x):
        shortcut = self.act1(self.conv1(x))

        resA1 = self.bn1(self.act2(self.conv2(x)))
        resA2 = self.bn2(self.act3(self.conv3(resA1)))
        resA3 = self.bn3(self.act4(self.conv4(resA2)))

        concat = torch.cat((resA1, resA2, resA3), dim=1)
        resA = self.bn4(self.act5(self.conv5(concat)))
        resA = shortcut + resA

        resB = self.dropout(resA) if self.drop_out else resA
        if self.pooling:
            return self.pool(resB), resA
        return resB
