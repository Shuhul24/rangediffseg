"""
Range Embedding Module.

RangeFormer's REM is a shared per-point MLP applied to every range-image pixel.
Implemented as 1x1 convolutions, which is exactly a shared MLP over pixels while
keeping the [B, C, H, W] range-view layout intact -- the input is never flattened
into an unordered point set. Normalisation and activation follow the
`BasicConv2d` of the reference implementation (conv -> BatchNorm -> LeakyReLU).
"""

import torch.nn as nn


class _PointMLP(nn.Sequential):
    """1x1 conv + BN + LeakyReLU: a shared MLP over range-image pixels."""

    def __init__(self, cin, cout, relu=True):
        layers = [nn.Conv2d(cin, cout, kernel_size=1, bias=False),
                  nn.BatchNorm2d(cout)]
        if relu:
            layers.append(nn.LeakyReLU(inplace=True))
        super().__init__(*layers)


class REM(nn.Module):
    """[B, in_ch, H, W] -> [B, dims[-1], H, W], spatial resolution preserved."""

    def __init__(self, in_channels=5, dims=(64, 128, 128)):
        super().__init__()
        chans = [in_channels] + list(dims)
        self.net = nn.Sequential(*[
            _PointMLP(chans[i], chans[i + 1]) for i in range(len(dims))])
        self.out_channels = dims[-1]

    def forward(self, x):
        return self.net(x)
