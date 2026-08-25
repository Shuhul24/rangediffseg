"""
Helper functions shared by the stem, the DiT encoder and the decoders.

Adapted from RangeViT (github.com/valeoai/rangevit, Apache-2.0) and from the
sin-cos positional-embedding utilities of DiT / MAE
(github.com/facebookresearch/DiT, github.com/facebookresearch/mae).
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#                              Grid size helpers                              #
# --------------------------------------------------------------------------- #

def get_grid_size_1d(length, patch_size, stride):
    assert patch_size % stride == 0
    assert length % patch_size == 0
    return (length - patch_size) // stride + 1


def get_grid_size_2d(H, W, patch_size, patch_stride):
    """Number of tokens along (height, width) for an H x W image."""
    if isinstance(patch_size, int):
        PS_H = PS_W = patch_size
    else:
        PS_H, PS_W = patch_size

    if patch_stride is not None:
        if isinstance(patch_stride, int):
            patch_stride = (patch_stride, patch_stride)
        H_stride, W_stride = patch_stride
    else:
        H_stride, W_stride = PS_H, PS_W

    return (get_grid_size_1d(H, PS_H, H_stride),
            get_grid_size_1d(W, PS_W, W_stride))


# --------------------------------------------------------------------------- #
#                           Positional embeddings                             #
# --------------------------------------------------------------------------- #

def resize_pos_embed(posemb, grid_old_shape, grid_new_shape, num_extra_tokens=0):
    """
    Bilinearly resize a grid of positional embeddings.

    DiT has no class token, so `num_extra_tokens` defaults to 0 (RangeViT uses
    1 because the ViT of Segmenter keeps a CLS token).

    Args:
        posemb          : (1, num_extra_tokens + gs_old_h * gs_old_w, D)
        grid_old_shape  : (gs_old_h, gs_old_w) or None to assume a square grid
        grid_new_shape  : (gs_h, gs_w) target token grid
    Returns:
        (1, num_extra_tokens + gs_h * gs_w, D)
    """
    posemb_tok, posemb_grid = (posemb[:, :num_extra_tokens],
                               posemb[0, num_extra_tokens:])

    if grid_old_shape is None:
        gs_old_h = int(math.sqrt(len(posemb_grid)))
        gs_old_w = gs_old_h
    else:
        gs_old_h, gs_old_w = grid_old_shape
    assert gs_old_h * gs_old_w == len(posemb_grid)

    gs_h, gs_w = grid_new_shape
    posemb_grid = posemb_grid.reshape(1, gs_old_h, gs_old_w, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(gs_h, gs_w), mode='bilinear', align_corners=False)
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_h * gs_w, -1)

    return torch.cat([posemb_tok, posemb_grid], dim=1)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """embed_dim: output dim per position. pos: (M,) positions. Returns (M, D)."""
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega                       # (D/2,)

    pos = pos.reshape(-1)                             # (M,)
    out = np.einsum('m,d->md', pos, omega)            # (M, D/2)

    return np.concatenate([np.sin(out), np.cos(out)], axis=1)  # (M, D)


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    """
    2-D sin-cos positional embeddings, as used by DiT/MAE, but extended to
    rectangular token grids (range images are far wider than they are tall).

    Args:
        embed_dim : embedding dimension
        grid_size : int for a square grid, or (grid_h, grid_w)
    Returns:
        (grid_h * grid_w, embed_dim) numpy array
    """
    if isinstance(grid_size, int):
        grid_h_size = grid_w_size = grid_size
    else:
        grid_h_size, grid_w_size = grid_size

    grid_h = np.arange(grid_h_size, dtype=np.float32)
    grid_w = np.arange(grid_w_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)                # w goes first
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_h_size, grid_w_size])

    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])

    return np.concatenate([emb_h, emb_w], axis=1)     # (H*W, D)


# --------------------------------------------------------------------------- #
#                        Weight init / padding helpers                        #
# --------------------------------------------------------------------------- #

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        if m.elementwise_affine:
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


def adapt_input_conv(in_chans, conv_weight):
    """
    Re-purpose a pre-trained patch-embedding kernel for a different number of
    input channels (the DiT checkpoint expects 4 latent channels, range images
    have 5). From RangeViT / timm.
    """
    conv_type = conv_weight.dtype
    conv_weight = conv_weight.float()
    O, I, J, K = conv_weight.shape

    if in_chans == I:
        return conv_weight.to(conv_type)

    if in_chans == 1:
        conv_weight = conv_weight.sum(dim=1, keepdim=True)
    else:
        repeat = int(math.ceil(in_chans / I))
        conv_weight = conv_weight.repeat(1, repeat, 1, 1)[:, :in_chans, :, :]
        conv_weight *= (I / float(in_chans))

    return conv_weight.to(conv_type)


def padding(im, patch_size, fill_value=0):
    """Pad an image so that both sides are divisible by the patch size."""
    H, W = im.size(2), im.size(3)
    if isinstance(patch_size, int):
        patch_size_H = patch_size_W = patch_size
    else:
        patch_size_H, patch_size_W = patch_size

    pad_h = (patch_size_H - H % patch_size_H) % patch_size_H
    pad_w = (patch_size_W - W % patch_size_W) % patch_size_W
    if pad_h > 0 or pad_w > 0:
        im = F.pad(im, (0, pad_w, 0, pad_h), value=fill_value)
    return im


def unpadding(y, target_size):
    """Crop away the extra pixels introduced by `padding`."""
    H, W = target_size
    extra_h = y.size(2) - H
    extra_w = y.size(3) - W
    if extra_h > 0:
        y = y[:, :, :-extra_h]
    if extra_w > 0:
        y = y[:, :, :, :-extra_w]
    return y
