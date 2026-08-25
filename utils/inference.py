"""
Sliding-window inference over full-width range images.

The model is trained on narrow crops (e.g. 64 x 384) while a full SemanticKITTI
range image is much wider (64 x 512 ... 64 x 2048). At evaluation time we slide
the window across the image and average the overlapping logits, as RangeViT
does (github.com/valeoai/rangevit).
"""

import torch
import torch.nn.functional as F


def _window_anchors(length, window, stride):
    """Top-left positions of the windows covering `length` pixels."""
    if window >= length:
        return [0]
    anchors = list(range(0, length - window + 1, max(stride, 1)))
    if anchors[-1] + window < length:
        anchors.append(length - window)
    return anchors


@torch.no_grad()
def sliding_window_inference(model, im, window_size, window_stride, n_cls, batch_size=1):
    """
    Args:
        model         : takes (B, C, h, w) and returns (B, n_cls, h, w) logits
        im            : (1, C, H, W) range image
        window_size   : (h, w) of the window seen by the model
        window_stride : (h, w) step between two consecutive windows
    Returns:
        (n_cls, H, W) logits averaged over the overlapping windows
    """
    assert im.shape[0] == 1, 'sliding-window inference expects one scan at a time'
    _, C, H, W = im.shape
    ws_h, ws_w = window_size
    st_h, st_w = window_stride

    # Zero-pad when the image is smaller than one window.
    pad_h, pad_w = max(ws_h - H, 0), max(ws_w - W, 0)
    if pad_h or pad_w:
        im = F.pad(im, (0, pad_w, 0, pad_h), value=0)
    H_pad, W_pad = im.shape[2], im.shape[3]

    anchors = [(top, left)
               for top in _window_anchors(H_pad, ws_h, st_h)
               for left in _window_anchors(W_pad, ws_w, st_w)]

    logits = torch.zeros((n_cls, H_pad, W_pad), device=im.device, dtype=torch.float32)
    counts = torch.zeros((1, H_pad, W_pad), device=im.device, dtype=torch.float32)

    for start in range(0, len(anchors), batch_size):
        chunk = anchors[start:start + batch_size]
        windows = torch.cat([im[:, :, t:t + ws_h, l:l + ws_w] for t, l in chunk], dim=0)
        out = model(windows).float()
        for i, (t, l) in enumerate(chunk):
            logits[:, t:t + ws_h, l:l + ws_w] += out[i]
            counts[:, t:t + ws_h, l:l + ws_w] += 1

    logits = logits / counts.clamp(min=1e-6)
    return logits[:, :H, :W]
