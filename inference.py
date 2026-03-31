"""
Inference script: run DDPM reverse diffusion on range-view images,
then back-project 2-D predictions to per-point 3-D labels.
Adapted from RangeViT (back-projection) and SegDiff (sampling).

Usage:
    python inference.py --config config.yaml --checkpoint ckpt.pt
"""

import argparse
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import RangeViewDataset
from diffusion import GaussianDiffusion, UNet


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config',     default='config.yaml')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--split',      default='val',
                   help='which dataset split to evaluate (val / test)')
    return p.parse_args()


def iou_per_class(pred, gt, num_classes):
    """Compute per-class IoU; returns (num_classes,) array."""
    iou = np.zeros(num_classes, dtype=np.float64)
    for c in range(num_classes):
        tp = ((pred == c) & (gt == c)).sum()
        fp = ((pred == c) & (gt != c)).sum()
        fn = ((pred != c) & (gt == c)).sum()
        iou[c] = tp / max(tp + fp + fn, 1)
    return iou


def infer():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---- Dataset ----
    ds_cfg  = cfg['dataset']
    seqs    = ds_cfg.get(f'{args.split}_seqs', ds_cfg['val_seqs'])
    eval_ds = RangeViewDataset(
        root      = ds_cfg['root'],
        sequences = seqs,
        proj_h    = ds_cfg['proj_h'],
        proj_w    = ds_cfg['proj_w'],
        is_train  = False,
    )
    loader  = DataLoader(eval_ds, batch_size=1, shuffle=False, num_workers=2)

    # ---- Model ----
    m_cfg = cfg['model']
    model = UNet(
        img_channels   = 5,
        num_classes    = 20,
        base_ch        = m_cfg['base_ch'],
        ch_mults       = tuple(m_cfg['ch_mults']),
        num_res_blocks = m_cfg['num_res_blocks'],
        use_mid_attn   = m_cfg.get('use_mid_attn', True),
        dropout        = 0.0,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    diffusion = GaussianDiffusion(
        num_timesteps = cfg['diffusion']['num_timesteps'],
        schedule      = cfg['diffusion']['schedule'],
    )

    num_classes  = 20
    all_iou      = []

    print(f'Evaluating on {len(eval_ds)} scans...')
    for seg, img, mask, labels_2d, uproj in loader:
        img_dev = img.to(device)
        B, C, H, W = img_dev.shape

        # ---- DDPM reverse diffusion → predicted segmentation ----
        pred_seg = diffusion.p_sample_loop(
            model  = model,
            shape  = (B, num_classes, H, W),
            cond   = img_dev,
            device = device,
        )  # (1, 20, H, W) in [-1, 1]

        # Discrete 2-D predictions: argmax over class dimension
        pred_2d = pred_seg[0].argmax(dim=0).cpu().numpy()  # (H, W)

        # ---- Back-project 2-D predictions → per-point 3-D labels ----
        ux = uproj['x'][0].numpy()   # (N,)
        uy = uproj['y'][0].numpy()   # (N,)
        pred_3d = pred_2d[uy, ux]    # (N,) point-wise labels

        # Ground-truth 3-D labels reconstructed from 2-D label map
        gt_2d   = labels_2d[0].numpy()
        gt_3d   = gt_2d[uy, ux]

        # Ignore class 0 (unlabelled) in IoU
        valid   = gt_3d > 0
        iou     = iou_per_class(pred_3d[valid], gt_3d[valid], num_classes)
        all_iou.append(iou)

    all_iou = np.stack(all_iou)              # (N_scans, 20)
    miou    = all_iou[:, 1:].mean()          # exclude class-0 (ignore)
    per_cls = all_iou[:, 1:].mean(axis=0)   # per-class mean over scans

    print(f'\nmIoU (classes 1-19): {miou * 100:.2f}%')
    for c, v in enumerate(per_cls, start=1):
        print(f'  class {c:2d}: {v * 100:.2f}%')


if __name__ == '__main__':
    infer()
