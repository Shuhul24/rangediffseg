"""
Training script: DDPM for range-view LiDAR segmentation.
Combines preprocessing from RangeViT with the DDPM training loop from SegDiff.

Usage:
    python train.py --config config.yaml
"""

import argparse
import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import RangeViewDataset
from diffusion import GaussianDiffusion, UNet


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='config.yaml')
    p.add_argument('--resume', default=None, help='path to checkpoint')
    return p.parse_args()


def collate_fn(batch):
    """Custom collate: stack tensors, drop uproj dicts (not needed for training)."""
    segs, imgs, masks, labels, _ = zip(*batch)
    return (torch.stack(segs), torch.stack(imgs),
            torch.stack(masks), torch.stack(labels))


def train():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---- Dataset ----
    ds_cfg  = cfg['dataset']
    train_ds = RangeViewDataset(
        root       = ds_cfg['root'],
        sequences  = ds_cfg['train_seqs'],
        proj_h     = ds_cfg['proj_h'],
        proj_w     = ds_cfg['proj_w'],
        is_train   = True,
        aug_cfg    = ds_cfg.get('augmentation', {}),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size  = cfg['training']['batch_size'],
        shuffle     = True,
        num_workers = cfg['training'].get('num_workers', 4),
        collate_fn  = collate_fn,
        pin_memory  = True,
    )

    # ---- Model ----
    m_cfg = cfg['model']
    model = UNet(
        img_channels  = 5,
        num_classes   = 20,
        base_ch       = m_cfg['base_ch'],
        ch_mults      = tuple(m_cfg['ch_mults']),
        num_res_blocks = m_cfg['num_res_blocks'],
        use_mid_attn  = m_cfg.get('use_mid_attn', True),
        dropout       = m_cfg.get('dropout', 0.1),
    ).to(device)

    diffusion = GaussianDiffusion(
        num_timesteps = cfg['diffusion']['num_timesteps'],
        schedule      = cfg['diffusion']['schedule'],
    )

    # ---- Optimiser ----
    t_cfg     = cfg['training']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = t_cfg['lr'],
        weight_decay = t_cfg.get('weight_decay', 1e-4),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    start_step = 0
    if args.resume and os.path.isfile(args.resume):
        ckpt       = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_step = ckpt.get('step', 0)
        print(f'Resumed from step {start_step}')

    os.makedirs(t_cfg['log_dir'], exist_ok=True)
    max_steps    = t_cfg['max_steps']
    save_every   = t_cfg['save_every']
    log_every    = t_cfg.get('log_every', 100)
    step         = start_step

    model.train()
    print(f'Training on {len(train_ds)} scans | device={device}')

    while step < max_steps:
        for seg, img, mask, _ in train_loader:
            if step >= max_steps:
                break

            seg, img, mask = seg.to(device), img.to(device), mask.to(device)

            # Uniform timestep sampling
            t = torch.randint(0, diffusion.T, (seg.shape[0],), device=device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                loss = diffusion.training_loss(model, seg, t, img, mask)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            step += 1
            if step % log_every == 0:
                print(f'step {step:6d} / {max_steps}  loss={loss.item():.4f}')

            if step % save_every == 0:
                ckpt_path = os.path.join(t_cfg['log_dir'], f'ckpt_{step:07d}.pt')
                torch.save({
                    'step':      step,
                    'model':     model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, ckpt_path)
                print(f'Saved checkpoint → {ckpt_path}')

    print('Training complete.')


if __name__ == '__main__':
    train()
