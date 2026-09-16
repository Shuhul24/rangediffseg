"""
Train the range-image VAE (models/rangevae.py), stage 1 of the latent-diffusion
route: range image -> 4-channel latent -> (later) DiT with a diffusion loss.

Self-supervised: uses only the velodyne scans, no labels, so every SemanticKITTI
sequence with point clouds can be used.

    python train_vae.py --data_root /path/to/kitti/dataset --save_path ./vae_log
"""

import argparse
import datetime
import glob
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from dataset.projection import RangeProjection
from models.rangevae import RangeVAE

MAX_RANGE = 80.0


class RangeImageDataset(Dataset):
    """Range images only: [range, intensity, mask], each scaled to [-1, 1]."""

    def __init__(self, data_root, sequences, proj_h=64, proj_w=2048,
                 fov_up=3.0, fov_down=-25.0):
        self.files = []
        for seq in sequences:
            self.files += sorted(glob.glob(
                os.path.join(data_root, 'sequences', seq, 'velodyne', '*.bin')))
        if not self.files:
            raise FileNotFoundError(f'no scans found under {data_root} for {sequences}')
        self.projector = RangeProjection(proj_h, proj_w, fov_up, fov_down)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        pts = np.fromfile(self.files[idx], dtype=np.float32).reshape(-1, 4)
        proj_range, proj_feat, _, proj_mask, _, _ = self.projector.project(pts)

        rng = np.clip(proj_range, 0.0, MAX_RANGE) / MAX_RANGE * 2.0 - 1.0
        inten = np.clip(proj_feat[..., 4], 0.0, 1.0) * 2.0 - 1.0
        mask = proj_mask.astype(np.float32) * 2.0 - 1.0
        rng[~proj_mask] = -1.0
        inten[~proj_mask] = -1.0

        x = np.stack([rng, inten, mask], axis=0).astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(proj_mask.astype(np.float32))


def parse_args():
    p = argparse.ArgumentParser('Range-image VAE training')
    p.add_argument('--data_root', type=str, required=True)
    p.add_argument('--save_path', type=str, required=True)
    p.add_argument('--id', type=str, default='rangevae')
    p.add_argument('--train_seqs', type=str,
                   default='00,01,02,03,04,05,06,07,09,10')
    p.add_argument('--val_seqs', type=str, default='08')
    p.add_argument('--proj_h', type=int, default=64)
    p.add_argument('--proj_w', type=int, default=2048)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=1e-6)
    p.add_argument('--kl_weight', type=float, default=1e-6)   # RangeLDM's value
    p.add_argument('--ch', type=int, default=64)
    p.add_argument('--z_channels', type=int, default=4)       # must stay 4 for DiT
    p.add_argument('--warmup_steps', type=int, default=500)
    p.add_argument('--log_every', type=int, default=100)
    p.add_argument('--val_every', type=int, default=2)
    p.add_argument('--amp', action='store_true', default=True)
    return p.parse_args()


def masked_l1(recon, target, valid):
    """Reconstruction error over occupied pixels only, plus the mask channel."""
    m = valid.unsqueeze(1)
    geo = (recon[:, :2] - target[:, :2]).abs() * m
    geo = geo.sum() / m.sum().clamp(min=1.0) / 2.0
    occ = (recon[:, 2:] - target[:, 2:]).abs().mean()
    return geo, occ


@torch.no_grad()
def evaluate(model, loader, device, max_batches=60):
    model.eval()
    tot_geo = tot_occ = tot_rng_m = n = 0
    for i, (x, valid) in enumerate(loader):
        if i >= max_batches:
            break
        x, valid = x.to(device), valid.to(device)
        recon, _ = model(x, sample_posterior=False)
        geo, occ = masked_l1(recon, x, valid)
        # Range error back in metres, on occupied pixels.
        err = ((recon[:, 0] - x[:, 0]).abs() * valid).sum() / valid.sum().clamp(min=1.0)
        tot_geo += geo.item(); tot_occ += occ.item()
        tot_rng_m += (err.item() / 2.0 * MAX_RANGE); n += 1
    model.train()
    return tot_geo / n, tot_occ / n, tot_rng_m / n


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = os.path.join(args.save_path, f'log_{args.id}')
    os.makedirs(out_dir, exist_ok=True)

    train_ds = RangeImageDataset(args.data_root, args.train_seqs.split(','),
                                 args.proj_h, args.proj_w)
    val_ds = RangeImageDataset(args.data_root, args.val_seqs.split(','),
                               args.proj_h, args.proj_w)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=True,
                              pin_memory=True, persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    model = RangeVAE(in_channels=3, out_ch=3, ch=args.ch,
                     z_channels=args.z_channels).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    lat_h = args.proj_h // model.downsample_factor
    lat_w = args.proj_w // model.downsample_factor
    print(f'RangeVAE: {n_par/1e6:.1f}M params', flush=True)
    print(f'  {args.proj_h}x{args.proj_w} -> latent {args.z_channels}x{lat_h}x{lat_w} '
          f'({model.downsample_factor}x downsample)', flush=True)
    print(f'  train scans {len(train_ds)}, val scans {len(val_ds)}, '
          f'batch {args.batch_size}, {len(train_loader)} iters/epoch', flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            betas=(0.5, 0.9), weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(args.warmup_steps, 1)) *
                       (0.5 * (1 + np.cos(np.pi * min(s / total_steps, 1.0)))))
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)

    best = float('inf')
    step, t0 = 0, time.time()
    for epoch in range(args.epochs):
        for x, valid in train_loader:
            x, valid = x.to(device, non_blocking=True), valid.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=args.amp):
                recon, posterior = model(x)
                geo, occ = masked_l1(recon, x, valid)
                kl = posterior.kl()
                loss = geo + occ + args.kl_weight * kl

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()

            if step % args.log_every == 0:
                el = time.time() - t0
                eta = datetime.timedelta(seconds=int(el / max(step, 1) * (total_steps - step)))
                print(f'>>> E[{args.epochs:03d}|{epoch+1:03d}] S[{step:07d}|{total_steps:07d}] '
                      f'loss {loss.item():.4f} geo {geo.item():.4f} occ {occ.item():.4f} '
                      f'kl {kl.item():.1f} lr {sched.get_last_lr()[0]:.2e} ETA {eta}', flush=True)
            step += 1

        if (epoch + 1) % args.val_every == 0 or epoch == args.epochs - 1:
            g, o, rng_m = evaluate(model, val_loader, device)
            print(f'=== VAL epoch {epoch+1}: geo {g:.4f}  occ {o:.4f}  '
                  f'range MAE {rng_m:.3f} m ===', flush=True)
            torch.save({'model': model.state_dict(), 'epoch': epoch,
                        'args': vars(args), 'val_geo': g, 'val_range_mae_m': rng_m},
                       os.path.join(out_dir, 'last.pth'))
            if g < best:
                best = g
                torch.save({'model': model.state_dict(), 'epoch': epoch,
                            'args': vars(args), 'val_geo': g, 'val_range_mae_m': rng_m},
                           os.path.join(out_dir, 'best.pth'))
                print(f'    new best (geo {g:.4f})', flush=True)

    print(f'=== done in {datetime.timedelta(seconds=int(time.time()-t0))} ===', flush=True)


if __name__ == '__main__':
    main()
