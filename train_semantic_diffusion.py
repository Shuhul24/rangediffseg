"""Train conditional semantic DiT. Single-GPU, exact epoch-boundary restart."""
import argparse
import json
import math
import os
from pathlib import Path
import random
import subprocess
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
import yaml

import dataset
from semantic_diffusion import SemanticDiT
from utils.losses import Lovasz_softmax


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(_):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def atomic_save(payload, path):
    tmp = str(path) + '.tmp'
    torch.save(payload, tmp)
    os.replace(tmp, path)


def lr_at(step, total, warmup, peak):
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    progress = min(1., (step - warmup) / max(1, total - warmup - 1))
    return peak * 0.5 * (1 + math.cos(math.pi * progress))


def loss_fn(logits, labels, valid, lovasz, lovasz_weight):
    target = (labels - 1).masked_fill((labels == 0) | ~valid.bool(), -1)
    if not target.ge(0).any():
        return logits.sum() * 0
    ce = F.cross_entropy(logits.float(), target, ignore_index=-1)
    return ce + lovasz_weight * lovasz(logits.float().softmax(1), target)


def add_confusion(cm, prediction, target):
    good = (target > 0) & (target <= cm.shape[0])
    bins = (target[good].long() - 1) * cm.shape[0] + prediction[good].long()
    cm += torch.bincount(bins.cpu(), minlength=cm.numel()).reshape_as(cm)


def metrics(cm):
    inter = cm.diag().double()
    union = cm.sum(0) + cm.sum(1) - inter
    iou = inter / union.clamp_min(1)
    return {'miou': iou.mean().item(), 'class_iou': iou.tolist()}


@torch.no_grad()
def evaluate(model, loader, device, cfg, steps, amp):
    model.eval()
    k = model.classes
    pixel, point = torch.zeros(k, k, dtype=torch.long), torch.zeros(k, k, dtype=torch.long)
    width, stride = cfg['data']['crop'][1], cfg['evaluation']['stride']
    started = time.time()
    for index, batch in enumerate(loader):
        x, labels, valid, px, py, point_labels = batch[:6]
        x, valid = x.to(device), valid.to(device)
        h, w = x.shape[-2:]
        if w < width:
            raise ValueError('Evaluation image is narrower than configured crop')
        anchors = list(range(0, w - width + 1, stride))
        if anchors[-1] != w - width:
            anchors.append(w - width)
        logits = torch.zeros(1, k, h, w, device=device)
        count = torch.zeros(1, 1, h, w, device=device)
        for left in anchors:
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                out = model.sample(x[..., left:left+width], valid[..., left:left+width],
                                   steps=steps, seed=cfg['seed'] + index * 10007 + left)
            logits[..., left:left+width] += out.float()
            count[..., left:left+width] += 1
        pred = (logits / count).argmax(1).cpu()[0]
        add_confusion(pixel, pred.flatten(), labels[0].flatten())
        pp = pred[py[0].long(), px[0].long()]
        add_confusion(point, pp, point_labels[0])
        if (index + 1) % 100 == 0:
            print(f'validation scans={index+1}/{len(loader)} point_miou={metrics(point)["miou"]:.4f}', flush=True)
    return {'pixel': metrics(pixel), 'point': metrics(point), 'seconds': time.time() - started,
            'scans': len(loader), 'steps': steps}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config_semantic_diffusion.yaml')
    parser.add_argument('--output', required=True)
    parser.add_argument('--resume')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--steps', type=int)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.smoke:
        cfg['training'].update(epochs=1, batch_size=2, accumulation=1, workers=0, val_every=1)
    seed_all(cfg['seed'])
    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp = device.type == 'cuda' and torch.cuda.is_bf16_supported()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'last.pt').exists() and not args.resume and not args.eval_only:
        raise RuntimeError('Existing run: supply --resume or a fresh output directory')
    if args.eval_only and not args.resume:
        raise ValueError('--eval-only requires --resume')
    d, tr = cfg['data'], cfg['training']
    if set(d['train_seqs']) & set(d['val_seqs']):
        raise ValueError('Train/validation sequence overlap')
    common = dict(proj_h=d['height'], proj_w=d['width'], fov_up=3., fov_down=-25.)
    train_ds = dataset.RangeViewDataset(dataset.SemanticKITTI(d['root'], d['train_seqs']),
                    image_size=d['crop'], is_train=True, aug_cfg=d['augmentation'],
                    range_aug_cfg=d.get('range_augmentation'), **common)
    val_ds = dataset.RangeViewDataset(dataset.SemanticKITTI(d['root'], d['val_seqs']),
                    is_train=False, return_points=True, **common)
    if args.smoke:
        train_ds, val_ds = Subset(train_ds, range(4)), Subset(val_ds, range(2))
    generator = torch.Generator()
    loader = DataLoader(train_ds, batch_size=tr['batch_size'], shuffle=True,
                        num_workers=tr['workers'], drop_last=True, generator=generator,
                        worker_init_fn=seed_worker, pin_memory=device.type == 'cuda')
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=tr['workers'], worker_init_fn=seed_worker)
    if not len(loader):
        raise ValueError('Training loader is empty')
    model = SemanticDiT(**cfg['model']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tr['lr'], weight_decay=tr['weight_decay'])
    lovasz = Lovasz_softmax(ignore=-1)
    updates = math.ceil(len(loader) / tr['accumulation'])
    total, warmup = updates * tr['epochs'], updates * tr['warmup_epochs']
    start, step, best = 0, 0, -1.
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        if ckpt['config']['model'] != cfg['model']:
            raise ValueError('Model configuration differs from checkpoint')
        model.load_state_dict(ckpt['model'])
        if not args.eval_only:
            if ckpt['config'] != cfg:
                raise ValueError('Exact restart requires identical resolved config')
            optimizer.load_state_dict(ckpt['optimizer'])
            start, step, best = ckpt['epoch'] + 1, ckpt['step'], ckpt['best']
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    (output / 'resolved_config.yaml').write_text(yaml.safe_dump(cfg))
    print(json.dumps({'commit': revision, 'device': str(device), 'bf16': amp,
                      'parameters': sum(p.numel() for p in model.parameters()),
                      'train_batches': len(loader), 'val_scans': len(val_loader),
                      'effective_batch': tr['batch_size'] * tr['accumulation']}), flush=True)
    if args.eval_only:
        result = evaluate(model, val_loader, device, cfg, args.steps or cfg['evaluation']['steps'], amp)
        (output / f'eval_{result["steps"]}_steps.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
        return
    for epoch in range(start, tr['epochs']):
        # Reproduce loader order, augmentation and noise at epoch-boundary restart.
        seed_all(cfg['seed'] + epoch)
        generator.manual_seed(cfg['seed'] + epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for i, (x, labels, valid) in enumerate(loader):
            x, labels, valid = x.to(device), labels.to(device), valid.to(device)
            window_start = (i // tr['accumulation']) * tr['accumulation']
            window_size = min(tr['accumulation'], len(loader) - window_start)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                condition = model.encode(x, valid)
                t = torch.rand(x.shape[0], device=device)
                clean = model.label_codes(labels)
                sc = None
                if model.mode == 'diffusion':
                    if epoch >= tr['prediction_warmup_epochs'] and random.random() < tr['prediction_noise_probability']:
                        with torch.no_grad():
                            prior = torch.randn_like(clean) * valid[:, None]
                            first = model.denoise(prior, torch.ones_like(t), condition)
                            clean = model.codes(first.float().softmax(1)).detach()
                    z = model.corrupt(clean, t, torch.randn_like(clean)) * valid[:, None]
                    if random.random() < tr['self_condition_probability']:
                        with torch.no_grad():
                            sc = model.codes(model.denoise(z, t, condition).float().softmax(1)).detach() * valid[:, None]
                else:
                    t.zero_()
                    z = torch.zeros_like(clean)
                logits = model.denoise(z, t, condition, sc)
                loss = loss_fn(logits, labels, valid, lovasz, tr['lovasz_weight'])
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss epoch={epoch} batch={i}')
            (loss / window_size).backward()
            if (i + 1) % tr['accumulation'] == 0 or i + 1 == len(loader):
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tr['grad_clip'], error_if_nonfinite=True)
                lr = lr_at(step, total, warmup, tr['lr'])
                for group in optimizer.param_groups:
                    group['lr'] = lr
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if step % tr['log_every'] == 0 or args.smoke:
                    print(f'epoch={epoch+1} batch={i+1}/{len(loader)} step={step} loss={loss.item():.5f} lr={lr:.7g} grad={norm.item():.4f}', flush=True)
        result = None
        improved = False
        if epoch == 0 or (epoch + 1) % tr['val_every'] == 0 or epoch + 1 == tr['epochs']:
            result = evaluate(model, val_loader, device, cfg, cfg['evaluation']['steps'], amp)
            improved = result['point']['miou'] > best
            best = max(best, result['point']['miou'])
            record = {'epoch': epoch + 1, 'step': step, 'best_point_miou': best, **result}
            with (output / 'metrics.jsonl').open('a') as f:
                f.write(json.dumps(record) + '\n')
            print(json.dumps(record), flush=True)
        payload = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                   'epoch': epoch, 'step': step, 'best': best, 'config': cfg, 'commit': revision}
        atomic_save(payload, output / 'last.pt')
        if improved:
            atomic_save({k: v for k, v in payload.items() if k != 'optimizer'}, output / 'best.pt')
        print(f'checkpoint saved epoch={epoch+1} best_point_miou={best:.5f}', flush=True)


if __name__ == '__main__':
    main()
