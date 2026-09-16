"""
Is the frozen DiT trunk doing anything useful?

A plateau far below the RangeViT reference is usually not a tuning problem, and
this script tests the cheapest structural explanation without training anything:
it pushes one batch through the network and reports, per block, how much the
block actually changes its input and how strong its adaLN-Zero gates are at the
conditioning the model is configured with.

Three outcomes, three different fixes:

  * `|dx|/|x|` near zero in most blocks
        The trunk is close to the identity at this conditioning -- the 677 M
        parameters are a very expensive no-op and the decoder is working off the
        stem skip alone. Sweep `cond_timestep`; if that does not wake it up, the
        premise does not hold and `n_trunk_blocks: 0` should match the full model.

  * `|dx|/|x|` large and `cos(stem, out)` near zero
        The trunk is overwriting the stem's features rather than refining them.
        The decoder then has to undo it, which caps accuracy. Tap earlier blocks
        (`feature_taps`) or truncate (`n_trunk_blocks`).

  * effective rank collapsing with depth
        The representation is degenerating -- all tokens converging to the same
        vector. Read from before the collapse.

Usage:
    python -m tools.diagnose_trunk config.yaml                      # random input
    python -m tools.diagnose_trunk config.yaml --data_root <root>   # real scans
    python -m tools.diagnose_trunk config.yaml --checkpoint <ckpt>  # a trained model
    python -m tools.diagnose_trunk config.yaml --sweep_timestep 0 50 100 250 500
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from option import Option                     # noqa: E402
from train import build_rangedit_model        # noqa: E402


def effective_rank(x):
    """Participation ratio of the singular values: 1 = rank-1 collapse."""
    x = x.reshape(-1, x.shape[-1]).float()
    x = x - x.mean(0, keepdim=True)
    try:
        s = torch.linalg.svdvals(x.double())
    except Exception:
        return float('nan')
    s2 = (s ** 2)
    if s2.sum() <= 0:
        return float('nan')
    p = s2 / s2.sum()
    return float(torch.exp(-(p * p.clamp_min(1e-12).log()).sum()))


@torch.no_grad()
def probe(model, x, device):
    enc = model.rangedit.encoder
    enc.eval()

    tokens, _ = enc.patch_embed(x.to(device))
    pos = enc.pos_embed
    if tokens.shape[1] != pos.shape[1]:
        from models.model_utils import resize_pos_embed
        gh, gw = enc.get_grid_size(x.shape[2], x.shape[3])
        pos = resize_pos_embed(pos, enc.patch_embed.grid_size, (gh, gw), num_extra_tokens=0)
    h = tokens + pos
    stem_repr = h.clone()
    c = enc.get_conditioning(h)

    rows = []
    for i in range(enc.n_trunk_blocks):
        blk = enc.blocks[i]
        gates = blk.adaLN_modulation(c).chunk(6, dim=1)
        gate_msa, gate_mlp = gates[2], gates[5]

        h_in = h
        h = blk(h_in, c)

        delta = (h - h_in).norm(dim=-1)
        base = h_in.norm(dim=-1).clamp_min(1e-9)
        cos_stem = torch.nn.functional.cosine_similarity(
            h.reshape(-1, h.shape[-1]), stem_repr.reshape(-1, stem_repr.shape[-1]), dim=-1)

        rows.append({
            'block': i,
            'rel_dx': float((delta / base).mean()),
            'gate_msa': float(gate_msa.abs().mean()),
            'gate_mlp': float(gate_mlp.abs().mean()),
            'tok_std': float(h.std()),
            'cos_stem': float(cos_stem.mean()),
            'eff_rank': effective_rank(h[0]),
        })
    return rows, stem_repr


def print_table(rows, every=1):
    print(f"{'blk':>4} {'|dx|/|x|':>9} {'gate_msa':>9} {'gate_mlp':>9} "
          f"{'tok_std':>8} {'cos(stem)':>10} {'eff_rank':>9}")
    for r in rows:
        if r['block'] % every and r['block'] != rows[-1]['block']:
            continue
        print(f"{r['block']:>4} {r['rel_dx']:>9.4f} {r['gate_msa']:>9.4f} "
              f"{r['gate_mlp']:>9.4f} {r['tok_std']:>8.3f} {r['cos_stem']:>10.4f} "
              f"{r['eff_rank']:>9.1f}")


def verdict(rows):
    rel = np.array([r['rel_dx'] for r in rows])
    cos_final = rows[-1]['cos_stem']
    rank_first, rank_last = rows[0]['eff_rank'], rows[-1]['eff_rank']
    print('\n---- reading ----')

    quiet = (rel < 0.02).mean()
    if quiet > 0.5:
        print(f'* {quiet:.0%} of blocks change their input by <2%: the trunk is close to '
              f'the IDENTITY at this conditioning.')
        print('  -> sweep model.cond_timestep; if nothing changes, n_trunk_blocks: 0 '
              'should match the full model and the trunk is not earning its place.')
    else:
        print(f'* blocks are active (median |dx|/|x| = {np.median(rel):.3f}).')

    if cos_final < 0.1:
        print(f'* cos(stem, output) = {cos_final:.3f}: the trunk OVERWRITES the stem '
              f'representation rather than refining it.')
        print('  -> try feature_taps on earlier blocks, or truncate with n_trunk_blocks.')
    else:
        print(f'* cos(stem, output) = {cos_final:.3f}: the output still carries the stem signal.')

    if np.isfinite(rank_last) and np.isfinite(rank_first) and rank_last < 0.5 * rank_first:
        print(f'* effective rank collapses {rank_first:.0f} -> {rank_last:.0f}: the tokens '
              f'are degenerating with depth.')
        print('  -> read from before the collapse (feature_taps / n_trunk_blocks).')


def get_batch(settings, args, device):
    if args.data_root:
        import dataset
        settings.data_root = args.data_root
        parser = dataset.build_parser(settings.dataset, args.data_root, 'val', settings)
        ds = dataset.RangeViewDataset(
            parser, proj_h=settings.proj_h, proj_w=settings.proj_w,
            fov_up=settings.fov_up, fov_down=settings.fov_down,
            img_mean=settings.img_mean, img_stds=settings.img_stds,
            image_size=settings.image_size, is_train=False)
        feats = []
        for k in range(args.batch):
            f, _, _ = ds[k * max(1, len(ds) // max(args.batch, 1))]
            h, w = settings.image_size
            feats.append(f[:, :h, :w])
        print(f'Using {args.batch} real scans from the val split.')
        return torch.stack(feats).to(device)

    print('Using random input (pass --data_root for real scans).')
    return torch.randn(args.batch, settings.in_channels, *settings.image_size, device=device)


def main():
    ap = argparse.ArgumentParser(description='Probe the frozen DiT trunk')
    ap.add_argument('config_path', nargs='?', default='config.yaml')
    ap.add_argument('--data_root', default=None)
    ap.add_argument('--checkpoint', default=None, help='a trained RangeDiT checkpoint')
    ap.add_argument('--batch', type=int, default=2)
    ap.add_argument('--every', type=int, default=1, help='print every Nth block')
    ap.add_argument('--sweep_timestep', type=int, nargs='*', default=None)
    args = ap.parse_args()

    class _A:
        pass
    a = _A()
    a.data_root = args.data_root or '/tmp'
    settings = Option(args.config_path, a)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pretrained = None if args.checkpoint else settings.pretrained_model
    model = build_rangedit_model(settings, pretrained_path=pretrained).to(device).eval()
    if args.checkpoint:
        ck = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(ck.get('model', ck), strict=False)
        print(f'Loaded {args.checkpoint}')

    x = get_batch(settings, args, device)
    print(f'\nbackbone={settings.backbone}  pretrained={settings.pretrained_model}  '
          f'cond_timestep={settings.cond_timestep}  cond_mode={settings.cond_mode}  '
          f'input={tuple(x.shape)}')

    timesteps = args.sweep_timestep if args.sweep_timestep else [settings.cond_timestep]
    for t in timesteps:
        enc = model.rangedit.encoder
        enc.cond_timestep.fill_(float(t))
        print(f'\n================ cond_timestep = {t} ================')
        rows, _ = probe(model, x, device)
        print_table(rows, every=args.every)
        verdict(rows)

    if len(timesteps) > 1:
        print('\nPick the timestep with the highest sustained |dx|/|x| that does NOT '
              'collapse the effective rank, then retrain with model.cond_timestep set to it.')


if __name__ == '__main__':
    main()
