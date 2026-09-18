"""
Smoke test for the rangeformer_dit_seg branch. CPU or single GPU, 1-2 samples.

Covers the twelve checks required before training:
  1 five network channels          7 finite non-zero gradients everywhere
  2 labels/projection alignment    8 inference without GT
  3 shapes printed once            9 range-view -> point mapping
  4 forward pass                  10 checkpoint save/load reproduces a forward
  5 finite L_total                11 accumulation steps the optimizer once per window
  6 backward pass                 12 AMP produces no NaN/Inf

    python tools/smoke_test_rangeformer_dit.py --config config_rangeformer_dit.yaml \
        --data_root /path/to/kitti/dataset [--device cuda]
"""

import argparse
import os
import sys
import tempfile

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dataset  # noqa: E402
from dataset.projection import CHANNEL_ORDERS  # noqa: E402
from models.rangeformer_dit import RangeFormerDiTSeg  # noqa: E402
from option import Option  # noqa: E402

OK, FAIL = '  [ok]  ', '  [FAIL]'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='config_rangeformer_dit.yaml')
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--samples', type=int, default=2)
    ap.add_argument('--accum', type=int, default=4)
    a = ap.parse_args()

    dev = torch.device(a.device)
    torch.manual_seed(0)

    class _A:
        pass
    args = _A()
    args.data_root = a.data_root
    for k in ('save_path', 'id', 'num_workers', 'pretrained_model', 'checkpoint',
              'finetune_from', 'log_frequency', 'seed', 'batch_size', 'model_type',
              'gradient_accumulation_steps'):
        setattr(args, k, None)
    for k in ('val_only', 'test_split', 'save_eval_results', 'use_wandb',
              'unfreeze_attn', 'unfreeze_ffn', 'debug_grad_flow'):
        setattr(args, k, False)
    s = Option(a.config, args)
    failures = []

    # ---- 1. five channels, in the right order ---------------------------
    print('\n1) range projection channels')
    assert s.channel_order == 'xyzri', s.channel_order
    names = CHANNEL_ORDERS[s.channel_order]
    parser = dataset.build_parser(s.dataset, s.data_root, 'train', s)
    ds = dataset.RangeViewDataset(
        parser, proj_h=s.proj_h, proj_w=s.proj_w, fov_up=s.fov_up,
        fov_down=s.fov_down, image_size=s.image_size, is_train=False,
        return_points=True, channel_order=s.channel_order)
    feat, label, mask, px, py, plabels = ds[0]
    print(f'    channel names: {names}')
    assert feat.shape[0] == 5, f'expected 5 channels, got {feat.shape[0]}'
    assert names == ('x', 'y', 'z', 'range', 'intensity')
    print(OK + f'5 channels, order {names}')

    # verify channel 3 really is sqrt(x^2+y^2+z^2) on raw (un-normalised) values
    raw = ds.projector.project(parser[0][0])[1]
    m = raw[..., 3] > 0
    d = np.linalg.norm(raw[m][:, 0:3], axis=1)
    err = np.abs(d - raw[m][:, 3]).max()
    assert err < 1e-3, f'channel 3 != sqrt(x^2+y^2+z^2), max err {err}'
    print(OK + f'channel 3 == sqrt(x^2+y^2+z^2)  (max err {err:.2e})')
    print(OK + f'valid_mask shape {tuple(mask.shape)}, {int(mask.sum())} valid pixels')

    # ---- 2. labels / projection indices align ---------------------------
    print('\n2) label and projection alignment')
    full = dataset.RangeViewDataset(
        parser, proj_h=s.proj_h, proj_w=s.proj_w, fov_up=s.fov_up,
        fov_down=s.fov_down, image_size=None, is_train=False,
        return_points=True, channel_order=s.channel_order)
    f2, l2, m2, px2, py2, pl2 = full[0]
    back = l2.numpy()[py2.numpy(), px2.numpy()]
    agree = (back == pl2.numpy()).mean()
    print(f'    points {len(pl2)}, back-projected labels agree {agree*100:.2f}%')
    assert agree > 0.90, f'label/index misalignment: only {agree:.3f} agree'
    print(OK + 'labels align with projection indices')

    # ---- 3/4. build, shapes, forward ------------------------------------
    print('\n3) tensor shapes through the architecture')
    from train import build_rangeformer_dit_model
    model = build_rangeformer_dit_model(s).to(dev)
    x = torch.stack([ds[i][0] for i in range(a.samples)]).to(dev)
    y = torch.stack([ds[i][1] for i in range(a.samples)]).to(dev).long()
    y = y * y.ge(1).long()
    print(f'    input                {tuple(x.shape)}')
    with torch.no_grad():
        rem = model.rem(x);              print(f'    REM                  {tuple(rem.shape)}')
        f1 = model.stage1(rem);          print(f'    encoder stage 1 (F1) {tuple(f1.shape)}')
        f2e = model.stage2(f1);          print(f'    encoder stage 2      {tuple(f2e.shape)}')
        c = model.latent_proj(f2e);      print(f'    latent condition C   {tuple(c.shape)}')
        cl = model.coarse_head(c);       print(f'    coarse logits        {tuple(cl.shape)}')
        s0, _ = model.semantic_latent_from_labels(y, c.shape[-2:])
        print(f'    semantic latent S0   {tuple(s0.shape)}')
        print(f'    DiT tokens           {c.shape[-2]*c.shape[-1]}')

    print('\n4) forward pass (train mode, labels supplied)')
    model.train()
    out = model(x, labels=y)
    print(f'    final logits         {tuple(out["logits"].shape)}')
    print(f'    eps / eps_hat        {tuple(out["eps"].shape)}')
    assert out['logits'].shape[-2:] == x.shape[-2:]
    print(OK + 'forward pass succeeded')

    # ---- 5/6/7. loss, backward, gradients -------------------------------
    print('\n5) total loss')
    import utils
    lov = utils.losses.Lovasz_softmax(ignore=0).to(dev)
    ce = F.cross_entropy(out['logits'].float(), y, ignore_index=0)
    lv = lov(F.softmax(out['logits'].float(), 1), y)
    h, w = out['latent_hw']
    ys = F.interpolate(y.unsqueeze(1).float(), size=(h, w), mode='nearest').squeeze(1).long()
    coarse = (F.cross_entropy(out['coarse_logits'].float(), ys, ignore_index=0)
              + lov(F.softmax(out['coarse_logits'].float(), 1), ys))
    diff = F.mse_loss(out['eps_hat'].float(), out['eps'].float())
    total = ce + lv + s.rf_lambda_diff * diff + s.rf_lambda_coarse * coarse
    print(f'    CE {ce.item():.4f}  Lovasz {lv.item():.4f}  '
          f'coarse {coarse.item():.4f}  diff {diff.item():.4f}  total {total.item():.4f}')
    assert torch.isfinite(total), 'L_total is not finite'
    print(OK + 'L_total finite')

    print('\n6/7) backward and per-module gradients')
    model.zero_grad(set_to_none=True)
    total.backward()
    probes = {'REM': model.rem, 'encoder stage 1': model.stage1,
              'encoder stage 2': model.stage2, 'latent_proj': model.latent_proj,
              'coarse head': model.coarse_head, 'class embedding': model.class_embed,
              'DiT': model.dit, 'decoder': model.decoder}
    all_ok = True
    for name, mod in probes.items():
        gs = [p.grad for p in mod.parameters() if p.requires_grad and p.grad is not None]
        if not gs:
            print(FAIL + f'{name:<18} NO GRADIENT'); all_ok = False; continue
        n = torch.sqrt(sum((g.float() ** 2).sum() for g in gs)).item()
        good = np.isfinite(n) and n > 0
        print((OK if good else FAIL) + f'{name:<18} grad_norm {n:.6e}')
        all_ok &= good
    if not all_ok:
        failures.append('gradient flow')

    # ---- 8. inference without GT ----------------------------------------
    print('\n8) inference without ground truth')
    model.eval()
    inf = model.infer(x, seed=0)     # seeded => deterministic evaluation
    print(f'    logits {tuple(inf["logits"].shape)}   (no labels argument passed)')
    assert torch.isfinite(inf['logits']).all()
    print(OK + 'inference succeeded with no GT')

    # ---- 9. map back to points ------------------------------------------
    print('\n9) range-view -> point mapping')
    model_full_in = f2.unsqueeze(0).to(dev)
    with torch.no_grad():
        li = model.infer(model_full_in, seed=0)['logits']
    pred2d = li.argmax(1)[0].cpu().numpy()
    pred3d = pred2d[py2.numpy(), px2.numpy()]
    print(f'    range-view {tuple(li.shape)} -> points {pred3d.shape}, '
          f'scan has {len(pl2)} points')
    assert pred3d.shape[0] == len(pl2)
    print(OK + 'point count matches')

    # ---- 10. checkpoint round trip --------------------------------------
    print('\n10) checkpoint save/load')
    model.eval()
    with torch.no_grad():
        a1 = model.infer(x, seed=0)['logits']
    with tempfile.TemporaryDirectory() as td:
        ck = os.path.join(td, 'ck.pth')
        torch.save({'model': model.state_dict()}, ck)
        m2 = build_rangeformer_dit_model(s).to(dev)
        m2.load_state_dict(torch.load(ck, map_location=dev)['model'], strict=True)
        m2.eval()
        with torch.no_grad():
            a2 = m2.infer(x, seed=0)['logits']
    d = (a1 - a2).abs().max().item()
    print(f'    max |before - after| = {d:.3e}  (seeded evaluation)')
    assert d < 1e-4, 'checkpoint round trip changed the forward pass'
    print(OK + 'strict load reproduces the forward pass')
    # unseeded inference is stochastic by design (DDIM init noise)
    with torch.no_grad():
        u1 = model.infer(x)['logits']; u2 = model.infer(x)['logits']
    print(f'    unseeded runs differ by {(u1-u2).abs().max().item():.3e} '
          f'(expected > 0: DDIM init is stochastic)')

    # ---- 11. gradient accumulation --------------------------------------
    print(f'\n11) gradient accumulation (accum={a.accum})')
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    steps_taken = 0
    before = None
    for micro in range(a.accum * 2):
        if micro % a.accum == 0:
            opt.zero_grad(set_to_none=True)
            # snapshot at the START of each window
            before = model.decoder.head.weight.detach().clone()
        o = model(x, labels=y)
        l = F.cross_entropy(o['logits'].float(), y, ignore_index=0) / a.accum
        l.backward()
        if (micro + 1) % a.accum == 0:
            opt.step(); steps_taken += 1
            opt.zero_grad(set_to_none=True)
        else:
            # mid-window: gradients accumulate but weights must not move
            assert torch.equal(before, model.decoder.head.weight.detach()), \
                f'weights changed at micro-step {micro}, before the window closed'
    print(f'    {a.accum * 2} micro-batches -> {steps_taken} optimizer steps '
          f'(expected {2})')
    assert steps_taken == 2
    print(OK + 'optimizer steps only at window boundaries')

    # ---- 12. AMP --------------------------------------------------------
    print('\n12) AMP')
    if dev.type == 'cuda':
        scaler = torch.amp.GradScaler('cuda')
        model.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            o = model(x, labels=y)
            l = (F.cross_entropy(o['logits'].float(), y, ignore_index=0)
                 + F.mse_loss(o['eps_hat'].float(), o['eps'].float()))
        assert torch.isfinite(l), 'AMP loss not finite'
        scaler.scale(l).backward()
        scaler.unscale_(opt)
        bad = [n for n, p in model.named_parameters()
               if p.grad is not None and not torch.isfinite(p.grad).all()]
        print(f'    AMP loss {l.item():.4f}, non-finite grads in {len(bad)} tensors')
        assert not bad, f'non-finite gradients under AMP: {bad[:5]}'
        print(OK + 'AMP produced no NaN/Inf')
    else:
        print('    skipped (CPU); rerun with --device cuda')

    print('\n' + ('ALL SMOKE TESTS PASSED' if not failures
                  else f'FAILURES: {failures}'))
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
