"""
Run a set of training variants against one base config and tabulate the result.

The point of this script is the *first* question in README section 7b: does the
frozen DiT trunk earn its 677 M parameters? Everything else in the architecture
is only worth tuning once that is answered, so the default suite starts there.

    # the decisive one -- does the trunk do anything?
    python -m tools.run_ablations config.yaml --data_root <root> --suite trunk

    # then, given the answer, where to read it and how to decode it
    python -m tools.run_ablations config.yaml --data_root <root> --suite depth
    python -m tools.run_ablations config.yaml --data_root <root> --suite decoder

    # anything you like
    python -m tools.run_ablations config.yaml --data_root <root> \
        --variant baseline \
        --variant 'taps:model.feature_taps=[6,13,20]' \
        --variant 'dw:model.stem_pool=dwconv'

Each variant runs `train.py` as a subprocess with `--id <base_id>_<name>`, so the
runs land in separate log directories and none of them overwrite each other. The
table is rebuilt from the on-disk logs, so re-running with `--skip_existing`
picks up where an interrupted sweep stopped, and `--table_only` just reprints.
"""

import argparse
import copy
import json
import os
import re
import subprocess
import sys

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# name -> list of "dotted.key=yaml_value" overrides applied to the base config
SUITES = {
    # Does the pre-trained trunk contribute anything at all?
    'trunk': {
        'full': [],
        'bypass': ['model.n_trunk_blocks=0'],
        'scratch': ['model.pretrained_model=null'],
    },
    # How much depth is actually needed?
    'depth': {
        'd28': [],
        'd21': ['model.n_trunk_blocks=21'],
        'd14': ['model.n_trunk_blocks=14'],
        'd7': ['model.n_trunk_blocks=7'],
    },
    # Where in the stack are the useful features?
    'taps': {
        'last': [],
        'mid': ['model.feature_taps=[13]'],
        'multi': ['model.feature_taps=[6,13,20]'],
        'early': ['model.feature_taps=[3,6,9]'],
    },
    # Does the conditioning timestep matter?
    'timestep': {
        't0': [],
        't50': ['model.cond_timestep=50'],
        't100': ['model.cond_timestep=100'],
        't250': ['model.cond_timestep=250'],
    },
    # Stem and decoder.
    'decoder': {
        'base': [],
        'stages2': ['model.decoder_stages=2'],
        'dwpool': ['model.stem_pool=dwconv'],
        'stages2_dwpool': ['model.decoder_stages=2', 'model.stem_pool=dwconv'],
        'prior_bias': ['model.head_prior_bias=true'],
    },
    # Tail-class handling (see README section 4b).
    'tail': {
        'plain': ['dataset.rare_crop_p=0.0'],
        'rare_crop': ['dataset.rare_crop_p=0.5'],
        'salsa_w': ['dataset.rare_crop_p=0.5', 'dataset.cls_weight_scheme=salsanext'],
        'lovasz2': ['dataset.rare_crop_p=0.5', 'training.lovasz_weight=2.0'],
    },
}


def set_dotted(cfg, dotted, raw_value):
    node = cfg
    parts = dotted.split('.')
    for key in parts[:-1]:
        node = node.setdefault(key, {})
    node[parts[-1]] = yaml.safe_load(raw_value)


def build_variant_config(base_cfg, overrides, run_id, out_dir):
    cfg = copy.deepcopy(base_cfg)
    for override in overrides:
        if '=' not in override:
            raise ValueError(f'override must look like key.path=value, got {override!r}')
        dotted, raw = override.split('=', 1)
        set_dotted(cfg, dotted.strip(), raw.strip())
    cfg.setdefault('training', {})['id'] = run_id
    path = os.path.join(out_dir, f'{run_id}.yaml')
    with open(path, 'w') as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)
    return path


# `>>> Validation mIoU 0.5734 | tail mIoU 0.3011 (...) | worst pole 0.0121`
VAL_LINE = re.compile(
    r'Validation mIoU\s+([\d.]+)\s+\|\s+tail mIoU\s+([\d.]+).*?\|\s+worst\s+(\S+)\s+([\d.]+)')


def parse_log(log_path):
    """Best validation result recorded in a run's console log."""
    if not os.path.isfile(log_path):
        return None
    best = None
    with open(log_path, errors='replace') as fh:
        for line in fh:
            m = VAL_LINE.search(line)
            if not m:
                continue
            row = {'mIoU': float(m.group(1)), 'tail_mIoU': float(m.group(2)),
                   'worst_class': m.group(3), 'worst_iou': float(m.group(4))}
            if best is None or row['mIoU'] > best['mIoU']:
                best = row
    return best


def log_path_for(base_cfg, run_id, save_path_override):
    save_path = save_path_override or base_cfg.get('training', {}).get('save_path', './log')
    return os.path.join(save_path, f'log_{run_id}', 'log', 'console.log')


def render_table(rows):
    headers = ['variant', 'mIoU', 'tail mIoU', 'worst class', 'worst IoU', 'status']
    body = []
    for name, result, status in rows:
        if result is None:
            body.append([name, '-', '-', '-', '-', status])
        else:
            body.append([name, f"{result['mIoU']:.4f}", f"{result['tail_mIoU']:.4f}",
                         result['worst_class'], f"{result['worst_iou']:.4f}", status])

    widths = [max(len(h), *(len(r[i]) for r in body)) if body else len(h)
              for i, h in enumerate(headers)]
    sep = '+' + '+'.join('-' * (w + 2) for w in widths) + '+'

    def line(cells):
        return '| ' + ' | '.join(c.ljust(w) for c, w in zip(cells, widths)) + ' |'

    out = [sep, line(headers), sep] + [line(r) for r in body] + [sep]

    scored = [(r[1]['mIoU'], r[0]) for r in rows if r[1] is not None]
    if len(scored) > 1:
        best_iou, best_name = max(scored)
        base_iou = dict((n, v) for v, n in scored).get(rows[0][0])
        out.append(f'\nBest mIoU: {best_name} ({best_iou:.4f})')
        if base_iou is not None and best_name != rows[0][0]:
            out.append(f'  vs {rows[0][0]}: {best_iou - base_iou:+.4f}')
    return '\n'.join(out)


def main():
    ap = argparse.ArgumentParser(description='Run and tabulate RangeDiT ablations')
    ap.add_argument('config_path', nargs='?', default='config.yaml')
    ap.add_argument('--suite', choices=sorted(SUITES), default=None,
                    help='a predefined set of variants')
    ap.add_argument('--variant', action='append', default=[],
                    help="'name' or 'name:key.path=value[,key.path=value...]'")
    ap.add_argument('--data_root', default=None)
    ap.add_argument('--save_path', default=None)
    ap.add_argument('--n_epochs', type=int, default=None,
                    help='override for all variants (use a short budget to rank them first)')
    ap.add_argument('--extra', default='',
                    help='extra args appended verbatim to every train.py call')
    ap.add_argument('--work_dir', default='./ablations',
                    help='where the generated per-variant configs are written')
    ap.add_argument('--skip_existing', action='store_true',
                    help='skip a variant that already has a validation result on disk')
    ap.add_argument('--table_only', action='store_true',
                    help='do not train, just rebuild the table from the existing logs')
    ap.add_argument('--dry_run', action='store_true', help='print the commands and exit')
    args = ap.parse_args()

    if not args.suite and not args.variant:
        ap.error('pass --suite or at least one --variant')

    base_cfg = yaml.safe_load(open(args.config_path))
    base_id = base_cfg.get('training', {}).get('id', 'rangedit')

    variants = {}
    if args.suite:
        variants.update(SUITES[args.suite])
    for spec in args.variant:
        name, _, rest = spec.partition(':')
        variants[name.strip()] = [o for o in rest.split(',') if o.strip()]

    os.makedirs(args.work_dir, exist_ok=True)
    rows = []

    for name, overrides in variants.items():
        run_id = f'{base_id}_{name}'
        log_file = log_path_for(base_cfg, run_id, args.save_path)
        existing = parse_log(log_file)

        if args.table_only:
            rows.append((name, existing, 'on disk' if existing else 'missing'))
            continue
        if args.skip_existing and existing is not None:
            print(f'== {name}: already has a result, skipping')
            rows.append((name, existing, 'cached'))
            continue

        cfg_path = build_variant_config(base_cfg, overrides, run_id, args.work_dir)
        cmd = [sys.executable, os.path.join(REPO_ROOT, 'train.py'), cfg_path]
        if args.data_root:
            cmd += ['--data_root', args.data_root]
        if args.save_path:
            cmd += ['--save_path', args.save_path]
        if args.n_epochs is not None:
            cmd += ['--n_epochs', str(args.n_epochs)]
        if args.extra:
            cmd += args.extra.split()

        print(f'\n===== {name} =====')
        print('  overrides: ' + (', '.join(overrides) if overrides else '(base config)'))
        print('  ' + ' '.join(cmd))
        if args.dry_run:
            rows.append((name, None, 'dry-run'))
            continue

        rc = subprocess.call(cmd, cwd=REPO_ROOT)
        result = parse_log(log_path_for(base_cfg, run_id, args.save_path))
        status = 'ok' if rc == 0 else f'train.py exited {rc}'
        if rc == 0 and result is None:
            status = 'no validation line in log'
        rows.append((name, result, status))

    print('\n' + render_table(rows))

    summary = os.path.join(args.work_dir, 'results.json')
    with open(summary, 'w') as fh:
        json.dump([{'variant': n, 'result': r, 'status': s} for n, r, s in rows], fh, indent=2)
    print(f'\nWrote {summary}')


if __name__ == '__main__':
    main()
