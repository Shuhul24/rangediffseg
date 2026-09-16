"""
Rare-class instance bank: "copy-paste" augmentation for range-view LiDAR
segmentation.

Why this exists
---------------
A 64 x 384 training crop covers 18.75 % of a 64 x 2048 SemanticKITTI range
image. With the official point frequencies that crop contains, on average,
~2160 vegetation pixels but ~1.3 bicycle pixels and ~0.3 motorcyclist pixels.
No loss re-weighting closes a 2000:1 gap -- the rare classes simply are not in
the batch often enough to produce gradient. Pasting real instances of those
classes into every scan is the standard fix (Cylinder3D, PolarMix, RangeFormer)
and is worth far more on the tail classes than any change to the loss.

SemanticKITTI `.label` files carry a panoptic instance id in the upper 16 bits,
so instances can be cut out exactly, with no clustering. nuScenes-lidarseg does
not ship instance ids, so the bank is SemanticKITTI-only; use `mix.py`
(LaserMix / PolarMix) there instead.

Build the bank once:

    python -m dataset.instance_bank --data_root /path/to/SemanticKITTI \
        --out ./cache/instance_bank_kitti.npz

then point `dataset.copy_paste.bank_path` at the resulting file.
"""

import argparse
import os

import numpy as np

# SemanticKITTI training-label ids that carry a panoptic instance id ("things").
# Class 1 (car) is a thing too but is common enough that pasting it mostly adds
# noise, so the default bank covers the seven rare thing classes.
KITTI_THING_CLASSES = (1, 2, 3, 4, 5, 6, 7, 8)
KITTI_RARE_THING_CLASSES = (2, 3, 4, 5, 6, 7, 8)


class InstanceBank:
    """
    Holds cut-out instances and pastes them into a scan before projection.

    The bank stores every instance in the sensor frame. Pasting rotates an
    instance around the sensor's z axis by a random azimuth, which keeps its
    range and elevation -- and therefore its point density in the range image --
    physically plausible. A candidate position is rejected when the instance
    would land *behind* existing scene geometry (inside a wall), which is the
    one failure mode that produces labels the network can never satisfy.
    """

    def __init__(self, bank_path, classes=None, rng=None):
        if not os.path.isfile(bank_path):
            raise FileNotFoundError(
                f'instance bank not found: {bank_path}\n'
                f'Build it with:  python -m dataset.instance_bank '
                f'--data_root <SemanticKITTI root> --out {bank_path}')

        data = np.load(bank_path)
        self.points = data['points'].astype(np.float32)     # (P, 4) x,y,z,intensity
        self.offsets = data['offsets'].astype(np.int64)     # (K + 1,)
        self.inst_class = data['inst_class'].astype(np.int64)  # (K,)
        self.rng = rng if rng is not None else np.random

        wanted = set(classes) if classes is not None else set(self.inst_class.tolist())
        # Group instance indices by class so sampling can be class-balanced
        # rather than frequency-weighted -- otherwise the bank reproduces the
        # very imbalance it is meant to fix.
        self.by_class = {}
        for c in sorted(wanted):
            idx = np.nonzero(self.inst_class == c)[0]
            if idx.size:
                self.by_class[int(c)] = idx
        if not self.by_class:
            raise ValueError(f'instance bank {bank_path} holds none of the classes {sorted(wanted)}')
        self.classes = sorted(self.by_class.keys())

    def __len__(self):
        return len(self.inst_class)

    def summary(self):
        return {c: len(idx) for c, idx in self.by_class.items()}

    def _get(self, k):
        lo, hi = self.offsets[k], self.offsets[k + 1]
        return self.points[lo:hi].copy()

    def sample_instance(self):
        """One instance, class-balanced: (points (M, 4), class id)."""
        c = self.classes[self.rng.randint(len(self.classes))]
        pool = self.by_class[c]
        return self._get(pool[self.rng.randint(len(pool))]), c

    def paste(self, points, labels, n_instances=(1, 4), max_tries=4,
              occlusion_tolerance=0.5, projector=None):
        """
        Args:
            points   : (N, 4) scan points [x, y, z, intensity]
            labels   : (N,)   per-point training labels
            n_instances : (lo, hi) inclusive range of instances to paste
            max_tries: azimuths tried per instance before giving up
            occlusion_tolerance : reject a placement when more than this
                       fraction of the instance's pixels are already occupied by
                       *closer* scene points (i.e. the instance is buried)
            projector: a `RangeProjection`; when given, it supplies the pixel
                       grid used by the occlusion test. Without it every
                       placement is accepted.
        Returns:
            (points, labels) with the pasted instances appended
        """
        lo, hi = n_instances
        n = self.rng.randint(lo, hi + 1) if hi > lo else lo
        if n <= 0:
            return points, labels

        new_pts, new_lbls = [], []

        scene_range = None
        if projector is not None:
            scene_range = _range_image(projector, points)

        for _ in range(n):
            inst, cls = self.sample_instance()
            for _try in range(max_tries):
                yaw = self.rng.uniform(-np.pi, np.pi)
                cand = _rotate_z(inst, yaw)
                if scene_range is None or self._fits(cand, scene_range, projector,
                                                     occlusion_tolerance):
                    new_pts.append(cand)
                    new_lbls.append(np.full(len(cand), cls, dtype=labels.dtype))
                    if scene_range is not None:
                        _blit_range(projector, scene_range, cand)
                    break

        if not new_pts:
            return points, labels
        return (np.concatenate([points] + new_pts, axis=0),
                np.concatenate([labels] + new_lbls, axis=0))

    @staticmethod
    def _fits(cand, scene_range, projector, tolerance):
        py, px, depth = _pixel_of(projector, cand)
        occupied = scene_range[py, px]
        # A pixel is "blocked" when the scene already has a *closer* return
        # there: the pasted object would be hidden behind a wall or a car.
        blocked = (occupied > 0) & (occupied < depth - 0.3)
        return blocked.mean() <= tolerance


# --------------------------------------------------------------------------- #
#                                  helpers                                    #
# --------------------------------------------------------------------------- #

def _rotate_z(pts, yaw):
    out = pts.copy()
    c, s = np.cos(yaw), np.sin(yaw)
    x, y = out[:, 0].copy(), out[:, 1].copy()
    out[:, 0] = c * x - s * y
    out[:, 1] = s * x + c * y
    return out


def _pixel_of(projector, pts):
    depth = np.linalg.norm(pts[:, :3], axis=1)
    yaw = -np.arctan2(pts[:, 1], pts[:, 0])
    pitch = np.arcsin(np.clip(pts[:, 2] / np.maximum(depth, 1e-8), -1.0, 1.0))
    u = yaw / (2 * np.pi) + 0.5
    v = 1.0 - (pitch + abs(projector.fov_down)) / projector.fov_v
    px = np.floor(u * projector.W).clip(0, projector.W - 1).astype(np.int64)
    py = np.floor(v * projector.H).clip(0, projector.H - 1).astype(np.int64)
    return py, px, depth


def _range_image(projector, pts):
    py, px, depth = _pixel_of(projector, pts)
    img = np.zeros((projector.H, projector.W), dtype=np.float32)
    order = np.argsort(-depth)
    img[py[order], px[order]] = depth[order]
    return img


def _blit_range(projector, img, pts):
    """Write a pasted instance into the occupancy image so later pastes see it."""
    py, px, depth = _pixel_of(projector, pts)
    order = np.argsort(-depth)
    img[py[order], px[order]] = depth[order]


# --------------------------------------------------------------------------- #
#                                 bank builder                                #
# --------------------------------------------------------------------------- #

def build_bank(data_root, sequences, out_path, classes=KITTI_RARE_THING_CLASSES,
               min_points=20, max_per_class=6000, stride=1, seed=1):
    """One pass over the training split, cutting out every rare-class instance."""
    from .semantic_kitti import SemanticKITTI

    rng = np.random.RandomState(seed)
    parser = SemanticKITTI(data_root, sequences, has_label=True)
    files = parser.files[::stride]
    classes = tuple(int(c) for c in classes)

    # Reservoir sampling per class keeps the bank bounded and unbiased over the
    # whole split rather than front-loaded on the first sequences.
    kept = {c: [] for c in classes}
    seen = {c: 0 for c in classes}

    print(f'Scanning {len(files)} scans from sequences {sequences} ...')
    for i, entry in enumerate(files):
        if not os.path.exists(entry['label']):
            continue
        raw = np.fromfile(entry['label'], dtype=np.int32)
        sem = parser.lut[raw & 0xFFFF]
        inst = (raw.astype(np.uint32) >> 16).astype(np.int64)

        wanted = np.isin(sem, classes)
        if not wanted.any():
            continue
        pts = np.fromfile(entry['scan'], dtype=np.float32).reshape(-1, 4)
        if len(pts) != len(sem):
            continue

        # (class, instance) uniquely identifies an object within a scan.
        key = sem.astype(np.int64) * (1 << 20) + inst
        for k in np.unique(key[wanted]):
            sel = np.nonzero(key == k)[0]
            if len(sel) < min_points:
                continue
            c = int(k >> 20)
            seen[c] += 1
            if len(kept[c]) < max_per_class:
                kept[c].append(pts[sel].copy())
            else:
                j = rng.randint(seen[c])
                if j < max_per_class:
                    kept[c][j] = pts[sel].copy()

        if (i + 1) % 500 == 0:
            print(f'  [{i + 1}/{len(files)}] ' +
                  ' '.join(f'{c}:{len(kept[c])}' for c in classes))

    chunks, inst_class = [], []
    for c in classes:
        chunks.extend(kept[c])
        inst_class.extend([c] * len(kept[c]))
    if not chunks:
        raise RuntimeError('no instances found -- check the dataset root and the label files')

    offsets = np.zeros(len(chunks) + 1, dtype=np.int64)
    np.cumsum([len(x) for x in chunks], out=offsets[1:])

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or '.', exist_ok=True)
    np.savez_compressed(out_path,
                        points=np.concatenate(chunks, axis=0).astype(np.float32),
                        offsets=offsets,
                        inst_class=np.asarray(inst_class, dtype=np.int64))

    from .semantic_kitti import SemanticKITTI as SK
    print(f'\nWrote {out_path}')
    for c in classes:
        print(f'  {SK.CLASS_NAMES[c]:>15}: {len(kept[c]):6d} instances '
              f'(of {seen[c]} seen)')


def _cli():
    p = argparse.ArgumentParser(description='Build the rare-class instance bank')
    p.add_argument('--data_root', required=True)
    p.add_argument('--out', default='./cache/instance_bank_kitti.npz')
    p.add_argument('--sequences', nargs='+',
                   default=['00', '01', '02', '03', '04', '05', '06', '07', '09', '10'])
    p.add_argument('--classes', nargs='+', type=int, default=list(KITTI_RARE_THING_CLASSES))
    p.add_argument('--min_points', type=int, default=20)
    p.add_argument('--max_per_class', type=int, default=6000)
    p.add_argument('--stride', type=int, default=1,
                   help='scan every Nth frame (consecutive frames are near-duplicates, '
                        'so 2-3 costs little and is much faster)')
    p.add_argument('--seed', type=int, default=1)
    a = p.parse_args()
    build_bank(a.data_root, a.sequences, a.out, classes=tuple(a.classes),
               min_points=a.min_points, max_per_class=a.max_per_class,
               stride=a.stride, seed=a.seed)


if __name__ == '__main__':
    _cli()
