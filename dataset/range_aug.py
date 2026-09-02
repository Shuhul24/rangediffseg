"""
RangeAug: range-view-space augmentations.

From RangeFormer (Kong et al., "Rethinking Range View Representation for LiDAR
Segmentation", ICCV 2023, arXiv:2303.05367). The augmentations already used by
RangeViT -- global flips, translations, rotations, scaling -- are rigid 3-D
transforms: they move the whole scene as one body, so the projected image
always shows the same objects at the same density with the same class balance.
RangeFormer calls those the "common" augmentations and adds four operations
that act on the rasterised range image itself, using a second, randomly
sampled scan as the donor.

Operations and their paper defaults (probabilities [0.9, 0.2, 0.9, 1.0]):

    RangeMix    p=0.9   swap bands of rows (equal inclination ranges) between
                        two scans; k_mix drawn from [2, 3, 4, 5, 6]
    RangeUnion  p=0.2   fill empty grids of one scan from the other; 50% of
                        the empty grids are candidates
    RangePaste  p=0.9   copy tail-class pixels from the donor at their
                        corresponding range-image positions
    RangeShift  p=1.0   roll the image along azimuth by k_shift in [W/4, 3W/4]

All four operate on a triplet (feat H x W x C, label H x W, mask H x W) and
are applied before the random crop.
"""

import numpy as np

# SemanticKITTI tail classes in the 19-class learning mapping. Used by
# RangePaste when the parser does not supply class frequencies.
DEFAULT_TAIL_CLASSES = (2, 3, 4, 5, 6, 7, 8, 12, 18, 19)


def range_mix(a, b, k_choices=(2, 3, 4, 5, 6), rng=np.random):
    """Swap equal-inclination bands of rows between two scans."""
    feat_a, lab_a, mask_a = a
    feat_b, lab_b, mask_b = b
    H = lab_a.shape[0]

    k = int(rng.choice(k_choices))
    edges = np.linspace(0, H, k + 1).astype(int)

    feat, lab, mask = feat_a.copy(), lab_a.copy(), mask_a.copy()
    for i in range(k):
        # Swap a random subset of the bands, so the mix differs every call.
        if rng.random() < 0.5:
            continue
        lo, hi = edges[i], edges[i + 1]
        feat[lo:hi] = feat_b[lo:hi]
        lab[lo:hi] = lab_b[lo:hi]
        mask[lo:hi] = mask_b[lo:hi]
    return feat, lab, mask


def range_union(a, b, k_union=0.5, rng=np.random):
    """Fill empty grids of `a` with the corresponding grids of `b`."""
    feat_a, lab_a, mask_a = a
    feat_b, lab_b, mask_b = b

    # Candidates: empty in a, occupied in b.
    empty = (mask_a <= 0) & (mask_b > 0)
    if not empty.any():
        return feat_a, lab_a, mask_a

    idx = np.flatnonzero(empty.reshape(-1))
    n_take = int(round(k_union * idx.size))
    if n_take <= 0:
        return feat_a, lab_a, mask_a
    take = rng.choice(idx, size=n_take, replace=False)

    fill = np.zeros(empty.size, dtype=bool)
    fill[take] = True
    fill = fill.reshape(empty.shape)

    feat, lab, mask = feat_a.copy(), lab_a.copy(), mask_a.copy()
    feat[fill] = feat_b[fill]
    lab[fill] = lab_b[fill]
    mask[fill] = mask_b[fill]
    return feat, lab, mask


def range_paste(a, b, tail_classes=DEFAULT_TAIL_CLASSES):
    """Copy the donor's tail-class pixels into `a`, keeping their positions."""
    feat_a, lab_a, mask_a = a
    feat_b, lab_b, mask_b = b

    paste = np.isin(lab_b, tail_classes) & (mask_b > 0)
    if not paste.any():
        return feat_a, lab_a, mask_a

    feat, lab, mask = feat_a.copy(), lab_a.copy(), mask_a.copy()
    feat[paste] = feat_b[paste]
    lab[paste] = lab_b[paste]
    mask[paste] = mask_b[paste]
    return feat, lab, mask


def range_shift(a, rng=np.random):
    """Roll the range image along azimuth by k in [W/4, 3W/4]."""
    feat, lab, mask = a
    W = lab.shape[1]
    k = int(rng.randint(W // 4, (3 * W) // 4 + 1))
    return (np.roll(feat, k, axis=1),
            np.roll(lab, k, axis=1),
            np.roll(mask, k, axis=1))


class RangeAugmentor:
    """Applies the four RangeAug operations in the order used by the paper."""

    def __init__(self, cfg=None, tail_classes=None):
        cfg = cfg or {}
        self.p_mix = cfg.get('p_range_mix', 0.9)
        self.p_union = cfg.get('p_range_union', 0.2)
        self.p_paste = cfg.get('p_range_paste', 0.9)
        self.p_shift = cfg.get('p_range_shift', 1.0)
        self.k_mix = tuple(cfg.get('range_mix_k', (2, 3, 4, 5, 6)))
        self.k_union = cfg.get('range_union_ratio', 0.5)
        self.tail_classes = tuple(tail_classes if tail_classes is not None
                                  else DEFAULT_TAIL_CLASSES)

    @property
    def needs_donor(self):
        return max(self.p_mix, self.p_union, self.p_paste) > 0

    def __call__(self, a, b=None, rng=np.random):
        if b is not None:
            if rng.random() < self.p_mix:
                a = range_mix(a, b, self.k_mix, rng)
            if rng.random() < self.p_paste:
                a = range_paste(a, b, self.tail_classes)
            if rng.random() < self.p_union:
                a = range_union(a, b, self.k_union, rng)
        if rng.random() < self.p_shift:
            a = range_shift(a, rng)
        return a
