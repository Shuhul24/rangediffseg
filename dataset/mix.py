"""
Scan-level mixing augmentations: LaserMix and PolarMix.

Both take two scans and build a third by swapping regions between them. They
are cheap (pure numpy on the raw point cloud, before projection), they need no
instance annotations -- so unlike `instance_bank.py` they also work on nuScenes
-- and they attack the failure mode that hurts tail classes most: a rare object
only ever being seen in the one context it was recorded in.

LaserMix  (Kong et al., CVPR 2023) -- swap *inclination* bands, i.e. groups of
          laser beams. Respects the way a spinning LiDAR samples the world, so
          the mixed scan still looks like a plausible scan.
PolarMix  (Xiao et al., NeurIPS 2022) -- swap *azimuth* sectors.

On a range image LaserMix is a row swap and PolarMix a column swap, but doing
them in 3-D keeps them compatible with the geometric augmentations that run
before the projection.
"""

import numpy as np


def _pitch(points):
    depth = np.linalg.norm(points[:, :3], axis=1)
    return np.arcsin(np.clip(points[:, 2] / np.maximum(depth, 1e-8), -1.0, 1.0))


def _azimuth(points):
    return np.arctan2(points[:, 1], points[:, 0])          # [-pi, pi]


def lasermix(points_a, labels_a, points_b, labels_b,
             n_bands=(3, 4, 5, 6), fov_up=3.0, fov_down=-25.0, rng=np.random):
    """
    Split the vertical field of view into `n_bands` inclination bands and take
    the odd bands from scan A and the even bands from scan B.
    """
    k = n_bands[rng.randint(len(n_bands))] if isinstance(n_bands, (list, tuple)) else int(n_bands)
    lo, hi = np.deg2rad(fov_down), np.deg2rad(fov_up)
    edges = np.linspace(lo, hi, k + 1)

    band_a = np.digitize(_pitch(points_a), edges[1:-1])
    band_b = np.digitize(_pitch(points_b), edges[1:-1])

    keep_a = (band_a % 2) == 0
    keep_b = (band_b % 2) == 1

    return (np.concatenate([points_a[keep_a], points_b[keep_b]], axis=0),
            np.concatenate([labels_a[keep_a], labels_b[keep_b]], axis=0))


def polarmix(points_a, labels_a, points_b, labels_b,
             sector_range=(np.pi / 4, np.pi), rng=np.random):
    """
    Cut one azimuth sector out of scan A and fill it with the same sector of
    scan B (which is first rotated by a random yaw, so the pasted content does
    not have to come from the same heading).
    """
    width = rng.uniform(*sector_range)
    start = rng.uniform(-np.pi, np.pi)
    end = start + width

    def in_sector(az):
        shifted = np.mod(az - start, 2 * np.pi)
        return shifted < width

    yaw = rng.uniform(-np.pi, np.pi)
    c, s = np.cos(yaw), np.sin(yaw)
    pts_b = points_b.copy()
    x, y = points_b[:, 0], points_b[:, 1]
    pts_b[:, 0] = c * x - s * y
    pts_b[:, 1] = s * x + c * y

    keep_a = ~in_sector(_azimuth(points_a))
    keep_b = in_sector(_azimuth(pts_b))

    del end
    return (np.concatenate([points_a[keep_a], pts_b[keep_b]], axis=0),
            np.concatenate([labels_a[keep_a], labels_b[keep_b]], axis=0))
