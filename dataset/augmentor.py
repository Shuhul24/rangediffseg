"""
Random 3D point-cloud augmentations applied before range-view projection.
Adapted from RangeViT (github.com/valeoai/rangevit).
"""

import numpy as np
from scipy.spatial.transform import Rotation


class Augmentor:
    def __init__(self, cfg: dict):
        """
        cfg keys (all optional, sensible defaults provided):
            p_flipx/y, p_transx/y/z, transx/y/z_range,
            p_rot_yaw/pitch/roll, rot_yaw/pitch/roll_range, p_scale, scale_range,
            p_drop_points, drop_points_range,
            p_jitter_xyz, jitter_xyz_sigma,
            p_intensity_jitter, intensity_scale_range, intensity_noise_sigma,
            p_drop_intensity
        """
        self.cfg = cfg

    def augment(self, points: np.ndarray) -> np.ndarray:
        """
        Args:
            points: (N, 4) [x, y, z, intensity]
        Returns:
            augmented points: (N, 4)
        """
        c = self.cfg

        # Random axis flips
        if np.random.random() < c.get('p_flipx', 0.5):
            points[:, 0] *= -1
        if np.random.random() < c.get('p_flipy', 0.5):
            points[:, 1] *= -1

        # Random translation per axis
        for axis, key in enumerate(['transx', 'transy', 'transz']):
            if np.random.random() < c.get(f'p_{key}', 0.5):
                lo, hi = c.get(f'{key}_range', [-1.0, 1.0])
                points[:, axis] += np.random.uniform(lo, hi)

        # Random rotation (ZYX Euler, degrees)
        angles = []
        for key in ['rot_yaw', 'rot_pitch', 'rot_roll']:
            if np.random.random() < c.get(f'p_{key}', 0.5):
                lo, hi = c.get(f'{key}_range', [-5.0, 5.0])
                angles.append(np.deg2rad(np.random.uniform(lo, hi)))
            else:
                angles.append(0.0)
        rot = Rotation.from_euler('zyx', angles).as_matrix()
        points[:, :3] = points[:, :3] @ rot.T

        # Random uniform scaling (intensity left unchanged)
        if np.random.random() < c.get('p_scale', 0.5):
            lo, hi = c.get('scale_range', [0.95, 1.05])
            points[:, :3] *= np.random.uniform(lo, hi)

        # Per-point Gaussian jitter on the coordinates. Small enough not to
        # change which pixel a point lands in for most points, but it stops the
        # stem from keying on the exact quantisation of the beam pattern.
        if np.random.random() < c.get('p_jitter_xyz', 0.0):
            sigma = c.get('jitter_xyz_sigma', 0.02)
            points[:, :3] += np.random.normal(0.0, sigma, size=points[:, :3].shape)

        # Intensity is the channel that transfers worst across sensors and even
        # across sequences of the same sensor, so it is the easiest one for the
        # network to overfit. Jittering it (and occasionally blanking it) forces
        # the geometric channels to carry the decision.
        if np.random.random() < c.get('p_intensity_jitter', 0.0):
            lo, hi = c.get('intensity_scale_range', [0.9, 1.1])
            noise = c.get('intensity_noise_sigma', 0.01)
            # Clip back into the scan's own intensity range rather than a fixed
            # [0, 1]: SemanticKITTI intensity is 0-1 but nuScenes is 0-255, and
            # a hard [0, 1] clamp would flatten the whole nuScenes channel.
            upper = float(points[:, 3].max()) if points.shape[0] else 1.0
            scale = np.random.uniform(lo, hi)
            points[:, 3] = points[:, 3] * scale
            if noise > 0:
                points[:, 3] += np.random.normal(0.0, noise * max(upper, 1e-6),
                                                 size=points.shape[0])
            np.clip(points[:, 3], 0.0, max(upper, 1e-6), out=points[:, 3])

        if np.random.random() < c.get('p_drop_intensity', 0.0):
            points[:, 3] = 0.0

        return points

    def drop_points(self, points, labels):
        """
        Random point dropout, applied *with* the labels so the two stay aligned.

        Kept separate from `augment` because it changes the number of points.
        Dropping returns simulates a sparser / partially occluded sensor and is
        one of the few augmentations that reliably helps thin structures (pole,
        traffic-sign, trunk), which the network otherwise learns to recognise
        only at the exact density of the training LiDAR.
        """
        c = self.cfg
        if np.random.random() >= c.get('p_drop_points', 0.0):
            return points, labels
        lo, hi = c.get('drop_points_range', [0.0, 0.2])
        ratio = np.random.uniform(lo, hi)
        if ratio <= 0:
            return points, labels
        n = len(points)
        keep = np.random.rand(n) >= ratio
        if keep.sum() < 100:            # never empty a scan
            return points, labels
        return points[keep], labels[keep]
