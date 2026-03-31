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
            p_rot_yaw/pitch/roll, rot_yaw/pitch/roll_range, p_scale, scale_range
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

        return points
