"""
Range-view projection of 3D LiDAR point clouds.
Adapted from RangeViT (github.com/valeoai/rangevit).

Spherical projection maps each 3D point (x,y,z) to a 2D pixel (u,v)
via yaw/pitch angles, producing a dense H x W range image.
"""

import numpy as np


# Channel layouts for `proj_feat`. 'rxyzi' is the original layout and stays the
# default so existing models are bit-for-bit unaffected; 'xyzri' is what the
# RangeFormer-DiT branch consumes.
CHANNEL_ORDERS = {
    'rxyzi': ('range', 'x', 'y', 'z', 'intensity'),
    'xyzri': ('x', 'y', 'z', 'range', 'intensity'),
}


class RangeProjection:
    def __init__(self, proj_h=64, proj_w=2048, fov_up=3.0, fov_down=-25.0,
                 channel_order='rxyzi'):
        """
        Args:
            proj_h / proj_w : range image dimensions (64 x 2048 for SemanticKITTI)
            fov_up / fov_down: vertical field-of-view bounds in degrees
            channel_order   : 'rxyzi' (default, unchanged) or 'xyzri'
        """
        if channel_order not in CHANNEL_ORDERS:
            raise ValueError(f'unknown channel_order {channel_order!r}; '
                             f'expected one of {sorted(CHANNEL_ORDERS)}')
        self.channel_order = channel_order
        self.channel_names = CHANNEL_ORDERS[channel_order]
        self.H, self.W = proj_h, proj_w
        self.fov_down = np.deg2rad(fov_down)
        self.fov_v = np.deg2rad(fov_up) + abs(self.fov_down)

    def project(self, points):
        """
        Args:
            points: (N, 4) float32 array [x, y, z, intensity]
        Returns:
            proj_range : (H, W)    depth image (-1 for empty pixels)
            proj_feat  : (H, W, 5) channels in `self.channel_order`
                         'rxyzi' -> [range, x, y, z, intensity]  (default)
                         'xyzri' -> [x, y, z, range, intensity]
            proj_idx   : (H, W)    index of point occupying each pixel (-1 = empty)
            proj_mask  : (H, W)    bool, True for occupied pixels
            px, py     : (N,)      pixel each point was projected onto
        """
        H, W = self.H, self.W
        depth = np.linalg.norm(points[:, :3], axis=1)

        # Spherical angles -> normalised [0,1] image coordinates
        yaw = -np.arctan2(points[:, 1], points[:, 0])
        pitch = np.arcsin(np.clip(points[:, 2] / np.maximum(depth, 1e-8), -1.0, 1.0))

        u = (yaw / (2 * np.pi) + 0.5)                        # horizontal [0,1]
        v = 1.0 - (pitch + abs(self.fov_down)) / self.fov_v   # vertical   [0,1]

        px = np.floor(u * W).clip(0, W - 1).astype(np.int64)
        py = np.floor(v * H).clip(0, H - 1).astype(np.int64)

        # Sort by decreasing depth so closer points overwrite farther ones
        order = np.argsort(-depth)

        proj_range = np.full((H, W), -1.0, dtype=np.float32)
        proj_feat = np.zeros((H, W, 5), dtype=np.float32)
        proj_idx = np.full((H, W), -1, dtype=np.int32)

        proj_range[py[order], px[order]] = depth[order]
        if self.channel_order == 'rxyzi':
            proj_feat[py[order], px[order], 0] = depth[order]
            proj_feat[py[order], px[order], 1:4] = points[order, :3]
        else:                                   # 'xyzri': [x, y, z, range, intensity]
            proj_feat[py[order], px[order], 0:3] = points[order, :3]
            proj_feat[py[order], px[order], 3] = depth[order]
        proj_feat[py[order], px[order], 4] = points[order, 3]
        proj_idx[py[order], px[order]] = order

        return proj_range, proj_feat, proj_idx, proj_range > 0, px, py
