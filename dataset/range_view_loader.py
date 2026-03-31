"""
PyTorch Dataset that wraps SemanticKITTI: loads point clouds, applies 3D
augmentation, projects to range-view images, and returns tensors ready for
DDPM training.  Adapted from RangeViT (github.com/valeoai/rangevit).

Range image channels: [range, x, y, z, intensity]  (5 channels)
Segmentation tensor  : one-hot over NUM_CLASSES, scaled to [-1, 1]
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from .semantic_kitti import SemanticKITTI
from .projection    import RangeProjection
from .augmentor     import Augmentor


# Per-channel mean/std for SemanticKITTI (range, x, y, z, intensity)
_MEAN = np.array([12.12, 10.88,  0.23, -1.04, 0.21], dtype=np.float32)
_STD  = np.array([12.32, 11.47,  6.91,  0.86, 0.16], dtype=np.float32)


class RangeViewDataset(Dataset):
    def __init__(self, root: str, sequences: list,
                 proj_h: int = 64, proj_w: int = 512,
                 is_train: bool = True, aug_cfg: dict = None):
        """
        Args:
            root       : SemanticKITTI dataset root
            sequences  : sequence IDs to load
            proj_h/w   : range image size (height × width)
            is_train   : enables 3D augmentation when True
            aug_cfg    : augmentation config dict (passed to Augmentor)
        """
        self.parser    = SemanticKITTI(root, sequences)
        self.projector = RangeProjection(proj_h, proj_w)
        self.num_cls   = SemanticKITTI.NUM_CLASSES
        self.augmentor = Augmentor(aug_cfg or {}) if is_train else None

    def __len__(self):
        return len(self.parser)

    def __getitem__(self, idx):
        """
        Returns:
            seg    : (C, H, W) float32 one-hot segmentation in [-1, 1],  C=NUM_CLASSES
            img    : (5, H, W) float32 normalised range-view features
            mask   : (H, W)    float32 valid-pixel mask (1=valid, 0=empty)
            labels : (H, W)    int64   per-pixel class indices (ground truth)
            uproj  : dict with 'x', 'y' arrays for back-projection to 3-D
        """
        points, point_labels = self.parser[idx]

        if self.augmentor is not None:
            points = self.augmentor.augment(points.copy())

        proj_range, proj_feat, proj_idx, proj_mask = self.projector.project(points)
        H, W = proj_range.shape

        # Build 2-D label map by scattering point labels via projection indices
        labels_2d = np.zeros((H, W), dtype=np.int64)
        valid = proj_idx >= 0
        labels_2d[valid] = point_labels[proj_idx[valid]]

        # Normalise range-image features; zero out empty pixels → (5, H, W)
        feat = (proj_feat - _MEAN) / _STD
        feat[~proj_mask] = 0.0
        img = torch.from_numpy(feat.transpose(2, 0, 1))  # HWC → CHW

        # One-hot encode labels, scale [0,1] → [-1,1] for diffusion
        one_hot = np.zeros((self.num_cls, H, W), dtype=np.float32)
        for c in range(self.num_cls):
            one_hot[c] = (labels_2d == c).astype(np.float32)
        seg = torch.from_numpy(one_hot * 2.0 - 1.0)

        mask = torch.from_numpy(proj_mask.astype(np.float32))

        # Back-projection indices: point i maps to pixel (uproj_x[i], uproj_y[i])
        uproj = {
            'x': self.projector.cached['uproj_x'].copy(),
            'y': self.projector.cached['uproj_y'].copy(),
        }

        return seg, img, mask, torch.from_numpy(labels_2d), uproj
