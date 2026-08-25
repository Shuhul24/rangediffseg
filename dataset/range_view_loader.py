"""
PyTorch Dataset that wraps a point-cloud parser (SemanticKITTI or nuScenes):
loads point clouds, applies 3D augmentation, projects them to range-view images
and returns the tensors the segmentation network is trained on.
Adapted from RangeViT (github.com/valeoai/rangevit).

Range image channels: [range, x, y, z, intensity]  (5 channels)

Training samples are random crops of the full range image (RangeViT trains on
64 x 384 windows of a 64 x 2048 SemanticKITTI image, 32 x 384 windows of a
32 x 2048 nuScenes image); at evaluation time the full-width image is returned
and the model is applied with a sliding window.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from .augmentor import Augmentor
from .projection import RangeProjection

# Per-channel mean/std used by RangeViT for both SemanticKITTI and nuScenes
# (range, x, y, z, intensity). Override them through `dataset.img_mean` /
# `dataset.img_stds` in the config file.
DEFAULT_MEAN = [12.12, 10.88, 0.23, -1.04, 0.21]
DEFAULT_STD = [12.32, 11.47, 6.91, 0.86, 0.16]


class RangeViewDataset(Dataset):
    def __init__(self, parser,
                 proj_h: int = 64, proj_w: int = 2048,
                 fov_up: float = 3.0, fov_down: float = -25.0,
                 image_size=None, is_train: bool = True, aug_cfg: dict = None,
                 return_points: bool = False,
                 img_mean=None, img_stds=None):
        """
        Args:
            parser        : point-cloud parser yielding (points (N,4), labels (N,))
                            -- `SemanticKITTI` or `NuScenesLidarSeg`
            proj_h/proj_w : full range-image size (height x width)
            fov_up/fov_down: vertical field-of-view bounds, in degrees
            image_size    : (H, W) random crop taken during training; None = full image
            is_train      : enables 3D augmentation and random cropping
            aug_cfg       : augmentation config dict (passed to Augmentor)
            return_points : also return the per-point back-projection indices
                            and per-point labels (for 3-D evaluation)
            img_mean/img_stds : per-channel normalisation statistics
        """
        self.parser = parser
        self.projector = RangeProjection(proj_h, proj_w, fov_up, fov_down)
        self.num_cls = parser.NUM_CLASSES
        self.augmentor = Augmentor(aug_cfg or {}) if is_train else None
        self.is_train = is_train
        self.image_size = tuple(image_size) if image_size is not None else None
        self.return_points = return_points

        self.mean = np.array(img_mean if img_mean is not None else DEFAULT_MEAN,
                             dtype=np.float32)
        self.std = np.array(img_stds if img_stds is not None else DEFAULT_STD,
                            dtype=np.float32)

        self.mapped_cls_name = parser.mapped_cls_name
        self.cls_freq = parser.cls_freq

    def __len__(self):
        return len(self.parser)

    def _random_crop(self, feat, labels_2d, mask):
        """Random crop of the range image, as done by RangeViT during training."""
        H, W = labels_2d.shape
        crop_h, crop_w = self.image_size
        crop_h, crop_w = min(crop_h, H), min(crop_w, W)

        top = np.random.randint(0, H - crop_h + 1)
        left = np.random.randint(0, W - crop_w + 1)

        return (feat[top:top + crop_h, left:left + crop_w],
                labels_2d[top:top + crop_h, left:left + crop_w],
                mask[top:top + crop_h, left:left + crop_w])

    def __getitem__(self, idx):
        """
        Returns:
            feature : (5, H, W) float32 normalised range-view features
            label   : (H, W)    int64   per-pixel class indices (ground truth)
            mask    : (H, W)    float32 valid-pixel mask (1=valid, 0=empty)
        and, when `return_points` is set, additionally:
            px, py       : (N,) int64 pixel each 3-D point projects onto
            point_labels : (N,) int64 per-point ground-truth labels
        """
        points, point_labels = self.parser[idx]

        if self.augmentor is not None:
            points = self.augmentor.augment(points.copy())

        proj_range, proj_feat, proj_idx, proj_mask, px, py = self.projector.project(points)
        H, W = proj_range.shape

        # Build the 2-D label map by scattering point labels via projection indices
        labels_2d = np.zeros((H, W), dtype=np.int64)
        valid = proj_idx >= 0
        labels_2d[valid] = point_labels[proj_idx[valid]]

        # Normalise the range-image features; zero out empty pixels
        feat = (proj_feat - self.mean) / self.std
        feat[~proj_mask] = 0.0
        mask = proj_mask.astype(np.float32)

        if self.is_train and self.image_size is not None:
            feat, labels_2d, mask = self._random_crop(feat, labels_2d, mask)

        feature = torch.from_numpy(np.ascontiguousarray(feat.transpose(2, 0, 1)))  # HWC -> CHW
        label = torch.from_numpy(np.ascontiguousarray(labels_2d))
        mask = torch.from_numpy(np.ascontiguousarray(mask))

        if not self.return_points:
            return feature, label, mask

        return (feature, label, mask,
                torch.from_numpy(px), torch.from_numpy(py),
                torch.from_numpy(point_labels.astype(np.int64)))
