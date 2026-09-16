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

Tail-class handling
-------------------
Three things happen here, in this order, and all three exist because a plain
uniform 64 x 384 crop sees the tail classes far too rarely to learn them:

  1. `copy_paste`  -- paste real rare-class instances into the scan
                      (`instance_bank.py`)
  2. `mix`         -- LaserMix / PolarMix against a second random scan
                      (`mix.py`)
  3. `rare_crop_p` -- bias the crop window towards a tail-class pixel instead of
                      sampling it uniformly

All three are off by default, so the dataset reproduces RangeViT exactly unless
they are switched on in the config.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from .augmentor import Augmentor
from .mix import lasermix, polarmix
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
                 img_mean=None, img_stds=None,
                 instance_bank=None, copy_paste_cfg: dict = None,
                 mix_cfg: dict = None,
                 rare_crop_p: float = 0.0, rare_classes=None):
        """
        Args:
            parser        : point-cloud parser yielding (points (N,4), labels (N,))
                            -- `SemanticKITTI` or `NuScenesLidarSeg`
            proj_h/proj_w : full range-image size (height x width)
            fov_up/fov_down: vertical field-of-view bounds, in degrees
            image_size    : (H, W) random crop taken during training; None = full image
            is_train      : enables 3D augmentation and random cropping
            aug_cfg       : augmentation config dict (passed to Augmentor)
            return_points : also return the per-point back-projection indices,
                            per-point labels and the range images needed by the
                            KNN post-processing (for 3-D evaluation)
            img_mean/img_stds : per-channel normalisation statistics
            instance_bank : an `InstanceBank`, or None to disable copy-paste
            copy_paste_cfg: {'p', 'n_instances', 'occlusion_tolerance'}
            mix_cfg       : {'p_lasermix', 'p_polarmix', 'lasermix_bands',
                             'polarmix_sector_range'}
            rare_crop_p   : probability of centring the training crop on a pixel
                            of one of `rare_classes` instead of cropping uniformly
            rare_classes  : training-label ids treated as tail classes
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

        # ---- tail-class handling (training only) ----
        self.instance_bank = instance_bank if is_train else None
        self.copy_paste_cfg = copy_paste_cfg or {}
        self.mix_cfg = mix_cfg or {}
        self.rare_crop_p = float(rare_crop_p) if is_train else 0.0
        self.rare_classes = np.asarray(
            rare_classes if rare_classes is not None else [], dtype=np.int64)

    def __len__(self):
        return len(self.parser)

    # ------------------------------------------------------------------ #
    #                              cropping                              #
    # ------------------------------------------------------------------ #

    def _random_crop(self, feat, labels_2d, mask):
        """Random crop of the range image, as done by RangeViT during training."""
        H, W = labels_2d.shape
        crop_h, crop_w = self.image_size
        crop_h, crop_w = min(crop_h, H), min(crop_w, W)

        top, left = None, None

        # With probability `rare_crop_p`, centre the window on a randomly chosen
        # tail-class pixel. A uniform 64 x 384 window of a 64 x 2048 image holds
        # ~1 bicycle pixel and ~0.3 motorcyclist pixels on average against ~2000
        # vegetation pixels; no loss weighting closes a gap that large, but
        # steering where the window lands does.
        if self.rare_crop_p > 0 and self.rare_classes.size and \
                np.random.random() < self.rare_crop_p:
            hits = np.nonzero(np.isin(labels_2d, self.rare_classes))
            if hits[0].size:
                k = np.random.randint(hits[0].size)
                cy, cx = int(hits[0][k]), int(hits[1][k])
                # Jitter the window around the hit so the object is not always
                # pinned to the centre of the crop.
                top = cy - crop_h // 2 + np.random.randint(-crop_h // 4, crop_h // 4 + 1)
                left = cx - crop_w // 2 + np.random.randint(-crop_w // 4, crop_w // 4 + 1)
                top = int(np.clip(top, 0, H - crop_h))
                left = int(np.clip(left, 0, W - crop_w))

        if top is None:
            top = np.random.randint(0, H - crop_h + 1)
            left = np.random.randint(0, W - crop_w + 1)

        return (feat[top:top + crop_h, left:left + crop_w],
                labels_2d[top:top + crop_h, left:left + crop_w],
                mask[top:top + crop_h, left:left + crop_w])

    # ------------------------------------------------------------------ #
    #                      point-level augmentation                      #
    # ------------------------------------------------------------------ #

    def _augment_points(self, points, point_labels):
        """Copy-paste, scan mixing, geometric augmentation and point dropout."""
        cfg = self.mix_cfg

        # -- Scan mixing against a second random scan. Done before the
        #    geometric augmentation so both halves get the same global
        #    transform, which keeps the mixed scan self-consistent.
        p_laser = cfg.get('p_lasermix', 0.0)
        p_polar = cfg.get('p_polarmix', 0.0)
        if (p_laser > 0 or p_polar > 0) and len(self.parser) > 1:
            r = np.random.random()
            if r < p_laser + p_polar:
                j = np.random.randint(len(self.parser))
                pts_b, lbl_b = self.parser[j]
                pts_b = np.asarray(pts_b, dtype=np.float32)
                lbl_b = np.asarray(lbl_b)
                if r < p_laser:
                    points, point_labels = lasermix(
                        points, point_labels, pts_b, lbl_b,
                        n_bands=cfg.get('lasermix_bands', (3, 4, 5, 6)),
                        fov_up=np.rad2deg(self.projector.fov_v + self.projector.fov_down),
                        fov_down=np.rad2deg(self.projector.fov_down))
                else:
                    points, point_labels = polarmix(
                        points, point_labels, pts_b, lbl_b,
                        sector_range=tuple(cfg.get('polarmix_sector_range',
                                                   (np.pi / 4, np.pi))))

        # -- Paste rare-class instances. After mixing, so the pasted instances
        #    are never cut away by a sector swap.
        if self.instance_bank is not None:
            cp = self.copy_paste_cfg
            if np.random.random() < cp.get('p', 1.0):
                points, point_labels = self.instance_bank.paste(
                    points, point_labels,
                    n_instances=tuple(cp.get('n_instances', (1, 4))),
                    occlusion_tolerance=cp.get('occlusion_tolerance', 0.5),
                    projector=self.projector)

        # -- Geometric / intensity augmentation, then point dropout.
        points = self.augmentor.augment(np.ascontiguousarray(points))
        points, point_labels = self.augmentor.drop_points(points, point_labels)
        return points, point_labels

    # ------------------------------------------------------------------ #

    def __getitem__(self, idx):
        """
        Returns:
            feature : (5, H, W) float32 normalised range-view features
            label   : (H, W)    int64   per-pixel class indices (ground truth)
            mask    : (H, W)    float32 valid-pixel mask (1=valid, 0=empty)
        and, when `return_points` is set, additionally:
            px, py        : (N,) int64   pixel each 3-D point projects onto
            point_labels  : (N,) int64   per-point ground-truth labels
            proj_range    : (H, W) float32 projected depth (-1 where empty)
            unproj_range  : (N,) float32 true depth of every point
        """
        points, point_labels = self.parser[idx]
        points = np.asarray(points, dtype=np.float32)
        point_labels = np.asarray(point_labels)

        if self.augmentor is not None:
            points, point_labels = self._augment_points(points.copy(), point_labels.copy())

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

        unproj_range = np.linalg.norm(points[:, :3], axis=1).astype(np.float32)
        return (feature, label, mask,
                torch.from_numpy(px), torch.from_numpy(py),
                torch.from_numpy(point_labels.astype(np.int64)),
                torch.from_numpy(np.ascontiguousarray(proj_range)),
                torch.from_numpy(unproj_range))
