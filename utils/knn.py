"""
KNN post-processing for range-view LiDAR segmentation (RangeNet++,
Milioto et al., IROS 2019 -- github.com/PRBonn/lidar-bonnetal).

Why it matters here
-------------------
The spherical projection is lossy: several 3-D points fall on the same pixel and
only the closest one survives (`projection.py` sorts by decreasing depth). The
back-projection in `inference.py` gives every one of those points the label of
the pixel, so every occluded point silently inherits the label of whatever was
in front of it. On a 64 x 2048 SemanticKITTI image that is 10-20 % of the points,
and the error concentrates exactly on the classes that sit in front of, or
behind, something else -- person, pole, traffic-sign, bicyclist.

This module re-labels each 3-D point by a vote over the k nearest pixels in a
small window, where "near" is measured in *range*, not in pixel distance. It
needs no retraining and typically adds 1-3 points of 3-D mIoU, more on the thin
classes. It only changes the 3-D metric; the 2-D pixel metric is untouched.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian_kernel(kernel_size, sigma):
    x = torch.arange(kernel_size).float()
    grid_x = x.repeat(kernel_size).view(kernel_size, kernel_size)
    grid_y = grid_x.t()
    mean = (kernel_size - 1) / 2.0
    var = sigma ** 2
    k = (1.0 / (2.0 * math.pi * var)) * torch.exp(
        -((grid_x - mean) ** 2 + (grid_y - mean) ** 2) / (2 * var))
    return k / k.sum()


class KNNPostProcess(nn.Module):
    """
    Args:
        n_classes : number of training classes (class 0 = ignore)
        knn       : how many neighbours vote
        search    : side of the square pixel window searched (must be odd)
        sigma     : width of the spatial prior; larger = pixel distance matters less
        cutoff    : neighbours whose range differs by more than this (metres)
                    are discarded. 0 disables the cutoff.
    """

    def __init__(self, n_classes=20, knn=5, search=5, sigma=1.0, cutoff=1.0):
        super().__init__()
        if search % 2 != 1:
            raise ValueError(f'search window must be odd, got {search}')
        if knn > search * search:
            raise ValueError(f'knn={knn} exceeds the {search}x{search} window')
        self.n_classes = n_classes
        self.knn = knn
        self.search = search
        self.cutoff = cutoff
        inv_gauss = (1.0 - gaussian_kernel(search, sigma)).reshape(1, search * search, 1)
        self.register_buffer('inv_gauss', inv_gauss)

    @torch.no_grad()
    def forward(self, proj_range, unproj_range, proj_argmax, px, py):
        """
        Args:
            proj_range   : (H, W) projected depth image, <= 0 where empty
            unproj_range : (N,)   true depth of every 3-D point
            proj_argmax  : (H, W) predicted label per pixel
            px, py       : (N,)   pixel each point projects onto
        Returns:
            (N,) refined per-point labels
        """
        device = proj_range.device
        H, W = proj_range.shape
        pad = (self.search - 1) // 2
        centre = (self.search * self.search - 1) // 2
        flat_idx = (py * W + px).long()

        # Ranges of every pixel in the window around each point's pixel.
        neigh_range = F.unfold(proj_range[None, None].float(),
                               kernel_size=self.search,
                               padding=pad)[0][:, flat_idx]        # (search^2, N)
        neigh_range[neigh_range <= 0] = float('inf')               # empty pixels never vote
        neigh_range[centre] = unproj_range                         # the point itself

        # Distance = range difference + a mild penalty for being far away in the
        # image, so that two surfaces at the same depth but different pixels are
        # still distinguishable.
        dist = (neigh_range - unproj_range[None, :]).abs()
        dist = dist + self.inv_gauss[0].to(device)

        topk_dist, topk_idx = dist.topk(self.knn, dim=0, largest=False, sorted=False)

        neigh_label = F.unfold(proj_argmax[None, None].float(),
                               kernel_size=self.search,
                               padding=pad)[0][:, flat_idx].long()  # (search^2, N)
        votes = torch.gather(neigh_label, 0, topk_idx)              # (knn, N)

        if self.cutoff > 0:
            # Neighbours from a different surface are sent to a dump class
            # rather than voting for the wrong label.
            votes = votes.masked_fill(topk_dist > self.cutoff, self.n_classes)

        onehot = torch.zeros((self.n_classes + 1, votes.shape[1]),
                             device=device, dtype=torch.float32)
        onehot.scatter_add_(0, votes, torch.ones_like(votes, dtype=torch.float32))

        # Ignore both class 0 (unlabelled) and the dump class when counting votes.
        return onehot[1:self.n_classes].argmax(dim=0) + 1
