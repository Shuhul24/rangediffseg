"""
KNN post-processing for range-view predictions.

A range image keeps one point per pixel, so every other point that falls on
that pixel inherits its label when the 2-D prediction is back-projected. On
SemanticKITTI at 64 x 2048 that affects ~20% of the points, and ~2.4% of all
points are assigned a wrong label even when the 2-D prediction is perfect --
the error concentrates on occlusion boundaries (other-ground, fence, parking).

The fix, introduced by RangeNet++ (github.com/PRBonn/lidar-bonnetal, MIT) and
reused by RangeViT, is to relabel every 3-D point by a k-nearest-neighbour vote
among the predictions in a small window around its pixel, where "nearest" is
measured in range rather than in image space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian_kernel(sigma, size):
    """(size * size,) normalised 2-D Gaussian, peak 1 at the window centre."""
    grid = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
    yy, xx = torch.meshgrid(grid, grid, indexing='ij')
    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return (kernel / kernel.max()).reshape(-1)


class KNNPostprocess(nn.Module):
    """
    Args:
        n_classes : number of label ids (including the ignore id)
        knn       : number of neighbours that vote
        search    : side of the square window searched around each pixel (odd)
        sigma     : width of the Gaussian that penalises far-away pixels
        cutoff    : neighbours whose range differs by more than this (metres)
                    are discarded from the vote
    """

    def __init__(self, n_classes, knn=3, search=7, sigma=2.0, cutoff=1.0):
        super().__init__()
        assert search % 2 == 1, 'the search window has to be odd'
        assert knn <= search * search, 'cannot take more neighbours than the window holds'

        self.n_classes = n_classes
        self.knn = knn
        self.search = search
        self.cutoff = cutoff

        # Penalise neighbours by their distance to the window centre: the
        # kernel is inverted so that a far pixel *adds* to the range distance.
        self.register_buffer('inv_gauss', 1.0 - gaussian_kernel(sigma, search))

    @torch.no_grad()
    def forward(self, proj_range, unproj_range, proj_argmax, px, py):
        """
        Args:
            proj_range   : (H, W) range image, <= 0 for empty pixels
            unproj_range : (N,)   true range of every 3-D point
            proj_argmax  : (H, W) predicted label of every pixel
            px, py       : (N,)   pixel each 3-D point projects onto
        Returns:
            (N,) per-point labels after the KNN vote
        """
        device = proj_range.device
        H, W = proj_range.shape
        pad = (self.search - 1) // 2
        centre = (self.search * self.search - 1) // 2

        # Gather the search window around every pixel: (search*search, H, W)
        def unfold(img):
            out = F.unfold(img[None, None].float(),
                           kernel_size=(self.search, self.search),
                           padding=(pad, pad))
            return out.reshape(self.search * self.search, H, W)

        win_range = unfold(proj_range)[:, py, px]     # (S, N)
        win_pred = unfold(proj_argmax)[:, py, px]     # (S, N)

        # Empty pixels must never win the vote.
        win_range[win_range <= 0] = float('inf')
        # The centre slot stands for the point itself, at its own true range.
        win_range[centre] = unproj_range

        # Distance in range, penalised by distance in the image.
        dist = (win_range - unproj_range[None, :]).abs()
        dist = dist + self.inv_gauss.to(device)[:, None]

        knn_dist, knn_idx = torch.topk(dist, self.knn, dim=0, largest=False)
        knn_pred = torch.gather(win_pred, 0, knn_idx).long()   # (knn, N)

        # Drop neighbours that are too far away in range to be the same surface.
        invalid = knn_dist > self.cutoff
        if invalid.any():
            # Route rejected votes to the ignore id, which is stripped below.
            knn_pred = knn_pred.masked_fill(invalid, 0)

        votes = torch.zeros(self.n_classes, knn_pred.shape[1], device=device, dtype=torch.long)
        votes.scatter_add_(0, knn_pred, torch.ones_like(knn_pred))
        votes[0] = 0                                  # never elect the ignore id

        winner = votes.argmax(dim=0)

        # If every neighbour was rejected, keep the plain back-projected label.
        empty = votes.sum(dim=0) == 0
        if empty.any():
            winner[empty] = proj_argmax[py, px][empty].long()

        return winner
