"""
Training losses: focal loss + Lovasz-softmax, the combination used by RangeViT.

Lovasz-softmax from M. Berman et al. / SalsaNext (MIT),
focal softmax loss from PMF (github.com/ICEORY/PMF),
both as vendored by RangeViT (github.com/valeoai/rangevit).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from itertools import ifilterfalse
except ImportError:
    from itertools import filterfalse as ifilterfalse


# --------------------------------------------------------------------------- #
#                              Lovasz-softmax                                 #
# --------------------------------------------------------------------------- #

def isnan(x):
    return x != x


def mean(values, ignore_nan=False, empty=0):
    """nanmean compatible with generators."""
    values = iter(values)
    if ignore_nan:
        values = ifilterfalse(isnan, values)
    try:
        n = 1
        acc = next(values)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(values, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


def lovasz_grad(gt_sorted):
    """Gradient of the Lovasz extension w.r.t. sorted errors."""
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_softmax_flat(probas, labels, classes='present'):
    if probas.numel() == 0:
        return probas * 0.
    if probas.dim() == 1:
        probas = probas.unsqueeze(0)

    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float()
        if classes == 'present' and fg.sum() == 0:
            continue
        if C == 1:
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (fg - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        fg_sorted = fg[perm.data]
        losses.append(torch.dot(errors_sorted, lovasz_grad(fg_sorted)))
    return mean(losses)


def flatten_probas(probas, labels, ignore=None):
    """Flatten predictions and labels, dropping the ignored classes."""
    if probas.dim() == 3:
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)
    labels = labels.view(-1)

    if ignore is None:
        return probas, labels

    if isinstance(ignore, (list, tuple)):
        valid = (labels != ignore[0])
        for i in range(1, len(ignore)):
            valid = valid * (labels != ignore[i])
    else:
        valid = (labels != ignore)

    return probas[torch.nonzero(valid, as_tuple=False).squeeze()], labels[valid]


def lovasz_softmax(probas, labels, classes='present', per_image=False, ignore=None):
    """
    Multi-class Lovasz-softmax loss.

    Args:
        probas : (B, C, H, W) class probabilities (already softmax-ed)
        labels : (B, H, W) ground-truth labels in [0, C)
    """
    if per_image:
        return mean(
            lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore),
                                classes=classes)
            for prob, lab in zip(probas, labels))
    return lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), classes=classes)


class Lovasz_softmax(nn.Module):
    def __init__(self, classes='present', per_image=False, ignore=None):
        super().__init__()
        self.classes = classes
        self.per_image = per_image
        self.ignore = ignore

    def forward(self, probas, labels):
        return lovasz_softmax(probas, labels, self.classes, self.per_image, self.ignore)


# --------------------------------------------------------------------------- #
#                             Focal softmax loss                              #
# --------------------------------------------------------------------------- #

class FocalSoftmaxLoss(nn.Module):
    def __init__(self, n_classes, gamma=1, alpha=0.8, softmax=True):
        super().__init__()
        self.gamma = gamma
        self.n_classes = n_classes

        if isinstance(alpha, (list, tuple)):
            assert len(alpha) == n_classes
            self.alpha = torch.Tensor(alpha)
        elif isinstance(alpha, np.ndarray):
            assert alpha.shape[0] == n_classes
            self.alpha = torch.from_numpy(alpha).float()
        else:
            assert 0 < alpha < 1, f'invalid alpha: {alpha}'
            self.alpha = torch.zeros(n_classes)
            self.alpha[0] = alpha
            self.alpha[1:] += (1 - alpha)
        self.softmax = softmax

    def forward(self, x, target, mask=None):
        """
        Args:
            x      : (B, C, H, W) logits or probabilities (see `softmax`)
            target : (B, H, W) ground-truth labels
            mask   : (B, H, W) 1 for pixels that contribute to the loss
        """
        if x.dim() > 2:
            pred = x.view(x.size(0), x.size(1), -1).transpose(1, 2).contiguous().view(-1, x.size(1))
        else:
            pred = x

        target = target.view(-1, 1)
        pred_softmax = F.softmax(pred, 1) if self.softmax else pred
        pred_softmax = pred_softmax.gather(1, target).view(-1)
        pred_logsoft = pred_softmax.clamp(1e-6).log()

        self.alpha = self.alpha.to(x.device)
        alpha = self.alpha.gather(0, target.squeeze())

        loss = -(1 - pred_softmax).pow(self.gamma) * pred_logsoft * alpha

        if mask is not None:
            mask = mask.reshape(-1)
            return (loss * mask).sum() / mask.sum().clamp(min=1.0)
        return loss.mean()
