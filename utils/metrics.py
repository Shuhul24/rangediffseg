"""
Confusion-matrix based IoU / accuracy / recall evaluation.

From lidar-bonnetal / SalsaNext, as vendored by RangeViT
(github.com/valeoai/rangevit).
"""

import numpy as np
import torch


class IOUEval:
    def __init__(self, n_classes, device=torch.device('cpu'), ignore=None):
        self.n_classes = n_classes
        self.device = device
        self.ignore = torch.tensor(ignore if ignore is not None else []).long()
        self.include = torch.tensor(
            [n for n in range(self.n_classes) if n not in self.ignore]).long()
        print('[IOU EVAL] IGNORE: ', self.ignore)
        print('[IOU EVAL] INCLUDE: ', self.include)
        self.reset()

    def num_classes(self):
        return self.n_classes

    def reset(self):
        self.conf_matrix = torch.zeros((self.n_classes, self.n_classes), device=self.device).long()
        self.ones = None
        self.last_scan_size = None

    def addBatch(self, x, y):
        """x = predictions, y = targets (any shape, flattened internally)."""
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(np.array(x))
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(np.array(y))
        x_row = x.long().to(self.device).reshape(-1)
        y_row = y.long().to(self.device).reshape(-1)

        idxs = torch.stack([x_row, y_row], dim=0)
        if self.ones is None or self.last_scan_size != idxs.shape[-1]:
            self.ones = torch.ones((idxs.shape[-1]), device=self.device).long()
            self.last_scan_size = idxs.shape[-1]

        self.conf_matrix = self.conf_matrix.index_put_(tuple(idxs), self.ones, accumulate=True)

    def getStats(self):
        conf = self.conf_matrix.clone().double()
        conf[self.ignore] = 0
        conf[:, self.ignore] = 0
        tp = conf.diag()
        fp = conf.sum(dim=1) - tp
        fn = conf.sum(dim=0) - tp
        return tp, fp, fn

    def getIoU(self):
        tp, fp, fn = self.getStats()
        union = tp + fp + fn + 1e-15
        iou = tp / union
        return iou[self.include].mean(), iou

    def getIoUnAcc(self):
        tp, fp, fn = self.getStats()
        union = tp + fp + fn + 1e-15
        iou = tp / union
        acc = tp / (tp + fp + 1e-15)
        return iou[self.include].mean(), iou, acc[self.include].mean(), acc

    def getAcc(self):
        tp, fp, fn = self.getStats()
        acc = tp / (tp + fp + 1e-15)
        return acc[self.include].mean(), acc

    def getRecall(self):
        tp, fp, fn = self.getStats()
        recall = tp / (tp + fn + 1e-15)
        return recall[self.include].mean(), recall
