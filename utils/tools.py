"""
Logging helpers: running meters, remaining-time estimation, a file+console
recorder and the per-class evaluation tables.

The meters and the remaining-time estimator come from PMF
(github.com/ICEORY/PMF) via RangeViT (github.com/valeoai/rangevit); the table
formatting reproduces the `prettytable` output RangeViT prints, without the
extra dependency.
"""

import logging
import os
import sys


# --------------------------------------------------------------------------- #
#                                   Meters                                    #
# --------------------------------------------------------------------------- #

class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class RunningAvgMeter(object):
    """Exponential moving average: avg = alpha * avg + (1 - alpha) * val."""

    def __init__(self, alpha=0.95):
        assert 0 <= alpha <= 1, 'alpha should be in [0, 1]'
        self.alpha = alpha
        self.reset()

    def reset(self):
        self.is_init = False
        self.avg = 0

    def update(self, val):
        if self.is_init:
            self.avg = self.avg * self.alpha + (1 - self.alpha) * val
        else:
            self.avg = val
            self.is_init = True


class RemainTime(object):
    """Estimates the remaining training time from per-iteration timings."""

    def __init__(self, n_epochs):
        self.n_epochs = n_epochs
        self.timer_avg = {}
        self.total_iter = {}

    def update(self, cost_time, batch_size=1, mode='Train'):
        if mode not in self.timer_avg:
            self.timer_avg[mode] = RunningAvgMeter()
            self.total_iter[mode] = 0
        self.timer_avg[mode].update(cost_time)

    def reset(self):
        self.timer_avg = {}

    def getRemainTime(self, epoch, iters, total_iter, mode='Train'):
        if self.total_iter[mode] == 0:
            self.total_iter[mode] = total_iter

        remain_time = 0
        mode_idx = list(self.timer_avg.keys()).index(mode)
        count = 0
        for k, v in self.timer_avg.items():
            if k == mode:
                remain_iter = (self.n_epochs - epoch) * self.total_iter[k] - iters
            elif count < mode_idx:
                remain_iter = (self.n_epochs - epoch - 1) * self.total_iter[k]
            else:
                remain_iter = (self.n_epochs - epoch) * self.total_iter[k]
            count += 1
            remain_time += v.avg * remain_iter
        return max(remain_time, 0)


# --------------------------------------------------------------------------- #
#                                   Tables                                    #
# --------------------------------------------------------------------------- #

def format_table(field_names, rows):
    """Render an ASCII table in the style of `prettytable` (centred cells)."""
    field_names = [str(f) for f in field_names]
    rows = [[str(c) for c in row] for row in rows]
    widths = [len(f) for f in field_names]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    sep = '+' + '+'.join('-' * (w + 2) for w in widths) + '+'

    def render(cells):
        out = []
        for cell, w in zip(cells, widths):
            pad = w - len(cell)
            left = pad // 2
            out.append(' ' + ' ' * left + cell + ' ' * (pad - left) + ' ')
        return '|' + '|'.join(out) + '|'

    lines = [sep, render(field_names), sep]
    lines += [render(row) for row in rows]
    lines.append(sep)
    return '\n'.join(lines)


# --------------------------------------------------------------------------- #
#                                  Recorder                                   #
# --------------------------------------------------------------------------- #

class Recorder(object):
    """
    Writes the training log both to stdout and to `<save_path>/log/console.log`,
    and owns the checkpoint directory. Tensorboard is used when available.
    """

    def __init__(self, settings, save_path, use_tensorboard=True):
        print('>> Init a recorder at ', save_path)
        self.save_path = save_path
        self.settings = settings
        self.log_path = os.path.join(self.save_path, 'log')
        self.checkpoint_path = os.path.join(self.save_path, 'checkpoint')

        os.makedirs(self.log_path, exist_ok=True)
        os.makedirs(self.checkpoint_path, exist_ok=True)

        self.tensorboard = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tensorboard = SummaryWriter(log_dir=self.save_path)
            except ImportError:
                print('>> Tensorboard is not available, skipping it.')

        self.logger = self._initLogger()
        self._saveConfig()

    def _initLogger(self):
        logger = logging.getLogger('console')
        logger.propagate = False
        logger.handlers = []

        file_handler = logging.FileHandler(os.path.join(self.log_path, 'console.log'))
        file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter('%(message)s'))

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        logger.setLevel(logging.INFO)
        return logger

    def _saveConfig(self):
        with open(os.path.join(self.log_path, 'settings.log'), 'w') as f:
            for k, v in self.settings.__dict__.items():
                f.write('{}: {}\n'.format(k, v))


# --------------------------------------------------------------------------- #
#                          Evaluation result printing                         #
# --------------------------------------------------------------------------- #

def eval_results(pixel_or_point, settings, recorder, metrics_dict, class_names,
                 print_data_distribution=False, print_confusion_matrix=False):
    """Print the per-class IoU / accuracy / recall table (RangeViT format)."""
    mean_acc, class_acc = metrics_dict['mean_acc'], metrics_dict['class_acc']
    mean_recall, class_recall = metrics_dict['mean_recall'], metrics_dict['class_recall']
    mean_iou, class_iou = metrics_dict['mean_iou'], metrics_dict['class_iou']

    dim = '3D' if pixel_or_point == 'Point' else '2D'
    recorder.logger.info(f'============ {pixel_or_point}-wise Evaluation Results ({dim} Eval) ============')
    recorder.logger.info('Acc avg: {:.4f}, IOU avg: {:.4f}, Recall avg: {:.4f}'.format(
        mean_acc.item(), mean_iou.item(), mean_recall.item()))

    rows = []
    latext_str = ''
    for i, iou in enumerate(class_iou.cpu()):
        if i == 0:  # class 0 is the ignored 'unlabeled' class
            continue
        rows.append([i, class_names[i], iou.item(),
                     class_acc[i].cpu().item(), class_recall[i].cpu().item()])
        latext_str += ' & {:0.1f}'.format(iou * 100)
    latext_str += ' & {:0.1f}'.format(mean_iou.cpu().item() * 100)

    recorder.logger.info(format_table(['Class ID', 'Class Name', 'IOU', 'Acc', 'Recall'], rows))
    recorder.logger.info('---- Latext Format String ----')
    recorder.logger.info(latext_str)

    conf_matrix = metrics_dict['conf_matrix'].clone()
    conf_matrix[0] = 0
    conf_matrix[:, 0] = 0

    if print_confusion_matrix:
        recorder.logger.info('---- Confusion Matrix Original Data ----')
        recorder.logger.info(conf_matrix)

    if print_data_distribution:
        dist_data = conf_matrix.sum(0)
        total = dist_data.sum().clamp(min=1)
        rows = [[class_names[i], dist_data[i].item(),
                 '{:.4f}%'.format((dist_data[i] / total).item() * 100)]
                for i in range(settings.n_classes)]
        recorder.logger.info('---- Data Distribution ----')
        recorder.logger.info(format_table(['Class Name', 'Number of Points', 'Percentage'], rows))


# --------------------------------------------------------------------------- #
#                        Mixed-precision compatibility                        #
# --------------------------------------------------------------------------- #

def autocast(enabled=True, device_type='cuda'):
    """`torch.amp.autocast` on recent PyTorch, `torch.cuda.amp.autocast` before."""
    import torch
    try:
        return torch.amp.autocast(device_type, enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast(enabled=enabled)


def grad_scaler(enabled=True, device_type='cuda'):
    """`torch.amp.GradScaler` on recent PyTorch, `torch.cuda.amp.GradScaler` before."""
    import torch
    try:
        return torch.amp.GradScaler(device_type, enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)
