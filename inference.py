"""
Evaluation script: runs the fine-tuned DiT over full-width range images with a
sliding window, back-projects the 2-D predictions to per-point 3-D labels and
reports both the pixel-wise (2-D) and the point-wise (3-D) mIoU.

The sliding-window evaluation and the back-projection follow RangeViT
(github.com/valeoai/rangevit).

Usage:
    python inference.py config.yaml --data_root /path/to/SemanticKITTI \
                        --checkpoint log/log_<id>/checkpoint/best_mean_iou_model.pth
"""

import argparse
import os
import time

import numpy as np
import torch

import dataset
import utils
from option import Option
from train import build_rangedit_model
from utils.inference import sliding_window_inference
from utils.tools import Recorder, eval_results


def parse_args():
    parser = argparse.ArgumentParser(description='RangeDiT evaluation options')
    parser.add_argument('config_path', type=str, nargs='?', default='config.yaml',
                        help='path of the config file, type: string')
    parser.add_argument('--config', dest='config_path',
                        help='path of the config file (alias of the positional argument)')
    parser.add_argument('--data_root', type=str, default=None,
                        help='path to the SemanticKITTI root, type: string')
    parser.add_argument('--save_path', type=str, default=None,
                        help='path where the evaluation log is written, type: string')
    parser.add_argument('--id', type=str, default=None, help='name to identify the run')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='path of the RangeDiT checkpoint to evaluate, type: string')
    parser.add_argument('--split', type=str, default='val', choices=['val', 'test'],
                        help='dataset split to evaluate')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='number of threads used for data loading, type: int')
    parser.add_argument('--window_stride', type=int, default=None,
                        help='sliding window stride along the width, type: int')
    parser.add_argument('--save_eval_results', action='store_true',
                        help='write the per-point predictions as SemanticKITTI .label files')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    args = parser.parse_args()
    args.test_split = (args.split == 'test')
    args.val_only = True
    args.pretrained_model = None
    args.batch_size = None
    args.log_frequency = None
    return args


def save_predictions(pred_3d, entry, prediction_path, inv_lut):
    """Write predictions as SemanticKITTI .label files (raw label space)."""
    out_dir = os.path.join(prediction_path, 'sequences', entry['seq'], 'predictions')
    os.makedirs(out_dir, exist_ok=True)
    inv_lut[pred_3d].astype(np.uint32).tofile(os.path.join(out_dir, entry['stem'] + '.label'))


def main():
    args = parse_args()
    settings = Option(args.config_path, args)
    settings.save_path = os.path.join(settings.save_path, 'Eval_{}'.format(args.split))
    settings.check_path()

    torch.manual_seed(settings.seed)
    np.random.seed(settings.seed)

    recorder = Recorder(settings, settings.save_path, use_tensorboard=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---- Dataset ----
    seqs = settings.test_seqs if args.test_split else settings.val_seqs
    eval_ds = dataset.RangeViewDataset(
        root=settings.data_root,
        sequences=seqs,
        proj_h=settings.proj_h,
        proj_w=settings.proj_w,
        fov_up=settings.fov_up,
        fov_down=settings.fov_down,
        image_size=None,
        is_train=False,
        return_points=True,
        has_label=(args.test_split is False))

    loader = torch.utils.data.DataLoader(
        eval_ds, batch_size=1, shuffle=False,
        num_workers=settings.num_workers, drop_last=False)

    # ---- Model ----
    model = build_rangedit_model(settings, pretrained_path=None).to(device)
    checkpoint_data = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint_data.get('model', checkpoint_data), strict=True)
    model.eval()
    recorder.logger.info(f'Loaded checkpoint {args.checkpoint} '
                         f'(epoch {checkpoint_data.get("epoch", "n/a")})')

    # ---- Metrics ----
    metrics_2d = utils.metrics.IOUEval(n_classes=settings.n_classes,
                                       device=torch.device('cpu'), ignore=[0])
    metrics_3d = utils.metrics.IOUEval(n_classes=settings.n_classes,
                                       device=torch.device('cpu'), ignore=[0])

    prediction_path = os.path.join(settings.save_path, 'preds')
    total_iter = len(loader)
    t_start = time.time()

    recorder.logger.info(f'Evaluating on {len(eval_ds)} scans '
                         f'(window {settings.window_size}, stride {settings.window_stride})...')

    for i, (feature, label_2d, mask, px, py, point_labels) in enumerate(loader):
        feature = feature.to(device)

        with torch.no_grad():
            logits = sliding_window_inference(
                model, feature,
                window_size=settings.window_size,
                window_stride=settings.window_stride,
                n_cls=settings.n_classes,
                batch_size=1)

        pred_2d = logits.argmax(dim=0).cpu()                     # (H, W)

        # Back-project the 2-D predictions to per-point 3-D labels
        px_np, py_np = px[0].numpy(), py[0].numpy()
        pred_3d = pred_2d.numpy()[py_np, px_np]                  # (N,)

        if not args.test_split:
            gt_3d = point_labels[0].numpy()
            metrics_2d.addBatch(pred_2d, label_2d[0])
            metrics_3d.addBatch(torch.from_numpy(pred_3d), torch.from_numpy(gt_3d))

        if args.save_eval_results:
            save_predictions(pred_3d, eval_ds.parser.files[i], prediction_path,
                             eval_ds.parser.inv_lut)

        if (i % settings.log_frequency == 0) or (i == total_iter - 1):
            elapsed = time.time() - t_start
            log_str = '>>> {} I[{:04d}|{:04d}] PT[{:.3f}] '.format(
                'Evaluation', total_iter, i + 1, elapsed / (i + 1))
            if not args.test_split:
                mean_iou_3d, _ = metrics_3d.getIoU()
                log_str += 'IOU {:0.4F} '.format(mean_iou_3d.item())
            log_str += 'ET {}'.format(str(int(elapsed)) + 's')
            recorder.logger.info(log_str)

    if args.test_split:
        recorder.logger.info(f'Predictions written to {prediction_path}')
        return

    for name, metrics in (('Pixel', metrics_2d), ('Point', metrics_3d)):
        mean_acc, class_acc = metrics.getAcc()
        mean_recall, class_recall = metrics.getRecall()
        mean_iou, class_iou = metrics.getIoU()
        eval_results(pixel_or_point=name,
                     settings=settings,
                     recorder=recorder,
                     metrics_dict={
                         'mean_acc': mean_acc, 'class_acc': class_acc,
                         'mean_recall': mean_recall, 'class_recall': class_recall,
                         'mean_iou': mean_iou, 'class_iou': class_iou,
                         'conf_matrix': metrics.conf_matrix.clone().cpu(),
                     },
                     class_names=eval_ds.mapped_cls_name,
                     print_data_distribution=True)

    if args.save_eval_results:
        recorder.logger.info(f'Predictions written to {prediction_path}')


if __name__ == '__main__':
    main()
