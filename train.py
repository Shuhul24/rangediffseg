"""
Training script: an off-the-shelf Diffusion Transformer (DiT) fine-tuned for
LiDAR semantic segmentation on range-view images.

The pre-processing (spherical projection, 3-D augmentation, random crops), the
loss (focal + Lovasz-softmax), the warmup-cosine schedule and the training log
follow RangeViT (github.com/valeoai/rangevit); the transformer trunk and its
pre-trained weights come from DiT (github.com/facebookresearch/DiT).

Usage:
    python train.py config.yaml --data_root /path/to/SemanticKITTI --save_path ./log
"""

import argparse
import datetime
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import dataset
import models
import utils
import utils.tools as tools
from option import Option
from utils.inference import sliding_window_inference
from utils.tools import Recorder, eval_results


def build_rangedit_model(settings, pretrained_path=None):
    model = models.RangeDiT(
        in_channels=settings.in_channels,
        n_cls=settings.n_classes,
        backbone=settings.backbone,
        image_size=settings.image_size,
        pretrained_path=pretrained_path,
        new_patch_size=settings.patch_size,
        new_patch_stride=settings.patch_stride,
        reuse_pos_emb=settings.reuse_pos_emb,
        reuse_patch_emb=settings.reuse_patch_emb,
        conv_stem=settings.conv_stem,
        stem_base_channels=settings.stem_base_channels,
        stem_hidden_dim=settings.D_h,
        skip_filters=settings.skip_filters,
        decoder=settings.decoder,
        up_conv_d_decoder=settings.D_h,
        up_conv_scale_factor=settings.patch_stride,
        dropout=settings.dropout,
        drop_path_rate=settings.drop_path_rate,
        cond_timestep=settings.cond_timestep,
        cond_class=settings.cond_class,
        learnable_cond=settings.learnable_cond,
        cond_mode=settings.cond_mode,
        learnable_pos_emb=settings.learnable_pos_emb)
    return model


class Trainer(object):
    def __init__(self, settings: Option, model: nn.Module, recorder=None):
        self.settings = settings
        self.recorder = recorder
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.remain_time = tools.RemainTime(self.settings.n_epochs)

        # Init data loaders
        self.train_loader, self.val_loader = self._initDataloader()

        # Init criterion and optimizer
        self.criterion = self._initCriterion()
        self.optimizer = self._initOptimizer()

        # Get metrics
        self.metrics = utils.metrics.IOUEval(
            n_classes=self.settings.n_classes, device=torch.device('cpu'),
            ignore=self.ignore_class)
        self.metrics.reset()

        # Define scheduler (stepped once per iteration)
        self.scheduler = utils.scheduler.WarmupCosineLR(
            optimizer=self.optimizer,
            lr=self.settings.lr,
            warmup_steps=self.settings.warmup_epochs * len(self.train_loader),
            momentum=0.9,
            max_steps=len(self.train_loader) * (self.settings.n_epochs - self.settings.warmup_epochs))

        # For mixed precision training
        self.fp16_scaler = None
        if self.settings.use_fp16 and torch.cuda.is_available():
            self.fp16_scaler = tools.grad_scaler(enabled=True)

    def _initOptimizer(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        return torch.optim.AdamW(params=params,
                                 lr=self.settings.lr,
                                 weight_decay=self.settings.weight_decay)

    def _initDataloader(self):
        settings = self.settings
        print('----Using {} dataset----'.format(settings.dataset))

        eval_split = 'test' if settings.test_split else 'val'
        train_parser = dataset.build_parser(
            settings.dataset, settings.data_root, 'train', settings)
        eval_parser = dataset.build_parser(
            settings.dataset, settings.data_root, eval_split, settings)

        common = dict(proj_h=settings.proj_h, proj_w=settings.proj_w,
                      fov_up=settings.fov_up, fov_down=settings.fov_down,
                      img_mean=settings.img_mean, img_stds=settings.img_stds)

        self.train_range_loader = dataset.RangeViewDataset(
            train_parser,
            image_size=settings.image_size,
            is_train=True,
            aug_cfg=settings.augmentation,
            **common)

        self.val_range_loader = dataset.RangeViewDataset(
            eval_parser,
            image_size=None,          # full-width images + sliding-window inference
            is_train=False,
            **common)

        # Class weights for the focal loss. RangeViT weights SemanticKITTI by
        # point frequency and leaves nuScenes uniform.
        cls_freq = self.train_range_loader.cls_freq
        if settings.use_cls_freq_weights and cls_freq is not None:
            self.cls_weight = 1 / (cls_freq + 1e-3)
        else:
            self.cls_weight = np.ones(settings.n_classes)
        self.ignore_class = [0]
        self.cls_weight[self.ignore_class] = 0
        for cl, w in enumerate(self.cls_weight):
            if w < 1e-10 and cl not in self.ignore_class:
                self.ignore_class.append(cl)
        if self.recorder is not None:
            self.recorder.logger.info('weight: {}'.format(self.cls_weight))
        self.mapped_cls_name = self.train_range_loader.mapped_cls_name

        train_loader = torch.utils.data.DataLoader(
            self.train_range_loader,
            batch_size=self.settings.batch_size,
            num_workers=self.settings.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=torch.cuda.is_available())

        val_loader = torch.utils.data.DataLoader(
            self.val_range_loader,
            batch_size=self.settings.batch_size_val,
            num_workers=self.settings.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=torch.cuda.is_available())

        return train_loader, val_loader

    def _initCriterion(self):
        criterion = {}
        criterion['lovasz'] = utils.losses.Lovasz_softmax(ignore=0)

        alpha = np.log(1 + self.cls_weight)
        alpha = alpha / alpha.max()
        alpha[0] = 0
        if self.recorder is not None:
            self.recorder.logger.info('focal_loss alpha: {}'.format(alpha))

        criterion['focal_loss'] = utils.losses.FocalSoftmaxLoss(
            self.settings.n_classes, gamma=2, alpha=alpha, softmax=False)

        for _, v in criterion.items():
            v.to(self.device)
        return criterion

    def compute_losses(self, output, output_softmax, label, mask):
        loss_lovasz = self.criterion['lovasz'](output_softmax, label)
        loss_focal = self.criterion['focal_loss'](output_softmax, label, mask=mask)
        total_loss = loss_focal + loss_lovasz
        return total_loss, loss_lovasz, loss_focal

    def run(self, epoch, mode='Train', print_results=False):
        if mode == 'Train':
            dataloader = self.train_loader
            self.model.train()
        elif mode == 'Validation':
            dataloader = self.val_loader
            self.model.eval()
        else:
            raise ValueError('invalid mode: {}'.format(mode))

        # Init metrics
        loss_meter = tools.AverageMeter()
        self.metrics.reset()
        loss_lovasz = loss_focal = torch.zeros(1)

        total_iter = len(dataloader)
        t_start = time.time()

        for i, (input_feature, input_label, input_mask) in enumerate(dataloader):
            t_process_start = time.time()

            # Feature: range, x, y, z, intensity
            input_feature = input_feature.to(self.device)               # B x 5 x H x W
            input_label = input_label.to(self.device).long()
            input_label = input_label * input_label.ge(1).long()
            input_mask = input_mask.to(self.device) * input_label.ge(1).float()

            if mode == 'Train':
                with tools.autocast(enabled=self.fp16_scaler is not None):
                    output = self.model(input_feature)
                    output_softmax = F.softmax(output, dim=1)
                    total_loss, loss_lovasz, loss_focal = self.compute_losses(
                        output, output_softmax, input_label, input_mask)

                # Backward
                self.optimizer.zero_grad()
                if self.fp16_scaler is None:
                    total_loss.backward()
                    self.optimizer.step()
                else:
                    self.fp16_scaler.scale(total_loss).backward()
                    self.fp16_scaler.step(self.optimizer)
                    self.fp16_scaler.update()

                # Update lr after backward (required by pytorch)
                self.scheduler.step()
            else:
                with torch.no_grad():
                    assert input_feature.shape[0] == 1, 'the validation batch size has to be 1'

                    with tools.autocast(enabled=self.fp16_scaler is not None):
                        lidar_pred = sliding_window_inference(
                            self.model,
                            input_feature,
                            window_size=self.settings.window_size,
                            window_stride=self.settings.window_stride,
                            n_cls=self.settings.n_classes,
                            batch_size=1)

                    output = lidar_pred.unsqueeze(0)     # [C, H, W] ==> [1, C, H, W]
                    output_softmax = F.softmax(output, dim=1)
                    total_loss, loss_lovasz, loss_focal = self.compute_losses(
                        output, output_softmax, input_label, input_mask)

            # Measure IoU and record loss
            loss = total_loss.mean()
            with torch.no_grad():
                argmax = output.argmax(dim=1)
                self.metrics.addBatch(argmax.cpu(), input_label.cpu())   # 2D predictions

            loss_meter.update(loss.item(), input_feature.size(0))

            # Timer logger
            t_process_end = time.time()
            data_cost_time = t_process_start - t_start
            process_cost_time = t_process_end - t_process_start
            self.remain_time.update(cost_time=(time.time() - t_start), mode=mode)
            remain_time = datetime.timedelta(
                seconds=self.remain_time.getRemainTime(
                    epoch=epoch, iters=i, total_iter=total_iter, mode=mode))
            t_start = time.time()

            # Logging
            if (i % self.settings.log_frequency == 0) or (i == total_iter - 1):
                with torch.no_grad():
                    mean_iou, _, mean_acc, _ = self.metrics.getIoUnAcc()
                if self.recorder is not None:
                    lr = self.optimizer.param_groups[0]['lr']
                    log_str = '>>> {} E[{:03d}|{:03d}] I[{:04d}|{:04d}] DT[{:.3f}] PT[{:.3f}] '.format(
                        mode, self.settings.n_epochs, epoch + 1, total_iter, i + 1,
                        data_cost_time, process_cost_time)
                    log_str += 'LR {} Loss {:0.4f} Acc {:0.4f} IOU {:0.4F} '.format(
                        lr, loss.item(), mean_acc.item(), mean_iou.item())
                    log_str += 'RT {}'.format(remain_time)
                    self.recorder.logger.info(log_str)
                    self.recorder.log_wandb({
                        f'{mode}/iter_loss': loss.item(),
                        f'{mode}/iter_mean_acc': mean_acc.item(),
                        f'{mode}/iter_mean_iou': mean_iou.item(),
                        'train/lr': lr,
                        'epoch': epoch + 1,
                    })

        with torch.no_grad():
            mean_acc, class_acc = self.metrics.getAcc()
            mean_recall, class_recall = self.metrics.getRecall()
            mean_iou, class_iou = self.metrics.getIoU()

            metrics_dict = {
                'mean_acc': mean_acc,
                'class_acc': class_acc,
                'mean_recall': mean_recall,
                'class_recall': class_recall,
                'mean_iou': mean_iou,
                'class_iou': class_iou,
                'conf_matrix': self.metrics.conf_matrix.clone().cpu(),
            }

        loss_dict = {
            'loss_meter_avg': loss_meter.avg,
            'loss_focal': float(loss_focal),
            'loss_lovasz': float(loss_lovasz),
        }

        # Print results
        if self.recorder is not None:
            print_train = (mode == 'Train' and
                           ((epoch % self.settings.train_result_frequency == 0) or
                            (epoch == self.settings.n_epochs - 1)))
            print_val = (mode == 'Validation' and
                         (print_results or epoch == self.settings.n_epochs - 1))

            if print_train or print_val:
                eval_results(pixel_or_point='Pixel',
                             settings=self.settings,
                             recorder=self.recorder,
                             metrics_dict=metrics_dict,
                             class_names=self.mapped_cls_name,
                             print_data_distribution=True)

            if self.recorder.tensorboard is not None:
                self.recorder.tensorboard.add_scalar(f'{mode}/loss', loss_dict['loss_meter_avg'], epoch)
                self.recorder.tensorboard.add_scalar(f'{mode}/mean_iou', mean_iou.item(), epoch)
                self.recorder.tensorboard.add_scalar(f'{mode}/mean_acc', mean_acc.item(), epoch)

            self.recorder.log_wandb({
                f'{mode}/loss': loss_dict['loss_meter_avg'],
                f'{mode}/loss_focal': loss_dict['loss_focal'],
                f'{mode}/loss_lovasz': loss_dict['loss_lovasz'],
                f'{mode}/mean_iou': mean_iou.item(),
                f'{mode}/mean_acc': mean_acc.item(),
                f'{mode}/mean_recall': mean_recall.item(),
                'epoch': epoch + 1,
            })

        return {'mean_iou': mean_iou.item(), 'mean_acc': mean_acc.item()}


class Experiment(object):
    def __init__(self, settings: Option):
        self.settings = settings
        self.settings.check_path()

        # Set random seed
        torch.manual_seed(self.settings.seed)
        torch.cuda.manual_seed_all(self.settings.seed)
        np.random.seed(self.settings.seed)
        torch.backends.cudnn.benchmark = True

        self.recorder = Recorder(self.settings, self.settings.save_path)
        self.epoch_start = 0

        self.model = self._initModel()
        self.trainer = Trainer(self.settings, self.model, self.recorder)
        self._loadCheckpoint()

    def _initModel(self):
        model = build_rangedit_model(
            self.settings, pretrained_path=self.settings.pretrained_model)

        # Freezing the DiT encoder weights: only the stem, the positional
        # embedding, the conditioning and the decoder are learnt.
        if self.settings.freeze_dit_encoder:
            model.freeze_encoder(
                unfreeze_adaln=self.settings.unfreeze_adaln,
                adaln_bias_only=self.settings.adaln_bias_only,
                unfreeze_attn=self.settings.unfreeze_attn,
                unfreeze_ffn=self.settings.unfreeze_ffn)

        if self.recorder is not None:
            self.recorder.logger.info(f'model = {model}')
            stats = model.counter_model_parameters()
            self.recorder.logger.info('Number of trainable model parameters:')
            for key, val in stats.items():
                self.recorder.logger.info(f'==> {key}: {val}')

        return model

    def _loadCheckpoint(self):
        if self.settings.finetune_from is not None:
            print(f'Fine-tune model weights from checkpoint {self.settings.finetune_from}')
            if not os.path.isfile(self.settings.finetune_from):
                raise FileNotFoundError('fine-tune checkpoint file not found: {}'.format(
                    self.settings.finetune_from))

            checkpoint_data = torch.load(self.settings.finetune_from, map_location='cpu')
            state_dict = checkpoint_data.get('model', checkpoint_data)
            msg = self.model.load_state_dict(state_dict, strict=self.settings.finetune_strict)
            print(f'{msg}')
            return

        if self.settings.checkpoint is None:
            return

        print(f'Resume training from checkpoint {self.settings.checkpoint}')
        if not os.path.isfile(self.settings.checkpoint):
            raise FileNotFoundError('checkpoint file not found: {}'.format(self.settings.checkpoint))

        checkpoint_data = torch.load(self.settings.checkpoint, map_location='cpu')
        msg = self.model.load_state_dict(checkpoint_data['model'], strict=True)
        print(f'{msg}')

        print('==> Loading optimizer')
        if self.settings.val_only is False:
            self.trainer.optimizer.load_state_dict(checkpoint_data['optimizer'])
            self.epoch_start = checkpoint_data['epoch'] + 1
            if checkpoint_data.get('fp16_scaler') is not None and self.trainer.fp16_scaler is not None:
                self.trainer.fp16_scaler.load_state_dict(checkpoint_data['fp16_scaler'])

    def _save_checkpoint(self, epoch, path, extra=None):
        checkpoint_data = {
            'model': self.model.state_dict(),
            'optimizer': self.trainer.optimizer.state_dict(),
            'epoch': epoch,
        }
        if self.trainer.fp16_scaler is not None:
            checkpoint_data['fp16_scaler'] = self.trainer.fp16_scaler.state_dict()
        if extra is not None:
            checkpoint_data.update(extra)
        torch.save(checkpoint_data, path)

    def run(self):
        t_start = time.time()

        if self.settings.val_only:
            self.trainer.run(self.epoch_start, mode='Validation', print_results=True)
            self.recorder.logger.info('==== Total cost time: {}'.format(
                datetime.timedelta(seconds=time.time() - t_start)))
            self.recorder.finish_wandb()
            return

        best_val_result = None
        for epoch in range(self.epoch_start, self.settings.n_epochs):
            self.trainer.run(epoch, mode='Train')

            if (epoch % self.settings.val_frequency == 0 or
                    epoch == self.settings.n_epochs - 1 or
                    epoch == self.epoch_start):
                val_result = self.trainer.run(epoch, mode='Validation')

                self.recorder.logger.info(f'---- Best result after Epoch {epoch+1} ----')
                if best_val_result is None:
                    best_val_result = val_result
                for k, v in val_result.items():
                    if v >= best_val_result[k]:
                        self.recorder.logger.info('Get better {} model: {}'.format(k, v))
                        best_val_result[k] = v
                        self._save_checkpoint(
                            epoch,
                            os.path.join(self.recorder.checkpoint_path, f'best_{k}_model.pth'),
                            extra={k: v})

            # Save the last checkpoint
            self._save_checkpoint(
                epoch, os.path.join(self.recorder.checkpoint_path, 'checkpoint.pth'))

            if best_val_result is not None:
                log_str = '>>> Best Result: '
                for k, v in best_val_result.items():
                    log_str += '{}: {} '.format(k, v)
                self.recorder.logger.info(log_str + '\n')

        self.recorder.logger.info('=== Total cost time: {}'.format(
            datetime.timedelta(seconds=time.time() - t_start)))
        self.recorder.finish_wandb()


def parse_args():
    parser = argparse.ArgumentParser(description='RangeDiT training options')
    parser.add_argument('config_path', type=str, nargs='?', default='config.yaml',
                        help='path of the config file, type: string')
    parser.add_argument('--config', dest='config_path',
                        help='path of the config file (alias of the positional argument)')
    parser.add_argument('--data_root', type=str, default=None,
                        help='path to the SemanticKITTI root, type: string')
    parser.add_argument('--save_path', type=str, default=None,
                        help='path where logs and checkpoints are written, type: string')
    parser.add_argument('--id', type=str, default=None, help='name to identify the run')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='number of threads used for data loading, type: int')
    parser.add_argument('--batch_size', type=int, default=None, help='mini-batch size, type: int')
    parser.add_argument('--pretrained_model', type=str, default=None,
                        help='DiT checkpoint used to initialize the transformer trunk '
                             '(e.g. DiT-XL-2-256x256.pt or a local path), type: string')
    parser.add_argument('--checkpoint', '--resume', dest='checkpoint', type=str, default=None,
                        help='path of a RangeDiT checkpoint to resume from, type: string')
    parser.add_argument('--finetune_from', type=str, default=None,
                        help='load model weights from this checkpoint and start a fresh optimizer')
    parser.add_argument('--finetune_non_strict', action='store_true',
                        help='allow missing/unexpected keys when using --finetune_from')
    parser.add_argument('--window_stride', type=int, default=None,
                        help='sliding window stride along the width during validation, type: int')
    parser.add_argument('--n_epochs', type=int, default=None, help='number of epochs, type: int')
    parser.add_argument('--lr', type=float, default=None, help='learning rate, type: float')
    parser.add_argument('--warmup_epochs', type=int, default=None,
                        help='number of warmup epochs, type: int')
    parser.add_argument('--val_frequency', type=int, default=None,
                        help='validation frequency in epochs, type: int')
    parser.add_argument('--cond_mode', type=str, default=None, choices=['static', 'context'],
                        help='DiT conditioning mode')
    parser.add_argument('--unfreeze_attn', action='store_true',
                        help='also train the DiT attention layers')
    parser.add_argument('--unfreeze_ffn', action='store_true',
                        help='also train the DiT feed-forward layers')
    parser.add_argument('--use_wandb', action='store_true', help='enable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default=None, help='Weights & Biases project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='Weights & Biases entity/team')
    parser.add_argument('--wandb_name', type=str, default=None, help='Weights & Biases run name')
    parser.add_argument('--wandb_mode', type=str, default=None, help='Weights & Biases mode, e.g. online/offline')
    parser.add_argument('--val_only', action='store_true', help='run inference only')
    parser.add_argument('--test_split', action='store_true', help='run inference on the test split')
    parser.add_argument('--log_frequency', type=int, default=None, help='logging frequency')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    settings = Option(args.config_path, args)
    if settings.val_only:
        settings.save_path = os.path.join(settings.save_path, 'Eval')
    Experiment(settings).run()
