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
import contextlib
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


def build_rangeformer_dit_model(settings):
    """The new `rangeformer_dit_seg` branch. Existing models are untouched."""
    from models.rangeformer_dit import RangeFormerDiTSeg
    return RangeFormerDiTSeg(
        in_channels=settings.rf_in_channels,
        n_classes=settings.n_classes,
        rem_dims=settings.rf_rem_dims,
        encoder_dims=settings.rf_encoder_dims,
        encoder_depths=settings.rf_encoder_depths,
        encoder_heads=settings.rf_encoder_heads,
        encoder_sr_ratios=settings.rf_encoder_sr,
        circular_padding=settings.rf_circular,
        latent_downsample=settings.rf_latent_downsample,
        cond_dim=settings.rf_cond_dim,
        semantic_latent_dim=settings.rf_sem_latent_dim,
        dit_hidden_dim=settings.rf_dit_hidden,
        dit_depth=settings.rf_dit_depth,
        dit_heads=settings.rf_dit_heads,
        dit_mlp_ratio=settings.rf_dit_mlp_ratio,
        dit_dropout=settings.rf_dit_dropout,
        cond_sr_ratio=settings.rf_cond_sr_ratio,
        grad_checkpointing=settings.rf_grad_checkpointing,
        bottleneck_type=settings.rf_bottleneck_type,
        use_diffusion=settings.rf_use_diffusion,
        decoder_dims=settings.rf_decoder_dims,
        use_gated_skip=settings.rf_use_gated_skip,
        diffusion_timesteps=settings.rf_timesteps,
        diffusion_schedule=settings.rf_schedule,
        inference_steps=settings.rf_inference_steps,
        inference_t_start=settings.rf_inference_t_start,
        s0_clamp=settings.rf_s0_clamp)


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
        # Coarse-head mIoU, so the DiT can be compared against its own anchor.
        self.coarse_metrics = utils.metrics.IOUEval(
            n_classes=self.settings.n_classes, device=torch.device('cpu'),
            ignore=self.ignore_class)
        self.coarse_metrics.reset()

        # Define scheduler (stepped once per iteration)
        self.scheduler = utils.scheduler.WarmupCosineLR(
            optimizer=self.optimizer,
            lr=self.settings.lr,
            warmup_steps=self.settings.warmup_epochs * len(self.train_loader),
            momentum=0.9,
            max_steps=len(self.train_loader) * (self.settings.n_epochs - self.settings.warmup_epochs))

        self.is_rangeformer_dit = (settings.model_type == 'rangeformer_dit_seg')
        self._trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self._micro_step = 0
        self.global_step = 0
        self._last_grad_norm = torch.zeros(())
        self._last_parts = {}
        self._t_hist = []
        self._debug_grad_flow = getattr(settings, 'debug_grad_flow', False)

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
                      img_mean=settings.img_mean, img_stds=settings.img_stds,
                      channel_order=settings.channel_order)

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

    # Modules that must receive gradient for the architecture to be alive.
    _GRAD_PROBES = ('rem', 'stage1', 'stage2', 'latent_proj', 'coarse_head',
                    'class_embed', 'dit', 'decoder')

    def _report_grad_flow(self):
        """Runs once when --debug-grad-flow is passed; then disables itself."""
        model = self.model.module if hasattr(self.model, 'module') else self.model
        print('--- gradient flow (one batch) ---', flush=True)
        bad = []
        for name in self._GRAD_PROBES:
            mod = getattr(model, name, None)
            if mod is None:
                continue
            gs = [p.grad for p in mod.parameters() if p.requires_grad and p.grad is not None]
            if not gs:
                print(f'  {name:<16} NO GRADIENT', flush=True); bad.append(name); continue
            norm = torch.sqrt(sum((g.float() ** 2).sum() for g in gs))
            finite = torch.isfinite(norm)
            flag = '' if (finite and norm > 0) else '   <-- PROBLEM'
            if not finite or norm == 0:
                bad.append(name)
            print(f'  {name:<16} grad_norm {norm.item():.6e}{flag}', flush=True)
        if bad:
            print(f'  WARNING: no finite non-zero gradient for: {bad}', flush=True)
        else:
            print('  all probed modules received finite non-zero gradients', flush=True)
        print('---------------------------------', flush=True)

    def compute_rf_losses(self, out, label, mask):
        """
        L_total = L_sem + lambda_diff * L_diff + lambda_coarse * L_coarse

        L_sem    = CE(final_logits) + Lovasz(final_logits)
        L_coarse = CE(coarse_logits) + Lovasz(coarse_logits), at latent resolution
        L_diff   = MSE(eps_hat, eps)

        Ignore index 0 is respected in every semantic term. The semantic losses
        run in fp32: the focal/Lovasz path takes log() of a clamped probability
        and sorts per-pixel errors, both of which are fragile in fp16.
        """
        s = self.settings
        logits = out['logits'].float()
        parts = {}

        ce = F.cross_entropy(logits, label, ignore_index=0)
        lov = self.criterion['lovasz'](F.softmax(logits, dim=1), label)
        parts['ce'], parts['lovasz'] = ce, lov
        total = ce + lov

        if s.rf_lambda_coarse > 0:
            h, w = out['latent_hw']
            lab_small = F.interpolate(label.unsqueeze(1).float(), size=(h, w),
                                      mode='nearest').squeeze(1).long()
            cl = out['coarse_logits'].float()
            c_ce = F.cross_entropy(cl, lab_small, ignore_index=0)
            c_lov = self.criterion['lovasz'](F.softmax(cl, dim=1), lab_small)
            parts['coarse'] = c_ce + c_lov
            total = total + s.rf_lambda_coarse * parts['coarse']

        if out.get('eps') is not None and s.rf_lambda_diff > 0:
            parts['diff'] = F.mse_loss(out['eps_hat'].float(), out['eps'].float())
            total = total + s.rf_lambda_diff * parts['diff']

        parts['total'] = total
        return total, parts

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
        self.coarse_metrics.reset()
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
                accum = self.settings.gradient_accumulation_steps
                # zero_grad only at the start of an accumulation window
                if self._micro_step % accum == 0:
                    self.optimizer.zero_grad(set_to_none=True)

                with tools.autocast(enabled=self.fp16_scaler is not None):
                    if self.is_rangeformer_dit:
                        out = self.model(input_feature, labels=input_label)
                        output = out['logits']
                        if out.get('timestep') is not None:
                            self._t_hist = (self._t_hist +
                                            out['timestep'].detach().cpu().tolist())[-2000:]
                        total_loss, parts = self.compute_rf_losses(
                            out, input_label, input_mask)
                        loss_lovasz = parts.get('lovasz', torch.zeros((), device=self.device))
                        loss_focal = parts.get('ce', torch.zeros((), device=self.device))
                        self._last_parts = parts
                    else:
                        output = self.model(input_feature)
                        output_softmax = F.softmax(output, dim=1)
                        total_loss, loss_lovasz, loss_focal = self.compute_losses(
                            output, output_softmax, input_label, input_mask)

                # Scale so the accumulated gradient matches a single large batch
                scaled = total_loss / accum
                is_last_micro = ((self._micro_step + 1) % accum == 0 or
                                 i == total_iter - 1)

                # Skip the DDP all-reduce on non-final micro-steps when possible
                sync_ctx = (self.model.no_sync()
                            if (not is_last_micro and hasattr(self.model, 'no_sync'))
                            else contextlib.nullcontext())
                with sync_ctx:
                    if self.fp16_scaler is None:
                        scaled.backward()
                    else:
                        self.fp16_scaler.scale(scaled).backward()

                if self._debug_grad_flow:
                    self._report_grad_flow()
                    self._debug_grad_flow = False

                if is_last_micro:
                    if self.fp16_scaler is not None:
                        self.fp16_scaler.unscale_(self.optimizer)   # before clipping
                    if self.settings.gradient_clip > 0:
                        self._last_grad_norm = torch.nn.utils.clip_grad_norm_(
                            self._trainable_params, self.settings.gradient_clip)
                    if self.fp16_scaler is None:
                        self.optimizer.step()
                    else:
                        self.fp16_scaler.step(self.optimizer)
                        self.fp16_scaler.update()
                    # scheduler and global step advance per OPTIMIZER UPDATE
                    self.scheduler.step()
                    self.global_step += 1
                    self.optimizer.zero_grad(set_to_none=True)

                self._micro_step += 1
            else:
                with torch.no_grad():
                    assert input_feature.shape[0] == 1, 'the validation batch size has to be 1'

                    if self.is_rangeformer_dit:
                        # No labels are passed: inference must never see GT.
                        # Falls through to the shared metric/logging block below
                        # so the new branch reports exactly like the existing one.
                        m = self.model.module if hasattr(self.model, 'module') else self.model
                        with tools.autocast(enabled=self.fp16_scaler is not None):
                            inf = m.infer(input_feature)
                        output = inf['logits']
                        output_softmax = F.softmax(output, dim=1)
                        total_loss, loss_lovasz, loss_focal = self.compute_losses(
                            output, output_softmax, input_label, input_mask)
                        # coarse mIoU alongside the final one: is the DiT beating
                        # the anchor it was initialised from?
                        with torch.no_grad():
                            coarse_up = F.interpolate(
                                inf['coarse_logits'].float(), size=input_label.shape[-2:],
                                mode='bilinear', align_corners=False)
                            self.coarse_metrics.addBatch(
                                coarse_up.argmax(1).cpu(), input_label.cpu())

                    else:
                        with tools.autocast(enabled=self.fp16_scaler is not None):
                            lidar_pred = sliding_window_inference(
                                self.model,
                                input_feature,
                                window_size=self.settings.window_size,
                                window_stride=self.settings.window_stride,
                                n_cls=self.settings.n_classes,
                                batch_size=1)

                        output = lidar_pred.unsqueeze(0)  # [C,H,W] -> [1,C,H,W]
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
                    payload = {
                        f'{mode}/iter_loss': loss.item(),
                        f'{mode}/iter_mean_acc': mean_acc.item(),
                        f'{mode}/iter_mean_iou': mean_iou.item(),
                        'train/lr': lr,
                        'epoch': epoch + 1,
                    }
                    if self.is_rangeformer_dit and mode == 'Train':
                        # Individual terms, so the diffusion objective can be
                        # read separately from the segmentation one.
                        for k in ('ce', 'lovasz', 'coarse', 'diff'):
                            if k in self._last_parts:
                                payload[f'{mode}/loss_{k}'] = float(self._last_parts[k])
                        payload['train/grad_norm'] = float(self._last_grad_norm)
                        payload['train/micro_batch'] = input_feature.shape[0]
                        payload['train/effective_batch'] = (
                            input_feature.shape[0] *
                            self.settings.gradient_accumulation_steps *
                            self.settings.world_size)
                        payload['train/global_step'] = self.global_step
                        if torch.cuda.is_available():
                            payload['train/gpu_mem_gb'] = (
                                torch.cuda.max_memory_allocated() / 2 ** 30)
                        if self._t_hist:
                            payload['train/timestep_mean'] = float(np.mean(self._t_hist))
                    self.recorder.log_wandb(payload)

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

            epoch_payload = {
                f'{mode}/loss': loss_dict['loss_meter_avg'],
                f'{mode}/loss_focal': loss_dict['loss_focal'],
                f'{mode}/loss_lovasz': loss_dict['loss_lovasz'],
                f'{mode}/mean_iou': mean_iou.item(),
                f'{mode}/mean_acc': mean_acc.item(),
                f'{mode}/mean_recall': mean_recall.item(),
                'epoch': epoch + 1,
            }
            # Per-class IoU, and the coarse anchor the DiT has to beat.
            for ci, ciou in enumerate(class_iou.tolist()):
                if ci in self.ignore_class:
                    continue
                epoch_payload[f'{mode}/IoU_{self.mapped_cls_name[ci]}'] = ciou
            if self.is_rangeformer_dit and mode == 'Validation':
                c_miou, _ = self.coarse_metrics.getIoU()
                epoch_payload[f'{mode}/coarse_mean_iou'] = c_miou.item()
                epoch_payload[f'{mode}/final_minus_coarse'] = (
                    mean_iou.item() - c_miou.item())
                if self.recorder is not None:
                    self.recorder.logger.info(
                        f'>>> coarse mIoU {c_miou.item():.4f}  '
                        f'final mIoU {mean_iou.item():.4f}  '
                        f'diffusion gain {mean_iou.item() - c_miou.item():+.4f}')
            self.recorder.log_wandb(epoch_payload)

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
        if self.settings.model_type == 'rangeformer_dit_seg':
            model = build_rangeformer_dit_model(self.settings)
            if self.recorder is not None:
                stats = model.parameter_groups()
                self.recorder.logger.info('RangeFormerDiTSeg parameters:')
                for k, v in stats.items():
                    self.recorder.logger.info(f'==> {k}: {v:,}')
            return model

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
        checkpoint_data['scheduler'] = self.trainer.scheduler.state_dict()
        checkpoint_data['global_step'] = self.trainer.global_step
        checkpoint_data['micro_step'] = self.trainer._micro_step
        checkpoint_data['config'] = self.settings.config
        checkpoint_data['model_type'] = self.settings.model_type
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
    parser.add_argument('--model_type', type=str, default=None,
                        help="'rangedit' (default) or 'rangeformer_dit_seg'")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=None,
                        help='micro-batches per optimizer update')
    parser.add_argument('--debug-grad-flow', dest='debug_grad_flow',
                        action='store_true',
                        help='report per-module gradient norms for one batch, then stop')
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
