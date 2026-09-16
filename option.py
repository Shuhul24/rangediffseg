"""
Configuration handling: reads `config.yaml`, flattens it into a settings object
and applies the command-line overrides.

Same role as `option.py` of RangeViT (github.com/valeoai/rangevit), adapted to
the nested YAML layout of this repository.
"""

import os

import yaml


class Option(object):
    def __init__(self, config_path, args=None):
        self.config_path = config_path
        self.config = yaml.safe_load(open(config_path, 'r'))

        ds_cfg = self.config.get('dataset', {})
        m_cfg = self.config.get('model', {})
        t_cfg = self.config.get('training', {})

        # ---------------------------- General ---------------------------- #
        self.seed = t_cfg.get('seed', 1)
        self.num_workers = t_cfg.get('num_workers', 4)
        self.id = t_cfg.get('id', 'rangedit')

        # ----------------------------- Weights & Biases ------------------- #
        self.use_wandb = t_cfg.get('use_wandb', False)
        self.wandb_project = t_cfg.get('wandb_project', 'rangediffseg')
        self.wandb_entity = t_cfg.get('wandb_entity', None)
        self.wandb_name = t_cfg.get('wandb_name', self.id)
        self.wandb_mode = t_cfg.get('wandb_mode', None)

        # ------------------------------ Data ----------------------------- #
        self.dataset = ds_cfg.get('name', 'SemanticKitti')
        self.data_root = ds_cfg.get('root', None)
        self.n_classes = ds_cfg.get('n_classes', 20)
        self.in_channels = ds_cfg.get('in_channels', 5)
        self.proj_h = ds_cfg.get('proj_h', 64)
        self.proj_w = ds_cfg.get('proj_w', 2048)
        self.fov_up = ds_cfg.get('fov_up', 3.0)
        self.fov_down = ds_cfg.get('fov_down', -25.0)
        self.img_mean = ds_cfg.get('img_mean', None)
        self.img_stds = ds_cfg.get('img_stds', None)
        self.augmentation = ds_cfg.get('augmentation', {})

        # SemanticKITTI: sequence folders
        self.train_seqs = ds_cfg.get('train_seqs', [])
        self.val_seqs = ds_cfg.get('val_seqs', [])
        self.test_seqs = ds_cfg.get('test_seqs', [])

        # nuScenes: metadata version and where the keyframe index comes from
        self.version = ds_cfg.get('version', 'v1.0-trainval')
        self.info_path = ds_cfg.get('info_path', None)
        self.index_cache_dir = ds_cfg.get('index_cache_dir', './cache')

        # Focal-loss class weighting: RangeViT weights SemanticKITTI by point
        # frequency and leaves nuScenes uniform.
        self.use_cls_freq_weights = ds_cfg.get(
            'use_cls_freq_weights', self.dataset == 'SemanticKitti')
        # 'log_inv' (RangeViT), 'salsanext', 'inv_sqrt' or 'uniform'
        self.cls_weight_scheme = ds_cfg.get('cls_weight_scheme', 'log_inv')
        self.cls_weight_eps = float(ds_cfg.get('cls_weight_eps', 1e-3))

        # ---- Tail-class handling ---------------------------------------- #
        # Training-label ids treated as the tail. Used to bias the crop window,
        # and reported as a separate `tail_mean_iou` so a collapsing rare class
        # is visible without reading the per-class table.
        default_rare = ([2, 3, 4, 5, 6, 7, 8, 16, 18, 19]
                        if self.dataset == 'SemanticKitti' else [])
        self.rare_classes = ds_cfg.get('rare_classes', default_rare)
        self.rare_crop_p = float(ds_cfg.get('rare_crop_p', 0.0))
        self.copy_paste = ds_cfg.get('copy_paste', {}) or {}
        self.mix = ds_cfg.get('mix', {}) or {}

        # ----------------------------- Model ----------------------------- #
        self.backbone = m_cfg.get('backbone', 'DiT-XL/2')
        self.pretrained_model = m_cfg.get('pretrained_model', None)
        self.image_size = m_cfg.get('image_size', [64, 384])
        self.patch_size = m_cfg.get('patch_size', [2, 8])
        self.patch_stride = m_cfg.get('patch_stride', [2, 8])
        self.window_size = m_cfg.get('window_size', self.image_size)
        self.window_stride = m_cfg.get('window_stride', self.image_size)
        self.conv_stem = m_cfg.get('conv_stem', 'ConvStem')
        self.stem_base_channels = m_cfg.get('stem_base_channels', 32)
        self.D_h = m_cfg.get('D_h', 256)
        self.decoder = m_cfg.get('decoder', 'up_conv')
        self.decoder_stages = int(m_cfg.get('decoder_stages', 1))
        self.stem_pool = m_cfg.get('stem_pool', 'avg')
        self.skip_filters = m_cfg.get('skip_filters', 0)
        self.dropout = m_cfg.get('dropout', 0.0)
        self.drop_path_rate = m_cfg.get('drop_path_rate', 0.0)

        # Loading pre-trained patch and positional embeddings
        self.reuse_pos_emb = m_cfg.get('reuse_pos_emb', True)
        self.reuse_patch_emb = m_cfg.get('reuse_patch_emb', False)
        self.learnable_pos_emb = m_cfg.get('learnable_pos_emb', True)

        # adaLN-Zero conditioning of the DiT blocks
        self.cond_timestep = m_cfg.get('cond_timestep', 0)
        self.cond_class = m_cfg.get('cond_class', None)
        self.learnable_cond = m_cfg.get('learnable_cond', True)
        self.cond_mode = m_cfg.get('cond_mode', 'static')

        # ---- Trunk capacity / where features are read from ----
        # feature_taps: extra block indices whose output is fused into the
        #   encoder output (zero-initialised, so [] reproduces the plain model).
        # n_trunk_blocks: run only the first N blocks; 0 bypasses the trunk.
        # grad_checkpointing: recompute block activations instead of storing them.
        self.feature_taps = m_cfg.get('feature_taps', []) or []
        self.n_trunk_blocks = m_cfg.get('n_trunk_blocks', None)
        self.grad_checkpointing = m_cfg.get('grad_checkpointing', False)
        self.head_prior_bias = m_cfg.get('head_prior_bias', False)

        # Freezing the pre-trained DiT trunk
        self.freeze_dit_encoder = m_cfg.get('freeze_dit_encoder', True)
        self.unfreeze_adaln = m_cfg.get('unfreeze_adaln', True)
        self.adaln_bias_only = m_cfg.get('adaln_bias_only', True)
        self.unfreeze_attn = m_cfg.get('unfreeze_attn', False)
        self.unfreeze_ffn = m_cfg.get('unfreeze_ffn', False)

        # ---------------------------- Training --------------------------- #
        self.n_epochs = t_cfg.get('n_epochs', 50)
        self.batch_size = t_cfg.get('batch_size', 4)
        self.batch_size_val = t_cfg.get('batch_size_val', 1)
        self.lr = float(t_cfg.get('lr', 2e-4))
        self.weight_decay = float(t_cfg.get('weight_decay', 0.01))
        self.warmup_epochs = t_cfg.get('warmup_epochs', 2)
        self.use_fp16 = t_cfg.get('use_fp16', True)
        self.val_frequency = t_cfg.get('val_frequency', 5)
        self.grad_clip = float(t_cfg.get('grad_clip', 0.0))
        self.ema_decay = float(t_cfg.get('ema_decay', 0.0))
        self.ema_warmup_steps = int(t_cfg.get('ema_warmup_steps', 2000))
        self.val_live_weights = bool(t_cfg.get('val_live_weights', False))
        # Point-wise (3-D) validation: the metric the benchmark actually scores,
        # and therefore the one checkpoints should be selected on.
        self.val_3d = bool(t_cfg.get('val_3d', False))
        self.val_knn = bool(t_cfg.get('val_knn', False))
        self.knn_k = int(t_cfg.get('knn_k', 5))
        self.knn_search = int(t_cfg.get('knn_search', 5))
        self.knn_sigma = float(t_cfg.get('knn_sigma', 1.0))
        self.knn_cutoff = float(t_cfg.get('knn_cutoff', 1.0))
        self.focal_gamma = float(t_cfg.get('focal_gamma', 2.0))
        self.focal_weight = float(t_cfg.get('focal_weight', 1.0))
        self.lovasz_weight = float(t_cfg.get('lovasz_weight', 1.0))
        self.train_result_frequency = t_cfg.get('train_result_frequency', 10)
        self.log_frequency = t_cfg.get('log_frequency', 100)
        self.save_path = t_cfg.get('save_path', './log')

        # ---------------------------- Runtime ---------------------------- #
        self.checkpoint = t_cfg.get('checkpoint', None)
        self.finetune_from = t_cfg.get('finetune_from', None)
        self.finetune_strict = t_cfg.get('finetune_strict', True)
        self.val_only = False
        self.test_split = False
        self.save_eval_results = False

        if args is not None:
            self._apply_args(args)

        self.save_path = os.path.join(self.save_path, 'log_{}'.format(self.id))
        self._check_options()

    def _apply_args(self, args):
        """Command-line arguments override the YAML file when provided."""
        for name in ('data_root', 'save_path', 'id', 'num_workers', 'pretrained_model',
                     'checkpoint', 'finetune_from', 'log_frequency', 'seed', 'batch_size',
                     'n_epochs', 'lr', 'warmup_epochs', 'val_frequency', 'cond_mode',
                     'wandb_project', 'wandb_entity', 'wandb_name', 'wandb_mode',
                     'ema_decay', 'rare_crop_p', 'cls_weight_scheme', 'lovasz_weight',
                     'cond_timestep', 'n_trunk_blocks'):
            value = getattr(args, name, None)
            if value is not None:
                setattr(self, name, value)

        for name in ('val_only', 'test_split', 'save_eval_results', 'use_wandb',
                     'unfreeze_attn', 'unfreeze_ffn', 'grad_checkpointing',
                     'val_3d', 'val_knn'):
            if getattr(args, name, False):
                setattr(self, name, True)

        if getattr(args, 'finetune_non_strict', False):
            self.finetune_strict = False

        window_stride = getattr(args, 'window_stride', None)
        if window_stride is not None:
            self.window_stride = [self.window_stride[0], window_stride]

        # A checkpoint of this model supersedes the off-the-shelf DiT weights.
        if self.checkpoint is not None or self.finetune_from is not None:
            self.pretrained_model = None

    def _check_options(self):
        for name in ('patch_size', 'patch_stride', 'image_size', 'window_size', 'window_stride'):
            value = getattr(self, name)
            assert isinstance(value, (list, tuple)) and len(value) == 2, \
                f'{name} must be a list of two elements, got {value}'
            setattr(self, name, list(value))

        # No skip connection without a convolutional stem or with the linear decoder.
        if self.conv_stem == 'none' or self.decoder == 'linear':
            assert self.skip_filters == 0, \
                'skip_filters must be 0 without a convolutional stem / with the linear decoder'

        # The skip connection carries the hidden dimension of the stem.
        if self.skip_filters > 0:
            assert self.skip_filters == self.D_h, 'skip_filters must equal D_h'

        # With a convolutional stem there is no patch-embedding layer to reuse.
        if self.conv_stem != 'none':
            assert self.patch_size == self.patch_stride, \
                'patch_size must equal patch_stride when a convolutional stem is used'
            assert self.reuse_patch_emb is False, \
                'reuse_patch_emb requires conv_stem: none'

        # Nothing to reuse when training the transformer from scratch.
        if self.pretrained_model is None:
            self.reuse_patch_emb = False
            self.reuse_pos_emb = False

        assert self.dataset in ('SemanticKitti', 'nuScenes'), \
            f'invalid dataset: {self.dataset}'

        if self.dataset == 'SemanticKitti':
            assert self.train_seqs and self.val_seqs, \
                'Set dataset.train_seqs / dataset.val_seqs for SemanticKITTI'

        assert self.data_root is not None, \
            'Set dataset.root in the config file or pass --data_root'

        assert not (self.checkpoint and self.finetune_from), \
            'Use either checkpoint/resume or finetune_from, not both'

        assert self.cls_weight_scheme in ('log_inv', 'salsanext', 'inv_sqrt', 'uniform'), \
            f'invalid dataset.cls_weight_scheme: {self.cls_weight_scheme}'
        assert 0.0 <= self.rare_crop_p <= 1.0, 'dataset.rare_crop_p must be in [0, 1]'
        assert 0.0 <= self.ema_decay < 1.0, 'training.ema_decay must be in [0, 1)'

        if self.val_3d:
            assert self.batch_size_val == 1, \
                'training.val_3d needs batch_size_val: 1 (point counts vary per scan)'
        assert self.val_knn is False or self.val_3d, \
            'training.val_knn only applies when training.val_3d is on'
        assert self.knn_search % 2 == 1, 'training.knn_search must be odd'

        depth = {'DiT-XL': 28, 'DiT-L': 24, 'DiT-B': 12, 'DiT-S': 12}[self.backbone.split('/')[0]]
        n_run = depth if self.n_trunk_blocks is None else int(self.n_trunk_blocks)
        assert 0 <= n_run <= depth, \
            f'model.n_trunk_blocks must be in [0, {depth}] for {self.backbone}, got {n_run}'
        for t in self.feature_taps:
            assert 0 <= int(t) < n_run, \
                (f'model.feature_taps entry {t} is outside the {n_run} blocks that '
                 f'are actually run (block indices are 0-based)')
        assert self.decoder_stages in (1, 2), \
            f'model.decoder_stages must be 1 or 2, got {self.decoder_stages}'
        if self.decoder_stages == 2:
            assert self.decoder == 'up_conv', 'model.decoder_stages=2 needs the up_conv decoder'
            assert self.patch_stride[1] % 2 == 0, \
                f'model.decoder_stages=2 needs an even width patch_stride, got {self.patch_stride}'
        assert self.stem_pool in ('avg', 'dwconv', 'conv'), \
            f'model.stem_pool must be avg, dwconv or conv, got {self.stem_pool}'
        if self.stem_pool != 'avg':
            assert self.conv_stem != 'none', 'model.stem_pool only applies to the ConvStem'

        assert self.head_prior_bias is False or self.decoder == 'up_conv', \
            'model.head_prior_bias needs the up_conv decoder (the linear decoder has no conv head)'

        p_mix = self.mix.get('p_lasermix', 0.0) + self.mix.get('p_polarmix', 0.0)
        assert 0.0 <= p_mix <= 1.0, \
            'dataset.mix p_lasermix + p_polarmix must not exceed 1.0'

        if self.copy_paste.get('enabled', False):
            assert self.copy_paste.get('bank_path'), \
                ('dataset.copy_paste.enabled requires bank_path. Build it with:\n'
                 '  python -m dataset.instance_bank --data_root <root> '
                 '--out ./cache/instance_bank_kitti.npz')
            assert self.dataset == 'SemanticKitti', \
                ('copy-paste needs panoptic instance ids, which only SemanticKITTI '
                 'ships in its .label files; use dataset.mix on nuScenes')

    def check_path(self):
        if os.path.exists(self.save_path):
            print('WARNING: Directory exists: {}'.format(self.save_path))
        os.makedirs(self.save_path, exist_ok=True)
