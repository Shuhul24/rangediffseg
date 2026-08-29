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
                     'wandb_project', 'wandb_entity', 'wandb_name', 'wandb_mode'):
            value = getattr(args, name, None)
            if value is not None:
                setattr(self, name, value)

        for name in ('val_only', 'test_split', 'save_eval_results', 'use_wandb',
                     'unfreeze_attn', 'unfreeze_ffn'):
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

    def check_path(self):
        if os.path.exists(self.save_path):
            print('WARNING: Directory exists: {}'.format(self.save_path))
        os.makedirs(self.save_path, exist_ok=True)
