"""
RangeDiT -- an off-the-shelf Diffusion Transformer (DiT) turned into a LiDAR
semantic-segmentation network for range-view images.

The recipe follows RangeViT (github.com/valeoai/rangevit): keep the pre-trained
transformer trunk as it is, and learn everything that is specific to range
images, i.e.

    stem      : patchification, patch encoding and positional embedding
    decoder   : token grid -> per-pixel class logits

with, in addition, the piece that has no counterpart in a plain ViT:

    conditioning : DiT blocks are modulated by adaLN-Zero from a conditioning
                   vector. We feed the pre-trained (fixed timestep + null
                   class) conditioning and learn a zero-initialised offset on
                   top of it, so training starts exactly from the pre-trained
                   modulation.

Pre-trained weights come from the official DiT release
(github.com/facebookresearch/DiT); the released `DiT-XL-2-*.pt` checkpoints
correspond to the `DiT-XL/2` configuration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .decoders import DecoderLinear, DecoderUpConv
from .dit import DIT_CONFIGS, DiTEncoder, find_dit_checkpoint
from .model_utils import adapt_input_conv, padding, resize_pos_embed, unpadding


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class DiTSegmenter(nn.Module):
    """DiT encoder + segmentation decoder, applied to a whole range image."""

    def __init__(self, encoder, decoder, n_cls):
        super().__init__()
        self.n_cls = n_cls
        self.patch_size = encoder.patch_size
        self.patch_stride = encoder.patch_stride
        self.encoder = encoder
        self.decoder = decoder

    @torch.jit.ignore
    def no_weight_decay(self):
        def with_prefix(prefix, module):
            return set(prefix + name for name in module.no_weight_decay())

        return with_prefix('encoder.', self.encoder).union(with_prefix('decoder.', self.decoder))

    def forward(self, im):
        H_ori, W_ori = im.size(2), im.size(3)
        im = padding(im, self.patch_size)
        H, W = im.size(2), im.size(3)

        x, skip = self.encoder(im, return_features=True)   # (B, N, D)

        feats = self.decoder(x, (H, W), skip)              # (B, n_cls, H, W)
        feats = F.interpolate(feats, size=(H, W), mode='bilinear', align_corners=False)
        return unpadding(feats, (H_ori, W_ori))


class RangeDiT(nn.Module):
    def __init__(
        self,
        in_channels=5,
        n_cls=20,
        backbone='DiT-XL/2',
        image_size=(64, 384),
        pretrained_path=None,
        new_patch_size=(2, 8),
        new_patch_stride=(2, 8),
        reuse_pos_emb=True,
        reuse_patch_emb=False,
        conv_stem='ConvStem',
        stem_base_channels=32,
        stem_hidden_dim=256,
        skip_filters=0,
        decoder='up_conv',
        up_conv_d_decoder=64,
        up_conv_scale_factor=(2, 8),
        dropout=0.0,
        drop_path_rate=0.0,
        cond_timestep=0.0,
        cond_class=None,
        learnable_cond=True,
        cond_mode='static',
        learnable_pos_emb=True,
    ):
        super().__init__()

        if backbone not in DIT_CONFIGS:
            raise NameError(f'Unknown DiT backbone: {backbone}. '
                            f'Available: {sorted(DIT_CONFIGS.keys())}')
        cfg = DIT_CONFIGS[backbone]

        self.n_cls = n_cls
        self.backbone = backbone
        self.d_model = cfg['hidden_size']

        # ---- DiT encoder (pre-trained trunk) + range-image stem ----
        encoder = DiTEncoder(
            image_size=tuple(image_size),
            patch_size=tuple(new_patch_size),
            patch_stride=tuple(new_patch_stride),
            n_layers=cfg['depth'],
            d_model=cfg['hidden_size'],
            n_heads=cfg['num_heads'],
            mlp_ratio=4.0,
            channels=in_channels,
            dropout=dropout,
            drop_path_rate=drop_path_rate,
            conv_stem=conv_stem,
            stem_base_channels=stem_base_channels,
            stem_hidden_dim=stem_hidden_dim,
            cond_timestep=cond_timestep,
            cond_class=cond_class,
            learnable_cond=learnable_cond,
            cond_mode=cond_mode,
            learnable_pos_emb=learnable_pos_emb)

        # ---- Segmentation decoder ----
        if decoder == 'linear':
            seg_decoder = DecoderLinear(
                n_cls=n_cls,
                patch_size=encoder.patch_size,
                d_encoder=encoder.d_model,
                patch_stride=encoder.patch_stride)
            assert skip_filters == 0, 'the linear decoder has no skip connection'
        elif decoder == 'up_conv':
            seg_decoder = DecoderUpConv(
                n_cls=n_cls,
                patch_size=encoder.patch_size,
                d_encoder=encoder.d_model,
                d_decoder=up_conv_d_decoder,
                scale_factor=tuple(up_conv_scale_factor),
                patch_stride=encoder.patch_stride,
                skip_filters=skip_filters)
        else:
            raise ValueError(f'Unknown decoder: {decoder}')

        self.rangedit = DiTSegmenter(encoder, seg_decoder, n_cls=n_cls)

        # ---- Load the off-the-shelf DiT weights ----
        if pretrained_path is not None:
            self.load_pretrained_dit(
                pretrained_path,
                in_channels=in_channels,
                image_size=image_size,
                new_patch_size=new_patch_size,
                new_patch_stride=new_patch_stride,
                reuse_pos_emb=reuse_pos_emb,
                reuse_patch_emb=reuse_patch_emb,
                conv_stem=conv_stem)

    # ------------------------------------------------------------------ #
    #                     Pre-trained weight loading                     #
    # ------------------------------------------------------------------ #

    def load_pretrained_dit(self, pretrained_path, in_channels, image_size,
                            new_patch_size, new_patch_stride,
                            reuse_pos_emb, reuse_patch_emb, conv_stem):
        print(f'Loading pretrained parameters from {pretrained_path}')
        pretrained_state_dict = find_dit_checkpoint(pretrained_path)
        pretrained_state_dict = {'encoder.' + k: v for k, v in pretrained_state_dict.items()}

        old_state_dict = self.rangedit.state_dict()

        # -- Positional embeddings: the DiT grid is square (16x16 for the
        #    256x256 model), the range-view grid is wide, so we resize it.
        if reuse_pos_emb and 'encoder.pos_embed' in pretrained_state_dict:
            print('Reusing positional embeddings.')
            gs_new_h = int((image_size[0] - new_patch_size[0]) // new_patch_stride[0] + 1)
            gs_new_w = int((image_size[1] - new_patch_size[1]) // new_patch_stride[1] + 1)
            pretrained_state_dict['encoder.pos_embed'] = resize_pos_embed(
                pretrained_state_dict['encoder.pos_embed'],
                grid_old_shape=None,
                grid_new_shape=(gs_new_h, gs_new_w),
                num_extra_tokens=0)
        else:
            pretrained_state_dict.pop('encoder.pos_embed', None)

        # -- Patch embeddings: only meaningful without a convolutional stem.
        if reuse_patch_emb:
            assert conv_stem == 'none', 'no patch embedding to reuse when a convolutional stem is used'
            print('Reusing patch embeddings.')
            weight = adapt_input_conv(in_channels, pretrained_state_dict['encoder.x_embedder.proj.weight'])
            _, _, ks_h, ks_w = old_state_dict['encoder.patch_embed.proj.weight'].shape
            weight = F.interpolate(weight, size=(ks_h, ks_w), mode='bilinear', align_corners=False)
            pretrained_state_dict['encoder.patch_embed.proj.weight'] = weight
            pretrained_state_dict['encoder.patch_embed.proj.bias'] = \
                pretrained_state_dict['encoder.x_embedder.proj.bias']
        pretrained_state_dict.pop('encoder.x_embedder.proj.weight', None)
        pretrained_state_dict.pop('encoder.x_embedder.proj.bias', None)

        # -- The DiT output head predicts noise/latent patches: dropped in
        #    favour of the segmentation decoder. The modulated final LayerNorm
        #    (`final_layer.adaLN_modulation`) is kept.
        pretrained_state_dict.pop('encoder.final_layer.linear.weight', None)
        pretrained_state_dict.pop('encoder.final_layer.linear.bias', None)

        # -- Drop anything whose shape does not match (e.g. a checkpoint from
        #    a different DiT configuration).
        for key in list(pretrained_state_dict.keys()):
            if key not in old_state_dict:
                continue
            if pretrained_state_dict[key].shape != old_state_dict[key].shape:
                print(f'Skipping {key}: shape mismatch '
                      f'{tuple(pretrained_state_dict[key].shape)} vs {tuple(old_state_dict[key].shape)}')
                del pretrained_state_dict[key]

        msg = self.rangedit.load_state_dict(pretrained_state_dict, strict=False)
        print(f'{msg}')

    # ------------------------------------------------------------------ #
    #                          Freezing helpers                          #
    # ------------------------------------------------------------------ #

    def freeze_encoder(self, unfreeze_adaln=True, adaln_bias_only=True,
                       unfreeze_attn=False, unfreeze_ffn=False):
        """
        Freeze the pre-trained DiT trunk, exactly as RangeViT freezes the ViT
        encoder. The stem, the positional embedding, the conditioning offset
        and the decoder always stay trainable.

        DiT blocks use `elementwise_affine=False` LayerNorms, so there are no
        LayerNorm affine parameters to unfreeze: the DiT counterpart of
        RangeViT's `unfreeze_layernorm` is `unfreeze_adaln`, which trains the
        adaLN-Zero modulation (per-block scales, shifts and residual gates).

        `adaln_bias_only` trains just the bias of that modulation. The
        modulation is `Linear(c)`, and with a conditioning vector that does not
        depend on the input (`cond_mode: static`) its output is one constant
        vector per block, which the bias alone can already reach: the bias has
        the same expressive power here as the full matrix, for 1/d_model of the
        parameters (0.2 M instead of 223 M on DiT-XL/2).
        """
        print('==> Freeze the DiT encoder (without the pos_embed, stem and conditioning)')
        for param in self.rangedit.encoder.blocks.parameters():
            param.requires_grad = False
        for param in self.rangedit.encoder.t_embedder.parameters():
            param.requires_grad = False
        for param in self.rangedit.encoder.y_embedder.parameters():
            param.requires_grad = False
        for param in self.rangedit.encoder.final_layer.parameters():
            param.requires_grad = False

        if unfreeze_adaln:
            def unfreeze(modulation):
                params = (modulation[-1].bias,) if adaln_bias_only else modulation.parameters()
                for param in params:
                    param.requires_grad = True

            print('==> Unfreeze the adaLN-Zero modulation layers'
                  + (' (bias only)' if adaln_bias_only else ''))
            for block in self.rangedit.encoder.blocks:
                unfreeze(block.adaLN_modulation)
            unfreeze(self.rangedit.encoder.final_layer.adaLN_modulation)

        if unfreeze_attn:
            print('==> Unfreeze the ATTN layers: qkv and proj')
            for block in self.rangedit.encoder.blocks:
                for param in block.attn.parameters():
                    param.requires_grad = True

        if unfreeze_ffn:
            print('==> Unfreeze the FFN layers: mlp.fc1 and mlp.fc2')
            for block in self.rangedit.encoder.blocks:
                for param in block.mlp.parameters():
                    param.requires_grad = True

    def counter_model_parameters(self):
        stats = {}
        stats['total_num_parameters'] = count_parameters(self.rangedit)
        stats['decoder_num_parameters'] = count_parameters(self.rangedit.decoder)
        stats['stem_num_parameters'] = count_parameters(self.rangedit.encoder.patch_embed)
        stats['encoder_num_parameters'] = (
            count_parameters(self.rangedit.encoder) - stats['stem_num_parameters'])
        return stats

    def forward(self, *args):
        return self.rangedit(*args)
