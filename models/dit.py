"""
Diffusion Transformer (DiT) backbone re-used as an off-the-shelf feature
extractor for range-view semantic segmentation.

The module definitions below mirror github.com/facebookresearch/DiT exactly at
the `state_dict` level (same submodule names, same parameter shapes), so the
official `DiT-XL-2-{256x256,512x512}.pt` checkpoints load without any key
surgery beyond the two layers we deliberately replace:

  * `x_embedder` : the 2x2 latent patch embedding, replaced by the range-image
                   convolutional stem (see `stems.py`),
  * `final_layer.linear` : the projection back to noise/latent patches,
                   replaced by the segmentation decoder (see `decoders.py`).

`Attention` and `Mlp` are written out here rather than imported from timm so
that the parameter names stay pinned (`attn.qkv`, `attn.proj`, `mlp.fc1`,
`mlp.fc2`) across timm versions and so the repository keeps a small dependency
footprint.

Note on conditioning: every DiT block is modulated by adaLN-Zero from a
conditioning vector `c = t_embedder(t) + y_embedder(y)`. Semantic segmentation
has neither a diffusion timestep nor an ImageNet class, so we feed a fixed
timestep together with the class-free ("null") label embedding and let a
zero-initialised offset be learnt on top of it -- the pre-trained modulation
pathway is therefore the starting point rather than something we throw away.
"""

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_utils import (get_2d_sincos_pos_embed,
                          resize_pos_embed)


# --------------------------------------------------------------------------- #
#                      Released DiT model configurations                      #
# --------------------------------------------------------------------------- #

# Only the DiT-XL/2 weights have been publicly released by Meta; the smaller
# configurations are kept so the backbone can also be trained from scratch.
DIT_CONFIGS = {
    'DiT-XL/2': dict(depth=28, hidden_size=1152, patch_size=2,  num_heads=16),
    'DiT-XL/4': dict(depth=28, hidden_size=1152, patch_size=4,  num_heads=16),
    'DiT-XL/8': dict(depth=28, hidden_size=1152, patch_size=8,  num_heads=16),
    'DiT-L/2':  dict(depth=24, hidden_size=1024, patch_size=2,  num_heads=16),
    'DiT-L/4':  dict(depth=24, hidden_size=1024, patch_size=4,  num_heads=16),
    'DiT-L/8':  dict(depth=24, hidden_size=1024, patch_size=8,  num_heads=16),
    'DiT-B/2':  dict(depth=12, hidden_size=768,  patch_size=2,  num_heads=12),
    'DiT-B/4':  dict(depth=12, hidden_size=768,  patch_size=4,  num_heads=12),
    'DiT-B/8':  dict(depth=12, hidden_size=768,  patch_size=8,  num_heads=12),
    'DiT-S/2':  dict(depth=12, hidden_size=384,  patch_size=2,  num_heads=6),
    'DiT-S/4':  dict(depth=12, hidden_size=384,  patch_size=4,  num_heads=6),
    'DiT-S/8':  dict(depth=12, hidden_size=384,  patch_size=8,  num_heads=6),
}

PRETRAINED_DIT_MODELS = {'DiT-XL-2-256x256.pt', 'DiT-XL-2-512x512.pt'}
DIT_CHECKPOINT_URL = 'https://dl.fbaipublicfiles.com/DiT/models/{}'


def find_dit_checkpoint(model_name, cache_dir='pretrained_models'):
    """
    Load an official DiT checkpoint (downloading it on first use) or a local
    checkpoint file. Mirrors `download.find_model` of the DiT repository.
    """
    if model_name in PRETRAINED_DIT_MODELS:
        os.makedirs(cache_dir, exist_ok=True)
        local_path = os.path.join(cache_dir, model_name)
        if not os.path.isfile(local_path):
            url = DIT_CHECKPOINT_URL.format(model_name)
            print(f'Downloading the pre-trained DiT checkpoint from {url}')
            torch.hub.download_url_to_file(url, local_path)
    else:
        local_path = model_name
        if not os.path.isfile(local_path):
            raise FileNotFoundError(f'Could not find the DiT checkpoint at {local_path}')

    checkpoint = torch.load(local_path, map_location='cpu')
    for key in ('ema', 'model', 'state_dict'):
        if isinstance(checkpoint, dict) and key in checkpoint:
            checkpoint = checkpoint[key]
            break
    return checkpoint


# --------------------------------------------------------------------------- #
#                          DiT building blocks                                #
# --------------------------------------------------------------------------- #

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. The last row of the table
    is the "null" embedding used by DiT for classifier-free guidance; that is
    the row we use as the class-agnostic conditioning for segmentation.
    """

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def forward(self, labels):
        return self.embedding_table(labels)


class Attention(nn.Module):
    """Multi-head self-attention with timm-compatible parameter names."""

    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim must be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop_rate = attn_drop

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if hasattr(F, 'scaled_dot_product_attention'):
            x = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.attn_drop_rate if self.training else 0.)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = self.attn_drop(attn.softmax(dim=-1))
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class Mlp(nn.Module):
    """Feed-forward block with timm-compatible parameter names."""

    def __init__(self, in_features, hidden_features, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class DropPath(nn.Module):
    """Stochastic depth on the residual branches (disabled by default)."""

    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob)
        return x * mask / keep_prob


class DiTBlock(nn.Module):
    """A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning."""

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout=0.0, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, proj_drop=dropout)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim,
                       act_layer=lambda: nn.GELU(approximate='tanh'), drop=dropout)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )
        self.drop_path = DropPath(drop_path)

    def forward(self, x, c):
        (shift_msa, scale_msa, gate_msa,
         shift_mlp, scale_mlp, gate_mlp) = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + self.drop_path(gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa)))
        x = x + self.drop_path(gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp)))
        return x


class FinalModulation(nn.Module):
    """
    The DiT final layer without its `linear` projection: the modulated
    LayerNorm is kept (and pre-trained weights are reused) while the projection
    back to latent patches is replaced by the segmentation decoder. It plays
    the role of the `encoder.norm` LayerNorm of RangeViT.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        return modulate(self.norm_final(x), shift, scale)


# --------------------------------------------------------------------------- #
#                        DiT encoder for segmentation                         #
# --------------------------------------------------------------------------- #

class DiTEncoder(nn.Module):
    """
    The transformer trunk of DiT, wired for dense prediction:

      range image -> stem (patchification + patch encoding)
                  -> + positional embedding
                  -> N x DiTBlock(adaLN-Zero conditioning)
                  -> modulated final LayerNorm
                  -> patch tokens for the decoder

    Everything except the stem, the positional embedding and the conditioning
    offset comes from the pre-trained DiT checkpoint.
    """

    def __init__(
        self,
        image_size,
        patch_size,
        patch_stride,
        n_layers,
        d_model,
        n_heads,
        mlp_ratio=4.0,
        channels=5,
        dropout=0.0,
        drop_path_rate=0.0,
        conv_stem='ConvStem',
        stem_base_channels=32,
        stem_hidden_dim=None,
        dit_num_classes=1000,
        class_dropout_prob=0.1,
        cond_timestep=0.0,
        cond_class=None,
        learnable_cond=True,
        cond_mode='static',
        learnable_pos_emb=True,
        fusion_layers=None,
        adapter_layers=None,
        adapter_channels=64,
        adapter_heads=8,
    ):
        super().__init__()
        from .stems import ConvStem, PatchEmbedding  # local import: avoids a cycle
        from .adapter import SpatialInjector, SpatialPriorModule

        self.conv_stem = conv_stem
        if self.conv_stem == 'none':
            self.patch_embed = PatchEmbedding(
                image_size, patch_size, patch_stride, d_model, channels)
        else:
            assert tuple(patch_size) == tuple(patch_stride), \
                'patch_size must equal patch_stride when a convolutional stem is used'
            self.patch_embed = ConvStem(
                in_channels=channels,
                base_channels=stem_base_channels,
                img_size=image_size,
                patch_stride=patch_stride,
                embed_dim=d_model,
                flatten=True,
                hidden_dim=stem_hidden_dim)

        self.image_size = image_size
        self.patch_size = tuple(patch_size)
        self.patch_stride = tuple(patch_stride)
        self.n_layers = n_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)
        self.cond_mode = cond_mode

        # ---- Positional embedding (sin-cos, as in DiT; fine-tuned by default) ----
        grid_size = self.patch_embed.grid_size
        pos_embed = get_2d_sincos_pos_embed(d_model, grid_size)
        self.pos_embed = nn.Parameter(
            torch.from_numpy(pos_embed).float().unsqueeze(0),
            requires_grad=learnable_pos_emb)

        # ---- adaLN-Zero conditioning ----
        self.t_embedder = TimestepEmbedder(d_model)
        self.y_embedder = LabelEmbedder(dit_num_classes, d_model, class_dropout_prob)
        if cond_class is None:
            # The "null" row of the label table (class-free conditioning).
            cond_class = dit_num_classes if class_dropout_prob > 0 else 0
        self.register_buffer('cond_timestep', torch.tensor([float(cond_timestep)]))
        self.register_buffer('cond_label', torch.tensor([int(cond_class)], dtype=torch.long))

        self.learnable_cond = learnable_cond
        if learnable_cond:
            self.cond_offset = nn.Parameter(torch.zeros(1, d_model))
        if cond_mode == 'context':
            # Conditioning also sees the scene: zero-initialised so that
            # training starts exactly from the pre-trained modulation.
            self.context_proj = nn.Linear(d_model, d_model)
            nn.init.constant_(self.context_proj.weight, 0)
            nn.init.constant_(self.context_proj.bias, 0)
        elif cond_mode != 'static':
            raise ValueError(f'Unknown cond_mode: {cond_mode}')

        # ---- Transformer blocks ----
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.blocks = nn.ModuleList([
            DiTBlock(d_model, n_heads, mlp_ratio=mlp_ratio, dropout=dropout, drop_path=dpr[i])
            for i in range(n_layers)])

        self.final_layer = FinalModulation(d_model)

        # ---- Multi-level feature fusion ----
        # The last DiT block is specialised for noise prediction, so for a
        # frozen trunk it is not necessarily where the most semantic features
        # live. When `fusion_layers` is set we also tap intermediate blocks,
        # normalise each tap, concatenate and project back to `d_model`.
        self.fusion_layers = sorted(set(fusion_layers)) if fusion_layers else []
        for idx in self.fusion_layers:
            if not 0 <= idx < n_layers:
                raise ValueError(f'fusion layer {idx} out of range for {n_layers} blocks')

        if self.fusion_layers:
            n_taps = len(self.fusion_layers) + 1        # + the final_layer output
            self.fusion_norms = nn.ModuleList(
                [nn.LayerNorm(d_model, eps=1e-6) for _ in range(len(self.fusion_layers))])
            self.fusion_proj = nn.Linear(n_taps * d_model, d_model, bias=True)

        # ---- Range-view spatial-prior adapter (ViT-Adapter style) ----
        # The frozen blocks otherwise see LiDAR structure only once, at the
        # stem. These injectors re-supply it at several depths.
        self.adapter_layers = sorted(set(adapter_layers)) if adapter_layers else []
        for idx in self.adapter_layers:
            if not 0 <= idx < n_layers:
                raise ValueError(f'adapter layer {idx} out of range for {n_layers} blocks')

        if self.adapter_layers:
            self.spatial_prior = SpatialPriorModule(
                in_channels=channels, base_channels=adapter_channels, d_model=d_model)
            self.injectors = nn.ModuleList(
                [SpatialInjector(d_model, n_heads=adapter_heads, dropout=dropout)
                 for _ in self.adapter_layers])

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.blocks.apply(_basic_init)
        self.final_layer.apply(_basic_init)

        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # adaLN-Zero: zero-out the modulation layers (overwritten when the
        # pre-trained DiT weights are loaded).
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)

        # Fusion starts as a no-op: the projection passes the final_layer
        # output straight through and ignores the intermediate taps, so
        # training begins exactly where the single-tap model would.
        if self.fusion_layers:
            nn.init.constant_(self.fusion_proj.weight, 0)
            nn.init.constant_(self.fusion_proj.bias, 0)
            d = self.d_model
            with torch.no_grad():
                self.fusion_proj.weight[:, -d:] = torch.eye(d)

    @torch.jit.ignore
    def no_weight_decay(self):
        names = {'pos_embed'}
        if self.learnable_cond:
            names.add('cond_offset')
        return names

    def get_grid_size(self, H, W):
        return self.patch_embed.get_grid_size(H, W)

    def get_conditioning(self, x):
        """Build the adaLN-Zero conditioning vector `c` for a batch of tokens."""
        B = x.shape[0]
        t = self.cond_timestep.expand(B)
        y = self.cond_label.expand(B)
        c = self.t_embedder(t) + self.y_embedder(y)
        if self.learnable_cond:
            c = c + self.cond_offset
        if self.cond_mode == 'context':
            c = c + self.context_proj(x.mean(dim=1))
        return c

    def forward(self, im, return_features=False):
        B, _, H, W = im.shape

        x, skip = self.patch_embed(im)                 # (B, N, D), stem skip features

        pos_embed = self.pos_embed
        if x.shape[1] != pos_embed.shape[1]:
            grid_H, grid_W = self.get_grid_size(H, W)
            pos_embed = resize_pos_embed(
                pos_embed, self.patch_embed.grid_size, (grid_H, grid_W), num_extra_tokens=0)

        x = self.dropout(x + pos_embed)

        c = self.get_conditioning(x)

        prior = self.spatial_prior(im) if self.adapter_layers else None
        inject_at = {idx: n for n, idx in enumerate(self.adapter_layers)}
        tap_at = set(self.fusion_layers)

        taps = []
        for i, blk in enumerate(self.blocks):
            x = blk(x, c)
            if i in inject_at:
                x = self.injectors[inject_at[i]](x, prior)
            if i in tap_at:
                taps.append(x)

        x = self.final_layer(x, c)
        if not self.fusion_layers:
            return x, skip

        taps = [norm(t) for norm, t in zip(self.fusion_norms, taps)]
        taps.append(x)
        return self.fusion_proj(torch.cat(taps, dim=-1)), skip
