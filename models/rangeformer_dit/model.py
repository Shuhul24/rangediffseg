"""
RangeFormerDiTSeg -- the full experimental model.

Training                                   Inference (no GT anywhere)
--------                                   --------------------------
x [B,5,H,W]                                x [B,5,H,W]
 -> REM -> S1 -> F1 -> S2 -> proj -> C       -> ... -> C
 -> coarse_logits = head(C)                  -> coarse_logits = head(C)
 -> S0 = embed(downsample_nn(labels))        -> S_coarse = softmax(coarse) @ E
 -> S_t = q_sample(S0, t, eps)               -> S_tstart = q_sample(S_coarse, t_start)
 -> eps_hat = DiT(S_t, t, C)                 -> a few DDIM steps -> S0_hat
 -> S0_hat = (S_t - sqrt(1-ab) eps_hat)/sqrt(ab)
 -> logits = Decoder(S0_hat, F1)             -> logits = Decoder(S0_hat, F1)

The decoder is fed only by S0_hat, so L_sem backpropagates through the DiT and
into the encoder. There is no clean C -> decoder shortcut.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion import GaussianDiffusion

from .decoder import Decoder
from .encoder import EncoderStage
from .rem import REM


class RangeFormerDiTSeg(nn.Module):
    def __init__(self,
                 in_channels=5,
                 n_classes=20,
                 ignore_index=0,
                 rem_dims=(64, 128, 128),
                 encoder_dims=(128, 256),
                 encoder_depths=(2, 2),
                 encoder_heads=(2, 4),
                 encoder_sr_ratios=(8, 4),
                 circular_padding=False,
                 latent_downsample=2,
                 cond_dim=256,
                 semantic_latent_dim=256,
                 dit_hidden_dim=256,
                 dit_depth=3,
                 dit_heads=8,
                 dit_mlp_ratio=4.0,
                 dit_dropout=0.0,
                 cond_sr_ratio=2,
                 grad_checkpointing=False,
                 bottleneck_type='dit',          # 'dit' | 'transformer'  (ablation B)
                 use_diffusion=True,             # ablation C turns this off
                 decoder_dims=(128, 128),
                 use_gated_skip=False,           # ablation E
                 diffusion_timesteps=1000,
                 diffusion_schedule='cosine',
                 inference_steps=3,
                 inference_t_start=200,
                 s0_clamp=10.0):
        super().__init__()
        assert in_channels == 5, f'expected 5 network channels, got {in_channels}'
        assert bottleneck_type in ('dit', 'transformer')

        self.n_classes = n_classes
        self.ignore_index = ignore_index
        self.use_diffusion = use_diffusion and bottleneck_type == 'dit'
        self.semantic_latent_dim = semantic_latent_dim
        self.inference_steps = inference_steps
        self.inference_t_start = inference_t_start
        self.s0_clamp = s0_clamp

        # ---- encoder ------------------------------------------------------
        self.rem = REM(in_channels, rem_dims)
        self.stage1 = EncoderStage(self.rem.out_channels, encoder_dims[0],
                                   encoder_depths[0], encoder_heads[0],
                                   encoder_sr_ratios[0], circular=circular_padding)
        self.stage2 = EncoderStage(encoder_dims[0], encoder_dims[1],
                                   encoder_depths[1], encoder_heads[1],
                                   encoder_sr_ratios[1], circular=circular_padding)
        # Cheap patch merge so the DiT sees ~H/8 x W/8 rather than H/4 x W/4.
        self.latent_proj = (nn.Sequential(
            nn.Conv2d(encoder_dims[1], cond_dim, 3, stride=latent_downsample, padding=1),
            nn.BatchNorm2d(cond_dim), nn.LeakyReLU(inplace=True))
            if latent_downsample > 1 else nn.Conv2d(encoder_dims[1], cond_dim, 1))

        # ---- coarse head: auxiliary supervision + diffusion initialisation --
        self.coarse_head = nn.Conv2d(cond_dim, n_classes, kernel_size=1)

        # ---- semantic latent ----------------------------------------------
        self.class_embed = nn.Embedding(n_classes, semantic_latent_dim)
        nn.init.normal_(self.class_embed.weight, std=0.02)

        # ---- bottleneck ----------------------------------------------------
        from .dit import ConditionalDiT
        self.dit = ConditionalDiT(
            dim=dit_hidden_dim, depth=dit_depth, num_heads=dit_heads,
            mlp_ratio=dit_mlp_ratio, dropout=dit_dropout, cond_dim=cond_dim,
            cond_sr_ratio=cond_sr_ratio, grad_checkpointing=grad_checkpointing,
            timestep_conditioning=(bottleneck_type == 'dit'))
        assert dit_hidden_dim == semantic_latent_dim, \
            'the DiT operates directly on the semantic latent; keep the dims equal'

        # ---- decoder -------------------------------------------------------
        self.decoder = Decoder(semantic_latent_dim, decoder_dims, n_classes,
                               skip_dim=encoder_dims[0], use_gated_skip=use_gated_skip)

        # Reuses the repository's DDPM scheduler.
        self.diffusion = GaussianDiffusion(diffusion_timesteps, diffusion_schedule)

    # ------------------------------------------------------------------ utils

    def _sync_diffusion_device(self, device):
        """
        GaussianDiffusion is a plain object, not an nn.Module, so its schedule
        tensors do not follow model.to(device). Move them once, lazily.
        """
        if self.diffusion.sqrt_ab.device != device:
            self.diffusion._to(device)

    def _ab(self, name, t, ndim):
        arr = getattr(self.diffusion, name).to(t.device)
        return self.diffusion._expand(arr, t, ndim)

    def encode(self, x):
        """x [B,5,H,W] -> F1 [B,C1,H/2,W/2], C [B,Cc,H/8,W/8]"""
        f = self.rem(x)
        f1 = self.stage1(f)
        f2 = self.stage2(f1)
        return f1, self.latent_proj(f2)

    def semantic_latent_from_labels(self, labels, hw):
        """
        GT labels [B,H,W] -> S0 [B,D,h,w]. Nearest-neighbour downsampling only:
        integer class ids must never be interpolated.
        """
        lab = labels.unsqueeze(1).float()
        lab = F.interpolate(lab, size=hw, mode='nearest').squeeze(1).long()
        lab = lab.clamp(0, self.n_classes - 1)
        return self.class_embed(lab).permute(0, 3, 1, 2).contiguous(), lab

    def semantic_latent_from_coarse(self, coarse_logits):
        """Soft embedding sum_c P_c * E_c -- the inference-time initialisation."""
        p = coarse_logits.softmax(dim=1)                       # [B,K,h,w]
        return torch.einsum('bkhw,kd->bdhw', p, self.class_embed.weight)

    def predict_x0(self, s_t, t, eps_hat):
        """
        S0_hat = (S_t - sqrt(1-ab) * eps_hat) / sqrt(ab)

        Computed in fp32 and clamped. With a cosine schedule sqrt(alpha_bar)
        falls to 4.9e-05 at t=999, so 1/sqrt(alpha_bar) reaches 2.0e+04 and the
        division overflows fp16 under autocast -- which produced nan losses on
        whichever batches happened to draw a large t. Clamping the predicted x0
        is the standard DDPM `clip_denoised` step; the bound is generous next to
        the class-embedding scale, so it only ever engages on the pathological
        high-noise end.
        """
        sqrt_ab = self._ab('sqrt_ab', t, s_t.ndim).float()
        sqrt_1m = self._ab('sqrt_1m_ab', t, s_t.ndim).float()
        x0 = (s_t.float() - sqrt_1m * eps_hat.float()) / sqrt_ab.clamp(min=1e-3)
        return x0.clamp(-self.s0_clamp, self.s0_clamp).to(s_t.dtype)

    # ------------------------------------------------------------------ train

    def forward(self, x, labels=None, out_hw=None):
        """
        Training forward. `labels` are required only here -- never at inference.
        Returns a dict so nothing is silently dropped.
        """
        out_hw = out_hw or x.shape[-2:]
        self._sync_diffusion_device(x.device)
        f1, cond = self.encode(x)
        coarse_logits = self.coarse_head(cond)
        h, w = cond.shape[-2:]
        B = x.shape[0]

        eps = eps_hat = None
        if self.use_diffusion:
            assert labels is not None, 'training with diffusion needs labels for S0'
            s0, _ = self.semantic_latent_from_labels(labels, (h, w))
            t = torch.randint(0, self.diffusion.T, (B,), device=x.device)
            eps = torch.randn_like(s0)
            s_t = self.diffusion.q_sample(s0, t, noise=eps)
            eps_hat = self.dit(s_t, t, cond)
            s0_hat = self.predict_x0(s_t, t, eps_hat)
        else:
            # Ablations B and C: the same bottleneck runs on the coarse
            # embedding with no noise and no diffusion loss.
            t = torch.zeros(B, dtype=torch.long, device=x.device)
            s_in = self.semantic_latent_from_coarse(coarse_logits)
            s0_hat = s_in + self.dit(s_in, t, cond)

        logits = self.decoder(s0_hat, f1, out_hw)
        return {'logits': logits, 'coarse_logits': coarse_logits,
                'eps': eps, 'eps_hat': eps_hat, 'latent_hw': (h, w),
                'timestep': t if self.use_diffusion else None}

    # -------------------------------------------------------------- inference

    @torch.no_grad()
    def infer(self, x, out_hw=None, steps=None, t_start=None, generator=None,
              seed=None):
        """Deterministic when `seed` is given (or `generator` is supplied)."""
        if seed is not None and generator is None:
            generator = torch.Generator(device=x.device).manual_seed(int(seed))
        return self._infer(x, out_hw, steps, t_start, generator)

    @torch.no_grad()
    def _infer(self, x, out_hw=None, steps=None, t_start=None, generator=None):
        """
        No ground truth is touched. The diffusion is initialised from the coarse
        prediction, optionally perturbed to `t_start`, then refined with a few
        DDIM steps conditioned on C.
        """
        out_hw = out_hw or x.shape[-2:]
        self._sync_diffusion_device(x.device)
        steps = self.inference_steps if steps is None else steps
        t_start = self.inference_t_start if t_start is None else t_start

        f1, cond = self.encode(x)
        coarse_logits = self.coarse_head(cond)
        s0_hat = self.sample_latent(cond, coarse_logits, steps, t_start, generator)
        logits = self.decoder(s0_hat, f1, out_hw)
        return {'logits': logits, 'coarse_logits': coarse_logits}

    def sample_latent(self, cond, coarse_logits, steps, t_start, generator=None):
        s = self.semantic_latent_from_coarse(coarse_logits)
        if not self.use_diffusion:
            t0 = torch.zeros(cond.shape[0], dtype=torch.long, device=cond.device)
            return s + self.dit(s, t0, cond)

        B, dev = cond.shape[0], cond.device
        t_start = int(max(1, min(t_start, self.diffusion.T - 1)))
        noise = torch.randn(s.shape, device=dev, generator=generator, dtype=s.dtype)
        t_vec = torch.full((B,), t_start, dtype=torch.long, device=dev)
        s_t = self.diffusion.q_sample(s, t_vec, noise=noise)

        # DDIM (eta = 0) over a short, evenly spaced timestep sequence.
        ts = torch.linspace(t_start, 0, steps + 1).long().tolist()
        for t_cur, t_prev in zip(ts[:-1], ts[1:]):
            tv = torch.full((B,), t_cur, dtype=torch.long, device=dev)
            eps_hat = self.dit(s_t, tv, cond)
            s0_hat = self.predict_x0(s_t, tv, eps_hat)
            if t_prev <= 0:
                return s0_hat
            pv = torch.full((B,), t_prev, dtype=torch.long, device=dev)
            s_t = (self._ab('sqrt_ab', pv, s_t.ndim) * s0_hat +
                   self._ab('sqrt_1m_ab', pv, s_t.ndim) * eps_hat)
        return s_t

    # ------------------------------------------------------------- bookkeeping

    def parameter_groups(self):
        """Per-submodule parameter counts, for reporting."""
        def n(m):
            return sum(p.numel() for p in m.parameters() if p.requires_grad)
        return {'rem': n(self.rem),
                'encoder_stage1': n(self.stage1),
                'encoder_stage2': n(self.stage2),
                'latent_proj': n(self.latent_proj),
                'coarse_head': n(self.coarse_head),
                'class_embed': n(self.class_embed),
                'dit': n(self.dit),
                'decoder': n(self.decoder),
                'total': n(self)}
