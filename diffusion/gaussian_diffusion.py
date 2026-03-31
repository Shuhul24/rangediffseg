"""
Gaussian Diffusion (DDPM) for range-view segmentation.
Adapted from SegDiff (github.com/tomeramit/SegDiff).

The model predicts x_0 (clean segmentation) directly from (x_t, t, cond).
Loss = MSE(pred_x0, x_0), optionally masked to valid LiDAR pixels.
"""

import numpy as np
import torch
import torch.nn.functional as F


def _cosine_betas(T: int, s: float = 0.008) -> np.ndarray:
    """Cosine noise schedule (Nichol & Dhariwal, 2021)."""
    t     = np.linspace(0, T, T + 1)
    ab    = np.cos((t / T + s) / (1 + s) * np.pi / 2) ** 2
    ab   /= ab[0]
    betas = 1 - ab[1:] / ab[:-1]
    return np.clip(betas, 0, 0.999)


def _linear_betas(T: int, b0: float = 1e-4, bT: float = 0.02) -> np.ndarray:
    return np.linspace(b0, bT, T)


class GaussianDiffusion:
    """
    Encapsulates the forward (noising) and reverse (denoising) processes.
    No trainable parameters — acts as a training / sampling coordinator.
    """

    def __init__(self, num_timesteps: int = 100, schedule: str = 'cosine'):
        betas  = (_cosine_betas if schedule == 'cosine' else _linear_betas)(num_timesteps)
        betas  = betas.astype(np.float64)
        self.T = num_timesteps

        alphas      = 1.0 - betas
        ab          = np.cumprod(alphas)           # ᾱ_t
        ab_prev     = np.append(1.0, ab[:-1])      # ᾱ_{t-1}

        def f(x): return torch.from_numpy(x.astype(np.float32))

        self.sqrt_ab      = f(np.sqrt(ab))
        self.sqrt_1m_ab   = f(np.sqrt(1.0 - ab))

        # Posterior q(x_{t-1} | x_t, x_0)
        post_var          = betas * (1.0 - ab_prev) / (1.0 - ab)
        self.post_var     = f(post_var)
        self.post_log_var = f(np.log(np.append(post_var[1], post_var[1:])))  # clamp t=0
        self.post_c0      = f(np.sqrt(ab_prev) * betas / (1.0 - ab))        # coeff for x_0
        self.post_ct      = f(np.sqrt(alphas) * (1.0 - ab_prev) / (1.0 - ab))  # coeff for x_t

    # ------------------------------------------------------------------ helpers

    def _to(self, device):
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                setattr(self, k, v.to(device))

    @staticmethod
    def _expand(arr, t, ndim):
        """Gather arr[t] and broadcast to shape (B, 1, 1, ...) with ndim dims."""
        x = arr[t]
        while x.ndim < ndim:
            x = x.unsqueeze(-1)
        return x

    # ------------------------------------------------------------------ forward

    def q_sample(self, x0, t, noise=None):
        """Add noise to x0 at timestep t:  x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε."""
        if noise is None:
            noise = torch.randn_like(x0)
        return (self._expand(self.sqrt_ab, t, x0.ndim) * x0 +
                self._expand(self.sqrt_1m_ab, t, x0.ndim) * noise)

    # ------------------------------------------------------------------ reverse

    def _posterior(self, pred_x0, x_t, t):
        """Compute posterior mean and log-variance given predicted x_0."""
        pred_x0  = pred_x0.clamp(-1.0, 1.0)
        mean     = (self._expand(self.post_c0, t, x_t.ndim) * pred_x0 +
                    self._expand(self.post_ct, t, x_t.ndim) * x_t)
        log_var  = self._expand(self.post_log_var, t, x_t.ndim)
        return mean, log_var, pred_x0

    @torch.no_grad()
    def p_sample(self, model, x_t, t, cond):
        """One denoising step: x_{t-1} ~ p(x_{t-1} | x_t)."""
        pred_x0         = model(x_t, t, cond)
        mean, log_var, pred_x0 = self._posterior(pred_x0, x_t, t)
        noise           = torch.randn_like(x_t)
        nonzero         = (t != 0).float().view(-1, *([1] * (x_t.ndim - 1)))
        return mean + nonzero * (0.5 * log_var).exp() * noise, pred_x0

    @torch.no_grad()
    def p_sample_loop(self, model, shape, cond, device):
        """Full reverse diffusion: generates segmentation from pure noise."""
        self._to(device)
        x = torch.randn(*shape, device=device)
        for i in reversed(range(self.T)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            x, _ = self.p_sample(model, x, t, cond)
        return x  # (B, num_classes, H, W) in [-1, 1]

    # ------------------------------------------------------------------ training

    def training_loss(self, model, x0, t, cond, mask=None):
        """
        MSE loss in x_0-prediction mode.

        Args:
            model : U-Net, takes (x_t, t, cond) → pred_x0
            x0    : (B, C, H, W) clean one-hot segmentation in [-1, 1]
            t     : (B,)  sampled diffusion timesteps
            cond  : (B, 5, H, W) range-view image features
            mask  : (B, H, W) float32, 1=valid LiDAR pixel (optional)
        Returns:
            scalar loss
        """
        self._to(x0.device)
        noise  = torch.randn_like(x0)
        x_t    = self.q_sample(x0, t, noise)
        pred   = model(x_t, t, cond)

        loss   = F.mse_loss(pred, x0, reduction='none')  # (B, C, H, W)
        if mask is not None:
            # Average only over valid pixels
            m    = mask.unsqueeze(1)                  # (B, 1, H, W)
            loss = (loss * m).sum() / (m.sum() * loss.shape[1] + 1e-8)
        else:
            loss = loss.mean()
        return loss
