"""
Diffusion decoder with dual-path cross-attention for expression reconstruction.

DenoiseDecoder: takes noisy x_t, timestep t, shared/private tokens, and query biases;
    produces direct x_0 prediction via dual-path transformer blocks.

ExpressionDiffusion: wraps GaussianDiffusion1D schedule with DenoiseDecoder,
    providing training_loss (L1 on x_0) and sampling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .diffusion import SinusoidalTimeEmbedding, GaussianDiffusion1D
from .cross_attention import DualPathTransformerBlock


class DenoiseDecoder(nn.Module):
    """Denoiser that predicts x_0 directly from x_t using dual-path cross-attention."""

    def __init__(self, input_dim, token_dim=32, hidden_dim=256,
                 time_embed_dim=64, xattn_depth=2, xattn_heads=4,
                 xattn_dim_head=32, dropout=0.1):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            DualPathTransformerBlock(
                dim=hidden_dim,
                context_dim=token_dim,
                n_heads=xattn_heads,
                dim_head=xattn_dim_head,
                dropout=dropout,
            )
            for _ in range(xattn_depth)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, x_t, t, shared_tokens, private_tokens,
                q_bias_shared, q_bias_private):
        """
        Args:
            x_t:            (B, input_dim) -- noisy input
            t:              (B,) -- integer timesteps
            shared_tokens:  (B, K_s, token_dim)
            private_tokens: (B, K_p, token_dim)
            q_bias_shared:  (B, inner_dim)
            q_bias_private: (B, inner_dim)
        Returns:
            x_0_pred: (B, input_dim) -- direct x_0 prediction
        """
        t_emb = self.time_mlp(self.time_embed(t))           # (B, hidden_dim)
        h = self.input_proj(x_t) + t_emb                    # (B, hidden_dim)
        h = h.unsqueeze(1)                                   # (B, 1, hidden_dim)

        for block in self.blocks:
            h = block(h, shared_tokens, private_tokens,
                      q_bias_shared, q_bias_private)

        h = h.squeeze(1)                                     # (B, hidden_dim)
        return self.output_proj(self.norm(h))                # (B, input_dim)


class ExpressionDiffusion(nn.Module):
    """Wraps diffusion schedule + DenoiseDecoder for expression reconstruction.

    Uses direct x_0 prediction with L1 loss in log1p space.
    """

    def __init__(self, input_dim, token_dim=32, hidden_dim=256,
                 time_embed_dim=64, xattn_depth=2, xattn_heads=4,
                 xattn_dim_head=32, dropout=0.1,
                 diff_steps=100, beta_schedule="linear"):
        super().__init__()
        self.diffusion = GaussianDiffusion1D(
            num_steps=diff_steps,
            beta_schedule=beta_schedule,
        )
        self.denoiser = DenoiseDecoder(
            input_dim=input_dim,
            token_dim=token_dim,
            hidden_dim=hidden_dim,
            time_embed_dim=time_embed_dim,
            xattn_depth=xattn_depth,
            xattn_heads=xattn_heads,
            xattn_dim_head=xattn_dim_head,
            dropout=dropout,
        )
        self.time_embed_dim = time_embed_dim

    def training_loss(self, x_0, shared_tokens, private_tokens,
                      q_bias_shared, q_bias_private, t=None,
                      mask=None, eps=1e-8):
        """Compute L1 reconstruction loss for diffusion training.

        Args:
            x_0:            (B, input_dim) -- clean log1p expression
            shared_tokens:  (B, K_s, token_dim)
            private_tokens: (B, K_p, token_dim)
            q_bias_shared:  (B, inner_dim)
            q_bias_private: (B, inner_dim)
        Returns:
            loss: scalar L1 loss between x_0 and x_0_pred
        """
        B = x_0.shape[0]
        if t is None:
            t = self.diffusion.sample_timesteps(B, x_0.device)
        noise = torch.randn_like(x_0)
        x_t = self.diffusion.q_sample(x_0, t, noise)

        x_0_pred = self.denoiser(
            x_t, t, shared_tokens, private_tokens,
            q_bias_shared, q_bias_private,
        )
        abs_err = torch.abs(x_0_pred - x_0)
        if mask is not None:
            weighted_err = (abs_err * mask).sum(dim=1)
            denom = mask.sum(dim=1) + eps
            return (weighted_err / denom).mean()
        return abs_err.mean()

    @torch.no_grad()
    def sample(self, shared_tokens, private_tokens,
               q_bias_shared, q_bias_private, input_dim):
        """Generate samples via iterative denoising (DDPM-style).

        Simple single-step x_0 prediction at a random high-noise timestep.
        Full iterative sampling reserved for V2.
        """
        B = shared_tokens.shape[0]
        device = shared_tokens.device
        x_t = torch.randn(B, input_dim, device=device)
        # Use highest noise level for single-step prediction
        t = torch.full((B,), self.diffusion.num_steps - 1, device=device, dtype=torch.long)
        x_0_pred = self.denoiser(
            x_t, t, shared_tokens, private_tokens,
            q_bias_shared, q_bias_private,
        )
        return x_0_pred
