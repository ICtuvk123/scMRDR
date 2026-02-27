import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        device = t.device
        scale = math.log(10000.0) / max(half - 1, 1)
        freqs = torch.exp(torch.arange(half, device=device) * -scale)
        angles = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class LatentDenoiserMLP(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        cond_dim: int,
        hidden_dim: int = 512,
        time_embed_dim: int = 64,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cond_proj = nn.Linear(cond_dim, hidden_dim) if cond_dim > 0 else None
        in_dim = latent_dim + hidden_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        t_hidden = self.time_mlp(self.time_embed(t))
        if self.cond_proj is not None and cond is not None:
            t_hidden = t_hidden + self.cond_proj(cond)
        h = torch.cat([z_t, t_hidden], dim=1)
        return self.net(h)


class GaussianDiffusion1D(nn.Module):
    def __init__(self, num_steps: int = 200, beta_schedule: str = "linear"):
        super().__init__()
        if num_steps < 2:
            raise ValueError("num_steps must be >= 2")
        if beta_schedule not in {"linear", "cosine"}:
            raise ValueError("beta_schedule must be 'linear' or 'cosine'")

        self.num_steps = int(num_steps)
        betas = self._build_betas(self.num_steps, beta_schedule)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - alpha_bars))

    @staticmethod
    def _build_betas(num_steps: int, beta_schedule: str) -> torch.Tensor:
        if beta_schedule == "linear":
            return torch.linspace(1e-4, 2e-2, num_steps)

        # Cosine schedule from improved DDPM.
        s = 0.008
        x = torch.linspace(0, num_steps, num_steps + 1, dtype=torch.float32)
        alphas_bar = torch.cos(((x / num_steps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_bar = alphas_bar / alphas_bar[0]
        betas = 1.0 - (alphas_bar[1:] / alphas_bar[:-1])
        return betas.clamp(1e-5, 0.999)

    @staticmethod
    def _extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        out = a.gather(0, t).float()
        while len(out.shape) < len(x_shape):
            out = out.unsqueeze(-1)
        return out

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randint(0, self.num_steps, (batch_size,), device=device, dtype=torch.long)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = self._extract(self.sqrt_alpha_bars, t, x0.shape)
        sqrt_1mab = self._extract(self.sqrt_one_minus_alpha_bars, t, x0.shape)
        return sqrt_ab * x0 + sqrt_1mab * noise

    def predict_x0_from_eps(self, z_t: torch.Tensor, t: torch.Tensor, eps_pred: torch.Tensor) -> torch.Tensor:
        sqrt_ab = self._extract(self.sqrt_alpha_bars, t, z_t.shape)
        sqrt_1mab = self._extract(self.sqrt_one_minus_alpha_bars, t, z_t.shape)
        return (z_t - sqrt_1mab * eps_pred) / (sqrt_ab + 1e-8)

