"""
Token encoder and global query policy for diffusion cross-attention architecture.

TokenEncoder: maps input x (+ modality m, optional batch b, optional celltype w)
to shared/private semantic tokens via a backbone + reparameterization heads.

GlobalQueryPolicy: produces per-cell query biases for dual-path cross-attention,
using static cell gate + timestep FiLM modulation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenEncoder(nn.Module):
    """Encode input to shared and private semantic tokens.

    forward(x, m, b=None, w=None) -> dict with:
        shared_tokens  (B, K_s, token_dim)
        private_tokens (B, K_p, token_dim)
        shared_mu, shared_logvar   (B, K_s * token_dim)
        private_mu, private_logvar (B, K_p * token_dim)
        backbone_h  (B, backbone_out_dim)
    """

    def __init__(self, input_dim, modality_num, covariate_dim=0, celltype_num=0,
                 backbone_dims=(500, 100), token_dim=32,
                 num_shared_tokens=4, num_private_tokens=2,
                 dropout_rate=0.5, encoder_covariates=False):
        super().__init__()
        self.token_dim = token_dim
        self.num_shared_tokens = num_shared_tokens
        self.num_private_tokens = num_private_tokens
        self.encoder_covariates = encoder_covariates
        self.covariate_dim = covariate_dim
        self.celltype_num = celltype_num
        self.modality_num = modality_num

        # Backbone input dim
        bb_in = input_dim
        if encoder_covariates and covariate_dim > 0:
            bb_in += covariate_dim
        if celltype_num > 0:
            bb_in += celltype_num

        layers = []
        current = bb_in
        for dim in backbone_dims:
            layers.append(nn.Linear(current, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(dropout_rate))
            current = dim
        self.backbone = nn.Sequential(*layers)
        self.backbone_out_dim = backbone_dims[-1]

        # Shared head: backbone_h -> mu, logvar for K_s tokens
        shared_flat = num_shared_tokens * token_dim
        self.shared_mu = nn.Linear(self.backbone_out_dim, shared_flat)
        self.shared_logvar = nn.Linear(self.backbone_out_dim, shared_flat)

        # Private head: cat(backbone_h, m) -> mu, logvar for K_p tokens
        private_in = self.backbone_out_dim + modality_num
        private_flat = num_private_tokens * token_dim
        self.private_mu = nn.Linear(private_in, private_flat)
        self.private_logvar = nn.Linear(private_in, private_flat)

    def _build_backbone_input(self, x, b, w):
        parts = [x]
        if self.encoder_covariates and self.covariate_dim > 0 and b is not None:
            parts.append(b)
        if self.celltype_num > 0 and w is not None:
            parts.append(w)
        return torch.cat(parts, dim=-1) if len(parts) > 1 else x

    @staticmethod
    def _reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, m, b=None, w=None, sample=True):
        B = x.shape[0]
        bb_input = self._build_backbone_input(x, b, w)
        backbone_h = self.backbone(bb_input)                       # (B, backbone_out)

        # Shared tokens
        s_mu = self.shared_mu(backbone_h)                          # (B, K_s * d)
        s_logvar = self.shared_logvar(backbone_h)                  # (B, K_s * d)
        s_tokens = self._reparameterize(s_mu, s_logvar) if sample else s_mu
        shared_tokens = s_tokens.view(B, self.num_shared_tokens, self.token_dim)
        shared_mu_tokens = s_mu.view(B, self.num_shared_tokens, self.token_dim)

        # Private tokens
        p_input = torch.cat([backbone_h, m], dim=-1)
        p_mu = self.private_mu(p_input)                            # (B, K_p * d)
        p_logvar = self.private_logvar(p_input)                    # (B, K_p * d)
        p_tokens = self._reparameterize(p_mu, p_logvar) if sample else p_mu
        private_tokens = p_tokens.view(B, self.num_private_tokens, self.token_dim)
        private_mu_tokens = p_mu.view(B, self.num_private_tokens, self.token_dim)

        return {
            "shared_tokens": shared_tokens,
            "private_tokens": private_tokens,
            "shared_mu_tokens": shared_mu_tokens,
            "private_mu_tokens": private_mu_tokens,
            "shared_mu": s_mu,
            "shared_logvar": s_logvar,
            "private_mu": p_mu,
            "private_logvar": p_logvar,
            "backbone_h": backbone_h,
        }


class GlobalQueryPolicy(nn.Module):
    """Produce per-cell query biases for dual-path cross-attention.

    Uses static cell gate on backbone_h + timestep FiLM modulation.
    Prototype dimension = inner_dim (heads * dim_head), matching CrossAttention.to_q output.
    """

    def __init__(self, num_shared_protos=2, num_private_protos=2,
                 inner_dim=128, backbone_dim=100, time_embed_dim=64,
                 gate_hidden=64):
        super().__init__()
        total_protos = num_shared_protos + num_private_protos
        self.num_shared_protos = num_shared_protos
        self.num_private_protos = num_private_protos

        # Learnable query-space prototypes
        self.shared_protos = nn.Parameter(torch.randn(num_shared_protos, inner_dim))
        self.private_protos = nn.Parameter(torch.randn(num_private_protos, inner_dim))

        # Cell gate: backbone_h -> logits for all prototypes
        self.cell_gate = nn.Sequential(
            nn.Linear(backbone_dim, gate_hidden),
            nn.LeakyReLU(),
            nn.Linear(gate_hidden, total_protos),
        )

        # Timestep FiLM: t_emb -> scale + shift for all prototype logits
        self.time_mod = nn.Sequential(
            nn.Linear(time_embed_dim, gate_hidden),
            nn.SiLU(),
            nn.Linear(gate_hidden, total_protos * 2),
        )

    def forward(self, backbone_h, t_emb):
        """
        Args:
            backbone_h: (B, backbone_dim)
            t_emb:      (B, time_embed_dim) -- raw sinusoidal time embedding
        Returns:
            q_bias_shared:  (B, inner_dim)
            q_bias_private: (B, inner_dim)
            alpha_shared:   (B, num_shared_protos)
            alpha_private:  (B, num_private_protos)
        """
        n_s = self.num_shared_protos
        base_logits = self.cell_gate(backbone_h)              # (B, total_protos)

        film_params = self.time_mod(t_emb)                    # (B, total_protos * 2)
        scale, shift = film_params.chunk(2, dim=-1)           # each (B, total_protos)
        modulated = base_logits * (1.0 + scale) + shift       # (B, total_protos)

        alpha_shared = F.softmax(modulated[:, :n_s], dim=-1)  # (B, n_s)
        alpha_private = F.softmax(modulated[:, n_s:], dim=-1) # (B, n_p)

        # Query biases in attention inner_dim space
        q_bias_shared = alpha_shared @ self.shared_protos     # (B, inner_dim)
        q_bias_private = alpha_private @ self.private_protos  # (B, inner_dim)

        return q_bias_shared, q_bias_private, alpha_shared, alpha_private
