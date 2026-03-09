"""
Dual-path cross-attention for rare-cell-aware integration.

Adapted from CausCell/causcell/Modules.py CrossAttention, without einops.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    """Multi-head cross-attention with optional query bias injection."""

    def __init__(self, query_dim, context_dim, heads=4, dim_head=32, dropout=0.0):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = heads * dim_head
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(query_dim, inner_dim)
        self.to_k = nn.Linear(context_dim, inner_dim)
        self.to_v = nn.Linear(context_dim, inner_dim)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, context, query_bias=None):
        """
        Args:
            x:          (B, N_q, query_dim)   -- hidden state
            context:    (B, N_kv, context_dim) -- token group (shared OR private)
            query_bias: (B, inner_dim) or None -- from global query policy prototype
        Returns:
            (B, N_q, query_dim)
        """
        B, N_q, _ = x.shape
        N_kv = context.shape[1]
        h = self.heads
        d = self.dim_head

        q = self.to_q(x)                                    # (B, N_q, inner_dim)
        if query_bias is not None:
            q = q + query_bias.unsqueeze(1)                  # additive bias from prototype
        k = self.to_k(context)                               # (B, N_kv, inner_dim)
        v = self.to_v(context)                               # (B, N_kv, inner_dim)

        # Reshape for multi-head: (B, N, h*d) -> (B*h, N, d)
        q = q.view(B, N_q, h, d).permute(0, 2, 1, 3).reshape(B * h, N_q, d)
        k = k.view(B, N_kv, h, d).permute(0, 2, 1, 3).reshape(B * h, N_kv, d)
        v = v.view(B, N_kv, h, d).permute(0, 2, 1, 3).reshape(B * h, N_kv, d)

        # Scaled dot-product attention
        sim = torch.bmm(q, k.transpose(1, 2)) * self.scale  # (B*h, N_q, N_kv)
        attn = F.softmax(sim, dim=-1)                        # (B*h, N_q, N_kv)
        out = torch.bmm(attn, v)                             # (B*h, N_q, d)

        # Merge heads: (B*h, N_q, d) -> (B, N_q, h*d)
        out = out.view(B, h, N_q, d).permute(0, 2, 1, 3).reshape(B, N_q, h * d)
        return self.to_out(out)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class DualPathTransformerBlock(nn.Module):
    """One block of dual-path cross-attention.

    NOT a single attention over concatenated tokens.
    Structurally separates shared and private routing.
    """

    def __init__(self, dim, context_dim, n_heads=4, dim_head=32,
                 dropout=0.0, ff_mult=4):
        super().__init__()
        self.norm_s = nn.LayerNorm(dim)
        self.attn_shared = CrossAttention(dim, context_dim, n_heads, dim_head, dropout)
        self.norm_p = nn.LayerNorm(dim)
        self.attn_private = CrossAttention(dim, context_dim, n_heads, dim_head, dropout)
        self.norm_ff = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, mult=ff_mult, dropout=dropout)

    def forward(self, h, shared_tokens, private_tokens,
                q_bias_shared=None, q_bias_private=None):
        """
        Args:
            h:              (B, 1, dim)         -- current hidden
            shared_tokens:  (B, K_s, ctx_dim)   -- shared K/V
            private_tokens: (B, K_p, ctx_dim)   -- private K/V
            q_bias_shared:  (B, inner_dim)      -- prototype query bias for shared path
            q_bias_private: (B, inner_dim)      -- prototype query bias for private path
        Returns:
            (B, 1, dim)
        """
        # Shared path: Q modulated by Q_shared prototype
        A_shared = self.attn_shared(self.norm_s(h),
                                    context=shared_tokens,
                                    query_bias=q_bias_shared)

        # Private path: Q modulated by Q_private prototype
        A_private = self.attn_private(self.norm_p(h),
                                      context=private_tokens,
                                      query_bias=q_bias_private)

        # Merge: h' = h + A_shared + A_private
        h = h + A_shared + A_private

        # FFN
        h = h + self.ff(self.norm_ff(h))
        return h
