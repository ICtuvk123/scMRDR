import torch


def dropout_augment(x, drop_rate=0.1):
    """Input-level gene dropout for count data augmentation.

    Randomly zeros out genes per sample (mimics technical dropout).
    No rescaling — the encoder sees a naturally sparser version.
    """
    mask = torch.bernoulli(torch.full_like(x, 1.0 - drop_rate))
    return x * mask


class ConsistencyGating:
    """Gate preserve_loss by z_shared cross-view consistency.

    For each sample, two augmented views are passed through encoder_shared.
    c_i = ||mu_s(aug1) - mu_s(aug2)|| measures how stable z_shared is.

    Stable z_shared (low c_i)   -> high weight -> trust geometric signal
    Unstable z_shared (high c_i) -> low weight  -> filter noisy geometry
    """

    def __init__(self, w_floor=0.3, w_cap=1.0, tau=1.0,
                 ema_decay=0.99, warmup_steps=100, drop_rate=0.1):
        self.w_floor = w_floor
        self.w_cap = w_cap
        self.tau = tau
        self.ema_decay = ema_decay
        self.warmup_steps = warmup_steps
        self.drop_rate = drop_rate
        self.step_count = 0
        self.running_median = None
        self.running_mad = None

    def _update_stats(self, c):
        """EMA update of median and MAD (robust statistics)."""
        batch_median = c.median()
        batch_mad = (c - batch_median).abs().median()
        if self.running_median is None:
            self.running_median = batch_median.item()
            self.running_mad = batch_mad.item()
        else:
            self.running_median = (self.ema_decay * self.running_median
                                   + (1 - self.ema_decay) * batch_median.item())
            self.running_mad = (self.ema_decay * self.running_mad
                                + (1 - self.ema_decay) * batch_mad.item())

    def compute_consistency(self, x_raw, model, b=None, w=None):
        """Compute per-sample z_shared consistency distance.

        Args:
            x_raw: (B, D) raw counts (before log1p)
            model: EmbeddingNet (uses encode_shared_mu)
            b: (B, cov_dim) batch covariates, or None
            w: (B, ct_dim) celltype one-hot, or None
        Returns:
            c_i: (B,) L2 distance between two augmented z_shared
        """
        with torch.no_grad():
            x_aug1 = dropout_augment(x_raw, self.drop_rate)
            x_aug2 = dropout_augment(x_raw, self.drop_rate)
            mu_s1 = model.encode_shared_mu(x_aug1, b, w)
            mu_s2 = model.encode_shared_mu(x_aug2, b, w)
            c_i = torch.norm(mu_s1 - mu_s2, dim=1)
        return c_i

    def compute_weights(self, c_i):
        """Convert consistency distances to per-sample weights.

        Args:
            c_i: (B,) consistency distances (lower = more stable)
        Returns:
            w: (B,) weights in [w_floor, w_cap]
        """
        c = c_i.detach()
        self._update_stats(c)
        self.step_count += 1

        if self.step_count <= self.warmup_steps:
            return torch.ones_like(c)

        # Robust standardization (MAD -> pseudo-sigma)
        z = (c - self.running_median) / (self.running_mad * 1.4826 + 1e-8)
        z = z.clamp(-3.0, 3.0)

        # High c -> high z -> low weight
        w_raw = torch.sigmoid(-z / self.tau)

        # Map to [w_floor, w_cap]
        w = self.w_floor + (self.w_cap - self.w_floor) * w_raw
        return w.detach()
