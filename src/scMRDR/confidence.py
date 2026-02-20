import torch


class ReconGating:
    """基于重建损失的 per-sample 对抗权重门控。

    高 recon → 噪声样本 → 降低对抗权重
    低 recon → 干净样本 → 保持对抗权重
    所有样本至少保留 w_floor 的对抗压力
    """

    def __init__(self, w_floor=0.3, w_cap=1.0, tau=1.0,
                 ema_decay=0.99, stats_warmup_steps=50):
        self.w_floor = w_floor
        self.w_cap = w_cap
        self.tau = tau
        self.ema_decay = ema_decay
        self.stats_warmup_steps = stats_warmup_steps
        self.step_count = 0
        self.running_median = None
        self.running_mad = None

    def _update_stats(self, r):
        """用 EMA 更新 median 和 MAD（稳健统计量）"""
        batch_median = r.median()
        batch_mad = (r - batch_median).abs().median()
        if self.running_median is None:
            self.running_median = batch_median.item()
            self.running_mad = batch_mad.item()
        else:
            self.running_median = self.ema_decay * self.running_median + (1 - self.ema_decay) * batch_median.item()
            self.running_mad = self.ema_decay * self.running_mad + (1 - self.ema_decay) * batch_mad.item()

    def compute_weights(self, per_sample_recon):
        """
        per_sample_recon: (B,) 每样本重建损失
        Returns: (B,) 权重 ∈ [w_floor, w_cap]
        """
        r = per_sample_recon.detach()           # 护栏 2: detach
        self._update_stats(r)
        self.step_count += 1

        if self.step_count <= self.stats_warmup_steps:  # 护栏 3: stats warmup
            return torch.ones_like(r)

        # 护栏 1: 稳健标准化 + winsorize
        z = (r - self.running_median) / (self.running_mad * 1.4826 + 1e-8)
        z = z.clamp(-3.0, 3.0)  # winsorize

        # sigmoid: recon 高 → z 大 → 权重低
        w_raw = torch.sigmoid(-z / self.tau)

        # 映射到 [w_floor, w_cap]
        w = self.w_floor + (self.w_cap - self.w_floor) * w_raw
        return w.detach()
