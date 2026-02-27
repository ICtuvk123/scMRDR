import torch
import torch.nn.functional as F
import math


class ConfidenceWeighter:
    """
    Confidence-weighted adversarial training module.

    Computes per-sample weights for the adversarial loss based on:
    (a) discriminator entropy signal
    (b) cross-modal nearest-neighbor similarity
    Fused via linear combination and gated by per-modality soft budget.
    """

    def __init__(self, latent_dim, num_modalities, device,
                 queue_size=4096, alpha=0.5,
                 c_tau=1.0, tau_min=0.01, tau_max=2.0, tau_fallback=0.5,
                 eta=0.9, rho=0.5, tau_w=0.1, w_min=0.1, min_count=8):
        self.latent_dim = latent_dim
        self.num_modalities = num_modalities
        self.device = device
        self.queue_size = queue_size
        self.alpha = alpha
        self.c_tau = c_tau
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.tau_fallback = tau_fallback
        self.eta = eta
        self.rho = rho
        self.tau_w = tau_w
        self.w_min = w_min
        self.min_count = min_count

        # Circular buffer per modality
        self.queues = [torch.zeros(queue_size, latent_dim, device=device) for _ in range(num_modalities)]
        self.buffer_ptr = [0] * num_modalities
        self.buffer_count = [0] * num_modalities

        # Adaptive tau_nn via EMA
        self.tau_nn_ema = tau_fallback

    def update_queues(self, z_shared, modality_labels):
        """
        Write L2-normalized z_shared into per-modality circular buffers.

        Args:
            z_shared: (B, D) detached tensor
            modality_labels: (B,) integer tensor
        """
        z_norm = F.normalize(z_shared.detach(), dim=1)
        for mod_idx in range(self.num_modalities):
            mask = modality_labels == mod_idx
            if mask.sum() == 0:
                continue
            z_mod = z_norm[mask]
            n = z_mod.shape[0]
            ptr = self.buffer_ptr[mod_idx]
            q = self.queues[mod_idx]
            if n >= self.queue_size:
                # Overwrite the entire queue with the last queue_size samples
                q[:] = z_mod[-self.queue_size:]
                self.buffer_ptr[mod_idx] = 0
                self.buffer_count[mod_idx] = self.queue_size
            elif ptr + n <= self.queue_size:
                q[ptr:ptr + n] = z_mod
                self.buffer_ptr[mod_idx] = ptr + n
                self.buffer_count[mod_idx] = min(self.buffer_count[mod_idx] + n, self.queue_size)
            else:
                # Wrap around
                first = self.queue_size - ptr
                q[ptr:] = z_mod[:first]
                remainder = n - first
                q[:remainder] = z_mod[first:]
                self.buffer_ptr[mod_idx] = remainder
                self.buffer_count[mod_idx] = self.queue_size

    @torch.no_grad()
    def compute_weights(self, z_shared, modality_labels, discriminator_logits):
        """
        Compute per-sample confidence weights (all detached).

        Args:
            z_shared: (B, D) tensor (will be detached internally)
            modality_labels: (B,) integer tensor
            discriminator_logits: (B, M) tensor (will be detached internally)

        Returns:
            weights: (B,) detached tensor in [w_min, 1]
        """
        B = z_shared.shape[0]
        M = self.num_modalities
        logits = discriminator_logits.detach()
        z = z_shared.detach()

        # (a) Entropy signal
        p = F.softmax(logits, dim=1)
        log_p = torch.log(p + 1e-8)
        H = -(p * log_p).sum(dim=1) / math.log(M)  # normalized to [0, 1]
        s_H = H

        # (b) Cross-modal NN signal
        z_norm = F.normalize(z, dim=1)
        s_nn = torch.ones(B, device=self.device)
        valid_distances = []

        for mod_idx in range(M):
            mask = modality_labels == mod_idx
            if mask.sum() == 0:
                continue

            # Collect queue entries from all OTHER modalities
            other_parts = []
            for other_idx in range(M):
                if other_idx == mod_idx:
                    continue
                cnt = self.buffer_count[other_idx]
                if cnt > 0:
                    other_parts.append(self.queues[other_idx][:cnt])

            if len(other_parts) == 0:
                # No cross-modal data available; fallback s_nn = 1
                continue

            other_z = torch.cat(other_parts, dim=0)  # (K, D)
            z_mod = z_norm[mask]  # (N_m, D)

            # Cosine similarity (other_z already L2-normalized from update_queues)
            sim = z_mod @ other_z.t()  # (N_m, K)
            max_sim, _ = sim.max(dim=1)  # (N_m,)
            d_i = 1.0 - max_sim  # distance
            valid_distances.append(d_i)

            # Will apply exp(-d/tau) after tau is determined
            # Store indices for later
            s_nn[mask] = d_i  # temporarily store distances

        # (c) Adaptive tau_nn
        if len(valid_distances) > 0:
            all_d = torch.cat(valid_distances)
            min_valid = max(32, int(0.2 * B))
            if all_d.numel() >= min_valid:
                tau_raw = self.c_tau * torch.median(all_d).item()
                tau_nn = max(self.tau_min, min(self.tau_max, tau_raw))
                self.tau_nn_ema = self.eta * self.tau_nn_ema + (1 - self.eta) * tau_nn
            else:
                tau_nn = self.tau_nn_ema
        else:
            tau_nn = self.tau_nn_ema

        self.current_tau_nn = tau_nn

        # Now convert stored distances to s_nn values
        for mod_idx in range(M):
            mask = modality_labels == mod_idx
            if mask.sum() == 0:
                continue
            # Check if we had valid cross-modal data
            other_cnt = sum(self.buffer_count[j] for j in range(M) if j != mod_idx)
            if other_cnt == 0:
                s_nn[mask] = 1.0
            else:
                s_nn[mask] = torch.exp(-s_nn[mask] / tau_nn)

        # (d) Fusion
        s = self.alpha * s_H + (1.0 - self.alpha) * s_nn

        # (e) Per-modality soft budget gating
        w = torch.ones(B, device=self.device)

        # Compute global threshold as fallback
        global_threshold = torch.quantile(s, 1.0 - self.rho)

        for mod_idx in range(M):
            mask = modality_labels == mod_idx
            n_mod = mask.sum().item()
            if n_mod == 0:
                continue

            s_mod = s[mask]
            if n_mod < self.min_count:
                t_m = global_threshold
            else:
                t_m = torch.quantile(s_mod, 1.0 - self.rho)

            g = torch.sigmoid((s_mod - t_m) / self.tau_w)
            w[mask] = torch.clamp(self.w_min + (1.0 - self.w_min) * g, self.w_min, 1.0)

        return w.detach()


class RobustAdvGate:
    """
    Robust gating module for adversarial weighting.

    Design goals:
    1) Delay reliance on unstable early features (handled in training scheduler)
    2) Preserve global mixing pressure (used with base + gated adversarial branches)
    3) Protect rare/orphan samples from collapsing to near-zero weights
    """

    def __init__(
        self,
        latent_dim,
        num_modalities,
        device,
        queue_size=4096,
        alpha=0.6,
        c_tau=1.0,
        tau_min=0.01,
        tau_max=2.0,
        tau_fallback=0.5,
        eta=0.9,
        tau_w=0.1,
        min_count=8,
        rho_target=0.65,
        w_floor=0.15,
        w_orphan_min=0.45,
        rarity_boost=0.10,
        orphan_sim_threshold=0.15,
        orphan_margin_threshold=0.02,
    ):
        self.latent_dim = latent_dim
        self.num_modalities = num_modalities
        self.device = device

        self.queue_size = queue_size
        self.alpha = alpha
        self.c_tau = c_tau
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.tau_fallback = tau_fallback
        self.eta = eta
        self.tau_w = tau_w
        self.min_count = min_count

        self.rho_target = rho_target
        self.w_floor = w_floor
        self.w_orphan_min = w_orphan_min
        self.rarity_boost = rarity_boost
        self.orphan_sim_threshold = orphan_sim_threshold
        self.orphan_margin_threshold = orphan_margin_threshold

        # Circular buffer per modality
        self.queues = [torch.zeros(queue_size, latent_dim, device=device) for _ in range(num_modalities)]
        self.buffer_ptr = [0] * num_modalities
        self.buffer_count = [0] * num_modalities

        # EMA states
        self.tau_nn_ema = tau_fallback
        self.tau_h_ema = 0.5
        self.tau_mod_ema = [0.5 for _ in range(num_modalities)]

        # Diagnostics
        self.current_tau_nn = tau_fallback
        self.current_tau_h = 0.5
        self.current_modality_thresholds = {m: 0.5 for m in range(num_modalities)}
        self.current_mean_weight_by_modality = {m: 1.0 for m in range(num_modalities)}
        self.current_orphan_ratio_by_modality = {m: 0.0 for m in range(num_modalities)}

    def update_queues(self, z_shared, modality_labels):
        z_norm = F.normalize(z_shared.detach(), dim=1)
        for mod_idx in range(self.num_modalities):
            mask = modality_labels == mod_idx
            if mask.sum() == 0:
                continue
            z_mod = z_norm[mask]
            n = z_mod.shape[0]
            ptr = self.buffer_ptr[mod_idx]
            q = self.queues[mod_idx]
            if n >= self.queue_size:
                q[:] = z_mod[-self.queue_size:]
                self.buffer_ptr[mod_idx] = 0
                self.buffer_count[mod_idx] = self.queue_size
            elif ptr + n <= self.queue_size:
                q[ptr:ptr + n] = z_mod
                self.buffer_ptr[mod_idx] = ptr + n
                self.buffer_count[mod_idx] = min(self.buffer_count[mod_idx] + n, self.queue_size)
            else:
                first = self.queue_size - ptr
                q[ptr:] = z_mod[:first]
                remainder = n - first
                q[:remainder] = z_mod[first:]
                self.buffer_ptr[mod_idx] = remainder
                self.buffer_count[mod_idx] = self.queue_size

    @torch.no_grad()
    def compute_weights(self, z_shared, modality_labels, discriminator_logits):
        bsz = z_shared.shape[0]
        m_num = self.num_modalities
        logits = discriminator_logits.detach()
        z = z_shared.detach()

        # Reliability signal from discriminator confidence: (1 - entropy)
        p = F.softmax(logits, dim=1)
        if m_num > 1:
            entropy = -(p * torch.log(p + 1e-8)).sum(dim=1) / math.log(m_num)
        else:
            entropy = torch.zeros(bsz, device=self.device)
        reliability = 1.0 - entropy

        rho_global = min(0.95, max(0.05, self.rho_target))
        tau_h_batch = torch.quantile(reliability, 1.0 - rho_global).item()
        self.tau_h_ema = self.eta * self.tau_h_ema + (1.0 - self.eta) * tau_h_batch
        tau_h = self.tau_h_ema
        self.current_tau_h = tau_h
        reliability_gate = torch.sigmoid((reliability - tau_h) / max(1e-4, self.tau_w))

        # Transferability signal from cross-modal nearest neighbors
        z_norm = F.normalize(z, dim=1)
        top1_sim = torch.zeros(bsz, device=self.device)
        top2_sim = torch.zeros(bsz, device=self.device)
        has_cross = torch.zeros(bsz, dtype=torch.bool, device=self.device)
        valid_distances = []

        for mod_idx in range(m_num):
            mask = modality_labels == mod_idx
            if mask.sum() == 0:
                continue

            other_parts = []
            for other_idx in range(m_num):
                if other_idx == mod_idx:
                    continue
                cnt = self.buffer_count[other_idx]
                if cnt > 0:
                    other_parts.append(self.queues[other_idx][:cnt])
            if len(other_parts) == 0:
                continue

            other_z = torch.cat(other_parts, dim=0)
            z_mod = z_norm[mask]
            sim = z_mod @ other_z.t()
            k_use = 2 if sim.shape[1] >= 2 else 1
            vals, _ = sim.topk(k_use, dim=1)
            top1 = vals[:, 0]
            if k_use == 2:
                top2 = vals[:, 1]
            else:
                top2 = vals[:, 0]

            top1_sim[mask] = top1
            top2_sim[mask] = top2
            has_cross[mask] = True
            valid_distances.append(1.0 - top1)

        if len(valid_distances) > 0:
            all_d = torch.cat(valid_distances)
            min_valid = max(32, int(0.2 * bsz))
            if all_d.numel() >= min_valid:
                tau_raw = self.c_tau * torch.median(all_d).item()
                tau_nn = max(self.tau_min, min(self.tau_max, tau_raw))
                self.tau_nn_ema = self.eta * self.tau_nn_ema + (1.0 - self.eta) * tau_nn
            else:
                tau_nn = self.tau_nn_ema
        else:
            tau_nn = self.tau_nn_ema
        self.current_tau_nn = tau_nn

        transfer_sim = torch.full((bsz,), 0.5, device=self.device)
        if has_cross.any():
            d = (1.0 - top1_sim[has_cross]).clamp(min=0.0)
            transfer_sim[has_cross] = torch.exp(-d / max(1e-4, tau_nn))

        margin = (top1_sim - top2_sim).clamp(min=0.0)
        margin_score = torch.sigmoid((margin - self.orphan_margin_threshold) / max(1e-4, self.tau_w))
        transferability = self.alpha * transfer_sim + (1.0 - self.alpha) * margin_score

        orphan_mask = has_cross & (
            (top1_sim < self.orphan_sim_threshold) | (margin < self.orphan_margin_threshold)
        )

        pre_score = reliability_gate * transferability
        global_threshold = torch.quantile(pre_score, 1.0 - rho_global).item()

        weights = torch.full((bsz,), self.w_floor, device=self.device)
        counts = [int((modality_labels == mod_idx).sum().item()) for mod_idx in range(m_num)]
        max_count = max(counts) if len(counts) > 0 else 1

        mean_weight_by_modality = {}
        orphan_ratio_by_modality = {}
        thresholds_by_modality = {}

        for mod_idx in range(m_num):
            mask = modality_labels == mod_idx
            n_mod = int(mask.sum().item())
            if n_mod == 0:
                mean_weight_by_modality[mod_idx] = 0.0
                orphan_ratio_by_modality[mod_idx] = 0.0
                thresholds_by_modality[mod_idx] = float(self.tau_mod_ema[mod_idx])
                continue

            mod_score = pre_score[mask]
            rarity = 1.0 - (n_mod / max(1, max_count))
            rho_mod = min(0.95, max(0.05, self.rho_target + self.rarity_boost * rarity))

            if n_mod < self.min_count:
                tau_mod = global_threshold
            else:
                tau_batch = torch.quantile(mod_score, 1.0 - rho_mod).item()
                tau_mod = self.eta * self.tau_mod_ema[mod_idx] + (1.0 - self.eta) * tau_batch
            self.tau_mod_ema[mod_idx] = tau_mod
            thresholds_by_modality[mod_idx] = float(tau_mod)

            gate = torch.sigmoid((mod_score - tau_mod) / max(1e-4, self.tau_w))
            mod_w = self.w_floor + (1.0 - self.w_floor) * gate
            mod_w = mod_w + self.rarity_boost * rarity
            mod_w = torch.clamp(mod_w, self.w_floor, 1.0)

            mod_orphan = orphan_mask[mask]
            if mod_orphan.any():
                orphan_floor = torch.full_like(mod_w, self.w_orphan_min)
                mod_w = torch.where(mod_orphan, torch.maximum(mod_w, orphan_floor), mod_w)
            mod_w = torch.clamp(mod_w, self.w_floor, 1.0)

            weights[mask] = mod_w
            mean_weight_by_modality[mod_idx] = float(mod_w.mean().item())
            orphan_ratio_by_modality[mod_idx] = float(mod_orphan.float().mean().item())

        self.current_modality_thresholds = thresholds_by_modality
        self.current_mean_weight_by_modality = mean_weight_by_modality
        self.current_orphan_ratio_by_modality = orphan_ratio_by_modality

        return {
            "weights": weights.detach(),
            "reliability": reliability.detach(),
            "transferability": transferability.detach(),
            "orphan_mask": orphan_mask.detach(),
            "tau_nn": float(tau_nn),
            "tau_h": float(tau_h),
            "mean_weight_by_modality": mean_weight_by_modality,
            "orphan_ratio_by_modality": orphan_ratio_by_modality,
        }
