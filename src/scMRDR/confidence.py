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
