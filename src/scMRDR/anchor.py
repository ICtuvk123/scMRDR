"""
MNN-based anchor loss for cross-modal alignment.

Implements anchor regularization inspired by scMODAL:
    L_anchor = (1/q) * sum_{(i,j) in MNN} ||mu_shared_i - mu_shared_j||_2^2

MNN pairs are found per modality pair using their pairwise linked features
(intersection of feat_mask rows) with cosine/angle distance.
"""

import torch
import torch.nn.functional as F


@torch.no_grad()
def find_mnn_pairs(X, modality_labels, feat_mask, k=30, linked_feature_idx=None,
                   sim_threshold=0.0, margin=0.0):
    """
    Find Mutual Nearest Neighbor pairs between modalities using linked features.

    For each pair of modalities (a, b):
      1. Compute pairwise linked features from feat_mask[a] * feat_mask[b]
      2. L2-normalize linked features (angle distance)
      3. Find k-NN in both directions, take mutual pairs

    Args:
        X: (B, D) raw input data tensor
        modality_labels: (B,) integer tensor of modality labels
        feat_mask: (M, D) binary tensor, feat_mask[m] indicates features for modality m
        k: number of nearest neighbors for MNN (default: 30)
        linked_feature_idx: optional 1D tensor/list of globally allowed feature
            indices. If provided, per-modality linked features are intersected with
            this set.
        sim_threshold: minimum cosine similarity to keep a pair (0 = disabled)
        margin: minimum top1-top2 cosine gap for each query (0 = disabled)

    Returns:
        idx1, idx2: (P,) LongTensors of batch indices forming MNN pairs
    """
    device = X.device
    unique_mods = torch.unique(modality_labels)
    if len(unique_mods) < 2:
        empty = torch.tensor([], dtype=torch.long, device=device)
        return empty, empty

    all_idx1 = []
    all_idx2 = []

    allowed_feature_mask = None
    if linked_feature_idx is not None:
        linked_feature_idx = torch.as_tensor(linked_feature_idx, device=device, dtype=torch.long)
        if linked_feature_idx.numel() > 0:
            allowed_feature_mask = torch.zeros(feat_mask.shape[1], dtype=torch.bool, device=device)
            allowed_feature_mask[linked_feature_idx] = True

    for a in range(len(unique_mods)):
        for b in range(a + 1, len(unique_mods)):
            mod_a, mod_b = unique_mods[a].item(), unique_mods[b].item()

            # Pairwise linked features from feat_mask intersection
            pair_mask = (feat_mask[mod_a] * feat_mask[mod_b]) > 0
            if allowed_feature_mask is not None:
                pair_mask = pair_mask & allowed_feature_mask
            linked_idx = torch.where(pair_mask)[0]
            if len(linked_idx) < 2:
                continue

            mask_a = modality_labels == mod_a
            mask_b = modality_labels == mod_b
            batch_idx_a = torch.where(mask_a)[0]
            batch_idx_b = torch.where(mask_b)[0]
            n_a, n_b = len(batch_idx_a), len(batch_idx_b)

            if n_a < 2 or n_b < 2:
                continue

            # Extract linked features and L2-normalize
            Xa = F.normalize(X[batch_idx_a][:, linked_idx].float(), dim=1, eps=1e-8)
            Xb = F.normalize(X[batch_idx_b][:, linked_idx].float(), dim=1, eps=1e-8)

            # Cosine similarity
            sim = Xa @ Xb.t()  # (n_a, n_b)

            # k-NN in both directions
            k_a2b = min(k, n_b)
            k_b2a = min(k, n_a)
            topk_a2b_vals, nn_a2b = sim.topk(k_a2b, dim=1)
            topk_b2a_vals, nn_b2a = sim.t().topk(k_b2a, dim=1)

            # Margin filter (per-query)
            valid_a = torch.ones(n_a, dtype=torch.bool, device=device)
            valid_b = torch.ones(n_b, dtype=torch.bool, device=device)
            if margin > 0:
                if k_a2b >= 2:
                    valid_a = (topk_a2b_vals[:, 0] - topk_a2b_vals[:, 1]) >= margin
                if k_b2a >= 2:
                    valid_b = (topk_b2a_vals[:, 0] - topk_b2a_vals[:, 1]) >= margin

            # Boolean adjacency → mutual intersection
            adj_a2b = torch.zeros(n_a, n_b, dtype=torch.bool, device=device)
            adj_a2b.scatter_(1, nn_a2b, True)
            adj_a2b[~valid_a] = False
            adj_b2a = torch.zeros(n_b, n_a, dtype=torch.bool, device=device)
            adj_b2a.scatter_(1, nn_b2a, True)
            adj_b2a[~valid_b] = False

            mutual = adj_a2b & adj_b2a.t()
            if sim_threshold > 0:
                mutual = mutual & (sim >= sim_threshold)
            local_a, local_b = torch.where(mutual)
            if len(local_a) > 0:
                all_idx1.append(batch_idx_a[local_a])
                all_idx2.append(batch_idx_b[local_b])

    if len(all_idx1) == 0:
        empty = torch.tensor([], dtype=torch.long, device=device)
        return empty, empty

    return torch.cat(all_idx1), torch.cat(all_idx2)


@torch.no_grad()
def find_mnn_pairs_latent(mu_shared, modality_labels, k=30,
                          sim_threshold=0.0, margin=0.0):
    """
    Find MNN pairs in the shared latent space with confidence filtering.

    Pairing uses stop-gradient (@torch.no_grad) to prevent the model from
    gaming pair selection.  The returned indices are meant to be used with
    the *original* (gradient-enabled) mu_shared in anchor_loss().

    Args:
        mu_shared: (B, d) shared latent means
        modality_labels: (B,) integer tensor of modality labels
        k: number of nearest neighbors for MNN
        sim_threshold: minimum cosine similarity to keep a pair (0 = disabled)
        margin: minimum gap between top-1 and top-2 cosine sim (0 = disabled)

    Returns:
        idx1, idx2: (P,) LongTensors of batch indices forming MNN pairs
    """
    device = mu_shared.device
    unique_mods = torch.unique(modality_labels)
    if len(unique_mods) < 2:
        empty = torch.tensor([], dtype=torch.long, device=device)
        return empty, empty

    all_idx1 = []
    all_idx2 = []

    for a in range(len(unique_mods)):
        for b in range(a + 1, len(unique_mods)):
            mod_a, mod_b = unique_mods[a].item(), unique_mods[b].item()

            mask_a = modality_labels == mod_a
            mask_b = modality_labels == mod_b
            batch_idx_a = torch.where(mask_a)[0]
            batch_idx_b = torch.where(mask_b)[0]
            n_a, n_b = len(batch_idx_a), len(batch_idx_b)

            if n_a < 2 or n_b < 2:
                continue

            # L2-normalize latent embeddings (cosine distance)
            Za = F.normalize(mu_shared[batch_idx_a].float(), dim=1, eps=1e-8)
            Zb = F.normalize(mu_shared[batch_idx_b].float(), dim=1, eps=1e-8)

            sim = Za @ Zb.t()  # (n_a, n_b)

            k_a2b = min(k, n_b)
            k_b2a = min(k, n_a)
            topk_a2b_vals, nn_a2b = sim.topk(k_a2b, dim=1)
            topk_b2a_vals, nn_b2a = sim.t().topk(k_b2a, dim=1)

            # --- Margin filter (per-query) ---
            valid_a = torch.ones(n_a, dtype=torch.bool, device=device)
            valid_b = torch.ones(n_b, dtype=torch.bool, device=device)
            if margin > 0:
                if k_a2b >= 2:
                    valid_a = (topk_a2b_vals[:, 0] - topk_a2b_vals[:, 1]) >= margin
                if k_b2a >= 2:
                    valid_b = (topk_b2a_vals[:, 0] - topk_b2a_vals[:, 1]) >= margin

            # Boolean adjacency -> mutual intersection
            adj_a2b = torch.zeros(n_a, n_b, dtype=torch.bool, device=device)
            adj_a2b.scatter_(1, nn_a2b, True)
            adj_a2b[~valid_a] = False

            adj_b2a = torch.zeros(n_b, n_a, dtype=torch.bool, device=device)
            adj_b2a.scatter_(1, nn_b2a, True)
            adj_b2a[~valid_b] = False

            mutual = adj_a2b & adj_b2a.t()

            # Similarity threshold
            if sim_threshold > 0:
                mutual = mutual & (sim >= sim_threshold)

            local_a, local_b = torch.where(mutual)
            if len(local_a) > 0:
                all_idx1.append(batch_idx_a[local_a])
                all_idx2.append(batch_idx_b[local_b])

    if len(all_idx1) == 0:
        empty = torch.tensor([], dtype=torch.long, device=device)
        return empty, empty

    return torch.cat(all_idx1), torch.cat(all_idx2)


def anchor_loss(mu_shared, idx1, idx2):
    """
    Compute L2 anchor loss between MNN pairs in the shared latent space.

    L_anchor = (1/q) * sum_{(i,j)} ||mu_shared_i - mu_shared_j||_2^2

    Args:
        mu_shared: (B, latent_dim) shared latent means (requires grad)
        idx1, idx2: (P,) LongTensors of batch indices forming MNN pairs

    Returns:
        loss: scalar anchor loss (0.0 if no pairs)
    """
    if len(idx1) == 0:
        return torch.tensor(0.0, device=mu_shared.device)

    return ((mu_shared[idx1] - mu_shared[idx2]) ** 2).sum(dim=1).mean()
