import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import scanpy as sc
import numpy as np
import torch
import gc
import random
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, silhouette_score, silhouette_samples, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA, TruncatedSVD
from scipy.sparse import issparse, csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import chi2
from scmrdr.module import Integration

# ============================================================
# Config
# ============================================================
DATA_PATH = "./experiments/BMMC_codes/feature_aligned_sampled.h5ad"
CELLTYPE_KEY = "celltype"         # adata.obs column for ground truth cell types
LATENT_KEY = "latent_shared"      # obsm key for the shared latent embeddings
TB_LOG_DIR = "./runs/optuna"      # TensorBoard log directory
# Feature caps for better modality balance in shared encoder training.
# 0 means "use all available features for that modality".
RNA_FEATURE_CAP = int(os.getenv("SCMRDR_RNA_FEATURE_CAP", "0"))
ATAC_FEATURE_CAP = int(os.getenv("SCMRDR_ATAC_FEATURE_CAP", "0"))
PROT_FEATURE_CAP = int(os.getenv("SCMRDR_PROT_FEATURE_CAP", "0"))
SEARCH_CFG = {}
METRIC_WEIGHTS = {
    "nmi_kmeans": 1.0,
    "ari_kmeans": 1.0,
    "silhouette_label": 1.0,
    "isolated_labels_asw": 0.8,
    "isolated_labels_f1": 0.8,
    "clisi": 1.0,
    "batch_silhouette": 1.0,
    "batch_ilisi": 1.0,
    "batch_kbet": 1.0,
    "batch_pcr_comparison": 1.0,
    "modality_silhouette": 1.0,
    "modality_ilisi": 1.0,
    "modality_kbet": 1.0,
    "graph_connectivity": 1.0,
    "modality_pcr_comparison": 1.0,
}

# ============================================================
# 1. Load data (only once, shared across all trials)
# ============================================================
SUBSAMPLE = 50000  # subsample for faster HP search; set None for full data

adata_raw = sc.read_h5ad(DATA_PATH)
adata_raw.obs_names_make_unique()
print(f"Full data: {adata_raw.shape[0]} cells x {adata_raw.shape[1]} features")

if SUBSAMPLE is not None and adata_raw.shape[0] > SUBSAMPLE:
    sc.pp.subsample(adata_raw, n_obs=SUBSAMPLE, random_state=42)
    print(f"Subsampled to {adata_raw.shape[0]} cells for HP search")

print(f"Modalities: {adata_raw.obs['modality'].value_counts().to_dict()}")

n_celltypes = adata_raw.obs[CELLTYPE_KEY].nunique()
print(f"Cell types ({CELLTYPE_KEY}): {n_celltypes} classes")

rna_hvg = np.where(adata_raw.var_names.isin(adata_raw.uns['rna_hvg']))[0].tolist()
atac_hvg = np.where(adata_raw.var_names.isin(adata_raw.uns['atac_hvg']))[0].tolist()
prot_hvg = np.where(adata_raw.var_names.isin(adata_raw.uns['prot_hvg']))[0].tolist()

if RNA_FEATURE_CAP > 0:
    rna_hvg = rna_hvg[: min(RNA_FEATURE_CAP, len(rna_hvg))]
if ATAC_FEATURE_CAP > 0:
    atac_hvg = atac_hvg[: min(ATAC_FEATURE_CAP, len(atac_hvg))]
if PROT_FEATURE_CAP > 0:
    prot_hvg = prot_hvg[: min(PROT_FEATURE_CAP, len(prot_hvg))]

feature_list = {"0": rna_hvg, "1": atac_hvg, "2": prot_hvg}
print(
    f"Features: RNA={len(rna_hvg)}, ATAC={len(atac_hvg)}, Protein={len(prot_hvg)} "
    f"(caps: RNA={RNA_FEATURE_CAP}, ATAC={ATAC_FEATURE_CAP}, Protein={PROT_FEATURE_CAP})"
)

EVAL_SUBSAMPLE = int(os.getenv("SCMRDR_EVAL_SUBSAMPLE", "10000"))
EVAL_K = int(os.getenv("SCMRDR_EVAL_K", "30"))
PCA_COMPONENTS = int(os.getenv("SCMRDR_PCA_COMPONENTS", "30"))
if EVAL_SUBSAMPLE > 0 and adata_raw.n_obs > EVAL_SUBSAMPLE:
    rng_eval = np.random.RandomState(42)
    EVAL_IDX = rng_eval.choice(adata_raw.n_obs, size=EVAL_SUBSAMPLE, replace=False)
    EVAL_IDX.sort()
else:
    EVAL_IDX = np.arange(adata_raw.n_obs)
print(f"Metric eval subset: {len(EVAL_IDX)} cells (SCMRDR_EVAL_SUBSAMPLE={EVAL_SUBSAMPLE})")

EVAL_CELLTYPE = adata_raw.obs[CELLTYPE_KEY].to_numpy()[EVAL_IDX]
EVAL_BATCH = adata_raw.obs["batch"].to_numpy()[EVAL_IDX]
EVAL_MODALITY = adata_raw.obs["modality"].to_numpy()[EVAL_IDX]

if "counts" in adata_raw.layers:
    X_eval_raw = adata_raw.layers["counts"][EVAL_IDX]
else:
    X_eval_raw = adata_raw.X[EVAL_IDX]


# ============================================================
# 2. Multi-metric evaluation
# ============================================================
def _to_dense(x):
    return x.toarray() if issparse(x) else np.asarray(x)


def _scaled_embed(x):
    return StandardScaler().fit_transform(x)


def _knn_indices(embed, k):
    k_eff = max(2, min(k, embed.shape[0] - 1))
    nn = NearestNeighbors(n_neighbors=k_eff + 1, metric="euclidean")
    nn.fit(embed)
    idx = nn.kneighbors(embed, return_distance=False)
    return idx[:, 1:]


def _inverse_simpson_per_cell(nei_idx, labels):
    labels = np.asarray(labels)
    uniq, enc = np.unique(labels, return_inverse=True)
    n_labels = len(uniq)
    out = np.zeros(nei_idx.shape[0], dtype=float)
    for i in range(nei_idx.shape[0]):
        neigh = enc[nei_idx[i]]
        counts = np.bincount(neigh, minlength=n_labels).astype(float)
        p = counts / max(1.0, counts.sum())
        out[i] = 1.0 / max(1e-12, np.sum(p * p))
    return out, n_labels


def _ilisi_score(nei_idx, labels):
    inv_s, n_labels = _inverse_simpson_per_cell(nei_idx, labels)
    ilisi = float(np.mean(inv_s))
    if n_labels <= 1:
        return 1.0, ilisi
    score = (ilisi - 1.0) / (n_labels - 1.0)
    return float(np.clip(score, 0.0, 1.0)), ilisi


def _clisi_score(nei_idx, labels):
    inv_s, n_labels = _inverse_simpson_per_cell(nei_idx, labels)
    clisi = float(np.mean(inv_s))
    if n_labels <= 1:
        return 1.0, clisi
    score = (n_labels - clisi) / (n_labels - 1.0)  # lower cLISI is better -> higher score
    return float(np.clip(score, 0.0, 1.0)), clisi


def _kbet_acceptance(nei_idx, batch_labels):
    batches, enc = np.unique(batch_labels, return_inverse=True)
    if len(batches) <= 1:
        return 1.0
    global_p = np.bincount(enc).astype(float)
    global_p /= global_p.sum()
    reject = 0
    dof = len(batches) - 1
    for i in range(nei_idx.shape[0]):
        obs = np.bincount(enc[nei_idx[i]], minlength=len(batches)).astype(float)
        exp = global_p * obs.sum()
        stat = np.sum((obs - exp) ** 2 / np.clip(exp, 1e-8, None))
        pval = chi2.sf(stat, dof)
        if pval < 0.05:
            reject += 1
    return float(1.0 - reject / max(1, nei_idx.shape[0]))


def _graph_connectivity(nei_idx, labels):
    n = nei_idx.shape[0]
    rows = np.repeat(np.arange(n), nei_idx.shape[1])
    cols = nei_idx.reshape(-1)
    data = np.ones_like(cols, dtype=np.float32)
    graph = csr_matrix((data, (rows, cols)), shape=(n, n))
    graph = graph.maximum(graph.T)
    label_arr = np.asarray(labels)
    scores = []
    for lbl in np.unique(label_arr):
        idx = np.where(label_arr == lbl)[0]
        if idx.size <= 1:
            scores.append(1.0)
            continue
        sub = graph[idx][:, idx]
        n_comp, comp = connected_components(sub, directed=False, return_labels=True)
        largest = np.bincount(comp).max()
        scores.append(largest / idx.size)
    return float(np.mean(scores)) if scores else np.nan


def _batch_silhouette(embed, labels):
    uniq = np.unique(labels)
    if uniq.size < 2:
        return 1.0
    sil = silhouette_score(embed, labels, metric="euclidean")
    return float(np.clip(1.0 - abs(sil), 0.0, 1.0))


def _label_silhouette(embed, labels):
    uniq = np.unique(labels)
    if uniq.size < 2:
        return np.nan
    sil = silhouette_score(embed, labels, metric="euclidean")
    return float((sil + 1.0) / 2.0)


def _isolated_label_stats(embed, labels):
    labels = np.asarray(labels)
    uniq, counts = np.unique(labels, return_counts=True)
    if uniq.size < 2:
        return np.nan, np.nan

    threshold = np.percentile(counts, 25)
    isolated = uniq[counts <= threshold]
    if isolated.size == 0:
        isolated = uniq[np.argsort(counts)[: max(1, uniq.size // 4)]]
    mask_iso = np.isin(labels, isolated)

    sil_vals = silhouette_samples(embed, labels, metric="euclidean")
    iso_asw = float(np.mean((sil_vals[mask_iso] + 1.0) / 2.0)) if mask_iso.any() else np.nan

    if mask_iso.sum() < 20:
        return iso_asw, np.nan
    y_bin = mask_iso.astype(int)
    if y_bin.min() == y_bin.max():
        return iso_asw, np.nan
    X_train, X_test, y_train, y_test = train_test_split(
        embed, y_bin, test_size=0.3, random_state=42, stratify=y_bin
    )
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    iso_f1 = float(f1_score(y_test, pred))
    return iso_asw, iso_f1


def _pca_embed(x, n_components=PCA_COMPONENTS):
    sparse_input = issparse(x)
    if not sparse_input:
        x = np.asarray(x)
    if x.shape[1] <= n_components:
        return _to_dense(x) if sparse_input else x
    if sparse_input:
        return TruncatedSVD(n_components=n_components, random_state=42).fit_transform(x)
    return PCA(n_components=n_components, random_state=42).fit_transform(x)


def _categorical_r2(embed, labels):
    if embed.ndim == 1:
        embed = embed[:, None]
    uniq, enc = np.unique(labels, return_inverse=True)
    if len(uniq) <= 1:
        return 0.0
    onehot = np.eye(len(uniq))[enc]
    r2s = []
    for j in range(embed.shape[1]):
        y = embed[:, j]
        y_mean = y.mean()
        sst = np.sum((y - y_mean) ** 2)
        if sst <= 1e-12:
            continue
        XtX = onehot.T @ onehot
        XtY = onehot.T @ y
        beta = np.linalg.solve(XtX + 1e-6 * np.eye(XtX.shape[0]), XtY)
        y_hat = onehot @ beta
        sse = np.sum((y - y_hat) ** 2)
        r2s.append(1.0 - sse / sst)
    return float(np.mean(r2s)) if r2s else 0.0


RAW_PCS = _pca_embed(X_eval_raw)
RAW_BATCH_R2 = _categorical_r2(RAW_PCS, EVAL_BATCH)
RAW_MODALITY_R2 = _categorical_r2(RAW_PCS, EVAL_MODALITY)
print(
    f"PCR baselines on raw data: batch_r2={RAW_BATCH_R2:.4f}, "
    f"modality_r2={RAW_MODALITY_R2:.4f}"
)


def compute_metrics(adata, use_rep=LATENT_KEY):
    latent = _scaled_embed(adata.obsm[use_rep][EVAL_IDX])
    true_labels = EVAL_CELLTYPE
    batch_labels = EVAL_BATCH
    modality_labels = EVAL_MODALITY

    n_clusters = len(np.unique(true_labels))
    km = KMeans(n_clusters=n_clusters, n_init=50, random_state=42)
    pred = km.fit_predict(latent)

    metrics = {}
    metrics["nmi_kmeans"] = float(
        normalized_mutual_info_score(true_labels, pred, average_method="arithmetic")
    )
    metrics["ari_kmeans"] = float(adjusted_rand_score(true_labels, pred))
    metrics["silhouette_label"] = _label_silhouette(latent, true_labels)

    iso_asw, iso_f1 = _isolated_label_stats(latent, true_labels)
    metrics["isolated_labels_asw"] = iso_asw
    metrics["isolated_labels_f1"] = iso_f1

    nei_idx = _knn_indices(latent, EVAL_K)
    clisi_score, _ = _clisi_score(nei_idx, true_labels)
    metrics["clisi"] = clisi_score

    metrics["batch_silhouette"] = _batch_silhouette(latent, batch_labels)
    batch_ilisi, _ = _ilisi_score(nei_idx, batch_labels)
    metrics["batch_ilisi"] = batch_ilisi
    metrics["batch_kbet"] = _kbet_acceptance(nei_idx, batch_labels)

    post_batch_r2 = _categorical_r2(latent, batch_labels)
    if RAW_BATCH_R2 <= 1e-8:
        metrics["batch_pcr_comparison"] = 1.0
    else:
        metrics["batch_pcr_comparison"] = float(np.clip(1.0 - (post_batch_r2 / RAW_BATCH_R2), 0.0, 1.0))

    metrics["modality_silhouette"] = _batch_silhouette(latent, modality_labels)
    modality_ilisi, _ = _ilisi_score(nei_idx, modality_labels)
    metrics["modality_ilisi"] = modality_ilisi
    metrics["modality_kbet"] = _kbet_acceptance(nei_idx, modality_labels)
    metrics["graph_connectivity"] = _graph_connectivity(nei_idx, true_labels)

    post_modality_r2 = _categorical_r2(latent, modality_labels)
    if RAW_MODALITY_R2 <= 1e-8:
        metrics["modality_pcr_comparison"] = 1.0
    else:
        metrics["modality_pcr_comparison"] = float(
            np.clip(1.0 - (post_modality_r2 / RAW_MODALITY_R2), 0.0, 1.0)
        )

    return metrics


def composite_score(metrics):
    score = 0.0
    weight_sum = 0.0
    for k, w in METRIC_WEIGHTS.items():
        v = metrics.get(k, np.nan)
        if v is None or np.isnan(v):
            continue
        score += w * float(v)
        weight_sum += w
    if weight_sum == 0:
        return np.nan
    return score / weight_sum


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_fixed_experiment(
    *,
    name: str,
    seed: int,
    gamma: float,
    lambda_adv: float,
    lambda_sp_cls: float = 1.0,
    lr: float,
    dropout_rate: float,
    latent_dim: int,
    denoise_hidden_dim: int,
    batch_size: int,
    epoch_num: int,
    patience: int,
    num_warmup: int,
    use_causal_dag: bool,
    weighted: bool,
):
    set_global_seed(seed)

    print(f"\n{'='*60}")
    print(
        f"{name}: gamma={gamma:.3f}, lambda_adv={lambda_adv:.3f}, "
        f"lambda_sp_cls={lambda_sp_cls:.3f}, lr={lr:.5f}, dropout={dropout_rate:.2f}, "
        f"latent_dim={latent_dim}, denoise_hidden={denoise_hidden_dim}, "
        f"batch_size={batch_size}, causal_dag={use_causal_dag}, weighted={weighted}"
    )
    print(f"{'='*60}")

    model = Integration(
        data=adata_raw,
        layer='counts',
        modality_key="modality",
        batch_key="batch",
        distribution="ZINB",
        feature_list=feature_list,
    )
    model.setup(
        hidden_layers=[128, 128],
        latent_dim_shared=latent_dim,
        latent_dim_specific=latent_dim,
        gamma=gamma,
        lambda_adv=lambda_adv,
        lambda_sp_cls=lambda_sp_cls,
        dropout_rate=dropout_rate,
        use_causal_dag=use_causal_dag,
        denoise_hidden_dim=denoise_hidden_dim,
    )
    model.train(
        epoch_num=epoch_num,
        batch_size=batch_size,
        lr=lr,
        num_warmup=num_warmup,
        early_stopping=True,
        patience=patience,
        valid_prop=0.1,
        adaptlr=True,
        weighted=weighted,
        tensorboard=False,
    )

    model.inference(n_samples=1, update=True)
    adata_result = model.get_adata()
    metrics = compute_metrics(adata_result)
    score = composite_score(metrics)
    print(
        f">>> {name} score={score:.4f}\n"
        f"    [Bio] NMI={metrics['nmi_kmeans']:.4f}, ARI={metrics['ari_kmeans']:.4f}, "
        f"label_ASW={metrics['silhouette_label']:.4f}, cLISI={metrics['clisi']:.4f}, "
        f"iso_ASW={metrics.get('isolated_labels_asw', float('nan')):.4f}, "
        f"iso_F1={metrics.get('isolated_labels_f1', float('nan')):.4f}\n"
        f"    [Batch] sil={metrics['batch_silhouette']:.4f}, "
        f"iLISI={metrics['batch_ilisi']:.4f}, kBET={metrics['batch_kbet']:.4f}, "
        f"PCR={metrics['batch_pcr_comparison']:.4f}\n"
        f"    [Modality] sil={metrics['modality_silhouette']:.4f}, "
        f"iLISI={metrics['modality_ilisi']:.4f}, kBET={metrics['modality_kbet']:.4f}, "
        f"PCR={metrics['modality_pcr_comparison']:.4f}, "
        f"graph_conn={metrics['graph_connectivity']:.4f}"
    )

    del model, adata_result
    gc.collect()
    torch.cuda.empty_cache()
    return score, metrics


# ============================================================
# 3. Define Optuna objective
# ============================================================
def objective(trial):
    set_global_seed(42 + trial.number)

    # --- Hyperparameters to search ---
    gamma = trial.suggest_float(
        "gamma", SEARCH_CFG["gamma_min"], SEARCH_CFG["gamma_max"], log=True
    )
    lambda_adv = trial.suggest_float(
        "lambda_adv", SEARCH_CFG["lambda_min"], SEARCH_CFG["lambda_max"], log=True
    )
    lr = trial.suggest_float("lr", SEARCH_CFG["lr_min"], SEARCH_CFG["lr_max"], log=True)
    dropout_rate = trial.suggest_float(
        "dropout_rate", SEARCH_CFG["dropout_min"], SEARCH_CFG["dropout_max"]
    )
    latent_dim = trial.suggest_categorical("latent_dim", SEARCH_CFG["latent_dim_choices"])
    lambda_sp_cls = trial.suggest_float(
        "lambda_sp_cls", SEARCH_CFG["lambda_sp_cls_min"], SEARCH_CFG["lambda_sp_cls_max"], log=True
    )
    denoise_hidden_dim = SEARCH_CFG["denoise_hidden_dim"]
    batch_size = trial.suggest_categorical("batch_size", SEARCH_CFG["batch_size_choices"])

    print(f"\n{'='*60}")
    print(f"Trial {trial.number}: gamma={gamma:.3f}, lambda_adv={lambda_adv:.3f}, "
          f"lambda_sp_cls={lambda_sp_cls:.3f}, lr={lr:.5f}, dropout={dropout_rate:.2f}, "
          f"latent_dim={latent_dim}, denoise_hidden={denoise_hidden_dim}, "
          f"batch_size={batch_size}")
    print(f"{'='*60}")

    # --- Build model ---
    model = Integration(
        data=adata_raw,
        layer='counts',
        modality_key="modality",
        batch_key="batch",
        distribution="ZINB",
        feature_list=feature_list,
    )
    model.setup(
        hidden_layers=[128, 128],
        latent_dim_shared=latent_dim,
        latent_dim_specific=latent_dim,
        gamma=gamma,
        lambda_adv=lambda_adv,
        lambda_sp_cls=lambda_sp_cls,
        dropout_rate=dropout_rate,
        use_causal_dag=SEARCH_CFG["use_causal_dag"],
        denoise_hidden_dim=denoise_hidden_dim,
    )

    # --- Train (no Optuna pruning, early stopping handles termination) ---
    def score_prune_eval():
        model.inference(n_samples=1, update=True)
        adata_result = model.get_adata()
        return composite_score(compute_metrics(adata_result))

    model.train(
        epoch_num=SEARCH_CFG["epoch_num"],
        batch_size=batch_size,
        lr=lr,
        num_warmup=SEARCH_CFG["num_warmup"],
        early_stopping=True,
        patience=SEARCH_CFG["patience"],
        valid_prop=0.1,
        adaptlr=True,
        weighted=True,
        tensorboard=False,
        trial=trial,
        nmi_eval_fn=score_prune_eval,
        nmi_eval_interval=10,
        nmi_eval_start=30,
    )

    # --- Inference & compute multi-metric composite score ---
    model.inference(n_samples=1, update=True)
    adata_result = model.get_adata()
    metrics = compute_metrics(adata_result)
    score = composite_score(metrics)
    print(
        f">>> Trial {trial.number} score={score:.4f}\n"
        f"    [Bio] NMI={metrics['nmi_kmeans']:.4f}, ARI={metrics['ari_kmeans']:.4f}, "
        f"label_ASW={metrics['silhouette_label']:.4f}, cLISI={metrics['clisi']:.4f}, "
        f"iso_ASW={metrics.get('isolated_labels_asw', float('nan')):.4f}, "
        f"iso_F1={metrics.get('isolated_labels_f1', float('nan')):.4f}\n"
        f"    [Batch] sil={metrics['batch_silhouette']:.4f}, "
        f"iLISI={metrics['batch_ilisi']:.4f}, kBET={metrics['batch_kbet']:.4f}, "
        f"PCR={metrics['batch_pcr_comparison']:.4f}\n"
        f"    [Modality] sil={metrics['modality_silhouette']:.4f}, "
        f"iLISI={metrics['modality_ilisi']:.4f}, kBET={metrics['modality_kbet']:.4f}, "
        f"PCR={metrics['modality_pcr_comparison']:.4f}, "
        f"graph_conn={metrics['graph_connectivity']:.4f}"
    )
    trial.set_user_attr("metrics", {k: (None if np.isnan(v) else float(v)) for k, v in metrics.items()})
    trial.set_user_attr("composite_score", None if np.isnan(score) else float(score))

    # Cleanup GPU memory
    del model, adata_result
    gc.collect()
    torch.cuda.empty_cache()

    return score


# ============================================================
# 4. Run study
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna search for scMRDR.")
    parser.add_argument(
        "--phase",
        choices=["coarse", "refine"],
        default="coarse",
        help="Search phase: coarse (wide lambda) or refine (lambda segment).",
    )
    parser.add_argument(
        "--use-causal-dag",
        action="store_true",
        help="Enable causal DAG in search mode. Default is disabled.",
    )
    parser.add_argument(
        "--target-total-trials",
        type=int,
        default=100,
        help="Target total number of trials in this study (including existing trials).",
    )
    parser.add_argument(
        "--lambda-low",
        type=float,
        default=0.02,
        help="Lower bound of lambda_adv for refine phase.",
    )
    parser.add_argument(
        "--lambda-high",
        type=float,
        default=0.8,
        help="Upper bound of lambda_adv for refine phase.",
    )
    parser.add_argument(
        "--ablation4",
        action="store_true",
        help="Run 4 fixed ablation experiments: no_dag/no_adv baseline and +adv/+dag variants.",
    )
    parser.add_argument("--ablation-epochs", type=int, default=120)
    parser.add_argument("--ablation-patience", type=int, default=20)
    parser.add_argument("--ablation-num-warmup", type=int, default=10)
    parser.add_argument("--ablation-lr", type=float, default=8e-4)
    parser.add_argument("--ablation-gamma", type=float, default=40.0)
    parser.add_argument("--ablation-lambda-adv", type=float, default=0.2)
    parser.add_argument("--ablation-lambda-sp-cls", type=float, default=1.0)
    parser.add_argument("--ablation-dropout", type=float, default=0.22)
    parser.add_argument("--ablation-latent-dim", type=int, default=16)
    parser.add_argument("--ablation-batch-size", type=int, default=128)
    parser.add_argument("--ablation-denoise-hidden-dim", type=int, default=512)
    parser.add_argument(
        "--ablation-seeds",
        type=int,
        default=1,
        help="Number of random seeds to run for each ablation setting.",
    )
    args = parser.parse_args()

    if args.ablation4:
        if args.ablation_seeds < 1:
            raise ValueError("--ablation-seeds must be >= 1.")
        base_cfg = {
            "seed": 42,
            "gamma": args.ablation_gamma,
            "lambda_sp_cls": args.ablation_lambda_sp_cls,
            "lr": args.ablation_lr,
            "dropout_rate": args.ablation_dropout,
            "latent_dim": args.ablation_latent_dim,
            "denoise_hidden_dim": args.ablation_denoise_hidden_dim,
            "batch_size": args.ablation_batch_size,
            "epoch_num": args.ablation_epochs,
            "patience": args.ablation_patience,
            "num_warmup": args.ablation_num_warmup,
            "weighted": False,
        }
        ablations = [
            {
                "name": "A1_baseline_no_dag_no_adv_unweighted",
                "use_causal_dag": False,
                "lambda_adv": 0.0,
            },
            {
                "name": "A2_add_adv_no_dag_unweighted",
                "use_causal_dag": False,
                "lambda_adv": args.ablation_lambda_adv,
            },
            {
                "name": "A3_add_dag_no_adv_unweighted",
                "use_causal_dag": True,
                "lambda_adv": 0.0,
            },
            {
                "name": "A4_add_dag_add_adv_unweighted",
                "use_causal_dag": True,
                "lambda_adv": args.ablation_lambda_adv,
            },
        ]
        results = []
        print(f"Running ablation4 with seeds_per_setting={args.ablation_seeds}")
        for idx, cfg in enumerate(ablations):
            scores = []
            nmis = []
            for seed_offset in range(args.ablation_seeds):
                run_name = f"{cfg['name']}_seed{seed_offset+1}"
                run_seed = base_cfg["seed"] + idx * 1000 + seed_offset
                score, metrics = run_fixed_experiment(
                    name=run_name,
                    seed=run_seed,
                    gamma=base_cfg["gamma"],
                    lambda_adv=cfg["lambda_adv"],
                    lambda_sp_cls=base_cfg["lambda_sp_cls"],
                    lr=base_cfg["lr"],
                    dropout_rate=base_cfg["dropout_rate"],
                    latent_dim=base_cfg["latent_dim"],
                    denoise_hidden_dim=base_cfg["denoise_hidden_dim"],
                    batch_size=base_cfg["batch_size"],
                    epoch_num=base_cfg["epoch_num"],
                    patience=base_cfg["patience"],
                    num_warmup=base_cfg["num_warmup"],
                    use_causal_dag=cfg["use_causal_dag"],
                    weighted=base_cfg["weighted"],
                )
                scores.append(score)
                nmis.append(metrics["nmi_kmeans"])

            mean_score = float(np.mean(scores))
            std_score = float(np.std(scores))
            results.append(
                {
                    "name": cfg["name"],
                    "mean_score": mean_score,
                    "std_score": std_score,
                    "nmis": nmis,
                    "scores": scores,
                }
            )

        print("\n" + "=" * 60)
        print("ABLATION-4 RESULTS (sorted by mean composite score)")
        results_sorted = sorted(results, key=lambda x: x["mean_score"], reverse=True)
        for rank, item in enumerate(results_sorted, start=1):
            score_list = ", ".join(f"{v:.4f}" for v in item["scores"])
            nmi_list = ", ".join(f"{v:.4f}" for v in item["nmis"])
            print(
                f"  #{rank} score_mean={item['mean_score']:.4f} score_std={item['std_score']:.4f}  "
                f"{item['name']}  score_runs=[{score_list}]  nmi_runs=[{nmi_list}]"
            )
        sys.exit(0)

    # Import optuna only when running search mode.
    import optuna
    from optuna.integration import TensorBoardCallback

    if args.phase == "coarse":
        SEARCH_CFG = {
            "gamma_min": 10.0,
            "gamma_max": 500.0,
            "lambda_min": 0.01,
            "lambda_max": 5.0,
            "lambda_sp_cls_min": 0.1,
            "lambda_sp_cls_max": 10.0,
            "lr_min": 3e-4,
            "lr_max": 1.5e-3,
            "dropout_min": 0.18,
            "dropout_max": 0.30,
            "latent_dim_choices": [16, 20, 32, 64],
            "denoise_hidden_dim": 512,
            "batch_size_choices": [128, 256],
            "epoch_num": 180,
            "patience": 20,
            "num_warmup": 10,
            "use_causal_dag": args.use_causal_dag,
        }
    else:
        if not (args.lambda_low > 0 and args.lambda_high > args.lambda_low):
            raise ValueError("Invalid lambda range: require 0 < lambda_low < lambda_high.")
        SEARCH_CFG = {
            "gamma_min": 20.0,
            "gamma_max": 400.0,
            "lambda_min": args.lambda_low,
            "lambda_max": args.lambda_high,
            "lambda_sp_cls_min": 0.1,
            "lambda_sp_cls_max": 10.0,
            "lr_min": 3e-4,
            "lr_max": 1.5e-3,
            "dropout_min": 0.18,
            "dropout_max": 0.30,
            "latent_dim_choices": [16, 20, 32, 64],
            "denoise_hidden_dim": 512,
            "batch_size_choices": [128, 256],
            "epoch_num": 260,
            "patience": 30,
            "num_warmup": 20,
            "use_causal_dag": args.use_causal_dag,
        }

    print(f"Search phase: {args.phase}")
    print(f"Search config: {SEARCH_CFG}")
    print(f"Metric weights: {METRIC_WEIGHTS}")

    # TensorBoard callback: logs params & composite score for each trial
    tb_callback = TensorBoardCallback(TB_LOG_DIR, metric_name="CompositeScore")

    pruner = (
        optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=30, interval_steps=10)
        if args.phase == "coarse"
        else optuna.pruners.NopPruner()
    )

    study = optuna.create_study(
        study_name="scmrdr_bmmc_nmi_v2",
        direction="maximize",            # higher composite score = better
        storage="sqlite:///optuna_scmrdr.db",
        load_if_exists=True,
        pruner=pruner,
    )

    existing_trials = len(study.trials)
    remaining_trials = max(0, args.target_total_trials - existing_trials)
    print(
        f"Study has {existing_trials} existing trials; "
        f"target total={args.target_total_trials}; remaining to run={remaining_trials}"
    )

    if remaining_trials > 0:
        study.optimize(
            objective,
            n_trials=remaining_trials,
            gc_after_trial=True,
            callbacks=[tb_callback],
        )
    else:
        print("No remaining trials to run.")

    # --- Print results ---
    print("\n" + "=" * 60)
    print("BEST TRIAL:")
    print(f"  CompositeScore: {study.best_trial.value:.4f}")
    print("  Params:")
    for k, v in study.best_trial.params.items():
        print(f"    {k}: {v}")
    metrics_best = study.best_trial.user_attrs.get("metrics")
    if metrics_best:
        print("  Metrics:")
        for k in sorted(metrics_best.keys()):
            v = metrics_best[k]
            if v is not None:
                print(f"    {k}: {v:.4f}")

    print("\n" + "=" * 60)
    print("TOP 5 TRIALS:")
    trials_sorted = sorted(
        [t for t in study.trials if t.value is not None],
        key=lambda t: t.value, reverse=True
    )
    for i, t in enumerate(trials_sorted[:5]):
        nmi = None
        if isinstance(t.user_attrs.get("metrics"), dict):
            nmi = t.user_attrs["metrics"].get("nmi_kmeans")
        if nmi is None:
            print(f"  #{i+1} score={t.value:.4f}  params={t.params}")
        else:
            print(f"  #{i+1} score={t.value:.4f} nmi={nmi:.4f}  params={t.params}")
