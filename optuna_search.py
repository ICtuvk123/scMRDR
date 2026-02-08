import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import optuna
from optuna.integration import TensorBoardCallback
import scanpy as sc
import numpy as np
import torch
import gc
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from scmrdr.module import Integration

# ============================================================
# Config
# ============================================================
DATA_PATH = "/root/autodl-tmp/scMRDR/experiments/BMMC_codes/feature_aligned.h5ad"
CELLTYPE_KEY = "celltype"         # adata.obs column for ground truth cell types
LATENT_KEY = "latent_shared"      # obsm key for the shared latent embeddings
TB_LOG_DIR = "./runs/optuna"      # TensorBoard log directory

# ============================================================
# 1. Load data (only once, shared across all trials)
# ============================================================
adata_raw = sc.read_h5ad(DATA_PATH)
adata_raw.obs_names_make_unique()
print(f"Data loaded: {adata_raw.shape[0]} cells x {adata_raw.shape[1]} features")
print(f"Modalities: {adata_raw.obs['modality'].value_counts().to_dict()}")

n_celltypes = adata_raw.obs[CELLTYPE_KEY].nunique()
print(f"Cell types ({CELLTYPE_KEY}): {n_celltypes} classes")

rna_hvg = np.where(adata_raw.var_names.isin(adata_raw.uns['rna_hvg']))[0].tolist()
atac_hvg = np.where(adata_raw.var_names.isin(adata_raw.uns['atac_hvg']))[0].tolist()
prot_hvg = np.where(adata_raw.var_names.isin(adata_raw.uns['prot_hvg']))[0].tolist()
feature_list = {"0": rna_hvg, "1": atac_hvg, "2": prot_hvg}
print(f"Features: RNA={len(rna_hvg)}, ATAC={len(atac_hvg)}, Protein={len(prot_hvg)}")


# ============================================================
# 2. NMI evaluation (KMeans, consistent with scib_metrics)
# ============================================================
def compute_kmeans_nmi(adata, label_key=CELLTYPE_KEY, use_rep=LATENT_KEY):
    """Compute KMeans NMI: cluster the latent space and compare with ground truth."""
    n_clusters = adata.obs[label_key].nunique()
    latent = adata.obsm[use_rep]
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    pred_labels = kmeans.fit_predict(latent)
    true_labels = adata.obs[label_key].values
    nmi = normalized_mutual_info_score(true_labels, pred_labels, average_method='arithmetic')
    return nmi


# ============================================================
# 3. Define Optuna objective
# ============================================================
def objective(trial):
    # --- Hyperparameters to search ---
    gamma = trial.suggest_float("gamma", 0.1, 50.0, log=True)
    lambda_adv = trial.suggest_float("lambda_adv", 0.1, 50.0, log=True)
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.4)
    latent_dim = trial.suggest_categorical("latent_dim", [10, 16, 20, 32])
    denoise_hidden_dim = trial.suggest_categorical("denoise_hidden_dim", [256, 512])

    print(f"\n{'='*60}")
    print(f"Trial {trial.number}: gamma={gamma:.3f}, lambda_adv={lambda_adv:.3f}, "
          f"lr={lr:.5f}, dropout={dropout_rate:.2f}, "
          f"latent_dim={latent_dim}, denoise_hidden={denoise_hidden_dim}")
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
        dropout_rate=dropout_rate,
        use_causal_dag=True,
        denoise_hidden_dim=denoise_hidden_dim,
    )

    # --- Train (no Optuna pruning, early stopping handles termination) ---
    model.train(
        epoch_num=50,
        batch_size=128,
        lr=lr,
        num_warmup=10,
        early_stopping=True,
        patience=10,
        valid_prop=0.1,
        adaptlr=False,
        tensorboard=False,
    )

    # --- Inference & compute NMI ---
    model.inference(n_samples=1, update=True)
    adata_result = model.get_adata()
    nmi = compute_kmeans_nmi(adata_result)
    print(f">>> Trial {trial.number} NMI = {nmi:.4f}")

    # Cleanup GPU memory
    del model, adata_result
    gc.collect()
    torch.cuda.empty_cache()

    return nmi


# ============================================================
# 4. Run study
# ============================================================
if __name__ == "__main__":
    # TensorBoard callback: logs params & NMI for each trial
    tb_callback = TensorBoardCallback(TB_LOG_DIR, metric_name="NMI")

    study = optuna.create_study(
        study_name="scmrdr_bmmc_nmi",
        direction="maximize",            # higher NMI = better
        storage="sqlite:///optuna_scmrdr.db",
        load_if_exists=True,
    )

    n_trials = 20
    study.optimize(objective, n_trials=n_trials, gc_after_trial=True, callbacks=[tb_callback])

    # --- Print results ---
    print("\n" + "=" * 60)
    print("BEST TRIAL:")
    print(f"  NMI: {study.best_trial.value:.4f}")
    print("  Params:")
    for k, v in study.best_trial.params.items():
        print(f"    {k}: {v}")

    print("\n" + "=" * 60)
    print("TOP 5 TRIALS:")
    trials_sorted = sorted(
        [t for t in study.trials if t.value is not None],
        key=lambda t: t.value, reverse=True
    )
    for i, t in enumerate(trials_sorted[:5]):
        print(f"  #{i+1} NMI={t.value:.4f}  params={t.params}")
