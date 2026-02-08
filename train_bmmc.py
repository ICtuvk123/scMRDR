import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import scanpy as sc
import numpy as np
from scmrdr.module import Integration

# ============================================================
# 1. Load data
# ============================================================
data_path = "/root/autodl-tmp/scMRDR/experiments/BMMC_codes/feature_aligned.h5ad"
adata = sc.read_h5ad(data_path)
adata.obs_names_make_unique()
print(f"Data loaded: {adata.shape[0]} cells × {adata.shape[1]} features")
print(f"Modalities: {adata.obs['modality'].value_counts().to_dict()}")

# Feature list per modality (mask unrelated features)
rna_hvg = np.where(adata.var_names.isin(adata.uns['rna_hvg']))[0].tolist()
atac_hvg = np.where(adata.var_names.isin(adata.uns['atac_hvg']))[0].tolist()
prot_hvg = np.where(adata.var_names.isin(adata.uns['prot_hvg']))[0].tolist()
feature_list = {"0": rna_hvg, "1": atac_hvg, "2": prot_hvg}
print(f"Features: RNA={len(rna_hvg)}, ATAC={len(atac_hvg)}, Protein={len(prot_hvg)}")

# ============================================================
# 2. Initialize model (ZINB distribution → auto ZINB loss)
# ============================================================
model = Integration(
    data=adata,
    layer='counts',
    modality_key="modality",
    batch_key="batch",
    distribution="ZINB",
    feature_list=feature_list,
)

# ============================================================
# 3. Setup with Causal DAG
# ============================================================
model.setup(
    hidden_layers=[128, 128],
    latent_dim_shared=20,
    latent_dim_specific=20,
    gamma=10,
    lambda_adv=10,
    dropout_rate=0.2,
    use_causal_dag=True,        # ← Causal DAG: shared → specific
    denoise_hidden_dim=512,     # ← 3090 24GB 必须设这个，否则默认=10173 OOM
)

# ============================================================
# 4. Train
# ============================================================
model.train(
    epoch_num=200,
    batch_size=128,
    lr=1e-3,
    num_warmup=10,
    early_stopping=True,
    patience=25,
    valid_prop=0.1,
    adaptlr=False,
    tensorboard=True,
    savepath="./runs/bmmc_causal_dag",
)

# ============================================================
# 5. Inference & Save
# ============================================================
model.inference(n_samples=1, update=True)
adata_out = model.get_adata()
save_path = "/root/autodl-tmp/scMRDR/experiments/BMMC_codes/feature_aligned_trained_causal.h5ad"
adata_out.write(save_path)
print(f"Results saved to {save_path}")
