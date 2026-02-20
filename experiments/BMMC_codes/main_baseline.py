"""
scMRDR 基线训练脚本（无 Gating）
==================================
与 main_gating.py 参数完全一致，仅关闭 confidence_weighted。
用于与 Gating 版本做公平对比。
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

import scanpy as sc
from scMRDR.module import Integration
from sklearn.metrics import adjusted_rand_score

DATA_PATH = os.path.join(os.path.dirname(__file__), "feature_aligned_sampled.h5ad")
SAVE_PATH = os.path.join(os.path.dirname(__file__), "feature_aligned_trained_baseline.h5ad")

print("正在加载数据...")
adata = sc.read_h5ad(DATA_PATH)
print(f"数据加载完成: {adata.shape[0]} 细胞 × {adata.shape[1]} 特征")

model = Integration(
    data=adata,
    layer="counts",
    modality_key="modality",
    batch_key="batch",
    distribution="ZINB",
    feature_list=None,
)

model.setup(
    hidden_layers=[512, 512],
    latent_dim_shared=20,
    latent_dim_specific=20,
    dropout_rate=0.2,
    beta=2,
    gamma=5,
    lambda_adv=5,
    confidence_weighted=False,   # 基线：关闭 gating
)

model.train(
    epoch_num=200,
    batch_size=128,
    lr=1e-3,
    adaptlr=False,
    num_warmup=0,
    early_stopping=True,
    valid_prop=0.1,
    patience=10,
    weighted=False,
)

print("正在推断潜在空间...")
model.inference(n_samples=1, update=True, returns=False)
adata = model.get_adata()

print("正在计算 ARI...")
sc.pp.neighbors(adata, use_rep="latent_shared")
best_ari, best_res, best_key = 0, 0, ""
for res in [0.3, 0.5, 0.8, 1.0, 1.5]:
    key = f"leiden_{res}"
    sc.tl.leiden(adata, resolution=res, key_added=key)
    ari = adjusted_rand_score(adata.obs["celltype"], adata.obs[key])
    n_clusters = adata.obs[key].nunique()
    print(f"  resolution={res}, n_clusters={n_clusters}, ARI={ari:.4f}")
    if ari > best_ari:
        best_ari, best_res, best_key = ari, res, key
print(f"最优 ARI={best_ari:.4f} (resolution={best_res})")

adata.write(SAVE_PATH)
print(f"结果已保存至: {SAVE_PATH}")
