"""
改法 C：难样本反向加权（hard-sample boosting）
s = alpha*(1-s_H) + (1-alpha)*(1-s_nn)
判别器越确定、跨模态距离越远的细胞 → 权重越高 → 对抗压力越大
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

import scanpy as sc
from scMRDR.module import Integration
from sklearn.metrics import adjusted_rand_score

DATA_PATH = os.path.join(os.path.dirname(__file__), "feature_aligned_sampled.h5ad")
SAVE_PATH = os.path.join(os.path.dirname(__file__), "feature_aligned_trained_gating_hardc.h5ad")

print("加载数据...")
adata = sc.read_h5ad(DATA_PATH)
print(f"  {adata.shape[0]} 细胞 × {adata.shape[1]} 特征")

model = Integration(
    data=adata, layer="counts", modality_key="modality",
    batch_key="batch", distribution="ZINB", feature_list=None,
)

model.setup(
    hidden_layers=[512, 512], latent_dim_shared=20, latent_dim_specific=20,
    dropout_rate=0.2, beta=2, gamma=5, lambda_adv=5,
    confidence_weighted=True,
    cw_queue_size=4096, cw_alpha=0.5, cw_c_tau=1.0,
    cw_tau_range=(0.01, 2.0), cw_tau_fallback=0.5,
    cw_eta=0.9, cw_rho=0.5, cw_tau_w=0.1,
    cw_w_min=0.1, cw_min_count=8,
    cw_hard_boost=True,   # 关键：难样本反向加权
)

model.train(
    epoch_num=200, batch_size=128, lr=1e-3, adaptlr=False,
    num_warmup=0, early_stopping=True, valid_prop=0.1,
    patience=10, weighted=False,
    cw_adv_ramp_epochs=0, cw_lambda_target=None,
)

print("推断潜在空间...")
model.inference(n_samples=1, update=True, returns=False)
adata = model.get_adata()

print("计算 ARI...")
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
