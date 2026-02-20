"""
带 Recon-Gated Adversarial Training 的 scMRDR 训练脚本
================================================
Gating 机制：ReconGating 基于 per-sample 重建损失来调节对抗权重：
  - 高 recon loss → 噪声/离群样本 → 降低对抗权重
  - 低 recon loss → 干净样本 → 保持对抗权重
  - 所有样本至少保留 w_floor 的对抗压力（防止 iLISI 崩溃）

用法：
    python main_gating.py
"""

import sys
import os

# 将本地 src 目录插入路径（开发模式，不需要 pip install）
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

import scanpy as sc
from scMRDR.module import Integration
from sklearn.metrics import adjusted_rand_score

# ──────────────────────────────────────────────
# 1. 加载数据
# ──────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "feature_aligned_sampled.h5ad")
SAVE_PATH = os.path.join(os.path.dirname(__file__), "feature_aligned_trained_gating.h5ad")

print("正在加载数据...")
adata = sc.read_h5ad(DATA_PATH)
print(f"数据加载完成: {adata.shape[0]} 细胞 × {adata.shape[1]} 特征")

# ──────────────────────────────────────────────
# 2. 初始化 Integration 模块
# ──────────────────────────────────────────────
model = Integration(
    data=adata,
    layer="counts",
    modality_key="modality",
    batch_key="batch",
    distribution="ZINB",
    feature_list=None,    # 特征已对齐，不需要掩码
)

# ──────────────────────────────────────────────
# 3. 配置模型（开启 recon gating）
# ──────────────────────────────────────────────
model.setup(
    # 网络结构（与原始配比一致）
    hidden_layers=[512, 512],
    latent_dim_shared=20,
    latent_dim_specific=20,
    dropout_rate=0.2,

    # 损失权重
    beta=2,
    gamma=5,
    lambda_adv=5,

    # ── Recon Gating 参数 ──────────
    confidence_weighted=True,

    cw_w_floor=0.3,             # 最低对抗权重（防止 iLISI 崩溃）
    cw_w_cap=1.0,               # 最高对抗权重
    cw_tau=1.0,                 # sigmoid 温度
    cw_ema_decay=0.99,          # EMA 衰减系数
    cw_stats_warmup_steps=50,   # 统计量收集步数（期间权重全 1）
)

# ──────────────────────────────────────────────
# 4. 训练
# ──────────────────────────────────────────────
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

    # Lambda ramp：从第一个 epoch 就全力对抗
    cw_adv_ramp_epochs=0,

    # 若为 None，则使用 setup 里设置的 lambda_adv=5
    cw_lambda_target=None,
)

# ──────────────────────────────────────────────
# 6. 推断潜在嵌入
# ──────────────────────────────────────────────
print("正在推断潜在空间...")
model.inference(n_samples=1, update=True, returns=False)
adata = model.get_adata()   # latent_shared / latent_specific 存入 adata.obsm

# ──────────────────────────────────────────────
# 7. ARI 评估
# ──────────────────────────────────────────────
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

# ──────────────────────────────────────────────
# 8. 可视化
# ──────────────────────────────────────────────
print("正在计算 UMAP...")
sc.tl.umap(adata)
sc.pl.umap(
    adata,
    color=["modality", "celltype", "batch", best_key],
    size=2,
    wspace=0.5,
)

# ──────────────────────────────────────────────
# 8. 保存结果
# ──────────────────────────────────────────────
adata.write(SAVE_PATH)
print(f"结果已保存至: {SAVE_PATH}")
