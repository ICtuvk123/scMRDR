"""
scMRDR 结果评估脚本
===================
计算以下指标：
  - ARI  (Adjusted Rand Index)        聚类与真实细胞类型的一致性
  - NMI  (Normalized Mutual Info)     同上，另一种衡量方式
  - ASW  (Average Silhouette Width)   细胞类型在潜在空间的分离程度
  - Batch ASW                         批次效应残留（越低越好）

用法：
    python eval_metrics.py
    python eval_metrics.py --h5ad path/to/file.h5ad
"""

import sys
import os
import argparse
import numpy as np
import scanpy as sc
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    silhouette_samples,
)
from sklearn.preprocessing import LabelEncoder

# ──────────────────────────────────────────────
# 参数
# ──────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument(
    "--h5ad",
    default=os.path.join(os.path.dirname(__file__), "feature_aligned_trained_gating.h5ad"),
    help="待评估的 h5ad 文件路径",
)
parser.add_argument("--use_rep", default="latent_shared", help="用于计算的潜在空间键名")
parser.add_argument("--celltype_key", default="celltype", help="细胞类型列名")
parser.add_argument("--batch_key", default="batch", help="批次列名")
parser.add_argument("--modality_key", default="modality", help="模态列名")
args = parser.parse_args()

# ──────────────────────────────────────────────
# 1. 加载数据
# ──────────────────────────────────────────────
print(f"加载文件: {args.h5ad}")
adata = sc.read_h5ad(args.h5ad)
print(f"  {adata.shape[0]} 细胞 × {adata.shape[1]} 特征")
print(f"  潜在空间维度: {adata.obsm[args.use_rep].shape[1]}")

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

Z = adata.obsm["latent_shared"]
true_labels = adata.obs["celltype"]
n_clusters = true_labels.nunique()

kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
pred = kmeans.fit_predict(Z)

ari = adjusted_rand_score(true_labels, pred)
nmi = normalized_mutual_info_score(true_labels, pred, average_method="arithmetic")
print(f"KMeans ARI: {ari:.4f}")
print(f"KMeans NMI: {nmi:.4f}")