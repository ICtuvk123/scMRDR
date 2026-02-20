"""
scIB 完整评估脚本（修正版）
============================
Bio conservation:
    Isolated Labels, KMeans NMI, KMeans ARI, Silhouette label, cLISI
Batch correction:
    Silhouette batch, iLISI, kBET, Graph connectivity, PCR comparison
Modality integration:
    Silhouette modality, iLISI modality, kBET modality,
    Graph connectivity modality, PCR comparison modality
Total = mean(Bio, Batch, Modality)
"""

import argparse, os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scanpy as sc
import scib
from sklearn.cluster import KMeans
from sklearn.metrics import (adjusted_rand_score,
                              normalized_mutual_info_score)

# ──────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--h5ad", default=os.path.join(os.path.dirname(__file__),
                    "feature_aligned_trained_gating.h5ad"))
parser.add_argument("--use_rep",      default="latent_shared")
parser.add_argument("--celltype_key", default="celltype")
parser.add_argument("--batch_key",    default="batch")
parser.add_argument("--modality_key", default="modality")
args = parser.parse_args()

# ──────────────────────────────────────────────
# 工具函数
def try_metric(fn, name, *a, **kw):
    try:
        v = fn(*a, **kw)
        print(f"    {name}: {v:.4f}")
        return float(v)
    except Exception as e:
        print(f"    {name}: NaN  ({e})")
        return np.nan

def safe(x):
    return x if (isinstance(x, float) and not np.isnan(x)) else 0.0

# ──────────────────────────────────────────────
# 1. 加载 & 建图
print(f"加载: {args.h5ad}")
adata = sc.read_h5ad(args.h5ad)
adata.obs_names_make_unique()
print(f"  {adata.shape[0]} 细胞 × {adata.shape[1]} 特征")
adata.obsm["X_emb"] = adata.obsm[args.use_rep]
sc.pp.neighbors(adata, use_rep="X_emb")

Z          = adata.obsm["X_emb"]
true_labels = adata.obs[args.celltype_key]
n_ct       = true_labels.nunique()

# ──────────────────────────────────────────────
# 2. Bio conservation
print("\n── Bio conservation ──")

# Isolated Labels
print("  Isolated Labels...")
iso = try_metric(scib.me.isolated_labels,
                 "Isolated Labels",
                 adata,
                 label_key=args.celltype_key,
                 batch_key=args.batch_key,
                 embed="X_emb",
                 cluster=True,
                 iso_threshold=None,
                 verbose=False)

# KMeans NMI & ARI
print("  KMeans NMI / ARI...")
kmeans = KMeans(n_clusters=n_ct, random_state=42, n_init=10)
pred   = kmeans.fit_predict(Z)
km_nmi = normalized_mutual_info_score(true_labels, pred, average_method="arithmetic")
km_ari = adjusted_rand_score(true_labels, pred)
# scIB 将 ARI 映射到 [0,1]
km_ari_scaled = (km_ari + 1) / 2
print(f"    KMeans NMI: {km_nmi:.4f}")
print(f"    KMeans ARI: {km_ari:.4f}  (scaled: {km_ari_scaled:.4f})")

# Silhouette label
sil_label = try_metric(scib.me.silhouette,
                        "Silhouette label",
                        adata,
                        label_key=args.celltype_key,
                        embed="X_emb")

# cLISI
clisi = try_metric(scib.me.clisi_graph,
                   "cLISI",
                   adata,
                   label_key=args.celltype_key,
                   type_="embed",
                   use_rep="X_emb",
                   verbose=False)

bio = np.nanmean([safe(iso), km_nmi, km_ari_scaled, safe(sil_label), safe(clisi)])

# ──────────────────────────────────────────────
# 3. Batch correction
print("\n── Batch correction ──")

sil_batch = try_metric(scib.me.silhouette_batch,
                        "Silhouette batch",
                        adata,
                        batch_key=args.batch_key,
                        label_key=args.celltype_key,
                        embed="X_emb",
                        verbose=False)

ilisi = try_metric(scib.me.ilisi_graph,
                   "iLISI",
                   adata,
                   batch_key=args.batch_key,
                   type_="embed",
                   use_rep="X_emb",
                   verbose=False)

kbet = try_metric(scib.me.kBET,
                  "kBET",
                  adata,
                  batch_key=args.batch_key,
                  label_key=args.celltype_key,
                  type_="embed",
                  embed="X_emb",
                  verbose=False)

graph_conn = try_metric(scib.me.graph_connectivity,
                         "Graph connectivity",
                         adata,
                         label_key=args.celltype_key)

print("  PCR comparison...")
try:
    adata_pre = sc.read_h5ad(args.h5ad)
    adata_pre.obs_names_make_unique()
    adata_pre.obsm["X_emb"] = adata_pre.obsm[args.use_rep]
    pcr = scib.me.pcr_comparison(adata_pre, adata,
                                  covariate=args.batch_key,
                                  embed="X_emb",
                                  n_comps=50,
                                  verbose=False)
    print(f"    PCR comparison: {pcr:.4f}")
except Exception as e:
    print(f"    PCR comparison: NaN  ({e})")
    pcr = np.nan

batch = np.nanmean([safe(sil_batch), safe(ilisi), safe(kbet),
                    safe(graph_conn), safe(pcr)])

# ──────────────────────────────────────────────
# 4. Modality integration
print("\n── Modality integration ──")

sil_mod = try_metric(scib.me.silhouette_batch,
                      "Silhouette modality",
                      adata,
                      batch_key=args.modality_key,
                      label_key=args.celltype_key,
                      embed="X_emb",
                      verbose=False)

ilisi_mod = try_metric(scib.me.ilisi_graph,
                        "iLISI modality",
                        adata,
                        batch_key=args.modality_key,
                        type_="embed",
                        use_rep="X_emb",
                        verbose=False)

kbet_mod = try_metric(scib.me.kBET,
                       "kBET modality",
                       adata,
                       batch_key=args.modality_key,
                       label_key=args.celltype_key,
                       type_="embed",
                       embed="X_emb",
                       verbose=False)

graph_conn_mod = try_metric(scib.me.graph_connectivity,
                             "Graph connectivity modality",
                             adata,
                             label_key=args.modality_key)

print("  PCR comparison modality...")
try:
    pcr_mod = scib.me.pcr_comparison(adata_pre, adata,
                                      covariate=args.modality_key,
                                      embed="X_emb",
                                      n_comps=50,
                                      verbose=False)
    print(f"    PCR comparison modality: {pcr_mod:.4f}")
except Exception as e:
    print(f"    PCR comparison modality: NaN  ({e})")
    pcr_mod = np.nan

modality = np.nanmean([safe(sil_mod), safe(ilisi_mod), safe(kbet_mod),
                        safe(graph_conn_mod), safe(pcr_mod)])

# ──────────────────────────────────────────────
# 5. 汇总
total = np.nanmean([bio, batch, modality])

print("\n" + "=" * 52)
print(f"  {'指标':<32} {'得分':>8}")
print("=" * 52)
print(f"  {'── Bio conservation ──':<32}")
print(f"  {'  Isolated Labels':<32} {iso:>8.4f}")
print(f"  {'  KMeans NMI':<32} {km_nmi:>8.4f}")
print(f"  {'  KMeans ARI (scaled)':<32} {km_ari_scaled:>8.4f}")
print(f"  {'  Silhouette label':<32} {sil_label:>8.4f}")
print(f"  {'  cLISI':<32} {'NaN':>8}" if np.isnan(clisi) else
      f"  {'  cLISI':<32} {clisi:>8.4f}")
print(f"  {'── Batch correction ──':<32}")
print(f"  {'  Silhouette batch':<32} {sil_batch:>8.4f}")
print(f"  {'  iLISI':<32} {'NaN':>8}" if np.isnan(ilisi) else
      f"  {'  iLISI':<32} {ilisi:>8.4f}")
print(f"  {'  kBET':<32} {'NaN':>8}" if np.isnan(kbet) else
      f"  {'  kBET':<32} {kbet:>8.4f}")
print(f"  {'  Graph connectivity':<32} {graph_conn:>8.4f}")
print(f"  {'  PCR comparison':<32} {'NaN':>8}" if np.isnan(pcr) else
      f"  {'  PCR comparison':<32} {pcr:>8.4f}")
print(f"  {'── Modality integration ──':<32}")
print(f"  {'  Silhouette modality':<32} {sil_mod:>8.4f}")
print(f"  {'  iLISI modality':<32} {'NaN':>8}" if np.isnan(ilisi_mod) else
      f"  {'  iLISI modality':<32} {ilisi_mod:>8.4f}")
print(f"  {'  kBET modality':<32} {'NaN':>8}" if np.isnan(kbet_mod) else
      f"  {'  kBET modality':<32} {kbet_mod:>8.4f}")
print(f"  {'  Graph connectivity modality':<32} {graph_conn_mod:>8.4f}")
print(f"  {'  PCR comparison modality':<32} {'NaN':>8}" if np.isnan(pcr_mod) else
      f"  {'  PCR comparison modality':<32} {pcr_mod:>8.4f}")
print("=" * 52)
print(f"  {'Bio conservation':<32} {bio:>8.4f}")
print(f"  {'Batch correction':<32} {batch:>8.4f}")
print(f"  {'Modality integration':<32} {modality:>8.4f}")
print(f"  {'Total':<32} {total:>8.4f}")
print("=" * 52)

# 保存 CSV
rows = {
    "Isolated Labels": iso, "KMeans NMI": km_nmi,
    "KMeans ARI": km_ari, "KMeans ARI scaled": km_ari_scaled,
    "Silhouette label": sil_label, "cLISI": clisi,
    "Silhouette batch": sil_batch, "iLISI": ilisi,
    "kBET": kbet, "Graph connectivity": graph_conn, "PCR": pcr,
    "Silhouette modality": sil_mod, "iLISI modality": ilisi_mod,
    "kBET modality": kbet_mod,
    "Graph connectivity modality": graph_conn_mod, "PCR modality": pcr_mod,
    "Bio conservation": bio, "Batch correction": batch,
    "Modality integration": modality, "Total": total,
}
csv_path = args.h5ad.replace(".h5ad", "_scib_metrics.csv")
pd.DataFrame([rows]).to_csv(csv_path, index=False)
print(f"\n结果已保存至: {csv_path}")
