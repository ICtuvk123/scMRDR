"""
用论文原始管线（Benchmarker2）对比 baseline vs gating
将 metrics.py 中的核心类内联，避免硬编码路径问题
"""
import os
os.environ['JAX_PLATFORM_NAME'] = 'cpu'

import warnings
warnings.filterwarnings("ignore")

import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from enum import Enum
from functools import partial
from typing import Any

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib as mpl
import matplotlib.pyplot as plt
from anndata import AnnData
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

import scib_metrics
from scib_metrics.nearest_neighbors import NeighborsResults, pynndescent

# ──────────────────────────────────────────────
# 从 metrics.py 内联的核心代码
# ──────────────────────────────────────────────
Kwargs = dict[str, Any]
MetricType = bool | Kwargs

_LABELS = "labels"
_BATCH = "batch"
_MODALITY = "modality"
_X_PRE = "X_pre"
_METRIC_TYPE = "Metric Type"
_AGGREGATE_SCORE = "Aggregate score"
_METRIC_NAME = "Metric Name"

@dataclass(frozen=True)
class BioConservation2:
    isolated_labels: MetricType = True
    nmi_ari_cluster_labels_leiden: MetricType = False
    nmi_ari_cluster_labels_kmeans: MetricType = True
    silhouette_label: MetricType = True
    clisi_knn: MetricType = True

@dataclass(frozen=True)
class BatchCorrection2:
    silhouette_batch_b: MetricType = True
    ilisi_knn_b: MetricType = True
    kbet_per_label_b: MetricType = True
    pcr_comparison_b: MetricType = True

@dataclass(frozen=True)
class ModalityIntegration2:
    silhouette_batch_m: MetricType = True
    ilisi_knn_m: MetricType = True
    kbet_per_label_m: MetricType = True
    graph_connectivity: MetricType = True
    pcr_comparison_m: MetricType = True

metric_name_cleaner2 = {
    "silhouette_label": "Silhouette label",
    "silhouette_batch_b": "Silhouette batch",
    "silhouette_batch_m": "Silhouette modality",
    "isolated_labels": "Isolated labels",
    "nmi_ari_cluster_labels_kmeans_nmi": "KMeans NMI",
    "nmi_ari_cluster_labels_kmeans_ari": "KMeans ARI",
    "clisi_knn": "cLISI",
    "ilisi_knn_b": "iLISI",
    "ilisi_knn_m": "iLISI modality",
    "kbet_per_label_b": "KBET",
    "kbet_per_label_m": "KBET modality",
    "graph_connectivity": "Graph connectivity",
    "pcr_comparison_b": "PCR comparison",
    "pcr_comparison_m": "PCR comparison modality",
}

class MetricAnnDataAPI2(Enum):
    isolated_labels = lambda ad, fn: fn(ad.X, ad.obs[_LABELS], ad.obs[_MODALITY])
    nmi_ari_cluster_labels_leiden = lambda ad, fn: fn(ad.uns["15_neighbor_res"], ad.obs[_LABELS])
    nmi_ari_cluster_labels_kmeans = lambda ad, fn: fn(ad.X, ad.obs[_LABELS])
    silhouette_label = lambda ad, fn: fn(ad.X, ad.obs[_LABELS])
    clisi_knn = lambda ad, fn: fn(ad.uns["90_neighbor_res"], ad.obs[_LABELS])
    silhouette_batch_b = lambda ad, fn: fn(ad.X, ad.obs[_LABELS], ad.obs[_BATCH])
    pcr_comparison_b = lambda ad, fn: fn(ad.obsm[_X_PRE], ad.X, ad.obs[_BATCH], categorical=True)
    ilisi_knn_b = lambda ad, fn: fn(ad.uns["90_neighbor_res"], ad.obs[_BATCH])
    kbet_per_label_b = lambda ad, fn: fn(ad.uns["50_neighbor_res"], ad.obs[_BATCH], ad.obs[_LABELS])
    graph_connectivity = lambda ad, fn: fn(ad.uns["15_neighbor_res"], ad.obs[_LABELS])
    silhouette_batch_m = lambda ad, fn: fn(ad.X, ad.obs[_LABELS], ad.obs[_MODALITY])
    pcr_comparison_m = lambda ad, fn: fn(ad.obsm[_X_PRE], ad.X, ad.obs[_MODALITY], categorical=True)
    ilisi_knn_m = lambda ad, fn: fn(ad.uns["90_neighbor_res"], ad.obs[_MODALITY])
    kbet_per_label_m = lambda ad, fn: fn(ad.uns["50_neighbor_res"], ad.obs[_MODALITY], ad.obs[_LABELS])

class Benchmarker2:
    def __init__(self, adata, batch_key, label_key, modality_key,
                 embedding_obsm_keys, bio_conservation_metrics,
                 batch_correction_metrics, modality_integration_metrics,
                 pre_integrated_embedding_obsm_key=None, n_jobs=1, progress_bar=True):
        self._adata = adata
        self._embedding_obsm_keys = embedding_obsm_keys
        self._pre_integrated_embedding_obsm_key = pre_integrated_embedding_obsm_key
        self._bio_conservation_metrics = bio_conservation_metrics
        self._batch_correction_metrics = batch_correction_metrics
        self._modality_integration_metrics = modality_integration_metrics
        self._results = pd.DataFrame(columns=list(self._embedding_obsm_keys) + [_METRIC_TYPE])
        self._emb_adatas = {}
        self._neighbor_values = (15, 50, 90)
        self._prepared = False
        self._benchmarked = False
        self._batch_key = batch_key
        self._modality_key = modality_key
        self._label_key = label_key
        self._n_jobs = n_jobs
        self._progress_bar = progress_bar
        self._metric_collection_dict = {}
        if self._bio_conservation_metrics is not None:
            self._metric_collection_dict["Bio conservation"] = self._bio_conservation_metrics
        if self._batch_correction_metrics is not None:
            self._metric_collection_dict["Batch correction"] = self._batch_correction_metrics
        if self._modality_integration_metrics is not None:
            self._metric_collection_dict["Modality integration"] = self._modality_integration_metrics

    def prepare(self, neighbor_computer=None):
        if self._pre_integrated_embedding_obsm_key is None:
            sc.tl.pca(self._adata, use_highly_variable=False)
            self._pre_integrated_embedding_obsm_key = "X_pca"
        for emb_key in self._embedding_obsm_keys:
            self._emb_adatas[emb_key] = AnnData(self._adata.obsm[emb_key], obs=self._adata.obs)
            self._emb_adatas[emb_key].obs[_BATCH] = np.asarray(self._adata.obs[self._batch_key].values)
            self._emb_adatas[emb_key].obs[_MODALITY] = np.asarray(self._adata.obs[self._modality_key].values)
            self._emb_adatas[emb_key].obs[_LABELS] = np.asarray(self._adata.obs[self._label_key].values)
            self._emb_adatas[emb_key].obsm[_X_PRE] = self._adata.obsm[self._pre_integrated_embedding_obsm_key]
        progress = tqdm(self._emb_adatas.values(), desc="Computing neighbors") if self._progress_bar else self._emb_adatas.values()
        for ad in progress:
            neigh_result = pynndescent(ad.X, n_neighbors=max(self._neighbor_values), random_state=0, n_jobs=self._n_jobs)
            for n in self._neighbor_values:
                ad.uns[f"{n}_neighbor_res"] = neigh_result.subset_neighbors(n=n)
        self._prepared = True

    def benchmark(self):
        if not self._prepared:
            self.prepare()
        num_metrics = sum([sum([v is not False for v in asdict(mc)]) for mc in self._metric_collection_dict.values()])
        progress_embs = tqdm(self._emb_adatas.items(), desc="Embeddings", colour="green") if self._progress_bar else self._emb_adatas.items()
        for emb_key, ad in progress_embs:
            pbar = tqdm(total=num_metrics, desc="Metrics", leave=False, colour="blue") if self._progress_bar else None
            for metric_type, metric_collection in self._metric_collection_dict.items():
                for metric_name, use_metric in asdict(metric_collection).items():
                    if use_metric:
                        if pbar: pbar.set_postfix_str(f"{metric_type}: {metric_name}")
                        metric_fn = getattr(scib_metrics, re.sub(r'(_b|_m)$', '', metric_name))
                        if isinstance(use_metric, dict):
                            metric_fn = partial(metric_fn, **use_metric)
                        metric_value = getattr(MetricAnnDataAPI2, metric_name)(ad, metric_fn)
                        if isinstance(metric_value, dict):
                            for k, v in metric_value.items():
                                self._results.loc[f"{metric_type}_{metric_name}_{k}", emb_key] = v
                                self._results.loc[f"{metric_type}_{metric_name}_{k}", _METRIC_TYPE] = metric_type
                                self._results.loc[f"{metric_type}_{metric_name}_{k}", _METRIC_NAME] = f"{metric_name}_{k}"
                        else:
                            self._results.loc[f"{metric_type}_{metric_name}", emb_key] = metric_value
                            self._results.loc[f"{metric_type}_{metric_name}", _METRIC_TYPE] = metric_type
                            self._results.loc[f"{metric_type}_{metric_name}", _METRIC_NAME] = metric_name
                        if pbar: pbar.update(1)
        self._benchmarked = True

    def get_results(self, min_max_scale=True):
        df = self._results.transpose()
        df.index.name = "Embedding"
        df = df.loc[~df.index.isin([_METRIC_TYPE, _METRIC_NAME])]
        arr = MinMaxScaler().fit_transform(df) if min_max_scale else df.to_numpy()
        df = pd.DataFrame(arr, columns=self._results[_METRIC_NAME].values, index=df.index)
        df = df.transpose()
        df[_METRIC_TYPE] = self._results[_METRIC_TYPE].values
        per_class_score = df.groupby(_METRIC_TYPE).mean().transpose()
        per_class_score["Total"] = (
            0.4 * per_class_score["Bio conservation"] +
            0.3 * per_class_score["Batch correction"] +
            0.3 * per_class_score["Modality integration"]
        )
        df[_METRIC_NAME] = self._results[_METRIC_NAME].values
        df = pd.concat([df.transpose(), per_class_score], axis=1)
        df.loc[_METRIC_TYPE, per_class_score.columns] = _AGGREGATE_SCORE
        df.loc[_METRIC_NAME, per_class_score.columns] = per_class_score.columns
        return df

# ──────────────────────────────────────────────
# 主程序
# ──────────────────────────────────────────────
BASELINE_PATH  = os.path.join(os.path.dirname(__file__), "feature_aligned_trained_baseline.h5ad")
GATING_PATH    = os.path.join(os.path.dirname(__file__), "feature_aligned_trained_gating.h5ad")
WMIN04_PATH    = os.path.join(os.path.dirname(__file__), "feature_aligned_trained_gating_wmin04.h5ad")

print("加载数据...")
adata_b  = sc.read_h5ad(BASELINE_PATH);  adata_b.obs_names_make_unique()
adata_g  = sc.read_h5ad(GATING_PATH);    adata_g.obs_names_make_unique()
adata_w4 = sc.read_h5ad(WMIN04_PATH);    adata_w4.obs_names_make_unique()

adata = adata_b.copy()
adata.obsm['Baseline']      = adata_b.obsm['latent_shared'].copy()
adata.obsm['Gating_wmin01'] = adata_g.obsm['latent_shared'].copy()
adata.obsm['Gating_wmin04'] = adata_w4.obsm['latent_shared'].copy()

adata.obs['modality'] = adata.obs['modality'].astype(str)
adata.obs.loc[adata.obs['modality'] == '0', 'modality'] = 'RNA'
adata.obs.loc[adata.obs['modality'] == '1', 'modality'] = 'ATAC'
adata.obs.loc[adata.obs['modality'] == '2', 'modality'] = 'Protein'

print(f"细胞数: {adata.shape[0]}, 模态: {sorted(adata.obs['modality'].unique())}")

bm = Benchmarker2(
    adata,
    batch_key="batch",
    label_key="celltype",
    modality_key="modality",
    bio_conservation_metrics=BioConservation2(),
    batch_correction_metrics=BatchCorrection2(),
    modality_integration_metrics=ModalityIntegration2(),
    embedding_obsm_keys=['Baseline', 'Gating_wmin01', 'Gating_wmin04'],
    n_jobs=4,
)

print("\n开始 benchmark...")
bm.benchmark()

print("\n=== 原始分数（未缩放）===")
df_raw = bm.get_results(min_max_scale=False)
print(df_raw.to_string())

print("\n=== Min-Max 缩放后 ===")
df_scaled = bm.get_results(min_max_scale=True)
print(df_scaled.to_string())

out_dir = os.path.dirname(os.path.abspath(__file__))
df_raw.to_csv(os.path.join(out_dir, "paper_pipeline_unscaled.csv"))
df_scaled.to_csv(os.path.join(out_dir, "paper_pipeline_scaled.csv"))
print(f"\n结果已保存至 {out_dir}/paper_pipeline_*.csv")
