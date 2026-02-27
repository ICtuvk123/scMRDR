import anndata as ad
a = ad.read_h5ad("experiments/BMMC_codes/feature_aligned_sampled.h5ad")
print("obs columns:", list(a.obs.columns))
print("modality values:\n", a.obs["modality"].astype(str).value_counts())
print("uns keys:", list(a.uns.keys())[:20])