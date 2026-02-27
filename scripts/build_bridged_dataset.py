#!/usr/bin/env python3
import argparse
import itertools
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


def _normalize_feature_names(names, uppercase: bool) -> np.ndarray:
    values = pd.Index(names).astype(str).str.strip()
    if uppercase:
        values = values.str.upper()
    return values.to_numpy()


def _get_matrix(adata: ad.AnnData, layer: str | None):
    if layer is None:
        return adata.X
    if layer not in adata.layers:
        raise KeyError(f"Layer '{layer}' not found.")
    return adata.layers[layer]


def _to_csr(matrix):
    if sparse.issparse(matrix):
        return matrix.tocsr()
    return sparse.csr_matrix(np.asarray(matrix))


def _choose_mapping_columns(df: pd.DataFrame, protein_col: str | None, gene_col: str | None) -> tuple[str, str]:
    def _find_col(candidates: list[str]) -> str | None:
        lookup = {c.lower(): c for c in df.columns}
        for cand in candidates:
            if cand in lookup:
                return lookup[cand]
        return None

    if protein_col is None:
        protein_col = _find_col(["protein", "protein_name", "adt", "feature", "marker"])
    if gene_col is None:
        gene_col = _find_col(["gene", "gene_name", "symbol", "rna"])
    if protein_col is None or gene_col is None:
        raise ValueError(
            "Cannot infer protein/gene columns from mapping table. "
            "Please provide --protein-col and --gene-col."
        )
    return protein_col, gene_col


def _prepare_gene_modality(
    adata: ad.AnnData,
    label: str,
    modality_key: str,
    layer: str | None,
    uppercase_genes: bool,
) -> ad.AnnData:
    X = _get_matrix(adata, layer)
    var_names = _normalize_feature_names(adata.var_names, uppercase=uppercase_genes)
    out = ad.AnnData(X=_to_csr(X), obs=adata.obs.copy(), var=pd.DataFrame(index=var_names))
    out.var_names_make_unique()
    out.obs_names_make_unique()
    out.obs[modality_key] = label
    out.var["feature_group"] = "gene"
    return out


def _prepare_protein_modality(
    adata: ad.AnnData,
    label: str,
    modality_key: str,
    layer: str | None,
    mapping: pd.DataFrame | None,
    protein_col: str | None,
    gene_col: str | None,
    uppercase_genes: bool,
    uppercase_protein: bool,
    keep_native_protein: bool,
    protein_prefix: str,
) -> tuple[ad.AnnData, dict[str, int]]:
    X = _to_csr(_get_matrix(adata, layer))
    protein_names = _normalize_feature_names(adata.var_names, uppercase=uppercase_protein)
    if mapping is None:
        # Fallback: exact-name mapping (protein name -> same gene symbol).
        map_df = pd.DataFrame({"_protein": protein_names, "_gene": protein_names})
        protein_col, gene_col = "_protein", "_gene"
    else:
        assert protein_col is not None and gene_col is not None
        map_df = mapping[[protein_col, gene_col]].copy()
        map_df[protein_col] = _normalize_feature_names(map_df[protein_col], uppercase=uppercase_protein)
        map_df[gene_col] = _normalize_feature_names(map_df[gene_col], uppercase=uppercase_genes)
    map_df = map_df.dropna().drop_duplicates()
    protein_to_gene = dict(zip(map_df[protein_col], map_df[gene_col]))

    mapped_pairs: list[tuple[int, str]] = []
    for idx, prot in enumerate(protein_names):
        gene = protein_to_gene.get(prot)
        if gene is not None and gene != "":
            mapped_pairs.append((idx, gene))

    if len(mapped_pairs) == 0 and not keep_native_protein:
        raise ValueError("No mapped protein->gene pairs found and keep_native_protein=False.")

    if len(mapped_pairs) > 0:
        mapped_idx = np.array([p[0] for p in mapped_pairs], dtype=np.int64)
        mapped_genes = [p[1] for p in mapped_pairs]
        unique_genes = sorted(set(mapped_genes))
        gene_to_col = {g: i for i, g in enumerate(unique_genes)}
        assign_cols = np.array([gene_to_col[g] for g in mapped_genes], dtype=np.int64)
        assign = sparse.csr_matrix(
            (np.ones(len(assign_cols), dtype=np.float32), (np.arange(len(assign_cols)), assign_cols)),
            shape=(len(assign_cols), len(unique_genes)),
        )
        X_gene = X[:, mapped_idx] @ assign
        bridge_var_names = list(unique_genes)
        bridge_feature_group = ["gene_linked"] * len(unique_genes)
    else:
        X_gene = sparse.csr_matrix((X.shape[0], 0), dtype=np.float32)
        bridge_var_names = []
        bridge_feature_group = []

    if keep_native_protein:
        native_names = [f"{protein_prefix}{name}" for name in protein_names]
        X_bridge = sparse.hstack([X_gene, X], format="csr")
        bridge_var_names.extend(native_names)
        bridge_feature_group.extend(["protein_native"] * len(native_names))
    else:
        X_bridge = X_gene.tocsr()

    out = ad.AnnData(X=X_bridge, obs=adata.obs.copy(), var=pd.DataFrame(index=bridge_var_names))
    out.var["feature_group"] = bridge_feature_group
    out.obs[modality_key] = label
    out.var_names_make_unique()
    out.obs_names_make_unique()

    stats = {
        "protein_features_total": int(len(protein_names)),
        "protein_features_mapped": int(len(mapped_pairs)),
        "protein_unique_mapped_genes": int(X_gene.shape[1]),
        "protein_native_features_kept": int(X.shape[1] if keep_native_protein else 0),
    }
    return out, stats


def _extract_modality_from_combined(
    adata: ad.AnnData,
    modality_key: str,
    modality_label: str,
    feature_uns_key: str | None,
) -> ad.AnnData:
    if modality_key not in adata.obs:
        raise KeyError(f"obs key not found: {modality_key}")
    subset = adata[adata.obs[modality_key].astype(str) == str(modality_label)].copy()
    if subset.n_obs == 0:
        raise ValueError(f"No cells found for modality label '{modality_label}'.")

    if feature_uns_key is not None and feature_uns_key in adata.uns:
        feature_names = pd.Index(adata.uns[feature_uns_key]).astype(str)
        keep = subset.var_names.isin(feature_names)
        subset = subset[:, keep].copy()
    else:
        X = subset.X
        if sparse.issparse(X):
            keep = np.asarray(X.getnnz(axis=0)).ravel() > 0
        else:
            keep = np.asarray(X).sum(axis=0) != 0
        subset = subset[:, keep].copy()
    return subset


def _compute_linked_features(feature_sets: dict[str, set[str]]) -> tuple[list[str], dict[str, int]]:
    counts: dict[str, int] = {}
    for values in feature_sets.values():
        for feat in values:
            counts[feat] = counts.get(feat, 0) + 1
    linked = sorted([f for f, c in counts.items() if c >= 2])
    return linked, counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a bridged multi-omics dataset for raw-space MNN anchors. "
            "ATAC input should be gene-activity features (pseudo-RNA space)."
        )
    )
    parser.add_argument(
        "--input-h5ad",
        type=Path,
        default=None,
        help=(
            "Optional combined h5ad (all modalities in one file). "
            "If provided, --rna-h5ad/--atac-h5ad can be omitted."
        ),
    )
    parser.add_argument("--rna-h5ad", type=Path, required=False)
    parser.add_argument("--atac-h5ad", type=Path, required=False, help="ATAC gene-activity h5ad.")
    parser.add_argument("--protein-h5ad", type=Path, default=None)
    parser.add_argument("--protein-map-tsv", type=Path, default=None)
    parser.add_argument("--protein-col", type=str, default=None)
    parser.add_argument("--gene-col", type=str, default=None)
    parser.add_argument("--output-h5ad", type=Path, required=True)

    parser.add_argument("--rna-layer", type=str, default=None)
    parser.add_argument("--atac-layer", type=str, default=None)
    parser.add_argument("--protein-layer", type=str, default=None)

    parser.add_argument("--modality-key", type=str, default="modality")
    parser.add_argument("--rna-label", type=str, default="rna")
    parser.add_argument("--atac-label", type=str, default="atac")
    parser.add_argument("--protein-label", type=str, default="protein")
    parser.add_argument("--rna-feature-uns-key", type=str, default="rna_hvg")
    parser.add_argument("--atac-feature-uns-key", type=str, default="atac_hvg")
    parser.add_argument("--protein-feature-uns-key", type=str, default="prot_hvg")

    parser.add_argument("--uppercase-genes", action="store_true")
    parser.add_argument("--uppercase-protein", action="store_true")
    parser.add_argument("--keep-native-protein", action="store_true")
    parser.add_argument("--protein-prefix", type=str, default="PROT__")
    parser.add_argument("--linked-uns-key", type=str, default="linked_features")
    args = parser.parse_args()

    # Allow two modes:
    # 1) split from a combined input file
    # 2) load separate modality files
    if args.input_h5ad is None:
        if args.rna_h5ad is None or args.atac_h5ad is None:
            raise ValueError(
                "Either provide --input-h5ad, or provide both --rna-h5ad and --atac-h5ad."
            )
        for p in [args.rna_h5ad, args.atac_h5ad]:
            if not p.exists():
                raise FileNotFoundError(p)
    else:
        if not args.input_h5ad.exists():
            raise FileNotFoundError(args.input_h5ad)
    if args.protein_h5ad is not None and args.input_h5ad is None and not args.protein_h5ad.exists():
        raise FileNotFoundError(args.protein_h5ad)
    if args.protein_map_tsv is not None and not args.protein_map_tsv.exists():
        raise FileNotFoundError(args.protein_map_tsv)

    if args.input_h5ad is not None:
        print(f"Loading combined input: {args.input_h5ad}")
        combined_in = ad.read_h5ad(args.input_h5ad)
        rna = _extract_modality_from_combined(
            combined_in, args.modality_key, args.rna_label, args.rna_feature_uns_key
        )
        atac = _extract_modality_from_combined(
            combined_in, args.modality_key, args.atac_label, args.atac_feature_uns_key
        )
        protein = None
        if args.protein_label in combined_in.obs[args.modality_key].astype(str).unique().tolist():
            protein = _extract_modality_from_combined(
                combined_in, args.modality_key, args.protein_label, args.protein_feature_uns_key
            )
    else:
        print(f"Loading RNA: {args.rna_h5ad}")
        rna = ad.read_h5ad(args.rna_h5ad)
        print(f"Loading ATAC(gene-activity): {args.atac_h5ad}")
        atac = ad.read_h5ad(args.atac_h5ad)
        protein = ad.read_h5ad(args.protein_h5ad) if args.protein_h5ad is not None else None

    rna_p = _prepare_gene_modality(
        rna, args.rna_label, args.modality_key, args.rna_layer, args.uppercase_genes
    )
    atac_p = _prepare_gene_modality(
        atac, args.atac_label, args.modality_key, args.atac_layer, args.uppercase_genes
    )

    prepared = [(args.rna_label, rna_p), (args.atac_label, atac_p)]
    summary = {
        args.rna_label: int(rna_p.n_vars),
        args.atac_label: int(atac_p.n_vars),
    }
    protein_stats = {}
    if protein is not None:
        mapping = None
        protein_col = None
        gene_col = None
        if args.protein_map_tsv is not None:
            mapping = pd.read_csv(args.protein_map_tsv, sep=None, engine="python")
            protein_col, gene_col = _choose_mapping_columns(mapping, args.protein_col, args.gene_col)
        protein_p, protein_stats = _prepare_protein_modality(
            protein,
            args.protein_label,
            args.modality_key,
            args.protein_layer,
            mapping,
            protein_col,
            gene_col,
            args.uppercase_genes,
            args.uppercase_protein,
            args.keep_native_protein,
            args.protein_prefix,
        )
        prepared.append((args.protein_label, protein_p))
        summary[args.protein_label] = int(protein_p.n_vars)

    labels = [x[0] for x in prepared]
    adatas = [x[1] for x in prepared]
    combined = ad.concat(adatas, join="outer", fill_value=0, merge="same")
    combined.obs_names_make_unique()
    combined.var_names_make_unique()

    feature_sets = {label: set(a.var_names.astype(str).tolist()) for label, a in prepared}
    linked_features, _ = _compute_linked_features(feature_sets)

    combined.uns["rna_hvg"] = list(rna_p.var_names.astype(str))
    combined.uns["atac_hvg"] = list(atac_p.var_names.astype(str))
    if protein is not None:
        combined.uns["prot_hvg"] = list(dict(prepared)[args.protein_label].var_names.astype(str))
    combined.uns[args.linked_uns_key] = linked_features
    combined.uns["feature_set_sizes"] = summary

    pairwise_sizes: dict[str, int] = {}
    for m1, m2 in itertools.combinations(labels, 2):
        key = f"{m1}__{m2}"
        pairwise_sizes[key] = len(feature_sets[m1].intersection(feature_sets[m2]))
    combined.uns["pairwise_linked_feature_sizes"] = pairwise_sizes
    combined.uns["linked_feature_counts_ge2"] = int(len(linked_features))
    if protein_stats:
        combined.uns["protein_bridge_stats"] = protein_stats

    args.output_h5ad.parent.mkdir(parents=True, exist_ok=True)
    combined.write_h5ad(args.output_h5ad)

    print("Build finished.")
    print(f"  Cells: {combined.n_obs}, Features: {combined.n_vars}")
    print(f"  Modality feature sizes: {summary}")
    print(f"  Linked features (>=2 modalities): {len(linked_features)}")
    print(f"  Pairwise linked sizes: {pairwise_sizes}")
    if protein_stats:
        print(f"  Protein bridge stats: {protein_stats}")
    print(f"Saved: {args.output_h5ad}")


if __name__ == "__main__":
    main()
