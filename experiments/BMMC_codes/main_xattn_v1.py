from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import scanpy as sc

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scMRDR.module import Integration


def _build_feature_list(adata):
    required = ("rna_hvg", "atac_hvg", "prot_hvg")
    missing = [key for key in required if key not in adata.uns]
    if missing:
        raise KeyError(
            "BMMC xattn script requires adata.uns keys "
            f"{required}, missing: {missing}"
        )

    rna_hvg = np.where(adata.var_names.isin(adata.uns["rna_hvg"]))[0].tolist()
    atac_hvg = np.where(adata.var_names.isin(adata.uns["atac_hvg"]))[0].tolist()
    prot_hvg = np.where(adata.var_names.isin(adata.uns["prot_hvg"]))[0].tolist()
    return {"0": rna_hvg, "1": atac_hvg, "2": prot_hvg}


def parse_args():
    default_input = REPO_ROOT / "experiments" / "BMMC_codes" / "feature_aligned_sampled.h5ad"
    default_output = REPO_ROOT / "experiments" / "BMMC_codes" / "feature_aligned_xattn_v1.h5ad"
    default_tb = REPO_ROOT / "runs" / "bmmc_xattn_v1"

    parser = argparse.ArgumentParser(description="Train scMRDR xattn V1 on BMMC.")
    parser.add_argument("--input", type=Path, default=default_input, help="Input .h5ad file")
    parser.add_argument("--output", type=Path, default=default_output, help="Output .h5ad file")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--num-warmup", type=int, default=0, help="Warmup epochs")
    parser.add_argument("--patience", type=int, default=25, help="Early stopping patience")
    parser.add_argument("--valid-prop", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--tensorboard", action="store_true", help="Enable TensorBoard logging")
    parser.add_argument("--tensorboard-dir", type=Path, default=default_tb, help="TensorBoard log dir")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading {args.input}")
    adata = sc.read_h5ad(args.input)
    feature_list = _build_feature_list(adata)

    model = Integration(
        data=adata,
        layer="counts",
        modality_key="modality",
        batch_key="batch",
        feature_list=feature_list,
        distribution="ZINB",
    )

    model.setup(
        model_architecture="xattn",
        hidden_layers=[128, 128],
        dropout_rate=0.2,
        beta=5,
        lambda_adv=10,
        num_shared_tokens=4,
        num_private_tokens=2,
        token_dim=32,
        num_shared_protos=2,
        num_private_protos=2,
        xattn_depth=2,
        xattn_heads=4,
        xattn_dim_head=32,
        diff_hidden_dim=256,
        diff_steps=100,
        lambda_diff_recon=1.0,
        lambda_token_orth=0.0,
        lambda_private_cls=0.03,
        adv_stop_frac=0.25,
        support_ema_eta=0.9,
        support_threshold=0.15,
        support_min_updates=5,
    )

    tb_dir = str(args.tensorboard_dir) if args.tensorboard else "./"
    model.train(
        epoch_num=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        adaptlr=False,
        num_warmup=args.num_warmup,
        early_stopping=True,
        valid_prop=args.valid_prop,
        patience=args.patience,
        weighted=False,
        tensorboard=args.tensorboard,
        savepath=tb_dir,
        random_state=args.seed,
    )

    model.inference(n_samples=1, update=True, returns=False)
    out = model.get_adata()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.write(args.output)
    print(f"Saved trained AnnData to {args.output}")


if __name__ == "__main__":
    main()
