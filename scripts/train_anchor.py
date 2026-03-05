#!/usr/bin/env python3
import argparse
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import scanpy as sc
import torch

# Support running this script directly from a src-layout repo checkout.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.is_dir():
    sys.path.insert(0, str(SRC_DIR))

from scMRDR.module import Integration


def parse_hidden_layers(value: str) -> list[int]:
    try:
        return [int(x.strip()) for x in value.split(",") if x.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--hidden-layers must be a comma-separated list of integers, e.g. 512,512"
        ) from exc


def build_feature_list(adata, modality_key: str) -> dict[str, list[int]]:
    modality_labels = sorted(adata.obs[modality_key].astype(str).unique().tolist())
    canonical_uns_keys = ["rna_hvg", "atac_hvg", "prot_hvg"]
    available_uns_keys = [k for k in canonical_uns_keys if k in adata.uns]

    if not available_uns_keys:
        raise ValueError(
            "No HVG keys found in adata.uns. Expected one or more of: "
            "rna_hvg, atac_hvg, prot_hvg."
        )

    if len(available_uns_keys) < len(modality_labels):
        raise ValueError(
            f"Found {len(modality_labels)} modalities {modality_labels}, but only "
            f"{len(available_uns_keys)} uns HVG keys {available_uns_keys}. "
            "Use --disable-feature-list to bypass feature masking."
        )

    modality_to_uns = {
        "rna": "rna_hvg",
        "gex": "rna_hvg",
        "atac": "atac_hvg",
        "protein": "prot_hvg",
        "prot": "prot_hvg",
        "adt": "prot_hvg",
    }

    feature_list: dict[str, list[int]] = {}
    fallback_keys = [k for k in canonical_uns_keys if k in adata.uns]
    for modality in modality_labels:
        modality_lower = modality.lower()
        uns_key = None
        for token, key in modality_to_uns.items():
            if token in modality_lower and key in adata.uns:
                uns_key = key
                break
        if uns_key is None:
            if not fallback_keys:
                raise ValueError(
                    f"No available uns feature key for modality '{modality}'. "
                    "Use --disable-feature-list to bypass feature masking."
                )
            uns_key = fallback_keys.pop(0)

        idx = np.where(adata.var_names.isin(adata.uns[uns_key]))[0].tolist()
        feature_list[modality] = idx
        print(f"feature_list[{modality}] <- {uns_key}: {len(idx)} features")

    return feature_list


def parse_linked_feature_file(path: Path) -> list[Any]:
    values: list[Any] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            item = raw.strip()
            if not item or item.startswith("#"):
                continue
            try:
                values.append(int(item))
            except ValueError:
                values.append(item)
    return values


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def main() -> None:
    parser = argparse.ArgumentParser(description="Train scMRDR with MNN anchor loss.")
    parser.add_argument(
        "--input-h5ad",
        type=Path,
        default=Path("experiments/BMMC_codes/feature_aligned_sampled.h5ad"),
        help="Input h5ad path.",
    )
    parser.add_argument(
        "--output-h5ad",
        type=Path,
        default=None,
        help="Output h5ad path. Default: <input_stem>_trained_anchor.h5ad",
    )
    parser.add_argument(
        "--layer",
        type=str,
        default=None,
        help="Input layer name. Default None means using adata.X.",
    )
    parser.add_argument("--modality-key", type=str, default="modality")
    parser.add_argument("--batch-key", type=str, default="batch")
    parser.add_argument("--distribution", type=str, default="ZINB")
    parser.add_argument(
        "--disable-feature-list",
        action="store_true",
        help="Disable feature masking and pass feature_list=None.",
    )

    parser.add_argument("--hidden-layers", type=parse_hidden_layers, default=[512, 512])
    parser.add_argument("--latent-dim-shared", type=int, default=20)
    parser.add_argument("--latent-dim-specific", type=int, default=20)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--gamma", type=float, default=5.0)
    parser.add_argument("--lambda-adv", type=float, default=5.0)
    parser.add_argument("--confidence-weighted", action="store_true",
                        help="Enable confidence-weighted adversarial training.")
    parser.add_argument("--gate-mode", type=str, default="robust_adv",
                        choices=["legacy", "robust_adv"],
                        help="Gate backend when --confidence-weighted is enabled.")
    parser.add_argument("--cw-adv-ramp-epochs", type=int, default=10,
                        help="Epochs to ramp lambda_adv from 0 to target after warmup.")
    parser.add_argument("--gate-start-epoch", type=int, default=None,
                        help="Epoch to enable gated branch (default: equals --num-warmup).")
    parser.add_argument("--gate-ramp-epochs", type=int, default=10,
                        help="Epochs to ramp gated branch weight.")
    parser.add_argument("--lambda-adv-base-ratio", type=float, default=0.35,
                        help="Base-branch ratio in dual-branch adversarial loss.")
    parser.add_argument("--rho-target", type=float, default=0.65,
                        help="Target keep ratio for robust gate budget control.")
    parser.add_argument("--w-floor", type=float, default=0.15,
                        help="Minimum adversarial sample weight.")
    parser.add_argument("--w-orphan-min", type=float, default=0.45,
                        help="Minimum weight for orphan samples in robust gate.")
    parser.add_argument("--rarity-boost", type=float, default=0.10,
                        help="Rare-modality additive boost in robust gate.")
    parser.add_argument("--orphan-sim-threshold", type=float, default=0.15,
                        help="Top1 cross-modal similarity threshold to mark orphan samples.")
    parser.add_argument("--orphan-margin-threshold", type=float, default=0.02,
                        help="Top1-top2 margin threshold to mark orphan samples.")
    parser.add_argument("--dropout-rate", type=float, default=0.2)
    parser.add_argument("--latent-backend", type=str, default="vae", choices=["vae", "diffusion"])
    parser.add_argument("--lambda-prior-diff", type=float, default=None,
                        help="Weight for diffusion prior loss (only used when latent-backend=diffusion).")
    parser.add_argument("--lambda-diff", type=float, default=None,
                        help="Deprecated alias of --lambda-prior-diff.")
    parser.add_argument("--beta-specific", type=float, default=None,
                        help="KL weight for modality-specific latent branch. Default uses --beta.")
    parser.add_argument("--diffusion-steps", type=int, default=200)
    parser.add_argument("--diffusion-hidden-dim", type=int, default=512)
    parser.add_argument("--diffusion-time-embed-dim", type=int, default=64)
    parser.add_argument("--diffusion-beta-schedule", type=str, default="linear",
                        choices=["linear", "cosine"])
    parser.add_argument("--diffusion-prior-cond", type=str, default=None,
                        choices=["none", "modality", "modality_batch", "modality_batch_celltype"])
    parser.add_argument("--diffusion-cond", type=str, default=None,
                        choices=["none", "modality", "modality_batch", "modality_batch_celltype"],
                        help="Deprecated alias of --diffusion-prior-cond.")

    parser.add_argument("--epoch-num", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--adaptlr", action="store_true")
    parser.add_argument("--num-warmup", type=int, default=0)
    parser.add_argument("--valid-prop", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--no-early-stopping", action="store_true")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed.")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic backend settings (may reduce speed).",
    )
    parser.add_argument("--lambda-anchor", type=float, default=0.0)
    parser.add_argument("--k-mnn", type=int, default=30)
    parser.add_argument("--anchor-space", type=str, default="latent", choices=["raw", "latent"],
                        help="Space for MNN pairing: 'latent' (mu_shared) or 'raw' (input features).")
    parser.add_argument("--anchor-start-epoch", type=int, default=0,
                        help="Epoch at which anchor loss begins (Phase A has no anchor).")
    parser.add_argument("--anchor-ramp-epochs", type=int, default=10,
                        help="Number of epochs to linearly ramp lambda_anchor from 0 to target.")
    parser.add_argument("--anchor-sim-threshold", type=float, default=0.0,
                        help="Minimum cosine similarity for MNN pairs (raw/latent).")
    parser.add_argument("--anchor-margin", type=float, default=0.0,
                        help="Minimum top1-top2 similarity gap for MNN (raw/latent).")
    parser.add_argument(
        "--linked-features-uns-key",
        type=str,
        default=None,
        help=(
            "Optional adata.uns key that stores linked features (names or indices) "
            "for raw-space MNN pairing."
        ),
    )
    parser.add_argument(
        "--linked-features-file",
        type=Path,
        default=None,
        help=(
            "Optional text file with one linked feature per line "
            "(feature name or integer index)."
        ),
    )

    args = parser.parse_args()
    set_seed(args.seed, deterministic=args.deterministic)
    print(f"Global seed set to {args.seed} (deterministic={args.deterministic})")
    lambda_prior_diff = args.lambda_prior_diff
    if lambda_prior_diff is None:
        lambda_prior_diff = args.lambda_diff if args.lambda_diff is not None else 1.0
    elif args.lambda_diff is not None:
        print("Warning: --lambda-diff is deprecated and ignored because --lambda-prior-diff is provided.")

    diffusion_prior_cond = args.diffusion_prior_cond
    if diffusion_prior_cond is None:
        diffusion_prior_cond = args.diffusion_cond if args.diffusion_cond is not None else "none"
    elif args.diffusion_cond is not None:
        print("Warning: --diffusion-cond is deprecated and ignored because --diffusion-prior-cond is provided.")

    input_path = args.input_h5ad
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if args.output_h5ad is None:
        output_path = input_path.with_name(f"{input_path.stem}_trained_anchor.h5ad")
    else:
        output_path = args.output_h5ad

    print(f"Loading data: {input_path}")
    adata = sc.read_h5ad(input_path)
    selected_layer = args.layer
    if selected_layer is not None and selected_layer not in adata.layers:
        print(
            f"Warning: layer '{selected_layer}' not found in adata.layers. "
            "Fallback to adata.X."
        )
        selected_layer = None
    num_modalities = adata.obs[args.modality_key].astype(str).nunique()
    print(
        f"Discriminator CE random baseline (log K): {np.log(num_modalities):.4f} "
        f"(K={num_modalities})"
    )

    if args.disable_feature_list:
        feature_list = None
        print("feature_list disabled (feature_list=None).")
    else:
        feature_list = build_feature_list(adata, args.modality_key)

    linked_features = None
    if args.linked_features_uns_key is not None:
        if args.linked_features_uns_key not in adata.uns:
            raise KeyError(
                f"linked features uns key not found: {args.linked_features_uns_key}"
            )
        linked_features = list(adata.uns[args.linked_features_uns_key])
        print(
            f"linked_features loaded from adata.uns['{args.linked_features_uns_key}']: "
            f"{len(linked_features)} entries"
        )
    if args.linked_features_file is not None:
        if not args.linked_features_file.exists():
            raise FileNotFoundError(
                f"linked features file not found: {args.linked_features_file}"
            )
        linked_features = parse_linked_feature_file(args.linked_features_file)
        print(
            f"linked_features loaded from file {args.linked_features_file}: "
            f"{len(linked_features)} entries"
        )

    model = Integration(
        data=adata,
        layer=selected_layer,
        modality_key=args.modality_key,
        batch_key=args.batch_key,
        feature_list=feature_list,
        distribution=args.distribution,
    )
    model.setup(
        hidden_layers=args.hidden_layers,
        latent_dim_shared=args.latent_dim_shared,
        latent_dim_specific=args.latent_dim_specific,
        beta=args.beta,
        gamma=args.gamma,
        lambda_adv=args.lambda_adv,
        confidence_weighted=args.confidence_weighted,
        gate_mode=args.gate_mode,
        cw_w_min=args.w_floor,
        lambda_adv_base_ratio=args.lambda_adv_base_ratio,
        rho_target=args.rho_target,
        w_orphan_min=args.w_orphan_min,
        rarity_boost=args.rarity_boost,
        orphan_sim_threshold=args.orphan_sim_threshold,
        orphan_margin_threshold=args.orphan_margin_threshold,
        dropout_rate=args.dropout_rate,
        linked_features=linked_features,
        latent_backend=args.latent_backend,
        lambda_prior_diff=lambda_prior_diff,
        diffusion_steps=args.diffusion_steps,
        diffusion_hidden_dim=args.diffusion_hidden_dim,
        diffusion_time_embed_dim=args.diffusion_time_embed_dim,
        diffusion_beta_schedule=args.diffusion_beta_schedule,
        diffusion_prior_cond=diffusion_prior_cond,
        beta_specific=args.beta_specific,
    )
    model.train(
        epoch_num=args.epoch_num,
        batch_size=args.batch_size,
        lr=args.lr,
        adaptlr=args.adaptlr,
        num_warmup=args.num_warmup,
        early_stopping=not args.no_early_stopping,
        valid_prop=args.valid_prop,
        patience=args.patience,
        random_state=args.seed,
        cw_adv_ramp_epochs=args.cw_adv_ramp_epochs,
        gate_start_epoch=args.gate_start_epoch,
        gate_ramp_epochs=args.gate_ramp_epochs,
        lambda_anchor=args.lambda_anchor,
        k_mnn=args.k_mnn,
        anchor_space=args.anchor_space,
        anchor_start_epoch=args.anchor_start_epoch,
        anchor_ramp_epochs=args.anchor_ramp_epochs,
        anchor_sim_threshold=args.anchor_sim_threshold,
        anchor_margin=args.anchor_margin,
    )
    model.inference(n_samples=1, update=True, returns=False)
    model.get_adata().write_h5ad(output_path)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
