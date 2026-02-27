#!/usr/bin/env python3
import argparse
import csv
import itertools
import json
import random
import subprocess
import sys
from pathlib import Path


def parse_float_list(text: str) -> list[float]:
    values = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(float(item))
    if not values:
        raise ValueError("Empty float list.")
    return values


def parse_int_list(text: str) -> list[int]:
    values = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(int(item))
    if not values:
        raise ValueError("Empty int list.")
    return values


def dedup_grid(grid: list[tuple[float, float, float, int, float, float]]) -> list[tuple[float, float, float, int, float, float]]:
    seen: set[tuple[float, float, float, int, float, float]] = set()
    out: list[tuple[float, float, float, int, float, float]] = []
    for item in grid:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def fmt_num(x: float) -> str:
    if abs(x - int(x)) < 1e-12:
        return str(int(x))
    return f"{x:.4g}".replace(".", "p")


def make_run_name(prefix: str, cfg: dict) -> str:
    name = (
        f"{prefix}"
        f"_adv{fmt_num(cfg['lambda_adv'])}"
        f"_anc{fmt_num(cfg['lambda_anchor'])}"
        f"_k{cfg['k_mnn']}"
        f"_sim{fmt_num(cfg['anchor_sim_threshold'])}"
        f"_mar{fmt_num(cfg['anchor_margin'])}"
    )
    if cfg.get("latent_backend") == "diffusion":
        name += f"_ld{fmt_num(cfg['lambda_prior_diff'])}"
    return name


def run_command(cmd: list[str]) -> None:
    print(">>>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def load_total_score(unscaled_csv: Path, method_name: str) -> dict[str, float]:
    keys = ["Batch correction", "Modality integration", "Bio conservation", "Total"]
    with unscaled_csv.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Empty CSV: {unscaled_csv}")
        index_col = reader.fieldnames[0]
        for row in reader:
            if row.get(index_col) == method_name:
                missing = [k for k in keys if k not in row]
                if missing:
                    raise KeyError(f"Missing metric columns in {unscaled_csv}: {missing}")
                return {k: float(row[k]) for k in keys}
    raise KeyError(f"Method '{method_name}' not found in {unscaled_csv}.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Grid search for scMRDR anchor/adversarial hyperparameters. "
            "For each config: train -> run local metrics -> rank by Total."
        )
    )
    parser.add_argument("--input-h5ad", type=Path, required=True)
    parser.add_argument("--search-outdir", type=Path, required=True)

    parser.add_argument("--modality-key", type=str, default="modality")
    parser.add_argument("--batch-key", type=str, default="batch")
    parser.add_argument("--label-key", type=str, default="celltype")
    parser.add_argument("--embedding-key", type=str, default="latent_shared")
    parser.add_argument("--anchor-space", type=str, default="raw", choices=["raw", "latent"])
    parser.add_argument("--linked-features-uns-key", type=str, default="linked_features")
    parser.add_argument("--distribution", type=str, default="ZINB")
    parser.add_argument("--layer", type=str, default=None)
    parser.add_argument("--latent-backend", type=str, default="vae", choices=["vae", "diffusion"])
    parser.add_argument("--diffusion-steps", type=int, default=200)
    parser.add_argument("--diffusion-hidden-dim", type=int, default=512)
    parser.add_argument("--diffusion-time-embed-dim", type=int, default=64)
    parser.add_argument("--diffusion-beta-schedule", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--diffusion-prior-cond", type=str, default=None,
                        choices=["none", "modality", "modality_batch", "modality_batch_celltype"])
    parser.add_argument("--diffusion-cond", type=str, default=None,
                        choices=["none", "modality", "modality_batch", "modality_batch_celltype"],
                        help="Deprecated alias of --diffusion-prior-cond.")

    parser.add_argument("--lambda-adv-grid", type=parse_float_list, default=[5.0, 10.0, 15.0])
    parser.add_argument("--lambda-anchor-grid", type=parse_float_list, default=[0.01, 0.02, 0.03])
    parser.add_argument("--lambda-prior-diff-grid", type=parse_float_list, default=None)
    parser.add_argument("--lambda-diff-grid", type=parse_float_list, default=None,
                        help="Deprecated alias of --lambda-prior-diff-grid.")
    parser.add_argument("--k-mnn-grid", type=parse_int_list, default=[8, 10, 12])
    parser.add_argument("--anchor-sim-grid", type=parse_float_list, default=[0.0, 0.05])
    parser.add_argument("--anchor-margin-grid", type=parse_float_list, default=[0.0])

    parser.add_argument("--anchor-start-epoch", type=int, default=10)
    parser.add_argument("--anchor-ramp-epochs", type=int, default=15)
    parser.add_argument("--epoch-num", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--valid-prop", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--run-prefix", type=str, default="gs")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--max-runs", type=int, default=None, help="Optional cap for quick tests.")
    parser.add_argument(
        "--collapse-anchor-zero",
        action="store_true",
        default=True,
        help=(
            "When lambda-anchor <= 0, collapse k/sim/margin search to a single baseline "
            "(avoids redundant runs where anchor branch is inactive)."
        ),
    )
    parser.add_argument(
        "--no-collapse-anchor-zero",
        dest="collapse_anchor_zero",
        action="store_false",
        help="Disable anchor-zero collapsing and use full Cartesian product.",
    )
    parser.add_argument("--anchor-zero-k", type=int, default=None,
                        help="Baseline k-mnn used when lambda-anchor <= 0. Default: first value in --k-mnn-grid.")
    parser.add_argument("--anchor-zero-sim", type=float, default=None,
                        help="Baseline anchor-sim-threshold used when lambda-anchor <= 0. Default: first value in --anchor-sim-grid.")
    parser.add_argument("--anchor-zero-margin", type=float, default=None,
                        help="Baseline anchor-margin used when lambda-anchor <= 0. Default: first value in --anchor-margin-grid.")
    parser.add_argument("--shuffle-grid", action="store_true",
                        help="Shuffle grid order before truncating by --max-runs (recommended for quick coarse search).")
    parser.add_argument("--grid-seed", type=int, default=42,
                        help="Random seed used by --shuffle-grid.")
    args = parser.parse_args()

    if args.lambda_prior_diff_grid is not None:
        lambda_prior_diff_grid = args.lambda_prior_diff_grid
        if args.lambda_diff_grid is not None:
            print("Warning: --lambda-diff-grid is deprecated and ignored because --lambda-prior-diff-grid is provided.")
    elif args.lambda_diff_grid is not None:
        lambda_prior_diff_grid = args.lambda_diff_grid
    else:
        lambda_prior_diff_grid = [1.0]

    if args.diffusion_prior_cond is not None:
        diffusion_prior_cond = args.diffusion_prior_cond
        if args.diffusion_cond is not None:
            print("Warning: --diffusion-cond is deprecated and ignored because --diffusion-prior-cond is provided.")
    else:
        diffusion_prior_cond = args.diffusion_cond if args.diffusion_cond is not None else "none"

    if not args.input_h5ad.exists():
        raise FileNotFoundError(args.input_h5ad)

    search_outdir = args.search_outdir
    model_dir = search_outdir / "models"
    metric_dir = search_outdir / "metrics"
    model_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)

    lambda_prior_diff_values = lambda_prior_diff_grid if args.latent_backend == "diffusion" else [0.0]
    k_zero = args.anchor_zero_k if args.anchor_zero_k is not None else args.k_mnn_grid[0]
    sim_zero = args.anchor_zero_sim if args.anchor_zero_sim is not None else args.anchor_sim_grid[0]
    margin_zero = args.anchor_zero_margin if args.anchor_zero_margin is not None else args.anchor_margin_grid[0]

    grid: list[tuple[float, float, float, int, float, float]] = []
    if args.collapse_anchor_zero:
        for lambda_adv, lambda_anchor, lambda_prior_diff in itertools.product(
            args.lambda_adv_grid,
            args.lambda_anchor_grid,
            lambda_prior_diff_values,
        ):
            if lambda_anchor <= 0.0:
                grid.append((lambda_adv, lambda_anchor, lambda_prior_diff, k_zero, sim_zero, margin_zero))
            else:
                for k_mnn, sim_th, margin in itertools.product(
                    args.k_mnn_grid,
                    args.anchor_sim_grid,
                    args.anchor_margin_grid,
                ):
                    grid.append((lambda_adv, lambda_anchor, lambda_prior_diff, k_mnn, sim_th, margin))
        grid = dedup_grid(grid)
    else:
        grid = list(
            itertools.product(
                args.lambda_adv_grid,
                args.lambda_anchor_grid,
                lambda_prior_diff_values,
                args.k_mnn_grid,
                args.anchor_sim_grid,
                args.anchor_margin_grid,
            )
        )

    if args.shuffle_grid:
        rng = random.Random(args.grid_seed)
        rng.shuffle(grid)

    if args.max_runs is not None:
        grid = grid[: args.max_runs]

    print(f"Total configs: {len(grid)}")

    all_rows: list[dict] = []
    for run_idx, (lambda_adv, lambda_anchor, lambda_prior_diff, k_mnn, sim_th, margin) in enumerate(grid, start=1):
        cfg = dict(
            latent_backend=args.latent_backend,
            lambda_adv=lambda_adv,
            lambda_anchor=lambda_anchor,
            lambda_prior_diff=lambda_prior_diff,
            lambda_diff=lambda_prior_diff,
            k_mnn=k_mnn,
            anchor_sim_threshold=sim_th,
            anchor_margin=margin,
        )
        run_name = make_run_name(args.run_prefix, cfg)
        out_model = model_dir / f"{run_name}.h5ad"
        out_metrics = metric_dir / run_name
        out_metrics.mkdir(parents=True, exist_ok=True)
        unscaled_csv = out_metrics / "unscaled_metrics_local.csv"

        print(f"\n=== [{run_idx}/{len(grid)}] {run_name} ===")
        if args.skip_existing and out_model.exists() and unscaled_csv.exists():
            print("Skip existing run.")
        else:
            train_cmd = [
                sys.executable, "scripts/train_anchor.py",
                "--input-h5ad", str(args.input_h5ad),
                "--output-h5ad", str(out_model),
                "--modality-key", args.modality_key,
                "--batch-key", args.batch_key,
                "--distribution", args.distribution,
                "--anchor-space", args.anchor_space,
                "--linked-features-uns-key", args.linked_features_uns_key,
                "--latent-backend", args.latent_backend,
                "--lambda-adv", str(lambda_adv),
                "--lambda-anchor", str(lambda_anchor),
                "--lambda-prior-diff", str(lambda_prior_diff),
                "--diffusion-steps", str(args.diffusion_steps),
                "--diffusion-hidden-dim", str(args.diffusion_hidden_dim),
                "--diffusion-time-embed-dim", str(args.diffusion_time_embed_dim),
                "--diffusion-beta-schedule", args.diffusion_beta_schedule,
                "--diffusion-prior-cond", diffusion_prior_cond,
                "--k-mnn", str(k_mnn),
                "--anchor-start-epoch", str(args.anchor_start_epoch),
                "--anchor-ramp-epochs", str(args.anchor_ramp_epochs),
                "--anchor-sim-threshold", str(sim_th),
                "--anchor-margin", str(margin),
                "--epoch-num", str(args.epoch_num),
                "--batch-size", str(args.batch_size),
                "--lr", str(args.lr),
                "--valid-prop", str(args.valid_prop),
                "--patience", str(args.patience),
            ]
            if args.layer is not None:
                train_cmd.extend(["--layer", args.layer])
            run_command(train_cmd)

            metric_cmd = [
                sys.executable, "experiments/plots/metrics.py",
                "--local-bmmc",
                "--adata", str(out_model),
                "--outdir", str(out_metrics),
                "--embedding-key", args.embedding_key,
                "--method-name", run_name,
                "--batch-key", args.batch_key,
                "--label-key", args.label_key,
                "--modality-key", args.modality_key,
                "--n-jobs", str(args.n_jobs),
            ]
            run_command(metric_cmd)

        scores = load_total_score(unscaled_csv, run_name)
        row = {"run_name": run_name, **cfg, **scores, "model_path": str(out_model)}
        all_rows.append(row)
        print(
            f"Scores | Total={scores['Total']:.6f}, "
            f"Bio={scores['Bio conservation']:.6f}, "
            f"Batch={scores['Batch correction']:.6f}, "
            f"Modality={scores['Modality integration']:.6f}"
        )

    all_rows = sorted(all_rows, key=lambda x: x["Total"], reverse=True)
    result_csv = search_outdir / "grid_results.csv"
    if all_rows:
        fieldnames = [
            "run_name",
            "latent_backend",
            "lambda_adv",
            "lambda_anchor",
            "lambda_prior_diff",
            "lambda_diff",
            "k_mnn",
            "anchor_sim_threshold",
            "anchor_margin",
            "Batch correction",
            "Modality integration",
            "Bio conservation",
            "Total",
            "model_path",
        ]
        with result_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)

    if not all_rows:
        raise RuntimeError("No completed runs. Check grid settings and logs.")

    best = all_rows[0]
    best_json = search_outdir / "best_config.json"
    with best_json.open("w", encoding="utf-8") as handle:
        json.dump(best, handle, ensure_ascii=False, indent=2)

    print("\n=== Grid Search Done ===")
    print(f"Best run: {best['run_name']}")
    print(
        f"Best scores | Total={best['Total']:.6f}, "
        f"Bio={best['Bio conservation']:.6f}, "
        f"Batch={best['Batch correction']:.6f}, "
        f"Modality={best['Modality integration']:.6f}"
    )
    print(f"Saved ranking: {result_csv}")
    print(f"Saved best config: {best_json}")


if __name__ == "__main__":
    main()
