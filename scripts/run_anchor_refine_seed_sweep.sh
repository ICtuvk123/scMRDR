#!/usr/bin/env bash
set -euo pipefail

cd /root/rivermind-data/scMRDR
mkdir -p experiments/BMMC_codes/logs_refine_seed
PYTHON_BIN="/opt/conda/envs/scMRDR/bin/python"

# You can override these from the shell, e.g.:
# RUN_TAG=full_v1 INPUT_H5AD=experiments/BMMC_codes/feature_aligned.h5ad SEEDS="0 1 2 3 4" bash scripts/run_anchor_refine_seed_sweep.sh
INPUT_H5AD="${INPUT_H5AD:-experiments/BMMC_codes/feature_aligned.h5ad}"
SEEDS="${SEEDS:-0 1 2 3 4}"
FORCE="${FORCE:-0}" # set FORCE=1 to rerun existing outputs
RUN_TAG="${RUN_TAG:-}" # optional suffix to avoid name collisions across datasets/runs

ADV="${ADV:-4.5}"
LA="${LA:-0.25}"
K_MNN="${K_MNN:-10}"
ANCHOR_START_EPOCH="${ANCHOR_START_EPOCH:-20}"
SIM="${SIM:-0.06}"
MARGIN="${MARGIN:-0.012}"

adv_tag=${ADV/./_}
la_tag=${LA/./_}
sim_tag=${SIM/./_}
mar_tag=${MARGIN/./_}
base_name="anchor_refine_adv${adv_tag}_la${la_tag}_k${K_MNN}_st${ANCHOR_START_EPOCH}_sim${sim_tag}_m${mar_tag}"
if [[ -n "$RUN_TAG" ]]; then
  base_name="${base_name}_${RUN_TAG}"
fi

COMMON_ANCHOR=(
  --input-h5ad "$INPUT_H5AD"
  --modality-key modality
  --batch-key batch
  --layer counts
  --distribution ZINB
  --hidden-layers 512,512
  --latent-dim-shared 20
  --latent-dim-specific 20
  --beta 2
  --gamma 5
  --dropout-rate 0.2
  --epoch-num 200
  --batch-size 128
  --lr 1e-3
  --valid-prop 0.1
  --patience 10
  --disable-feature-list
)

for seed in $SEEDS; do
  name="${base_name}_s${seed}"
  out="experiments/BMMC_codes/${name}.h5ad"
  tlog="experiments/BMMC_codes/logs_refine_seed/train_${name}.log"
  mlog="experiments/BMMC_codes/logs_refine_seed/metrics_${name}.log"
  mout="experiments/BMMC_codes/metrics_${name}/scaled_metrics_local.csv"

  if [[ "$FORCE" == "1" || ! -f "$out" ]]; then
    echo "[RUN][TRAIN] $name"
    "$PYTHON_BIN" scripts/train_anchor.py "${COMMON_ANCHOR[@]}" \
      --seed "$seed" \
      --lambda-adv "$ADV" \
      --lambda-anchor "$LA" \
      --anchor-space raw \
      --linked-features-file experiments/BMMC_codes/linked_features_160.txt \
      --k-mnn "$K_MNN" \
      --anchor-start-epoch "$ANCHOR_START_EPOCH" \
      --anchor-ramp-epochs 30 \
      --anchor-sim-threshold "$SIM" \
      --anchor-margin "$MARGIN" \
      --output-h5ad "$out" > "$tlog" 2>&1 || { echo "[ERR][TRAIN] $name"; continue; }
  else
    echo "[SKIP][TRAIN] $name (h5ad exists)"
  fi

  if [[ "$FORCE" == "1" || ! -f "$mout" ]]; then
    echo "[RUN][METRICS] $name"
    "$PYTHON_BIN" experiments/plots/metrics.py \
      --local-bmmc \
      --adata "$out" \
      --outdir "experiments/BMMC_codes/metrics_${name}" \
      --embedding-key latent_shared \
      --method-name "$name" \
      --batch-key batch \
      --label-key celltype \
      --modality-key modality \
      --n-jobs 8 > "$mlog" 2>&1 || echo "[ERR][METRICS] $name"
  else
    echo "[SKIP][METRICS] $name (metrics exists)"
  fi
done

echo "[DONE] seed sweep finished for ${base_name}"
