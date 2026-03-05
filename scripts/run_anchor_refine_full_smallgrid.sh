#!/usr/bin/env bash
set -euo pipefail

cd /root/rivermind-data/scMRDR
mkdir -p experiments/BMMC_codes/logs_refine_full_smallgrid
PYTHON_BIN="/opt/conda/envs/scMRDR/bin/python"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache}"

# Full dataset by default; keep seed=0 for fast screening.
INPUT_H5AD="${INPUT_H5AD:-experiments/BMMC_codes/feature_aligned.h5ad}"
SEED="${SEED:-0}"
FORCE="${FORCE:-0}"        # FORCE=1 to rerun completed jobs
RUN_TAG="${RUN_TAG:-full_smallgrid_s0}"
N_JOBS="${N_JOBS:-8}"
BATCH_SIZE="${BATCH_SIZE:-128}"
EPOCH_NUM="${EPOCH_NUM:-200}"
PATIENCE="${PATIENCE:-10}"
LR="${LR:-1e-3}"

# Small-range hyperparameter grids (space-separated).
ADV_GRID="${ADV_GRID:-4.0 4.5 5.0}"
LA_GRID="${LA_GRID:-0.20 0.25 0.30}"

# Additional grids (space-separated). Defaults keep prior behavior.
K_MNN_GRID="${K_MNN_GRID:-10}"
SIM_GRID="${SIM_GRID:-0.06}"
MARGIN_GRID="${MARGIN_GRID:-0.012}"

# Keep schedule fixed unless explicitly overridden.
ANCHOR_START_EPOCH="${ANCHOR_START_EPOCH:-20}"
ANCHOR_RAMP_EPOCHS="${ANCHOR_RAMP_EPOCHS:-30}"

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
  --epoch-num "$EPOCH_NUM"
  --batch-size "$BATCH_SIZE"
  --lr "$LR"
  --valid-prop 0.1
  --patience "$PATIENCE"
  --disable-feature-list
  --seed "$SEED"
)

count=0
for ADV in $ADV_GRID; do
  for LA in $LA_GRID; do
    for K_MNN in $K_MNN_GRID; do
      for SIM in $SIM_GRID; do
        for MARGIN in $MARGIN_GRID; do
          adv_tag=${ADV/./_}
          la_tag=${LA/./_}
          sim_tag=${SIM/./_}
          mar_tag=${MARGIN/./_}
          name="anchor_refine_adv${adv_tag}_la${la_tag}_k${K_MNN}_st${ANCHOR_START_EPOCH}_sim${sim_tag}_m${mar_tag}_${RUN_TAG}"
          out="experiments/BMMC_codes/${name}.h5ad"
          tlog="experiments/BMMC_codes/logs_refine_full_smallgrid/train_${name}.log"
          mlog="experiments/BMMC_codes/logs_refine_full_smallgrid/metrics_${name}.log"
          mout="experiments/BMMC_codes/metrics_${name}/scaled_metrics_local.csv"

          count=$((count + 1))
          echo ""
          echo "[${count}] ${name}"

          if [[ "$FORCE" == "1" || ! -f "$out" ]]; then
            echo "[RUN][TRAIN] $name"
            "$PYTHON_BIN" scripts/train_anchor.py "${COMMON_ANCHOR[@]}" \
              --lambda-adv "$ADV" \
              --lambda-anchor "$LA" \
              --anchor-space raw \
              --linked-features-file experiments/BMMC_codes/linked_features_160.txt \
              --k-mnn "$K_MNN" \
              --anchor-start-epoch "$ANCHOR_START_EPOCH" \
              --anchor-ramp-epochs "$ANCHOR_RAMP_EPOCHS" \
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
              --n-jobs "$N_JOBS" > "$mlog" 2>&1 || echo "[ERR][METRICS] $name"
          else
            echo "[SKIP][METRICS] $name (metrics exists)"
          fi
        done
      done
    done
  done
done

echo "[DONE] full small-grid finished. total_runs=${count}"
