#!/usr/bin/env bash
set -euo pipefail

cd /root/rivermind-data/scMRDR
mkdir -p experiments/BMMC_codes/logs_refine_local
PYTHON_BIN="/opt/conda/envs/scMRDR/bin/python"

COMMON_ANCHOR="--input-h5ad experiments/BMMC_codes/feature_aligned_sampled.h5ad --modality-key modality --batch-key batch --layer counts --distribution ZINB --hidden-layers 512,512 --latent-dim-shared 20 --latent-dim-specific 20 --beta 2 --gamma 5 --dropout-rate 0.2 --epoch-num 200 --batch-size 128 --lr 1e-3 --valid-prop 0.1 --patience 10 --disable-feature-list"

# Around current best:
# anchor_full_adv4_5_la0_25_st20_sim0_05_m0_01
# We only perturb (adv, lambda_anchor, k_mnn, sim, margin).
CONFIGS=(
  "4.5 0.30 10 0.05 0.01"
  "5.0 0.25 10 0.05 0.01"
  "5.0 0.30 10 0.05 0.01"
  "4.5 0.25 12 0.05 0.01"
  "4.5 0.25 10 0.04 0.008"
  "4.5 0.25 10 0.06 0.012"
)

for cfg in "${CONFIGS[@]}"; do
  set -- $cfg
  adv=$1
  la=$2
  k=$3
  sim=$4
  mar=$5

  adv_tag=${adv/./_}
  la_tag=${la/./_}
  sim_tag=${sim/./_}
  mar_tag=${mar/./_}
  name="anchor_refine_adv${adv_tag}_la${la_tag}_k${k}_st20_sim${sim_tag}_m${mar_tag}"
  out="experiments/BMMC_codes/${name}.h5ad"
  tlog="experiments/BMMC_codes/logs_refine_local/train_${name}.log"
  mlog="experiments/BMMC_codes/logs_refine_local/metrics_${name}.log"
  mout="experiments/BMMC_codes/metrics_${name}/scaled_metrics_local.csv"

  if [ ! -f "$out" ]; then
    echo "[RUN][TRAIN] $name"
    "$PYTHON_BIN" scripts/train_anchor.py $COMMON_ANCHOR \
      --lambda-adv "$adv" \
      --lambda-anchor "$la" \
      --anchor-space raw \
      --linked-features-file experiments/BMMC_codes/linked_features_160.txt \
      --k-mnn "$k" \
      --anchor-start-epoch 20 \
      --anchor-ramp-epochs 30 \
      --anchor-sim-threshold "$sim" \
      --anchor-margin "$mar" \
      --output-h5ad "$out" > "$tlog" 2>&1 || { echo "[ERR][TRAIN] $name"; continue; }
  else
    echo "[SKIP][TRAIN] $name (h5ad exists)"
  fi

  if [ ! -f "$mout" ]; then
    echo "[RUN][METRICS] $name"
    "$PYTHON_BIN" experiments/plots/metrics.py \
      --local-bmmc \
      --adata "$out" \
      --outdir "experiments/BMMC_codes/metrics_${name}" \
      --embedding-key latent_shared \
      --method-name "$name" \
      --batch-key batch --label-key celltype --modality-key modality \
      --n-jobs 8 > "$mlog" 2>&1 || echo "[ERR][METRICS] $name"
  else
    echo "[SKIP][METRICS] $name (metrics exists)"
  fi
done

echo "[DONE] local refine sweep finished."
