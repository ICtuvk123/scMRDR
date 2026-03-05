#!/usr/bin/env bash
set -euo pipefail

cd /root/rivermind-data/scMRDR
mkdir -p experiments/BMMC_codes/logs_s2

COMMON_ANCHOR="--input-h5ad experiments/BMMC_codes/feature_aligned_sampled.h5ad --modality-key modality --batch-key batch --layer counts --distribution ZINB --hidden-layers 512,512 --latent-dim-shared 20 --latent-dim-specific 20 --beta 2 --gamma 5 --dropout-rate 0.2 --epoch-num 200 --batch-size 128 --lr 1e-3 --valid-prop 0.1 --patience 10 --disable-feature-list"

for adv in 3.5 4.0 4.5; do
  for la in 0.25 0.30 0.35 0.40; do
    adv_tag=${adv/./_}
    la_tag=${la/./_}
    name="anchor_s2_adv${adv_tag}_la${la_tag}_st20_sim0_05_m0_01"
    out="experiments/BMMC_codes/${name}.h5ad"
    tlog="experiments/BMMC_codes/logs_s2/train_${name}.log"
    mlog="experiments/BMMC_codes/logs_s2/metrics_${name}.log"
    mout="experiments/BMMC_codes/metrics_${name}/scaled_metrics_local.csv"

    if [ -f "$out" ] && [ -f "$mout" ]; then
      echo "[SKIP] $name (train+metrics exists)"
      continue
    fi

    if [ ! -f "$out" ]; then
      echo "[RUN][TRAIN] $name"
      python scripts/train_anchor.py $COMMON_ANCHOR \
        --lambda-adv "$adv" \
        --lambda-anchor "$la" \
        --anchor-space raw \
        --linked-features-file experiments/BMMC_codes/linked_features_160.txt \
        --k-mnn 10 \
        --anchor-start-epoch 20 \
        --anchor-ramp-epochs 30 \
        --anchor-sim-threshold 0.05 \
        --anchor-margin 0.01 \
        --output-h5ad "$out" > "$tlog" 2>&1 || { echo "[ERR][TRAIN] $name"; continue; }
    else
      echo "[SKIP][TRAIN] $name (h5ad exists)"
    fi

    if [ ! -f "$mout" ]; then
      echo "[RUN][METRICS] $name"
      python experiments/plots/metrics.py \
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
done

echo "[DONE] stage2 finished."
