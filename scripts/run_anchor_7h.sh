#!/usr/bin/env bash
set -euo pipefail

cd /root/rivermind-data/scMRDR
mkdir -p experiments/BMMC_codes/logs

COMMON_ANCHOR="--input-h5ad experiments/BMMC_codes/feature_aligned_sampled.h5ad --modality-key modality --batch-key batch --layer counts --distribution ZINB --hidden-layers 512,512 --latent-dim-shared 20 --latent-dim-specific 20 --beta 2 --gamma 5 --dropout-rate 0.2 --epoch-num 200 --batch-size 128 --lr 1e-3 --valid-prop 0.1 --patience 10 --disable-feature-list"

END_TS=$(( $(date +%s) + 7*3600 ))
STOP=0

for adv in 4 5 6 7; do
  [ "$STOP" -eq 1 ] && break
  for la in 0.2 0.3 0.4 0.5 0.6; do
    [ "$STOP" -eq 1 ] && break
    for st in 20 30 40; do
      [ "$STOP" -eq 1 ] && break
      for cfg in "0.0 0.0" "0.02 0.005" "0.05 0.01"; do
        now=$(date +%s)
        if [ "$now" -ge "$END_TS" ]; then
          echo "[TIME] 7h budget reached."
          STOP=1
          break
        fi

        set -- $cfg
        sim=$1
        mar=$2

        la_tag=${la/./_}
        sim_tag=${sim/./_}
        mar_tag=${mar/./_}
        out="experiments/BMMC_codes/anchor_7h_adv${adv}_la${la_tag}_st${st}_sim${sim_tag}_m${mar_tag}.h5ad"
        name=$(basename "$out" .h5ad)
        train_log="experiments/BMMC_codes/logs/train_${name}.log"
        metrics_log="experiments/BMMC_codes/logs/metrics_${name}.log"
        metrics_csv="experiments/BMMC_codes/metrics_${name}/scaled_metrics_local.csv"

        if [ ! -f "$out" ]; then
          echo "[RUN ][TRAIN] $name"
          python scripts/train_anchor.py $COMMON_ANCHOR \
            --lambda-adv "$adv" \
            --lambda-anchor "$la" \
            --anchor-space raw \
            --linked-features-file experiments/BMMC_codes/linked_features_160.txt \
            --k-mnn 10 \
            --anchor-start-epoch "$st" \
            --anchor-ramp-epochs 30 \
            --anchor-sim-threshold "$sim" \
            --anchor-margin "$mar" \
            --output-h5ad "$out" > "$train_log" 2>&1
        else
          echo "[SKIP][TRAIN] $name"
        fi

        if [ ! -f "$metrics_csv" ]; then
          echo "[RUN ][METRICS] $name"
          python experiments/plots/metrics.py \
            --local-bmmc \
            --adata "$out" \
            --outdir "experiments/BMMC_codes/metrics_${name}" \
            --embedding-key latent_shared \
            --method-name "$name" \
            --batch-key batch \
            --label-key celltype \
            --modality-key modality \
            --n-jobs 8 > "$metrics_log" 2>&1
        else
          echo "[SKIP][METRICS] $name"
        fi
      done
    done
  done
done

echo "[DONE] sweep stopped (time limit or grid end)."
