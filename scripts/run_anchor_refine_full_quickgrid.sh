#!/usr/bin/env bash
set -euo pipefail

cd /root/rivermind-data/scMRDR
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache}"

# Quick full-data screening defaults (single-seed, short budget).
INPUT_H5AD="${INPUT_H5AD:-experiments/BMMC_codes/feature_aligned.h5ad}"
SEED="${SEED:-0}"
RUN_TAG="${RUN_TAG:-full_quickgrid_s0}"
FORCE="${FORCE:-0}"
TOPK="${TOPK:-15}"

# Grids
BATCH_SIZE_GRID="${BATCH_SIZE_GRID:-128 256}"
ADV_GRID="${ADV_GRID:-3.0 4.0 4.5 5.0 6.0}"
LA_GRID="${LA_GRID:-0.20 0.25 0.30}"
K_MNN_GRID="${K_MNN_GRID:-10}"
SIM_GRID="${SIM_GRID:-0.06}"
MARGIN_GRID="${MARGIN_GRID:-0.012}"

# Fast budget for screening
EPOCH_NUM="${EPOCH_NUM:-80}"
PATIENCE="${PATIENCE:-4}"
LR="${LR:-1e-3}"
N_JOBS="${N_JOBS:-8}"

echo "[INFO] quick grid start"
echo "[INFO] INPUT_H5AD=$INPUT_H5AD"
echo "[INFO] RUN_TAG=$RUN_TAG"
echo "[INFO] BATCH_SIZE_GRID=$BATCH_SIZE_GRID"
echo "[INFO] ADV_GRID=$ADV_GRID"
echo "[INFO] LA_GRID=$LA_GRID"

for bs in $BATCH_SIZE_GRID; do
  echo ""
  echo "[INFO] ===== batch_size=$bs ====="
  INPUT_H5AD="$INPUT_H5AD" \
  SEED="$SEED" \
  FORCE="$FORCE" \
  RUN_TAG="${RUN_TAG}_bs${bs}" \
  N_JOBS="$N_JOBS" \
  BATCH_SIZE="$bs" \
  EPOCH_NUM="$EPOCH_NUM" \
  PATIENCE="$PATIENCE" \
  LR="$LR" \
  ADV_GRID="$ADV_GRID" \
  LA_GRID="$LA_GRID" \
  K_MNN_GRID="$K_MNN_GRID" \
  SIM_GRID="$SIM_GRID" \
  MARGIN_GRID="$MARGIN_GRID" \
  bash scripts/run_anchor_refine_full_smallgrid.sh
done

echo ""
echo "[INFO] ===== ranking by Total ====="
/opt/conda/envs/scMRDR/bin/python - <<'PY'
import csv
import glob
import os
from pathlib import Path

root = Path("experiments/BMMC_codes")
run_tag = os.environ.get("RUN_TAG", "full_quickgrid_s0")
topk = int(os.environ.get("TOPK", "15"))

rows = []
for csv_path in glob.glob(str(root / f"metrics_*{run_tag}_bs*/unscaled_metrics_local.csv")):
    p = Path(csv_path)
    with p.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            continue
        key_col = reader.fieldnames[0]
        for row in reader:
            name = row.get(key_col, "")
            if not name or run_tag not in name:
                continue
            try:
                total = float(row["Total"])
                bio = float(row["Bio conservation"])
                batch = float(row["Batch correction"])
                mod = float(row["Modality integration"])
            except Exception:
                continue
            rows.append((total, name, batch, bio, mod, str(p.parent)))

rows.sort(key=lambda x: x[0], reverse=True)
print(f"count={len(rows)}")
for total, name, batch, bio, mod, outdir in rows[:topk]:
    print(
        f"{name}\tTotal={total:.6f}\tBatch={batch:.6f}\tBio={bio:.6f}\tMod={mod:.6f}\tout={outdir}"
    )
PY

echo "[DONE] quick grid finished"
