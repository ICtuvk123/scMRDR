#!/usr/bin/env bash
set -euo pipefail
cd /root/rivermind-data/scMRDR

/opt/conda/envs/scMRDR/bin/python - <<'PY'
import glob,re
rows=[]
for p in glob.glob('experiments/BMMC_codes/logs_s2/metrics_*.log'):
    t=open(p,encoding='utf-8',errors='ignore').read()
    m1=re.search(r'Batch correction\s+([0-9]*\.?[0-9]+)',t)
    m2=re.search(r'Modality integration\s+([0-9]*\.?[0-9]+)',t)
    m3=re.search(r'Bio conservation\s+([0-9]*\.?[0-9]+)',t)
    m4=re.search(r'Total\s+([0-9]*\.?[0-9]+)',t)
    m5=re.search(r'Name:\s*([^,\n]+)',t)
    if all([m1,m2,m3,m4,m5]):
        rows.append((float(m4.group(1)),m5.group(1),float(m1.group(1)),float(m2.group(1)),float(m3.group(1))))
rows.sort(reverse=True)
print(f'COUNT={len(rows)}')
for total,name,b,m,bio in rows[:20]:
    print(f'{name}\tTotal={total:.6f}\tBatch={b:.6f}\tMod={m:.6f}\tBio={bio:.6f}')
PY
