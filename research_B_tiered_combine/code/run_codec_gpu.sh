#!/usr/bin/env bash
# Entire qualification + benchmark owns the existing shared GPU lock.
set -euo pipefail
cd "$(dirname "$0")/.."
RUN=${1:?Usage: run_codec_gpu.sh unique_run_name}
[[ "$RUN" =~ ^[A-Za-z0-9_-]+$ ]] || exit 2
mkdir "runs/$RUN"
exec > >(tee "runs/$RUN/driver.log") 2>&1
exec 9>/root/autodl-tmp/moe-research-gpu.lock
flock -n 9 || { echo 'GPU lock busy; experiment not started'; exit 3; }
PYTHON=/root/miniconda3/bin/python
"$PYTHON" - <<'PY'
import subprocess
q=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
print(q,flush=True)
rows=[list(map(int,s.split(','))) for s in q.strip().splitlines()]
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
if len(rows)!=1 or apps or any(mem>64 or util!=0 for mem,util in rows):
    raise SystemExit('GPU not verified idle; no experiment launched')
PY
tar -czf "runs/$RUN/code.tar.gz" code/*.py code/*.sh
"$PYTHON" code/check_codec.py --gpu --output "runs/$RUN/check.json"
"$PYTHON" code/bench_codec.py --output "runs/$RUN/bench.json"
echo COMPLETE
