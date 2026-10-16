#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
RUN=${1:?Usage: run_quality_smoke.sh unique_run model_path}
MODEL=${2:?model_path required}
[[ "$RUN" =~ ^[A-Za-z0-9_-]+$ ]] || exit 2
mkdir "runs/$RUN"
exec > >(tee "runs/$RUN/driver.log") 2>&1
exec 9>/root/autodl-tmp/moe-research-gpu.lock
flock -n 9 || { echo 'GPU lock busy; smoke not started'; exit 3; }
PYTHON=/root/miniconda3/bin/python
"$PYTHON" - <<'PY'
import subprocess
q=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
print(q,flush=True)
r=[list(map(int,s.split(','))) for s in q.strip().splitlines()]
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
if len(r)!=1 or apps or any(m>64 or u!=0 for m,u in r):raise SystemExit('GPU not verified idle')
PY
export VLLM_USE_FLASHINFER_SAMPLER=0
export LD_LIBRARY_PATH="/root/miniconda3/lib/python3.12/site-packages/nvidia/cu13/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$PWD/code${PYTHONPATH:+:$PYTHONPATH}"
tar -czf "runs/$RUN/code.tar.gz" code/*.py code/*.sh
# The timeout applies only to this new process group, never an existing task.
timeout --signal=TERM --kill-after=30 600 "$PYTHON" code/smoke_vllm_quality.py --model "$MODEL" --output "runs/$RUN/smoke.json"
echo COMPLETE
