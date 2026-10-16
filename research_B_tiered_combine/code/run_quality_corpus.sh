#!/usr/bin/env bash
# One bounded waiter and one quality run; all input/output stays in this project.
set -euo pipefail
cd "$(dirname "$0")/.."
run_name=${1:?Usage: run_quality_corpus.sh unique_run evaluate_quality_args...}
shift
[[ "$run_name" =~ ^[A-Za-z0-9_-]+$ ]] || exit 2
mkdir "runs/$run_name"
exec > >(tee "runs/$run_name/driver.log") 2>&1
trap 'rc=$?; echo EXIT_CODE=$rc; date -u +FINISHED=%Y-%m-%dT%H:%M:%SZ; echo "$rc" > "runs/$run_name/exit_code"' EXIT
date -u +STARTED=%Y-%m-%dT%H:%M:%SZ
exec 9>/root/autodl-tmp/moe-research-gpu.lock
echo WAITING_FOR_SHARED_LOCK
flock -w 600 9 || exit 3
echo LOCK_ACQUIRED
/root/miniconda3/bin/python - <<'PY'
import subprocess,time
for attempt in range(3):
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    print(raw,flush=True)
    rows=[list(map(int,s.split(','))) for s in raw.strip().splitlines()]
    if len(rows)==1 and not apps and all(m<=64 and u==0 for m,u in rows):break
    if apps or any(m>64 for m,u in rows):raise SystemExit('GPU occupied')
    if attempt==2:raise SystemExit('GPU utilization did not settle')
    time.sleep(2)
PY
export VLLM_USE_FLASHINFER_SAMPLER=0
export LD_LIBRARY_PATH="/root/miniconda3/lib/python3.12/site-packages/nvidia/cu13/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$PWD/code${PYTHONPATH:+:$PYTHONPATH}"
tar -czf "runs/$run_name/code.tar.gz" code/*.py code/*.sh
printf '%q ' "$@" > "runs/$run_name/arguments.txt"
timeout --signal=TERM --kill-after=30 3600 /root/miniconda3/bin/python code/evaluate_quality.py \
  --output "runs/$run_name/evaluation" "$@"
echo COMPLETE
