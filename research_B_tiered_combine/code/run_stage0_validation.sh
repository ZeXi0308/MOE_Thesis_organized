#!/usr/bin/env bash
# One bounded waiter, then one exclusive validation group; never kills peers.
set -euo pipefail
cd "$(dirname "$0")/.."
RUN=${1:?Usage: run_stage0_validation.sh unique_run model_path}
MODEL=${2:?model_path required}
[[ "$RUN" =~ ^[A-Za-z0-9_-]+$ ]] || exit 2
mkdir "runs/$RUN"
exec > >(tee "runs/$RUN/driver.log") 2>&1
trap 'rc=$?; echo EXIT_CODE=$rc; date -u +FINISHED=%Y-%m-%dT%H:%M:%SZ; echo "$rc" > "runs/$RUN/exit_code"' EXIT
date -u +STARTED=%Y-%m-%dT%H:%M:%SZ
echo WAITING_FOR_SHARED_LOCK
exec 9>/root/autodl-tmp/moe-research-gpu.lock
flock -w 600 9 || { echo 'Lock wait deadline reached; no GPU work started'; exit 3; }
echo LOCK_ACQUIRED
PYTHON=/root/miniconda3/bin/python
"$PYTHON" - <<'PY'
import subprocess,time
# Utilization can contain the preceding completed task's last sample.
for attempt in range(3):
    q=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    print(q,flush=True);r=[list(map(int,s.split(','))) for s in q.strip().splitlines()]
    if len(r)==1 and not apps and all(m<=64 and u==0 for m,u in r):break
    if apps or any(m>64 for m,u in r):raise SystemExit('GPU has active memory/processes')
    if attempt==2:raise SystemExit('GPU utilization did not settle')
    time.sleep(2)
PY
export VLLM_USE_FLASHINFER_SAMPLER=0
export LD_LIBRARY_PATH="/root/miniconda3/lib/python3.12/site-packages/nvidia/cu13/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$PWD/code${PYTHONPATH:+:$PYTHONPATH}"
tar -czf "runs/$RUN/code.tar.gz" code/*.py code/*.sh
smoke_rc=0
timeout --signal=TERM --kill-after=30 600 "$PYTHON" code/smoke_vllm_quality.py --model "$MODEL" --output "runs/$RUN/smoke.json" || smoke_rc=$?
echo "SMOKE_EXIT=$smoke_rc"
bench_rc=0
timeout --signal=TERM --kill-after=30 300 "$PYTHON" code/bench_codec.py --output "runs/$RUN/bench.json" || bench_rc=$?
echo "BENCH_EXIT=$bench_rc"
if [[ $smoke_rc -ne 0 || $bench_rc -ne 0 ]]; then exit 1; fi
if [[ -n ${3:-} ]]; then
    timeout --signal=TERM --kill-after=30 300 "$PYTHON" code/evaluate_quality.py \
      --model "$3" --articles runs/quality_articles_r02/validation.jsonl \
      --output "runs/$RUN/corpus_integration" --layers 16 --top-k 16 --hidden 512 \
      --context 64 --limit-docs 1 --max-article-tokens 64 --purpose integration
fi
echo COMPLETE
