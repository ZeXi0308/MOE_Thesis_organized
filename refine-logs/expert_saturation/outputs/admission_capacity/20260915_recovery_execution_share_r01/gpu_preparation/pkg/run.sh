#!/bin/bash
set -eu
PYTHON_BIN=${1:?Pass the existing vLLM environment Python path}
BASE=$(cd -- "$(dirname -- "$0")/.." && pwd)
cd "$BASE/pkg"
export HF_HOME=/root/autodl-tmp/hf-cache HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0
sha256sum -c SHA256SUMS
export CUDA_VISIBLE_DEVICES=GPU-4015b79d-bed6-3a4b-9d2b-0c17be96d0e5
[ "$(nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader)" = "$CUDA_VISIBLE_DEVICES" ] || exit 94
exec 9>/root/autodl-tmp/moe-research-gpu.lock
flock -n 9 || exit 90
if [ -e "$BASE/results" ]; then echo REFUSE_EXISTING_RESULTS; exit 91; fi
mkdir "$BASE/results"
run() {
 label=$1; variant=$2
 pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader) || exit 92
 if [ -n "$pids" ]; then echo "BLOCKED_BEFORE_GPU_INITIALIZATION $pids"; exit 93; fi
 echo "START $(date -u +%FT%TZ) $label"
 timeout 600 "$PYTHON_BIN" -u run_probe.py --variant "$variant" --domain heterogeneous --reservation-policy full --cap 32 --gpu-memory-utilization 0.9 --kv-cache-bytes 13960740864 --output-dir "$BASE/results/$label" > "$BASE/results/$label.log" 2>&1
 echo "END $(date -u +%FT%TZ) $label"
}
run recovery-block0-most_output most_output
run recovery-block0-least_feasible least_feasible
run recovery-block0-least_feasible_native_guard least_feasible_native_guard
run recovery-block1-least_feasible_native_guard least_feasible_native_guard
run recovery-block1-least_feasible least_feasible
run recovery-block1-most_output most_output
echo "CAMPAIGN_FINISHED $(date -u +%FT%TZ)"
