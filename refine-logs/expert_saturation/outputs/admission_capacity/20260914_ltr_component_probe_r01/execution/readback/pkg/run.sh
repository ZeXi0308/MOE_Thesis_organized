#!/bin/bash
set -eu
PYTHON_BIN=${1:?Pass the existing vLLM environment Python path}
BASE=$(cd -- "$(dirname -- "$0")/.." && pwd)
cd "$BASE/pkg"
export HF_HOME=/root/autodl-tmp/hf-cache HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0
sha256sum -c SHA256SUMS
exec 9>/root/autodl-tmp/moe-research-gpu.lock
flock -n 9 || exit 90
if [ -e "$BASE/results" ]; then echo REFUSE_EXISTING_RESULTS; exit 91; fi
mkdir "$BASE/results"
run() {
 label=$1; boost=$2
 pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader) || exit 92
 if [ -n "$pids" ]; then echo "BLOCKED_BEFORE_GPU_INITIALIZATION $pids"; exit 93; fi
 echo "START $(date -u +%FT%TZ) $label"
 timeout 600 "$PYTHON_BIN" -u run_probe.py --domain long --reservation-policy full --completion-policy native --cap 32 --gpu-memory-utilization 0.9 --kv-cache-bytes 13960740864 --fairness-boost "$boost" --output-dir "$BASE/results/$label" > "$BASE/results/$label.log" 2>&1
 echo "END $(date -u +%FT%TZ) $label"
}
run block0-d6-boost-off off
run block0-d6-boost-on on
run block1-d6-boost-on on
run block1-d6-boost-off off
echo "CAMPAIGN_FINISHED $(date -u +%FT%TZ)"
