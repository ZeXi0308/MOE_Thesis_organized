#!/bin/bash
set -eu
PYTHON_BIN=${1:?Pass installed vLLM Python}
BASE=$(cd -- "$(dirname -- "$0")/.." && pwd)
cd "$BASE/pkg"
export HF_HOME=/root/autodl-tmp/hf-cache HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0
sha256sum -c SHA256SUMS
exec 9>/root/autodl-tmp/moe-research-gpu.lock
flock -n 9 || exit 90
test ! -e "$BASE/results"
mkdir "$BASE/results"
run() {
 label=$1; variant=$2
 pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader) || exit 92
 test -z "$pids" || exit 93
 echo "START $(date -u +%FT%TZ) $label"
 "$PYTHON_BIN" -u host_budget_observed.py --parent-cgroup /sys/fs/cgroup --expected-parent-memory-max 96636764160 --declared-host-kv-offload-bytes 0 --output "$BASE/results/$label-host" --interval-s 1 -- timeout --kill-after=30 600 "$PYTHON_BIN" -u run_streaming_recovery.py --inputs inputs --warmup-inputs warmups --variant "$variant" --output-dir "$BASE/results/$label"
 echo "END $(date -u +%FT%TZ) $label"
}
run stream-block0-native native
run stream-block0-most_output most_output
run stream-block1-most_output most_output
run stream-block1-native native
echo "CAMPAIGN_FINISHED $(date -u +%FT%TZ)"
