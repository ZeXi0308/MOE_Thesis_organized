#!/bin/bash
set -u
BASE=/root/psweep-review-westc-r02
V=/root/autodl-tmp/expert-saturation/vllm-0.26/bin/python
export HF_HOME=/root/autodl-tmp/hf-cache HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0
cd "$BASE/pkg" || exit 1
mkdir -p "$BASE/results"
exec 9>"$BASE/campaign.lock"
flock -n 9 || exit 90
run() {
 label=$1; policy=$2; kv=$3
 if [ -e "$BASE/results/$label" ]; then echo "EXISTS $label; refusing restart"; exit 91; fi
 pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader) || exit 92
 if [ -n "$pids" ]; then echo "BLOCKED_BEFORE_GPU_INITIALIZATION $pids"; exit 93; fi
 echo "START $(date -u +%FT%TZ) $label"
 timeout 600 "$V" -u run_probe.py --domain long --reservation-policy full --completion-policy "$policy" --cap 32 --gpu-memory-utilization 0.9 --kv-cache-bytes "$kv" --output-dir "$BASE/results/$label" > "$BASE/results/$label.log" 2>&1
 rc=$?
 echo "END $(date -u +%FT%TZ) $label exit=$rc"
 sleep 2
}
run block0-d0-native native 17181966336
run block0-d0-rotate rotate 17181966336
run block0-d2-native native 16089350144
run block0-d2-rotate rotate 16089350144
run block0-d4-native native 15034482688
run block0-d4-rotate rotate 15034482688
run block0-d6-native native 13960740864
run block0-d6-rotate rotate 13960740864
run block1-d6-rotate rotate 13960740864
run block1-d6-native native 13960740864
run block1-d4-rotate rotate 15034482688
run block1-d4-native native 15034482688
run block1-d2-rotate rotate 16089350144
run block1-d2-native native 16089350144
run block1-d0-rotate rotate 17181966336
run block1-d0-native native 17181966336
echo "CAMPAIGN_FINISHED $(date -u +%FT%TZ)"
