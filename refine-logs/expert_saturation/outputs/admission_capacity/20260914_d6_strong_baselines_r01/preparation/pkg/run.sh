#!/bin/bash
set -u
BASE=/root/d6-strong-baselines-20260914-r01
V=/root/autodl-tmp/expert-saturation/vllm-0.26/bin/python
export HF_HOME=/root/autodl-tmp/hf-cache HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0
cd "$BASE/pkg" || exit 1
mkdir -p "$BASE/results"
exec 9>"$BASE/campaign.lock"
flock -n 9 || exit 90
run() {
 label=$1; policy=$2; victim=$3
 if [ -e "$BASE/results/$label" ]; then echo "EXISTS $label"; exit 91; fi
 pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader) || exit 92
 if [ -n "$pids" ]; then echo "BLOCKED_BEFORE_GPU_INITIALIZATION $pids"; exit 93; fi
 echo "START $(date -u +%FT%TZ) $label"
 timeout 600 "$V" -u run_probe.py --domain long --reservation-policy full --completion-policy "$policy" --victim-order "$victim" --cap 32 --gpu-memory-utilization 0.9 --kv-cache-bytes 13960740864 --output-dir "$BASE/results/$label" > "$BASE/results/$label.log" 2>&1
 rc=$?
 echo "END $(date -u +%FT%TZ) $label exit=$rc"
 if [ "$rc" -ne 0 ]; then echo CAMPAIGN_STOPPED_FAILURE; exit "$rc"; fi
 sleep 2
}
run block0-d6-native native least_progress
run block0-d6-headroom headroom least_progress
run block0-d6-least_progress rotate least_progress
run block0-d6-most_output rotate most_output
run block1-d6-most_output rotate most_output
run block1-d6-least_progress rotate least_progress
run block1-d6-headroom headroom least_progress
run block1-d6-native native least_progress
echo "CAMPAIGN_FINISHED $(date -u +%FT%TZ)"
