#!/bin/bash
set -u
BASE=/root/d6-action-state-20260914-r01
V=/root/autodl-tmp/expert-saturation/vllm-0.26/bin/python
export HF_HOME=/root/autodl-tmp/hf-cache HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0
cd "$BASE/pkg" || exit 1
mkdir -p "$BASE/results"
for label in repeat0 repeat1; do
 test ! -e "$BASE/results/$label" || exit 91
 pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader) || exit 92
 test -z "$pids" || { echo "BLOCKED $pids"; exit 93; }
 echo "START $(date -u +%FT%TZ) $label"
 timeout 600 "$V" -u run_probe.py --domain long --reservation-policy full --completion-policy rotate --victim-order least_progress --cap 32 --gpu-memory-utilization 0.9 --kv-cache-bytes 13960740864 --output-dir "$BASE/results/$label" > "$BASE/results/$label.log" 2>&1
 rc=$?; echo "END $(date -u +%FT%TZ) $label exit=$rc"
 test "$rc" = 0 || exit "$rc"
 sleep 2
done
echo "CAMPAIGN_FINISHED $(date -u +%FT%TZ)"
