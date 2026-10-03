#!/usr/bin/env bash
set -euo pipefail
export HF_HOME=/root/autodl-tmp/hf-cache HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0
cd -- "$(dirname -- "$0")"
exec 9>/root/autodl-tmp/moe-research-gpu.lock
flock -n 9 || { echo ABORT_GPU_GROUP_LOCK_BUSY; exit 75; }
py=/root/autodl-tmp/expert-saturation/vllm-0.26/bin/python
"$py" preflight.py
for arm in off on; do
  "$py" run_probe.py --domain long --reservation-policy full --completion-policy native --cap 32 --gpu-memory-utilization 0.90 --kv-cache-bytes 13960740864 --selective-save "$arm" --offload-gib 16 --output-dir "../results/save-$arm"
done
