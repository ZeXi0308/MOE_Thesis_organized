#!/usr/bin/env bash
set -euo pipefail
export HF_HOME=/root/autodl-tmp/hf-cache HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0
export CUDA_VISIBLE_DEVICES=GPU-4015b79d-bed6-3a4b-9d2b-0c17be96d0e5
cd -- "$(dirname -- "$0")"
exec 9>/root/autodl-tmp/moe-research-gpu.lock
flock -n 9 || { echo ABORT_GPU_GROUP_LOCK_BUSY; exit 75; }
py=/root/autodl-tmp/expert-saturation/vllm-0.26/bin/python
"$py" preflight.py
for cell in block0-off block0-on block1-on block1-off; do
  arm=${cell##*-}
  mode=performance
  if [[ "$cell" == diag-* ]]; then mode=diagnostic; fi
  "$py" run_probe.py --domain long --reservation-policy full --completion-policy native --cap 32 --gpu-memory-utilization 0.90 --kv-cache-bytes 13960740864 --selective-save "$arm" --offload-gib 16 --measurement-mode "$mode" --output-dir "../results/$cell"
done
