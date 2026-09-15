#!/usr/bin/env bash
set -euo pipefail
export HF_HOME=/root/autodl-tmp/hf-cache HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0
export CUDA_VISIBLE_DEVICES=GPU-4015b79d-bed6-3a4b-9d2b-0c17be96d0e5
cd -- "$(dirname -- "$0")"
exec 9>/root/autodl-tmp/moe-research-gpu.lock
flock -n 9 || { echo ABORT_GPU_GROUP_LOCK_BUSY; exit 75; }
py=/root/autodl-tmp/expert-saturation/vllm-0.26/bin/python
"$py" preflight.py
for cell in block0-native_full_native block0-current block0-eager block1-eager block1-current block1-native_full_native; do
  variant=${cell#*-}
  "$py" run_recovery_cadence.py --inputs inputs --warmup-inputs warmups --variant "$variant" --measurement-mode performance --output-dir "../results/$cell"
done
