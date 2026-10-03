#!/usr/bin/env bash
set -euo pipefail
export HF_HOME=/root/autodl-tmp/hf-cache HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0
export CUDA_VISIBLE_DEVICES=GPU-4015b79d-bed6-3a4b-9d2b-0c17be96d0e5
cd -- "$(dirname -- "$0")"
exec 9>/root/autodl-tmp/moe-research-gpu.lock
flock -n 9 || { echo ABORT_GPU_GROUP_LOCK_BUSY; exit 75; }
py=/root/autodl-tmp/expert-saturation/vllm-0.26/bin/python
"$py" preflight.py
"$py" run_ltr_style.py --inputs inputs --warmup-inputs warmups --ltr-threshold 30 --ltr-quantum 10 --output-dir ../results/diagnostic-ltr-t30-q10
