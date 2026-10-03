#!/bin/bash
set -euo pipefail
cd /root/autodl-tmp/moe-step-action-20260906
export HF_HOME=/root/autodl-tmp/hf-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=25 MKL_NUM_THREADS=25
for capacity_trial in a0_c0_hold a0_c32_hold b0_c32_up b0_c0_up b1_c0_up b1_c32_up a1_c32_hold a1_c0_hold; do
  capacity_input=${capacity_trial#*_}
  /root/autodl-tmp/longrun-b-venv/bin/python -u refine-logs/expert_saturation/experiments/admission_capacity/run_capacity.py --prepared-dir "refine-logs/expert_saturation/outputs/admission_capacity/20260906_step_action_r01/${capacity_input}_inputs" --output-dir "/root/autodl-tmp/moe-step-action-20260906-results-r02/${capacity_trial}"
done
