#!/bin/bash
set -euo pipefail
cd /root/autodl-tmp/moe-capacity-20260906
export HF_HOME=/root/autodl-tmp/hf-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1
for capacity_trial in a0_hold b0_pulse b1_pulse a1_hold; do
  capacity_arm=${capacity_trial#*_}
  /root/autodl-tmp/longrun-b-venv/bin/python -u refine-logs/expert_saturation/experiments/admission_capacity/run_capacity.py --prepared-dir "${capacity_arm}_inputs" --output-dir "/root/autodl-tmp/moe-capacity-20260906-actions-repeat/${capacity_trial}"
done
