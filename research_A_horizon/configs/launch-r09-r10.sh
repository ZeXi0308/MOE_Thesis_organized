#!/bin/bash
# One-shot continuation; existing run_group owns the public GPU lock per group.
set -eu
task_root=/root/autodl-tmp/research_A_horizon
task_python=/root/miniconda3/bin/python
trap 'printf "queue exit=%s at " "$?"; date -u' EXIT
printf 'Waiting for asset process 12459 at '; date -u
while [ -d /proc/12459 ]; do sleep 15; done
"$task_python" -c 'import json; from pathlib import Path; p=Path("/root/autodl-tmp/research_A_horizon/models/qwen-assets.json"); r=json.loads(p.read_text()); assert r["status"]=="COMPLETE", r["status"]'
printf 'Starting r09 at '; date -u
"$task_python" -B "$task_root/code-r12/run_group.py" --plan "$task_root/configs/w2-24-calibration-r09.json" --wait-lock-seconds 1800
printf 'Starting r10 at '; date -u
"$task_python" -B "$task_root/code-r12/run_group.py" --plan "$task_root/configs/w2-48-ablations-r10.json" --wait-lock-seconds 1800
printf 'Both groups completed at '; date -u
