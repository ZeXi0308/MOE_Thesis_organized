#!/bin/bash
set -eu
task_root=/root/autodl-tmp/research_A_horizon
task_python=/root/miniconda3/bin/python
trap 'printf "queue exit=%s at " "$?"; date -u' EXIT
printf 'Waiting for r09/r10 queue 12998 at '; date -u
while [ -d /proc/12998 ]; do sleep 15; done
"$task_python" - <<'PY'
import json
from pathlib import Path
root=Path('/root/autodl-tmp/research_A_horizon')
for name in ['models/qwen-assets.json','runs/w2-24-calibration-r09/receipt.json',
             'runs/w2-48-ablations-r10/receipt.json']:
    record=json.loads((root/name).read_text())
    assert record['status']=='COMPLETE', (name,record['status'])
PY
"$task_python" -B "$task_root/code-r12/run_group.py" --plan "$task_root/configs/qwen-calibration-r11.json" --wait-lock-seconds 1800
