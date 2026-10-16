#!/bin/bash
# One-shot continuation after the existing Qwen calibration controller.
set -eu
task_root=/root/autodl-tmp/research_A_horizon
task_python=/root/miniconda3/bin/python
printf 'Waiting for Qwen controller 29079 at '; date -u
while [ -d /proc/29079 ]; do sleep 15; done
"$task_python" - "$task_root" <<'PY'
import hashlib,json,sys
from pathlib import Path
p=Path(sys.argv[1])
for name in ('w2-48-ablations-r10','qwen-calibration-r11'):
    r=json.loads((p/'runs'/name/'receipt.json').read_text())
    assert r['status']=='COMPLETE',(name,r['status'])
plan=json.loads((p/'configs/olmoe24-w2-corrected-frontier.json').read_text())
assert hashlib.sha256(Path(plan['inputs']).read_bytes()).hexdigest()==plan['provenance']['inputs_sha256']
assert not Path(plan['session']).exists(),'Session already exists; do not repeat'
PY
printf 'Starting OLMoE24 W2 frontier at '; date -u
exec "$task_python" -B "$task_root/code-r12/run_group.py" \
  --plan "$task_root/configs/olmoe24-w2-corrected-frontier.json" --wait-lock-seconds 1800
