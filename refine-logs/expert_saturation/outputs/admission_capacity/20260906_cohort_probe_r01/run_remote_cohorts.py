"""Launch the two declared cohorts sequentially from the exported repo root."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path.cwd()
base = root / "refine-logs/expert_saturation/outputs/admission_capacity/20260906_cohort_probe_r01"
results = root / "results"
results.mkdir(exist_ok=False)
env = dict(os.environ, HF_HUB_CACHE="/root/autodl-tmp/hf-cache/hub",
           HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", CUDA_VISIBLE_DEVICES="0")
state = dict(status="STARTED", queue_pid=os.getpid(), completed_offsets=[])


def save():
    temporary = results / "queue_status.tmp"
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(results / "queue_status.json")


save()
for offset in (16, 32):
    command = [sys.executable, "-u",
               "refine-logs/expert_saturation/experiments/admission_capacity/run_capacity.py",
               "--prepared-dir", str(base / f"offset{offset}_inputs"),
               "--output-dir", str(results / f"offset{offset}")]
    with (results / f"offset{offset}.console.log").open("x") as log:
        child = subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
        state.update(status="RUNNING", offset=offset, child_pid=child.pid, started_unix_s=time.time())
        save()
        code = child.wait()
    state.update(child_exit_code=code, finished_unix_s=time.time())
    if code:
        state["status"] = "STOPPED_AFTER_CHILD_FAILURE"
        save()
        sys.exit(code)
    state["completed_offsets"].append(offset)
state["status"] = "COMPLETE"
save()
