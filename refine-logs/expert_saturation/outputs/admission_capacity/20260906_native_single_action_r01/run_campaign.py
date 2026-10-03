"""Run the frozen one-shot action block in two sequential fresh engines."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path.cwd()
results = root / "results"
results.mkdir(exist_ok=False)
env = dict(os.environ, HF_HUB_CACHE="/root/autodl-tmp/hf-cache/hub", HF_HUB_OFFLINE="1",
           TRANSFORMERS_OFFLINE="1", CUDA_VISIBLE_DEVICES="0", VLLM_ENABLE_V1_MULTIPROCESSING="0",
           VLLM_USE_FLASHINFER_SAMPLER="0")
state = dict(status="STARTED", queue_pid=os.getpid(), completed=[])


def save():
    temporary = results / "queue_status.tmp"
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(results / "queue_status.json")


save()
for label, reverse in (("forward", False), ("reverse", True)):
    cmd = [sys.executable, "-u", "run_native_capacity.py", "--prepared-dir", "prepared",
           "--output-dir", str(results / label), "--caps", "8,12,16,32",
           "--engine-max-seqs", "32", "--arrival-scales", "1", "--repeats", "1",
           "--single-action-probe", "--ttft-slo-s", "0.20", "--tpot-slo-s", "0.009",
           "--warmup-condition-rounds", "1"]
    if reverse:
        cmd.append("--reverse-conditions")
    with (results / f"{label}.console.log").open("x") as log:
        state.update(status="RUNNING", label=label, started_unix_s=time.time())
        child = subprocess.Popen(cmd, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
        state["child_pid"] = child.pid
        save()
        code = child.wait()
    state.update(child_exit_code=code, finished_unix_s=time.time())
    if code:
        state["status"] = "STOPPED_AFTER_CHILD_FAILURE"
        save()
        sys.exit(code)
    state["completed"].append(label)
state["status"] = "COMPLETE"
save()
