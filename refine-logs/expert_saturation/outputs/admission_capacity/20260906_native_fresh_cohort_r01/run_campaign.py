"""Run the declared native A/B/B/A processes sequentially; stop on failure."""
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
    temp = results / "queue_status.tmp"
    temp.write_text(json.dumps(state, indent=2) + "\n")
    temp.replace(results / "queue_status.json")


save()
for label, cap in (("a0_cap6", 6), ("b0_cap8", 8), ("b1_cap8", 8), ("a1_cap6", 6)):
    cmd = [sys.executable, "-u", "run_native_capacity.py", "--prepared-dir", "prepared",
           "--output-dir", str(results / label), "--cap", str(cap),
           "--engine-max-seqs", "8", "--arrival-scales", "0.02",
           "--ttft-slo-s", "0.20", "--tpot-slo-s", "0.009", "--warmup-condition-rounds", "1"]
    with (results / f"{label}.console.log").open("x") as log:
        child = subprocess.Popen(cmd, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
        state.update(status="RUNNING", label=label, child_pid=child.pid, started_unix_s=time.time())
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
