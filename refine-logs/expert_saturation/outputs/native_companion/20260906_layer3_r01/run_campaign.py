"""Two sequential fresh native engines; preserve the captured fixture."""
from pathlib import Path
import json
import os
import subprocess
import sys

root = Path.cwd()
results = root / "results"
results.mkdir(exist_ok=False)
env = dict(os.environ, HF_HUB_CACHE="/root/autodl-tmp/hf-cache/hub", HF_HUB_OFFLINE="1",
           TRANSFORMERS_OFFLINE="1", CUDA_VISIBLE_DEVICES="0", VLLM_ENABLE_V1_MULTIPROCESSING="0",
           VLLM_USE_FLASHINFER_SAMPLER="0")
state = dict(status="STARTED", pid=os.getpid(), completed=[])
for process in range(2):
    label = f"process-{process}"
    command = [sys.executable, "-u", "run_native_companion_probe.py", "--config", "config.json",
        "--inputs", "inputs.json", "--fixture", str(results / "fixture.pt"),
        "--output-dir", str(results / label)]
    with (results / f"{label}.log").open("x") as log:
        child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        state.update(status="RUNNING", label=label, child_pid=child.pid)
        (results / "campaign_status.json").write_text(json.dumps(state, indent=2) + "\n")
        code = child.wait()
    state.update(exit_code=code)
    if code:
        state["status"] = "STOPPED_AFTER_FAILURE"
        (results / "campaign_status.json").write_text(json.dumps(state, indent=2) + "\n")
        sys.exit(code)
    state["completed"].append(label)
state["status"] = "COMPLETE"
(results / "campaign_status.json").write_text(json.dumps(state, indent=2) + "\n")
