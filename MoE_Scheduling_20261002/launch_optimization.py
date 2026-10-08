"""Run the declared A/B/C/C/B/A optimization group under the shared whole-group lock."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parent
results = root / "results"
results.mkdir(exist_ok=False)
lock = open("/root/autodl-tmp/moe-research-gpu.lock", "a")
try:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    (results / "group_status.json").write_text(json.dumps({"status": "ABORT_LOCK_BUSY"}))
    raise SystemExit(73)

def status(**fields):
    (results / "group_status.json").write_text(json.dumps(fields, indent=2) + "\n")

def gpu_check():
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
                         text=True, capture_output=True, check=True)
    procs = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                           text=True, capture_output=True, check=True)
    if gpu.stdout.strip() != "GPU-3fc910c2-bf65-5273-e6b5-6c0d8b6ce03e" or procs.stdout.strip():
        raise RuntimeError("GPU identity differs or occupied: " + gpu.stdout + procs.stdout)

started = time.time()
env = os.environ.copy()
env.update(PYTHONPATH=str(root / "vendor/WiSP/src"),
           WISP_PLUGIN_DISABLE="1", WISP_PREFETCH="0", WISP_DYNAMIC="0",
           WISP_COOCCUR_ONLINE="0", WISP_COOCCUR_PATH="",
           VLLM_ENABLE_V1_MULTIPROCESSING="0", VLLM_USE_FLASHINFER_SAMPLER="0",
           HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
completed = []
try:
    for index, arm in enumerate(("default24", "serial12", "overlap12", "overlap12", "serial12", "default24")):
        gpu_check()
        name = f"{index:02d}_{arm}"
        command = [sys.executable, str(root / "optimization_src/run_native_pager.py"),
            "--output", str(results / name), "--prepared", str(root / "inputs/olmoe_gsm8k16"),
            "--model", str(root / "model"), "--expert-cap", "24",
            "--execution", "expert", "--group-retention", "none",
            "--prompt-tokens", "0", "--output-tokens", "32", "--requests", "16",
            "--token-budget", "512", "--kv-bytes", "1073741824",
            "--arrival-interval", "0", "--max-seconds", "300", "--warmup-full"]
        if arm != "default24":
            command += ["--expert-group-size", "12"]
        if arm == "overlap12":
            command += ["--group-overlap"]
        status(status="RUNNING", arm=name, controller_pid=os.getpid(),
               started_unix_s=started, completed=completed, command=command)
        with (results / (name + ".log")).open("x") as log:
            run = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                 timeout=600)
        if run.returncode:
            raise RuntimeError(name + " exited " + str(run.returncode))
        completed.append(name)
    status(status="COMPLETE", started_unix_s=started, finished_unix_s=time.time(), completed=completed)
except BaseException as exc:
    status(status="FAILED", started_unix_s=started, finished_unix_s=time.time(),
           completed=completed, error=repr(exc))
    raise
finally:
    fcntl.flock(lock, fcntl.LOCK_UN)
    lock.close()
