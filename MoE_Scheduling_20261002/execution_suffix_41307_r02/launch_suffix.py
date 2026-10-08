"""Small full-request suffix prototype comparison; all engines use identical resources."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--static-cap", type=int, required=True, choices=(8, 10, 16))
parser.add_argument("--arms", default="n2off,n2fixed0", help="comma separated arms, executed in stated order")
parser.add_argument("--output-name", default="results_suffix_r01")
parser.add_argument("--suffix-layer", type=int, default=0)
args = parser.parse_args()
arm_specs = {"ar16": (0,"off"), **{f"n{n}{p}": (n,p) for n in (1,2,4) for p in ("off","fixed0","fixed1","conservative","horizon")}}
order = args.arms.split(",")
if any(a not in arm_specs for a in order):
    parser.error("unknown arm")
root = Path(__file__).resolve().parent
results = root / args.output_name
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
    if gpu.stdout.strip() != "GPU-273c48a6-cbbb-f4e4-ae0e-78f1876a0f51" or procs.stdout.strip():
        raise RuntimeError("GPU identity differs or occupied: " + gpu.stdout + procs.stdout)

started = time.time()
env = os.environ.copy()
env.update(PYTHONPATH=str(root / "vendor/WiSP/src"),
           WISP_PLUGIN_DISABLE="1", WISP_PREFETCH="0", WISP_DYNAMIC="0",
           WISP_COOCCUR_ONLINE="0", WISP_COOCCUR_PATH="",
           VLLM_ENABLE_V1_MULTIPROCESSING="0", VLLM_USE_FLASHINFER_SAMPLER="0",
           HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
nvidia_root = Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages/nvidia"
lib_dirs = sorted((str(p) for p in nvidia_root.rglob("lib") if p.is_dir()), key=lambda p: ("/cu13/" not in p, p))
env["LD_LIBRARY_PATH"] = ":".join(lib_dirs + [env.get("LD_LIBRARY_PATH", "")])
completed = []
try:
    for index, arm in enumerate(order):
        gpu_check()
        name = f"{index:02d}_{arm}"
        command = [sys.executable, str(root / "suffix_src/run_native_pager.py"),
            "--output", str(results / name), "--prepared", str(root / "inputs/olmoe_gsm8k_natural16"),
            "--warmup-prepared", str(root / "inputs/olmoe_gsm8k16"),
            "--admission-profile", str(root / "analysis/cost_profile.json"),
            "--model", str(root / "model"), "--expert-cap", "24",
            "--execution", "expert", "--group-retention", "none",
            "--prompt-tokens", "0", "--output-tokens", "512", "--warmup-output-tokens", "32", "--natural-eos", "--requests", "16",
            "--token-budget", "4096", "--warmup-token-budget", "512",
            "--measured-token-budget", "2048", "--kv-bytes", "1073741824",
            "--arrival-interval", "0", "--max-seconds", "600", "--warmup-full"]
        cap = 16
        command += ["--cpu-kv-gib", "1", "--ngram-speculative-tokens", str(arm_specs[arm][0]), "--suffix-policy", arm_specs[arm][1], "--suffix-layer", str(args.suffix_layer)]
        command += ["--admission-cap", str(cap), "--admission-policy",
                    "static"]
        status(status="RUNNING", arm=name, controller_pid=os.getpid(),
               started_unix_s=started, completed=completed, command=command)
        with (results / (name + ".log")).open("x") as log:
            run = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                 timeout=1200)
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
