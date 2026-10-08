"""AR expert/KV static partitions at equal actual 5.5GiB combined pool bytes."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--static-cap", type=int, default=16, choices=(16,))
parser.add_argument("--output-name", default="results_partition_r01")
parser.add_argument("--wait-for-lock", action="store_true", help="Queue behind an entire group; never enter between its cells")
parser.add_argument("--prepared", default="olmoe_gsm8k_natural16")
args = parser.parse_args()
# Fixed forward/reverse order; each cell starts a separate native AR engine.
arm_specs = {"expert24": (24, 1073741824), "expert23": (23, 1275068416),
             "expert20": (20, 1879048192)}
order = ("expert24", "expert23", "expert20", "expert20", "expert23", "expert24")
root = Path(__file__).resolve().parent
results = root / args.output_name
results.mkdir(exist_ok=False)
lock = open("/root/autodl-tmp/moe-research-gpu.lock", "a")
if args.wait_for_lock:
    (results / "group_status.json").write_text(json.dumps({"status": "WAITING_FOR_GROUP_LOCK", "controller_pid": os.getpid(), "queued_unix_s": time.time()}))
try:
    fcntl.flock(lock, fcntl.LOCK_EX | (0 if args.wait_for_lock else fcntl.LOCK_NB))
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
        command = [sys.executable, str(root / "partition_src/run_native_pager.py"),
            "--output", str(results / name), "--prepared", str(root / "inputs" / args.prepared),
            "--warmup-prepared", str(root / "inputs/olmoe_gsm8k16"),
            "--admission-profile", str(root / "analysis/cost_profile.json"),
            "--model", str(root / "model"), "--expert-cap", str(arm_specs[arm][0]),
            "--execution", "expert", "--group-retention", "none",
            "--prompt-tokens", "0", "--output-tokens", "512", "--warmup-output-tokens", "32", "--natural-eos", "--requests", "16",
            "--token-budget", "4096", "--warmup-token-budget", "512",
            "--measured-token-budget", "2048", "--kv-bytes", str(arm_specs[arm][1]),
            "--arrival-interval", "0", "--max-seconds", "600", "--warmup-full"]
        cap = args.static_cap
        command += ["--cpu-kv-gib", "1", "--ngram-speculative-tokens", "0", "--suffix-policy", "off"]
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
