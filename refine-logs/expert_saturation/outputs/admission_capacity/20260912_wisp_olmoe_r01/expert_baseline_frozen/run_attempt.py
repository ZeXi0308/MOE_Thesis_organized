"""Execute one retained qualification or ABBA block; never retry a failed cell."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parent
kind = sys.argv[1]
if kind not in ("qualification", "performance"):
    raise ValueError("choose qualification or performance")
out = root / kind
out.mkdir(exist_ok=False)
record = dict(status="RUNNING", kind=kind, started_unix_s=time.time(), cells=[])


def save():
    (out / "execution.json").write_text(json.dumps(record, indent=2) + "\n")


save()
try:
    plans = [("expert_validation", "expert")] if kind == "qualification" else [
        ("token_r0", "token"), ("expert_r0", "expert"),
        ("expert_r1", "expert"), ("token_r1", "token")]
    for label, execution in plans:
        cmd = [sys.executable, "-u", str(root / "source/run_native_pager.py"),
               "--output", str(out / label), "--expert-cap", "24",
               "--execution", execution, "--warmup-full", "--warmup-executions",
               "--prepared", "/root/autodl-tmp/expert-union/inputs_preparation/prepared/short",
               "--model", "/root/autodl-tmp/hf-cache/hub/models--allenai--OLMoE-1B-7B-0924/snapshots/6d84c48581ece794365f2b8e9cfb043c68ade9c5"]
        if kind == "qualification":
            cmd.append("--verify-kernel")
        cell = dict(label=label, command=cmd, cpu_affinity=list(range(8)),
                    started_unix_s=time.time(), status="RUNNING")
        record["cells"].append(cell)
        save()
        with (out / (label + ".log")).open("x") as log:
            result = subprocess.run(["taskset", "-c", "0-7", "timeout", "-s", "TERM", "600"] + cmd,
                env=dict(os.environ, OMP_NUM_THREADS="8", TOKENIZERS_PARALLELISM="false",
                         PYTHONUNBUFFERED="1", PYTHONPATH="/root/autodl-tmp/wisp-pager-smoke-20260912/source/src:" + str(root / "source")),
                stdout=log, stderr=subprocess.STDOUT)
        cell.update(returncode=result.returncode, finished_unix_s=time.time())
        if (out / label / "status.json").exists():
            cell["terminal"] = json.loads((out / label / "status.json").read_text())
        cell["status"] = "COMPLETE" if result.returncode == 0 else "FAILED"
        save()
        if result.returncode:
            raise RuntimeError(f"{label} failed; diagnose retained attempt before another run")
    record["status"] = "COMPLETE"
except BaseException as exc:
    record.update(status="STOPPED", error=f"{type(exc).__name__}: {exc}")
    raise
finally:
    record["finished_unix_s"] = time.time()
    save()
