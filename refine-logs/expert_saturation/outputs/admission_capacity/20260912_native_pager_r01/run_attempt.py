"""Run this archived integration sequence once; preserve failures, never retry."""
import json
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parent
out = root / "results"
out.mkdir(exist_ok=False)
record = dict(status="RUNNING", started_unix_s=time.time(), cells=[])


def save():
    (out / "execution.json").write_text(json.dumps(record, indent=2) + "\n")


save()
try:
    plans_path = root / "run_cells.json"
    plans = json.loads(plans_path.read_text()) if plans_path.exists() else [
        dict(label=f"cap{cap}", cap=cap, extra=[]) for cap in (64, 24)]
    for plan in plans:
        cap, label = plan["cap"], plan["label"]
        cmd = [sys.executable, "-u", str(root / "source/run_native_pager.py"),
               "--output", str(out / label), "--expert-cap", str(cap),
               "--prepared", "/root/autodl-tmp/expert-union/inputs_preparation/prepared/short",
               "--model", "/root/autodl-tmp/hf-cache/hub/models--allenai--OLMoE-1B-7B-0924/snapshots/6d84c48581ece794365f2b8e9cfb043c68ade9c5", *plan.get("extra", [])]
        cell = dict(label=label, command=cmd, started_unix_s=time.time(), status="RUNNING")
        record["cells"].append(cell)
        save()
        with (out / (label + ".log")).open("x") as log:
            result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=600)
        cell.update(returncode=result.returncode, finished_unix_s=time.time())
        if (out / label / "status.json").exists():
            cell["terminal"] = json.loads((out / label / "status.json").read_text())
        cell["status"] = "COMPLETE" if result.returncode == 0 else "FAILED"
        save()
        if result.returncode:
            raise RuntimeError(f"{label} failed; inspect retained attempt before fixing")
    record["status"] = "COMPLETE"
except BaseException as exc:
    record.update(status="STOPPED", error=f"{type(exc).__name__}: {exc}")
    raise
finally:
    record["finished_unix_s"] = time.time()
    save()
