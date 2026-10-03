"""Read-only nvidia-smi preflight. Persist every decision; ABORT exits with 91."""
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time


def check_gpu(path, *, allow_pid=None):
    """Only this calling process may be exempted for persistent-engine episodes."""
    utc = lambda: datetime.now(timezone.utc).isoformat()
    record = dict(start_utc=utc(), start_unix=time.time(), start_pid=os.getpid(),
                  allow_pid=allow_pid, status="CHECKING", gpus=[], compute_processes=[], queries=[], errors=[])
    if allow_pid is not None and (type(allow_pid) is not int or allow_pid != os.getpid()):
        record["errors"].append("allow_pid may only equal the calling process PID")
    else:
        specs = (
            ("gpus", "--query-gpu", "index,uuid,name,pstate,memory.used,memory.total,utilization.gpu,temperature.gpu"),
            ("compute_processes", "--query-compute-apps", "gpu_uuid,pid,process_name,used_gpu_memory"),
        )
        for key, flag, fields in specs:
            command = ["nvidia-smi", flag + "=" + fields, "--format=csv,noheader,nounits"]
            query = dict(command=command)
            record["queries"].append(query)
            try:
                result = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
                query.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
                if result.returncode:
                    raise RuntimeError(f"nvidia-smi returned {result.returncode}")
                names = fields.split(",")
                for values in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
                    if not values:
                        continue
                    if len(values) != len(names):
                        raise RuntimeError("unexpected nvidia-smi CSV columns")
                    row = dict(zip(names, (v.strip() for v in values)))
                    if key == "compute_processes":
                        row["pid"] = int(row["pid"])
                    record[key].append(row)
                if key == "gpus" and not record[key]:
                    raise RuntimeError("nvidia-smi returned no GPUs")
            except Exception as error:
                query["error"] = f"{type(error).__name__}: {error}"
                record["errors"].append(query["error"])
    record["foreign_processes"] = [p for p in record["compute_processes"] if p["pid"] != allow_pid]
    abort = bool(record["errors"] or record["foreign_processes"])
    record.update(status="ABORT" if abort else "PASS", end_utc=utc(),
                  reason="QUERY_OR_ARGUMENT_ERROR" if record["errors"] else
                  "GPU_BUSY" if record["foreign_processes"] else "NO_FOREIGN_GPU_PROCESS",
                  exit_code=91 if abort else 0)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:  # Each attempt keeps its own record.
        json.dump(record, handle, indent=2)
        handle.write("\n")
    if abort:
        raise SystemExit(91)
    return record
