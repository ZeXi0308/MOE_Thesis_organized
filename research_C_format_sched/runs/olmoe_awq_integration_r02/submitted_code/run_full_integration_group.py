#!/usr/bin/env python3
"""Three fresh arm processes under ONE existing run_locked ancestor.

  python code/run_locked.py --name integration_group_r01 --timeout 1800 -- \
    python code/run_full_integration_group.py --stage forward \
    --output runs/integration_group_r01/group \
    --lock-status runs/integration_group_r01/status.json

For --stage serving use a new job and outer timeout 1200. Inner arm budgets
are 900 seconds for forward and 360 for serving. The outer budget bounds the
whole group, even when it is less than the sum of inner budgets. This runner
does not start a session, acquire a lock, retry, or continue after a failure.
Back up and hash the whole terminal group before starting the next GPU job.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from bench_serving import verify_lock
from prepare_quantization import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=("forward", "serving"), default="forward")
    parser.add_argument("--output", required=True)
    parser.add_argument("--lock-status", required=True)
    args = parser.parse_args()
    lock_path = Path(args.lock_status).resolve()
    lock = verify_lock(lock_path)
    controller = json.loads(lock_path.read_text())
    pgid = os.getpgrp()
    # run_locked launches GNU timeout, which owns this process group. Verify
    # that exact live ancestor before any group signal can be sent.
    timeout_pid = controller["worker_pid"]
    timeout_cmd = Path(f"/proc/{timeout_pid}/cmdline").read_bytes().split(b"\0")
    if pgid != timeout_pid or os.getpgid(timeout_pid) != pgid or Path(os.fsdecode(timeout_cmd[0])).name != "timeout" or os.getpgid(lock["controller_pid"]) == pgid:
        raise RuntimeError("refuse execution without the verified outer timeout-owned process group")
    output = Path(args.output).resolve()
    if not output.is_relative_to(lock_path.parent):
        raise ValueError("group output must be inside this run_locked directory")
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    arm_timeout = 900 if args.stage == "forward" else 360
    record = {"schema_version": 1, "status": "running", "stage": args.stage,
              "lock": lock, "verified_timeout_pgid": pgid, "started_unix_s": time.time(),
              "per_arm_timeout_s": arm_timeout, "arms": [
                  {"arm": arm, "status": "not_started"} for arm in ("single_fp8", "single_awq", "dual_phase")]}
    def save():
        temporary = output / "group.tmp"
        write_json(temporary, record)
        temporary.replace(output / "group.json")
    def interrupted(signum, frame):
        raise InterruptedError(f"received signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    save()
    process, current = None, None
    try:
        for current in record["arms"]:
            arm = current["arm"]
            arm_output = output / arm
            stdout = output / f"{arm}.stdout.log"
            if args.stage == "forward":
                command = [sys.executable, str(root / "code/run_full_model_integration.py"),
                           "--worker-config", str(root / f"configs/olmoe_integration_r01/{arm}.json"),
                           "--output", str(arm_output / "integration")]
            else:
                command = [sys.executable, str(root / "code/run_serving_group.py"),
                           "--config", str(root / f"configs/minimal_serving_r01/{arm}.json"),
                           "--output", str(arm_output)]
            command += ["--lock-status", str(lock_path), "--run"]
            current.update(status="running", command=command, stdout=str(stdout), started_unix_s=time.time())
            save()
            with stdout.open("w") as log:
                try:
                    process = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
                    current["pid"] = process.pid
                    save()
                    current["exit_code"] = process.wait(timeout=arm_timeout)
                finally:
                    # Consumers require a new output directory; move the log
                    # inside only after their creation attempt, including failure.
                    arm_output.mkdir(parents=True, exist_ok=True)
                    stdout.replace(arm_output / "stdout.log")
                    current["stdout"] = str(arm_output / "stdout.log")
            current["ended_unix_s"] = time.time()
            if current["exit_code"] != 0:
                raise RuntimeError(f"{arm} exited {current['exit_code']}; remaining arms were not started")
            current["status"] = "complete"
            save()
        record.update(status="complete", ended_unix_s=time.time())
        save()
    except BaseException as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}", ended_unix_s=time.time())
        if current is not None:
            current.update(status="timeout" if isinstance(error, subprocess.TimeoutExpired) else "failed",
                           error=record["error"], ended_unix_s=time.time())
        save()  # Durable failure precedes group termination, including this process.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        os.killpg(pgid, signal.SIGTERM)
        if process is not None:
            try:
                process.wait(timeout=25)
            except subprocess.TimeoutExpired:
                os.killpg(pgid, signal.SIGKILL)
        raise SystemExit(1)
    print(json.dumps({"status": record["status"], "stage": args.stage, "output": str(output)}))


if __name__ == "__main__":
    main()
