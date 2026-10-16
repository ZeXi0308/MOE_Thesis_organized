#!/usr/bin/env python3
"""Run the existing reduced-domain protocol on prepared code/summary inputs.

This CPU coordinator follows the original evaluation sequence, never owns a
GPU lock, and delegates every GPU stage to the existing budgeted run_campaign.
It neither retries failed stages nor reuses existing run-directory names.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from prepare_reduced_evaluation import domain_name

ROOT = Path(__file__).resolve().parents[1]
ROLES = {"ar", "static", "native_unrestricted", "native_threshold", "adaptive"}


def read(path):
    return json.loads(Path(path).read_text())


def project_file(name):
    path = (ROOT / name).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("manifest path leaves the project")
    return path


def verify_files(files):
    for name, expected in files.items():
        actual = hashlib.sha256(project_file(name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"prepared input changed: {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="configs/auxiliary_campaigns.json")
    parser.add_argument("--run", default="auxiliary_evaluation_sequence_r01", type=domain_name)
    parser.add_argument("--predecessor", default="runs/olmoe_evaluation_sequence_r02/status.json")
    parser.add_argument("--describe", action="store_true")
    args = parser.parse_args()
    manifest_path = project_file(args.manifest)
    plan = read(manifest_path)
    if plan.get("status") != "prepared_pending_gpu" or plan.get("total_gpu_budget_seconds") != 72000:
        raise ValueError("expected the prepared 20-hour-budget auxiliary manifest")
    domains = [domain_name(item["domain"]) for item in plan["domains"]]
    if len(domains) != 12 or len(set(domains)) != 12:
        raise ValueError("expected twelve unique model/source/task domains")
    verify_files(plan["input_sha256"])
    predecessor = project_file(args.predecessor)
    if not predecessor.is_relative_to(ROOT / "runs"):
        raise ValueError("predecessor must be a project run status")
    if args.describe:
        print(json.dumps({"status": "prepared_pending_gpu", "domains": domains,
                          "predecessor": str(predecessor), "gpu_campaigns": 12 * 7,
                          "total_gpu_budget_seconds": 72000,
                          "boundary": "Prepared commands, not measured results."}))
        return

    output = ROOT / "runs" / args.run
    output.mkdir(exist_ok=False)
    shutil.copyfile(__file__, output / "coordinator.py")
    shutil.copyfile(manifest_path, output / "manifest.json")
    state = {"status": "waiting_for_predecessor", "pid": os.getpid(),
             "started_unix_s": time.time(), "args": vars(args), "stages": []}

    def save():
        temp = output / "status.tmp"
        temp.write_text(json.dumps(state, indent=2) + "\n")
        temp.replace(output / "status.json")

    def execute(label, argv, status_file=None):
        verify_files(plan["input_sha256"])
        stage = {"label": label, "argv": argv, "started_unix_s": time.time()}
        state["stages"].append(stage)
        state.update(status="running_stage", active_stage=label)
        save()
        with (output / (label + ".log")).open("x") as log:
            child = subprocess.Popen(argv, cwd=ROOT, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT)
            stage["pid"] = child.pid
            save()
            code = child.wait()
        stage.update(returncode=code, end_unix_s=time.time())
        if status_file is not None:
            path = project_file(status_file)
            if not path.is_relative_to(ROOT / "runs"):
                raise ValueError("child status leaves project runs")
            stage["result_status"] = read(path).get("status") if path.is_file() else "missing"
        save()
        if code or (status_file is not None and stage["result_status"] != "complete"):
            raise RuntimeError(f"{label}: exit {code}, result {stage.get('result_status')}")

    def campaign(domain, suffix, extra=()):
        run = f"{domain}_{suffix}_r01"
        execute(run, [sys.executable, "-u", "code/run_campaign.py", "--config",
                      f"configs/{domain}_{suffix}.json", "--run", run,
                      "--wait-lock-seconds", "21600", "--skip-trace", *extra],
                f"runs/{run}/status.json")

    save()
    try:
        deadline = time.monotonic() + 21600
        while True:
            previous = read(predecessor)
            if previous.get("status") == "complete":
                break
            if previous.get("status") not in ("waiting_for_predecessor", "running_stage"):
                raise RuntimeError("predecessor needs attention: " + str(previous.get("status")))
            pid = previous["pid"]
            os.kill(pid, 0)
            argv = (Path("/proc") / str(pid) / "cmdline").read_bytes().split(b"\0")
            if predecessor.parent.name.encode() not in argv:
                raise RuntimeError("predecessor PID does not identify the expected run")
            if time.monotonic() >= deadline:
                raise TimeoutError("predecessor observation deadline; inspect the original process")
            time.sleep(10)

        # Native matrices for all workload domains precede policy calibration.
        # Existing 80-record OLMoE workloads remain separate historical plans.
        for domain in domains:
            campaign(domain, "reduced")
        for domain in domains:
            campaign(domain, "reduced_calibration",
                     ["--gammas", "0", "3", "7", "--mode", "static", "--proposal-budget", "7"])
            preparation = f"runs/{domain}_controller_preparation_r01"
            execute(domain + "_prepare_evaluation", [sys.executable, "-u",
                    "code/prepare_reduced_evaluation.py", "--domain", domain])
            commands = read(ROOT / preparation / "evaluation_commands.json")["commands"]
            if len(commands) != 5 or {c["role"] for c in commands} != ROLES or len({c["run"] for c in commands}) != 5:
                raise ValueError("expected five unique actual policy commands")
            for item in commands:
                if item["argv"][:3] != [sys.executable, "-u", "code/run_campaign.py"] or Path(item["cwd"]).resolve() != ROOT:
                    raise ValueError("unexpected evaluation command")
            for item in commands:
                execute(item["run"], item["argv"], item["status_file"])
        state.update(status="complete", end_unix_s=time.time())
        save()
    except BaseException as error:
        state.update(status="needs_attention", error=repr(error), end_unix_s=time.time())
        save()
        raise


if __name__ == "__main__":
    main()
