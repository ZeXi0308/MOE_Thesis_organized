#!/usr/bin/env python3
"""Bounded OLMoE task/free-generation suite; execute inside run_locked.py.

The default runs all five fixed quality arms together, sharing each job's native
reference. Jobs are independent task families plus natural free generation.
Failures retain their files and do not prevent subsequent jobs within budget.
--plan-only uses only the standard library and does not load a model or datasets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
TASKS = ("gsm8k", "mmlu", "hellaswag", "arc_challenge")


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolved(path, root):
    return (path if path.is_absolute() else root / path).expanduser().resolve()


def inputs(args):
    from free_generate import resolve_arms
    # Reuse the runners' CPU-only policy/name validation, retaining the exact
    # user-supplied specs and order rather than selecting or adding any arms.
    resolve_arms(args.arms_file)
    payload = json.loads(args.arms_file.read_text())
    entries = payload["arms"] if isinstance(payload, dict) else payload
    if entries[0]["name"] != "native" or entries[0]["spec"] is not None:
        raise ValueError("arms-file must start with native and spec=null")
    arms = payload if isinstance(payload, dict) else {"arms": entries}
    warnings = []
    provenance = {"arms_file": str(args.arms_file), "arms_sha256": sha(args.arms_file)}
    for name, path in (("model_config", args.model / "config.json"),
                       ("dataset_source", args.data_dir / "source.json"),
                       ("prompts", args.prompts)):
        if not path.is_file():
            if not args.plan_only:
                raise ValueError(f"Missing {name}: {path}")
            warnings.append(f"Unresolved on this planning host: {name}={path}")
            continue
        provenance[name] = {"path": str(path), "sha256": sha(path)}
        if name == "model_config":
            config = json.loads(path.read_text())
            if config.get("model_type") != "olmoe" or config.get("quantization_config"):
                raise ValueError("Use one unquantized local OLMoE checkpoint for BF16 and online FP8")
        elif name == "dataset_source":
            source = json.loads(path.read_text())
            if source.get("status") != "READY" or not set(args.tasks) <= set(source.get("tasks", [])):
                raise ValueError("Task datasets must already be READY and cover requested tasks")
        else:
            seen = set()
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                prompt = row if isinstance(row, str) else row.get("prompt", row.get("text"))
                if not isinstance(prompt, str) or not prompt.strip() or prompt in seen:
                    raise ValueError("Free prompts must be nonempty and distinct")
                seen.add(prompt)
                if len(seen) == args.free_limit:
                    break
            if len(seen) != args.free_limit:
                raise ValueError(f"Need {args.free_limit} distinct free-generation prompts")
    return arms, provenance, warnings


def verify_outer(args):
    """Require direct launch by the existing controller and its inherited lock."""
    if args.outer_run_dir is None:
        raise ValueError("Execution requires --outer-run-dir pointing to this run_locked job")
    status_path = args.outer_run_dir / "status.json"
    # run_locked writes worker_pid immediately after Popen; allow that short race.
    for _ in range(40):
        state = json.loads(status_path.read_text())
        if state.get("worker_pid") == os.getpid():
            break
        time.sleep(.05)
    if state.get("state") != "RUNNING" or state.get("worker_pid") != os.getpid() or state.get("controller_pid") != os.getppid():
        raise RuntimeError("Launch this script directly as run_locked.py's child command")
    lock = Path(state["lock_path"]).resolve()
    inherited = any(p.resolve() == lock for p in Path("/proc/self/fd").iterdir())
    if not inherited:
        raise RuntimeError("The outer GPU lock descriptor was not inherited")
    if args.budget_seconds + 30 > state["timeout_s"]:
        raise ValueError("Outer timeout must exceed this suite budget by at least 30 seconds")
    return {"status_file": str(status_path), "controller_pid": os.getppid(),
            "lock_path": str(lock), "outer_timeout_s": state["timeout_s"]}


def build_plan(args, arms, provenance, warnings):
    variants = [("all_arms", arms["arms"])]
    if args.per_arm_isolation:
        variants = [(row["name"], [arms["arms"][0], row] if row["name"] != "native" else [row])
                    for row in arms["arms"]]
    jobs, arm_files = [], {}
    for tag, rows in variants:
        arm_files[tag] = {"purpose": arms.get("purpose"), "arms": rows}
    for fmt in args.formats:
        for unit in ("free_generation", *args.tasks):
            for tag, rows in variants:
                job_id = f"{fmt}.{unit}.{tag}"
                output = args.output / job_id
                command = [sys.executable, str(args.root / "code" / ("free_generate.py" if unit == "free_generation" else "eval_tasks.py"))]
                if unit != "free_generation":
                    command.append("standard")
                command += ["--model", str(args.model), "--output", str(output),
                            "--arms-file", str(args.output / "arms" / f"{tag}.json"),
                            "--batch", str(args.batch), "--max-tokens", str(args.max_tokens),
                            "--max-model-len", str(args.max_model_len), "--dtype", "bfloat16",
                            "--gpu-memory-utilization", str(args.gpu_memory_utilization),
                            "--seed", str(args.seed)]
                if fmt == "fp8":
                    command += ["--quantization", "fp8"]
                if unit == "free_generation":
                    command += ["--prompts", str(args.prompts), "--limit", str(args.free_limit),
                                "--prompt-tokens", str(args.prompt_tokens), "--temperature", "0", "--top-p", "1"]
                else:
                    command += ["--data-dir", str(args.data_dir), "--tasks", unit]
                    if args.limit is not None:
                        command += ["--limit", str(args.limit)]
                jobs.append({"id": job_id, "format": fmt, "unit": unit,
                             "arms": [row["name"] for row in rows], "command": command,
                             "output": str(output), "log": str(args.output / f"{job_id}.log"),
                             "status": "NOT_STARTED"})
    return {"schema": 1, "status": "PLANNED", "created_unix": time.time(),
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "inputs": provenance, "planning_warnings": warnings,
            "budget_seconds": args.budget_seconds, "job_timeout_seconds": args.job_timeout,
            "jobs": jobs, "arm_files": arm_files,
            "semantics": {
                "formats": "Same local BF16 checkpoint; fp8 means online weight quantization. Model dtype and auto KV remain BF16.",
                "mc": "MMLU 5-shot, HellaSwag 0-shot, ARC-c 0-shot are prefill loglikelihood identity controls, not decode-quality evidence.",
                "generation": "GSM8K official 5-shot and 200 natural free completions exercise decode; EOS remains enabled, max_tokens is a cap.",
                "limit": "Task --limit applies independently to every leaf, including each MMLU subject; pilot subsets are exploratory.",
                "failure": "A failed job preserves partial files; later independent families still run within budget. Arms not reached by the existing runner are NOT_RUN, never inferred successful.",
                "native_reuse": "Shares native within each job's complete supplied arm sequence (five by default). No automatic score reuse or arm selection. Per-arm isolation explicitly repeats native.",
                "memory": "Eager existing runners; no CUDA graph or timing claims. Startup and all jobs consume the outer GPU budget."}}


def stop_child(child):
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    if child.poll() is None:
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
    # The process may exit before its own workers. Clean only this job's group,
    # which was created by Popen(start_new_session=True), before releasing lock.
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    child.wait()


def summarize_child(job):
    path = Path(job["output"]) / "summary.json"
    if not path.is_file():
        return {"status": "NO_SUMMARY", "arms": {name: "UNKNOWN_OR_NOT_RUN" for name in job["arms"]}}
    summary = json.loads(path.read_text())
    statuses = {}
    if job["unit"] == "free_generation":
        for name in job["arms"]:
            row = summary.get("arms", {}).get(name, {})
            statuses[name] = ("NOT_RUN" if not row.get("attempted") else
                              "COMPLETE" if row.get("not_run") == 0 and not row.get("termination_counts", {}).get("fail") else "PARTIAL_OR_FAILED")
    else:
        completed = {row["arm"] for row in summary.get("evaluations", [])}
        for name in job["arms"]:
            statuses[name] = "COMPLETE" if name in completed else "NOT_COMPLETED"
    return {"status": summary.get("status"), "path": str(path), "arms": statuses,
            "error": summary.get("error")}


def execute(args, plan):
    manifest = args.output / "suite.json"
    started = time.monotonic()
    plan.update(status="RUNNING", started_unix=time.time())
    save(manifest, plan)
    def interrupted(signum, frame):
        raise InterruptedError(f"Suite received signal {signum}")
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupted)
    child = None
    try:
        for job in plan["jobs"]:
            remaining = args.budget_seconds - (time.monotonic() - started)
            if remaining <= 5:
                job.update(status="NOT_RUN_BUDGET")
                continue
            job.update(status="RUNNING", started_unix=time.time(), timeout_s=min(args.job_timeout, remaining - 5))
            save(manifest, plan)
            try:
                with Path(job["log"]).open("x") as log:
                    child = subprocess.Popen(job["command"], cwd=args.root, stdout=log,
                                             stderr=subprocess.STDOUT, start_new_session=True)
                    job["pid"] = child.pid
                    save(manifest, plan)
                    try:
                        rc = child.wait(timeout=job["timeout_s"])
                        job.update(status="COMPLETE" if rc == 0 else "FAILED", exit_code=rc)
                    except subprocess.TimeoutExpired:
                        stop_child(child)
                        job.update(status="TIMEOUT", exit_code=child.returncode)
            except (OSError, ValueError) as error:
                job.update(status="FAILED", error=f"{type(error).__name__}: {error}")
            finally:
                if child is not None:
                    stop_child(child)
                    child = None
                job["ended_unix"] = time.time()
                try:
                    job["runner_summary"] = summarize_child(job)
                    if job["status"] == "COMPLETE" and job["runner_summary"]["status"] != "COMPLETE":
                        job.update(status="FAILED", error="Child exited zero without a COMPLETE runner summary")
                except (OSError, ValueError) as error:
                    job["runner_summary"] = {"status": "UNREADABLE", "error": str(error)}
                save(manifest, plan)
        plan["status"] = "COMPLETE" if all(job["status"] == "COMPLETE" for job in plan["jobs"]) else "INCOMPLETE"
    except BaseException as error:
        plan.update(status="INTERRUPTED", error=f"{type(error).__name__}: {error}")
        for job in plan["jobs"]:
            if job["status"] == "RUNNING":
                job["status"] = "INTERRUPTED"
        raise
    finally:
        if child is not None:
            stop_child(child)
        plan.update(ended_unix=time.time(), elapsed_seconds=time.monotonic() - started)
        save(manifest, plan)
    return 0 if plan["status"] == "COMPLETE" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New unique directory, e.g. outer run directory / quality")
    parser.add_argument("--outer-run-dir", type=Path, help="Existing run_locked run directory; required except for plan-only")
    parser.add_argument("--arms-file", type=Path, default=Path("configs/quality_eval_arms.json"),
                        help="Run every listed arm in order; native/spec=null must be first; default is the five quality arms")
    parser.add_argument("--formats", nargs="+", choices=("bf16", "fp8"), default=["bf16"])
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--limit", type=int, help="Exploratory task cap per leaf; omit for full tasks")
    parser.add_argument("--free-limit", type=int, default=200, help="Use explicit smaller value for a pilot")
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=.35)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--budget-seconds", type=int, required=True)
    parser.add_argument("--job-timeout", type=int, default=1200)
    parser.add_argument("--per-arm-isolation", action="store_true", help="Explicit extra native runs to isolate arm failures; default shares native")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    args.root = args.root.expanduser().resolve()
    for key in ("model", "data_dir", "prompts", "output", "outer_run_dir", "arms_file"):
        if getattr(args, key) is not None:
            setattr(args, key, resolved(getattr(args, key), args.root))
    if (min(args.free_limit, args.prompt_tokens, args.batch, args.max_tokens, args.max_model_len,
            args.budget_seconds, args.job_timeout) <= 0 or args.free_limit > 200 or
            args.max_tokens >= args.max_model_len or args.prompt_tokens + args.max_tokens > args.max_model_len or
            not 0 < args.gpu_memory_utilization < 1 or (args.limit is not None and args.limit <= 0)):
        parser.error("Invalid sample, context, memory or time bounds")
    if len(set(args.formats)) != len(args.formats) or len(set(args.tasks)) != len(args.tasks):
        parser.error("Duplicate formats or tasks are not allowed")
    arms, provenance, warnings = inputs(args)
    plan = build_plan(args, arms, provenance, warnings)
    if not args.plan_only:
        plan["outer_run"] = verify_outer(args)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "arms").mkdir()
    for tag, value in plan["arm_files"].items():
        save(args.output / "arms" / f"{tag}.json", value)
    save(args.output / "suite.json", plan)
    print(json.dumps({"status": "PLANNED", "jobs": len(plan["jobs"]), "manifest": str(args.output / "suite.json"),
                      "warnings": warnings}), flush=True)
    if args.plan_only:
        return 0
    result = execute(args, plan)
    print(json.dumps({"status": plan["status"], "manifest": str(args.output / "suite.json"),
                      "jobs": {job["id"]: job["status"] for job in plan["jobs"]}}), flush=True)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
