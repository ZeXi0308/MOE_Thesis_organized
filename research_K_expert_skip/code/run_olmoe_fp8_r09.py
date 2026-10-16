#!/usr/bin/env python3
"""Run the complete r09 online-FP8 sequence directly under run_locked.py.

--plan-only prints every command/dependency without creating a run directory.
The controller owns status.json; this driver only owns stages.json and its
stage outputs. Existing stages.json is never overwritten.
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
from types import SimpleNamespace

from run_task_quality import save, sha, stop_child, verify_outer


def absolute(root, path):
    path = Path(path).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def build_plan(root, output, config, budget):
    batches = config["batches"]
    paths = {name: absolute(root, config[name]) for name in
             ("performance_arms", "quality_arms", "prompts", "articles", "rolling_manifest")}
    for name, count in (("performance_arms", 9), ("quality_arms", 10)):
        payload = json.loads(paths[name].read_text())
        arms = payload["arms"] if isinstance(payload, dict) else payload
        if (len(arms) != count or len({a["name"] for a in arms}) != count
                or arms[0] != {"name": "native", "spec": None}):
            raise ValueError(f"{name} must preserve the complete native-first {count}-arm sequence")
    if batches != [1, 2, 4, 8, 16, 32, 64] or config["performance"]["expected_cells"] != 63:
        raise ValueError("r09 requires the complete seven-batch, 63-cell matrix")
    model = ["--model", config["model"], "--dtype", config["dtype"],
             "--quantization", config["quantization"], "--gpu-memory-utilization",
             str(config["gpu_memory_utilization"])]
    tuned = output / "native_tuning" / "tuned_configs"
    trace_files = [output / "native_traces" / f"trace_native_b{b}" / "snapshot.json" for b in batches]
    stages = []

    def add(name, script, argv, artifact, expected_status, dependencies=(), use_tuned=True):
        stages.append({"name": name, "command": [sys.executable, str(root / "code" / script), *map(str, argv)],
                       "dependencies": list(dependencies), "timeout_cap_seconds": config["stage_timeout_seconds"][name],
                       "tuned_config_folder": str(tuned) if use_tuned else None,
                       "artifact": str(artifact), "expected_artifact_status": expected_status,
                       "log": str(output / "stage_logs" / f"{name}.log"), "status": "NOT_STARTED",
                       "started_unix": None, "ended_unix": None, "exit_code": None})

    add("native_traces", "bench_decode.py", model + ["--prompts", paths["prompts"],
        "--output-dir", output / "native_traces", "--arms-file", paths["performance_arms"],
        "--arms", "native", "--batches", *batches, "--traceonly", "--trace-steps", config["trace"]["steps"],
        "--newtokens", 2, "--prompt-tokens", 128, "--max-model-len", config["trace"]["max_model_len"],
        "--repeats", 1, "--warmup-seconds", 0, "--no-energy"],
        output / "native_traces" / "status.json", "COMPLETE", use_tuned=False)
    tune = config["tuning"]
    add("native_tuning", "tune_moe.py", ["--model", config["model"], "--quantization", "fp8",
        "--traces", *trace_files, "--output-dir", output / "native_tuning", "--batches", *batches,
        "--events-per-batch", tune["events_per_batch"], "--rounds", tune["rounds"],
        "--replays-per-sample", tune["replays_per_sample"], "--evict-mib", tune["evict_mib"]],
        output / "native_tuning" / "summary.json", "PASS", ["native_traces"], False)
    specs = {"graph_dense": {"mode": "dense", "backend": "triton"},
             "graph_marginal": {"mode": "marginal", "tail_rank": 4, "gate_threshold": .15,
                                "mass_budget": .30, "backend": "triton"}}
    for name, spec in specs.items():
        artifact = output / f"{name}.json"
        add(name, "smoke_graph.py", model + ["--output", artifact, "--spec-json", json.dumps(spec)],
            artifact, "PASS", ["native_tuning"])
    graph_dependencies = ["native_tuning", "graph_dense", "graph_marginal"]
    micro = config["microbench"]
    add("microbench_policy", "microbench_policy.py", ["--output", output / "microbench_policy.json",
        "--experts", micro["experts"], "--top-k", micro["top_k"], "--batches", *micro["batches"],
        "--repetitions", micro["repetitions"]], output / "microbench_policy.json", "PASS", graph_dependencies)
    quality = config["quality"]
    quality_common = model + ["--arms-file", paths["quality_arms"], "--reference-max-mib", quality["reference_max_mib"]]
    rolling = config["rolling_path"]
    add("rolling_path", "quality_decode.py", quality_common + ["--tokenized-windows", paths["rolling_manifest"],
        "--batch", rolling["batch"], "--limit-windows", rolling["limit_windows"],
        "--output", output / "quality_rolling_path.json"], output / "quality_rolling_path.json", "PASS", graph_dependencies)
    for batch in quality["batches"]:
        artifact = output / f"quality_b{batch}.json"
        add(f"quality_b{batch}", "quality_decode.py", quality_common + ["--articles", paths["articles"],
            "--output", artifact, "--batch", batch, "--tokens", quality["decode_tokens"],
            "--prefix-tokens", quality["prefix_tokens"], "--limit-docs", quality["documents"]],
            artifact, "PASS", ["native_tuning"])
    perf = config["performance"]
    add("bench_graph", "bench_graph.py", model + ["--prompts", paths["prompts"], "--output-dir", output / "bench",
        "--arms-file", paths["performance_arms"], "--batches", *batches, "--repeats", perf["repeats"],
        "--newtokens", perf["newtokens"], "--prompt-tokens", perf["prompt_tokens"],
        "--max-model-len", perf["max_model_len"], "--warmup-seconds", perf["warmup_seconds"]],
        output / "bench" / "status.json", "COMPLETE", graph_dependencies)
    return {"status": "PLANNED", "output": str(output), "budget_seconds": budget,
            "config": config, "stages": stages, "trace_files": list(map(str, trace_files)),
            "environment": {"HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                            "VLLM_ENABLE_V1_MULTIPROCESSING": "0", "TOKENIZERS_PARALLELISM": "false",
                            "VLLM_USE_FLASHINFER_SAMPLER": "0"},
            "input_files": {name: {"path": str(path), "sha256": sha(path)} for name, path in paths.items()}}


def check_artifact(stage, plan):
    artifact = json.loads(Path(stage["artifact"]).read_text())
    if artifact.get("status") != stage["expected_artifact_status"]:
        raise RuntimeError(f"Child artifact status is {artifact.get('status')!r}")
    if stage["name"] == "native_traces" and not all(Path(p).is_file() for p in plan["trace_files"]):
        raise RuntimeError("Missing one or more of the seven native trace snapshots")
    if stage["name"] == "native_tuning":
        if artifact.get("quantization") != "fp8" or artifact.get("config_dtype") != "fp8_w8a8":
            raise RuntimeError("Tuner summary is not online FP8 W8A8")
        folder = Path(stage["artifact"]).parent / "tuned_configs"
        filename = artifact["config_filename"]
        if Path(filename).name != filename or "fp8" not in filename:
            raise RuntimeError("Tuner did not emit an FP8 configuration filename")
        payload = json.loads((folder / filename).read_text())
        if not all(str(b) in payload for b in plan["config"]["batches"]):
            raise RuntimeError("FP8 tuned configuration does not cover all seven batches")
    if stage["name"] == "bench_graph" and artifact.get("completed_cells") != 63:
        raise RuntimeError("Full graph benchmark did not complete all 63 paired cells")
    if stage["name"] == "microbench_policy" and (not artifact.get("rows") or
            any("eager_thread_cpu_us_per_call" not in row for row in artifact["rows"])):
        raise RuntimeError("F6 output lacks the new eager thread CPU measurement")


def execute(root, output, plan, started):
    manifest = output / "stages.json"
    (output / "stage_logs").mkdir(exist_ok=False)
    plan.update(status="RUNNING", started_unix=time.time())
    save(manifest, plan)
    def interrupted(signum, frame):
        raise InterruptedError(f"r09 driver received signal {signum}")
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupted)
    child, current = None, None
    try:
        for row in plan["stages"]:
            current = row
            failed = [name for name in row["dependencies"]
                      if next(s for s in plan["stages"] if s["name"] == name)["status"] != "COMPLETE"]
            remaining = plan["budget_seconds"] - (time.monotonic() - started)
            row["remaining_budget_at_start_seconds"] = remaining
            if failed:
                row.update(status="NOT_RUN_DEPENDENCY", failed_dependencies=failed)
            elif remaining <= 10:
                row.update(status="NOT_RUN_BUDGET")
            else:
                row.update(status="RUNNING", started_unix=time.time(),
                           timeout_seconds=min(row["timeout_cap_seconds"], remaining - 10))
                save(manifest, plan)
                env = {**os.environ, **plan["environment"]}
                env.pop("VLLM_TUNED_CONFIG_FOLDER", None)
                if row["tuned_config_folder"]:
                    env["VLLM_TUNED_CONFIG_FOLDER"] = row["tuned_config_folder"]
                try:
                    with Path(row["log"]).open("x") as log:
                        child = subprocess.Popen(row["command"], cwd=root, env=env,
                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                        row["pid"] = child.pid
                        save(manifest, plan)
                        try:
                            row["exit_code"] = child.wait(timeout=row["timeout_seconds"])
                            row["status"] = "COMPLETE" if row["exit_code"] == 0 else "FAILED"
                        except subprocess.TimeoutExpired:
                            row["status"] = "TIMEOUT"
                            stop_child(child)
                            row["exit_code"] = child.returncode
                        if row["status"] == "COMPLETE":
                            check_artifact(row, plan)
                except InterruptedError:
                    raise
                except (OSError, ValueError, KeyError, RuntimeError) as error:
                    row.update(status="FAILED", error=f"{type(error).__name__}: {error}")
                finally:
                    if child is not None:
                        stop_child(child)
                        row["exit_code"] = child.returncode
                        child = None
                    row["ended_unix"] = time.time()
            save(manifest, plan)
            print(json.dumps(row), flush=True)
        plan["status"] = "COMPLETE" if all(s["status"] == "COMPLETE" for s in plan["stages"]) else "INCOMPLETE"
    except BaseException as error:
        if current and current["status"] == "RUNNING":
            current.update(status="INTERRUPTED", ended_unix=time.time())
        for row in plan["stages"]:
            if row["status"] == "NOT_STARTED":
                row["status"] = "NOT_RUN_INTERRUPTED"
        plan.update(status="INTERRUPTED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if child is not None:
            stop_child(child)
        plan.update(ended_unix=time.time(), elapsed_seconds=time.monotonic() - started)
        save(manifest, plan)
    return 0 if plan["status"] == "COMPLETE" else 1


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("runs/olmoe_fp8_matrix_r09"))
    parser.add_argument("--config", type=Path, default=Path("configs/olmoe_fp8_matrix_r09.json"))
    parser.add_argument("--budget-seconds", type=int, default=7170)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    output, source = absolute(root, args.output), absolute(root, args.config)
    config = json.loads(source.read_text())
    if not 0 < args.budget_seconds <= min(7170, config["maximum_internal_budget_seconds"]):
        parser.error("--budget-seconds must be 1..7170 and no greater than the configured budget")
    if (config["quantization"] != "fp8" or config["dtype"] != "bfloat16"
            or config["gpu_memory_utilization"] != .35):
        raise ValueError("r09 requires online FP8, BF16 model dtype/auto KV and the common .35 budget")
    plan = build_plan(root, output, config, args.budget_seconds)
    plan["config_source"] = {"path": str(source), "sha256": sha(source)}
    if args.plan_only:
        print(json.dumps(plan, indent=2))
        return 0
    if (output / "stages.json").exists():
        raise FileExistsError("Refusing to overwrite existing stages.json")
    plan["outer_run"] = verify_outer(SimpleNamespace(outer_run_dir=output, budget_seconds=args.budget_seconds))
    return execute(root, output, plan, started)


if __name__ == "__main__":
    raise SystemExit(main())
