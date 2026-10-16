#!/usr/bin/env python3
"""Complete Qwen3-30B-A3B matrix, one BF16/online-FP8 format per locked job.

--plan-only prints all eight stages without loading weights, touching the GPU,
or creating the output directory. Run directly under the existing run_locked.py;
the controller owns status.json and this driver owns stages.json.
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

# The deployment entry remains outside live code; existing collection CLIs stay shared.
DEFAULT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(DEFAULT_ROOT / 'code'))

from run_olmoe_fp8_r09 import absolute
from run_task_quality import save, sha, stop_child, verify_outer
from tune_moe import model_geometry


def build_plan(root, output, config, weight_format, budget, plan_only):
    paths = {key: absolute(root, config[key]) for key in
             ("performance_arms", "quality_arms", "prompts", "articles", "model_config_reference", "input_compatibility")}
    warnings = []
    model_config = Path(config["model"]) / "config.json"
    if not model_config.is_file():
        if not plan_only:
            raise FileNotFoundError(f"Qwen checkpoint config is unavailable: {model_config}")
        warnings.append("Actual Qwen checkpoint is unavailable on this planning host; using the fixed-revision config reference. No weight-completeness or GPU-load claim.")
        model_config = paths["model_config_reference"]
    reference_sha = sha(paths["model_config_reference"])
    if reference_sha != config["model_config_sha256"] or sha(model_config) != reference_sha:
        raise ValueError("Qwen config must match the pinned official revision exactly")
    model_payload = json.loads(model_config.read_text())
    if model_payload.get("norm_topk_prob") is not True:
        raise ValueError("Preserve Qwen normalized top-k routing")
    geometry = model_geometry(model_payload)
    if any(geometry.get(key) != value for key, value in config["expected_geometry"].items()):
        raise ValueError("Expected the prepared Qwen3-30B-A3B E128/k8/I768/layer0/shared0 geometry")
    compatibility = json.loads(paths["input_compatibility"].read_text())["models"]["qwen3_bf16_ad44e777"]
    for key, name, count in (("prompts", "bench64", 64), ("articles", "validation_quality32", 32)):
        entry = compatibility["inputs"][name]
        if (entry["status"] != "PASS" or entry["sha256"] != sha(paths[key])
                or entry["rows"] != count or entry["distinct_prefix128"] != count
                or entry["short_rows"] or entry["duplicate_prefixes"]):
            raise ValueError("Qwen input differs from the prepared tokenizer compatibility record")
    for key, count in (("performance_arms", 9), ("quality_arms", 10)):
        payload = json.loads(paths[key].read_text())
        arms = payload["arms"] if isinstance(payload, dict) else payload
        if len(arms) != count or len({a["name"] for a in arms}) != count or arms[0] != {"name": "native", "spec": None}:
            raise ValueError(f"Keep all {count} native-first arms in {key}")
    batches = config["batches"]
    if batches != [1, 2, 4, 8, 16, 32, 64] or config["performance"]["expected_cells"] != 63:
        raise ValueError("The complete matrix requires seven batches and 63 cells")
    model = ["--model", config["model"], "--dtype", config["dtype"],
             "--gpu-memory-utilization", str(config["gpu_memory_utilization"])]
    if weight_format == "fp8":
        model += ["--quantization", "fp8"]
    tuned = output / "native_tuning" / "tuned_configs"
    traces = [output / "native_traces" / f"trace_native_b{batch}" / "snapshot.json" for batch in batches]
    stages = []

    def add(name, script, argv, artifact, expected, dependencies=(), use_tuned=True):
        stages.append({"name": name, "command": [sys.executable, str(root / "code" / script), *map(str, argv)],
                       "dependencies": list(dependencies), "timeout_cap_seconds": config["stage_timeout_seconds"][name],
                       "tuned_config_folder": str(tuned) if use_tuned else None,
                       "artifact": str(artifact), "expected_artifact_status": expected,
                       "log": str(output / "stage_logs" / f"{name}.log"), "status": "NOT_STARTED",
                       "started_unix": None, "ended_unix": None, "exit_code": None})

    add("native_traces", "bench_decode.py", model + ["--prompts", paths["prompts"],
        "--output-dir", output / "native_traces", "--arms-file", paths["performance_arms"],
        "--arms", "native", "--batches", *batches, "--traceonly", "--trace-steps", config["trace"]["steps"],
        "--newtokens", 2, "--prompt-tokens", 128, "--max-model-len", config["trace"]["max_model_len"],
        "--repeats", 1, "--warmup-seconds", 0, "--no-energy"],
        output / "native_traces" / "status.json", "COMPLETE", use_tuned=False)
    tuning = config["tuning"]
    if tuning["layer"] != geometry["first_moe_layer"]:
        raise ValueError("Tune the first routed Qwen layer, layer0")
    add("native_tuning", "tune_moe.py", ["--model", config["model"], "--quantization", weight_format,
        "--layer", tuning["layer"], "--traces", *traces, "--output-dir", output / "native_tuning",
        "--batches", *batches, "--events-per-batch", tuning["events_per_batch"], "--rounds", tuning["rounds"],
        "--replays-per-sample", tuning["replays_per_sample"], "--evict-mib", tuning["evict_mib"]],
        output / "native_tuning" / "summary.json", "PASS", ["native_traces"], False)
    for name, spec in [("graph_dense", {"mode": "dense", "backend": "triton"}),
            ("graph_marginal", {"mode": "marginal", "tail_rank": 4, "gate_threshold": .15,
                                "mass_budget": .30, "backend": "triton"})]:
        artifact = output / f"{name}.json"
        add(name, "smoke_graph.py", model + ["--output", artifact, "--spec-json", json.dumps(spec),
            "--batches", *batches, "--capture-sizes", *batches],
            artifact, "PASS", ["native_tuning"])
    quality = config["quality"]
    for batch in quality["batches"]:
        artifact = output / f"quality_b{batch}.json"
        add(f"quality_b{batch}", "quality_decode.py", model + ["--articles", paths["articles"],
            "--output", artifact, "--arms-file", paths["quality_arms"], "--batch", batch,
            "--prefix-tokens", quality["prefix_tokens"], "--tokens", quality["decode_tokens"],
            "--limit-docs", quality["documents"], "--reference-max-mib", quality["reference_max_mib"]],
            artifact, "PASS", ["native_tuning"])
    perf = config["performance"]
    add("bench_graph", "bench_graph.py", model + ["--prompts", paths["prompts"], "--output-dir", output / "bench",
        "--arms-file", paths["performance_arms"], "--batches", *batches, "--repeats", perf["repeats"],
        "--newtokens", perf["newtokens"], "--prompt-tokens", perf["prompt_tokens"],
        "--max-model-len", perf["max_model_len"], "--warmup-seconds", perf["warmup_seconds"]],
        output / "bench" / "status.json", "COMPLETE", ["native_tuning", "graph_dense", "graph_marginal"])
    return {"status": "PLANNED", "format": weight_format, "budget_seconds": budget,
            "output": str(output), "config": config, "stages": stages, "planning_warnings": warnings,
            "model_geometry": geometry, "model_default_norm_topk_prob": model_payload.get("norm_topk_prob"),
            "tokenizer_identity": compatibility["tokenizer"]["sha256"],
            "model_config": {"path": str(model_config), "sha256": sha(model_config)},
            "trace_files": list(map(str, traces)),
            "input_files": {key: {"path": str(path), "sha256": sha(path)} for key, path in paths.items()},
            "environment": {"HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                            "VLLM_ENABLE_V1_MULTIPROCESSING": "0", "TOKENIZERS_PARALLELISM": "false",
                            "VLLM_USE_FLASHINFER_SAMPLER": "0"}}


def check_artifact(stage, plan):
    artifact = json.loads(Path(stage["artifact"]).read_text())
    if artifact.get("status") != stage["expected_artifact_status"]:
        raise RuntimeError(f"Child artifact is not successful: {artifact.get('status')!r}")
    if stage["name"] == "native_traces" and not all(Path(p).is_file() for p in plan["trace_files"]):
        raise RuntimeError("Missing native trace snapshots for one or more batches")
    if stage["name"] == "native_tuning":
        fmt = plan["format"]
        expected_dtype = "fp8_w8a8" if fmt == "fp8" else None
        if artifact.get("quantization") != fmt or artifact.get("config_dtype") != expected_dtype:
            raise RuntimeError("Tuning metadata belongs to a different weight format")
        folder = Path(stage["artifact"]).parent
        filename = artifact["config_filename"]
        if Path(filename).name != filename or (fmt == "fp8" and "fp8_w8a8" not in filename) or (fmt == "bf16" and "fp8" in filename):
            raise RuntimeError("Tuned filename does not match the selected BF16/FP8-W8A8 format")
        payload = json.loads((folder / "tuned_configs" / filename).read_text())
        if not all(str(b) in payload for b in plan["config"]["batches"]):
            raise RuntimeError("Tuned configuration does not cover all seven batches")
        weights = json.loads((folder / "weights.json").read_text())
        if weights["layer"] != 0 or weights["geometry"] != plan["model_geometry"]:
            raise RuntimeError("Tuned weights do not retain Qwen layer0/k8/routed I768 geometry")
        if fmt == "fp8" and not (folder / "fp8_conversion.json").is_file():
            raise RuntimeError("Missing online-FP8 conversion metadata")
    if stage["name"] == "bench_graph" and artifact.get("completed_cells") != 63:
        raise RuntimeError("Performance matrix did not complete all 63 paired cells")


def execute(root, output, plan, started):
    """Same bounded stage pattern as r09, with Qwen format checks above."""
    manifest = output / "stages.json"
    (output / "stage_logs").mkdir(exist_ok=False)
    plan.update(status="RUNNING", started_unix=time.time())
    save(manifest, plan)
    def interrupted(signum, frame):
        raise InterruptedError(f"Qwen driver received signal {signum}")
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
                row.update(status="RUNNING", started_unix=time.time(), timeout_seconds=min(row["timeout_cap_seconds"], remaining - 10))
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
        plan["status"] = "COMPLETE" if all(row["status"] == "COMPLETE" for row in plan["stages"]) else "INCOMPLETE"
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
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--config", type=Path, default=Path("deployments/qwen_matrix_20261016/configs/qwen_matrix.json"))
    parser.add_argument("--format", choices=("bf16", "fp8"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget-seconds", type=int, default=7170)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    output, source = absolute(root, args.output), absolute(root, args.config)
    config = json.loads(source.read_text())
    if not 0 < args.budget_seconds <= min(7170, config["maximum_internal_budget_seconds"]):
        parser.error("--budget-seconds must be 1..7170 and no greater than the configured cap")
    if config["dtype"] != "bfloat16" or config["gpu_memory_utilization"] != .85:
        raise ValueError("Use BF16 model dtype/auto KV and the common .85 total-memory budget")
    plan = build_plan(root, output, config, args.format, args.budget_seconds, args.plan_only)
    plan["config_source"] = {"path": str(source), "sha256": sha(source)}
    plan["driver_source"] = {"path": str(Path(__file__)), "sha256": sha(Path(__file__))}
    plan["reused_driver_structure"] = config["reused_driver_structure"]
    if args.plan_only:
        print(json.dumps(plan, indent=2))
        return 0
    if (output / "stages.json").exists():
        raise FileExistsError("Refusing to overwrite existing stages.json")
    plan["outer_run"] = verify_outer(SimpleNamespace(outer_run_dir=output, budget_seconds=args.budget_seconds))
    return execute(root, output, plan, started)


if __name__ == "__main__":
    raise SystemExit(main())
