#!/usr/bin/env python3
"""Prepare five compiled policies from complete compiled measurements (CPU only).

  python code/prepare_compiled_controller_evaluation.py \
    --calibration-run runs/olmoe_compiled_fullbudget_calibration_r01 \
    --direct-run runs/olmoe_ngram_compiled_r01 \
    --output runs/controller_preparation_compiled_r01

Validate the full 56-cell native matrix; select on protocol B={1,4,16,64}.
Costs come only from the separate 32-cell, fixed proposal-budget7 campaign.
Choices are computed from these inputs, including gamma0 and all-zero tables.
No campaign, GPU work, remote access, or held-out selection occurs here.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

from build_calibration import analyze_run
from cost_model import AcceptanceEWMA, CalibrationTable
from plot_results import _context, _requests, load_runs, resolve_regime, route_instrumented
from prepare_controller_evaluation import (
    BATCHES, GAMMAS, CRITICAL_SOURCE, PROTOCOL_SHA, behavior_sha,
    read, require_complete_status, sha,
)
from prepare_reduced_evaluation import domain_signature, expected_spec, positive
from select_controller_baselines import select_regime, validate_protocol, write_selection


ROOT = Path(__file__).resolve().parents[1]
NATIVE_BATCHES = [1, 2, 4, 8, 16, 32, 64]
DOMAIN = "olmoe_ngram_compiled"
ENGINE_FIELDS = ("model", "dtype", "quantization", "kv_cache_memory_bytes",
                 "max_model_len", "max_num_seqs", "max_num_batched_tokens",
                 "enforce_eager", "kernel_config")


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def check_config(plan, *, calibration):
    required = {"batches": BATCHES, "repeats": 1 if calibration else 2,
        "measured_requests": 64, "output_tokens": 64 if calibration else 128,
        "input_offset": 0 if calibration else 64, "warmup_offset": 0,
        "shape_warmup_offset": 0, "warmup_each_batch": True, "enforce_eager": False}
    if any(plan.get(key) != value for key, value in required.items()):
        raise ValueError("compiled config differs from the fixed population/warmup/output protocol")
    if plan.get("gammas") != (GAMMAS if calibration else [7]):
        raise ValueError("unexpected configured gamma grid")
    sig = domain_signature(plan)
    if (sig["max_num_seqs"] != 64 or sig["admission_mode"] != "wave"
            or sig["draft_source"] != "ngram" or plan.get("native_route_return", False)
            or plan.get("native_capture_padding", False)
            or plan.get("candidate_gammas", GAMMAS) != GAMMAS):
        raise ValueError("compiled evaluation requires route-off ngram, wave cap64, gamma0..7")
    if plan.get("total_gpu_budget_seconds") != 72000:
        raise ValueError("compiled evaluation must preserve the shared 20-hour budget")


def warmup_evidence(plan, records):
    """Actual source IDs, including historical native general warmup exposure."""
    general = plan.get("warmup_offset", 80)
    shape = plan.get("shape_warmup_offset", plan.get("input_offset", 0))
    general_records = records[general:general + max(plan["batches"])]
    shape_records = records[shape:shape + 64] if plan.get("warmup_each_batch") else []
    heldout = {row["source_index"] for row in records[64:128]}
    general_ids = [row["source_index"] for row in general_records]
    shape_ids = [row["source_index"] for row in shape_records]
    if len(general_records) != max(plan["batches"]) or (plan.get("warmup_each_batch") and len(shape_records) != 64):
        raise ValueError("workload lacks the declared warmup population")
    return {"warmup_offset": general, "shape_warmup_offset": shape,
        "general_warmup_record_range": [general, general + len(general_records) - 1],
        "general_warmup_source_indices": general_ids, "shape_warmup_source_indices": shape_ids,
        "warmup_overlap_heldout_source_indices": sorted((set(general_ids) | set(shape_ids)) & heldout),
        "warmup_boundary": "Measured development and evaluation populations are disjoint; historical native general warmup may have exposed held-out prompts."}


def inspect_campaign(directory, expected_plan, *, calibration):
    directory = Path(directory).resolve()
    require_complete_status(read(directory / "status.json"))
    plan = read(directory / "plan.json")
    if domain_signature(plan) != domain_signature(expected_plan) or plan.get("enforce_eager") is not False:
        raise ValueError("campaign differs from the compiled model/resource/execution domain")
    batches = BATCHES if calibration else NATIVE_BATCHES
    repeats, length, mode = (1, 64, "static") if calibration else (2, 128, "native")
    if (plan.get("batches"), plan.get("gammas"), plan.get("repeats"),
            plan.get("output_tokens"), plan.get("measured_requests")) != (batches, GAMMAS, repeats, length, 64):
        raise ValueError("campaign does not contain the complete compiled grid/population")
    if plan.get("input_offset", 0) != 0 or not plan.get("warmup_each_batch"):
        raise ValueError("development requires input_offset0 and full shape warmup")
    if calibration:
        check_config(plan, calibration=True)
    workload = ROOT / plan["workload"]
    records, workload_hash = read(workload)["records"], sha(workload)
    development_ids = sorted(row["source_index"] for row in records[:64])
    heldout_ids = sorted(row["source_index"] for row in records[64:128])
    if len(set(development_ids + heldout_ids)) != 128:
        raise ValueError("need 64 distinct development and 64 disjoint held-out prompt identities")
    warmup = warmup_evidence(plan, records)
    paths, provenance, regimes, environments, costs, initial_states = [], [], [], [], [], []
    source_hashes = {name: sha(directory / "source" / name) for name in CRITICAL_SOURCE}
    behaviors = {name: behavior_sha(directory / "source" / name) for name in CRITICAL_SOURCE}
    if calibration and any(behaviors[name] != behavior_sha(ROOT / "code" / name) for name in CRITICAL_SOURCE):
        raise ValueError("executable behavior changed since compiled cost calibration")
    sig = domain_signature(plan)
    for gamma in GAMMAS:
        folder = directory / f"gamma{gamma}"
        if read(folder / "status.json").get("status") != "complete":
            raise ValueError(f"gamma{gamma} did not finish")
        config, engine, env = (read(folder / name) for name in ("configuration.json", "engine_args.json", "environment.json"))
        args, budget = config["args"], 7 if calibration else gamma
        if config.get("plan") != plan or config.get("workload_sha256") != workload_hash:
            raise ValueError("engine plan/workload differs from the campaign")
        if (args.get("mode") != mode or args.get("gamma") != gamma
                or args.get("natural_stop", False) or args.get("native_table")
                or not (args.get("skip_trace") or plan.get("skip_trace"))):
            raise ValueError("wrong runtime mode, gamma, stop rule, table, or skip-trace setting")
        for key, value in (("batches", batches), ("repeats", repeats), ("output_tokens", length), ("admission_mode", "wave")):
            if args.get(key) is not None and args[key] != value:
                raise ValueError(f"unexpected runtime {key} override")
        if (args.get("proposal_budget") if args.get("proposal_budget") is not None else gamma) != budget:
            raise ValueError("native direct budgets cannot substitute for fixed-budget7 costs")
        if any(engine.get(key) != sig[key] for key in ENGINE_FIELDS):
            raise ValueError("engine settings differ from the intended compiled domain")
        if (engine.get("enable_return_routed_experts", False) or engine.get("async_scheduling") is not False
                or engine.get("enable_prefix_caching") is not False
                or engine.get("speculative_config") != expected_spec(plan, budget)):
            raise ValueError("instrumented/asynchronous/prefix-cache engine or wrong speculative configuration")
        if env.get("vllm") != "0.26.0" or env.get("VLLM_USE_V2_MODEL_RUNNER") != "0":
            raise ValueError("expected vLLM0.26 explicit V1 runner")
        keys = ("vllm", "torch", "cuda", "gpu", "model_runner_version", "VLLM_USE_V2_MODEL_RUNNER")
        if any(not env.get(key) for key in keys):
            raise ValueError("incomplete environment provenance")
        environments.append({key: env[key] for key in keys})
        if any(env.get("source_sha256", {}).get(name) != digest for name, digest in source_hashes.items()):
            raise ValueError("captured source differs from actual engine source")
        initial = read(folder / "evaluation_initial_state.json")
        if any(initial.get(key) != value for key, value in (("warmup_offset", warmup["warmup_offset"]),
                ("shape_warmup_offset", warmup["shape_warmup_offset"]), ("input_offset", 0))):
            raise ValueError("actual initial state differs from declared warmup population")
        if initial.get("acceptance_snapshot_before", {}).get("alpha_estimator") != "exposure_ratio":
            raise ValueError("initial acceptance state must use exposure_ratio")
        initial_states.append({"gamma": gamma, "source": str(folder / "evaluation_initial_state.json"),
                               "sha256": sha(folder / "evaluation_initial_state.json"), **initial})
        required = {folder / f"perf-r{repeat}-b{batch}.json" for repeat in range(repeats) for batch in batches}
        if set(folder.glob("perf-*.json")) != required:
            raise ValueError("missing or extra performance runs in compiled campaign")
        for path in sorted(required):
            record, context = read(path), _context(path)
            batch = int(path.stem.rsplit("-b", 1)[1])
            if route_instrumented(record, context) or not record.get("all_finished"):
                raise ValueError("incomplete or route-instrumented performance population")
            if (record.get("batch"), record.get("gamma"), record.get("proposal_budget"), record.get("mode")) != (batch, gamma, budget, mode):
                raise ValueError("performance B/gamma/budget/mode differs from the declared grid")
            requests = _requests(record)
            if len(requests) != 64 or sorted(row.get("source_index") for row in requests) != development_ids:
                raise ValueError("performance population differs from the 64 development requests")
            if any(not row.get("finished") or len(row.get("output_token_ids", [])) != length
                   or not positive(row.get("effective_tpot_ms")) for row in requests):
                raise ValueError("incomplete request outputs or unmeasured effective TPOT")
            if record.get("output_tokens") != 64 * length or not positive(record.get("wall_s")):
                raise ValueError("invalid output count or wall time")
            for episode in record.get("episodes", [record]):
                plugin = episode.get("plugin", {})
                if len(episode.get("host_steps", [])) != len(plugin.get("steps", [])):
                    raise ValueError("host/plugin step mismatch")
                if plugin.get("summary", {}).get("timing_eligible") is False:
                    raise ValueError("runtime marked this episode ineligible for performance")
            regime = resolve_regime(record, context)
            if regime["execution_mode"] != "compiled" or regime["measurement_kind"] != "fixed_length_generation":
                raise ValueError("compiled fixed-length performance records required")
            regimes.append(regime)
            if calibration:
                cost = analyze_run(record, str(path), context)["calibration"]
                if cost is None or cost["eligible_steps"] <= 0:
                    raise ValueError("cost cell lacks full-B nonterminal decode measurements")
                costs.append(cost)
            paths.append(path)
            provenance.append({"source": str(path), "sha256": sha(path)})
    if any(row != regimes[0] for row in regimes) or any(row != environments[0] for row in environments):
        raise ValueError("mixed execution or environment within the campaign")
    return {"path": str(directory), "files": paths, "source_files": provenance, "costs": costs,
        "source_inputs_sha256": hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest(),
        "regime": regimes[0], "environment": environments[0], "domain": sig,
        "runtime_behavior": behaviors, "runtime_source_sha256": source_hashes,
        "workload_sha256": workload_hash, "development_source_indices": development_ids,
        "heldout_source_indices": heldout_ids, "evaluation_initial_states": initial_states, **warmup,
        "validated_cells": len(batches) * len(GAMMAS), "validated_performance_runs": len(paths),
        "plan_sha256": sha(directory / "plan.json"), "status_sha256": sha(directory / "status.json")}


def validate_cost_grid(records):
    observed = Counter((row["batch"], row["gamma"], row.get("proposal_budget")) for row in records)
    if set(observed) != {(b, g, 7) for b in BATCHES for g in GAMMAS} or any(n != 1 for n in observed.values()):
        raise ValueError("need exactly 32 measured budget7 cost cells")
    if any(row.get("evidence_kind") != "measured_complete_step_cost" or row.get("eligible_steps", 0) <= 0 for row in records):
        raise ValueError("unmeasured complete-step cost cell")
    if CalibrationTable(records).gammas != GAMMAS:
        raise ValueError("compiled adaptive action set must be gamma0..7")


def make_commands(output, selection_dir, selection, config_path):
    gamma = selection["static"]["selected_gamma"]
    definitions = [("ar", 0, 0, "native", []), ("static", gamma, gamma, "native", []),
        ("native_unrestricted", 7, 7, "native", ["--native-table", str(selection_dir / "native_unrestricted_table.json")]),
        ("native_threshold", 7, 7, "native", ["--native-table", str(selection_dir / "native_threshold_table.json")]),
        ("adaptive", 7, 7, "adaptive", ["--calibration", str(output / "calibration/calibration.json")])]
    commands = []
    for role, gamma, budget, mode, extra in definitions:
        run = f"olmoe_compiled_controller_eval_{role}_r01"
        if (ROOT / "runs" / run).exists():
            raise FileExistsError(f"evaluation run already exists; do not resubmit: {run}")
        argv = [sys.executable, "-u", "code/run_campaign.py", "--config", str(config_path),
            "--run", run, "--gammas", str(gamma), "--mode", mode, "--proposal-budget", str(budget),
            "--wait-lock-seconds", "21600", "--skip-trace", *extra]
        commands.append({"role": role, "run": run, "config": str(config_path), "argv": argv,
            "cwd": str(ROOT), "status_file": str(ROOT / "runs" / run / "status.json"),
            "expected_performance_run_records": 8})
    return commands


def prepare(calibration_run, direct_run, output, evaluation_config):
    output, evaluation_config = Path(output).resolve(), Path(evaluation_config).resolve()
    if output.exists():
        raise FileExistsError("choose a fresh preparation output directory")
    plan = read(evaluation_config)
    check_config(plan, calibration=False)
    protocol_path = ROOT / "configs/controller_protocol.json"
    if sha(protocol_path) != PROTOCOL_SHA:
        raise ValueError("declared controller objective/protocol changed")
    protocol = read(protocol_path)
    parsed = validate_protocol(protocol)
    if parsed["batches"] != BATCHES or parsed["gammas"] != GAMMAS:
        raise ValueError("unexpected compiled controller action grid")
    # Validate all dependencies before creating any preparation artifacts.
    full = inspect_campaign(calibration_run, plan, calibration=True)
    native = inspect_campaign(direct_run, plan, calibration=False)
    for key in ("regime", "environment", "runtime_behavior", "domain", "workload_sha256"):
        if full[key] != native[key]:
            raise ValueError(f"compiled native selection and budget7 calibration differ in {key}")
    validate_cost_grid(full["costs"])
    selection_files = [path for path in native["files"] if int(path.stem.rsplit("-b", 1)[1]) in BATCHES]
    rows, exclusions = load_runs(selection_files)
    selected = select_regime(rows, protocol)
    if (exclusions or len(rows) != 64 or selected["status"] != "selected_pending_evaluation"
            or selected["exclusions"] or len(selected["measured_cells"]) != 32
            or any(cell["n_runs"] != 2 for cell in selected["measured_cells"])):
        raise ValueError("selection requires all 32 protocol cells with two original run values")
    if AcceptanceEWMA().snapshot().get("alpha_estimator") != "exposure_ratio":
        raise ValueError("expected exposure_ratio acceptance estimator")
    native["selection_performance_runs"] = len(selection_files)
    native["selection_source_files"] = [{"source": str(path), "sha256": sha(path)} for path in selection_files]
    make_commands(output, output / "selection", selected, evaluation_config)
    output.mkdir(parents=True)
    subprocess.run([sys.executable, "-B", str(ROOT / "code/build_calibration.py"), full["path"],
                    "--output", str(output / "calibration")], check=True, cwd=ROOT)
    table_path = output / "calibration/calibration.json"
    table = read(table_path)
    validate_cost_grid(table["records"])
    if read(output / "calibration/exclusions.json"):
        raise ValueError("complete-step cost builder excluded calibration evidence")
    CalibrationTable(table["records"], table["metadata"])
    manifest = write_selection(output / "selection", protocol, protocol_path, rows, [], [])
    if len(manifest["regimes"]) != 1:
        raise ValueError("unexpected multiple compiled selection domains")
    selection_path = Path(manifest["regimes"][0]["selection"])
    selection = read(selection_path)
    commands = make_commands(output, selection_path.parent, selection, evaluation_config)
    evidence = {"schema_version": 1, "status": "ready_for_serialized_evaluation", "domain": DOMAIN,
        "candidate_gammas": GAMMAS, "commands": commands,
        "calibration_table": str(table_path), "calibration_sha256": sha(table_path), "calibration_cells": 32,
        "selection": str(selection_path), "selection_sha256": sha(selection_path),
        "protocol": str(protocol_path), "protocol_sha256": sha(protocol_path),
        "evaluation_config": str(evaluation_config), "evaluation_config_sha256": sha(evaluation_config),
        "calibration_evidence": {key: value for key, value in full.items() if key not in ("files", "costs")},
        "selection_evidence": {key: value for key, value in native.items() if key not in ("files", "costs")},
        "current_source_sha256": {name: sha(ROOT / "code" / name) for name in CRITICAL_SOURCE},
        "budget": {"group_gpu_budget_seconds": plan["group_gpu_budget_seconds"],
            "cumulative_gpu_budget_seconds": plan["total_gpu_budget_seconds"],
            "enforced_by": "existing run_campaign.py shared lock and cumulative budget"},
        "execution_policy": "Generated only; coordinator executes five distinct campaigns serially and checks actual completion.",
        "boundaries": [
            "Selection uses only native compiled development records0-63 at protocol B1/4/16/64 after validating the full 56-cell matrix. No eager costs or choices are reused.",
            "All simple baselines are selected by the unchanged effective-TPOT objective; gamma0, threshold cutoff0 and all-zero native tables are legal outcomes, not skipped evaluations.",
            "Native tables use an engine max budget7 even if every entry is zero. This is distinct from AR and direct static gamma0 engines; their actual evaluation remains separate.",
            "Fixed-budget7 post-proposal gamma0 still pays draft-generation cost and is not AR. Adaptive lookup contains all 32 measured budget7 cells.",
            "Measured development0-63 and evaluation64-127 populations are disjoint. Historical native general warmup exposure is recorded, so the full pipeline is not claimed prompt-unseen.",
            "New calibration/evaluation warmup begins at0. Initial EWMA state continues across warmup, waves, batches and repeats; per-engine snapshots are retained.",
            "Compiled and graph-capable configuration does not establish timed-step FULL/PIECEWISE dispatch fractions; route-off performance runs do not record them.",
            "One cost repeat is calibration evidence. Evaluation keeps two raw repeats; requests and steps are not independent system repeats. No natural-stop run is executed or generated here."]}
    write(output / "evaluation_commands.json", evidence)
    (output / "evaluation_commands.txt").write_text("# Generated CPU preparation; no command executed here.\n" + "\n".join(shlex.join(command["argv"]) for command in commands) + "\n")
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--calibration-run", "--fullbudget", default="runs/olmoe_compiled_fullbudget_calibration_r01")
    parser.add_argument("--direct-run", default="runs/olmoe_ngram_compiled_r01")
    parser.add_argument("--evaluation-config", default=str(ROOT / "configs/olmoe_ngram_compiled_controller_evaluation.json"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = prepare(args.calibration_run, args.direct_run, args.output, args.evaluation_config)
    print(json.dumps({"status": result["status"], "calibration_cells": result["calibration_cells"],
        "selection": result["selection"], "commands_file": str(Path(args.output) / "evaluation_commands.json")}))


if __name__ == "__main__":
    main()
