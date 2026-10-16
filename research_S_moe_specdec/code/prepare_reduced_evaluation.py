#!/usr/bin/env python3
"""Prepare, but never execute, five held-out policies for one reduced domain.

  python code/prepare_reduced_evaluation.py --domain olmoe_dense_draft --describe
  python code/prepare_reduced_evaluation.py --domain olmoe_dense_draft

Defaults: native inputs runs/DOMAIN_reduced_r01, fixed-budget inputs
runs/DOMAIN_reduced_calibration_r01, output runs/DOMAIN_controller_preparation_r01.
DOMAIN may be any safe name matching [A-Za-z0-9][A-Za-z0-9_-]* with frozen
configs/DOMAIN_reduced{,_calibration,_evaluation}.json files. New names undergo
the same configuration, campaign, population and source checks as existing ones.
The native 12-cell, two-repeat development matrix selects the simple baselines.
A separate 12-cell, one-repeat static post-truncation matrix supplies complete
budget-7 costs. Only gamma={0,3,7} is available to the adaptive controller.
Both campaigns must finish before any output is created. The emitted five argv
arrays are for a caller which serializes run_campaign and checks actual status.
No GPU, remote connection, download or policy evaluation is started here.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import unittest

from build_calibration import analyze_run
from cost_model import AcceptanceEWMA, CalibrationTable
from plot_results import _context, _requests, load_runs, resolve_regime, route_instrumented
from select_controller_baselines import select_regime, validate_protocol, write_selection


ROOT = Path(__file__).resolve().parents[1]
DOMAINS = ("olmoe_dense_draft", "qwen3_awq_ngram", "qwen3_awq_eagle3",
           "granite3b_ngram", "granite3b_neural_draft")
BATCHES, GAMMAS = [1, 4, 16, 64], [0, 3, 7]
ROLES = ("ar", "static", "native_unrestricted", "native_threshold", "adaptive")
RUNTIME_SOURCE = ("run_matrix.py", "vllm_s_plugin.py", "cost_model.py", "power.py")


def domain_name(value):
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value) is None:
        raise ValueError("domain must be one safe filename component matching [A-Za-z0-9][A-Za-z0-9_-]*")
    return value


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def behavior_sha(path):
    tree = ast.parse(Path(path).read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
                node.body = node.body[1:]
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()


def domain_signature(plan):
    defaults = {"model_revision": None, "draft_model": None, "draft_revision": None,
        "speculative_overrides": {}, "dtype": "bfloat16", "quantization": None,
        "kv_cache_memory_bytes": 6 * 1024**3, "max_model_len": 1024,
        "max_num_seqs": 64, "max_num_batched_tokens": 16384, "enforce_eager": True,
        "kernel_config": {"moe_backend": "triton"}, "admission_mode": "wave",
        "semantic_token_boundary": None}
    return {**{k: plan.get(k, v) for k, v in defaults.items()},
            **{k: plan[k] for k in ("model", "draft_source", "workload")}}


def check_config(plan, *, calibration):
    required = {"batches": BATCHES, "gammas": GAMMAS, "measured_requests": 64,
        "repeats": 1 if calibration else 2, "output_tokens": 64 if calibration else 128,
        "input_offset": 0 if calibration else 64, "warmup_offset": 0,
        "shape_warmup_offset": 0, "warmup_each_batch": True, "max_num_seqs": 64,
        "admission_mode": "wave", "skip_trace": True, "native_route_return": False,
        "candidate_gammas": GAMMAS}
    for key, value in required.items():
        if plan.get(key) != value:
            raise ValueError(f"invalid reduced {'calibration' if calibration else 'evaluation'} config: {key}")


def require_complete(status):
    cells = status.get("cells", [])
    if (status.get("status") != "complete" or len(cells) != 3
            or sorted(c.get("gamma", -1) for c in cells) != GAMMAS
            or any(c.get("returncode") != 0 for c in cells)):
        raise ValueError("campaign must contain exactly three completed successful gamma cells: 0,3,7")


def expected_spec(plan, budget):
    if budget == 0:
        return None
    spec = {"method": plan["draft_source"], "num_speculative_tokens": budget}
    if plan["draft_source"] in ("ngram", "ngram_gpu"):
        spec.update(prompt_lookup_min=2, prompt_lookup_max=5)
    if plan.get("draft_model"):
        spec["model"] = plan["draft_model"]
    spec.update(plan.get("speculative_overrides", {}))
    return spec


def inspect_campaign(directory, expected_plan, *, calibration):
    """Require completed populations and reuse the existing cost/metric readers."""
    directory = Path(directory).resolve()
    require_complete(read(directory / "status.json"))
    plan = read(directory / "plan.json")
    if domain_signature(plan) != domain_signature(expected_plan):
        raise ValueError("campaign model/draft/resource/execution domain differs from its target config")
    repeats, length, mode = (1, 64, "static") if calibration else (2, 128, "native")
    if (plan.get("batches"), plan.get("gammas"), plan.get("repeats"), plan.get("output_tokens"), plan.get("measured_requests")) != (BATCHES, GAMMAS, repeats, length, 64):
        raise ValueError("campaign plan does not contain the required reduced grid/population")
    if plan.get("input_offset", 0) != 0:
        raise ValueError("baseline selection and cost calibration must use development offset0")
    if calibration:
        check_config(plan, calibration=True)
    workload = ROOT / plan["workload"]
    records = read(workload)["records"]
    if len(records) < 128:
        raise ValueError("workload does not contain the development and held-out populations")
    development_ids = sorted(r["source_index"] for r in records[:64])
    heldout_ids = sorted(r["source_index"] for r in records[64:128])
    if len(set(development_ids + heldout_ids)) != 128:
        raise ValueError("development and held-out request identities are not disjoint")
    workload_hash = sha(workload)
    if plan.get("workload_sha256") not in (None, workload_hash):
        raise ValueError("workload differs from the config's pinned content hash")
    paths, provenance, regimes, environments, behaviors, costs = [], [], [], [], [], []
    for gamma in GAMMAS:
        folder = directory / f"gamma{gamma}"
        if read(folder / "status.json").get("status") != "complete":
            raise ValueError(f"gamma{gamma} did not finish")
        config, engine, env = (read(folder / name) for name in ("configuration.json", "engine_args.json", "environment.json"))
        args = config["args"]
        if config.get("plan") != plan or config.get("workload_sha256") != workload_hash:
            raise ValueError("engine plan/workload does not match the campaign")
        budget = 7 if calibration else gamma
        if args.get("mode") != mode or args.get("gamma") != gamma or args.get("natural_stop", False) or args.get("native_table"):
            raise ValueError("wrong execution mode, gamma, termination rule or unexpected native table")
        for key, expected in (("batches", BATCHES), ("repeats", repeats), ("output_tokens", length)):
            if args.get(key) is not None and args[key] != expected:
                raise ValueError(f"unexpected runtime {key} override")
        if (args.get("proposal_budget") if args.get("proposal_budget") is not None else gamma) != budget:
            raise ValueError("native direct budgets cannot substitute for fixed-budget7 truncation costs")
        sig = domain_signature(plan)
        for key in ("model", "dtype", "quantization", "kv_cache_memory_bytes", "max_model_len", "max_num_seqs", "max_num_batched_tokens", "enforce_eager", "kernel_config"):
            if engine.get(key) != sig[key]:
                raise ValueError(f"engine {key} differs from the intended domain")
        if engine.get("enable_return_routed_experts", False) or engine.get("async_scheduling") is not False or engine.get("speculative_config") != expected_spec(plan, budget):
            raise ValueError("instrumented/asynchronous engine or mismatched speculative configuration")
        if env.get("vllm") != "0.26.0" or env.get("VLLM_USE_V2_MODEL_RUNNER") != "0":
            raise ValueError("expected vLLM0.26 with explicit V1 runner")
        if any(not env.get(key) for key in ("torch", "cuda", "gpu", "model_runner_version")):
            raise ValueError("incomplete software/device environment provenance")
        environments.append({key: env.get(key) for key in ("vllm", "torch", "cuda", "gpu", "model_runner_version", "VLLM_USE_V2_MODEL_RUNNER")})
        source_behavior, source_hashes = {}, {}
        for name in RUNTIME_SOURCE:
            captured = directory / "source" / name
            source_hashes[name] = sha(captured)
            if env.get("source_sha256", {}).get(name) != source_hashes[name]:
                raise ValueError(f"captured source does not match actual engine source: {name}")
            source_behavior[name] = behavior_sha(captured)
            if calibration and source_behavior[name] != behavior_sha(ROOT / "code" / name):
                raise ValueError(f"runtime behavior changed since cost calibration: {name}")
        behaviors.append(source_behavior)
        required_paths = {folder / f"perf-r{r}-b{b}.json" for r in range(repeats) for b in BATCHES}
        if set(folder.glob("perf-*.json")) != required_paths:
            raise ValueError("missing or extra performance run files")
        for path in sorted(required_paths):
            record, context = read(path), _context(path)
            if route_instrumented(record, context) or not record.get("all_finished"):
                raise ValueError("incomplete or instrumented performance run")
            batch = int(path.stem.rsplit("-b", 1)[1])
            if (record.get("batch"), record.get("gamma"), record.get("proposal_budget"), record.get("mode")) != (batch, gamma, budget, mode):
                raise ValueError("performance B/gamma/budget/mode differs from the declared grid")
            requests = _requests(record)
            if len(requests) != 64 or sorted(r.get("source_index") for r in requests) != development_ids:
                raise ValueError("performance run does not contain the complete development population")
            if any(not r.get("finished") or len(r.get("output_token_ids", [])) != length or not positive(r.get("effective_tpot_ms")) for r in requests):
                raise ValueError("incomplete outputs or unmeasured request effective TPOT")
            if record.get("output_tokens") != 64 * length or not positive(record.get("wall_s")):
                raise ValueError("invalid complete-population token count or wall time")
            regime = resolve_regime(record, context)
            if regime["measurement_kind"] != "fixed_length_generation":
                raise ValueError("only fixed-length performance populations are supported")
            regimes.append(regime)
            if calibration:
                cost = analyze_run(record, str(path), context)["calibration"]
                if cost is None or cost["eligible_steps"] <= 0:
                    raise ValueError("cost cell has no full-B nonterminal decode measurements")
                costs.append(cost)
            paths.append(path)
            provenance.append({"source": str(path), "sha256": sha(path)})
    for group in (regimes, environments, behaviors):
        if any(row != group[0] for row in group):
            raise ValueError("mixed execution, environment or runtime source within a campaign")
    return {"path": str(directory), "files": paths, "source_files": provenance,
        "regime": regimes[0], "environment": environments[0], "runtime_behavior": behaviors[0],
        "runtime_source_sha256": source_hashes, "domain": domain_signature(plan),
        "workload_sha256": workload_hash, "development_source_indices": development_ids,
        "heldout_source_indices": heldout_ids, "costs": costs,
        "warmup_offset": plan.get("warmup_offset", 80),
        "shape_warmup_offset": plan.get("shape_warmup_offset", plan.get("input_offset", 0)),
        "plan_sha256": sha(directory / "plan.json"), "status_sha256": sha(directory / "status.json")}


def positive(value):
    return isinstance(value, (int, float)) and math.isfinite(value) and value > 0


def validate_cost_grid(records):
    observed = Counter((r["batch"], r["gamma"], r.get("proposal_budget")) for r in records)
    if set(observed) != {(b, g, 7) for b in BATCHES for g in GAMMAS} or any(n != 1 for n in observed.values()):
        raise ValueError("need exactly 12 measured budget7 cost cells; native different-budget timings are insufficient")
    if any(r.get("evidence_kind") != "measured_complete_step_cost" or r.get("eligible_steps", 0) <= 0 for r in records):
        raise ValueError("unmeasured complete-step cost cell")
    if CalibrationTable(records).gammas != GAMMAS:
        raise ValueError("adaptive action set must be exactly {0,3,7}")


def make_commands(domain, output, selection_dir, selection, config_path, prefix=None):
    gamma = selection["static"]["selected_gamma"]
    definitions = [("ar", 0, 0, "native", []), ("static", gamma, gamma, "native", []),
        ("native_unrestricted", 7, 7, "native", ["--native-table", str(selection_dir / "native_unrestricted_table.json")]),
        ("native_threshold", 7, 7, "native", ["--native-table", str(selection_dir / "native_threshold_table.json")]),
        ("adaptive", 7, 7, "adaptive", ["--calibration", str(output / "calibration/calibration.json")])]
    prefix = prefix or f"{domain}_controller_eval_r01"
    if Path(prefix).name != prefix or prefix in (".", ".."):
        raise ValueError("run prefix must be one directory-name component")
    commands = []
    for role, gamma, budget, mode, extra in definitions:
        run = f"{prefix}_{role}"
        if (ROOT / "runs" / run).exists():
            raise FileExistsError(f"existing evaluation run must not be resubmitted: {run}")
        argv = [sys.executable, "-u", "code/run_campaign.py", "--config", str(config_path),
            "--run", run, "--gammas", str(gamma), "--mode", mode, "--proposal-budget", str(budget),
            "--skip-trace", "--wait-lock-seconds", "21600", *extra]
        commands.append({"role": role, "run": run, "config": str(config_path), "argv": argv,
            "cwd": str(ROOT), "status_file": str(ROOT / "runs" / run / "status.json"),
            "expected_performance_run_records": 8})
    return commands


def describe(domain, native_run=None, calibration_run=None, output=None):
    domain = domain_name(domain)
    calibration_config = ROOT / "configs" / f"{domain}_reduced_calibration.json"
    evaluation_config = ROOT / "configs" / f"{domain}_reduced_evaluation.json"
    base = read(ROOT / "configs" / f"{domain}_reduced.json")
    for path, calibration in ((calibration_config, True), (evaluation_config, False)):
        plan = read(path)
        check_config(plan, calibration=calibration)
        if plan.get("preparation_domain") != domain or domain_signature(plan) != domain_signature(base):
            raise ValueError("derived config differs from the original reduced model/draft domain")
    calibration_run = Path(calibration_run or ROOT / "runs" / f"{domain}_reduced_calibration_r01").resolve()
    local_campaign = calibration_run.parent == (ROOT / "runs").resolve()
    return {"domain": domain,
        "native_run": str(Path(native_run or ROOT / "runs" / f"{domain}_reduced_r01").resolve()),
        "calibration_run": str(calibration_run),
        "output": str(Path(output or ROOT / "runs" / f"{domain}_controller_preparation_r01").resolve()),
        "calibration_config": str(calibration_config), "evaluation_config": str(evaluation_config),
        "calibration_argv": [sys.executable, "-u", "code/run_campaign.py", "--config", str(calibration_config),
            "--run", calibration_run.name, "--gammas", "0", "3", "7", "--mode", "static",
            "--proposal-budget", "7", "--skip-trace", "--wait-lock-seconds", "21600"] if local_campaign else None,
        "calibration_command_status": "generated_not_executed" if local_campaign else "external_input_directory_no_launch_command_inferred",
        "calibration_status_file": str(calibration_run / "status.json"),
        "calibration_cwd": str(ROOT), "expected_calibration_cells": 12,
        "candidate_gammas": GAMMAS, "status": "description_only_no_execution"}


def prepare(info, prefix=None):
    output = Path(info["output"])
    if output.exists():
        raise FileExistsError("choose a fresh preparation output directory")
    calibration_plan, evaluation_plan = read(info["calibration_config"]), read(info["evaluation_config"])
    if domain_signature(calibration_plan) != domain_signature(evaluation_plan):
        raise ValueError("calibration and evaluation configs differ in domain")
    protocol_path = ROOT / "configs/reduced_protocol.json"
    protocol = read(protocol_path)
    parsed_protocol = validate_protocol(protocol)
    if parsed_protocol["batches"] != BATCHES or parsed_protocol["gammas"] != GAMMAS:
        raise ValueError("reduced protocol changed the declared action grid")
    # No partial output if a dependency is still running or has missing evidence.
    full = inspect_campaign(info["calibration_run"], calibration_plan, calibration=True)
    native = inspect_campaign(info["native_run"], calibration_plan, calibration=False)
    for key in ("regime", "environment", "runtime_behavior", "domain", "workload_sha256"):
        if full[key] != native[key]:
            raise ValueError(f"native selection and budget7 cost calibration differ in {key}")
    validate_cost_grid(full["costs"])
    rows, exclusions = load_runs(native["files"])
    selected = select_regime(rows, protocol)
    if exclusions or len(rows) != 24 or selected["status"] != "selected_pending_evaluation" or selected["exclusions"] or any(c["n_runs"] != 2 for c in selected["measured_cells"]):
        raise ValueError("baseline selection needs all 12 direct-draft cells with two original run values")
    if AcceptanceEWMA().snapshot().get("alpha_estimator") != "exposure_ratio":
        raise ValueError("expected the exposure_ratio acceptance estimator")
    # Check duplicate run names before any output is created.
    make_commands(info["domain"], output, output / "selection", selected, Path(info["evaluation_config"]), prefix)
    output.mkdir(parents=True)
    subprocess.run([sys.executable, "-B", str(ROOT / "code/build_calibration.py"),
        full["path"], "--output", str(output / "calibration")], check=True, cwd=ROOT)
    table_path = output / "calibration/calibration.json"
    table = read(table_path)
    validate_cost_grid(table["records"])
    if read(output / "calibration/exclusions.json"):
        raise ValueError("cost builder excluded calibration evidence")
    manifest = write_selection(output / "selection", protocol, protocol_path, rows, [], [])
    if len(manifest["regimes"]) != 1:
        raise ValueError("unexpected multiple baseline selection domains")
    selection_path = Path(manifest["regimes"][0]["selection"])
    selection = read(selection_path)
    commands = make_commands(info["domain"], output, selection_path.parent, selection, Path(info["evaluation_config"]), prefix)
    evidence = {**info, "schema_version": 1, "status": "ready_for_serialized_evaluation", "commands": commands,
        "calibration_table": str(table_path), "calibration_sha256": sha(table_path), "calibration_cells": 12,
        "selection": str(selection_path), "selection_sha256": sha(selection_path),
        "protocol": str(protocol_path), "protocol_sha256": sha(protocol_path),
        "evaluation_config_sha256": sha(info["evaluation_config"]),
        "calibration_evidence": {k: v for k, v in full.items() if k not in ("files", "costs")},
        "selection_evidence": {k: v for k, v in native.items() if k not in ("files", "costs")},
        "budget": {"group_gpu_budget_seconds": evaluation_plan["group_gpu_budget_seconds"],
            "cumulative_gpu_budget_seconds": evaluation_plan["total_gpu_budget_seconds"],
            "enforced_by": "existing run_campaign.py shared lock and cumulative budget"},
        "boundaries": [
            "Adaptive candidates are exactly gamma={0,3,7}, not all integers0..7; runtime tail batches use existing within-range B interpolation.",
            "Native development runs alone select all simple baselines; held-out prompt64-127 runs never reselect them.",
            "Only measured budget7 post-truncation complete-step costs feed adaptive decisions. Native gamma0 is AR; post-truncation gamma0 still pays full draft cost.",
            "Selection scores and the per-B native table are not measured policy performance or a future-information oracle; all five policies require actual separate evaluation.",
            "Evaluation warmup and initial EWMA use development prompts0-63; online state then remains continuous across waves, batches and repeats.",
            "Native development general/shape warmup offsets are retained in selection_evidence; prompt separation is assessed from those actual offsets, rather than an assumed runner default.",
            "One cost repeat per cell is calibration evidence, not independent confirmation. Evaluation retains two raw run values; requests/steps are not independent repeats."]}
    write(output / "evaluation_commands.json", evidence)
    (output / "evaluation_commands.txt").write_text("# Generated CPU preparation; none of these commands has been executed here.\n" + "\n".join(shlex.join(c["argv"]) for c in commands) + "\n")
    return evidence


def self_test():
    from unittest.mock import patch
    import shutil
    class Checks(unittest.TestCase):
        def test_complete_fixture_uses_real_builders(self):
            # Entirely synthetic CPU fixtures live only in a deleted temporary
            # directory. No synthetic cost or performance artifact enters runs/.
            domain = DOMAINS[0]
            original_root = ROOT
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                (root / "configs").mkdir()
                (root / "code").mkdir()
                for name in (*RUNTIME_SOURCE, "build_calibration.py", "plot_results.py"):
                    shutil.copyfile(original_root / "code" / name, root / "code" / name)
                for name in (f"{domain}_reduced.json", f"{domain}_reduced_calibration.json",
                             f"{domain}_reduced_evaluation.json", "reduced_protocol.json"):
                    shutil.copyfile(original_root / "configs" / name, root / "configs" / name)
                workload = read(original_root / "configs/sharegpt_olmoe.json")
                write(root / "configs/sharegpt_olmoe.json", workload)
                for calibration in (False, True):
                    suffix = "reduced_calibration" if calibration else "reduced"
                    plan = read(root / "configs" / f"{domain}_{suffix}.json")
                    run = root / "runs" / f"{domain}_{suffix}_r01"
                    (run / "source").mkdir(parents=True)
                    hashes = {}
                    for name in RUNTIME_SOURCE:
                        shutil.copyfile(root / "code" / name, run / "source" / name)
                        hashes[name] = sha(run / "source" / name)
                    write(run / "plan.json", plan)
                    write(run / "status.json", {"status": "complete", "cells": [{"gamma": g, "returncode": 0} for g in GAMMAS]})
                    for gamma in GAMMAS:
                        cell = run / f"gamma{gamma}"
                        cell.mkdir()
                        budget, mode = (7, "static") if calibration else (gamma, "native")
                        write(cell / "status.json", {"status": "complete"})
                        write(cell / "configuration.json", {"plan": plan,
                            "args": {"mode": mode, "gamma": gamma, "proposal_budget": budget},
                            "workload_sha256": sha(root / plan["workload"])})
                        sig = domain_signature(plan)
                        engine = {k: sig[k] for k in ("model", "dtype", "quantization", "kv_cache_memory_bytes",
                            "max_model_len", "max_num_seqs", "max_num_batched_tokens", "enforce_eager", "kernel_config")}
                        engine.update(async_scheduling=False, speculative_config=expected_spec(plan, budget))
                        write(cell / "engine_args.json", engine)
                        write(cell / "environment.json", {"vllm": "0.26.0", "gpu": "synthetic_cpu_fixture",
                            "torch": "synthetic", "cuda": "synthetic", "VLLM_USE_V2_MODEL_RUNNER": "0",
                            "model_runner_version": "v1_explicit", "source_sha256": hashes})
                        for repeat in range(plan["repeats"]):
                            for batch in BATCHES:
                                label = f"perf-r{repeat}-b{batch}"
                                tpot = (10 if gamma == 0 else (6 if batch <= 4 else 9) if gamma == 3 else (8 if batch <= 4 else 5))
                                episodes = []
                                for offset in range(0, 64, batch):
                                    wave = f"{label}-w{offset // batch}"
                                    ids = [f"req{i}" for i in range(offset, offset + batch)]
                                    requests = [{"source_index": workload["records"][i]["source_index"],
                                        "finished": True, "output_token_ids": [1] * plan["output_tokens"],
                                        "effective_tpot_ms": tpot,
                                        "chunks": [{"elapsed_s": 2}]} for i in range(offset, offset + batch)]
                                    steps = [{"step": j, "episode": wave, "phase": "decode", "gamma": gamma,
                                        "decode_batch": batch, "scheduled_tokens": batch * (gamma + 1),
                                        "scheduled_decode_request_ids": ids, "proposal_budget_next": budget,
                                        "proposed_before_by_request": {rid: budget for rid in ids},
                                        "scheduled_drafts_by_request": {rid: gamma for rid in ids},
                                        "accepted_by_request": {rid: 0 for rid in ids},
                                        "ewma_censored_requests": ids if j == 1 else []} for j in (0, 1)]
                                    episodes.append({"label": wave, "requests": requests,
                                        "host_steps": [{"index": j, "elapsed_s": j + 1, "host_step_ms": 2 + gamma} for j in (0, 1)],
                                        "plugin": {"steps": steps}})
                                write(cell / f"{label}.json", {"label": label, "model": plan["model"],
                                    "draft_source": plan["draft_source"], "batch": batch, "gamma": gamma,
                                    "proposal_budget": budget, "mode": mode, "all_finished": True,
                                    "measurement_kind": "fixed_length_generation", "episodes": episodes,
                                    "wall_s": 2, "output_tokens": 64 * plan["output_tokens"]})
                with patch.dict(globals(), {"ROOT": root}):
                    info = describe(domain)
                    evidence = prepare(info)
                    self.assertEqual(evidence["calibration_cells"], 12)
                    self.assertEqual(read(evidence["selection"])["static"]["selected_gamma"], 7)
                    self.assertEqual([c["role"] for c in evidence["commands"]], list(ROLES))
                    self.assertEqual(CalibrationTable.from_json(evidence["calibration_table"]).gammas, GAMMAS)
                    self.assertTrue((Path(info["output"]) / "evaluation_commands.json").is_file())
                    with self.assertRaises(FileExistsError):
                        prepare(info)
                    broken = describe(domain, output=root / "runs/refused")
                    raw = Path(info["calibration_run"]) / "gamma0/perf-r0-b1.json"
                    obj = read(raw)
                    obj["proposal_budget"] = 0
                    write(raw, obj)
                    with self.assertRaisesRegex(ValueError, "budget/mode"):
                        prepare(broken)
                    self.assertFalse(Path(broken["output"]).exists())

        def test_all_domain_interfaces(self):
            import run_campaign
            original_parse = argparse.ArgumentParser.parse_args
            class Parsed(Exception):
                pass
            for domain in DOMAINS:
                info = describe(domain)
                self.assertEqual(domain_signature(read(info["calibration_config"])), domain_signature(read(info["evaluation_config"])))
                selection = {"static": {"selected_gamma": 3}}
                with patch.object(Path, "exists", return_value=False):
                    commands = make_commands(domain, Path("/tmp/prep"), Path("/tmp/selection"), selection, Path(info["evaluation_config"]))
                self.assertEqual([c["role"] for c in commands], list(ROLES))
                self.assertEqual(len({c["run"] for c in commands}), 5)
                for argv in [info["calibration_argv"]] + [c["argv"] for c in commands]:
                    self.assertEqual(argv[:3], [sys.executable, "-u", "code/run_campaign.py"])
                    self.assertEqual(argv.count("--wait-lock-seconds"), 1)
                    def stop(parser):
                        args = original_parse(parser, argv[3:])
                        self.assertEqual(args.wait_lock_seconds, 21600)
                        self.assertIsNone(args.batches)
                        self.assertIsNone(args.repeats)
                        self.assertIsNone(args.output_tokens)
                        raise Parsed
                    with patch.object(argparse.ArgumentParser, "parse_args", stop), self.assertRaises(Parsed):
                        run_campaign.main()

        def test_incomplete_and_different_budgets_rejected(self):
            good = {"status": "complete", "cells": [{"gamma": g, "returncode": 0} for g in GAMMAS]}
            require_complete(good)
            for obj in ({**good, "status": "running"}, {**good, "cells": good["cells"][:-1]}):
                with self.assertRaises(ValueError):
                    require_complete(obj)
            records = [{"batch": b, "gamma": g, "proposal_budget": 7, "total_step_ms": 2,
                "eligible_steps": 1, "evidence_kind": "measured_complete_step_cost"} for b in BATCHES for g in GAMMAS]
            validate_cost_grid(records)
            records[0]["proposal_budget"] = 0
            with self.assertRaises(ValueError):
                validate_cost_grid(records)

        def test_missing_dependency_creates_no_output(self):
            with tempfile.TemporaryDirectory() as folder:
                info = describe(DOMAINS[0], calibration_run=Path(folder) / "missing", output=Path(folder) / "output")
                with self.assertRaises(FileNotFoundError):
                    prepare(info)
                self.assertFalse(Path(info["output"]).exists())
    result = unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    if not result.wasSuccessful():
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--domain", type=domain_name,
        help="Safe filename component [A-Za-z0-9][A-Za-z0-9_-]*; requires all three frozen reduced configs")
    parser.add_argument("--native-run")
    parser.add_argument("--calibration-run")
    parser.add_argument("--output")
    parser.add_argument("--run-prefix", help="Unique evaluation directory prefix; defaults to DOMAIN_controller_eval_r01")
    parser.add_argument("--describe", action="store_true", help="Print calibration/evaluation interface without reading run results")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.domain:
        parser.error("--domain is required")
    info = describe(args.domain, args.native_run, args.calibration_run, args.output)
    if args.describe:
        print(json.dumps(info, indent=2))
        return
    result = prepare(info, args.run_prefix)
    print(json.dumps({"status": result["status"], "domain": args.domain, "calibration_cells": 12,
        "selected_static_gamma": read(result["selection"])["static"]["selected_gamma"],
        "commands_file": str(Path(info["output"]) / "evaluation_commands.json")}))


if __name__ == "__main__":
    main()
