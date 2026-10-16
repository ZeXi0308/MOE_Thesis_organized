#!/usr/bin/env python3
"""Compare logged selected-action costs with the same synchronous host steps.

  python code/analyze_policy_cost_prediction.py runs/ADAPTIVE_CAMPAIGN \
    --output runs/ADAPTIVE_cost_transfer

The campaign's copied calibration.json is used by default. --calibration must
match that actual artifact. Supports both full and reduced action grids.
Only the executed action is scored: no unchosen-action, ranking or service
benefit inference. Native missing predictions remain pending, never zero.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

from analyze_acceptance_tracking import iter_steps, write_csv
from analyze_online_acceptance import execution_order
from compare_reduced_evaluations import ENV_FIELDS, relocate
from cost_model import CalibrationTable
from plot_results import REGIME_FIELDS, _context, _finite, _requests, resolve_regime, route_instrumented
from prepare_reduced_evaluation import RUNTIME_SOURCE, behavior_sha, domain_signature, expected_spec


class Pending(ValueError):
    """Evidence is not present; no invented prediction or measurement."""


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def positive(value):
    value = _finite(value)
    return value is not None and value > 0


def engine_evidence(folder):
    folder = Path(folder)
    config, engine, env = (read(folder / name) for name in ("configuration.json", "engine_args.json", "environment.json"))
    plan, args = config["plan"], config["args"]
    spec = engine.get("speculative_config") or {}
    budget = args.get("proposal_budget")
    budget = args["gamma"] if budget is None else budget
    if int(spec.get("num_speculative_tokens", 0)) != budget:
        raise ValueError("engine/config proposal budget mismatch")
    if engine.get("async_scheduling") is not False or engine.get("enable_return_routed_experts", False):
        raise ValueError("cost transfer requires synchronous, route-off performance execution")
    if env.get("vllm") != "0.26.0" or env.get("VLLM_USE_V2_MODEL_RUNNER") != "0":
        raise ValueError("unsupported engine: require vLLM0.26 explicit V1")
    if any(not env.get(key) for key in ENV_FIELDS):
        raise Pending("incomplete execution environment provenance")
    base_spec = {key: value for key, value in spec.items() if key != "num_speculative_tokens_per_batch_size"}
    if (base_spec or None) != expected_spec(plan, budget):
        raise ValueError("actual speculative source/budget differs from its config")
    if args.get("mode") == "adaptive" and ("num_speculative_tokens_per_batch_size" in spec or args.get("native_table")):
        raise ValueError("adaptive engine unexpectedly uses a native table")
    domain = domain_signature(plan)
    for key in ("model", "dtype", "quantization", "kv_cache_memory_bytes", "max_model_len", "max_num_seqs",
                "max_num_batched_tokens", "enforce_eager", "kernel_config"):
        if engine.get(key) != domain[key]:
            raise ValueError(f"engine/config domain mismatch: {key}")
    source_hashes, behavior = {}, {}
    for name in RUNTIME_SOURCE:
        path = folder.parent / "source" / name
        source_hashes[name] = sha(path)
        if env.get("source_sha256", {}).get(name) != source_hashes[name]:
            raise ValueError(f"recorded/captured runtime source mismatch: {name}")
        behavior[name] = behavior_sha(path)
    return {"config": config, "engine": engine, "budget": budget, "domain": domain,
        "environment": {key: env.get(key) for key in ENV_FIELDS}, "runtime_behavior": behavior,
        "runtime_source_sha256": source_hashes, "configuration_sha256": sha(folder / "configuration.json"),
        "engine_args_sha256": sha(folder / "engine_args.json")}


def validate_table(table, regime, budget):
    for key in REGIME_FIELDS:
        if key not in table.metadata or str(table.metadata[key]) != str(regime[key]):
            raise ValueError(f"calibration/evaluation execution-domain mismatch: {key}")
    if not table.cells or any(key[2] != budget for key in table.cells):
        raise ValueError("calibration/evaluation production budget mismatch")


class CalibrationEvidence:
    """Read each measured calibration source once, without refitting its costs."""
    def __init__(self, path, execution):
        self.path = Path(path).resolve()
        self.hash = sha(self.path)
        obj = read(self.path)
        self.table = CalibrationTable(obj["records"], obj.get("metadata"))
        self.prompt_ids, self.sources = set(), []
        self.records = obj["records"]
        source_engines = {}
        sources = sorted({row.get("source") for row in self.records if row.get("source")})
        if not sources or any(not row.get("source") for row in self.records):
            raise Pending("calibration lacks measured source provenance")
        for row in self.records:
            if row.get("evidence_kind") != "measured_complete_step_cost" or not positive(row.get("total_step_ms")) or row.get("eligible_steps", 0) <= 0:
                raise ValueError("calibration must contain measured complete-step costs")
            if any(key in row and str(row[key]) != str(self.table.metadata.get(key)) for key in REGIME_FIELDS):
                raise ValueError("calibration row/metadata execution-domain mismatch")
        for value in sources:
            source = relocate(value)
            record = read(source)
            context = _context(source)
            regime = resolve_regime(record, context)
            validate_table(self.table, regime, execution["budget"])
            if route_instrumented(record, context) or record.get("all_finished") is not True or record.get("mode") != "static":
                raise ValueError("calibration source is not a complete route-off static population")
            if record.get("proposal_budget") != execution["budget"]:
                raise ValueError("calibration source production budget mismatch")
            for row in self.records:
                if row["source"] == value and any(row[key] != record.get(key) for key in ("batch", "gamma", "proposal_budget")):
                    raise ValueError("calibration row differs from its measured source action/batch/budget")
            if source.parent not in source_engines:
                source_engines[source.parent] = engine_evidence(source.parent)
            evidence = source_engines[source.parent]
            for key in ("domain", "environment", "runtime_behavior"):
                if evidence[key] != execution[key]:
                    raise ValueError(f"calibration/evaluation differs in {key}")
            ids = [row.get("source_index") for row in _requests(record)]
            if not ids or None in ids:
                raise Pending("calibration source request identities unavailable")
            self.prompt_ids.update(ids)
            self.sources.append({"source": str(source), "sha256": sha(source),
                                 "runtime_source_sha256": evidence["runtime_source_sha256"]})
        self.source_paths = {row["source"] for row in self.sources}

    def check_run(self, record, source, regime, execution):
        validate_table(self.table, regime, execution["budget"])
        ids = [row.get("source_index") for row in _requests(record)]
        if not ids or None in ids:
            raise Pending("evaluation request identities unavailable")
        if str(Path(source).resolve()) in self.source_paths or set(ids) & self.prompt_ids:
            raise ValueError("evaluation and calibration measured sources/prompts overlap")


def metrics(rows):
    pairs = [row for row in rows if row["eligible"]]
    if not pairs:
        return {"n": 0, "predicted_mean_ms": None, "observed_mean_ms": None,
                "bias_ms": None, "mae_ms": None, "rmse_ms": None}
    errors = [row["predicted_total_step_ms"] - row["observed_host_step_ms"] for row in pairs]
    return {"n": len(pairs), "predicted_mean_ms": statistics.mean(row["predicted_total_step_ms"] for row in pairs),
        "observed_mean_ms": statistics.mean(row["observed_host_step_ms"] for row in pairs),
        "bias_ms": statistics.mean(errors), "mae_ms": statistics.mean(abs(error) for error in errors),
        "rmse_ms": math.sqrt(statistics.mean(error * error for error in errors))}


def analyze_record(record, source, context, table=None):
    """Strict event identity matching; all aligned steps retain reasons/costs."""
    if route_instrumented(record, context) or record.get("all_finished") is not True:
        raise ValueError("require complete route-off performance populations")
    identity = {**resolve_regime(record, context), "source": str(source), "run_id": record.get("label"),
        "nominal_batch": record["batch"], "configured_gamma": record["gamma"],
        "proposal_budget": record["proposal_budget"], "mode": record.get("mode")}
    matched = re.search(r"perf-r(\d+)-b", str(record.get("label", source)))
    identity["repeat"] = int(matched.group(1)) if matched else None
    rows, pending = [], set()
    for wave_index, wave, host, step, config, observations, _ in iter_steps(record):
        decision = step.get("decision") or {}
        prediction, measured = _finite(decision.get("total_step_ms")), _finite(host.get("host_step_ms"))
        actual, control = step.get("decode_batch"), step.get("control_batch")
        selected = decision.get("gamma")
        reasons = []
        if step.get("phase") != "decode": reasons.append("not_pure_decode")
        if not isinstance(actual, int) or actual <= 0 or step.get("scheduled_requests") != actual or step.get("scheduled_tokens", 0) <= 0:
            reasons.append("not_complete_scheduled_decode_batch")
        if control is None: reasons.append("missing_precontrol_batch"); pending.add("missing_precontrol_batch")
        elif control != actual: reasons.append("precontrol_actual_batch_mismatch")
        if len(observations) != actual:
            reasons.append("missing_decode_request_identity"); pending.add("missing_decode_request_identity")
        if any(row["terminal"] for row in observations) or step.get("ewma_censored_requests"):
            reasons.append("terminal_or_output_budget_censored")
        if any(not row["terminal_known"] for row in observations):
            reasons.append("unknown_terminal_status"); pending.add("unknown_terminal_status")
        next_budget = step.get("proposal_budget_next")
        if next_budget is None: reasons.append("missing_next_proposal_budget"); pending.add("missing_next_proposal_budget")
        elif next_budget != identity["proposal_budget"]: reasons.append("next_proposal_budget_changed")
        if measured is None or measured <= 0:
            reasons.append("missing_or_invalid_host_step_ms"); pending.add("missing_or_invalid_host_step_ms")
        if identity["mode"] != "adaptive":
            reasons.append("not_adaptive_no_selected_cost_prediction"); pending.add("not_adaptive_no_selected_cost_prediction")
        elif config.get("mode") != "adaptive" or config.get("proposal_budget") != "max" or config.get("max_gamma") != identity["proposal_budget"]:
            raise ValueError("step plugin/config mode or production budget mismatch")
        if prediction is None:
            reasons.append("uncovered_calibration_fallback" if decision.get("fallback") else "missing_decision_cost")
            # Prefill with no existing decode requests legitimately has no decision.
            if step.get("phase") == "decode": pending.add(reasons[-1])
        elif prediction <= 0:
            raise ValueError("nonpositive recorded decision cost")
        if prediction is not None and identity["mode"] == "adaptive":
            missing_action = selected is None or step.get("gamma") is None
            missing_batch = control is None or decision.get("batch") is None
            if missing_action:
                reasons.append("missing_selected_action"); pending.add("missing_selected_action")
            elif not isinstance(selected, int) or selected != step.get("gamma") or not 0 <= selected <= identity["proposal_budget"]:
                raise ValueError("logged selected action differs from the executed controller cap")
            if missing_batch:
                reasons.append("missing_decision_batch"); pending.add("missing_decision_batch")
            elif decision["batch"] != control:
                raise ValueError("decision batch differs from its precontrol batch")
            if table is None:
                reasons.append("calibration_unavailable"); pending.add("calibration_unavailable")
            elif not missing_action and not missing_batch:
                expected = table.estimate(control, selected, identity["proposal_budget"])
                if not math.isclose(prediction, expected.total_step_ms, rel_tol=1e-9, abs_tol=1e-10):
                    raise ValueError("recorded selected cost differs from the executed calibration lookup")
        row = {"source": str(source), "wave": wave, "wave_index": wave_index, "step": step["step"],
            "phase": step.get("phase"), "nominal_batch": identity["nominal_batch"],
            "actual_batch": actual, "precontrol_batch": control, "selected_gamma": selected,
            "proposal_budget": identity["proposal_budget"], "proposal_budget_next": next_budget,
            "scheduled_tokens": step.get("scheduled_tokens"), "scheduled_drafts": step.get("scheduled_drafts"),
            "available_drafts": step.get("drafts_available_before"), "truncated_drafts": step.get("truncated_by_controller"),
            "predicted_total_step_ms": prediction, "observed_host_step_ms": measured,
            "prediction_source": decision.get("source"), "decision_cpu_us": decision.get("decision_cpu_us"),
            "signed_error_ms": prediction - measured if prediction is not None and measured is not None else None,
            "eligible": not reasons, "exclusion_reasons": reasons,
            "terminal_request_count": sum(row["terminal"] for row in observations),
            "terminal_recovered_count": sum(row["terminal_recovered"] for row in observations)}
        rows.append(row)
    result = {**identity, **metrics(rows), "total_steps": len(rows),
        "excluded_steps": sum(not row["eligible"] for row in rows),
        "exclusion_reason_counts": dict(Counter(reason for row in rows for reason in row["exclusion_reasons"])),
        "pending": sorted(pending)}
    result["status"] = "pending" if pending or not result["n"] else "complete"
    groups = defaultdict(list)
    for row in rows:
        groups[row["actual_batch"], row["selected_gamma"]].append(row)
    diagnostics = [{**identity, "actual_batch": batch, "selected_gamma": gamma, **metrics(group),
        "total_steps": len(group), "excluded_steps": sum(not row["eligible"] for row in group)}
        for (batch, gamma), group in groups.items()]
    return result, diagnostics, rows


def analyze(inputs, output, calibration=None):
    output = Path(output).resolve()
    if output.exists(): raise FileExistsError("choose a fresh cost-transfer output directory")
    paths, missing, invalid = set(), [], []
    for value in inputs:
        path = relocate(value)
        if not path.exists():
            missing.append({"source": str(path), "reason": "input not present"}); continue
        if path.is_file(): paths.add(path.resolve()); continue
        folders = [path] if (path / "configuration.json").is_file() else sorted(
            folder for folder in path.glob("gamma*") if folder.is_dir())
        if not folders: missing.append({"source": str(path), "reason": "no recorded engines"})
        for folder in folders:
            try:
                config = read(folder / "configuration.json")
                for repeat, batch in execution_order(config):
                    source = folder / f"perf-r{repeat}-b{batch}.json"
                    if source.is_file(): paths.add(source.resolve())
                    else: missing.append({"source": str(source), "reason": "expected run not present"})
            except FileNotFoundError as exc:
                missing.append({"source": str(folder), "reason": str(exc)})
    executions, calibrations, runs, diagnostics, steps, provenance = {}, {}, [], [], [], []
    for path in sorted(paths):
        try:
            record, context = read(path), _context(path)
            if path.parent not in executions: executions[path.parent] = engine_evidence(path.parent)
            execution = executions[path.parent]
            if record.get("proposal_budget") != execution["budget"] or record.get("mode") != execution["config"]["args"]["mode"]:
                raise ValueError("record/engine mode or proposal budget mismatch")
            costs = None
            if record.get("mode") == "adaptive":
                actual = path.parent.parent / "calibration.json"
                if not actual.is_file():
                    configured = execution["config"]["args"].get("calibration")
                    if not configured: raise Pending("executed calibration artifact is unavailable")
                    actual = relocate(configured)
                table_path = relocate(calibration) if calibration else actual
                if sha(actual) != sha(table_path): raise ValueError("supplied table differs from the executed calibration artifact")
                key = str(table_path.resolve()), str(path.parent)
                if key not in calibrations: calibrations[key] = CalibrationEvidence(table_path, execution)
                costs = calibrations[key]
                costs.check_run(record, path, resolve_regime(record, context), execution)
            run, cells, events = analyze_record(record, path, context, costs.table if costs else None)
            run["source_sha256"] = sha(path)
            run["calibration_sha256"] = costs.hash if costs else None
            runs.append(run); diagnostics.extend(cells); steps.extend(events)
            provenance.append({"source": str(path), "sha256": run["source_sha256"],
                "configuration_sha256": execution["configuration_sha256"], "engine_args_sha256": execution["engine_args_sha256"],
                "runtime_source_sha256": execution["runtime_source_sha256"],
                "calibration": str(costs.path) if costs else None, "calibration_sha256": costs.hash if costs else None})
        except (FileNotFoundError, Pending) as exc:
            missing.append({"source": str(path), "reason": str(exc)})
        except (KeyError, ValueError, TypeError) as exc:
            invalid.append({"source": str(path), "reason": str(exc)})
    output.mkdir(parents=True)
    write_csv(output / "run_cost_prediction.csv", runs)
    write_csv(output / "selected_action_diagnostics.csv", diagnostics)
    with gzip.open(output / "step_cost_predictions.csv.gz", "wt", newline="") as stream:
        if steps:
            writer = csv.DictWriter(stream, fieldnames=list(steps[0])); writer.writeheader()
            writer.writerows({**row, "exclusion_reasons": json.dumps(row["exclusion_reasons"])} for row in steps)
    report = {"status": "invalid" if invalid else "pending" if missing or not runs or any(run["status"] != "complete" for run in runs) else "complete",
        "status_scope": "selected original runs; not campaign completion",
        "runs": runs, "selected_action_diagnostics": diagnostics, "missing_inputs": missing, "invalid_inputs": invalid,
        "source_files": provenance, "eligible_steps": sum(run["n"] for run in runs), "total_steps": len(steps),
        "calibrations": [{"path": str(item.path), "sha256": item.hash, "metadata": item.table.metadata,
            "records": item.records, "verified_sources": item.sources} for item in calibrations.values()],
        "analysis_source_sha256": sha(__file__), "boundaries": [
            "Only the logged selected action is evaluated. No unchosen-action, action-ranking, counterfactual speedup or service-benefit claim.",
            "Primary rows require pure decode, positive complete synchronous host duration, equal precontrol/actual batch, known request identities/termination and no terminal request, with unchanged full production budget. Every aligned excluded step and its reasons remain in the CSV.",
            "Prediction minus observation defines signed bias. N counts within-run eligible steps, not independent system repetitions; each original run is preserved separately, including run-local selected-gamma/actual-B diagnostics.",
            "Recorded predicted cost is checked against the exact executed table for its chosen action, including native B interpolation. No policy decision is recomputed and no value is substituted for a missing prediction.",
            "Measured calibration sources and evaluation request IDs are disjoint; historical warmup exposure is not asserted disjoint. Eager/compiled, model/draft, resource, environment, runtime-source behavior and production budgets must match.",
            "Calibration rows average eligible wave means from fixed-action runs; this diagnostic averages selected dynamic steps. History, context length, availability, graph shapes and controller work may change their distributions. Transfer error does not identify a causal component.",
            "Host step includes verification, next drafting, control and synchronous host work. Components are not subtracted or renamed GPU kernel time. This diagnostic complements rather than replaces full-service evaluation."]}
    (output / "policy_cost_prediction.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--calibration", help="optional table; must equal the campaign's actually loaded copy")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = analyze(args.inputs, args.output, args.calibration)
    print(json.dumps({"status": report["status"], "runs": len(report["runs"]),
                      "eligible_steps": report["eligible_steps"], "output": args.output}))
    return {"complete": 0, "pending": 2, "invalid": 1}[report["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
