#!/usr/bin/env python3
"""F5 actual online pre-step acceptance/availability tracking (CPU only).

  python code/analyze_online_acceptance.py runs/POLICY_CAMPAIGN --output runs/F5_online_POLICY
  python code/analyze_online_acceptance.py runs/MATRIX/gamma1 --output runs/F5_online_gamma1
  python code/analyze_online_acceptance.py --preparation runs/DOMAIN_controller_preparation_r01 \
      --output runs/F5_online_DOMAIN

Unlike analyze_acceptance_tracking.py, this starts from the recorded post-warmup
evaluation_initial_state.json and follows the runner's exact batch/repeat order,
without resetting at perf files or waves. It reconstructs the executed estimator,
checks step alpha/decision predictions and wave-end snapshots, and never fits a
prior. Missing initial state/predecessors/updates remain pending. A past wave-end
snapshot can re-establish state only for later steps, never retroactively.

Metrics are weighted within each original performance run by tested positions,
rank exposures, or decode request-steps. Raw run values are retained; steps and
requests are not independent repetitions. Observation rates are noisy outcomes,
not latent ground-truth probabilities. Logged decision progress is distinct from
an estimator-derived diagnostic and from a counterfactual policy result.
Completion refers to analysis of the selected engines. Campaign execution and
gamma coverage are reported separately; one complete engine does not complete
the surrounding matrix.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import tempfile
import unittest

from analyze_acceptance_tracking import iter_steps, write_csv
from cost_model import AcceptanceEWMA
from plot_results import REGIME_FIELDS, _context, _finite, resolve_regime, route_instrumented


PROJECT = Path(__file__).resolve().parents[1]
STATE_KEYS = ("alpha", "decay", "alpha_estimator", "alpha_prior", "alpha_counts",
    "prefix_survival", "exposures", "updates", "observations", "availability",
    "availability_exposures", "availability_updates")
ERROR_NAMES = ("conditional_alpha", "logged_decision_progress")
STEP_CSV_FIELDS = ("source", "policy_role", "run_id", "repeat", "nominal_batch", "mode", "wave", "step",
    "host_elapsed_s", "phase", "actual_decode_batch", "selected_gamma", "proposal_budget",
    "online_state_recovered", "state_anchor", "conditional_alpha_before", "conditional_alpha_observed",
    "conditional_alpha_error", "conditional_accepted", "conditional_tested",
    "logged_decision_progress_before", "progress_observed", "logged_decision_progress_error", "progress_requests",
    "state_derived_progress_before", "one_plus_accepted_proxy", "proxy_emission_mismatches",
    "availability_rank1_before", "availability_rank1_observed", "availability_request_steps",
    "eligible_request_steps", "terminal_request_steps", "terminal_recovered_from_runner",
    "terminal_status_unknown_request_steps", "alpha_estimator", "alpha_count_accepted_before", "alpha_count_tested_before")


class Pending(ValueError):
    """Evidence does not identify a complete actual pre-step estimator state."""


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def same(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        a, b = {str(k): v for k, v in a.items()}, {str(k): v for k, v in b.items()}
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-10)
    return a == b


def state_view(snapshot):
    if not isinstance(snapshot, dict) or any(k not in snapshot for k in STATE_KEYS):
        raise Pending("missing or incomplete AcceptanceEWMA snapshot")
    return {k: snapshot[k] for k in STATE_KEYS}


def restore(snapshot):
    """Restore recorded count/prior state, not a newly chosen cold prior."""
    s = state_view(snapshot)
    prior, counts = s["alpha_prior"], s["alpha_counts"]
    try:
        state = AcceptanceEWMA(alpha=prior["alpha"], decay=s["decay"],
            alpha_estimator=s["alpha_estimator"], prior_tested_count=prior["tested_count"])
        state.alpha = s["alpha"]
        state.ewma_accepted, state.ewma_tested = counts["ewma_accepted"], counts["ewma_tested"]
        state.observed_accepted, state.observed_tested = counts["observed_accepted"], counts["observed_tested"]
        state.survival = {int(k): v for k, v in s["prefix_survival"].items()}
        state.exposures.update({int(k): v for k, v in s["exposures"].items()})
        state.availability = {int(k): v for k, v in s["availability"].items()}
        state.availability_exposures.update({int(k): v for k, v in s["availability_exposures"].items()})
        state.update_count, state.observation_count = s["updates"], s["observations"]
        state.availability_update_count = s["availability_updates"]
        if not 0 <= state.alpha <= 1 or not all(math.isfinite(x) and x >= 0 for x in
                (state.ewma_accepted, state.ewma_tested, state.observed_accepted, state.observed_tested)):
            raise ValueError("invalid scalar/count state")
        for values in (state.survival, state.availability):
            if any(k < 1 or not 0 <= v <= 1 for k, v in values.items()):
                raise ValueError("invalid probability state")
        if s["alpha_estimator"] == "exposure_ratio" and state.ewma_tested and not same(state.alpha, state.ewma_accepted / state.ewma_tested):
            raise ValueError("alpha differs from recorded count ratio")
        if not same(state_view(state.snapshot()), s):
            raise ValueError("snapshot does not round-trip")
    except (KeyError, TypeError, ValueError) as exc:
        raise Pending(f"unsupported/inconsistent snapshot: {exc}") from exc
    return state


def ast_signature(path, names):
    tree = ast.parse(Path(path).read_text())
    nodes = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names:
            for child in ast.walk(node):
                if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    if child.body and isinstance(child.body[0], ast.Expr) and isinstance(child.body[0].value, ast.Constant) and isinstance(child.body[0].value.value, str):
                        child.body = child.body[1:]
            nodes.append(ast.dump(node, include_attributes=False))
    if len(nodes) != len(names):
        raise Pending("recorded estimator/update implementation is unidentified")
    return hashlib.sha256(json.dumps(nodes).encode()).hexdigest()


def check_algorithm(folder):
    """Only inspect captured code; never execute a captured implementation."""
    environment = read(folder / "environment.json")
    result = {}
    for filename, names in (("cost_model.py", {"AcceptanceEWMA"}),
                            ("vllm_s_plugin.py", {"finish_update", "observe_acceptance"})):
        recorded = folder.parent / "source" / filename
        if not recorded.is_file():
            raise Pending(f"missing captured {filename}")
        if environment.get("source_sha256", {}).get(filename) != sha(recorded):
            raise Pending(f"captured {filename} differs from the engine's recorded source SHA")
        if ast_signature(recorded, names) != ast_signature(Path(__file__).with_name(filename), names):
            raise Pending(f"unsupported recorded {filename} estimator/update behavior")
        result[filename] = sha(recorded)
    return result


def apply_recorded_update(state, step, config):
    """Exactly the executed plugin update population, including its censor mask."""
    required = ("accepted_by_request", "observed_drafts_by_request", "ewma_censored_requests",
                "availability_updated", "scheduled_decode_request_ids", "proposed_before_by_request")
    if any(key not in step for key in required):
        raise Pending("missing per-request update/censor/availability log fields")
    accepted, verified = step["accepted_by_request"], step["observed_drafts_by_request"]
    if accepted.keys() != verified.keys():
        raise Pending("accepted/verified event identities differ")
    if any(type(verified[r]) is not int or verified[r] <= 0 or type(accepted[r]) is not int
           or not 0 <= accepted[r] <= verified[r] for r in verified):
        raise ValueError("invalid recorded acceptance event")
    for name, values in (("accepted_drafts", accepted), ("observed_drafts", verified)):
        if name in step and step[name] != sum(values.values()):
            raise Pending(f"{name} aggregate does not match recorded events")
    censored = set(step["ewma_censored_requests"])
    eligible = [rid for rid in verified if rid not in censored]
    if "ewma_eligible_requests" in step and step["ewma_eligible_requests"] != len(eligible):
        raise Pending("eligible acceptance count differs from recorded censor mask")
    if eligible:
        state.update([accepted[r] for r in eligible], [verified[r] for r in eligible])
    if step["availability_updated"]:
        if config.get("proposal_budget") != "max" or config.get("mode") == "native":
            raise Pending("availability update is inconsistent with recorded plugin mode")
        lengths = [step["proposed_before_by_request"].get(rid, 0)
                   for rid in step["scheduled_decode_request_ids"] if rid not in censored]
        if not lengths or "max_gamma" not in config:
            raise Pending("availability update lacks its request population or full budget")
        state.update_availability(lengths, int(config["max_gamma"]))


def prediction_row(state, host, step, config, observations, identity, wave, anchor):
    budget = int(config.get("max_gamma", identity["proposal_budget"]))
    decision = step.get("decision") or {}
    selected = int(decision.get("gamma", step.get("gamma", identity["gamma"])))
    if not 0 <= selected <= budget:
        raise ValueError("selected cap outside recorded maximum")
    logged = _finite(decision.get("expected_tokens_per_request"))
    if logged is not None and not 1 <= logged <= selected + 1 + 1e-9:
        raise ValueError("invalid recorded decision progress prediction")
    eligible = [o for o in observations if not o["terminal"] and o["terminal_known"]]
    valid = [o for o in eligible if o["accepted"] is not None and o["verified"] is not None]
    # Full budget exposure is observable for max-mode post-truncation, or for
    # fixed native budgets. Native per-batch production can vary across steps;
    # absent per-request generating-budget logs do not identify its higher ranks.
    full_known = ((config.get("proposal_budget") == "max" and config.get("mode") != "native")
        or (identity["mode"] == "native" and not identity["native_batch_table"]
            and int(step.get("proposal_budget_next", -1)) == budget == identity["proposal_budget"]))
    available = [o for o in eligible if o["available"] is not None] if full_known else []
    a = sum(o["accepted"] for o in valid)
    tested = sum(min(o["accepted"] + 1, o["verified"]) for o in valid)
    emitted = [o["emitted"] for o in eligible]
    progress = statistics.mean(emitted) if emitted and all(v is not None for v in emitted) else None
    prefix = state.prefix_survival(budget) if state else None
    availability = state.availability_survival(budget) if state else None
    reconstructed = state.expected_tokens(selected) if state else None
    row = {"source": identity["source"], "policy_role": identity.get("policy_role", "unspecified"),
        "run_id": identity.get("run_id"), "repeat": identity.get("repeat"), "nominal_batch": identity["batch"],
        "mode": identity["mode"], "wave": wave, "step": int(step["step"]),
        "host_elapsed_s": host.get("elapsed_s"), "phase": step.get("phase"),
        "actual_decode_batch": len(observations), "selected_gamma": selected, "proposal_budget": budget,
        "state_anchor": anchor if state else None, "online_state_recovered": state is not None,
        "conditional_alpha_before": state.alpha if state else None,
        "prefix_survival_before": prefix, "availability_survival_before": availability,
        "alpha_estimator": state.alpha_estimator if state else None,
        "alpha_count_accepted_before": state.ewma_accepted if state else None,
        "alpha_count_tested_before": state.ewma_tested if state else None,
        "prefix_exposures_before": [state.exposures.get(j, 0) for j in range(1, budget + 1)] if state else None,
        "availability_exposures_before": [state.availability_exposures.get(j, 0) for j in range(1, budget + 1)] if state else None,
        "logged_decision_progress_before": logged, "state_derived_progress_before": reconstructed,
        "progress_observed": progress, "progress_requests": len(eligible) if progress is not None else 0,
        "one_plus_accepted_proxy": 1 + a / len(valid) if valid else None,
        "proxy_emission_mismatches": sum(o["emitted"] != 1 + o["accepted"] for o in valid if o["emitted"] is not None),
        "conditional_accepted": a, "conditional_tested": tested,
        "conditional_alpha_observed": a / tested if tested else None,
        "conditional_alpha_error": state.alpha - a / tested if state and tested else None,
        "logged_decision_progress_error": logged - progress if logged is not None and progress is not None else None,
        "eligible_request_steps": len(eligible), "acceptance_request_steps": len(valid),
        "terminal_request_steps": sum(o["terminal"] for o in observations),
        "terminal_recovered_from_runner": sum(o["terminal_recovered"] for o in observations),
        "terminal_status_unknown_request_steps": sum(not o["terminal_known"] for o in observations),
        "availability_request_steps": len(available), "full_proposal_budget_identified": full_known,
        "availability_rank1_before": availability[0] if availability else None,
        "availability_rank1_observed": sum(o["available"] >= 1 for o in available) / len(available) if available and budget else None,
        "availability_updated_by_runtime": step.get("availability_updated"),
        "ranks": [{"rank": j, "available": sum(o["available"] >= j for o in available),
            "availability_exposure": len(available), "verified_exposure": sum(o["verified"] >= j for o in valid),
            "accepted": sum(o["accepted"] >= j for o in valid), "accepted_population": len(valid)}
            for j in range(1, budget + 1)],
        "evidence_kind": "actual_online_recorded_decision_and_reconstructed_pre_step_state"}
    if state and logged is not None and not same(logged, reconstructed):
        raise Pending("recorded decision progress differs from reconstructed pre-step state")
    if state:
        for candidate in decision.get("candidates", []):
            if not same(candidate["expected_tokens_per_request"], state.expected_tokens(int(candidate["gamma"]))):
                raise Pending("recorded candidate progress differs from reconstructed pre-step state")
    return row


def remove_reconstructed(row):
    row["online_state_recovered"] = False
    for key in ("conditional_alpha_before", "prefix_survival_before", "availability_survival_before",
                "alpha_estimator", "alpha_count_accepted_before", "alpha_count_tested_before",
                "prefix_exposures_before", "availability_exposures_before", "state_derived_progress_before", "state_anchor",
                "conditional_alpha_error", "availability_rank1_before"):
        row[key] = None


def error_add(counter, prediction, observation, weight):
    if prediction is not None and observation is not None and weight:
        delta = prediction - observation
        counter["weight"] += weight
        counter["absolute"] += weight * abs(delta)
        counter["squared"] += weight * delta * delta
        counter["signed"] += weight * delta


def error_metrics(counter, prefix):
    weight = counter["weight"]
    return {prefix + "_weight": weight,
        prefix + "_mae": counter["absolute"] / weight if weight else None,
        prefix + "_rmse": math.sqrt(counter["squared"] / weight) if weight else None,
        prefix + "_bias": counter["signed"] / weight if weight else None}


def aggregate_run(identity, rows, pending):
    totals, errors, ranks = Counter(), defaultdict(Counter), defaultdict(Counter)
    rank_errors = defaultdict(Counter)
    for row in rows:
        totals["host_steps"] += 1
        totals["online_state_recovered_steps"] += row["online_state_recovered"]
        totals["recorded_decision_prediction_steps"] += row["logged_decision_progress_before"] is not None
        for name in ("actual_decode_batch", "eligible_request_steps", "acceptance_request_steps", "terminal_request_steps",
                     "terminal_recovered_from_runner", "terminal_status_unknown_request_steps", "conditional_accepted",
                     "conditional_tested", "availability_request_steps", "proxy_emission_mismatches"):
            totals[name] += row[name]
        error_add(errors["conditional_alpha"], row["conditional_alpha_before"], row["conditional_alpha_observed"], row["conditional_tested"])
        error_add(errors["logged_decision_progress"], row["logged_decision_progress_before"], row["progress_observed"], row["progress_requests"])
        for observed in row["ranks"]:
            j = observed["rank"]
            ranks[j].update({k: v for k, v in observed.items() if k != "rank"})
            if row["online_state_recovered"]:
                for name, count, exposure, predictions in (
                    ("availability", observed["available"], observed["availability_exposure"], row["availability_survival_before"]),
                    ("conditional_prefix", observed["accepted"], observed["verified_exposure"], row["prefix_survival_before"])):
                    error_add(rank_errors[j, name], predictions[j - 1], count / exposure if exposure else None, exposure)
    run = {**identity, **dict(totals), "pending": pending,
        "status": "pending" if pending or totals["online_state_recovered_steps"] < len(rows) else "complete",
        "conditional_alpha_pooled": totals["conditional_accepted"] / totals["conditional_tested"] if totals["conditional_tested"] else None,
        "weight_boundary": "within-run tested positions, verified-rank exposures or eligible decode request-steps; not independent repeats"}
    for name in ERROR_NAMES:
        run.update(error_metrics(errors[name], name + "_prestep"))
    rank_rows = []
    for rank, c in sorted(ranks.items()):
        row = {**identity, "rank": rank, **dict(c), "status": run["status"],
            "availability_survival_observed": c["available"] / c["availability_exposure"] if c["availability_exposure"] else None,
            "conditional_prefix_survival_observed": c["accepted"] / c["verified_exposure"] if c["verified_exposure"] else None,
            "accepted_survival_unconditional_observed": c["accepted"] / c["accepted_population"] if c["accepted_population"] else None}
        for name in ("availability", "conditional_prefix"):
            row.update(error_metrics(rank_errors[rank, name], name + "_prestep"))
        rank_rows.append(row)
    return run, rank_rows


def execution_order(config):
    plan, args = config["plan"], config.get("args", {})
    batches, repeats = args.get("batches") or plan["batches"], args.get("repeats") or plan["repeats"]
    return [(repeat, batch) for repeat in range(repeats)
            for batch in (batches if repeat % 2 == 0 else list(reversed(batches)))]


def analyze_engine(folder, emit=lambda row: None, policy_role="unspecified"):
    folder = Path(folder).resolve()
    config = read(folder / "configuration.json")
    engine_info = {"engine": str(folder), "policy_role": policy_role,
        "configuration_sha256": sha(folder / "configuration.json"), "pending": []}
    status_path = folder / "status.json"
    engine_info["engine_recorded_status"] = read(status_path).get("status") if status_path.is_file() else "unavailable"
    algorithm_ok, state, anchor = True, None, None
    try:
        engine_info["recorded_algorithm_sha256"] = check_algorithm(folder)
    except (OSError, KeyError, ValueError) as exc:
        algorithm_ok = False
        engine_info["pending"].append(str(exc))
    initial_path = folder / "evaluation_initial_state.json"
    try:
        initial = read(initial_path)
        engine_info.update(initial_state_file=str(initial_path), initial_state_sha256=sha(initial_path),
            initial_state_metadata={k: v for k, v in initial.items() if k != "acceptance_snapshot_before"},
            initial_acceptance_snapshot=initial.get("acceptance_snapshot_before"))
        if initial.get("acceptance_state_policy") != "continuous across warmup, batches, repeats and waves; no reset":
            raise Pending("initial-state continuity policy is missing or unsupported")
        if algorithm_ok:
            state, anchor = restore(initial["acceptance_snapshot_before"]), str(initial_path)
    except (OSError, KeyError, ValueError) as exc:
        engine_info["pending"].append("initial state unavailable: " + str(exc))
    runs, ranks, order = [], [], execution_order(config)
    engine_info["execution_order"] = [f"perf-r{r}-b{b}.json" for r, b in order]
    for order_index, (repeat, batch) in enumerate(order):
        path = folder / f"perf-r{repeat}-b{batch}.json"
        if not path.is_file():
            engine_info["pending"].append(f"missing predecessor/run: {path.name}")
            state, anchor = None, None
            continue
        record, context = read(path), _context(path)
        if route_instrumented(record, context) or not record.get("all_finished"):
            engine_info["pending"].append(f"incomplete or instrumented performance record: {path.name}")
            state, anchor = None, None
            continue
        if record.get("batch") != batch or record.get("label") != path.stem:
            raise ValueError("performance file identity differs from runner execution order")
        identity = {**resolve_regime(record, context), "source": str(path), "source_sha256": sha(path),
            "engine": str(folder), "run_id": path.stem, "repeat": repeat, "execution_order_index": order_index,
            "policy_role": policy_role, "policy_campaign": str(folder.parent),
            "batch": batch, "gamma": int(record["gamma"]), "proposal_budget": int(record.get("proposal_budget", record["gamma"])),
            "mode": record.get("mode", "native"), "native_batch_table": bool(context.get("native_batch_table")),
            "prediction_role": "adaptive_decision" if record.get("mode") == "adaptive" else "observed_estimator_state_diagnostic"}
        rows, pending = [], []
        for episode in record.get("episodes", [record]):
            wave_rows, wave_mismatch = [], False
            for _, wave, host, step, plugin_config, observations, _ in iter_steps({"episodes": [episode]}):
                if state is not None and plugin_config.get("acceptance_class") != "AcceptanceEWMA":
                    pending.append(f"{wave}: recorded estimator class is missing or unsupported")
                    state, anchor = None, None
                if state is None and not wave_rows:
                    pending.append(f"{wave}: missing actual pre-step state/predecessor")
                try:
                    row = prediction_row(state, host, step, plugin_config, observations, identity, wave, anchor)
                except Pending as exc:
                    pending.append(f"{wave}/step{step['step']}: {exc}")
                    wave_mismatch = True
                    state, anchor = None, None
                    row = prediction_row(None, host, step, plugin_config, observations, identity, wave, None)
                if state is not None:
                    try:
                        apply_recorded_update(state, step, plugin_config)
                        if _finite(step.get("alpha_after")) is not None and not same(state.alpha, step["alpha_after"]):
                            raise Pending("reconstructed alpha_after differs from the actual step log")
                        row["alpha_after_checkpoint"] = step.get("alpha_after")
                    except Pending as exc:
                        pending.append(f"{wave}/step{step['step']}: {exc}")
                        state, anchor = None, None
                wave_rows.append(row)
            snapshot = episode.get("plugin", {}).get("acceptance_snapshot")
            if snapshot is not None and algorithm_ok:
                try:
                    endpoint = restore(snapshot)
                    if state is not None and not same(state_view(state.snapshot()), state_view(snapshot)):
                        pending.append(f"{episode.get('label')}: reconstructed wave-end state differs from recorded snapshot")
                        wave_mismatch = True
                    # This actual past checkpoint can seed ONLY later steps.
                    state, anchor = endpoint, f"{path}#{episode.get('label')}:end_snapshot"
                except Pending as exc:
                    pending.append(f"{episode.get('label')}: {exc}")
                    state, anchor = None, None
            if wave_mismatch:
                for row in wave_rows:
                    remove_reconstructed(row)
            rows.extend(wave_rows)
        run, rank_rows = aggregate_run(identity, rows, pending)
        runs.append(run)
        ranks.extend(rank_rows)
        for row in rows:
            emit(row)
    engine_info["status"] = "pending" if engine_info["pending"] or any(r["status"] != "complete" for r in runs) else "complete"
    engine_info["analysis_status"] = engine_info["status"]
    engine_info["engine_complete"] = engine_info["engine_recorded_status"] == "complete" and engine_info["status"] == "complete"
    engine_info["analyzed_run_records"] = len(runs)
    return engine_info, runs, ranks


def repeat_summary(runs):
    groups = defaultdict(list)
    fields = (*REGIME_FIELDS, "policy_role", "policy_campaign", "batch", "gamma", "proposal_budget", "mode", "native_batch_table")
    for run in runs:
        groups[tuple(run[k] for k in fields)].append(run)
    result = []
    for key, rows in sorted(groups.items()):
        row = {**dict(zip(fields, key)), "n_run_records": len(rows),
            "sources": [r["source"] for r in rows], "raw_status": [r["status"] for r in rows]}
        for name in ERROR_NAMES:
            for metric in ("mae", "rmse", "bias"):
                field = name + "_prestep_" + metric
                values = [r.get(field) for r in rows]
                row[field + "_raw"] = values
                row[field + "_median"] = statistics.median(values) if all(v is not None for v in values) and all(r["status"] == "complete" for r in rows) else None
        result.append(row)
    return result


def relocate(value):
    path = Path(value)
    if path.exists():
        return path
    marker = "/runs/"
    return PROJECT / "runs" / str(path).split(marker, 1)[1] if marker in str(path) else path


def discover(inputs, preparation=None):
    paths, pending, roles = list(map(Path, inputs)), [], {}
    if preparation:
        manifest_path = Path(preparation)
        manifest_path = manifest_path / "evaluation_commands.json" if manifest_path.is_dir() else manifest_path
        manifest = read(manifest_path)
        for command in manifest["commands"]:
            campaign = relocate(command["status_file"]).parent
            paths.append(campaign)
            roles[str(campaign.resolve())] = command["role"]
    engines = set()
    for path in paths:
        if not path.exists():
            pending.append({"source": str(path), "reason": "campaign/input not yet present"})
        elif path.is_file():
            engines.add(path.parent.resolve())
        elif (path / "configuration.json").is_file():
            engines.add(path.resolve())
        else:
            found = list(path.glob("gamma*/configuration.json"))
            if not found:
                pending.append({"source": str(path), "reason": "no recorded engine configuration yet"})
            engines.update(p.parent.resolve() for p in found)
    return sorted(engines), pending, roles


def campaign_scope(engines, metadata):
    """Keep selected-engine reconstruction separate from campaign completion."""
    def optional_json(path):
        try:
            return read(path)
        except (OSError, ValueError):
            return {}
    analyzed = {row["engine"]: row for row in metadata}
    campaigns = []
    for campaign in sorted({folder.parent for folder in engines}):
        selected = [folder for folder in engines if folder.parent == campaign]
        status_path, plan_path = campaign / "status.json", campaign / "plan.json"
        status, plan = optional_json(status_path), optional_json(plan_path)
        expected = status.get("args", {}).get("gammas") or plan.get("gammas")
        gamma_by_engine = {str(folder): optional_json(folder / "configuration.json").get("args", {}).get("gamma")
                           for folder in selected}
        selected_gammas = sorted({int(g) for g in gamma_by_engine.values() if g is not None})
        unknown_gamma = [folder for folder, gamma in gamma_by_engine.items() if gamma is None]
        recorded_complete = sorted({int(cell["gamma"]) for cell in status.get("cells", [])
                                    if cell.get("returncode") == 0})
        covers = expected is not None and not unknown_gamma and set(expected) == set(selected_gammas)
        execution_complete = (status.get("status") == "complete" and expected is not None
                              and set(expected).issubset(recorded_complete))
        campaigns.append({"campaign": str(campaign), "recorded_status": status.get("status", "unavailable"),
            "status_file": str(status_path), "status_sha256": sha(status_path) if status_path.is_file() else None,
            "expected_gammas": expected, "selected_gammas": selected_gammas,
            "engines_with_unknown_gamma": unknown_gamma,
            "recorded_completed_gammas": recorded_complete,
            "selected_analysis_complete": all(analyzed.get(str(folder), {}).get("analysis_status") == "complete" for folder in selected),
            "analysis_covers_campaign": covers, "campaign_execution_complete": execution_complete,
            "campaign_partial": not (execution_complete and covers)})
    return campaigns


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="*")
    parser.add_argument("--preparation", help="Main or reduced preparation directory/evaluation_commands.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if not args.output or not (args.inputs or args.preparation):
        parser.error("inputs or --preparation, and --output, are required")
    engines, missing, roles = discover(args.inputs, args.preparation)
    args.output.mkdir(parents=True, exist_ok=False)
    runs, ranks, metadata, invalid = [], [], [], []
    with gzip.open(args.output / "tracking_steps.jsonl.gz", "wt", encoding="utf-8") as stream, \
            gzip.open(args.output / "tracking_steps.csv.gz", "wt", encoding="utf-8", newline="") as csv_stream:
        csv_writer = csv.DictWriter(csv_stream, fieldnames=STEP_CSV_FIELDS, extrasaction="ignore")
        csv_writer.writeheader()
        def emit(row):
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
            csv_writer.writerow(row)
        for folder in engines:
            try:
                info, found_runs, found_ranks = analyze_engine(folder, emit,
                    policy_role=roles.get(str(folder.parent), "unspecified"))
                metadata.append(info)
                runs.extend(found_runs)
                ranks.extend(found_ranks)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                invalid.append({"engine": str(folder), "reason": str(exc)})
    status = "invalid" if invalid else "pending" if missing or not metadata or any(m["status"] != "complete" for m in metadata) else "complete"
    campaigns = campaign_scope(engines, metadata)
    summary = repeat_summary(runs)
    write_csv(args.output / "online_tracking_runs.csv", runs)
    write_csv(args.output / "online_tracking_summary.csv", summary)
    write_csv(args.output / "rank_coverage_runs.csv", ranks)
    manifest = {"status": status, "analysis_status": status, "analysis_scope": "selected_engines",
        "campaign_partial": bool(missing) or not campaigns or any(c["campaign_partial"] for c in campaigns),
        "campaigns": campaigns, "engines": metadata, "missing_inputs": missing, "invalid_inputs": invalid,
        "runs": runs, "rank_coverage": ranks, "summary": summary,
        "analysis_source_sha256": {p.name: sha(p) for p in (Path(__file__), Path(__file__).with_name("cost_model.py"), Path(__file__).with_name("analyze_acceptance_tracking.py"))},
        "boundaries": [
            "Analysis status applies only to selected engines. Campaign execution status and gamma coverage are separate; an engine-complete result may be campaign-partial.",
            "Actual online state is reconstructed from the recorded post-warmup snapshot and executed updates in original runner order. No cold prior reset at waves, batches or repeats.",
            "Missing predecessors never receive invented updates. A past wave-end snapshot can restore later states but cannot repair earlier predictions using future information.",
            "State reconstruction uses exactly the plugin's recorded censor mask; evaluation additionally censors terminal requests recovered from client output timestamps and reports their count.",
            "Conditional alpha is accepted/tested positions; prefix survival conditions on verified-rank exposure. Full-proposal availability includes zero proposals and remains distinct from acceptance.",
            "Availability for native varying-budget tables is unidentified without generating-budget provenance; higher ranks are not filled with zeros.",
            "Logged decision progress was used online. State-derived progress is a diagnostic and is not scored as a policy decision where no prediction was logged.",
            "Error weights apply within each original run. Two raw repeated-run values are retained; shared requests/steps are not independent repeats, and EWMA state persists across these runs.",
            "CSV step errors are prediction minus observation; positive bias means overprediction. Run MAE/RMSE/bias use the stated exposure weights.",
            "Observed rates are noisy realized observations, not latent probabilities. No counterfactual speedup, theoretical bound or GPU result is inferred from these errors."]}
    (args.output / "online_acceptance.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"status": status, "analysis_scope": "selected_engines", "campaign_partial": manifest["campaign_partial"],
                      "engines": len(metadata), "run_records": len(runs), "output": str(args.output)}))
    return 1 if invalid else 2 if status == "pending" else 0


def self_test():
    import shutil
    def fixture(directory):
        """Synthetic logs with a non-cold initial state; discarded after tests."""
        directory = Path(directory)
        folder = directory / "gamma2"
        folder.mkdir(parents=True)
        (directory / "source").mkdir()
        hashes = {}
        for name in ("cost_model.py", "vllm_s_plugin.py"):
            shutil.copyfile(Path(__file__).with_name(name), directory / "source" / name)
            hashes[name] = sha(directory / "source" / name)
        def dump(path, value):
            path.write_text(json.dumps(value))
        config = {"plan": {"batches": [1, 2], "repeats": 2}, "args": {"mode": "adaptive"},
                  "workload_sha256": "synthetic_cpu_fixture"}
        dump(folder / "configuration.json", config)
        dump(folder / "environment.json", {"source_sha256": hashes, "gpu": "synthetic_cpu_fixture"})
        state = AcceptanceEWMA()
        state.update([0, 1], [2, 2])
        state.update_availability([2, 0], 2)
        initial = state.snapshot()
        dump(folder / "evaluation_initial_state.json", {"acceptance_snapshot_before": initial,
            "warmup_offset": 0, "shape_warmup_offset": 0, "input_offset": 64,
            "acceptance_state_policy": "continuous across warmup, batches, repeats and waves; no reset"})
        for order_index, (repeat, batch) in enumerate(execution_order(config)):
            label = f"perf-r{repeat}-b{batch}"
            episodes = []
            for wave_index in range(4 // batch):
                wave = f"{label}-w{wave_index}"
                ids = [f"{wave}-r{i}" for i in range(batch)]
                available = {rid: (2 if (i + wave_index + order_index) % 2 == 0 else 0) for i, rid in enumerate(ids)}
                accepted = {rid: (i + order_index) % 3 for i, rid in enumerate(ids) if available[rid]}
                emitted = {rid: 1 + accepted.get(rid, 0) for rid in ids}
                steps = []
                for step_index in (0, 1):
                    decision = {"gamma": 2, "expected_tokens_per_request": state.expected_tokens(2),
                        "candidates": [{"gamma": g, "expected_tokens_per_request": state.expected_tokens(g)} for g in (0, 2)]}
                    a = accepted if step_index == 0 else {rid: 2 for rid in ids}
                    p = {rid: available[rid] for rid in accepted} if step_index == 0 else {rid: 2 for rid in ids}
                    before = available if step_index == 0 else {rid: 2 for rid in ids}
                    if step_index == 0:
                        state.update(list(a.values()), list(p.values()))
                        state.update_availability(list(before.values()), 2)
                    steps.append({"episode": wave, "step": step_index, "phase": "decode", "decode_batch": batch,
                        "scheduled_tokens": batch + sum(p.values()), "scheduled_decode_request_ids": ids,
                        "scheduled_tokens_by_request": {rid: 1 + p.get(rid, 0) for rid in ids},
                        "proposed_before_by_request": before, "scheduled_drafts_by_request": p,
                        "accepted_by_request": a, "observed_drafts_by_request": p,
                        "accepted_drafts": sum(a.values()), "observed_drafts": sum(p.values()),
                        "scheduled_drafts": sum(p.values()), "ewma_eligible_requests": len(a) if step_index == 0 else 0,
                        "ewma_censored_requests": [] if step_index == 0 else ids,
                        "availability_updated": step_index == 0, "proposal_budget_next": 2,
                        "decision": decision, "gamma": 2, "alpha_after": state.alpha})
                episodes.append({"label": wave, "host_steps": [{"index": i, "elapsed_s": i + 1, "host_step_ms": 1} for i in (0, 1)],
                    "requests": [{"request_id": rid, "finished": True, "chunks": [
                        {"elapsed_s": 1, "tokens": emitted[rid]}, {"elapsed_s": 2, "tokens": 1}]} for rid in ids],
                    "plugin": {"config": {"mode": "adaptive", "max_gamma": 2, "proposal_budget": "max", "acceptance_class": "AcceptanceEWMA"},
                        "steps": steps, "acceptance_snapshot": state.snapshot()}})
            dump(folder / (label + ".json"), {"label": label, "batch": batch, "gamma": 2, "proposal_budget": 2,
                "mode": "adaptive", "draft_source": "draft_model", "model": "synthetic_cpu_fixture",
                "measurement_kind": "fixed_length_generation", "all_finished": True, "episodes": episodes})
        return folder, initial

    class Tests(unittest.TestCase):
        def test_exact_continuous_state_and_run_error_weights(self):
            with tempfile.TemporaryDirectory() as directory:
                folder, initial = fixture(directory)
                tracking = []
                engine, runs, ranks = analyze_engine(folder, tracking.append)
                self.assertEqual(engine["status"], "complete")
                self.assertEqual(len(runs), 4)
                self.assertTrue(all(r["status"] == "complete" for r in runs))
                self.assertEqual(tracking[0]["conditional_alpha_before"], initial["alpha"])
                self.assertNotEqual(tracking[0]["conditional_alpha_before"], .7)
                for row in tracking:
                    self.assertAlmostEqual(row["logged_decision_progress_before"], row["state_derived_progress_before"])
                self.assertEqual(sum(r["conditional_alpha_prestep_weight"] for r in runs), sum(r["conditional_tested"] for r in tracking))
                self.assertEqual(sum(r["terminal_request_steps"] for r in runs), 16)
                self.assertEqual(sum(r["eligible_request_steps"] for r in runs), 16)
                self.assertTrue(all(r["availability_exposure"] == 4 for r in ranks))

        def test_missing_initial_never_backfills_first_wave_from_its_future(self):
            with tempfile.TemporaryDirectory() as directory:
                folder, _ = fixture(directory)
                (folder / "evaluation_initial_state.json").unlink()
                tracking = []
                engine, runs, _ = analyze_engine(folder, tracking.append)
                self.assertEqual(engine["status"], "pending")
                self.assertIsNone(tracking[0]["conditional_alpha_before"])
                self.assertIsNone(tracking[1]["prefix_survival_before"])
                self.assertIsNotNone(tracking[0]["logged_decision_progress_before"])
                self.assertTrue(tracking[2]["online_state_recovered"])
                self.assertEqual(runs[0]["status"], "pending")
                self.assertEqual(runs[1]["status"], "complete")

        def test_missing_predecessor_and_mismatched_checkpoint_are_pending(self):
            with tempfile.TemporaryDirectory() as directory:
                folder, _ = fixture(directory)
                (folder / "perf-r0-b2.json").unlink()
                tracking = []
                engine, runs, _ = analyze_engine(folder, tracking.append)
                self.assertEqual(engine["status"], "pending")
                following = [r for r in tracking if "perf-r1-b2" in r["source"]]
                self.assertIsNone(following[0]["prefix_survival_before"])
                self.assertTrue(following[2]["online_state_recovered"])
                self.assertEqual(runs[-1]["status"], "complete")
            with tempfile.TemporaryDirectory() as directory:
                folder, _ = fixture(directory)
                path = folder / "perf-r0-b1.json"
                obj = read(path)
                obj["episodes"][0]["plugin"]["acceptance_snapshot"]["availability"]["2"] = .123
                path.write_text(json.dumps(obj))
                tracking = []
                engine, _, _ = analyze_engine(folder, tracking.append)
                self.assertEqual(engine["status"], "pending")
                self.assertFalse(tracking[0]["online_state_recovered"])
                self.assertIsNotNone(tracking[0]["logged_decision_progress_before"])

        def test_restore_nontrivial_warm_state(self):
            state = AcceptanceEWMA()
            state.update([0, 2], [2, 2])
            state.update_availability([0, 2], 2)
            restored = restore(json.loads(json.dumps(state.snapshot())))
            self.assertTrue(same(state_view(restored.snapshot()), state_view(state.snapshot())))
            self.assertNotEqual(restored.alpha, .7)

        def test_runner_order_is_not_lexicographic(self):
            self.assertEqual(execution_order({"plan": {"batches": [1, 4, 16], "repeats": 2}}),
                [(0, 1), (0, 4), (0, 16), (1, 16), (1, 4), (1, 1)])

        def test_terminal_update_exclusion_and_zero_availability(self):
            state = AcceptanceEWMA()
            before = state.alpha
            step = {"accepted_by_request": {"terminal": 2}, "observed_drafts_by_request": {"terminal": 2},
                "ewma_censored_requests": ["terminal"], "availability_updated": True,
                "scheduled_decode_request_ids": ["terminal", "zero"], "proposed_before_by_request": {"terminal": 2, "zero": 0}}
            apply_recorded_update(state, step, {"mode": "adaptive", "proposal_budget": "max", "max_gamma": 2})
            self.assertEqual(state.alpha, before)
            self.assertEqual(state.expected_tokens(2), 1)
            del step["observed_drafts_by_request"]
            with self.assertRaises(Pending):
                apply_recorded_update(state, step, {})
    result = unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
