#!/usr/bin/env python3
"""Build complete-step timing tables, acceptance summaries and control overhead.

  python build_calibration.py runs/CAMPAIGN --output runs/CAMPAIGN/calibration

Inputs are run_matrix.py perf-r*-b*.json runs, including fixed-population waves.
Calibration uses only target pure-decode steps whose actual scheduled batch is
the requested B. It excludes terminal/censored steps and trace-on episodes.
One host_steps[index] must match one plugin.steps[index]. Host timing contains
verification, next-proposal work and host gaps under the supported synchronous
runner. No asynchronously overlapping component times are added/subtracted.

Each row supplied to CalibrationTable is one RUN's mean of its eligible wave
mean step costs. Wave and step counts remain diagnostic subsamples, not repeat
counts. The full proposal budget is kept explicitly. Static gamma=3,budget=3
cannot calibrate truncating gamma=3 AFTER generating budget=7.

F5 includes zero-proposal requests, available/scheduled prefix counts, conditional
acceptance, and uncensored accepted-prefix distributions. F6 reports measured
controller CPU costs and discarded draft token counts; discarded tokens are not
a measurement of wasted GPU time or energy.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics

from cost_model import CalibrationTable
from plot_results import (REGIME_FIELDS, _bool, _context, _finite, write_csv,
                          resolve_regime, route_instrumented)


def _quantile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lo = int(position)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def _decode_ids(step: dict) -> list[str]:
    if "scheduled_decode_request_ids" in step:
        return list(step["scheduled_decode_request_ids"])
    if step.get("phase") == "decode":
        return list(step.get("scheduled_tokens_by_request", {}))
    return []


def analyze_episode(episode: dict, target_batch: int, configured_gamma: int,
                    proposal_budget: int, source: str) -> dict:
    """Return diagnostics and one wave cost; raises on ambiguous step alignment."""
    hosts = episode.get("host_steps", [])
    steps = episode.get("plugin", {}).get("steps", [])
    if len(hosts) != len(steps):
        raise ValueError(f"{source}/{episode.get('label')}: host/plugin step count mismatch "
                         f"{len(hosts)} != {len(steps)}")
    costs, progress, overhead = [], [], []
    skips = Counter()
    acceptance = {"eligible_requests": 0, "zero_scheduled_requests": 0,
                  "accepted_drafts": 0, "tested_positions": 0, "scheduled_drafts": 0,
                  "missing_acceptance_requests": 0, "available_before": 0,
                  "available_prefix": Counter(), "scheduled_prefix": Counter(),
                  "accepted_prefix": Counter(), "accepted_histogram": Counter()}
    # Older plugin snapshots did not mark zero-draft terminal requests. The
    # runner records each final emitted chunk at the exact host-step timestamp;
    # use that independent evidence as well, without dropping cold outliers.
    terminal_times = {float(r["chunks"][-1]["elapsed_s"])
                      for r in episode.get("requests", [])
                      if r.get("finished") and r.get("chunks")}
    for index, (host, step) in enumerate(zip(hosts, steps)):
        if int(host.get("index", -1)) != index or int(step.get("step", -1)) != index:
            raise ValueError(f"{source}: nonmatching host/plugin index at {index}")
        if step.get("episode") not in (None, episode.get("label")):
            raise ValueError(f"{source}: step belongs to a different wave")
        duration = _finite(host.get("host_step_ms"))
        if duration is None or duration <= 0:
            raise ValueError(f"{source}: invalid complete-step host duration")
        ids = _decode_ids(step)
        censored = set(step.get("ewma_censored_requests", []))
        if _finite(host.get("elapsed_s")) in terminal_times:
            censored.add("__runner_observed_terminal_chunk__")
        full_decode = step.get("phase") == "decode" and int(step.get("decode_batch", -1)) == target_batch
        if not full_decode:
            skips["not_full_target_decode_batch"] += 1
        elif censored:
            skips["terminal_or_output_budget_censored"] += 1
        elif int(step.get("proposal_budget_next", proposal_budget)) != proposal_budget:
            skips["next_proposal_budget_changed"] += 1
        else:
            costs.append(duration)

        if int(step.get("scheduled_tokens", 0)) > 0:
            decision = step.get("decision") or {}
            overhead.append({"source": source, "wave": episode.get("label"), "step": index,
                "phase": step.get("phase"), "target_batch": target_batch,
                "actual_decode_batch": step.get("decode_batch"), "configured_gamma": configured_gamma,
                "selected_gamma": step.get("gamma"), "proposal_budget": proposal_budget,
                "control_us": float(step.get("control_ns", 0)) / 1000,
                "policy_choose_us": _finite(decision.get("decision_cpu_us")),
                "acceptance_update_us": float(step.get("acceptance_update_ns", 0)) / 1000,
                "schedule_instrumentation_us": float(step.get("schedule_instrumentation_ns", 0)) / 1000,
                "drafts_available_before": int(step.get("drafts_available_before", 0)),
                "drafts_truncated_by_controller": int(step.get("truncated_by_controller", 0)),
                "scheduled_drafts": int(step.get("scheduled_drafts", 0)),
                "host_step_ms": duration})

        if not full_decode or censored:
            continue
        if len(ids) != target_batch:
            skips["missing_decode_request_identity_for_acceptance"] += 1
            continue
        available = step.get("proposed_before_by_request", {})
        scheduled = step.get("scheduled_drafts_by_request", {})
        verified = step.get("observed_drafts_by_request")
        accepted = step.get("accepted_by_request", {})
        sample_outputs = []
        tested_here = accepted_here = 0
        for rid in ids:
            p = int(scheduled.get(rid, 0))
            if p > 0 and rid not in accepted:
                acceptance["missing_acceptance_requests"] += 1
                continue
            actual_p = int(verified.get(rid, p)) if verified is not None else p
            a = int(accepted.get(rid, 0))
            if not 0 <= a <= actual_p <= p:
                raise ValueError(f"{source}: invalid accepted/verified/scheduled counts for {rid}")
            full_p = int(available.get(rid, 0))
            if full_p < p:
                raise ValueError(f"{source}: scheduled drafts exceed pre-truncation available drafts")
            acceptance["eligible_requests"] += 1
            acceptance["zero_scheduled_requests"] += int(p == 0)
            acceptance["accepted_drafts"] += a
            acceptance["tested_positions"] += min(a + 1, actual_p)
            acceptance["scheduled_drafts"] += p
            acceptance["available_before"] += full_p
            acceptance["accepted_histogram"][a] += 1
            sample_outputs.append(a + 1)
            tested_here += min(a + 1, actual_p)
            accepted_here += a
            for rank in range(1, max(proposal_budget, full_p) + 1):
                acceptance["available_prefix"][rank] += int(full_p >= rank)
                acceptance["scheduled_prefix"][rank] += int(actual_p >= rank)
                acceptance["accepted_prefix"][rank] += int(a >= rank)
        decision = step.get("decision") or {}
        predicted = _finite(decision.get("expected_tokens_per_request"))
        observed = statistics.mean(sample_outputs) if sample_outputs else None
        observed_alpha = accepted_here / tested_here if tested_here else None
        post_alpha = _finite(step.get("alpha_after"))
        progress.append({"source": source, "wave": episode.get("label"), "step": index,
            "batch": target_batch, "configured_gamma": configured_gamma,
            "selected_gamma": step.get("gamma"), "proposal_budget": proposal_budget,
            "eligible_requests": len(sample_outputs), "mean_output_tokens": observed,
            "predicted_output_tokens_before_step": predicted,
            "output_prediction_error": predicted - observed if predicted is not None and observed is not None else None,
            "conditional_alpha_observed": observed_alpha, "alpha_after_update": post_alpha,
            "alpha_postupdate_tracking_error": post_alpha - observed_alpha
                if post_alpha is not None and observed_alpha is not None else None,
            "draft_count_source": "native_verified" if verified is not None else "scheduled_uncensored_proxy"})
    return {"source": source, "wave": episode.get("label"), "batch": target_batch,
            "gamma": configured_gamma, "proposal_budget": proposal_budget,
            "eligible_steps": len(costs), "total_steps": len(steps),
            "mean_step_ms": statistics.mean(costs) if costs else None,
            "median_step_ms": statistics.median(costs) if costs else None,
            "step_ms_min": min(costs) if costs else None, "step_ms_max": max(costs) if costs else None,
            "excluded_steps": dict(skips), "acceptance": acceptance,
            "progress": progress, "overhead": overhead}


def analyze_run(record: dict, source: str, context: dict | None = None) -> dict:
    context = context or {}
    if route_instrumented(record, context):
        raise ValueError("trace-on episodes cannot calibrate performance")
    if record.get("all_finished") is not None and not _bool(record["all_finished"]):
        raise ValueError("incomplete run cannot calibrate a completed-population service")
    batch, gamma = int(record["batch"]), int(record["gamma"])
    budget = int(record.get("proposal_budget", gamma))
    episodes = record.get("episodes", [record])
    wave_rows = [analyze_episode(episode, batch, gamma, budget, source) for episode in episodes]
    means = [wave["mean_step_ms"] for wave in wave_rows if wave["mean_step_ms"] is not None]
    regime = resolve_regime(record, context)
    calibration = None
    if means and record.get("mode", "native") in ("native", "static") and not context.get("native_batch_table"):
        calibration = {**regime, "batch": batch, "gamma": gamma, "proposal_budget": budget,
            "total_step_ms": statistics.mean(means), "run_id": record.get("label", Path(source).stem),
            "source": source, "eligible_waves": len(means), "total_waves": len(wave_rows),
            "eligible_steps": sum(w["eligible_steps"] for w in wave_rows),
            "timing_boundary": "complete synchronous host engine.step; includes current verify and next draft",
            "aggregation": "one run: mean of eligible wave mean step times",
            "evidence_kind": "measured_complete_step_cost"}
    counters = Counter()
    available, scheduled, accepted, histogram = Counter(), Counter(), Counter(), Counter()
    for wave in wave_rows:
        stats = wave["acceptance"]
        for key in ("eligible_requests", "zero_scheduled_requests", "accepted_drafts", "tested_positions",
                    "scheduled_drafts", "missing_acceptance_requests", "available_before"):
            counters[key] += stats[key]
        available.update(stats["available_prefix"])
        scheduled.update(stats["scheduled_prefix"])
        accepted.update(stats["accepted_prefix"])
        histogram.update(stats["accepted_histogram"])
    count = counters["eligible_requests"]
    acceptance_rows = [{**regime, "source": source, "run_id": record.get("label"),
        "batch": batch, "gamma": gamma, "proposal_budget": budget, "rank": rank,
        "eligible_requests": count, "available_at_rank": available[rank],
        "scheduled_at_rank": scheduled[rank], "accepted_at_rank": accepted[rank],
        "availability_survival": available[rank] / count if count else None,
        "conditional_acceptance_survival": accepted[rank] / scheduled[rank] if scheduled[rank] else None,
        "unconditional_accepted_survival": accepted[rank] / count if count else None,
        "conditional_alpha": counters["accepted_drafts"] / counters["tested_positions"]
                             if counters["tested_positions"] else None,
        "mean_output_tokens": 1 + counters["accepted_drafts"] / count if count else None,
        "accepted_length_histogram": json.dumps(dict(sorted(histogram.items()))),
        "zero_scheduled_requests": counters["zero_scheduled_requests"],
        "missing_acceptance_requests": counters["missing_acceptance_requests"],
        "boundary": "uncensored full-B decode; ranks/requests are not independent run repeats"}
        for rank in range(1, budget + 1)]
    all_overhead = [row for wave in wave_rows for row in wave["overhead"]]
    control_rows = []
    for phase in sorted({row["phase"] for row in all_overhead}):
        rows = [row for row in all_overhead if row["phase"] == phase]
        control = {**regime, "source": source, "run_id": record.get("label"), "mode": record.get("mode"),
            "batch": batch, "gamma": gamma, "proposal_budget": budget, "phase": phase, "steps": len(rows),
            "drafts_available_before": sum(r["drafts_available_before"] for r in rows),
            "drafts_truncated_by_controller": sum(r["drafts_truncated_by_controller"] for r in rows),
            "scheduled_drafts": sum(r["scheduled_drafts"] for r in rows)}
        for metric in ("control_us", "policy_choose_us", "acceptance_update_us", "schedule_instrumentation_us"):
            values = [r[metric] for r in rows if r[metric] is not None]
            control[metric + "_mean"] = statistics.mean(values) if values else None
            control[metric + "_p50"] = _quantile(values, .5)
            control[metric + "_p95"] = _quantile(values, .95)
            control[metric + "_max"] = max(values) if values else None
        control["discarded_available_token_fraction"] = control["drafts_truncated_by_controller"] / control["drafts_available_before"] \
            if control["drafts_available_before"] else None
        control["boundary"] = "discarded token counts do not quantify wasted GPU time or energy"
        control_rows.append(control)
    mode = "native_batch_table" if context.get("native_batch_table") else record.get("mode", "native")
    for row in acceptance_rows + control_rows:
        row["mode"] = mode
    progress_rows = [{**row, **regime, "mode": mode} for wave in wave_rows for row in wave["progress"]]
    return {"calibration": calibration, "waves": wave_rows, "acceptance": acceptance_rows,
            "progress": progress_rows, "control": control_rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    paths = set()
    for item in map(Path, args.inputs):
        paths.update(path.resolve() for path in item.rglob("perf-*.json") if path.is_file()) if item.is_dir() else paths.add(item.resolve())
    groups = defaultdict(list)
    waves, acceptance, progress, control, exclusions = [], [], [], [], []
    for path in sorted(paths):
        record = json.loads(path.read_text())
        try:
            result = analyze_run(record, str(path), _context(path))
        except (KeyError, TypeError, ValueError) as exc:
            exclusions.append({"source": str(path), "reason": str(exc)})
            continue
        if result["calibration"] is not None:
            key = tuple(result["calibration"][k] for k in REGIME_FIELDS)
            groups[key].append(result["calibration"])
        else:
            exclusions.append({"source": str(path), "reason": "no eligible static full-B nonterminal decode cost"})
        waves.extend({k: (json.dumps(v) if k == "excluded_steps" else v) for k, v in wave.items()
                      if k not in ("acceptance", "progress", "overhead")} for wave in result["waves"])
        acceptance.extend(result["acceptance"])
        progress.extend(result["progress"])
        control.extend(result["control"])
    write_csv(output / "wave_costs.csv", waves)
    write_csv(output / "acceptance_survival.csv", acceptance)
    write_csv(output / "acceptance_tracking.csv", progress)
    write_csv(output / "control_overhead.csv", control)
    (output / "exclusions.json").write_text(json.dumps(exclusions, indent=2))
    index = []
    for key, records in sorted(groups.items()):
        metadata = {**dict(zip(REGIME_FIELDS, key)),
            "draft_cost_nonoverlap": False,
            "timing_boundary": "complete synchronous host step; no asynchronous component arithmetic",
            "record_unit": "run mean of eligible wave mean costs",
            "budget_semantics": "full generated budget, before truncation; retain proposal_budget in lookup",
            "evidence_kind": "measured_complete_step_calibration"}
        CalibrationTable(records, metadata)  # Validate before making a usable artifact.
        name = "calibration.json" if len(groups) == 1 else "calibration_" + hashlib.sha256(json.dumps(key).encode()).hexdigest()[:10] + ".json"
        (output / name).write_text(json.dumps({"metadata": metadata, "records": records}, indent=2))
        index.append({"path": str(output / name), "regime": dict(zip(REGIME_FIELDS, key)), "run_records": len(records)})
    (output / "calibration_index.json").write_text(json.dumps(index, indent=2))
    print(json.dumps({"calibration_tables": index, "excluded_files": len(exclusions), "output": str(output)}))


if __name__ == "__main__":
    main()
