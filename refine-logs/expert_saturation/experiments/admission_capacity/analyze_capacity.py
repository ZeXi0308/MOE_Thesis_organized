#!/usr/bin/env python3
"""Recompute finite-episode measurements from retained cells, never fit a policy."""
import argparse
import hashlib
import json
import math
from collections import Counter
from itertools import zip_longest
from pathlib import Path

from metrics import summarize_episode_requests


def read_json(path):
    return json.loads(path.read_text())


def delta(a, b):
    return None if a is None or b is None else a - b


def mean(values):
    return sum(values) / len(values) if values else None


def values(cell):
    m = cell["metrics"]
    return dict(goodput_rps=m["goodput_rps"], ttft_p50_s=m["latency_s"]["ttft"]["p50"],
                tpot_p50_s=m["latency_s"]["tpot"]["p50"])


def execution_exposure(raw, metrics, plan, config):
    """Describe recorded decode snapshots; missing legacy fields are unmeasured."""
    steps = raw.get("steps", [])
    ordinary = [s.get("ordinary", {}) for s in steps]

    def coverage(n):
        return "MEASURED" if n and n == len(steps) else "PARTIAL" if n else "UNMEASURED"

    distributions = {}
    for key in ("actual_active", "decode_requests", "waiting_requests", "future_requests", "target_cap"):
        observed = [s[key] for s in ordinary if s.get(key) is not None]
        if any(type(v) is not int or v < 0 for v in observed):
            raise ValueError(f"invalid recorded {key}")
        distributions[key] = dict(status=coverage(len(observed)), n=len(observed),
            min=min(observed) if observed else None, max=max(observed) if observed else None,
            mean=mean(observed), counts=dict(sorted(Counter(observed).items())))
    pairs = [(s["actual_active"], s["target_cap"]) for s in ordinary
             if s.get("actual_active") is not None and s.get("target_cap") is not None]
    waiting = [s["waiting_requests"] for s in ordinary if s.get("waiting_requests") is not None]
    helper = [s["telemetry_s"] for s in steps if s.get("telemetry_s") is not None]
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in helper):
        raise ValueError("invalid recorded telemetry_s")
    helper_total = sum(helper) if coverage(len(helper)) == "MEASURED" else None
    duration = metrics["observation_duration_s"]
    queue = [r["queue_s"] for r in metrics["per_request"] if r["queue_s"] is not None]
    completed = [r for r in metrics["per_request"] if r["status"] == "completed"]
    complete_cohort = bool(completed) and len(completed) == metrics["n_planned"]
    slo = {}
    for name, key in (("ttft", "ttft_pass"), ("tpot", "tpot_pass"), ("joint", "slo_pass")):
        n_pass = sum(r[key] for r in completed)
        hint = "ALL_PASS" if n_pass == len(completed) else "NONE_PASS" if n_pass == 0 else "MIXED"
        slo[name] = dict(n_completed_pass=n_pass, n_arrived=metrics["n_arrived"],
            complete_cohort_hint=hint if complete_cohort else "INCOMPLETE_OR_EMPTY_COHORT")
    active, target = (distributions[k] for k in ("actual_active", "target_cap"))
    if config.get("cap_schedule"):
        cap_status = "NOT_APPLICABLE_DYNAMIC_CAP"
    elif active["status"] != "MEASURED" or target["status"] != "MEASURED":
        cap_status = "UNAVAILABLE"
    elif set(target["counts"]) != {plan["cap"]}:
        cap_status = "TARGET_MISMATCH"
    else:
        cap_status = "REACHED" if active["max"] >= plan["cap"] else "NOT_REACHED"
    return dict(n_decode_snapshots=len(steps), distributions=distributions,
        static_cap_reached=cap_status, slo_checks=slo,
        target_snapshot=dict(status=coverage(len(pairs)), n=len(pairs),
            at_target_steps=sum(a == b for a, b in pairs) if pairs else None,
            above_target_steps=sum(a > b for a, b in pairs) if pairs else None,
            below_target_steps=sum(a < b for a, b in pairs) if pairs else None,
            interpretation="decode snapshots; above target is allowed during non-preemptive drain; not action-effect timing"),
        waiting_positive_steps=sum(v > 0 for v in waiting) if waiting else None,
        waiting_sample_fraction=sum(v > 0 for v in waiting) / len(steps) if coverage(len(waiting)) == "MEASURED" else None,
        sampling_interpretation="Recorded decode snapshots only; not wall-time occupancy or independent statistical samples. Reaching cap alone does not establish action headroom.",
        admitted_request_queue_s=dict(n=len(queue), n_positive=sum(v > 0 for v in queue),
            n_without_admission=metrics["n_arrived"] - len(queue), mean=mean(queue), max=max(queue) if queue else None,
            interpretation="Arrival-to-admission duration; per-request values remain in metrics. Unadmitted requests are unavailable, not zero."),
        telemetry_helper=dict(status=coverage(len(helper)), n=len(helper), total_s=helper_total,
            episode_wall_fraction=helper_total / duration if helper_total is not None and duration > 0 else None,
            interpretation="telemetry_s helper only; already included in episode time, never add again. Excludes router-logit return and other ON costs; OFF/ON reruns do not isolate causal overhead."))


def exposure_hints(group):
    hints = []
    for cell in group:
        exposure, cap = cell["execution_exposure"], cell["plan"]["cap"]
        measured = exposure["distributions"]
        if any(measured[k]["status"] != "MEASURED" for k in ("actual_active", "decode_requests", "waiting_requests", "target_cap")):
            hints.append(f"EXPOSURE_UNMEASURED_OR_PARTIAL cap={cap}: retain unknown fields; inspect the raw format")
        if measured["waiting_requests"]["status"] == "MEASURED" and exposure["waiting_positive_steps"] == 0:
            hints.append(f"NO_WAITING_AT_DECODE_SNAPSHOTS cap={cap}: prefill or short waits may still occur; inspect request queue times")
        target = exposure["target_snapshot"]
        if target["status"] == "MEASURED" and target["at_target_steps"] == 0:
            hints.append(f"TARGET_NOT_OBSERVED cap={cap}: actual active never equalled its target at decode snapshots")
    widths = [c["execution_exposure"]["distributions"]["decode_requests"] for c in group]
    if len(widths) > 1 and all(w["status"] == "MEASURED" for w in widths) and len({w["max"] for w in widths}) == 1:
        hints.append("SAME_MAX_DECODE_WIDTH: target caps did not expand observed maximum width; inspect distributions before interpreting capacity response")
    return hints


def expected_identity(config, workload, regime):
    source, token_ids = workload["source_requests"], workload["actual_prompt_token_ids"]
    if len(source) != len(token_ids) or len(source) != config["requests"]:
        raise ValueError("frozen workload/token ID lengths disagree")
    if regime not in ("steady", "bursty"):
        raise ValueError("unknown arrival regime")
    expected = {}
    for i, (row, ids) in enumerate(zip(source, token_ids)):
        position = i if regime == "steady" else (i // config["burst_size"]) * config["burst_size"]
        arrival = position * config["arrival_gap_s"]
        if "arrival_traces_s" in workload:
            trace = workload["arrival_traces_s"][regime]
            if len(trace) != len(source):
                raise ValueError("frozen arrival trace length disagrees")
            arrival = trace[i]
        if not math.isfinite(arrival) or arrival < 0:
            raise ValueError("frozen arrival must be finite and nonnegative")
        if row["request_id"] in expected:
            raise ValueError("duplicate frozen workload request ID")
        expected[row["request_id"]] = (row["document_id"], arrival,
            hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest(), len(ids))
    return expected


def identity_errors(raw, expected):
    rows = raw["requests"]
    actual = {r["request_id"]: r for r in rows}
    errors = []
    if len(actual) != len(rows) or set(actual) != set(expected):
        errors.append("request cohort differs from frozen workload")
    for request_id in set(actual) & set(expected):
        row, (document, arrival, prompt_hash, tokens) = actual[request_id], expected[request_id]
        if (row.get("document_id"), row.get("prompt_token_ids_sha256"), row.get("prompt_tokens")) != (document, prompt_hash, tokens):
            errors.append(f"document/prompt identity mismatch: {request_id}")
        if abs(row["arrival_s"] - arrival) > 1e-9:
            errors.append(f"arrival identity mismatch: {request_id}")
    return errors


def paired_telemetry(off, on):
    a, b = off["raw"], on["raw"]
    ar, br = ({r["request_id"]: r for r in x["requests"]} for x in (a, b))
    sequences = [[s["request_ids"] for s in x["steps"]] for x in (a, b)]
    differences = {}
    for key in ("admission_s", "completion_s", "token_times_s", "output_token_ids"):
        differences[key] = [rid for rid in sorted(ar) if ar[rid][key] != br[rid][key]]
    latency = {key: delta(value, values(off)[key]) for key, value in values(on).items()}
    itls = [[v for r in c["metrics"]["per_request"] for v in r["itl_s"]] for c in (off, on)]
    return dict(repeat=off["plan"]["repeat"], regime=off["plan"]["regime"], cap=off["plan"]["cap"],
        off_cell=off["cell"], on_cell=on["cell"], interpretation="independent rerun diagnostic, not isolated causal overhead",
        on_minus_off= dict(wall_time_s=delta(on["metrics"]["observation_duration_s"],
                                           off["metrics"]["observation_duration_s"]),
                           itl_mean_s=delta(mean(itls[1]), mean(itls[0])), **latency),
        differing_request_ids=differences,
        membership_sequence_equal=sequences[0] == sequences[1],
        membership_different_step_positions=sum(x != y for x, y in zip_longest(*sequences)),
        output_mismatch_is_diagnostic=True,
        pressure_association="ON trajectory only; post-action association; never assigned to OFF")


def analyze(run_dir):
    root = Path(run_dir)
    config = read_json(root / "config.json")
    plans, cells = config["run_order"], []
    present = [i for i in range(len(plans)) if (root / f"cell-{i:03d}.json").exists()]
    result = dict(input_kind=config.get("fixture_kind", "recorded_bundle_provenance_not_certified"),
        verdict="UNRUN", complete_scan=False, identity_validation="not_checked_no_cells",
        evidence_type="CUSTOM_CONTINUOUS_RUNTIME" if present else "UNRUN",
        claim_ceiling="exploratory request measurements; no native-serving or causal controller claim",
        action_increment_verdict="NO_ACTION_INCREMENT_ESTABLISHED",
        oracle_status="only observed static cap comparison; no action-conditioned oracle" if present else "UNRUN",
        low_sample_warning="Per-cell n < 100 makes tail percentiles descriptive; repeats are reported separately.",
        denominator="all arrived requests; failed and unfinished retain their place in attainment denominator",
        cells=cells, static_comparisons=[], telemetry_pairs=[], issues=[])
    workload = None
    if present:
        try:
            workload = read_json(root / "workload.json")
        except (OSError, ValueError) as exc:
            result["issues"].append(f"workload unavailable: {exc}")
    for index, plan in enumerate(plans):
        cell = dict(cell=index, plan=plan, state="MISSING", issues=[])
        cells.append(cell)
        if index not in present:
            continue
        try:
            raw = read_json(root / f"cell-{index:03d}.json")
            cell["raw"] = raw
            if raw.get("plan") != plan:
                cell["issues"].append("cell plan differs from config.run_order")
            if config.get("cell_checks_required"):
                try:
                    checks = read_json(root / f"checks-{index:03d}.json")
                    if checks.get("status") != "PASS":
                        cell["issues"].append(f"post-cell isolation check not PASS: {checks.get('reason', checks.get('status'))}")
                except (OSError, ValueError, AttributeError) as exc:
                    cell["issues"].append(f"post-cell isolation check unavailable: {type(exc).__name__}")
            if workload is None:
                cell["issues"].append("frozen workload unavailable")
            else:
                cell["issues"].extend(identity_errors(raw, expected_identity(config, workload, plan["regime"])))
            cell["metrics"] = summarize_episode_requests(raw["requests"], observation_end_s=raw["observation_end_s"],
                ttft_slo_s=config["ttft_slo_s"], tpot_slo_s=config["tpot_slo_s"])
            cell["execution_exposure"] = execution_exposure(raw, cell["metrics"], plan, config)
            complete = raw["status"] == "COMPLETE" and cell["metrics"]["n_completed"] == len(raw["requests"])
            cell["state"] = "INVALID" if cell["issues"] else ("COMPLETE" if complete else "INCOMPLETE")
            if plan["telemetry"]:
                own = [p for s in raw["steps"] for p in s["pressure"]]
                cell["own_on_pressure"] = {k: dict(n=len(v), mean=mean(v)) for k in ("U", "C")
                    for v in [[p[k] for p in own if p[k] is not None]]}
                cell["pressure_interpretation"] = "ON step/layer unweighted means; post-action association only"
        except (OSError, KeyError, TypeError, ValueError) as exc:
            cell["state"] = "INVALID"
            cell["issues"].append(f"raw validation: {type(exc).__name__}: {exc}")
    if present:
        result["identity_validation"] = "failed" if any(c["state"] == "INVALID" for c in cells) else "passed_present_cells_only"
        result["complete_scan"] = bool(cells) and all(c["state"] == "COMPLETE" for c in cells)
        result["verdict"] = ("INVALID_EXPERIMENT" if result["identity_validation"] == "failed" else
                             "MEASUREMENT_ONLY" if result["complete_scan"] else "PARTIAL_MEASUREMENT")
    eligible = [c for c in cells if c["state"] in ("COMPLETE", "INCOMPLETE")]
    keys = [(c["plan"]["repeat"], c["plan"]["regime"], c["plan"]["cap"], c["plan"]["telemetry"]) for c in eligible]
    if len(keys) != len(set(keys)):
        result.update(verdict="INVALID_EXPERIMENT", complete_scan=False)
        result["issues"].append("duplicate repeat/regime/cap/telemetry plan; comparisons withheld")
        eligible = []
    if config.get("cap_schedule"):
        result["issues"].append("schedule experiment: static optimum comparison is inapplicable")
    else:
        for repeat, regime in sorted({(c["plan"]["repeat"], c["plan"]["regime"]) for c in eligible}):
            group = [c for c in eligible if c["state"] == "COMPLETE" and not c["plan"]["telemetry"] and
                     (c["plan"]["repeat"], c["plan"]["regime"]) == (repeat, regime)]
            group = [c for c in group if c["metrics"]["goodput_rps"] is not None]
            if not group:
                continue
            best = max(group, key=lambda c: (c["metrics"]["goodput_rps"], -c["plan"]["cap"]))
            candidates = [dict(cell=c["cell"], cap=c["plan"]["cap"], state=c["state"], **values(c),
                minus_observed_best={k: delta(v, values(best)[k]) for k, v in values(c).items()}) for c in group]
            planned = [p for p in plans if not p["telemetry"] and (p["repeat"], p["regime"]) == (repeat, regime)]
            complete = len(group) == len(planned) and all(c["state"] == "COMPLETE" for c in group)
            ordered = sorted(group, key=lambda c: c["plan"]["cap"])
            adjacent = [dict(lower_cap=lo["plan"]["cap"], higher_cap=hi["plan"]["cap"],
                high_minus_low={**{k: delta(v, values(lo)[k]) for k, v in values(hi).items()},
                    "n_slo_pass": hi["metrics"]["n_slo_pass"] - lo["metrics"]["n_slo_pass"]})
                for lo, hi in zip(ordered, ordered[1:])]
            hints = []
            if complete and all(c["metrics"]["goodput_rps"] == 0 for c in group):
                hints.append("ALL_ZERO_GOODPUT: current SLO/load provides no passing-request separation; inspect latency before a newly preregistered calibration run")
            if complete and all(c["metrics"]["slo_attainment"] == 1 for c in group):
                hints.append("ALL_ATTAIN_ONE: no SLO violations observed at this load; a separately frozen higher-load/tighter-SLO run may expose the boundary")
            hints.extend(exposure_hints(group))
            result["static_comparisons"].append(dict(repeat=repeat, regime=regime,
                observed_best_cap=best["plan"]["cap"], tie_rule="smaller cap", candidates=candidates,
                complete_off_scan=complete, adjacent_cap_deltas=adjacent, calibration_hints=hints,
                interpretation="hindsight observed static grid; no online selection or causal increment"))
    lookup = {(c["plan"]["repeat"], c["plan"]["regime"], c["plan"]["cap"], c["plan"]["telemetry"]): c for c in eligible}
    for key, off in lookup.items():
        if not key[-1] and (*key[:-1], True) in lookup:
            result["telemetry_pairs"].append(paired_telemetry(off, lookup[(*key[:-1], True)]))
    for cell in cells:
        cell.pop("raw", None)
    return result


def report(result):
    lines = [f"# Admission capacity analysis: {result['verdict']}", "",
        f"Input: {result['input_kind']}. Complete scan: {result['complete_scan']}.", "",
        result["claim_ceiling"], result["low_sample_warning"], result["denominator"], "",
        "Incomplete cells remain in the accounting table and are excluded from best-cap selection.", "",
        "ON pressure belongs only to its own evolving trajectory. OFF/ON differences are paired rerun diagnostics,",
        "including token/membership/timing changes; they do not isolate causal telemetry overhead.", "",
        "| Cell | Regime | Repeat | Cap | Telemetry | State | Goodput rps | Arrived / completed / failed / unfinished |",
        "|---|---|---|---|---|---|---|---|"]
    for c in result["cells"]:
        p, m = c["plan"], c.get("metrics", {})
        counts = " / ".join(str(m.get(k, "—")) for k in ("n_arrived", "n_completed", "n_failed", "n_unfinished"))
        lines.append(f"| {c['cell']} | {p['regime']} | {p['repeat']} | {p['cap']} | {p['telemetry']} | {c['state']} | {m.get('goodput_rps', '—')} | {counts} |")
        lines.extend(f"\nCell {c['cell']} issue: {issue}" for issue in c["issues"])
        if "own_on_pressure" in c:
            lines.append(f"\nCell {c['cell']} own ON pressure: {json.dumps(c['own_on_pressure'])}; post-action association only.")
        if "execution_exposure" in c:
            lines.append(f"\nCell {c['cell']} execution exposure: {json.dumps(c['execution_exposure'])}.")
    for group in result["static_comparisons"]:
        lines.extend(["", f"Repeat {group['repeat']} / {group['regime']}: observed best OFF cap = {group['observed_best_cap']}; "
                      f"complete OFF scan = {group['complete_off_scan']}. Hindsight comparison only."])
        lines.extend(["", "| Cap | Goodput rps | TTFT p50 s | TPOT p50 s | Differences from observed best |",
                      "|---|---|---|---|---|"])
        for c in group["candidates"]:
            lines.append(f"| {c['cap']} | {c['goodput_rps']} | {c['ttft_p50_s']} | {c['tpot_p50_s']} | {json.dumps(c['minus_observed_best'])} |")
        for change in group["adjacent_cap_deltas"]:
            lines.append(f"\nAdjacent OFF caps {change['lower_cap']} → {change['higher_cap']}, high minus low: {json.dumps(change['high_minus_low'])}.")
        lines.extend(f"\nCalibration: {hint}. Retain the original run and SLO thresholds unchanged." for hint in group["calibration_hints"])
    for pair in result["telemetry_pairs"]:
        lines.extend(["", f"OFF/ON cells {pair['off_cell']}/{pair['on_cell']}: ON minus OFF {json.dumps(pair['on_minus_off'])}.",
            f"Membership sequence equal: {pair['membership_sequence_equal']}; differing request identities by field: "
            f"{json.dumps(pair['differing_request_ids'])}. Token mismatch is diagnostic and does not alone invalidate a rerun."])
    answer = ("No cells were run; capacity response and action increment remain unmeasured."
              if result["verdict"] == "UNRUN" else
              "Raw validation errors prevent a scientific conclusion from this bundle."
              if result["verdict"] == "INVALID_EXPERIMENT" else
              "Only the observed static response is described; an online action increment is not established.")
    lines.extend(["", *result["issues"], "", f"Action increment verdict: {result['action_increment_verdict']}.",
                  f"Research answer: {answer}"])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.run_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "analysis.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    (args.output_dir / "report.md").write_text(report(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
