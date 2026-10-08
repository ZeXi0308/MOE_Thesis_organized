#!/usr/bin/env python3
"""Offline, prespecified one-at-a-time SLO sensitivity of all old high-load runs.

Reads existing traces without changing them. Uses only Python's standard library.
Run: python3 -B sensitivity.py [raw.json or directories ...] [--output RESULT.json]
Without inputs, analyzes the three original high-load phases. Explicit inputs
default to sensitivity_custom.json, preserving the old sensitivity.json.
"""

import argparse
import collections
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys


PHASES = ("high-normal-r01", "extra-controls-high", "extra-controls-high384")
BASE_SLO = {"ttft_s": 4.0, "gap_s": 0.100, "completion_s": 20.0}
SCENARIOS = (
    ("primary", BASE_SLO),
    ("ttft_2s", {**BASE_SLO, "ttft_s": 2.0}),
    ("ttft_6s", {**BASE_SLO, "ttft_s": 6.0}),
    ("gap_50ms", {**BASE_SLO, "gap_s": 0.050}),
    ("gap_150ms", {**BASE_SLO, "gap_s": 0.150}),
    ("completion_15s", {**BASE_SLO, "completion_s": 15.0}),
    ("completion_25s", {**BASE_SLO, "completion_s": 25.0}),
)
CONDITIONS = ("ttft_s", "gap_s", "completion_s", "unfinished", "invalid_measurement")
COHORTS = ("background_t0", "background_t3", "long_wave1", "long_wave2", "long_wave3")


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def percentile(values, p=0.95):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * p
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def numeric_request_id(value):
    prefix, suffix = value.rsplit("-", 1)
    return prefix, int(suffix)


def cohort_of(request):
    request_id, arrival = request["request_id"], request["arrival_s"]
    if request_id.startswith("high-background-") and arrival in (0, 3):
        return "background_t%d" % arrival
    if request_id.startswith("high-long-"):
        index = int(request_id.rsplit("-", 1)[1])
        if 0 <= index < 43:
            return "long_wave1"
        if 43 <= index < 86:
            return "long_wave2"
        if 86 <= index < 128:
            return "long_wave3"
    raise ValueError("Unrecognized old high-load cohort: %s at %s" % (request_id, arrival))


def extract_request(request, elapsed):
    arrival = request["arrival_s"]
    times = request["token_times_s"]
    token_count = len(request["output_token_ids"])
    finished = request.get("finished") is True
    valid = (
        finite(arrival) and 0 <= arrival <= elapsed
        and token_count == len(times) == request["max_tokens"] and token_count > 0
        and all(finite(t) and arrival <= t <= elapsed for t in times)
        and all(a <= b for a, b in zip(times, times[1:]))
    )
    completion = request.get("completion_s")
    if finished and (not finite(completion) or not arrival <= completion <= elapsed
                     or (times and finite(times[-1]) and completion < times[-1])):
        valid = False
    return {
        "request_id": request["request_id"], "cohort": cohort_of(request),
        "arrival_s": arrival, "prompt_tokens": request["prompt_tokens"],
        "max_tokens": request["max_tokens"], "output_tokens": token_count,
        "finished": finished, "valid_measurement": bool(valid),
        "ttft_s": times[0] - arrival if valid else None,
        "gap_s": max((b - a for a, b in zip(times, times[1:])), default=0.0) if valid else None,
        "completion_s": completion - arrival if valid and finished else None,
    }


def failed_conditions(request, slo):
    failed = [key for key in BASE_SLO if request[key] is None or request[key] > slo[key]]
    if not request["finished"]:
        failed.append("unfinished")
    if not request["valid_measurement"]:
        failed.append("invalid_measurement")
    return failed


def evaluate(requests, slo, elapsed):
    marginal = collections.Counter()
    exclusive = collections.Counter()
    qualified_ids = []
    qualified_tokens = 0
    for request in requests:
        failed = failed_conditions(request, slo)
        marginal.update(failed)
        exclusive["+".join(failed) if failed else "pass"] += 1
        if not failed:
            qualified_ids.append(request["request_id"])
            qualified_tokens += request["output_tokens"]
    assert sum(exclusive.values()) == len(requests)
    return {
        "request_count": len(requests),
        "finished_count": sum(q["finished"] for q in requests),
        "qualified_request_count": len(qualified_ids),
        "qualified_request_fraction": len(qualified_ids) / len(requests) if requests else 0.0,
        "elapsed_s_denominator": elapsed,
        "goodput_requests_per_s": len(qualified_ids) / elapsed,
        "qualified_output_tokens": qualified_tokens,
        "goodput_output_tokens_per_s": qualified_tokens / elapsed,
        "single_constraint_fail_counts_nonexclusive": {key: marginal[key] for key in CONDITIONS},
        "exclusive_failure_combinations": dict(sorted(exclusive.items())),
        "qualified_request_ids": sorted(qualified_ids, key=numeric_request_id),
    }


def load_runs(data_root):
    runs = []
    workload_signature = None
    workload_hash = None
    phase_metadata = {}
    for phase in PHASES:
        phase_root = data_root / phase
        protocol = json.loads((phase_root / "protocol.json").read_text())
        summary = json.loads((phase_root / "summary.json").read_text())
        if protocol["slo"] != BASE_SLO:
            raise ValueError("Unexpected original SLO in " + phase)
        if workload_hash is None:
            workload_hash = protocol["workload_sha256"]
        if protocol["workload_sha256"] != workload_hash:
            raise ValueError("High-load phases do not use the same workload hash")
        phase_metadata[phase] = {
            "protocol_path": str((phase_root / "protocol.json").resolve()),
            "workload_sha256": protocol["workload_sha256"],
            "original_slo": protocol["slo"],
            "expected_requests_per_run": protocol["request_count"],
        }
        for report in sorted(summary["runs"], key=lambda x: x["summary"]["run"]):
            row = report["summary"]
            relative = Path(row["run"])
            if row.get("is_warmup") or any(p.lower().startswith("warm") for p in relative.parts):
                continue
            path = phase_root / relative / "raw.json"
            raw_bytes = path.read_bytes()
            raw = json.loads(raw_bytes)
            elapsed = raw["elapsed_s"]
            if not finite(elapsed) or elapsed <= 0:
                raise ValueError("Invalid whole-run elapsed_s: " + str(path))
            requests = [extract_request(q, elapsed) for q in raw["requests"]]
            ids = [q["request_id"] for q in requests]
            if len(ids) != len(set(ids)):
                raise ValueError("Duplicate request IDs in " + str(path))
            signature = sorted((q["request_id"], q["arrival_s"], q["prompt_tokens"], q["max_tokens"]) for q in requests)
            if workload_signature is None:
                workload_signature = signature
            if signature != workload_signature:
                raise ValueError("Actual request identities differ in " + str(path))
            if len(requests) != protocol["request_count"]:
                raise ValueError("Missing requests relative to protocol in " + str(path))
            evaluated = {}
            for scenario, slo in SCENARIOS:
                overall = evaluate(requests, slo, elapsed)
                cohorts = {name: evaluate([q for q in requests if q["cohort"] == name], slo, elapsed) for name in COHORTS}
                assert sum(x["qualified_request_count"] for x in cohorts.values()) == overall["qualified_request_count"]
                evaluated[scenario] = {"overall": overall, "cohorts": cohorts}
            primary = evaluated["primary"]["overall"]
            if primary["qualified_request_count"] != row["joint_slo_qualified_requests"]:
                raise ValueError("Recomputed primary SLO differs from old summary: " + str(path))
            if not math.isclose(primary["goodput_requests_per_s"], row["joint_slo_requests_per_s"], rel_tol=1e-12):
                raise ValueError("Recomputed primary goodput differs from old summary: " + str(path))
            runs.append({
                "run_id": phase + "/" + row["run"], "phase": phase,
                "run": row["run"], "policy": raw["policy"],
                "raw_path": str(path.resolve()), "raw_sha256": hashlib.sha256(raw_bytes).hexdigest(),
                "elapsed_s": elapsed, "request_count": len(requests),
                "finished_count": sum(q["finished"] for q in requests),
                "output_tokens": sum(q["output_tokens"] for q in requests),
                "output_tokens_per_s": sum(q["output_tokens"] for q in requests) / elapsed,
                "original_summary_valid": row["valid"],
                "all_request_measurements_valid": all(q["valid_measurement"] for q in requests),
                "scenarios": evaluated, "_requests": requests,
            })
    return runs, phase_metadata


def load_analyzer():
    """Reuse the established token-accounting validator without writing caches."""
    sys.dont_write_bytecode = True
    path = Path(__file__).resolve().parent.parent / "analyze.py"
    spec = importlib.util.spec_from_file_location("d_trace_analyzer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def discover_raw_paths(inputs):
    found = set()
    for source in inputs:
        source = Path(source).resolve()
        if not source.exists():
            continue
        candidates = [source] if source.is_file() else source.rglob("raw.json")
        for path in candidates:
            if path.name == "raw.json" and not any(p.lower().startswith("warm") for p in path.parent.parts):
                found.add(path)
    return sorted(found)


def load_custom_runs(inputs):
    """Accept raw files, cell directories, block directories, or a block parent."""
    analyzer = load_analyzer()
    runs, phase_metadata, signatures = [], {}, {}
    for path in discover_raw_paths(inputs):
        protocol_path = analyzer.find_protocol(path, None)
        protocol = json.loads(protocol_path.read_text())
        phase_root = protocol_path.parent
        phase = phase_root.name
        if phase in phase_metadata and phase_metadata[phase]["protocol_path"] != str(protocol_path):
            raise ValueError("Ambiguous duplicate phase directory name: " + phase)
        if protocol["slo"] != BASE_SLO:
            raise ValueError("Unexpected source primary SLO in " + str(protocol_path))
        phase_metadata[phase] = {
            "protocol_path": str(protocol_path), "workload_sha256": protocol.get("workload_sha256"),
            "original_slo": protocol["slo"], "expected_requests_per_run": protocol.get("request_count"),
            "expected_policies": protocol.get("policies", []),
        }
        relative = str(path.parent.relative_to(phase_root))
        report = analyzer.summarize(path, protocol_path, relative)
        row = report["summary"]
        raw_bytes = path.read_bytes()
        raw = json.loads(raw_bytes)
        elapsed = raw["elapsed_s"]
        requests = [extract_request(q, elapsed) for q in raw["requests"]]
        signature = sorted((q["request_id"], q["arrival_s"], q["prompt_tokens"], q["max_tokens"]) for q in requests)
        if phase in signatures and signatures[phase] != signature:
            raise ValueError("Within-phase request identities or arrivals differ in " + str(path))
        signatures[phase] = signature
        evaluated = {}
        for name, slo in SCENARIOS:
            evaluated[name] = {
                "overall": evaluate(requests, slo, elapsed),
                "cohorts": {cohort: evaluate([q for q in requests if q["cohort"] == cohort], slo, elapsed) for cohort in COHORTS},
            }
        if evaluated["primary"]["overall"]["qualified_request_count"] != row["joint_slo_qualified_requests"]:
            raise ValueError("Primary SLO recomputation differs from analyzer: " + str(path))
        if not math.isclose(evaluated["primary"]["overall"]["goodput_requests_per_s"], row["joint_slo_requests_per_s"], rel_tol=1e-12):
            raise ValueError("Primary goodput recomputation differs from analyzer: " + str(path))
        runs.append({
            "run_id": phase + "/" + relative, "phase": phase, "run": relative,
            "policy": raw["policy"], "raw_path": str(path),
            "raw_sha256": hashlib.sha256(raw_bytes).hexdigest(), "elapsed_s": elapsed,
            "request_count": len(requests), "finished_count": sum(q["finished"] for q in requests),
            "output_tokens": sum(q["output_tokens"] for q in requests),
            "output_tokens_per_s": sum(q["output_tokens"] for q in requests) / elapsed,
            "original_summary_valid": row["valid"],
            "all_request_measurements_valid": all(q["valid_measurement"] for q in requests),
            "native_summary": row, "validation_errors": report["validation_errors"],
            "cohort_observed_arrival_ranges_s": {cohort: [
                min(q["arrival_s"] for q in requests if q["cohort"] == cohort),
                max(q["arrival_s"] for q in requests if q["cohort"] == cohort)]
                for cohort in COHORTS if any(q["cohort"] == cohort for q in requests)},
            "scenarios": evaluated, "_requests": requests,
        })
    return runs, phase_metadata


def summarize_policies(runs):
    grouped = collections.defaultdict(list)
    for run in runs:
        grouped[(run["phase"], run["policy"])].append(run)
    result = []
    for (phase, policy), members in grouped.items():
        scenarios = {}
        for scenario, _ in SCENARIOS:
            rows = [run["scenarios"][scenario]["overall"] for run in members]
            scenarios[scenario] = {
                "qualified_counts_by_run": [x["qualified_request_count"] for x in rows],
                "elapsed_s_by_run": [x["elapsed_s_denominator"] for x in rows],
                "goodput_requests_per_s_by_run": [x["goodput_requests_per_s"] for x in rows],
                "mean_of_run_goodput_requests_per_s": statistics.fmean(x["goodput_requests_per_s"] for x in rows),
                "aggregate_qualified_requests": sum(x["qualified_request_count"] for x in rows),
                "aggregate_elapsed_s": sum(x["elapsed_s_denominator"] for x in rows),
                "ratio_of_totals_goodput_requests_per_s": sum(x["qualified_request_count"] for x in rows) / sum(x["elapsed_s_denominator"] for x in rows),
                "cohort_qualified_counts_by_run": {name: [run["scenarios"][scenario]["cohorts"][name]["qualified_request_count"] for run in members] for name in COHORTS},
            }
        result.append({"phase": phase, "policy": policy, "run_count": len(members),
                       "run_ids": [run["run_id"] for run in members], "scenarios": scenarios})
    return result


def timing_damage(feedback, baseline):
    result = {}
    for metric in ("ttft_s", "completion_s"):
        values = [(qid, feedback[qid][metric] - baseline[qid][metric]) for qid in feedback
                  if feedback[qid][metric] is not None and baseline[qid][metric] is not None]
        result[metric] = {
            "comparable_count": len(values),
            "unobserved_count": len(feedback) - len(values),
            "worsened_count": sum(delta > 0 for _, delta in values),
            "improved_count": sum(delta < 0 for _, delta in values),
            "equal_count": sum(delta == 0 for _, delta in values),
            "mean_delta_s": statistics.fmean(delta for _, delta in values) if values else None,
            "p95_delta_s": percentile([delta for _, delta in values]),
            "worst_delta_s": max((delta for _, delta in values), default=None),
            "worsened_request_ids": sorted([qid for qid, delta in values if delta > 0], key=numeric_request_id),
        }
    return result


def compare_pair(feedback_run, baseline_run, ordinal):
    feedback = {q["request_id"]: q for q in feedback_run["_requests"]}
    baseline = {q["request_id"]: q for q in baseline_run["_requests"]}
    if feedback.keys() != baseline.keys():
        raise ValueError("Cannot request-match runs with different request IDs")
    scenarios = {}
    for name, _ in SCENARIOS:
        a = feedback_run["scenarios"][name]["overall"]
        b = baseline_run["scenarios"][name]["overall"]
        passed_a, passed_b = set(a["qualified_request_ids"]), set(b["qualified_request_ids"])
        added, lost = passed_a - passed_b, passed_b - passed_a
        scenarios[name] = {
            "feedback_qualified_count": len(passed_a), "baseline_qualified_count": len(passed_b),
            "added_qualified_count": len(added), "lost_qualified_count": len(lost),
            "both_qualified_count": len(passed_a & passed_b),
            "neither_qualified_count": len(feedback) - len(passed_a | passed_b),
            "added_request_ids": sorted(added, key=numeric_request_id),
            "lost_request_ids": sorted(lost, key=numeric_request_id),
            "feedback_elapsed_s": feedback_run["elapsed_s"],
            "baseline_elapsed_s": baseline_run["elapsed_s"],
            "feedback_goodput_requests_per_s": a["goodput_requests_per_s"],
            "baseline_goodput_requests_per_s": b["goodput_requests_per_s"],
            "goodput_delta_requests_per_s": a["goodput_requests_per_s"] - b["goodput_requests_per_s"],
            "goodput_change_percent": 100 * (a["goodput_requests_per_s"] / b["goodput_requests_per_s"] - 1) if b["goodput_requests_per_s"] else None,
            "cohorts": {cohort: {
                "added_qualified_count": sum(feedback[qid]["cohort"] == cohort for qid in added),
                "lost_qualified_count": sum(feedback[qid]["cohort"] == cohort for qid in lost),
            } for cohort in COHORTS},
        }
    same_phase = feedback_run["phase"] == baseline_run["phase"]
    return {
        "feedback_run": feedback_run["run_id"], "baseline_run": baseline_run["run_id"],
        "feedback_phase": feedback_run["phase"], "baseline_phase": baseline_run["phase"],
        "baseline_policy": baseline_run["policy"], "ordinal_within_policy": ordinal,
        "request_id_matched_count": len(feedback), "same_phase": same_phase,
        "strict_paired_execution": False,
        "pairing_note": "Same-phase separate executions, request-ID matched; no simultaneous execution or randomized paired-trial claim." if same_phase else "DIFFERENT PHASES: matching by request ID and ordinal is descriptive, NOT strict experimental pairing or a causal estimate.",
        "timing_damage_all_requests": timing_damage(feedback, baseline),
        "timing_damage_by_cohort": {cohort: timing_damage(
            {qid: q for qid, q in feedback.items() if q["cohort"] == cohort},
            {qid: q for qid, q in baseline.items() if q["cohort"] == cohort}) for cohort in COHORTS},
        "scenarios": scenarios,
    }


def request_matched_comparisons(runs, original=True):
    grouped = collections.defaultdict(list)
    for run in runs:
        grouped[(run["phase"], run["policy"])].append(run)
    comparisons = []
    feedback_phases = ("high-normal-r01", "extra-controls-high") if original else sorted({r["phase"] for r in runs if r["policy"] == "feedback"})
    for feedback_phase in feedback_phases:
        baseline_sources = (("extra-controls-high384", "fixed384"), ("extra-controls-high", "fixed304"), ("high-normal-r01", "decode")) if original else tuple((feedback_phase, policy) for policy in ("fixed384", "fixed304", "decode_v2", "decode") if (feedback_phase, policy) in grouped)
        for baseline_phase, baseline_policy in baseline_sources:
            feedback_runs = grouped[(feedback_phase, "feedback")]
            baseline_runs = grouped[(baseline_phase, baseline_policy)]
            if len(feedback_runs) != len(baseline_runs):
                raise ValueError("Unequal repeat counts require an explicit new matching design")
            pairs = [compare_pair(a, b, i + 1) for i, (a, b) in enumerate(zip(feedback_runs, baseline_runs))]
            scenario_summary = {}
            for name, _ in SCENARIOS:
                rows = [pair["scenarios"][name] for pair in pairs]
                ga = statistics.fmean(x["feedback_goodput_requests_per_s"] for x in rows)
                gb = statistics.fmean(x["baseline_goodput_requests_per_s"] for x in rows)
                scenario_summary[name] = {
                    "added_qualified_counts_by_pair": [x["added_qualified_count"] for x in rows],
                    "lost_qualified_counts_by_pair": [x["lost_qualified_count"] for x in rows],
                    "feedback_mean_goodput_requests_per_s": ga,
                    "baseline_mean_goodput_requests_per_s": gb,
                    "change_of_mean_goodput_percent": 100 * (ga / gb - 1) if gb else None,
                }
            comparisons.append({
                "feedback_phase": feedback_phase, "baseline_phase": baseline_phase,
                "baseline_policy": baseline_policy, "same_phase": feedback_phase == baseline_phase,
                "strict_paired_execution": False, "scenarios": scenario_summary, "pairs": pairs,
            })
    return comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path, help="Explicit raw.json files or directories; defaults to the three original high-load phases")
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--output", type=Path, help="Defaults to sensitivity.json for old data, sensitivity_custom.json for explicit inputs")
    args = parser.parse_args()
    args.output = args.output or Path(__file__).resolve().with_name("sensitivity_custom.json" if args.inputs else "sensitivity.json")
    runs, phases = load_custom_runs(args.inputs) if args.inputs else load_runs(args.data_root.resolve())
    if not runs:
        parser.error("No measured raw.json files found; no report written")
    policy_summaries = summarize_policies(runs)
    comparisons = request_matched_comparisons(runs, original=not args.inputs)
    for run in runs:
        del run["_requests"]
    result = {
        "schema_version": 1,
        "scope": "Offline sensitivity of explicitly selected raw files/directories; no GPU runs and no edits to source artifacts." if args.inputs else "Offline sensitivity of all formal episodes in the three original high-load phases; no GPU runs and no edits to old artifacts.",
        "scenario_selection": "All seven user-prespecified one-at-a-time thresholds are retained; no favorable threshold selection and no new SLO is promoted to the primary endpoint.",
        "scenarios": [{"name": name, "thresholds": slo} for name, slo in SCENARIOS],
        "definitions": {
            "ttft_s": "First observed output timestamp minus scheduled workload arrival timestamp.",
            "gap_s": "Maximum adjacent observed output-token timestamp difference within a request; excludes TTFT; not device-only latency or network ITL.",
            "completion_s": "Completion timestamp minus scheduled workload arrival timestamp; unfinished requests cannot qualify.",
            "joint_pass": "Finished, complete valid fixed-length output, and all three inclusive <= thresholds satisfied.",
            "goodput_denominator": "Entire raw.elapsed_s, including waiting and final drain; also used for each cohort, whose goodputs sum to the whole run.",
            "marginal_failures": "Single-constraint counts overlap; exclusive_failure_combinations partition all requests, including pass.",
            "policy_summary": "Arithmetic means of complete-run goodput. Ratio-of-summed-counts-to-summed-time is also reported separately.",
            "cohort_origin": "Original source request-ID membership: background at t=0 or t=3; long IDs 0..42, 43..85, and 86..127 define waves. Membership is preserved under confirmation arrival jitter; no generated content is used.",
            "damage": "Feedback timing strictly greater than the matched baseline by any positive amount; no practical-significance tolerance and no statistical-significance interpretation.",
            "matching": "Equal request IDs and equal ordinal within each phase/policy. Different phases are explicitly NOT strict paired experiments. Same-phase runs are still separate executions.",
        },
        "cohort_definitions": {
            "background_t0": {"arrival_s": [0, 0], "requests_per_run": 24},
            "background_t3": {"arrival_s": [3, 3], "requests_per_run": 8},
            "long_wave1": {"source_id_suffix": [0, 42], "original_arrival_s": [0.60, 1.02], "requests_per_run": 43},
            "long_wave2": {"source_id_suffix": [43, 85], "original_arrival_s": [2.50, 2.92], "requests_per_run": 43},
            "long_wave3": {"source_id_suffix": [86, 127], "original_arrival_s": [5.00, 5.41], "requests_per_run": 42},
        },
        "phases": phases,
        "validation": {
            "measured_episodes": len(runs), "recorded_requests": sum(x["request_count"] for x in runs),
            "finished_requests": sum(x["finished_count"] for x in runs),
            "output_tokens": sum(x["output_tokens"] for x in runs),
            "all_original_summaries_valid": all(x["original_summary_valid"] for x in runs),
            "all_recomputed_measurements_valid": all(x["all_request_measurements_valid"] for x in runs),
            "all_primary_counts_and_goodputs_match_reference": True,
            "primary_comparison_reference": "parent analyze.summarize" if args.inputs else "existing phase summaries",
            "identical_workload_hash_and_request_identities": True if not args.inputs else "Within each phase; distinct confirmation blocks may have different arrival hashes.",
        },
        "policy_summaries": policy_summaries, "runs": runs,
        "request_matched_comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print("Wrote %s: %d episodes, %d scenarios, %d request-matched repeat pairs" %
          (args.output, len(runs), len(SCENARIOS), sum(len(c["pairs"]) for c in comparisons)))
    print("Scenario order: " + ", ".join(name for name, _ in SCENARIOS))
    for policy in policy_summaries:
        print("%s/%s: %s" % (policy["phase"], policy["policy"], "; ".join(
            "%s %.4f (%s)" % (name, policy["scenarios"][name]["mean_of_run_goodput_requests_per_s"],
                                "/".join(str(x) for x in policy["scenarios"][name]["qualified_counts_by_run"]))
            for name, _ in SCENARIOS)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
