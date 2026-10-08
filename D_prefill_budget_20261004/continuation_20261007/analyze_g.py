#!/usr/bin/env python3
"""CPU-only adapter for the frozen three-block F/R/A/G screening experiment.

Run: python3 -B analyze_g.py [block directories or raw.json paths]
Writes only g_results.json (or --output). Missing blocks produce no promotion
decision. No new-arrival results were inspected to choose the frozen criteria.
"""

import argparse
import collections
import json
from pathlib import Path
import statistics
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / "confirmation_v2"
sys.path.insert(0, str(OLD))
import sensitivity
from paired_comparison import backlog_statistics
from ablation_elapsed_v3.analyze_ablation import pool_policy_runs, public_run, run_valid


POLICIES = {"F": "fixed384", "R": "elapsed_replay", "A": "feedback",
            "G": "feedback_floor_recover"}
ORDERS = {"block01": "FRAG", "block02": "GARF", "block03": "AFGR"}
REFERENCES = ("R", "A", "F")
CRITERIA = {
    "purpose": "Prespecified screening for whether G merits further study; not statistical proof or a main-method claim.",
    "minimum_primary_goodput_gain_percent": 3.0,
    "required_positive_blocks": 3,
    "maximum_output_throughput_loss_percent": 3.0,
    "maximum_ttft_p95_increase_percent": 5.0,
    "maximum_completion_flow_p95_increase_percent": 5.0,
    "maximum_request_max_gap_p95_increase_percent": 10.0,
    "all_requests_must_finish_with_valid_fixed_length_work": True,
    "aggregation": "Goodput and output throughput use sums of numerators divided by sums of full-run elapsed times. Tail guardrails compare means of the three per-run P95s, not pooled request/token percentiles.",
    "unit": "Three complete arrival blocks; requests/tokens are not independent repetitions.",
    "budget": "Exactly three blocks, four formal policies per block, frozen FRAG/GARF/AFGR order. No threshold changes or additional runs to reach significance.",
    "decision": "G/R, G/A and G/F are evaluated separately. Passing only G/A is a self-feedback repair, not independent contribution. Even passing all three only justifies budget-matched and strong-dynamic-baseline follow-up.",
}


def relative_percent(candidate, reference):
    if candidate is None or reference is None:
        return None
    if reference == 0:
        return 0.0 if candidate == 0 else None
    return 100.0 * (candidate / reference - 1.0)


def rename_pair_fields(value):
    """The reused helper's first input is G; remove its historical A naming."""
    if isinstance(value, dict):
        return {key.replace("feedback_", "G_").replace("baseline_", "reference_"):
                rename_pair_fields(item) for key, item in value.items()}
    if isinstance(value, list):
        return [rename_pair_fields(item) for item in value]
    return value


def compare_pools(candidate, reference):
    if candidate is None or reference is None:
        return None
    scenarios = {}
    for name, _ in sensitivity.SCENARIOS:
        a = candidate["scenarios"][name]["overall"]
        b = reference["scenarios"][name]["overall"]
        scenarios[name] = {
            "G_qualified_numerator": a["qualified_request_numerator"],
            "reference_qualified_numerator": b["qualified_request_numerator"],
            "G_elapsed_s_denominator": a["elapsed_s_denominator"],
            "reference_elapsed_s_denominator": b["elapsed_s_denominator"],
            "G_goodput_requests_per_s": a["goodput_requests_per_s"],
            "reference_goodput_requests_per_s": b["goodput_requests_per_s"],
            "G_relative_goodput_percent": relative_percent(a["goodput_requests_per_s"], b["goodput_requests_per_s"]),
        }
    metrics = {}
    for name, value in candidate["metrics"].items():
        a, b = value["mean_of_run_metrics"], reference["metrics"][name]["mean_of_run_metrics"]
        metrics[name] = {"G_mean_of_run_metrics": a, "reference_mean_of_run_metrics": b,
                         "G_minus_reference": a - b if a is not None and b is not None else None,
                         "G_relative_percent": relative_percent(a, b)}
    a = candidate["prefill_backlog"]["mean_actual_prefill_tokens"]
    b = reference["prefill_backlog"]["mean_actual_prefill_tokens"]
    return {
        "scenarios": scenarios, "full_service_metrics": metrics,
        "output_throughput_ratio_of_totals_change_percent": relative_percent(candidate["output_tokens_per_s"], reference["output_tokens_per_s"]),
        "backlog_actual_prefill": {"G_mean": a, "reference_mean": b,
                                    "G_relative_percent": relative_percent(a, b)},
    }


def screen(comparison, block_comparisons, ready):
    if not ready:
        return {"status": "pending_complete_three_block_screen", "passed": None}
    primary = comparison["scenarios"]["primary"]["G_relative_goodput_percent"]
    effects = [row["scenarios"]["primary"]["G_relative_goodput_percent"] for row in block_comparisons]
    costs = {
        "output_throughput_percent": comparison["output_throughput_ratio_of_totals_change_percent"],
        **{metric: comparison["full_service_metrics"][metric]["G_relative_percent"]
           for metric in ("ttft_p95_s", "completion_flow_p95_s", "request_max_gap_p95_s")},
    }
    guards = {
        "pooled_primary_gain_at_least_3_percent": primary is not None and primary >= CRITERIA["minimum_primary_goodput_gain_percent"],
        "three_of_three_primary_directions_positive": len(effects) == CRITERIA["required_positive_blocks"] and all(v is not None and v > 0 for v in effects),
        "output_throughput_loss_at_most_3_percent": costs["output_throughput_percent"] is not None and costs["output_throughput_percent"] >= -CRITERIA["maximum_output_throughput_loss_percent"],
        **{metric + "_within_limit": costs[metric] is not None and costs[metric] <= limit
           for metric, limit in (("ttft_p95_s", CRITERIA["maximum_ttft_p95_increase_percent"]),
                                 ("completion_flow_p95_s", CRITERIA["maximum_completion_flow_p95_increase_percent"]),
                                 ("request_max_gap_p95_s", CRITERIA["maximum_request_max_gap_p95_increase_percent"]))},
    }
    return {
        "status": "passes_screen_only" if all(guards.values()) else "does_not_pass_screen",
        "passed": all(guards.values()), "guard_checks": guards,
        "primary_block_effects_percent": effects, "aggregate_cost_changes_percent": costs,
        "primary_block_effect_range_percent": [min(effects), max(effects)] if all(v is not None for v in effects) else None,
        "primary_block_effect_median_percent": statistics.median(effects) if all(v is not None for v in effects) else None,
        "uncertainty": "Descriptive three-block screening; no significance, equivalence, confidence-interval or generalization claim.",
    }


def historical_workload_hashes():
    hashes = set()
    for name in ORDERS:
        path = OLD / name / "protocol.json"
        if path.exists():
            value = json.loads(path.read_text()).get("workload_sha256")
            if value:
                hashes.add(value)
    return hashes


def build_result(runs, metadata):
    grouped = collections.defaultdict(list)
    for run in runs:
        run["_backlog"] = backlog_statistics(run["raw_path"])
        grouped[run["phase"]].append(run)
    old_hashes = historical_workload_hashes()
    hashes = [metadata[name].get("workload_sha256") for name in grouped]
    blocks = []
    for name, members in sorted(grouped.items()):
        def position(run):
            prefix = Path(run["run"]).name.split("_", 1)[0]
            return int(prefix) if prefix.isdigit() else -1
        members = sorted(members, key=position)
        labels = {value: key for key, value in POLICIES.items()}
        order = "".join(labels.get(run["policy"], "?") for run in members)
        errors = []
        if name not in ORDERS or order != ORDERS.get(name) or [position(run) for run in members] != [0, 1, 2, 3]:
            errors.append("Formal positions/policy order differ from frozen FRAG/GARF/AFGR.")
        if metadata[name].get("expected_policies") != [POLICIES[label] for label in ORDERS.get(name, "")]:
            errors.append("Protocol policy order differs from frozen order.")
        digest = metadata[name].get("workload_sha256")
        if not digest or digest in old_hashes or hashes.count(digest) != 1:
            errors.append("Require a distinct new workload hash, different from the three old arrival blocks.")
        pooled = {label: pool_policy_runs([run for run in members if run["policy"] == policy])
                  for label, policy in POLICIES.items()}
        pairs, comparisons = {}, {}
        for label in REFERENCES:
            candidates = [run for run in members if run["policy"] == POLICIES["G"]]
            references = [run for run in members if run["policy"] == POLICIES[label]]
            if len(candidates) == len(references) == 1:
                pair = rename_pair_fields(sensitivity.compare_pair(candidates[0], references[0], 1))
                pair["reference_label"] = label
                pair["pairing_note"] = "Same frozen arrival block and request IDs; separate executions in the prespecified interleaved order, not necessarily adjacent. No cross-block pairing."
                pairs["G_vs_" + label] = pair
                comparisons["G_vs_" + label] = compare_pools(pooled["G"], pooled[label])
        complete = not errors and len(members) == 4 and all(run_valid(run) for run in members)
        blocks.append({"block": name, "expected_order": ORDERS.get(name), "observed_order": order,
                       "complete": complete, "errors": errors, "protocol": metadata[name],
                       "runs": [public_run(run) for run in members], "request_matched_comparisons": pairs,
                       "policy_pooled_counts_and_time": pooled, "comparisons": comparisons})
    complete_names = {block["block"] for block in blocks if block["complete"]}
    ready = complete_names == set(ORDERS) and len(blocks) == 3
    valid_runs = [run for run in runs if run["phase"] in complete_names]
    aggregate = None
    screens = {"G_vs_" + label: screen(None, [], False) for label in REFERENCES}
    if valid_runs:
        pooled = {label: pool_policy_runs([run for run in valid_runs if run["policy"] == policy])
                  for label, policy in POLICIES.items()}
        comparisons = {"G_vs_" + label: compare_pools(pooled["G"], pooled[label]) for label in REFERENCES}
        screens = {key: screen(value, [block["comparisons"][key] for block in blocks if block["complete"]], ready)
                   for key, value in comparisons.items()}
        aggregate = {"block_ids": sorted(complete_names), "run_count": len(valid_runs),
                     "request_exposures": sum(run["request_count"] for run in valid_runs),
                     "finished_request_exposures": sum(run["finished_count"] for run in valid_runs),
                     "output_tokens": sum(run["output_tokens"] for run in valid_runs),
                     "policy_pooled_counts_and_time": pooled, "comparisons": comparisons}
    return {
        "schema_version": 1, "policies": POLICIES, "orders": ORDERS,
        "primary_slo": sensitivity.BASE_SLO,
        "scenarios": [{"name": name, "thresholds": slo} for name, slo in sensitivity.SCENARIOS],
        "frozen_screening_criteria": CRITERIA, "observed_formal_runs": len(runs),
        "complete_block_count": len(complete_names), "all_three_blocks_complete": ready,
        "missing_blocks": sorted(set(ORDERS) - set(grouped)), "blocks": blocks,
        "aggregate_complete_blocks": aggregate, "screening_by_comparison": screens,
        "all_three_comparisons_pass_screen": all(row["passed"] for row in screens.values()) if ready else None,
        "definitions": {
            "validation": "Reuse parent analyze.summarize through sensitivity.load_custom_runs: all requests, fixed output lengths, token accounting, budget, timestamps and protocol count. Same-block IDs/arrivals/work requirements must match.",
            "backlog": "Reuse paired_comparison.backlog_statistics: positive pre-step backlog, including P=0; reconstruct from add_s and cumulative scheduled prompt work.",
            "observation_window": "Unchanged arrival-relative request metrics and entire elapsed_s including final drain; all requests remain in fractions and throughput denominators. Unfinished requests never qualify and prevent screening completion.",
            "sensitivity": "Exactly seven existing one-at-a-time scenarios; primary 4s/100ms/20s never replaced by a favorable alternative.",
            "scope": "Only this directory's exact block01..03 roots. warm* paths excluded by the reused loader; failed/retry archive siblings and historical phases are not pooled.",
            "damage": "Any strictly positive G-minus-reference request timing delta, descriptive only. Cohorts retain source request-ID wave membership.",
            "freshness": "Protocol workload hashes must differ across the three blocks and from the old confirmation arrival hashes; complete within-block request signatures are checked by the existing loader.",
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "g_results.json")
    args = parser.parse_args()
    allowed_roots = [(ROOT / name).resolve() for name in ORDERS]
    paths = sensitivity.discover_raw_paths(args.inputs or allowed_roots)
    paths = [path for path in paths if any(root in path.resolve().parents for root in allowed_roots)]
    runs, metadata = sensitivity.load_custom_runs(paths)
    result = build_result(runs, metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print("Wrote %s: %d formal runs, %d complete blocks" %
          (args.output, result["observed_formal_runs"], result["complete_block_count"]))
    if result["complete_block_count"] == 0:
        print("0 complete blocks; no measured aggregate or screening conclusion.")
    for name, row in result["screening_by_comparison"].items():
        print(name, row["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
