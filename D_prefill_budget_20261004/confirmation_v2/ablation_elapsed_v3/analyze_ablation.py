#!/usr/bin/env python3
"""Analyze the frozen feedback versus elapsed_replay ablation on CPU.

Usage: python3 -B analyze_ablation.py [block directories or raw.json paths]
Only ablation_results.json is written. With no inputs, discover block01..03
beside this script. Missing/invalid blocks stay visible and are not pooled.
"""

import argparse
import collections
import json
from pathlib import Path
import statistics
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import sensitivity
from paired_comparison import METRICS, backlog_statistics, cohort_timing


POLICIES = {"A": "feedback", "R": "elapsed_replay"}
EXPECTED_ORDERS = {"block01": "ARRA", "block02": "RAAR", "block03": "ARRA"}


def numeric_cell_order(run):
    prefix = Path(run["run"]).name.split("_", 1)[0]
    if not prefix.isdigit():
        raise ValueError("Formal cell lacks a numeric order prefix: " + run["run_id"])
    return int(prefix)


def pool_evaluations(evaluations):
    if not evaluations:
        return None
    elapsed = sum(row["elapsed_s_denominator"] for row in evaluations)
    qualified = sum(row["qualified_request_count"] for row in evaluations)
    tokens = sum(row["qualified_output_tokens"] for row in evaluations)
    exclusive = collections.Counter()
    for row in evaluations:
        exclusive.update(row["exclusive_failure_combinations"])
    total = sum(row["request_count"] for row in evaluations)
    return {
        "request_count": total,
        "finished_count": sum(row["finished_count"] for row in evaluations),
        "qualified_request_numerator": qualified, "elapsed_s_denominator": elapsed,
        "qualified_request_fraction": qualified / total if total else 0.0,
        "goodput_requests_per_s": qualified / elapsed,
        "qualified_output_tokens": tokens, "goodput_output_tokens_per_s": tokens / elapsed,
        "single_constraint_fail_counts_nonexclusive": {
            condition: sum(row["single_constraint_fail_counts_nonexclusive"][condition] for row in evaluations)
            for condition in sensitivity.CONDITIONS
        },
        "exclusive_failure_combinations": dict(sorted(exclusive.items())),
    }


def pool_backlogs(runs):
    backlogs = [run["_backlog"] for run in runs]
    steps = sum(row["steps"] for row in backlogs)
    actual = sum(row["sum_actual_prefill_tokens"] for row in backlogs)
    budgets = sum(row["sum_budget_tokens"] for row in backlogs)
    caps = collections.Counter()
    for row in backlogs:
        caps.update(row["cap_frequencies"])
    return {
        "selection": "Recorded pre-step backlog > 0, including zero-prefill steps; pooled step-weighted means.",
        "steps": steps, "zero_prefill_steps": sum(row["zero_prefill_steps"] for row in backlogs),
        "sum_actual_prefill_tokens": actual,
        "mean_actual_prefill_tokens": actual / steps if steps else None,
        "mean_budget_tokens": budgets / steps if steps else None,
        "cap_frequencies": dict(sorted(caps.items(), key=lambda pair: int(pair[0]))),
        "all_reconstructions_valid": all(row["reconstruction"]["all_fields_match"] for row in backlogs),
    }


def pool_policy_runs(runs):
    if not runs:
        return None
    elapsed = sum(run["elapsed_s"] for run in runs)
    output = sum(run["output_tokens"] for run in runs)
    scenarios = {}
    for name, _ in sensitivity.SCENARIOS:
        scenarios[name] = {
            "overall": pool_evaluations([run["scenarios"][name]["overall"] for run in runs]),
            "cohorts": {cohort: pool_evaluations([run["scenarios"][name]["cohorts"][cohort] for run in runs])
                        for cohort in sensitivity.COHORTS},
        }
    return {
        "run_count": len(runs), "run_ids": [run["run_id"] for run in runs],
        "request_exposure_note": "Each policy execution counts all its requests; repeated source request IDs are separate measured exposures.",
        "output_tokens": output, "elapsed_s": elapsed, "output_tokens_per_s": output / elapsed,
        "metrics": {metric: {
            "mean_of_run_metrics": statistics.fmean(run["native_summary"][metric] for run in runs)
            if all(run["native_summary"].get(metric) is not None for run in runs) else None,
            "run_values": [run["native_summary"].get(metric) for run in runs],
        } for metric in METRICS},
        "prefill_backlog": pool_backlogs(runs), "scenarios": scenarios,
    }


def pooled_effects(a, r):
    if a is None or r is None:
        return None
    effects = {}
    for name, _ in sensitivity.SCENARIOS:
        av = a["scenarios"][name]["overall"]
        rv = r["scenarios"][name]["overall"]
        ag, rg = av["goodput_requests_per_s"], rv["goodput_requests_per_s"]
        effects[name] = {
            "A_qualified_numerator": av["qualified_request_numerator"],
            "R_qualified_numerator": rv["qualified_request_numerator"],
            "A_elapsed_s_denominator": av["elapsed_s_denominator"],
            "R_elapsed_s_denominator": rv["elapsed_s_denominator"],
            "A_goodput_requests_per_s": ag, "R_goodput_requests_per_s": rg,
            "A_minus_R_goodput_requests_per_s": ag - rg,
            "A_relative_to_R_goodput_percent": 100 * (ag / rg - 1) if rg else None,
        }
    am, rm = a["prefill_backlog"]["mean_actual_prefill_tokens"], r["prefill_backlog"]["mean_actual_prefill_tokens"]
    return {
        "scenarios": effects,
        "A_relative_to_R_output_throughput_percent": 100 * (a["output_tokens_per_s"] / r["output_tokens_per_s"] - 1),
        "A_minus_R_mean_actual_prefill_per_backlog_step": am - rm if am is not None and rm is not None else None,
        "A_relative_to_R_mean_actual_prefill_percent": 100 * (am / rm - 1) if am is not None and rm else None,
        "full_service_mean_of_run_metric_deltas": {
            metric: a["metrics"][metric]["mean_of_run_metrics"] - r["metrics"][metric]["mean_of_run_metrics"]
            if a["metrics"][metric]["mean_of_run_metrics"] is not None and r["metrics"][metric]["mean_of_run_metrics"] is not None else None
            for metric in METRICS
        },
    }


def run_valid(run):
    return (run["original_summary_valid"] and run["all_request_measurements_valid"]
            and run["finished_count"] == run["request_count"]
            and run["_backlog"]["reconstruction"]["all_fields_match"])


def public_run(run):
    return {
        "run_id": run["run_id"], "policy": run["policy"],
        "raw_path": run["raw_path"], "raw_sha256": run["raw_sha256"],
        "all_valid_and_finished": run_valid(run),
        "summary": run["native_summary"], "validation_errors": run["validation_errors"],
        "prefill_backlog": run["_backlog"],
        "cohort_timing": cohort_timing(run["_requests"]),
        "scenarios": run["scenarios"],
    }


def make_pair(first, second, pair_number):
    if {first["policy"], second["policy"]} != set(POLICIES.values()):
        return None
    a = first if first["policy"] == POLICIES["A"] else second
    r = second if first["policy"] == POLICIES["A"] else first
    compared = sensitivity.compare_pair(a, r, pair_number)
    return {
        "pair_number": pair_number,
        "execution_order": "AR" if first is a else "RA",
        "first_run": first["run_id"], "second_run": second["run_id"],
        "all_valid_and_finished": run_valid(a) and run_valid(r),
        "request_id_matched_count": compared["request_id_matched_count"],
        "pairing": "Adjacent opposite-policy executions of the same frozen arrival block; the second pair reverses order.",
        "scenarios": compared["scenarios"],
        "timing_damage_all_requests": compared["timing_damage_all_requests"],
        "timing_damage_by_cohort": compared["timing_damage_by_cohort"],
        "A_minus_R_full_service_metric_deltas": {
            metric: a["native_summary"][metric] - r["native_summary"][metric]
            if a["native_summary"].get(metric) is not None and r["native_summary"].get(metric) is not None else None
            for metric in METRICS
        },
    }


def summarize_block(name, runs, metadata):
    runs = sorted(runs, key=numeric_cell_order)
    orders = [numeric_cell_order(run) for run in runs]
    by_position = dict(zip(orders, runs))
    labels = {policy: label for label, policy in POLICIES.items()}
    observed = "".join(labels.get(run["policy"], "?") for run in runs)
    expected = EXPECTED_ORDERS.get(name)
    errors = []
    if expected is None:
        errors.append("Block is not one of the three frozen block names")
    if orders != [0, 1, 2, 3]:
        errors.append("Formal positions must be exactly 0,1,2,3")
    if observed != expected:
        errors.append("Formal order differs from the frozen order")
    declared = metadata.get("expected_policies", [])
    if expected and declared != [POLICIES[label] for label in expected]:
        errors.append("Protocol policies differ from the frozen order")
    pairs = []
    for number, (left, right) in enumerate(((0, 1), (2, 3)), 1):
        if left in by_position and right in by_position:
            pair = make_pair(by_position[left], by_position[right], number)
            if pair is None:
                errors.append("Adjacent pair %d does not contain one A and one R" % number)
            else:
                pairs.append(pair)
    if len(pairs) == 2 and pairs[0]["execution_order"] == pairs[1]["execution_order"]:
        errors.append("Adjacent pairs do not reverse execution order")
    policies = {label: pool_policy_runs([run for run in runs if run["policy"] == policy])
                for label, policy in POLICIES.items()}
    complete = not errors and len(pairs) == 2 and all(run_valid(run) for run in runs)
    return {
        "block": name, "protocol": metadata, "expected_order": expected,
        "observed_order": observed, "complete": complete,
        "order_or_completeness_errors": errors,
        "all_present_runs_valid_and_finished": all(run_valid(run) for run in runs),
        "runs": [public_run(run) for run in runs], "adjacent_pairs": pairs,
        "policy_pooled_counts_and_time": policies,
        "A_vs_R_pooled_effects": pooled_effects(policies["A"], policies["R"]),
        "pooling_note": "Within each policy, add qualified counts and whole-run elapsed times over its two exposures. Partial blocks are descriptive only and never enter the complete-block aggregate.",
    }


def build_result(runs, metadata):
    grouped = collections.defaultdict(list)
    for run in runs:
        if "_backlog" not in run:
            run["_backlog"] = backlog_statistics(run["raw_path"])
        grouped[run["phase"]].append(run)
    blocks = [summarize_block(name, members, metadata[name]) for name, members in sorted(grouped.items())]
    complete_names = {block["block"] for block in blocks if block["complete"]}
    complete_runs = [run for run in runs if run["phase"] in complete_names]
    aggregate = None
    if complete_runs:
        policies = {label: pool_policy_runs([run for run in complete_runs if run["policy"] == policy])
                    for label, policy in POLICIES.items()}
        aggregate = {
            "block_count": len(complete_names), "block_ids": sorted(complete_names),
            "run_count": len(complete_runs),
            "request_exposures": sum(run["request_count"] for run in complete_runs),
            "finished_request_exposures": sum(run["finished_count"] for run in complete_runs),
            "output_tokens": sum(run["output_tokens"] for run in complete_runs),
            "policy_pooled_counts_and_time": policies,
            "A_vs_R_pooled_effects": pooled_effects(policies["A"], policies["R"]),
            "block_primary_effects": [{"block": block["block"], **block["A_vs_R_pooled_effects"]["scenarios"]["primary"]}
                                      for block in blocks if block["complete"]],
        }
    return {
        "schema_version": 1, "policies": POLICIES,
        "primary_slo": sensitivity.BASE_SLO,
        "scenarios": [{"name": name, "thresholds": slo} for name, slo in sensitivity.SCENARIOS],
        "expected_orders": EXPECTED_ORDERS,
        "observed_formal_runs": len(runs), "complete_block_count": len(complete_names),
        "all_three_blocks_complete": complete_names == set(EXPECTED_ORDERS),
        "not_yet_observed_blocks": sorted(set(EXPECTED_ORDERS) - set(grouped)),
        "incomplete_or_invalid_blocks": [block["block"] for block in blocks if not block["complete"]],
        "definitions": {
            "validation": "sensitivity.load_custom_runs calls the parent analyze.summarize for all requests, token accounting, budget, timestamps, and protocol count; existing backlog_statistics checks add_s and cumulative scheduled-prompt reconstruction.",
            "warmup": "All raw files under any warm* parent path segment are excluded.",
            "primary_denominator": "Full elapsed_s including waiting and final drain; per-block and overall policy goodputs divide summed qualified counts by summed elapsed times.",
            "metrics": "TTFT, completion flow and per-request maximum observed gap retain each run's values. Summarized P95 values are means of per-run P95s, not pooled token gaps.",
            "joint_slo": "Finished valid fixed-length outputs satisfying all three inclusive <= thresholds; unfinished requests never qualify. All seven frozen sensitivity scenarios are retained.",
            "backlog": "Positive recorded pre-step backlog, including actual P=0 steps. Actual prompt work and budget are pooled over those steps, with cap frequencies and reconstruction status.",
            "pairing": "Pairs are formal positions (0,1) and (2,3), matched by request ID within the same arrival block, with reversed exposure order. No cross-block request pairing.",
            "damage": "Any strictly positive A-minus-R TTFT or completion-flow delta; descriptive, not a statistical significance test.",
            "failure_counts": "Marginal single-constraint counts overlap; exclusive combinations partition all request exposures. Cohort goodput uses the same full-run denominator.",
            "cohorts": "Original source request-ID wave membership remains unchanged by arrival permutations.",
        },
        "limitations": [
            "This tests feedback against one frozen elapsed-free replay schedule. A winning result cannot establish that elapsed information is universally necessary or that every elapsed-free controller loses.",
            "The replay switches were frozen from OLD development data and retain the no-decode branch; executed state trajectories and actual mean prompt work may still differ and must be reported.",
            "The three existing arrival realizations were already used in the confirmation campaign; this ablation is not a new untouched workload holdout.",
            "ABBA/BAAB order supplies adjacent reversed comparisons, not a guarantee against GPU/host drift, dependence, or sufficient statistical power.",
            "One model/GPU and fixed output lengths bound generalization. Host-observed output gaps do not isolate GPU latency or include network delivery; output contents do not establish semantic equivalence.",
        ],
        "blocks": blocks, "aggregate_complete_blocks": aggregate,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().with_name("ablation_results.json"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    inputs = args.inputs or [root / name for name in EXPECTED_ORDERS]
    runs, metadata = sensitivity.load_custom_runs(inputs)
    result = build_result(runs, metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print("Wrote %s: %d formal runs, %d complete blocks" %
          (args.output, result["observed_formal_runs"], result["complete_block_count"]))
    if result["aggregate_complete_blocks"] is None:
        print("0 complete blocks; no measured aggregate available.")
    for block in result["blocks"]:
        print(block["block"], "complete=" + str(block["complete"]), "order=" + block["observed_order"],
              {label: (data["scenarios"]["primary"]["overall"]["qualified_request_numerator"],
                       data["scenarios"]["primary"]["overall"]["elapsed_s_denominator"],
                       data["scenarios"]["primary"]["overall"]["goodput_requests_per_s"])
               for label, data in block["policy_pooled_counts_and_time"].items() if data is not None})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
