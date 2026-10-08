#!/usr/bin/env python3
"""Summarize F/A/M/N confirmation blocks with the frozen primary SLO.

Usage: python3 -B paired_comparison.py [block directories or raw.json paths]
With no inputs, discovers block?? next to this script. Old campaign summaries
are never modified. Partial blocks remain visible but are not aggregated.
"""

import argparse
import collections
import json
from pathlib import Path
import statistics
import sys

sys.dont_write_bytecode = True
import sensitivity


POLICIES = {"F": "fixed384", "A": "feedback", "M": "fixed304", "N": "decode_v2"}
METRICS = (
    "output_tokens_per_s", "joint_slo_requests_per_s", "joint_slo_output_tokens_per_s",
    "ttft_mean_s", "ttft_p95_s", "ttft_max_s", "completion_flow_mean_s",
    "completion_flow_p95_s", "completion_flow_max_s", "completion_makespan_s",
    "request_max_gap_mean_s", "request_max_gap_p95_s", "request_max_gap_max_s",
    "arrival_to_first_schedule_mean_s", "arrival_to_first_schedule_p95_s",
)


def backlog_statistics(raw_path):
    """Check recorded pre-step backlog against submitted, unserved prompt work."""
    raw = json.loads(Path(raw_path).read_text())
    requests = {q["request_id"]: q for q in raw["requests"]}
    computed = collections.Counter()
    caps = collections.Counter()
    errors = []
    error_counts = collections.Counter()
    selected_steps = zero_steps = sum_actual = sum_budget = 0
    reconstructed_steps = missing_field_steps = 0

    def mismatch(kind, index, detail):
        error_counts[kind] += 1
        if len(errors) < 12:
            errors.append({"kind": kind, "step_index": index, **detail})

    for index, step in enumerate(raw["steps"]):
        start = step["start_s"]
        remaining = {
            request_id: request["prompt_tokens"] - computed[request_id]
            for request_id, request in requests.items()
            if sensitivity.finite(request.get("add_s")) and request["add_s"] <= start
            and computed[request_id] < request["prompt_tokens"]
        }
        rebuilt_tokens = sum(remaining.values())
        rebuilt_count = len(remaining)
        reconstructed_steps += int(rebuilt_tokens > 0)
        tokens = step.get("prefill_backlog_tokens_before")
        count = step.get("prefill_backlog_requests_before")
        tokens_valid = isinstance(tokens, int) and not isinstance(tokens, bool) and tokens >= 0
        count_valid = isinstance(count, int) and not isinstance(count, bool) and count >= 0
        if not tokens_valid or not count_valid:
            missing_field_steps += 1
            mismatch("missing_or_invalid_backlog_field", index, {
                "recorded_tokens": tokens, "recorded_requests": count,
            })
        if tokens_valid and tokens != rebuilt_tokens:
            mismatch("backlog_token_total", index, {
                "recorded": tokens, "reconstructed": rebuilt_tokens,
                "reconstructed_remaining_tokens_by_request": remaining,
            })
        if count_valid and count != rebuilt_count:
            mismatch("backlog_request_count", index, {
                "recorded": count, "reconstructed": rebuilt_count,
                "reconstructed_request_ids": sorted(remaining, key=sensitivity.numeric_request_id),
            })
        # The selection uses the recorded pre-step backlog, never P > 0.
        if tokens_valid and tokens > 0:
            selected_steps += 1
            actual = step["prefill_tokens"]
            zero_steps += int(actual == 0)
            sum_actual += actual
            sum_budget += step["budget"]
            caps[str(step["budget"])] += 1
        for entry in step.get("requests", []):
            request_id, actual = entry["request_id"], entry.get("prefill_tokens", 0)
            if actual > 0 and request_id not in remaining:
                mismatch("scheduled_prompt_outside_reconstructed_backlog", index, {
                    "request_id": request_id, "actual_prefill_tokens": actual,
                    "add_s": requests.get(request_id, {}).get("add_s"), "step_start_s": start,
                })
            elif actual > remaining.get(request_id, 0):
                mismatch("scheduled_prompt_exceeds_remaining", index, {
                    "request_id": request_id, "actual_prefill_tokens": actual,
                    "remaining_tokens_before": remaining.get(request_id, 0),
                })
            computed[request_id] += actual
        if step.get("preempted"):
            mismatch("preemption_invalidates_simple_cumulative_reconstruction", index, {
                "preempted": step["preempted"],
            })
    return {
        "selection": "Recorded prefill_backlog_tokens_before > 0, including actual prefill_tokens == 0.",
        "steps": selected_steps, "zero_prefill_steps": zero_steps,
        "sum_actual_prefill_tokens": sum_actual,
        "mean_actual_prefill_tokens": sum_actual / selected_steps if selected_steps else None,
        "sum_budget_tokens": sum_budget,
        "mean_budget_tokens": sum_budget / selected_steps if selected_steps else None,
        "cap_frequencies": dict(sorted(caps.items(), key=lambda pair: int(pair[0]))),
        "reconstruction": {
            "definition": "Before each step, include requests with add_s <= step.start_s and prompt_tokens minus cumulative previously scheduled per-request prefill_tokens > 0; reconstruct the request set, its cardinality, and sum of remaining tokens.",
            "all_fields_match": not error_counts,
            "total_steps_checked": len(raw["steps"]),
            "reconstructed_positive_backlog_steps": reconstructed_steps,
            "steps_with_missing_or_invalid_fields": missing_field_steps,
            "mismatch_counts": dict(error_counts), "first_mismatches": errors,
            "set_observability": "Raw records backlog request count and remaining-token sum, not a backlog-ID list. Both aggregates are checked against the reconstructed set; every scheduled prefill ID must belong to that set.",
        },
    }


def backlog_mean_comparison(feedback, matched_fixed):
    a = feedback["mean_actual_prefill_tokens"]
    m = matched_fixed["mean_actual_prefill_tokens"]
    return {
        "A_mean_actual_prefill_tokens": a, "M_mean_actual_prefill_tokens": m,
        "A_minus_M_actual_prefill_tokens_per_backlog_step": a - m if a is not None and m is not None else None,
        "A_relative_to_M_percent": 100 * (a / m - 1) if a is not None and m else None,
    }


def cohort_timing(requests):
    result = {}
    for cohort in sensitivity.COHORTS:
        members = [q for q in requests if q["cohort"] == cohort]
        result[cohort] = {"request_count": len(members)}
        for field in ("ttft_s", "completion_s", "gap_s"):
            values = [q[field] for q in members if q[field] is not None]
            complete = len(values) == len(members)
            result[cohort][field] = {
                "observed_count": len(values), "missing_count": len(members) - len(values),
                "mean": statistics.fmean(values) if values and complete else None,
                "p95": sensitivity.percentile(values) if complete else None,
                "max": max(values) if values and complete else None,
            }
    return result


def block_report(name, runs, metadata):
    by_policy = {}
    for run in runs:
        if run["policy"] in by_policy:
            raise ValueError("Expected one execution per policy per block: " + name)
        by_policy[run["policy"]] = run
    missing = [policy for policy in POLICIES.values() if policy not in by_policy]
    unknown = sorted(set(by_policy) - set(POLICIES.values()))
    valid = all(run["original_summary_valid"] and run["all_request_measurements_valid"]
                and run["finished_count"] == run["request_count"] for run in runs)
    backlogs = {run["policy"]: backlog_statistics(run["raw_path"]) for run in runs}
    backlog_valid = all(stats["reconstruction"]["all_fields_match"] for stats in backlogs.values())
    policies = {}
    for letter, policy in POLICIES.items():
        if policy not in by_policy:
            continue
        run = by_policy[policy]
        policies[letter] = {
            "policy": policy, "run_id": run["run_id"], "raw_path": run["raw_path"],
            "raw_sha256": run["raw_sha256"], "summary": run["native_summary"],
            "validation_errors": run["validation_errors"],
            "primary": run["scenarios"]["primary"],
            "cohort_timing": cohort_timing(run["_requests"]),
            "cohort_observed_arrival_ranges_s": run["cohort_observed_arrival_ranges_s"],
            "prefill_backlog": backlogs[policy],
        }
    comparisons = {}
    if "feedback" in by_policy:
        for letter in ("F", "M", "N"):
            baseline = POLICIES[letter]
            if baseline not in by_policy:
                continue
            paired = sensitivity.compare_pair(by_policy["feedback"], by_policy[baseline], 1)
            comparisons["A_vs_" + letter] = {
                "feedback_run": paired["feedback_run"], "baseline_run": paired["baseline_run"],
                "request_id_matched_count": paired["request_id_matched_count"],
                "pairing": "Same prespecified arrival block and matched request IDs; separate policy executions in the frozen block order.",
                "primary": paired["scenarios"]["primary"],
                "timing_damage_all_requests": paired["timing_damage_all_requests"],
                "timing_damage_by_cohort": paired["timing_damage_by_cohort"],
                "full_service_metric_deltas": {
                    metric: by_policy["feedback"]["native_summary"][metric] - by_policy[baseline]["native_summary"][metric]
                    if by_policy["feedback"]["native_summary"].get(metric) is not None
                    and by_policy[baseline]["native_summary"].get(metric) is not None else None
                    for metric in METRICS
                },
            }
            if letter == "M":
                comparisons["A_vs_M"]["prefill_backlog_mean_comparison"] = backlog_mean_comparison(backlogs["feedback"], backlogs["fixed304"])
    return {
        "block": name, "protocol": metadata,
        "complete_four_policy_block": not missing and not unknown and valid and backlog_valid,
        "all_present_runs_valid_and_finished": valid,
        "all_backlog_fields_match_reconstruction": backlog_valid,
        "missing_policies": missing, "unexpected_policies": unknown,
        "policies": policies, "comparisons": comparisons,
    }


def aggregate_complete_blocks(blocks):
    complete = [block for block in blocks if block["complete_four_policy_block"]]
    result = {
        "block_count": len(complete), "block_ids": [block["block"] for block in complete],
        "excluded_partial_or_invalid_blocks": [block["block"] for block in blocks if not block["complete_four_policy_block"]],
        "policies": {}, "comparisons": {},
    }
    if not complete:
        return result
    for letter, policy in POLICIES.items():
        rows = [block["policies"][letter]["summary"] for block in complete]
        elapsed = sum(row["elapsed_s"] for row in rows)
        qualified = sum(row["joint_slo_qualified_requests"] for row in rows)
        qualified_tokens = sum(row["joint_slo_qualified_output_tokens"] for row in rows)
        result["policies"][letter] = {
            "policy": policy, "block_count": len(rows),
            "request_count": sum(row["request_count"] for row in rows),
            "finished_count": sum(row["finished_count"] for row in rows),
            "qualified_request_numerator": qualified, "elapsed_s_denominator": elapsed,
            "ratio_of_totals_goodput_requests_per_s": qualified / elapsed,
            "ratio_of_totals_goodput_output_tokens_per_s": qualified_tokens / elapsed,
            "metrics": {metric: {
                "mean_of_block_metrics": statistics.fmean(row[metric] for row in rows),
                "block_values": [row[metric] for row in rows],
            } for metric in METRICS},
        }
        backlogs = [block["policies"][letter]["prefill_backlog"] for block in complete]
        steps = sum(x["steps"] for x in backlogs)
        actual = sum(x["sum_actual_prefill_tokens"] for x in backlogs)
        budgets = sum(x["sum_budget_tokens"] for x in backlogs)
        caps = collections.Counter()
        for stats in backlogs:
            caps.update(stats["cap_frequencies"])
        result["policies"][letter]["prefill_backlog"] = {
            "aggregation": "Pooled over all pre-step backlog-positive steps of complete blocks, including zero-service steps; step-weighted means.",
            "steps": steps, "zero_prefill_steps": sum(x["zero_prefill_steps"] for x in backlogs),
            "sum_actual_prefill_tokens": actual,
            "mean_actual_prefill_tokens": actual / steps if steps else None,
            "sum_budget_tokens": budgets, "mean_budget_tokens": budgets / steps if steps else None,
            "cap_frequencies": dict(sorted(caps.items(), key=lambda pair: int(pair[0]))),
            "mean_actual_prefill_tokens_by_block": [x["mean_actual_prefill_tokens"] for x in backlogs],
        }
    for letter in ("F", "M", "N"):
        key = "A_vs_" + letter
        pairs = [block["comparisons"][key] for block in complete]
        ag = result["policies"]["A"]["metrics"]["joint_slo_requests_per_s"]["mean_of_block_metrics"]
        bg = result["policies"][letter]["metrics"]["joint_slo_requests_per_s"]["mean_of_block_metrics"]
        result["comparisons"][key] = {
            "block_count": len(pairs),
            "added_qualified_counts_by_block": [p["primary"]["added_qualified_count"] for p in pairs],
            "lost_qualified_counts_by_block": [p["primary"]["lost_qualified_count"] for p in pairs],
            "goodput_change_percent_by_block": [p["primary"]["goodput_change_percent"] for p in pairs],
            "change_of_mean_goodput_percent": 100 * (ag / bg - 1) if bg else None,
            "timing_damage": {field: {
                "worsened_counts_by_block": [p["timing_damage_all_requests"][field]["worsened_count"] for p in pairs],
                "mean_deltas_s_by_block": [p["timing_damage_all_requests"][field]["mean_delta_s"] for p in pairs],
            } for field in ("ttft_s", "completion_s")},
            "mean_of_block_full_service_metric_deltas": {metric: statistics.fmean(p["full_service_metric_deltas"][metric] for p in pairs) for metric in METRICS},
        }
        if letter == "M":
            result["comparisons"][key]["pooled_prefill_backlog_mean_comparison"] = backlog_mean_comparison(
                result["policies"]["A"]["prefill_backlog"], result["policies"]["M"]["prefill_backlog"])
            result["comparisons"][key]["prefill_backlog_actual_mean_differences_by_block"] = [
                p["prefill_backlog_mean_comparison"]["A_minus_M_actual_prefill_tokens_per_backlog_step"] for p in pairs]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().with_name("confirmation_results.json"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    inputs = args.inputs or sorted(path for path in root.glob("block[0-9][0-9]") if path.is_dir())
    runs, metadata = sensitivity.load_custom_runs(inputs)
    by_block = collections.defaultdict(list)
    for run in runs:
        by_block[run["phase"]].append(run)
    blocks = [block_report(name, members, metadata[name]) for name, members in sorted(by_block.items())]
    expected_blocks = []
    design_path = root / "design.json"
    if not args.inputs and design_path.exists():
        design = json.loads(design_path.read_text())
        expected_blocks = ["block%02d" % item["block"] for item in design.get("blocks", [])]
    result = {
        "schema_version": 1, "primary_slo": sensitivity.BASE_SLO,
        "policy_labels": POLICIES,
        "scope": "Frozen confirmation blocks only; no old-episode pooling or primary-SLO retuning.",
        "validation": "Every discovered run is checked with parent analyze.summarize, including completion, fixed output lengths, per-request and total token accounting, budget, timestamps, and expected request count.",
        "aggregation": "Only blocks with all four valid complete policies enter the aggregate. Per-block outcomes remain visible. Mean-of-block metrics and ratio-of-total-counts-to-total-time are separate.",
        "damage_definition": "Any strictly positive feedback-minus-baseline TTFT or completion-flow change; descriptive counts, not significance tests. Missing or unfinished requests never qualify.",
        "cohort_definition": "Original source membership from request IDs; arrival jitter/permutation does not change long-wave membership.",
        "prefill_backlog_definition": "Use recorded prefill_backlog_tokens_before > 0, include P=0 steps, and check request counts/remaining tokens against add_s plus cumulative scheduled prompt reconstruction. A mismatch prevents the block from entering complete-block aggregation.",
        "expected_blocks": expected_blocks,
        "not_yet_observed_blocks": [name for name in expected_blocks if name not in by_block],
        "observed_runs": len(runs), "observed_blocks": len(blocks),
        "blocks": blocks, "aggregate_complete_blocks": aggregate_complete_blocks(blocks),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print("Wrote %s: %d observed runs, %d complete four-policy blocks" %
          (args.output, len(runs), result["aggregate_complete_blocks"]["block_count"]))
    for block in blocks:
        print(block["block"], "complete=" + str(block["complete_four_policy_block"]),
              {letter: (entry["primary"]["overall"]["qualified_request_count"],
                        entry["primary"]["overall"]["elapsed_s_denominator"],
                        entry["primary"]["overall"]["goodput_requests_per_s"])
               for letter, entry in block["policies"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
