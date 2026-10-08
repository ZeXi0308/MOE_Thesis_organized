#!/usr/bin/env python3
"""Can fixed cap + free-KV rules express each observed budget-arm action set?

python capacity_rule_diagnostic.py runs/westb-20261008/declared-budget-r01 \
    --output analysis/westb-declared-budget-r01.capacity-rules.json

This is an observed-state representation diagnostic, not a service replay,
threshold selection, or new policy. Arms are never pooled.
"""
import argparse
import hashlib
import json
from pathlib import Path


ARMS = ("probe-01-declaredbudget", "probe-02-declaredbudget")
MAX_CAP = 256
MAX_BLOCKS = 32768


def rule_allows(row, cap, watermark, age_bypass):
    return (row["native_fit"] is True and row["active"] < cap and
            (row["free_blocks"] >= watermark or
             (age_bypass and row["age_s"] >= 10)))


def witness(row):
    if row is None:
        return None
    keys = ("decision_index", "request_id", "t", "active", "admitted_inflight",
            "free_blocks", "age_s", "native_fit", "native_allocation_result",
            "budget_before_blocks", "budget_required_blocks",
            "budget_after_if_admitted_blocks", "budget_limit_blocks",
            "declared_prompt_tokens", "declared_max_tokens")
    return {key: row[key] for key in keys}


def family(allow, deny, age_bypass):
    """The smallest cap and largest permitted floor give the strictest rule."""
    active_witness = max(allow, key=lambda row: row["active"])
    cap_min = active_witness["active"] + 1
    constrained = [row for row in allow if not age_bypass or row["age_s"] < 10]
    free_witness = min(constrained, key=lambda row: row["free_blocks"], default=None)
    watermark_max = free_witness["free_blocks"] if free_witness else MAX_BLOCKS
    within_domain = 1 <= cap_min <= MAX_CAP and 0 <= watermark_max <= MAX_BLOCKS
    # Each allow requires C >= Cmin; each non-bypassed allow requires W <= Wmax.
    # Raising C or lowering W can only admit more rows, so a denial admitted by
    # this strictest combination is a contradiction for every legal parameter.
    conflicts = [row for row in deny
                 if rule_allows(row, cap_min, watermark_max, age_bypass)] if within_domain else []
    if within_domain and not all(rule_allows(row, cap_min, watermark_max, age_bypass)
                                 for row in allow):
        raise ValueError("Strictest combination failed an allow constraint")
    return dict(age_bypass_s=10 if age_bypass else None, Cmin=cap_min,
        Wmax=watermark_max, within_parameter_domain=within_domain,
        watermark_constraining_allow_evaluations=len(constrained),
        all_allow_constraints_satisfied=within_domain,
        fixed_rule_can_represent_observed_labels=within_domain and not conflicts,
        conflicting_denial_evaluations=len(conflicts) if within_domain else None,
        conflicting_denial_unique_request_ids=(len({row["request_id"] for row in conflicts})
                                               if within_domain else None),
        conflict_denial_witness=witness(conflicts[0]) if conflicts else None,
        cap_constraint_allow_witness=witness(active_witness),
        watermark_constraint_allow_witness=witness(free_witness),
        watermark_bound_source="observed_allow" if free_witness else "parameter_domain_upper_bound",
        verdict=("NO_PARAMETER_PRESERVES_ALLOWS" if not within_domain else
                 "NO_FIXED_PARAMETER_REPRESENTATION" if conflicts else
                 "FIXED_PARAMETER_REPRESENTATION_EXISTS"))


def arm_diagnostic(path):
    payload = path.read_bytes()
    report = json.loads(payload)
    if (report["mode"] != "declared_budget" or report["cap"] != MAX_CAP or
            report["budget_blocks"] != MAX_BLOCKS or report["block_size"] != 16):
        raise ValueError(f"{path}: not the declared v15 budget configuration")
    starts = {row["request_id"]: row["first_prefill_perf_s"] for row in report["starts"]}
    if len(starts) != len(report["starts"]):
        raise ValueError(f"{path}: duplicate first-prefill start identity")
    rows = [dict(row, decision_index=index) for index, row in enumerate(report["decisions"])]
    local = [row for row in rows if row["reason"] == "allow" and row["denied"] is False
        and row.get("native_fit") is True and row.get("native_allocation_result") is True
        and row.get("final_allowed") is True and row.get("num_new_tokens", 0) > 0
        and row.get("load_kv_async") is False]
    allow = [row for row in local if starts.get(row["request_id"], -float("inf")) >=
             report["origin_perf_s"] + row["t"]]
    if len(allow) != 384 or len({row["request_id"] for row in allow}) != 384 or len(local) != 384:
        raise ValueError(f"{path}: expected 384 successful local allocations with matched starts")
    deny = [row for row in rows if row["reason"] == "declared_budget"
            and row["denied"] is True and row.get("native_fit") is True
            and row.get("baseline_allowed") is True]
    if not deny:
        raise ValueError(f"{path}: no native-fit direct budget denial")
    for row in allow + deny:
        if (row["fifo_held"] or row["active"] != row["admitted_inflight"] or
                not 0 <= row["active"] < MAX_CAP or not 0 <= row["free_blocks"] <= MAX_BLOCKS or
                row["budget_before_blocks"] + row["budget_required_blocks"] !=
                row["budget_after_if_admitted_blocks"]):
            raise ValueError(f"{path}: inconsistent selected decision {row['decision_index']}")
    return dict(source_path=str(path), source_sha256=hashlib.sha256(payload).hexdigest(),
        origin_perf_s=report["origin_perf_s"],
        allow_evaluations=len(allow),
        allow_unique_request_ids=len({row["request_id"] for row in allow}),
        direct_budget_denial_evaluations=len(deny),
        direct_budget_denial_unique_request_ids=len({row["request_id"] for row in deny}),
        families={"without_age_bypass": family(allow, deny, False),
                  "with_10s_age_bypass": family(allow, deny, True)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = dict(schema_version=1,
        question="Can fixed concurrency + actual free-KV rules reproduce all selected observed actions?",
        rule="native_fit && active < C && (free_blocks >= W || (age_bypass && age_s >= 10))",
        parameter_domain=dict(C=[1, MAX_CAP], W=[0, MAX_BLOCKS], integer_parameters=True),
        method="For each arm separately, Cmin=max(allow.active)+1; Wmax=min(allow.free_blocks), "
            "excluding age>=10 allows only in the bypass family. If no allow constrains W, "
            "Wmax=32768. This is the strictest combination preserving all allows. Any selected "
            "denial allowed by it rules out every parameter in that family; no grid search is used.",
        selected_rows="384 actual successful new local allocations per arm, with exact request-ID "
            "first-prefill host schedule-return timestamps at/after the decision; versus native-fit, "
            "baseline_allowed direct declared-budget denials. FIFO, native-fit failures, async loads "
            "and unvisited scan-break suffixes are excluded. baseline_allowed refers to cap256, "
            "not a fixed128 counterfactual.",
        evidence_limits="Observed-state action representation only, not counterfactual service, "
            "threshold tuning, a new strategy, or proof of service benefit/novelty. States across "
            "different decisions need not have identical KV or active values; each rule is evaluated "
            "on its own recorded state. The two arms are not pooled. No future output is used. "
            "Allocation and a matched host schedule-return do not establish GPU execution time. "
            "Repeated denial evaluations are not independent requests or experiments.",
        witness_units="t: seconds from admission origin_perf_s; age_s: recorded external-arrival age; "
            "free_blocks: physical free KV pages; budget_*_blocks: separate declared virtual pages. "
            "Each page has 16 tokens. Unique IDs are native request IDs within one arm.",
        arms={arm: arm_diagnostic(args.run_directory/arm/"admission.json") for arm in ARMS})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        json.dump(result, output, indent=2)
        output.write("\n")
    for arm, data in result["arms"].items():
        for name, item in data["families"].items():
            print(arm, name, "Cmin", item["Cmin"], "Wmax", item["Wmax"],
                  "conflicts", item["conflicting_denial_evaluations"],
                  "unique", item["conflicting_denial_unique_request_ids"], item["verdict"])


if __name__ == "__main__":
    main()
