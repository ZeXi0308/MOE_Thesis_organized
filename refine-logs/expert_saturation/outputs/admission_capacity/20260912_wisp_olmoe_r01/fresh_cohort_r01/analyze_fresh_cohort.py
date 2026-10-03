"""Describe two fresh F/X finite-arrival cohorts; preserve every engine and cost."""
import argparse
import json
import statistics
from pathlib import Path
import analyze_covered
import finite_metrics
ROOT = Path(__file__).resolve().parent
EXPECTED = [
    ("cohort0", "c0_0_fullstage", "fullstage"),
    ("cohort0", "c0_1_oneshot", "oneshot"),
    ("cohort0", "c0_2_oneshot", "oneshot"),
    ("cohort0", "c0_3_fullstage", "fullstage"),
    ("cohort1", "c1_0_oneshot", "oneshot"),
    ("cohort1", "c1_1_fullstage", "fullstage"),
    ("cohort1", "c1_2_fullstage", "fullstage"),
    ("cohort1", "c1_3_oneshot", "oneshot"),
]
ADJACENT = [
    ("cohort0", [0, 1], "c0_0_fullstage", "c0_1_oneshot"),
    ("cohort0", [2, 3], "c0_3_fullstage", "c0_2_oneshot"),
    ("cohort1", [0, 1], "c1_1_fullstage", "c1_0_oneshot"),
    ("cohort1", [2, 3], "c1_2_fullstage", "c1_3_oneshot"),
]
RETESTS = [
    ("cohort0", "fullstage", "c0_0_fullstage", "c0_3_fullstage"),
    ("cohort0", "oneshot", "c0_1_oneshot", "c0_2_oneshot"),
    ("cohort1", "oneshot", "c1_0_oneshot", "c1_3_oneshot"),
    ("cohort1", "fullstage", "c1_1_fullstage", "c1_2_fullstage"),
]
def read(path):
    return json.loads(path.read_text())
def metric_means(rows):
    fields = set(rows[0]["metrics"])
    for row in rows[1:]:
        fields &= row["metrics"].keys()
    return {key: statistics.mean(row["metrics"][key] for row in rows)
            for key in sorted(fields)}
def direct_comparison(baseline, target):
    fields = baseline.keys() & target.keys()
    return dict(delta_x_minus_f={key: target[key] - baseline[key] for key in sorted(fields)},
                delta_pct={key: 100 * (target[key] / baseline[key] - 1) if baseline[key] else None
                           for key in sorted(fields)})
def directional_answer(values):
    signs = ["decrease" if value < 0 else "increase" if value > 0 else "equal" for value in values]
    return dict(by_cohort=signs, agreement=(signs[0] == signs[1]),
                classification=(signs[0] + "_in_both_cohorts" if signs[0] == signs[1]
                                else "direction_differs_between_cohorts"))
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.input_dir.resolve()
    protocol, plans = read(root / "protocol.json"), read(root / "run_cells.json")
    execution_path = root / "results" / "execution.json"
    execution = read(execution_path) if execution_path.exists() else {"status": "UNRUN", "cells": []}
    entries = {cell["label"]: cell for cell in execution["cells"]}
    issues, missing, rows, cohort_ids = [], [], [], {}
    declared = [(p.get("cohort"), p.get("label"), p.get("mode")) for p in plans]
    if declared != EXPECTED:
        issues.append("run_cells does not match frozen cohort/order/mode design")
    if set(entries) - {label for _, label, _ in EXPECTED}:
        issues.append("execution contains undeclared result cells")
    for plan in plans:
        label, cohort = plan["label"], plan["cohort"]
        if entries.get(label, {}).get("status") != "COMPLETE":
            missing.append(label)
            continue
        try:
            workload_path = root / "prepared" / cohort / "workload.json"
            workload = read(workload_path)
            sources, prompts = workload.get("source_requests"), workload.get("actual_prompt_token_ids")
            if (workload.get("schema") != "olmoe-admission-inputs-v1" or not isinstance(sources, list)
                    or not isinstance(prompts, list) or len(sources) != 16 or len(prompts) != 16
                    or len({source["request_id"] for source in sources}) != 16
                    or any(len(ids) != 128 for ids in prompts)):
                raise ValueError("invalid prepared cohort schema or coverage")
            warmup_document_id = sources[0]["document_id"]
            cohort_ids[cohort] = {source["request_id"] for source in sources}
            row = analyze_covered.analyze_cell(root, plan, entries[label],
                workload_path=workload_path, warmup_document_id=warmup_document_id)
            raw, command = read(root / "results" / label / "raw.json"), [str(x) for x in entries[label].get("command", [])]
            if raw.get("cpu_diagnostics") is not False or raw.get("runtime_observer_enabled") is not False:
                row["issues"].append("host CPU/runtime observer unexpectedly enabled")
            if sum(x.endswith("/instrumentation/run_covered.py") for x in command) != 1 or any(x.endswith("/instrumentation/run_host_cost.py") for x in command):
                row["issues"].append("execution did not use the unobserved covered runner")
            row["cohort"] = cohort
            rows.append(row)
            issues.extend(label + ": " + issue for issue in row["issues"])
        except Exception as exc:
            issues.append(label + ": " + repr(exc))
    if len(cohort_ids) == 2 and cohort_ids["cohort0"] & cohort_ids["cohort1"]:
        issues.append("prepared cohort request IDs overlap")
    by_label = {row["label"]: row for row in rows}
    adjacent_pairs = []
    for cohort, positions, fullstage, oneshot in ADJACENT:
        if fullstage in by_label and oneshot in by_label:
            item = finite_metrics.comparison(by_label[fullstage], by_label[oneshot])
            item.update(cohort=cohort, adjacent_positions=positions,
                        direction="X minus F; F is denominator")
            adjacent_pairs.append(item)
    same_mode_retests = []
    for cohort, mode, first, second in RETESTS:
        if first in by_label and second in by_label:
            item = finite_metrics.comparison(by_label[first], by_label[second])
            item.update(cohort=cohort, mode=mode, direction="later minus earlier engine")
            same_mode_retests.append(item)
    cohort_summaries = []
    for cohort in ("cohort0", "cohort1"):
        selected = [row for row in rows if row["cohort"] == cohort]
        fullstage = [row for row in selected if row["mode"] == "fullstage"]
        oneshot = [row for row in selected if row["mode"] == "oneshot"]
        if len(fullstage) == len(oneshot) == 2:
            f_mean, x_mean = metric_means(fullstage), metric_means(oneshot)
            cohort_summaries.append(dict(cohort=cohort, engines_per_mode=2,
                descriptive_requests_per_mode=32, fullstage_engine_metric_means=f_mean,
                oneshot_engine_metric_means=x_mean, x_vs_f_engine_mean=direct_comparison(f_mean, x_mean),
                denominator="mean of two F engine-level metrics; requests are not independent n"))
    question_result = None
    if len(cohort_summaries) == 2:
        capture = [row["x_vs_f_engine_mean"]["delta_pct"]["capture_wall_s"]
                   for row in cohort_summaries]
        completion = [row["x_vs_f_engine_mean"]["delta_pct"]["completion_latency_s_mean"]
                      for row in cohort_summaries]
        question_result = dict(primary_capture_delta_pct=dict(zip(("cohort0", "cohort1"), capture)),
            primary_capture_direction=directional_answer(capture),
            key_secondary_mean_completion_delta_pct=dict(zip(("cohort0", "cohort1"), completion)),
            key_secondary_mean_completion_direction=directional_answer(completion),
            interpretation="Descriptive engine-level directions only; no request-level independence or inference.")
    done = (execution["status"] == "COMPLETE" and len(rows) == len(plans) == 8
            and len(adjacent_pairs) == len(same_mode_retests) == 4
            and len(cohort_summaries) == 2 and not missing and not issues)
    status = ("DESCRIPTIVE_FRESH_COHORT_FX" if done else "INVALID_ANALYSIS_INPUT" if issues
              else "UNRUN" if not entries else "INCOMPLETE")
    report = dict(status=status, issues=issues, missing_cells=missing, cells=rows,
        research_question=protocol["question"],
        primary_metric="capture_wall_s: first planned arrival through complete finite-cohort drain",
        key_secondary_metric="completion_latency_s_mean", byte_metrics={"h2d": "payload_bytes", "d2d": "d2d_bytes"}, adjacent_x_vs_f=adjacent_pairs,
        same_mode_retests=same_mode_retests, cohort_engine_mean_x_vs_f=cohort_summaries,
        research_question_result=question_result,
        aggregation_scope="Two engine-level values per mode within each cohort; no pooling of 128 requests as independent n.",
        claim_ceiling="Two fresh source-order 16-document cohorts, two engines/mode/cohort. Report all four adjacent X/F pairs and all costs. Descriptive continuation evidence only; no 3x-max-difference noise rule, p-value, post-hoc output filtering, quality, SLO, stable benefit or method GO.")
    with args.out.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: report[key] for key in ("status", "issues", "missing_cells")}))
if __name__ == "__main__":
    main()
