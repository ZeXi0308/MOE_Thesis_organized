#!/usr/bin/env python3
"""Describe one donor pilot against both prior FIFO and retirement executions."""
import argparse
import json
from pathlib import Path
import evaluate_goodput_a_frozen as goodput
from C_NATIVE_STATIC_MATCHED_ANALYZE_V1 import need, read, summarize
from C_NATIVE_BOUND_MATCHED_ANALYZE_V1 import paired

REFERENCES = (("bound_fifo_1", "native_max_bound"),
              ("retirement_1", "native_retirement"),
              ("retirement_2", "native_retirement"),
              ("bound_fifo_2", "native_max_bound"))


def analyze(root, reference_root):
    complete = read(reference_root / "block-receipt.json")
    need(complete["status"] == "COMPLETE", "reference block incomplete")
    folder = root / "native_temporal_donor_1"
    receipt = read(folder / "launcher-receipt.json")
    need(receipt["status"] == "ACCEPTED_PILOT_ONLY" and not receipt["errors"],
         "donor pilot did not qualify")
    gate = read(folder / "temporal-cap-donor.json")
    need(gate["status"] == "DRAINED" and gate["drained"]
         and gate["admitted"] == gate["released"] == 128
         and gate["preemptions"] == 0 and gate["violations"] == []
         and gate["promises"] == gate["transfers"] + gate["cancellations"]
         and gate["outstanding_promise"] is None
         and 0 <= gate["physical_blocks_peak"] <= gate["logical_quota_peak_blocks"] <= 4096,
         "donor lifecycle or capacity incomplete")
    labels, raws, values, cells = [], [], [], []
    paths = [(label, arm, reference_root / label / arm / "raw.json")
             for label, arm in REFERENCES]
    paths.append(("native_temporal_donor_1", "native_temporal_donor", folder / "raw.json"))
    for label, arm, path in paths:
        raw = read(path)
        value, metrics = summarize(raw)
        labels.append(label); raws.append(raw); values.append(value)
        cells.append(dict(label=label, arm=arm, metrics=metrics,
                          all_request_metrics=value["requests"]))
    comparisons = []
    for index in range(len(REFERENCES)):
        result = paired(index, 4, labels, raws, values, cells)
        result["per_request_deltas"] = goodput.pair(values[index], values[4])["per_request"]
        comparisons.append(result)
    fields = ("policy", "admitted", "released", "preemptions", "promises", "transfers",
              "cancellations", "logical_quota_peak_blocks", "full_bound_sum_peak_blocks",
              "physical_blocks_peak", "eligible_calls", "schedule_calls", "hold_calls",
              "decision_seconds", "bookkeeping_seconds", "policy_wall_seconds")
    return dict(schema="c-native-temporal-cap-donor-analysis-v1",
        status="COMPLETE_DEVELOPMENT_PILOT_ONLY", cells=cells,
        reservation={key: gate[key] for key in fields},
        reference_to_donor_comparisons=comparisons,
        limits="One temporally separated donor pilot on a now-viewed cohort; both FIFO and both retirement references retained. No matched-repeat, equal-work causality, full CacheOPT reproduction, or generalization claim.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.root, args.reference_root)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
