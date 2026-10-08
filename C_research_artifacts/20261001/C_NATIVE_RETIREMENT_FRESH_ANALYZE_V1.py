#!/usr/bin/env python3
"""Describe the frozen fresh-cohort native/bound/retirement mirrored block."""
import argparse
import json
from pathlib import Path
import evaluate_goodput_a_frozen as goodput
from C_NATIVE_STATIC_MATCHED_ANALYZE_V1 import need, read, summarize
from C_NATIVE_BOUND_MATCHED_ANALYZE_V1 import paired

ORDER = (("native_1", "native_full"), ("bound_fifo_1", "native_max_bound"),
         ("retirement_1", "native_retirement"), ("retirement_2", "native_retirement"),
         ("bound_fifo_2", "native_max_bound"), ("native_2", "native_full"))


def analyze(root):
    labels = [label for label, _ in ORDER]
    block = read(root / "block-receipt.json")
    need(block["status"] == "COMPLETE" and block["error"] is None
         and block["order"] == block["completed"] == labels, "block incomplete")
    raws, values, cells = [], [], []
    for i, (label, arm) in enumerate(ORDER):
        folder = root / label
        receipt = read(folder / "launcher-receipt.json")
        need(receipt["sequence_index"] == i and receipt["label"] == label
             and receipt["arm"] == arm and receipt["status"] == "ACCEPTED_FOR_MATCHED_BLOCK"
             and receipt["errors"] == [], "cell incomplete")
        raw = read(folder / arm / "raw.json")
        value, metrics = summarize(raw)
        need(metrics["output_tokens"] == receipt["output_tokens"], "output total differs")
        reservation = {"policy": "native_no_offload"}
        if arm != "native_full":
            filename = "max-bound-admission.json" if arm == "native_max_bound" else "retirement-envelope.json"
            gate = read(folder / arm / filename)
            need(gate["status"] == "DRAINED" and gate["admitted"] == gate["released"] == 128
                 and gate["preemptions"] == 0 and gate["violations"] == [], "gate incomplete")
            fields = ["status", "preemptions", "admitted", "released", "schedule_calls", "hold_calls"]
            fields += (["reserved_blocks_peak"] if arm == "native_max_bound" else
                       ["policy", "full_bound_sum_peak", "envelope_peak_blocks", "physical_blocks_peak",
                        "eligible_calls", "incremental_admissions", "decision_seconds"])
            reservation = {k: gate[k] for k in fields}
        cells.append(dict(index=i, label=label, arm=arm, metrics=metrics,
                          all_request_metrics=value["requests"],
                          reservation=reservation))
        raws.append(raw); values.append(value)
    adjacent = []
    for a, b in ((1, 2), (4, 3), (0, 2), (5, 3)):
        result = paired(a, b, labels, raws, values, cells)
        result["per_request_deltas"] = goodput.pair(values[a], values[b])["per_request"]
        adjacent.append(result)
    return dict(schema="c-native-retirement-fresh-analysis-v1",
        status="COMPLETE_FROZEN_FRESH_COHORT_BLOCK", order=labels, cells=cells,
        reference_to_retirement_pairs=adjacent,
        within_arm_repeats={arm: paired(a, b, labels, raws, values, cells)
                           for arm, a, b in (("native_full", 0, 5), ("native_max_bound", 1, 4), ("native_retirement", 2, 3))},
        limits="One frozen fresh cohort relative to recorded C selection, two runs per arm on one replacement GPU. This is not a population confidence interval or final paper validation. Changed outputs preclude equal-work causality. Host-return gaps omit chunk interiors.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = analyze(args.root)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
