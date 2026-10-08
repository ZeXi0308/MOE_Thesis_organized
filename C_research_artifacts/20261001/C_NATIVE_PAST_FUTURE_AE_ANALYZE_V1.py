#!/usr/bin/env python3
"""Describe one batch4096 author-AE core pilot without cross-regime pairing."""
import argparse
import json
from C_NATIVE_STATIC_MATCHED_ANALYZE_V1 import need, read, summarize
from pathlib import Path


def analyze(cell):
    receipt = read(cell / "launcher-receipt.json")
    need(receipt["status"] == "ACCEPTED_PILOT_ONLY" and not receipt["errors"],
         "native pilot not qualified")
    args = read(cell / "engine_args.json")
    need(args["max_num_batched_tokens"] == 4096 and args["max_num_seqs"] == 32,
         "new batch4096 regime changed")
    raw = read(cell / "raw.json")
    values, metrics = summarize(raw)
    gate = read(cell / "past-future-ae.json")
    need(gate["status"] == "DRAINED" and not gate["violations"], "policy not drained")
    return dict(schema="c-native-past-future-ae-analysis-v1",
        status="COMPLETE_AUTHOR_AE_CORE_PORT_PILOT_ONLY", metrics=metrics,
        all_request_metrics=values["requests"],
        policy_summary={k: v for k, v in gate.items() if k not in ("steps", "events", "decisions")},
        limits="Viewed cohort, one batch4096 native port. No matched references, "
               "no equal-work or quality claim, no full LightLLM reproduction. "
               "Prior batch1024 outcomes are not paired with this run.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    with args.output.open("x") as stream:
        json.dump(analyze(args.cell), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
