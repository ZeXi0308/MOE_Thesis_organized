#!/usr/bin/env python3
"""Describe one qualified full-cap/PF/PF/full-cap batch4096 ablation block."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import evaluate_goodput_a_frozen as goodput
from C_NATIVE_STATIC_MATCHED_ANALYZE_V1 import need, read, summarize
from C_NATIVE_BOUND_MATCHED_ANALYZE_V1 import paired
from C_NATIVE_PAST_FUTURE_AE_MATCHED_ANALYZE_V1 import sha

ORDER = (("cap_1", "native_past_future_cap"),
         ("past_future_1", "native_past_future_ae"),
         ("past_future_2", "native_past_future_ae"),
         ("cap_2", "native_past_future_cap"))
INPUT_SHA = {"config.json": "634e7715618879775daf312fc98d77c1b25f4cd8e97007c101604ded317738a3",
             "workload.json": "9576395c9540c71be86dd182df03ab6d39cf1c961a6a24dc851d40991cf0a985",
             "INPUT_STATS.json": "797ccd408a60ee01bed2ac370f51f7145fa4f9ea3bcaecc5a1a4535afd034ac1"}
WRAPPER_SHA = "49bdec4150c7b66fef2f3dfd512f5f430bcf2d1a559c9a9eb255f1c7b11910f8"
CELL_SHA = "3479f00852a015f1786eedd164bd62ecd5281713d737a6a73d05794d85bdaee8"
CAP_SHA = "aca4720a69cade192505c61e9d3da7844e89356cfd1a0d73f886c40bf1d1609f"
PF_SHA = "28001321554a72bf1a451d26e15e6d4e2192c26d074891c6efd9bf5091ed3dc7"
PREDICTOR_SHA = "e37d50ee262ae3325521bc95edee14079119451cd96f182c8236e905bc9a7773"
POLICIES = {"native_past_future_cap": ("past-future-cap.json",
             "author_AE_full_output_cap_ablation_native_batch4096"),
            "native_past_future_ae": ("past-future-ae.json",
             "author_AE_statistical_peak_core_native_batch4096")}
BLOCK_SCHEMA = "c-native-past-future-cap-ablation-block-v1"
CELL_SCHEMA = "c-native-past-future-cap-ablation-cell-v1"


def qualified_cell(root: Path, index: int, label: str, arm: str,
                   prior_gate: dict) -> tuple[dict, dict, dict]:
    folder = root / label
    cell = folder / arm
    receipt = read(folder / "launcher-receipt.json")
    need(receipt.get("schema") == CELL_SCHEMA
         and receipt.get("sequence_index") == index
         and receipt.get("label") == label and receipt.get("arm") == arm
         and receipt.get("status") == "ACCEPTED_FOR_ABLATION_BLOCK"
         and receipt.get("errors") == []
         and receipt.get("runner_source_sha256") == WRAPPER_SHA
         and receipt.get("cell_source_sha256") == CELL_SHA
         and receipt.get("cap_adapter_source_sha256") == CAP_SHA
         and receipt.get("pf_adapter_source_sha256") == PF_SHA
         and receipt.get("pf_predictor_source_sha256") == PREDICTOR_SHA
         and receipt.get("input_file_sha256") == INPUT_SHA
         and receipt.get("batch_tokens") == 4096
         and receipt.get("arrival_scale") == 1.0
         and receipt.get("prior_gate") == prior_gate,
         f"{label}: frozen ablation cell receipt differs")
    config, args = (read(cell / name) for name in ("config.json", "engine_args.json"))
    source, status = (read(cell / name) for name in ("source-receipt.json", "status.json"))
    audit = read(cell / "baseline-result-audit.json")
    raw_path = cell / "raw.json"
    raw = read(raw_path)
    need(config.get("reservation_policy") == status.get("arm") == arm
         and config.get("requests") == 128 and config.get("cap") == 32
         and config.get("output_mode") == "eos" and config.get("output_tokens") == 1024
         and config.get("seed") == args.get("seed") == 20260905
         and config.get("policy_seed") == 20261001
         and config.get("arrival_regime") == raw.get("regime") == "poisson_v1"
         and config.get("arrival_span_s") == 22.682329
         and raw.get("arrival_scale") == 1.0
         and config.get("max_num_batched_tokens") == args.get("max_num_batched_tokens") == 4096
         and args.get("max_num_seqs") == 32 and args.get("max_model_len") == 4096
         and args.get("long_prefill_token_threshold") == 0
         and source.get("input_sha256") == INPUT_SHA
         and config.get("development_input_receipt", {}).get("input_sha256") == INPUT_SHA
         and status.get("status") == raw.get("status") == "COMPLETE"
         and status.get("requests_completed") == 128
         and audit.get("status") == "PASS" and audit.get("requests") == 128,
         f"{label}: input, batch or completion contract differs")
    value, metrics = summarize(raw)
    need(metrics["output_tokens"] == receipt.get("output_tokens")
         == audit.get("output_tokens") == status.get("output_tokens")
         and sha(raw_path) == receipt.get("raw_sha256")
         and type(raw.get("actual_preemption_count")) is int,
         f"{label}: raw SHA, output or preemption accounting differs")
    filename, policy_name = POLICIES[arm]
    gate_path = cell / filename
    gate = read(gate_path)
    need(gate.get("status") == "DRAINED" and gate.get("drained") is True
         and gate.get("policy") == policy_name
         and gate.get("admitted") == gate.get("completed") == 128
         and gate.get("violations") == []
         and gate.get("preemptions") == raw["actual_preemption_count"]
         and type(gate.get("physical_blocks_peak")) is int
         and 0 <= gate["physical_blocks_peak"] <= 4096
         and receipt.get("policy_summary", {}).get("gate_sha256") == sha(gate_path),
         f"{label}: policy did not drain or honor native KV capacity")
    fields = ("policy", "admitted", "completed", "resumptions", "preemptions",
              "schedule_calls", "hold_calls", "decision_seconds", "physical_blocks_peak")
    policy = {field: gate[field] for field in fields}
    need(all(receipt["policy_summary"].get(field) == policy[field]
             for field in fields if field != "schedule_calls"),
         f"{label}: policy summary differs from qualified receipt")
    policy["gate_sha256"] = sha(gate_path)
    row = dict(index=index, label=label, arm=arm, metrics=metrics,
               all_request_metrics=value["requests"], policy_summary=policy,
               pressure_qualified=receipt.get("pressure_qualified"),
               raw_sha256=sha(raw_path))
    return raw, value, row


def analyze(root: Path) -> dict:
    need(root.is_dir() and not root.is_symlink(), "ablation root absent or linked")
    labels = [label for label, _ in ORDER]
    start = read(root / "block-start.json")
    block = read(root / "block-receipt.json")
    need(start.get("schema") == block.get("schema") == BLOCK_SCHEMA
         and start.get("order") == block.get("order") == block.get("completed") == labels
         and block.get("status") == "COMPLETE" and block.get("error") is None
         and start.get("runner_source_sha256") == block.get("runner_source_sha256") == WRAPPER_SHA
         and start.get("cell_source_sha256") == CELL_SHA
         and start.get("cap_adapter_source_sha256") == CAP_SHA
         and start.get("pf_adapter_source_sha256") == PF_SHA
         and start.get("pf_predictor_source_sha256") == PREDICTOR_SHA
         and start.get("input_file_sha256") == INPUT_SHA
         and start.get("batch_tokens") == 4096 and start.get("arrival_scale") == 1.0,
         "full-cap/PF four-cell block incomplete or source/input differs")
    prior_gate = start.get("prior_gate")
    need(type(prior_gate) is dict and type(prior_gate.get("pilot")) is dict
         and len(prior_gate.get("matched_block_receipt_sha256", "")) == 64,
         "previous pilot/matched-block provenance absent")
    raws, values, cells = [], [], []
    for index, (label, arm) in enumerate(ORDER):
        raw, value, row = qualified_cell(root, index, label, arm, prior_gate)
        if values:
            goodput.pair(values[0], value)  # Same 128 IDs, prompts, arrivals and caps.
        raws.append(raw)
        values.append(value)
        cells.append(row)

    def comparison(reference: int, candidate: int) -> dict:
        result = paired(reference, candidate, labels, raws, values, cells)
        ref, cand = cells[reference]["metrics"], cells[candidate]["metrics"]
        result["primary_20_4"] = dict(
            reference_requests=ref["joint_20_4"], candidate_requests=cand["joint_20_4"],
            candidate_minus_reference_requests=cand["joint_20_4"] - ref["joint_20_4"],
            reference_goodput_requests_s=ref["goodput_20_4_requests_s"],
            candidate_goodput_requests_s=cand["goodput_20_4_requests_s"],
            candidate_minus_reference_goodput_requests_s=(
                cand["goodput_20_4_requests_s"] - ref["goodput_20_4_requests_s"]))
        result["output_tokens"] = dict(reference=ref["output_tokens"],
            candidate=cand["output_tokens"],
            candidate_minus_reference=cand["output_tokens"] - ref["output_tokens"])
        result["per_request_deltas"] = goodput.pair(
            values[reference], values[candidate])["per_request"]
        return result

    return dict(schema="c-native-past-future-cap-ablation-analysis-v1",
        status="COMPLETE_DESCRIPTIVE_DEVELOPMENT_ONLY", order=labels,
        input_file_sha256=INPUT_SHA, batch_tokens=4096,
        previous_qualification_provenance=prior_gate, cells=cells,
        policy_summaries={row["label"]: row["policy_summary"] for row in cells},
        cap_to_past_future_adjacent_pairs=[comparison(a, b) for a, b in ((0, 1), (3, 2))],
        within_arm_repeats={"native_past_future_cap": comparison(0, 3),
                            "native_past_future_ae": comparison(1, 2)},
        limits="One viewed cohort, two runs per arm on one batch4096 GPU regime. This isolates "
               "the service contribution and decision cost of history/sampling relative to full-cap AE peak scoring. "
               "Changed output IDs/lengths preclude equal-work causality; host-return gaps omit chunk interiors. "
               "It is neither a new C method, unseen confirmation nor a full LightLLM reproduction.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = analyze(args.root)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
