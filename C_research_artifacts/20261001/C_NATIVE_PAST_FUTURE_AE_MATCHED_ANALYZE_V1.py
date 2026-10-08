#!/usr/bin/env python3
"""Analyze one qualified native/PF/PF/native batch4096 development block."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import evaluate_goodput_a_frozen as goodput
from C_NATIVE_STATIC_MATCHED_ANALYZE_V1 import need, read, summarize
from C_NATIVE_BOUND_MATCHED_ANALYZE_V1 import paired

ORDER = (("native_1", "native_full"),
         ("past_future_1", "native_past_future_ae"),
         ("past_future_2", "native_past_future_ae"),
         ("native_2", "native_full"))
INPUT_SHA = {"config.json": "634e7715618879775daf312fc98d77c1b25f4cd8e97007c101604ded317738a3",
             "workload.json": "9576395c9540c71be86dd182df03ab6d39cf1c961a6a24dc851d40991cf0a985",
             "INPUT_STATS.json": "797ccd408a60ee01bed2ac370f51f7145fa4f9ea3bcaecc5a1a4535afd034ac1"}
WRAPPER_SHA = "240e22083a1b4480ac53273ded832cfb335273f910bbf76f07a706373e37ea8b"
CELL_SHA = "b9d1a9609cd355449a31c9b0800ab0d0df27960787381fe5b86d8c0e0de22cd3"
PF_ADAPTER_SHA = "28001321554a72bf1a451d26e15e6d4e2192c26d074891c6efd9bf5091ed3dc7"
PF_PREDICTOR_SHA = "e37d50ee262ae3325521bc95edee14079119451cd96f182c8236e905bc9a7773"
BLOCK_SCHEMA = "c-native-past-future-ae-matched-block-v1"
CELL_SCHEMA = "c-native-past-future-ae-matched-cell-v1"
PF_POLICY = "author_AE_statistical_peak_core_native_batch4096"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for piece in iter(lambda: stream.read(1 << 20), b""):
            h.update(piece)
    return h.hexdigest()


def qualified_cell(root: Path, index: int, label: str, arm: str,
                   pilot_gate: dict) -> tuple[dict, dict, dict]:
    folder = root / label
    cell = folder / arm
    receipt = read(folder / "launcher-receipt.json")
    need(receipt.get("schema") == CELL_SCHEMA
         and receipt.get("sequence_index") == index
         and receipt.get("label") == label and receipt.get("arm") == arm
         and receipt.get("status") == "ACCEPTED_FOR_MATCHED_BLOCK"
         and receipt.get("errors") == []
         and receipt.get("runner_source_sha256") == WRAPPER_SHA
         and receipt.get("cell_source_sha256") == CELL_SHA
         and receipt.get("input_file_sha256") == INPUT_SHA
         and receipt.get("batch_tokens") == 4096
         and receipt.get("arrival_scale") == 1.0
         and receipt.get("pilot_gate") == pilot_gate,
         f"{label}: matched launcher receipt differs")
    life = receipt.get("child_lifecycle", {})
    need(type(life) is dict and life.get("exit_code") == life.get("child_returncode") == 0
         and life.get("child_reaped") is True and life.get("timed_out") is False
         and life.get("launcher_error") is None and life.get("interruption_signal") is None,
         f"{label}: child lifetime incomplete")
    config = read(cell / "config.json")
    args = read(cell / "engine_args.json")
    environment = read(cell / "environment.json")
    source = read(cell / "source-receipt.json")
    status = read(cell / "status.json")
    audit = read(cell / "baseline-result-audit.json")
    raw_path = cell / "raw.json"
    raw = read(raw_path)
    need(config.get("reservation_policy") == status.get("arm") == arm
         and config.get("requests") == config.get("cap") * 4 == 128
         and config.get("output_mode") == "eos"
         and config.get("output_tokens") == 1024
         and config.get("prompt_tokens") == 3072
         and config.get("seed") == args.get("seed") == 20260905
         and config.get("policy_seed") == 20261001
         and config.get("arrival_regime") == raw.get("regime") == "poisson_v1"
         and config.get("arrival_rate_per_s") == 5.0
         and config.get("arrival_span_s") == 22.682329
         and raw.get("arrival_scale") == 1.0
         and args.get("max_num_batched_tokens") == config.get("max_num_batched_tokens") == 4096
         and args.get("long_prefill_token_threshold") == 0
         and args.get("max_num_seqs") == 32
         and args.get("max_model_len") == 4096
         and args.get("async_scheduling") is False
         and args.get("enable_prefix_caching") is False
         and args.get("scheduler_reserve_full_isl") is True
         and "kv_offloading_size" not in args and "kv_transfer_config" not in args
         and source.get("status") == "READY_FRESH_POLICY_TEST_INPUT"
         and source.get("input_sha256") == INPUT_SHA
         and config.get("development_input_receipt", {}).get("input_sha256") == INPUT_SHA
         and environment.get("fresh_input_sha256") == INPUT_SHA
         and environment.get("pilot_source_sha256") == CELL_SHA
         and environment.get("past_future_adapter_sha256") == PF_ADAPTER_SHA
         and environment.get("past_future_predictor_sha256") == PF_PREDICTOR_SHA
         and status.get("status") == raw.get("status") == "COMPLETE"
         and status.get("requests_completed") == 128
         and audit.get("status") == "PASS" and audit.get("requests") == 128,
         f"{label}: input/arrival/batch/runtime/output contract differs")
    value, metrics = summarize(raw)
    need(metrics["output_tokens"] == receipt.get("output_tokens")
         == audit.get("output_tokens") == status.get("output_tokens")
         and sha(raw_path) == receipt.get("raw_sha256")
         and type(raw.get("actual_preemption_count")) is int,
         f"{label}: raw output/preemption receipt differs")
    if arm == "native_full":
        need(not (cell / "past-future-ae.json").exists()
             and receipt.get("policy_summary", {}).get("policy") == "native_full",
             f"{label}: native cell unexpectedly used PF policy")
        policy = dict(policy="native_full", native_preemptions=raw["actual_preemption_count"])
    else:
        gate_path = cell / "past-future-ae.json"
        gate = read(gate_path)
        need(gate.get("status") == "DRAINED" and gate.get("drained") is True
             and gate.get("policy") == PF_POLICY
             and gate.get("admitted") == gate.get("completed") == 128
             and gate.get("violations") == []
             and gate.get("preemptions") == raw["actual_preemption_count"]
             and type(gate.get("physical_blocks_peak")) is int
             and 0 <= gate["physical_blocks_peak"] <= 4096
             and receipt.get("policy_summary", {}).get("gate_sha256") == sha(gate_path),
             f"{label}: Past-Future gate did not qualify")
        fields = ("policy", "admitted", "completed", "resumptions", "preemptions",
                  "schedule_calls", "hold_calls", "decision_seconds", "physical_blocks_peak")
        policy = {key: gate[key] for key in fields}
        policy["history_window_size_at_drain"] = len(gate["history"])
        policy["gate_sha256"] = sha(gate_path)
        need(all(receipt["policy_summary"].get(key) == policy[key]
                 for key in fields if key != "schedule_calls"),
             f"{label}: PF policy summary differs from receipt")
    row = dict(index=index, label=label, arm=arm, metrics=metrics,
               all_request_metrics=value["requests"], policy_summary=policy,
               pressure_qualified=receipt.get("pressure_qualified"),
               raw_sha256=sha(raw_path))
    return raw, value, row


def analyze(root: Path) -> dict:
    need(root.is_dir() and not root.is_symlink(), "matched block root absent or linked")
    labels = [label for label, _ in ORDER]
    start = read(root / "block-start.json")
    block = read(root / "block-receipt.json")
    need(start.get("schema") == block.get("schema") == BLOCK_SCHEMA
         and start.get("order") == block.get("order") == block.get("completed") == labels
         and block.get("status") == "COMPLETE" and block.get("error") is None
         and start.get("runner_source_sha256") == block.get("runner_source_sha256") == WRAPPER_SHA
         and start.get("cell_source_sha256") == CELL_SHA
         and start.get("input_file_sha256") == INPUT_SHA
         and start.get("batch_tokens") == 4096 and start.get("arrival_scale") == 1.0,
         "four-cell batch4096 block incomplete or source/input differs")
    pilot_gate = start.get("pilot_gate")
    need(type(pilot_gate) is dict
         and len(pilot_gate.get("pilot_receipt_sha256", "")) == 64
         and len(pilot_gate.get("pilot_gate_sha256", "")) == 64,
         "pilot qualification provenance absent")
    raws, values, cells = [], [], []
    for index, (label, arm) in enumerate(ORDER):
        raw, value, row = qualified_cell(root, index, label, arm, pilot_gate)
        if values:
            goodput.pair(values[0], value)  # Exact request/prompt/arrival/cap identity.
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

    return dict(schema="c-native-past-future-ae-matched-analysis-v1",
        status="COMPLETE_DESCRIPTIVE_DEVELOPMENT_ONLY", order=labels,
        input_file_sha256=INPUT_SHA, batch_tokens=4096,
        pilot_gate_provenance=pilot_gate, cells=cells,
        past_future_policy_summaries={cells[i]["label"]: cells[i]["policy_summary"]
                                      for i in (1, 2)},
        native_to_past_future_adjacent_pairs=[comparison(a, b) for a, b in ((0, 1), (3, 2))],
        within_arm_repeats={"native_full": comparison(0, 3),
                            "native_past_future_ae": comparison(1, 2)},
        limits="Two runs per arm on one viewed development cohort and one batch4096 GPU regime. "
               "Different output IDs and lengths preclude equal-work speedup. "
               "The native port scores each co-batched decode/prefill call; it does not reproduce "
               "LightLLM's separate prefill/merge or max_wait_tokens=2 router cadence. "
               "Host-return gaps omit chunk interiors; this is not unseen confirmation or full LightLLM reproduction.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = analyze(args.root)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
