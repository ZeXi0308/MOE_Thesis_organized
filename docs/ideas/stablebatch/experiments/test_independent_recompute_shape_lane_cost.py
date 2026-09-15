from __future__ import annotations

import copy
import json
from pathlib import Path
import statistics
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import independent_recompute_shape_lane_cost as recompute


def zero_mismatch_fixture() -> dict[str, dict[str, int]]:
    return {
        arm: {
            "raw_repeat_mismatches": 0,
            "route_repeat_mismatches": 0,
            "final_logits_repeat_mismatches": 0,
        }
        for arm in recompute.ARMS
    }


class IndependentGateTests(unittest.TestCase):
    def test_frozen_ratio_boundaries_and_correctness_precedence(self) -> None:
        metrics = {
            recompute.ARM_NATIVE: {
                "median_total_expert_gpu_ms": 4.0,
                "median_token_step_p99_ms": 10.0,
            },
            recompute.ARM_SERIAL: {
                "median_total_expert_gpu_ms": 10.0,
                "median_token_step_p99_ms": 20.0,
            },
            recompute.ARM_C8: {
                "median_total_expert_gpu_ms": 8.0,
                "median_token_step_p99_ms": 10.5,
            },
        }
        result = recompute.classify_independent(
            zero_mismatch_fixture(), metrics, recompute.FROZEN_GATE
        )
        self.assertEqual(result["verdict"], recompute.FROZEN_GATE["pass"])

        cost_metrics = copy.deepcopy(metrics)
        cost_metrics[recompute.ARM_C8]["median_total_expert_gpu_ms"] = 8.000001
        result = recompute.classify_independent(
            zero_mismatch_fixture(), cost_metrics, recompute.FROZEN_GATE
        )
        self.assertEqual(result["verdict"], recompute.FROZEN_GATE["cost_fail"])

        correctness = zero_mismatch_fixture()
        correctness[recompute.ARM_C8]["route_repeat_mismatches"] = 1
        result = recompute.classify_independent(
            correctness, cost_metrics, recompute.FROZEN_GATE
        )
        self.assertEqual(result["verdict"], recompute.FROZEN_GATE["correctness_fail"])

    def test_two_measured_replays_use_frozen_median_and_aggregate_formulas(self) -> None:
        replays = {}
        for arm_index, arm in enumerate(recompute.ARMS):
            for repeat in (0, 1):
                base = float(10 * arm_index + 2 * repeat + 1)
                ledger_row = {"arm": arm, "phase": "measured", "repeat": repeat}
                replays[(arm, "measured", repeat)] = {
                    "total_expert_gpu_ms": base,
                    "total_whole_step_wall_ms": base + 10.0,
                    "token_step_p99_ms": base + 20.0,
                    "mean_frozen_token_nll": base + 30.0,
                    "kernel_calls": repeat + 1,
                    "real_rows": 4,
                    "dummy_rows": repeat,
                    "occupied_experts": 3,
                    "natural_m_histogram": {"2": 3},
                    "ledger_row": ledger_row,
                }
        metrics = recompute.summarize_measured_arms(replays)
        native = metrics[recompute.ARM_NATIVE]
        self.assertEqual(native["median_total_expert_gpu_ms"], 2.0)
        self.assertEqual(native["kernel_calls"], 3)
        self.assertEqual(native["real_rows"], 8)
        self.assertEqual(native["dummy_rows"], 1)
        self.assertEqual(native["natural_m_histogram"], {"2": 6})


class IndependentIntegrityTests(unittest.TestCase):
    def test_batch_digest_rebuild_accepts_producer_rank_major_call_rows(self) -> None:
        request_ids = ["r0", "r1"]
        decode_steps = [0, 0]
        roster = {
            "request_ids": request_ids,
            "decode_steps": decode_steps,
            "input_token_ids": [10, 20],
            "native_predicted_next_token_ids": [11, 21],
            "position_ids": [3, 4],
            "native_route_membership_sha256": "d" * 64,
            "native_final_logits_sha256": "e" * 64,
        }

        def make_call(expert: int, positions: list[tuple[int, int]]) -> dict[str, object]:
            slot_ids = [
                f"{request_ids[token]}:decode:000000:layer:00:topk:{rank}"
                for token, rank in positions
            ]
            row_ids = [f"{slot}:expert:{expert:02d}" for slot in slot_ids]
            return {
                "schema_version": recompute.CALL_SCHEMA,
                "arm": recompute.ARM_NATIVE,
                "phase": "measured",
                "repeat": 0,
                "batch_index": 0,
                "layer": 0,
                "expert_id": expert,
                "logical_m": 2,
                "physical_m": 2,
                "physical_m_per_kernel": [2],
                "padding_rows": 0,
                "kernel_calls": 1,
                "slot_ids": slot_ids,
                "row_ids": row_ids,
                "route_membership_sha256": recompute.canonical_sha256(row_ids),
                "raw_bf16_sha256": f"{expert + 1:x}" * 64,
                "raw_row_sha256": [
                    {"slot_id": slot, "row_id": row, "sha256": "a" * 64}
                    for slot, row in zip(slot_ids, row_ids)
                ],
            }

        calls = [
            make_call(0, [(0, 0), (1, 1)]),
            # torch.where over [topk_rank, token] yields this rank-major order.
            make_call(1, [(1, 0), (0, 1)]),
        ]
        route_entries = []
        for token, ranked_experts in enumerate(((0, 1), (1, 0))):
            for rank, expert in enumerate(ranked_experts):
                slot = (
                    f"{request_ids[token]}:decode:000000:layer:00:topk:{rank}"
                )
                route_entries.append({
                    "layer": 0,
                    "token_index": token,
                    "request_id": request_ids[token],
                    "decode_step": 0,
                    "topk_rank": rank,
                    "expert_id": expert,
                    "slot_id": slot,
                    "row_id": f"{slot}:expert:{expert:02d}",
                })
        raw_payload = [
            {
                "layer": call["layer"],
                "expert_id": call["expert_id"],
                "logical_m": call["logical_m"],
                "physical_m": call["physical_m"],
                "kernel_calls": call["kernel_calls"],
                "row_ids": call["row_ids"],
                "raw_bf16_sha256": call["raw_bf16_sha256"],
                "raw_row_sha256": call["raw_row_sha256"],
            }
            for call in calls
        ]
        roster_identity = {
            "batch_index": 0,
            "request_ids": request_ids,
            "decode_steps": decode_steps,
            "input_token_ids": [10, 20],
            "position_ids": [3, 4],
        }
        nll = [0.1, 0.2]
        step = {
            "schema_version": recompute.STEP_SCHEMA,
            "arm": recompute.ARM_NATIVE,
            "phase": "measured",
            "repeat": 0,
            "batch_index": 0,
            "request_ids": request_ids,
            "decode_steps": decode_steps,
            "input_token_ids": [10, 20],
            "teacher_forced_target_token_ids": [11, 21],
            "frozen_token_nll": nll,
            "whole_step_wall_ms": 2.0,
            "expert_stage_gpu_ms": 1.25,
            "timing_boundaries": {
                "whole_step": "cuda_sync_before_model_to_logits_ready_cuda_sync",
                "expert_stage": (
                    "sum_of_layer_cuda_events_after_router_topk_through_"
                    "dispatch_padding_expert_and_index_add"
                ),
            },
            "native_roster_route_membership_sha256": "d" * 64,
            "native_roster_final_logits_sha256": "e" * 64,
            "roster_identity_sha256": recompute.canonical_sha256(roster_identity),
            "raw_calls_sha256": recompute.canonical_sha256(raw_payload),
            "route_membership_sha256": recompute.canonical_sha256(route_entries),
            "final_logits_sha256": "f" * 64,
            "greedy_token_ids": [11, 21],
            "mean_frozen_token_nll": statistics.fmean(nll),
        }
        stage = {
            "schema_version": recompute.STAGE_SCHEMA,
            "arm": recompute.ARM_NATIVE,
            "phase": "measured",
            "repeat": 0,
            "batch_index": 0,
            "layer": 0,
            "expert_stage_gpu_ms": 1.25,
            "occupied_experts": 2,
            "timing_boundary": (
                "after_router_topk_through_dispatch_padding_expert_and_index_add"
            ),
        }
        result = recompute.reconstruct_batch_evidence(
            key=(recompute.ARM_NATIVE, "measured", 0),
            batch_index=0,
            step=step,
            roster=roster,
            stages=[stage],
            calls=calls,
            num_layers=1,
            num_experts=2,
            top_k=2,
        )
        self.assertEqual(result["route_rows"], 4)
        self.assertEqual(result["real_rows"], 4)
        self.assertEqual(result["signature"]["raw_calls_sha256"], step["raw_calls_sha256"])

    def test_manifest_size_and_hash_are_checked_without_writes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = root / "evidence.json"
            evidence.write_text(json.dumps({"sealed": True}) + "\n", encoding="utf-8")
            manifest = {
                "schema_version": recompute.MANIFEST_SCHEMA,
                "required_artifacts": ["evidence.json"],
                "files": {
                    "evidence.json": {
                        "size_bytes": evidence.stat().st_size,
                        "sha256": recompute.sha256_file(evidence),
                    }
                },
            }
            before = sorted(path.name for path in root.iterdir())
            report = recompute.verify_manifest(
                root, manifest, required_names=("evidence.json",)
            )
            after = sorted(path.name for path in root.iterdir())
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(before, after)

            evidence.write_text(json.dumps({"sealed": False}) + "\n", encoding="utf-8")
            with self.assertRaises(recompute.RecomputeError):
                recompute.verify_manifest(
                    root, manifest, required_names=("evidence.json",)
                )


if __name__ == "__main__":
    unittest.main()
