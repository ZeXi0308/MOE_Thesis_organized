#!/usr/bin/env python3
"""Pure-CPU protocol tests for the source-localization runner."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "source_localization_under_test", HERE / "run_source_localization.py"
)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNNER
SPEC.loader.exec_module(RUNNER)


class SourceLocalizationProtocolTest(unittest.TestCase):
    def test_identity_and_alignment(self) -> None:
        config = json.loads((HERE / "events.json").read_text(encoding="utf-8"))
        event = copy.deepcopy(config["events"][0])
        batch_index = event["batch_index"]
        target = event["target_request_id"]
        step = event["decode_step"]
        batch_rows = [{"batch_index": index} for index in range(batch_index + 1)]
        batch_rows[batch_index] = {
            "batch_index": batch_index,
            "request_ids": event["original_batch_request_ids"],
            "decode_steps": [step] * len(event["original_batch_request_ids"]),
        }
        ledger_steps = [{} for _ in range(step + 1)]
        ledger_steps[step] = {"decode_step": step, "batch_index": batch_index}
        route_index = {}
        serial_examples = []
        for difference in event["recorded_difference_examples"]:
            layer = difference["layer"]
            serial_examples.append(
                {
                    "request_id": target,
                    "decode_step": step,
                    "layer": layer,
                    "batched_experts": difference["batched_experts"],
                    "serial_experts": difference["serial_experts"],
                }
            )
            first = event["source_rows"]["routes_csv_rows_by_layer"][str(layer)][0]
            route_index[(target, step, layer)] = [
                {
                    "topk_slot": slot,
                    "expert_id": expert,
                    "_source_line": first + slot,
                }
                for slot, expert in enumerate(difference["batched_experts"])
            ]
        capture = {
            "batch_rows": batch_rows,
            "ledger": {target: {"steps": ledger_steps}},
            "ledger_lines": {
                target: event["source_rows"]["request_ledger_jsonl_line"]
            },
            "serial_audit": {"difference_examples": serial_examples},
            "route_index": route_index,
        }
        aligned = RUNNER.validate_event_alignment(event, capture)
        self.assertEqual(
            aligned["recorded_difference_layers"],
            [row["layer"] for row in event["recorded_difference_examples"]],
        )
        broken = copy.deepcopy(capture)
        broken["batch_rows"][batch_index]["request_ids"] = list(
            reversed(event["original_batch_request_ids"])
        )
        with self.assertRaises(RUNNER.ProtocolError):
            RUNNER.validate_event_alignment(event, broken)

    def test_replay_plan_never_reads_event_or_future(self) -> None:
        rows = [{"batch_index": index} for index in range(5)]
        plan = RUNNER.build_replay_plan(rows, 3)
        self.assertEqual([row["batch_index"] for row in plan], [0, 1, 2])
        self.assertTrue(all(row["batch_index"] < 3 for row in plan))

        source = (HERE / "run_source_localization.py").read_text(encoding="utf-8")
        function = source.split("def _run_capture_events(", 1)[1].split(
            "def _write_json_exclusive", 1
        )[0]
        event_loop = function.split("for event in events:", 1)[1]
        prefill = event_loop.find("canonical = _prefill_states(")
        replay = event_loop.find("while cursor < event_batch:")
        execute = event_loop.find("_run_event(")
        self.assertGreaterEqual(prefill, 0)
        self.assertGreater(replay, prefill)
        self.assertGreater(execute, replay)

    def test_arm_isolation_and_deterministic_shuffle(self) -> None:
        original = ["r0", "target", "r2", "r3"]
        arms = RUNNER.deterministic_arm_orders(original, "target")
        self.assertEqual(arms["A_serial"], ["target"])
        self.assertEqual(arms["B_original"], original)
        self.assertEqual(arms["C_shuffled"].index("target"), 1)
        self.assertEqual(set(arms["C_shuffled"]), set(original))
        self.assertEqual(arms["C_shuffled"], ["r3", "target", "r2", "r0"])

    def test_frozen_classification_logic_and_scope(self) -> None:
        base = {
            "target_state_identical": True,
            "original_arm_reproduced": True,
            "within_arm_stable": True,
            "pairwise_repeat_consistent": True,
            "ab_assignment_changed": True,
            "bc_assignment_changed": False,
            "ab_first_pre_router_hidden_layer": None,
            "ab_first_router_logit_layer": 4,
            "ab_first_divergence_signal": "router_logits",
            "ab_first_divergence_layer": 4,
            "ab_router_candidate_pre_hidden_exact_digest_equal": True,
            "ab_near_tie_concentrated_crossing": True,
        }
        localized = RUNNER.classify_facts(base)
        self.assertEqual(
            localized["frozen_classification"],
            ["ROUTER_KERNEL_SHAPE_EFFECT", "NEAR_TIE_AMPLIFICATION"],
        )
        self.assertEqual(
            localized["secondary_findings"], ["WIDTH_VS_COMPANION_CONTEXT_UNRESOLVED"]
        )
        self.assertEqual(
            localized["scope"]["physical_shape_effect"],
            "NOT_IDENTIFIED_BY_SHUFFLE_ONLY_C",
        )
        self.assertEqual(localized["scope"]["companion_identity_externality"], "NOT_TESTED")
        row_order = RUNNER.classify_facts({**base, "bc_assignment_changed": True})
        self.assertEqual(
            row_order["secondary_findings"], ["COMPANION_ROW_ORDER_OR_LAYOUT_EFFECT"]
        )
        self.assertNotIn("COMPANION_IDENTITY_EXTERNALITY", str(row_order))
        upstream = RUNNER.classify_facts(
            {
                **base,
                "ab_first_pre_router_hidden_layer": 2,
                "ab_first_router_logit_layer": 2,
                "ab_first_divergence_signal": "pre_router_hidden",
                "ab_first_divergence_layer": 2,
            }
        )
        self.assertIn("UPSTREAM_BATCH_CONTEXT_EFFECT", upstream["frozen_classification"])
        unstable = RUNNER.classify_facts({**base, "within_arm_stable": False})
        self.assertEqual(unstable["frozen_classification"], ["NONDETERMINISTIC_RUNTIME"])
        not_reproduced = RUNNER.classify_facts({**base, "ab_assignment_changed": False})
        self.assertEqual(not_reproduced["frozen_classification"], ["NOT_REPRODUCED"])
        pairwise_unstable = RUNNER.classify_facts(
            {**base, "pairwise_repeat_consistent": False}
        )
        self.assertEqual(
            pairwise_unstable["status"], "STOP_PAIRWISE_REPEAT_INCONSISTENT"
        )
        hidden_delta = RUNNER.classify_facts(
            {**base, "ab_router_candidate_pre_hidden_exact_digest_equal": False}
        )
        self.assertEqual(
            hidden_delta["status"],
            "STABLE_DIVERGENCE_INPUT_DELTA_VS_KERNEL_UNRESOLVED",
        )
        self.assertEqual(
            hidden_delta["frozen_classification"],
            ["UNRESOLVED_INPUT_DELTA_VS_KERNEL"],
        )

    def test_cross_signal_repeat_and_profile_qualification(self) -> None:
        def record(
            repeat: int,
            *,
            hidden: float,
            router: tuple[float, float, float],
            selected: list[int],
            margin: float,
        ) -> dict:
            def tensor(values: tuple[float, ...] | list[float]) -> dict:
                floats = [float(value) for value in values]
                return {
                    "dtype_before_float32_copy": "torch.bfloat16",
                    "shape": [len(floats)],
                    "sha256_float32": json.dumps(floats),
                    "values": floats,
                }

            return {
                "repeat": repeat,
                "signals": {
                    "pre_router_hidden": {"0": tensor([hidden])},
                    "router_logits": {"0": tensor(router)},
                    "topk_margin": {"0": margin},
                    "selected_experts": {"0": selected},
                    "combined_expert_output": {"0": tensor([hidden])},
                    "next_token_logits": tensor([0.0, 1.0]),
                },
            }

        left = [
            record(index, hidden=0.0, router=(1.0, 0.999, 0.998), selected=[0, 1], margin=0.001)
            for index in range(3)
        ]
        right = [
            record(index, hidden=0.0, router=(0.999, 1.0, 0.998), selected=[1, 2], margin=0.002)
            for index in range(3)
        ]
        paired = RUNNER._compare_repeat_pairs(left, right)
        self.assertTrue(paired["repeat_consistent"])
        self.assertTrue(paired["all_repeats_assignment_changed"])
        consensus = paired["consensus"]
        self.assertEqual(
            consensus["first_cross_signal_divergence"], {"signal": "router_logits", "layer": 0}
        )
        self.assertTrue(consensus["router_candidate_pre_hidden_exact_digest_equal"])
        self.assertTrue(consensus["near_tie_concentrated_crossing"])

        allclose_only_right = [
            record(
                index,
                hidden=5e-7,
                router=(0.999, 1.0, 0.998),
                selected=[1, 2],
                margin=0.002,
            )
            for index in range(3)
        ]
        allclose_only = RUNNER._compare_repeat_pairs(left, allclose_only_right)
        self.assertTrue(allclose_only["repeat_consistent"])
        self.assertEqual(
            allclose_only["consensus"]["first_cross_signal_divergence"],
            {"signal": "router_logits", "layer": 0},
        )
        self.assertFalse(
            allclose_only["consensus"]["router_candidate_pre_hidden_exact_digest_equal"]
        )

        def event(event_id: str, regime: str, source_class: str) -> dict:
            return {
                "event_id": event_id,
                "arrival_regime": regime,
                "summary": {
                    "classification": {
                        "status": "LOCALIZED_DEVELOPMENT_SIGNAL",
                        "frozen_classification": [source_class],
                    }
                },
            }

        consistent = RUNNER.summarize_profile_results(
            [
                event("s", "steady", "ROUTER_KERNEL_SHAPE_EFFECT"),
                event("b", "bursty", "ROUTER_KERNEL_SHAPE_EFFECT"),
            ]
        )
        self.assertTrue(consistent["continue_gate_passed"])
        mixed = RUNNER.summarize_profile_results(
            [
                event("s", "steady", "ROUTER_KERNEL_SHAPE_EFFECT"),
                event("b", "bursty", "UPSTREAM_BATCH_CONTEXT_EFFECT"),
            ]
        )
        self.assertFalse(mixed["continue_gate_passed"])


if __name__ == "__main__":
    unittest.main()
