from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from docs.ideas.route_shape_slo.experiments.build_route_windows import (
    ProtocolError,
    build_stablebatch_capture,
)


class StableBatchAdapterTest(unittest.TestCase):
    def _write_fixture(self, root: Path, *, duplicate_slot: bool = False) -> None:
        (root / "RUN_STATUS.json").write_text(
            json.dumps(
                {
                    "schema_version": "stablebatch-shape-lane-cost-run-status-v1",
                    "status": "COMPLETE",
                    "serving_result": False,
                }
            ),
            encoding="utf-8",
        )
        (root / "COMPLETE.json").write_text(
            json.dumps(
                {
                    "schema_version": "stablebatch-shape-lane-cost-complete-v1",
                    "status": "COMPLETE",
                }
            ),
            encoding="utf-8",
        )
        (root / "config_snapshot.json").write_text(
            json.dumps(
                {
                    "schema_version": "stablebatch-shape-lane-continuous-cost-gate-v1",
                    "execution": {
                        "measured_arm_orders": [
                            ["native_variable_m", "serial_m1", "fixed_c8"]
                        ]
                    },
                    "model": {
                        "repo_id": "fixture/moe",
                        "revision": "frozen",
                        "num_experts": 4,
                        "num_experts_per_tok": 2,
                        "num_hidden_layers": 2,
                    },
                    "workload": {
                        "expected_requests": 4,
                        "expected_decode_steps_per_request": 2,
                        "expected_request_steps": 8,
                        "max_batch_size": 2,
                    },
                }
            ),
            encoding="utf-8",
        )
        requests = [
            {
                "request_id": f"r{index}",
                "document_id": f"d{index}",
                "prompt_token_count": 5,
            }
            for index in range(4)
        ]
        (root / "workload_snapshot.json").write_text(
            json.dumps(
                {
                    "schema": "bcrd-continuous-workload-v1",
                    "model": {"id": "fixture/moe", "revision": "frozen"},
                    "requests": requests,
                }
            ),
            encoding="utf-8",
        )

        roster_rows = []
        step_rows = []
        call_rows = []
        for batch_index in range(4):
            cohort, step = divmod(batch_index, 2)
            request_ids = [f"r{2 * cohort}", f"r{2 * cohort + 1}"]
            route_hash = f"route-{batch_index}"
            roster_rows.append(
                {
                    "schema_version": "stablebatch-shape-lane-native-roster-row-v1",
                    "batch_index": batch_index,
                    # Deliberately includes future/cohort-external requests: it
                    # must never become a serving running/queue counter.
                    "active_request_ids": ["r0", "r1", "r2", "r3"],
                    "request_ids": request_ids,
                    "decode_steps": [step, step],
                    "prior_cache_lengths": [5 + step, 5 + step],
                    "native_route_membership_sha256": route_hash,
                }
            )
            step_rows.append(
                {
                    "schema_version": "stablebatch-shape-lane-decode-step-v1",
                    "arm": "native_variable_m",
                    "phase": "measured",
                    "repeat": 0,
                    "batch_index": batch_index,
                    "request_ids": request_ids,
                    "decode_steps": [step, step],
                    "route_membership_sha256": route_hash,
                    "whole_step_wall_ms": 10.0 + batch_index,
                }
            )
            for layer in range(2):
                for expert in range(2):
                    slot_ids = [
                        f"{request_id}:decode:{step:06d}:layer:{layer:02d}:topk:{expert}"
                        for request_id in request_ids
                    ]
                    call_rows.append(
                        {
                            "schema_version": "stablebatch-shape-lane-expert-call-v1",
                            "arm": "native_variable_m",
                            "phase": "measured",
                            "repeat": 0,
                            "batch_index": batch_index,
                            "layer": layer,
                            "expert_id": expert,
                            "logical_m": 2,
                            "row_ids": [
                                f"{slot_id}:expert:{expert:02d}"
                                for slot_id in slot_ids
                            ],
                            "slot_ids": slot_ids,
                        }
                    )
        if duplicate_slot:
            call_rows[1]["slot_ids"][0] = call_rows[0]["slot_ids"][0]
        with (root / "native_roster.jsonl").open("w", encoding="utf-8") as handle:
            for row in roster_rows:
                handle.write(json.dumps(row) + "\n")
        with (root / "decode_step_ledger.jsonl").open("w", encoding="utf-8") as handle:
            for row in step_rows:
                handle.write(json.dumps(row) + "\n")
        with (root / "expert_call_ledger.jsonl").open("w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "arm": "native_variable_m",
                        "phase": "warmup",
                        "repeat": 0,
                    }
                )
                + "\n"
            )
            for row in call_rows:
                handle.write(json.dumps(row) + "\n")
            # This closes the selected contiguous block. The invalid tail must
            # not be parsed, proving that the 1.2 GiB source is streamed only
            # through the selected block.
            handle.write(json.dumps({"arm": "serial_m1", "phase": "measured", "repeat": 0}) + "\n")
            handle.write("not-json-and-must-not-be-read\n")

    def test_builds_request_disjoint_nonserving_proxy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_fixture(root)
            rows, metadata = build_stablebatch_capture(root)

        self.assertEqual(len(rows), 4)
        self.assertEqual(len({row["episode_id"] for row in rows}), 2)
        self.assertTrue(all(row["running_sequences"] == 2 for row in rows))
        self.assertTrue(all(row["queue_depth"] == 0 for row in rows))
        self.assertTrue(
            all(
                row["queue_depth_semantics"]
                == "NOT_OBSERVED_SENTINEL_ZERO_EXCLUDED_FROM_CLAIMS"
                for row in rows
            )
        )
        self.assertTrue(
            all(row["evidence_type"] == "[Observed isolated GPU primitive]" for row in rows)
        )
        self.assertTrue(all(row["runtime_representative"] == "false" for row in rows))
        self.assertTrue(all(row["top_k"] == 2 for row in rows))
        self.assertEqual(rows[0]["route_max_mean"], 2.0)
        self.assertFalse(metadata["serving_queue_observed"])
        self.assertFalse(metadata["slo_observed"])
        self.assertFalse(metadata["gate_weight_observed"])

    def test_rejects_cross_expert_duplicate_route_slot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_fixture(root, duplicate_slot=True)
            with self.assertRaisesRegex(ProtocolError, "duplicated across experts"):
                build_stablebatch_capture(root)


if __name__ == "__main__":
    unittest.main()
