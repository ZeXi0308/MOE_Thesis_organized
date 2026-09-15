from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

try:
    from . import smoke_multirank_trace_gpu as subject
except ImportError:  # direct unittest discovery
    import smoke_multirank_trace_gpu as subject


def _plan() -> dict:
    contributions = []
    for slot in range(2):
        route_message_id = subject._object_sha256({"slot": slot})
        contributions.append(
            {
                "message_id": route_message_id,
                "payload_bytes": 16,
                "topk_slot": slot,
                "sender_rank": slot,
                "receiver_rank": 1,
                "expert_id": slot,
                "route_weight": 0.5,
            }
        )
    plan = {
        "schema_version": "multirank-native-wave-plan-v1",
        "status": "FROZEN_PHYSICAL_INPUT_NOT_EXECUTION_RESULT",
        "model": "test",
        "waves": [
            {
                "wave_id": "wave",
                "request_id": "request",
                "layer_id": 0,
                "token_position": 0,
                "top_k": 2,
                "hidden": 8,
                "intermediate": 4,
                "contributions": contributions,
            }
        ],
    }
    plan["artifact_sha256"] = subject._object_sha256(plan)
    return plan


def _records(plan: dict) -> list[dict]:
    rows = []
    for slot, contribution in enumerate(plan["waves"][0]["contributions"]):
        start = 10 + slot * 20
        rows.append(
            {
                "message_id": subject.execution_message_id(
                    contribution["message_id"], credit_b=1, measured_index=0
                ),
                "route_message_id": contribution["message_id"],
                "wave_id": "wave",
                "credit_b": 1,
                "measured_index": 0,
                "payload_bytes": 16,
                "transport": "LOCAL_CLONE",
                "expert_start_ns": start,
                "expert_ready_ns": start + 1,
                "credit_recv_ns": start + 1,
                "send_start_ns": start + 2,
                "send_end_ns": start + 3,
                "recv_visible_ns": start + 3,
                "unpack_start_ns": start + 4,
                "unpack_end_ns": start + 5,
                "join_close_ns": 35,
            }
        )
    return rows


class TraceSmokeTests(unittest.TestCase):
    def test_plan_self_hash_and_records_pass(self) -> None:
        plan = _plan()
        subject.validate_plan(plan)
        subject.validate_records(_records(plan), plan)

    def test_chronology_violation_is_rejected(self) -> None:
        plan = _plan()
        rows = _records(plan)
        rows[0]["recv_visible_ns"] = rows[0]["send_start_ns"] - 1
        with self.assertRaises(subject.TraceSmokeError):
            subject.validate_records(rows, plan)

    def test_missing_sibling_is_rejected(self) -> None:
        plan = _plan()
        with self.assertRaises(subject.TraceSmokeError):
            subject.validate_records(_records(plan)[:-1], plan)

    def test_payload_mismatch_is_rejected(self) -> None:
        plan = _plan()
        rows = _records(plan)
        rows[0]["payload_bytes"] += 2
        with self.assertRaises(subject.TraceSmokeError):
            subject.validate_records(rows, plan)

    def test_join_close_must_equal_last_unpack(self) -> None:
        plan = _plan()
        rows = _records(plan)
        rows[0]["join_close_ns"] -= 1
        with self.assertRaises(subject.TraceSmokeError):
            subject.validate_records(rows, plan)

    def test_writer_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            output = Path(value) / "trace.json"
            subject.write_once(output, {"value": 1})
            with self.assertRaises(subject.TraceSmokeError):
                subject.write_once(output, {"value": 2})

    def test_execution_message_identity_changes_by_cell_and_trial(self) -> None:
        values = {
            subject.execution_message_id("route", credit_b=credit_b, measured_index=index)
            for credit_b in (1, 2, 4, 8)
            for index in range(3)
        }
        self.assertEqual(len(values), 12)


if __name__ == "__main__":
    unittest.main()
