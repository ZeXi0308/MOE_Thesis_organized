from __future__ import annotations

import unittest

try:
    from . import run_multirank_rr_census as subject
except ImportError:  # direct unittest discovery
    import run_multirank_rr_census as subject


def _plan() -> dict:
    contributions = []
    for slot, sender in enumerate((0, 0, 1, 2)):
        contributions.append(
            {
                "message_id": f"route-{slot}",
                "topk_slot": slot,
                "sender_rank": sender,
                "receiver_rank": 3,
                "expert_id": slot,
                "payload_bytes": 16,
            }
        )
    return {
        "waves": [
            {
                "wave_id": "wave",
                "request_id": "request",
                "layer_id": 2,
                "token_position": 3,
                "top_k": 4,
                "contributions": contributions,
            }
        ]
    }


def _halves() -> list[dict]:
    values = []
    close = 93
    for slot in range(4):
        message_id = f"message-{slot}"
        common = {
            "run_id": "run",
            "wave_id": "wave",
            "route_message_id": f"route-{slot}",
            "message_id": message_id,
            "credit_b": 2,
            "trial_kind": "measured",
            "trial_index": 0,
            "payload_bytes": 16,
            "physical_sender_rank": slot % 3,
            "physical_receiver_rank": 3,
        }
        values.append(
            {
                **common,
                "event_side": "sender",
                "expert_start_ns": 10 + slot,
                "expert_ready_ns": 20 + slot,
                "credit_recv_ns": 30 + slot,
                "send_start_ns": 40 + slot,
                "send_end_ns": 50 + slot,
            }
        )
        values.append(
            {
                **common,
                "event_side": "receiver",
                "recv_visible_ns": 60 + slot,
                "unpack_start_ns": 70 + slot,
                "unpack_end_ns": 90 + slot,
                "join_close_ns": close,
            }
        )
    return values


class MultiRankRuntimeTests(unittest.TestCase):
    def test_physical_rank_mapping(self) -> None:
        self.assertEqual(subject.physical_rank(7, 8), 7)
        self.assertEqual(subject.physical_rank(7, 4), 3)
        with self.assertRaises(subject.MultiRankRuntimeError):
            subject.physical_rank(0, 2)

    def test_rr_order_rotates_senders(self) -> None:
        order = subject.rr_order(_plan()["waves"][0]["contributions"], 4)
        self.assertEqual([row["sender_rank"] for row in order], [0, 1, 2, 0])

    def test_merge_requires_exact_halves_and_conservation(self) -> None:
        merged = subject.merge_event_halves(_plan(), _halves())
        self.assertEqual(len(merged), 4)
        self.assertTrue(all(row["join_close_ns"] == 93 for row in merged))
        self.assertTrue(all(row["request_id"] == "request" for row in merged))

    def test_recv_may_be_visible_before_sender_wait_returns(self) -> None:
        rows = _halves()
        for row in rows:
            if row["event_side"] == "receiver":
                row["recv_visible_ns"] = 45
                row["unpack_start_ns"] = 70
        merged = subject.merge_event_halves(_plan(), rows)
        self.assertEqual(len(merged), 4)

    def test_merge_rejects_missing_receiver(self) -> None:
        rows = _halves()
        rows.pop()
        with self.assertRaises(subject.MultiRankRuntimeError):
            subject.merge_event_halves(_plan(), rows)

    def test_merge_rejects_duplicate_sender(self) -> None:
        rows = _halves()
        rows.append(dict(rows[0]))
        with self.assertRaises(subject.MultiRankRuntimeError):
            subject.merge_event_halves(_plan(), rows)

    def test_execution_id_changes_across_warmup_and_measured(self) -> None:
        warmup = subject.execution_message_id(
            "route", credit_b=1, trial_kind="warmup", trial_index=0
        )
        measured = subject.execution_message_id(
            "route", credit_b=1, trial_kind="measured", trial_index=0
        )
        self.assertNotEqual(warmup, measured)


if __name__ == "__main__":
    unittest.main()
