from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from . import compile_multirank_wave_plan as subject


def _rows(model: str, wave_index: int) -> list[dict]:
    spec = subject.MODEL_SPECS[model]
    receiver = wave_index % spec["virtual_ep"]
    rows = []
    for slot in range(spec["top_k"]):
        rows.append(
            {
                "model_key": model,
                "model_revision": f"{model}@test",
                "request_id": f"request:{wave_index // 2}",
                "forward_id": f"forward:{wave_index // 2}",
                "phase": "prefill",
                "decode_step": 0,
                "layer_id": wave_index // 4,
                "token_position": wave_index,
                "token_id": f"token:{wave_index}",
                "topk_slot": slot,
                "expert_id": slot,
                "sender_rank": slot % spec["virtual_ep"],
                "receiver_rank": receiver,
                "route_weight": 1.0 / spec["top_k"],
                "native_route_tuple_sha256": hashlib.sha256(
                    f"native:{model}:{wave_index}".encode()
                ).hexdigest(),
                "valid": True,
            }
        )
    return rows


class CompileWavePlanTests(unittest.TestCase):
    def _source(self, root: Path, model: str, waves: int) -> Path:
        path = root / "route.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for wave_index in range(waves):
                for row in _rows(model, wave_index):
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
        return path

    def test_hash_selection_is_deterministic_and_timing_blind(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            source = self._source(Path(value), "olmoe", 12)
            first = subject.compile_plan(source, "olmoe", 5)
            second = subject.compile_plan(source, "olmoe", 5)
            self.assertEqual(first, second)
            self.assertEqual(first["selected_wave_count"], 5)
            self.assertFalse(first["selection_contract"]["uses_sender_multiplicity"])
            self.assertFalse(first["selection_contract"]["uses_timing_or_policy_outcome"])
            observed = [wave["selection_hash"] for wave in first["waves"]]
            self.assertEqual(observed, sorted(observed))

    def test_complete_topk_and_message_identity(self) -> None:
        wave = subject.compile_wave(_rows("llmjp", 3), "llmjp")
        self.assertEqual(len(wave["contributions"]), 16)
        self.assertEqual(
            [row["topk_slot"] for row in wave["contributions"]], list(range(16))
        )
        self.assertEqual(len({row["message_id"] for row in wave["contributions"]}), 16)
        self.assertTrue(all(row["payload_bytes"] == 1024 for row in wave["contributions"]))

    def test_incomplete_wave_is_rejected(self) -> None:
        with self.assertRaises(subject.WavePlanError):
            subject.compile_wave(_rows("olmoe", 0)[:-1], "olmoe")

    def test_noncontiguous_repeated_wave_is_rejected(self) -> None:
        rows = _rows("olmoe", 0) + _rows("olmoe", 1) + _rows("olmoe", 0)
        with self.assertRaises(subject.WavePlanError):
            list(subject.iter_contiguous_waves(rows))

    def test_atomic_writer_refuses_overwrite_and_self_hash_is_valid(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            plan = subject.compile_plan(self._source(root, "olmoe", 3), "olmoe", 2)
            expected = plan["artifact_sha256"]
            payload = dict(plan)
            payload.pop("artifact_sha256")
            self.assertEqual(expected, subject._object_sha256(payload))
            output = root / "plan.json"
            subject.write_once(output, plan)
            self.assertTrue(output.exists())
            with self.assertRaises(subject.WavePlanError):
                subject.write_once(output, plan)


if __name__ == "__main__":
    unittest.main()

