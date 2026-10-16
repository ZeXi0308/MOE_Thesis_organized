"""CPU checks for joining independently captured graph performance/counters."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench_graph import pair_rows, validate_snapshot


def fixture(counted, spec=None):
    row = {"repeat": 0, "batch": 2, "arm": "native", "execution": "graph",
           "graph_counter_pass": counted, "spec": spec, "decode_output_tokens": 6,
           "output_tokens_sha256": "same", "logical_counters":
           {"events": 6., "tokens": 12.} if counted else None}
    snapshot = {"spec": spec, "phase_steps": {"prefill": 1, "decode": 3},
        "execution": {f"layer{i}": {"original_top_k": 8, "current_router_top_k": 8} for i in range(2)},
        "layers": {"totals": row["logical_counters"]},
        "graph": {"fixed_arm": True, "compilation_mode": 0,
            "cudagraph_mode": "FULL_DECODE_ONLY", "count_routes": counted,
            "captured_token_buckets": [1, 2], "runtime_rows": [
                {"phase": "decode", "runtime_mode": "FULL", "actual_tokens": 2,
                 "padded_tokens": 2} for _ in range(3)]}}
    return row, snapshot


class GraphBenchmarkAccounting(unittest.TestCase):
    def test_native_none_pairs_without_changing_spec(self):
        performance, snap = fixture(False)
        counted, counted_snap = fixture(True)
        validate_snapshot(performance, snap, False)
        validate_snapshot(counted, counted_snap, True)
        merged = pair_rows([performance], [counted])[0]
        self.assertIsNone(merged["spec"])
        self.assertIsNone(merged["counter_spec"])
        self.assertEqual(merged["logical_counters"]["tokens"], 12.)

    def test_native_reduced_k_allows_only_counter_toggle(self):
        performance, _ = fixture(False, {"mode": "global_native", "drop_j": 4, "count_routes": False})
        counted, _ = fixture(True, {"mode": "global_native", "drop_j": 4, "count_routes": True})
        self.assertTrue(pair_rows([performance], [counted])[0]["counter_identity_verified"])
        counted["spec"]["drop_j"] = 2
        with self.assertRaisesRegex(ValueError, "action spec"):
            pair_rows([performance], [counted])

    def test_missing_duplicate_or_wrong_role_is_rejected(self):
        performance, _ = fixture(False)
        counted, _ = fixture(True)
        for left, right in [([performance, performance], [counted, counted]),
                            ([performance], []), ([counted], [performance])]:
            with self.assertRaises(ValueError):
                pair_rows(left, right)
        counted["output_tokens_sha256"] = "different"
        with self.assertRaisesRegex(RuntimeError, "output tokens"):
            pair_rows([performance], [counted])

    def test_capture_counter_reset_or_padded_replay_is_rejected(self):
        row, snapshot = fixture(True)
        changed = copy.deepcopy(snapshot)
        changed["layers"]["totals"]["events"] = 2.
        with self.assertRaisesRegex(RuntimeError, "exactly once"):
            validate_snapshot(row, changed, True)
        for field, value in [("padded_tokens", 4), ("runtime_mode", "NONE")]:
            changed = copy.deepcopy(snapshot)
            changed["graph"]["runtime_rows"][0][field] = value
            with self.assertRaisesRegex(RuntimeError, "unpadded FULL"):
                validate_snapshot(row, changed, True)

    def test_counter_toggle_and_router_restoration_are_checked(self):
        row, snapshot = fixture(False)
        changed = copy.deepcopy(snapshot)
        changed["graph"]["count_routes"] = True
        with self.assertRaisesRegex(RuntimeError, "settings"):
            validate_snapshot(row, changed, False)
        changed = copy.deepcopy(snapshot)
        changed["execution"]["layer0"]["current_router_top_k"] = 4
        with self.assertRaisesRegex(RuntimeError, "not restored"):
            validate_snapshot(row, changed, False)


if __name__ == "__main__":
    unittest.main()
