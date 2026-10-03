from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


MODULE_PATH = Path(__file__).with_name("run_vllm_decode_cap_branch.py")
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("decode_cap_branch", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def fake_output(index: int, prompt: list[int], output_tokens: int = 4):
    completion = SimpleNamespace(token_ids=list(range(output_tokens)), finish_reason="length")
    metrics = SimpleNamespace(
        num_generation_tokens=output_tokens,
        queued_ts=1.0,
        scheduled_ts=1.1,
        first_token_ts=1.3,
        last_token_ts=1.6,
        first_token_latency=0.4,
    )
    return SimpleNamespace(
        request_id=str(index),
        prompt_token_ids=prompt,
        outputs=[completion],
        metrics=metrics,
    )


class DecodeCapBranchRunnerTest(unittest.TestCase):
    def test_embedded_producer_bundle_is_self_contained_for_cli(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for relative, payload in MODULE.producer_source_bundle().items():
                (root / relative).write_bytes(payload)
            result = subprocess.run(
                [sys.executable, str(root / "producer_source.py"), "--help"],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--decode-cap", result.stdout)

    def test_complete_request_denominator_and_metrics(self) -> None:
        prompts = [[1, 2], [3, 4]]
        rows = MODULE.summarize_request_outputs(
            [fake_output(0, prompts[0]), fake_output(1, prompts[1])],
            prompts,
            4,
            branch_started_s=10.0,
            branch_finished_s=12.0,
        )
        summary = MODULE.summarize_branch(rows, wall_s=2.0, output_tokens=4)
        self.assertEqual(summary["completed_request_count"], 2)
        self.assertEqual(summary["total_generated_tokens"], 8)
        self.assertEqual(summary["total_decode_intervals"], 6)
        self.assertAlmostEqual(rows[0]["queue_ms"], 100.0)
        self.assertAlmostEqual(rows[0]["tpot_ms"], 100.0)
        self.assertAlmostEqual(rows[0]["e2e_ms"], 700.0)
        self.assertEqual(
            rows[0]["raw_timing_s"]["branch_started_perf_counter"], 10.0
        )

    def test_short_generation_fails_closed(self) -> None:
        output = fake_output(0, [1, 2])
        output.outputs[0].token_ids = [1, 2, 3]
        with self.assertRaisesRegex(ValueError, "denominator"):
            MODULE.summarize_request_outputs(
                [output],
                [[1, 2]],
                4,
                branch_started_s=10.0,
                branch_finished_s=12.0,
            )

    def test_prompt_order_drift_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "identity/order"):
            MODULE.summarize_request_outputs(
                [fake_output(0, [2, 1])],
                [[1, 2]],
                4,
                branch_started_s=10.0,
                branch_finished_s=12.0,
            )

    def test_nonfinite_or_nonpositive_raw_timing_fails_closed(self) -> None:
        output = fake_output(0, [1, 2])
        output.metrics.last_token_ts = float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            MODULE.summarize_request_outputs(
                [output],
                [[1, 2]],
                4,
                branch_started_s=10.0,
                branch_finished_s=12.0,
            )

        with self.assertRaisesRegex(ValueError, "positive"):
            MODULE.summarize_request_outputs(
                [fake_output(0, [1, 2])],
                [[1, 2]],
                4,
                branch_started_s=10.0,
                branch_finished_s=10.0,
            )

    def test_fcfs_wave_pressure_keeps_cap_denominator(self) -> None:
        routes = np.array(
            [
                [[[0, 1]], [[0, 1]]],
                [[[2, 3]], [[2, 3]]],
                [[[0, 0]], [[0, 0]]],
                [[[0, 0]], [[0, 0]]],
            ],
            dtype=np.int64,
        )
        pressure = MODULE.summarize_fcfs_waves(routes, decode_cap=2, num_experts=4)
        self.assertEqual(pressure["wave_count"], 2)
        self.assertEqual(pressure["waves"][0]["max_layer_step_load"], 1.0)
        self.assertEqual(pressure["waves"][1]["max_layer_step_load"], 4.0)
        self.assertFalse(pressure["scheduler_trace_captured"])


if __name__ == "__main__":
    unittest.main()
