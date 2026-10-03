"""CPU-only checks for frozen paired plans; no GPU performance evidence."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from admission_feedback import AdmissionFeedback
import run_native_capacity as runner


class NativeLadderPlanTest(unittest.TestCase):
    def test_paired_prepare_reverse_and_independent_ladders(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared = root / "input"
            prepared.mkdir()
            ids = [1, 2]
            token_hash = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
            workload = dict(source_requests=[dict(request_id=f"r{i}", document_id=f"d{i}",
                prompt_token_ids_sha256=token_hash) for i in range(32)],
                actual_prompt_token_ids=[list(ids) for _ in range(32)],
                arrival_traces_s={regime: [0.01 * i for i in range(32)] for regime in ("steady", "bursty")})
            original = dict(model=dict(id="fixture", revision="fixed", tokenizer_revision="fixed"),
                requests=32, prompt_tokens=2, output_tokens=2, ttft_slo_s=1.0, tpot_slo_s=0.01, seed=0,
                workload_sha256=hashlib.sha256(json.dumps(workload, sort_keys=True).encode()).hexdigest())
            runner.dump(prepared / "config.json", original)
            runner.dump(prepared / "workload.json", workload)
            configs = []
            for reverse in (False, True):
                output = root / ("reverse" if reverse else "forward")
                argv = ["run_native_capacity.py", "--prepared-dir", str(prepared), "--output-dir", str(output),
                    "--caps", "8,12,16,24,32", "--engine-max-seqs", "32", "--arrival-scales", "0.02",
                    "--repeats", "1", "--compare-feedback-ladders", "--prepare-only"]
                if reverse:
                    argv.append("--reverse-conditions")
                with patch.object(sys, "argv", argv), patch.object(runner, "gpu_state") as gpu, \
                        patch.object(runner, "capture_episode") as capture, \
                        patch.dict(sys.modules, {"torch": None, "vllm": None}):
                    runner.main()
                    gpu.assert_not_called()
                    capture.assert_not_called()
                status = json.loads((output / "status.json").read_text())
                self.assertEqual(status["status"], "PREPARED")
                self.assertFalse(status["gpu_initialized"])
                self.assertEqual(status["cells_completed"], 0)
                self.assertEqual(status["cells_planned"], 16)
                self.assertEqual(json.loads((output / "workload.json").read_text()), workload)
                self.assertIn("--prepare-only", (output / "commands.txt").read_text())
                configs.append(json.loads((output / "config.json").read_text()))

            forward, reverse = configs
            self.assertEqual(reverse["plans"], list(reversed(forward["plans"])))
            self.assertEqual(forward["caps"], [8, 12, 16, 24, 32])
            self.assertEqual(forward["engine_max_num_seqs"], 32)
            expected = [(cap, "static", "static") for cap in (8, 12, 16, 24, 32)] + [
                (32, "shadow", "legacy"), (32, "feedback", "legacy"), (32, "feedback", "aligned")]
            for regime in ("steady", "bursty"):
                self.assertEqual([(p["cap"], p["policy"], p["ladder_name"]) for p in forward["plans"]
                                  if p["regime"] == regime], expected)
            for plan in forward["plans"]:
                cfg = runner.episode_config(forward, plan)
                expected_caps = [8, 16, 24, 32] if plan["ladder_name"] == "aligned" else [8, 12, 16, 32]
                self.assertEqual(cfg["feedback_caps"], expected_caps)
                self.assertIsNot(cfg["feedback_caps"], plan["feedback_caps"])
                if plan["policy"] != "static":
                    controller = AdmissionFeedback(cfg, 32)
                    for step in range(1, 5):
                        controller.observe({"r0": 0.012}, received_s=step, available_s=step)
                    decision = controller.decide(decision_start_s=4.01, active=32, waiting=4)
                    self.assertEqual(decision["intent_target"], 24 if plan["ladder_name"] == "aligned" else 16)
                    self.assertEqual(decision["target_cap"], 32 if plan["policy"] == "shadow" else decision["intent_target"])
                cfg["feedback_caps"].append(99)
                self.assertEqual(plan["feedback_caps"], expected_caps)
            self.assertEqual(forward["feedback_caps"], [8, 12, 16, 32])


if __name__ == "__main__":
    unittest.main()
