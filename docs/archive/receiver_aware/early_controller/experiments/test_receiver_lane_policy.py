from __future__ import annotations


# --- shared-lib bootstrap (auto) ---
import sys
from pathlib import Path as _Path

def _ensure_shared_on_path() -> None:
    here = _Path(__file__).resolve().parent
    for p in [here, *here.parents]:
        cand = p / "experiments" / "shared"
        if (cand / "capture_moe.py").exists():
            s = str(cand)
            if s not in sys.path:
                sys.path.insert(0, s)
            return
        if (p / "capture_moe.py").exists():
            s = str(p)
            if s not in sys.path:
                sys.path.insert(0, s)
            return

_ensure_shared_on_path()
del _ensure_shared_on_path, _Path
# --- end bootstrap ---

import unittest

import torch

from fake_quant import apply_precision
from receiver_lane_policy import ReceiverLaneController, ReceiverPolicyConfig


class ReceiverLanePolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(19)
        self.raw = (torch.randn(2, 2, 32) * 3.0).to(torch.bfloat16)
        self.weights = torch.tensor([[0.7, 0.3], [0.6, 0.4]], dtype=torch.bfloat16)
        self.experts = torch.tensor([[3, 0], [3, 3]], dtype=torch.long)
        self.receivers = torch.tensor([0])
        self.requests = torch.tensor([17])
        self.mask = torch.tensor([[1, 0]])

    def _controller(self, arm: str, **kwargs) -> ReceiverLaneController:
        config = ReceiverPolicyConfig(
            arm=arm,
            ep_size=4,
            gpus_per_node=2,
            placement="round_robin",
            high_precision="fp8",
            low_precision="int4",
            **kwargs,
        )
        return ReceiverLaneController(config)

    def _apply(self, controller: ReceiverLaneController):
        controller.begin_forward(self.receivers, self.mask, self.requests)
        output, weights = controller.apply_layer(
            0, self.raw, self.weights, self.experts, num_experts=4
        )
        controller.end_forward()
        return output, weights

    def test_padding_and_local_pairs_are_not_quantized(self) -> None:
        controller = self._controller("uniform_low")
        output, returned_weights = self._apply(controller)
        expected_low = apply_precision(self.raw[0, 0].unsqueeze(0), "int4").squeeze(0)
        self.assertTrue(torch.equal(output[0, 0], expected_low))
        self.assertTrue(torch.equal(output[0, 1], self.raw[0, 1]))
        self.assertTrue(torch.equal(output[1], self.raw[1]))
        self.assertTrue(torch.equal(returned_weights, self.weights))
        exposure = controller.request_exposure()[17]
        self.assertEqual(exposure["remote_pairs"], 1)
        self.assertEqual(exposure["low_pairs"], 1)
        self.assertEqual(exposure["low_frac"], 1.0)

    def test_uniform_full_uses_high_format_on_remote_only(self) -> None:
        controller = self._controller("uniform_full")
        output, _ = self._apply(controller)
        expected_high = apply_precision(self.raw[0, 0].unsqueeze(0), "fp8").squeeze(0)
        self.assertTrue(torch.equal(output[0, 0], expected_high))
        self.assertTrue(torch.equal(output[0, 1], self.raw[0, 1]))
        self.assertEqual(controller.request_exposure()[17]["low_pairs"], 0)

    def test_causal_policy_cannot_use_same_step_load(self) -> None:
        controller = self._controller(
            "causal_no_hysteresis", threshold_high=1.0, threshold_low=0.5
        )
        self._apply(controller)
        self.assertEqual(controller.step_rows[0]["low_pairs"], 0)
        self._apply(controller)
        self.assertEqual(controller.step_rows[1]["low_pairs"], 1)

    def test_controller_uses_previous_step_and_hysteresis(self) -> None:
        controller = self._controller(
            "controller",
            alpha=1.0,
            threshold_high=1.0,
            threshold_low=0.0,
            dwell_min=1,
        )
        self._apply(controller)
        self.assertEqual(controller.step_rows[0]["low_lane_count"], 0)
        self._apply(controller)
        self.assertEqual(controller.step_rows[1]["low_lane_count"], 1)

    def test_calibration_excludes_padding(self) -> None:
        controller = self._controller("uniform_full")
        self._apply(controller)
        self.assertEqual(controller.observed_steps, 1)
        self.assertEqual(controller.static_profile(), {(3, 0): 1.0})
        high, low, static = controller.fitted_thresholds(0.6, 0.5)
        self.assertEqual(high, 1.0)
        self.assertEqual(low, 0.5)
        self.assertEqual(static, 1.0)


if __name__ == "__main__":
    unittest.main()
