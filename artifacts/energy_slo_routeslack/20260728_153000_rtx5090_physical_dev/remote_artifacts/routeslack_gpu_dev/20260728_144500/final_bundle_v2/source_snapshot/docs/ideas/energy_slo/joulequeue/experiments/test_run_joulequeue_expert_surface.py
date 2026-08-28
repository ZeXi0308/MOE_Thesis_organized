from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import torch
from torch import nn


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from joulequeue_policy import ProtocolError  # noqa: E402
from run_joulequeue_expert_surface import (  # noqa: E402
    _bootstrap_mean_ci,
    _numerical_errors,
    _run_arm,
    _surface_curves,
    _surface_metadata,
)
from run_joulequeue_oracle import _load_surface  # noqa: E402


def surface_config() -> dict[str, object]:
    return {
        "row_grid": [1],
        "independent_trials": 10,
        "bootstrap_resamples": 200,
        "bootstrap_seed": 7,
        "maximum_energy_cv": 0.1,
        "minimum_window_seconds": 2.0,
        "power_sample_interval_seconds": 0.005,
        "numerical_gate": {
            "max_abs_error": 0.02,
            "mean_abs_error": 0.002,
            "max_cosine_error": 0.0001,
        },
        "artifact_contract": {
            "evidence_level": "REAL_5090_NATIVE_ACTIVATIONS",
            "energy_basis": "total_during_launch",
            "paired_order": "AB_BA",
            "counter_sample_logical_window_bracketed": True,
            "background_sampler_exceptions_propagated": True,
        },
    }


def trial_rows() -> list[dict[str, object]]:
    return [
        {
            "layer_id": 2,
            "expert_id": 3,
            "rows": 1,
            "trial": trial,
            "arm_order": (
                "separate>coalesced" if trial % 2 == 0 else "coalesced>separate"
            ),
            "coalesced_energy_j": 1.0 + trial / 1000.0,
            "coalesced_latency_us": 10.0 + trial / 100.0,
            "delta_energy_j": 0.2,
            "energy_source": "nvml_total_energy_counter",
            "gpu_uuid": "GPU-ABC",
            "max_sample_gap_s": 0.006,
            "counter_sample_logical_window_bracketed": True,
            "separate_workload_start_ns": 0,
            "separate_workload_end_ns": 2_000_000_000,
            "coalesced_workload_start_ns": 3_000_000_000,
            "coalesced_workload_end_ns": 5_000_000_000,
            "numerical_pass": True,
        }
        for trial in range(10)
    ]


def formal_surface_payload() -> dict[str, object]:
    rows = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    metadata = {
        "artifact_schema": "joulequeue-expert-surface-v1",
        "model_revision": "model@revision",
        "evidence_level": "REAL_5090_NATIVE_ACTIVATIONS",
        "native_activations": True,
        "formal_eligible": True,
        "formal_eligibility_failed_gates": [],
        "phase4_signoff_verified": True,
        "phase4_signoff_sha256": "a" * 64,
        "gpu_name": "NVIDIA GeForce RTX 5090",
        "gpu_uuid": "GPU-ABC",
        "energy_source": "nvml_total_energy_counter",
        "energy_basis": "total_during_launch",
        "paired_order": "AB_BA",
        "counter_sample_logical_window_bracketed": True,
        "counter_sample_boundary_relation": "SEQUENTIAL_BRACKETING_NOT_ATOMIC",
        "background_sampler_exceptions_propagated": True,
        "minimum_window_s": 2.0,
        "independent_trials": 10,
        "sampling_interval_ms": 5.0,
        "max_observed_gap_ms": 6.0,
        "all_measurements_valid": True,
        "all_numerical_gates_passed": True,
        "numerical_gate": {
            "max_abs_error": 0.02,
            "mean_abs_error": 0.002,
            "max_cosine_error": 0.0001,
        },
    }
    points = [
        {
            "rows": rows_value,
            "energy_j": float(rows_value),
            "energy_ucb95_j": float(rows_value) + 0.1,
            "latency_us": float(rows_value) + 10.0,
            "latency_ucb95_us": float(rows_value) + 10.1,
            "independent_trials": 10,
            "measurement_valid": True,
            "numerical_gate_passed": True,
            "paired_order_complete": True,
        }
        for rows_value in rows
    ]
    return {
        "metadata": metadata,
        "curves": [
            {
                "layer_id": layer_id,
                "expert_id": expert_id,
                "energy_basis": "total_during_launch",
                "all_measurements_valid": True,
                "all_numerical_gates_passed": True,
                "points": points,
            }
            for layer_id in (2, 4, 6, 8)
            for expert_id in (3, 5, 7, 9)
        ],
    }


class KernelAndStatisticsTest(unittest.TestCase):
    def test_separate_and_coalesced_preserve_row_identity(self) -> None:
        expert = nn.Linear(4, 3, bias=False)
        activation = torch.arange(20, dtype=torch.float32).reshape(5, 4)
        _run_arm(expert, activation, "separate", 2)
        _run_arm(expert, activation, "coalesced", 2)
        errors = _numerical_errors(expert, activation)
        # GEMM batch shape may change the floating-point accumulation path even
        # when row identity is preserved.  Bound that roundoff by dtype epsilon
        # and the observed output scale instead of a platform-flaky absolute
        # constant.  This remains orders of magnitude tighter than the frozen
        # scientific max-absolute-error gate (2e-2).
        output_scale = max(1.0, float(expert(activation).abs().max().item()))
        roundoff_bound = 8.0 * torch.finfo(activation.dtype).eps * output_scale
        self.assertLessEqual(errors["max_abs_error"], roundoff_bound)
        self.assertLessEqual(errors["mean_abs_error"], roundoff_bound)
        self.assertLessEqual(errors["max_cosine_error"], 1e-6)

    def test_unknown_measurement_arm_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown arm"):
            _run_arm(nn.Identity(), torch.ones((1, 4)), "future", 1)

    def test_separate_arm_concatenates_exactly_once_per_repeat(self) -> None:
        activation = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        real_cat = torch.cat
        with mock.patch.object(torch, "cat", wraps=real_cat) as cat:
            _run_arm(nn.Identity(), activation, "separate", 3)
        self.assertEqual(cat.call_count, 3)

    def test_bootstrap_uses_nonempty_independent_units(self) -> None:
        mean, lower, upper = _bootstrap_mean_ci([1.0, 2.0, 3.0], 200, 7)
        self.assertLessEqual(lower, mean)
        self.assertLessEqual(mean, upper)
        with self.assertRaisesRegex(ValueError, "non-empty"):
            _bootstrap_mean_ci([], 10, 1)

    def test_per_expert_curve_uses_trials_not_inner_repeats(self) -> None:
        curves = _surface_curves(trial_rows(), ((2, 3),), surface_config())
        self.assertEqual(len(curves), 1)
        point = curves[0]["points"][0]
        self.assertEqual(point["independent_trials"], 10)
        self.assertTrue(point["paired_order_complete"])
        self.assertTrue(point["measurement_valid"])


class ArtifactContractTest(unittest.TestCase):
    def test_formal_eligibility_requires_mode_and_signoff(self) -> None:
        config = surface_config()
        trials = trial_rows()
        curves = _surface_curves(trials, ((2, 3),), config)
        common = {
            "model_revision": "model@revision",
            "gpu_name": "NVIDIA GeForce RTX 5090",
            "gpu_uuid": "GPU-ABC",
            "capture_metadata": {
                "capture_kind": "joulequeue_real_bf16_expert_inputs",
                "input_source": "measured_same_gpu_model_forward",
            },
            "config": config,
            "trials": trials,
            "curves": curves,
        }
        unsigned = _surface_metadata(
            mode="formal", signoff_verified=False, signoff_sha256=None, **common
        )
        self.assertFalse(unsigned["formal_eligible"])
        self.assertIn(
            "phase4_signoff_verified", unsigned["formal_eligibility_failed_gates"]
        )
        development = _surface_metadata(
            mode="dev", signoff_verified=True, signoff_sha256="a" * 64, **common
        )
        self.assertFalse(development["formal_eligible"])
        signed = _surface_metadata(
            mode="formal", signoff_verified=True, signoff_sha256="a" * 64, **common
        )
        self.assertTrue(signed["formal_eligible"])
        self.assertEqual(signed["formal_eligibility_failed_gates"], [])

    def test_formal_loader_accepts_total_launch_per_expert_curves(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "surface.json"
            path.write_text(json.dumps(formal_surface_payload()), encoding="utf-8")
            surface = _load_surface(path, formal=True)
        self.assertEqual(surface.energy_basis, "total_during_launch")
        self.assertEqual(len(surface.curves), 16)
        self.assertIn((2, 3), surface.curves)
        self.assertAlmostEqual(surface.curves[(2, 3)].estimate(2).energy_j, 2.1)

    def test_formal_loader_rejects_self_asserted_eligibility_with_failed_point(self) -> None:
        payload = formal_surface_payload()
        payload["curves"][0]["points"][0]["measurement_valid"] = False
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "surface.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ProtocolError, "frozen gate"):
                _load_surface(path, formal=True)


if __name__ == "__main__":
    unittest.main()
