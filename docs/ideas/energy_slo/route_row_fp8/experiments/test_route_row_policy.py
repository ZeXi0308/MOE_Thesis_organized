#!/usr/bin/env python3
"""CPU-only logic tests for the frozen route-row mechanism.

These tests deliberately do not emulate CUDA FP8.  The production path must
hard-fail on CPU; native-kernel/counter acceptance remains a real-GPU Phase-4
review item.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from route_row_policy import (
    ACTION_BF16,
    ACTION_FP8,
    ACTION_SKIP,
    FP8_RECIPE_E4M3FN,
    ROW_BIN_LABELS,
    DualResidentExpertMLP,
    RouteRowLUT,
    RuntimeCounters,
    SurfaceCell,
    SurfaceKey,
    code_config_sha256,
    energy_per_completed_token,
    require_cuda_fp8,
    route_row_counts,
    routing_blind_expected_rows,
    row_bin_for_count,
    verify_formal_signoff,
)


QUALITY_SHA256 = "a" * 64


def make_key(*, gpu_uuid: str = "GPU-TEST") -> SurfaceKey:
    return SurfaceKey(
        model_revision="example/model@0123456789abcdef",
        gpu_uuid=gpu_uuid,
        fp8_recipe=FP8_RECIPE_E4M3FN,
        expert_mlp_shape=(16, 32, 16),
    )


def make_cell(
    row_bin: str,
    *,
    sample_count: int = 30,
    energy_lcb: float = 0.5,
    latency_ucb: float = 0.0,
    mass: float = 1.0,
) -> SurfaceCell:
    return SurfaceCell(
        row_bin=row_bin,
        sample_count=sample_count,
        delta_energy_mean_j=0.75,
        delta_energy_lcb95_j=energy_lcb,
        delta_energy_ucb95_j=max(energy_lcb, 1.0),
        delta_latency_mean_ms=-0.2 if latency_ucb <= 0 else 0.2,
        delta_latency_lcb95_ms=min(-0.5, latency_ucb),
        delta_latency_ucb95_ms=latency_ucb,
        bf16_energy_mass_j=mass,
    )


class RowBinTests(unittest.TestCase):
    def test_exact_frozen_boundaries(self) -> None:
        cases = {
            0: None,
            1: "1",
            2: "2",
            3: "3-4",
            4: "3-4",
            5: "5-8",
            8: "5-8",
            9: "9-16",
            16: "9-16",
            17: "17-32",
            32: "17-32",
            33: "33-64",
            64: "33-64",
            65: "65-128",
            128: "65-128",
            129: ">=129",
            10000: ">=129",
        }
        self.assertEqual(tuple(ROW_BIN_LABELS), tuple(dict.fromkeys(cases.values()))[1:])
        for value, expected in cases.items():
            with self.subTest(value=value):
                self.assertEqual(row_bin_for_count(value), expected)

    def test_invalid_row_counts_fail(self) -> None:
        with self.assertRaises(ValueError):
            row_bin_for_count(-1)
        with self.assertRaises(TypeError):
            row_bin_for_count(True)
        with self.assertRaises(TypeError):
            row_bin_for_count(1.0)  # type: ignore[arg-type]


class RouteAccountingTests(unittest.TestCase):
    def test_row_count_identity(self) -> None:
        selected = torch.tensor([[0, 1], [1, 3], [2, 3]], dtype=torch.int64)
        counts = route_row_counts(
            selected,
            4,
            expected_active_decode_tokens=3,
            expected_top_k=2,
        )
        self.assertEqual(counts.tolist(), [1, 2, 1, 2])
        self.assertEqual(int(counts.sum()), 3 * 2)

    def test_bad_shape_dtype_and_identity_fail(self) -> None:
        with self.assertRaises(ValueError):
            route_row_counts(torch.tensor([0, 1]), 2)
        with self.assertRaises(TypeError):
            route_row_counts(torch.tensor([[0.0, 1.0]]), 2)
        with self.assertRaises(ValueError):
            route_row_counts(torch.tensor([[0, 2]]), 2)
        with self.assertRaises(ValueError):
            route_row_counts(torch.tensor([[1, 1]]), 2)
        with self.assertRaises(ValueError):
            route_row_counts(
                torch.tensor([[0, 1]]), 2, expected_active_decode_tokens=2
            )

    def test_routing_blind_signal_is_expectation_only(self) -> None:
        self.assertEqual(routing_blind_expected_rows(8, 8, 64), 1.0)
        with self.assertRaises(ValueError):
            routing_blind_expected_rows(1, 3, 2)


class FailClosedPolicyTests(unittest.TestCase):
    def test_valid_cell_selects_fp8_and_zero_skips(self) -> None:
        key = make_key()
        lut = RouteRowLUT(
            key=key,
            cells={"3-4": make_cell("3-4")},
            quality_gate_passed=True,
            min_samples_per_bin=30,
            quality_artifact_sha256=QUALITY_SHA256,
        )
        self.assertEqual(lut.decide(3, key).action, ACTION_FP8)
        decision = lut.decide(0, key)
        self.assertEqual(decision.action, ACTION_SKIP)
        self.assertEqual(decision.reason, "empty_expert")

    def test_every_uncertain_case_falls_back_bf16(self) -> None:
        key = make_key()
        cases = {
            "quality": RouteRowLUT(key, {"1": make_cell("1")}, False, 30),
            "missing": RouteRowLUT(key, {}, True, 30, QUALITY_SHA256),
            "underpowered": RouteRowLUT(
                key, {"1": make_cell("1", sample_count=29)}, True, 30, QUALITY_SHA256
            ),
            "energy_ci": RouteRowLUT(
                key, {"1": make_cell("1", energy_lcb=0.0)}, True, 30, QUALITY_SHA256
            ),
            "latency_ci": RouteRowLUT(
                key,
                {"1": make_cell("1", latency_ucb=0.001)},
                True,
                30,
                QUALITY_SHA256,
            ),
        }
        for name, lut in cases.items():
            with self.subTest(name=name):
                self.assertEqual(lut.decide(1, key).action, ACTION_BF16)
        valid = RouteRowLUT(
            key, {"1": make_cell("1")}, True, 30, QUALITY_SHA256
        )
        self.assertEqual(
            valid.decide(1, make_key(gpu_uuid="GPU-DIFFERENT")).action,
            ACTION_BF16,
        )

    def test_existence_uses_bf16_energy_mass_not_event_count(self) -> None:
        key = make_key()
        lut = RouteRowLUT(
            key,
            cells={
                "1": make_cell("1", mass=8.9),
                "2": make_cell("2", energy_lcb=-0.1, mass=1.1),
            },
            quality_gate_passed=True,
            min_samples_per_bin=30,
            quality_artifact_sha256=QUALITY_SHA256,
        )
        summary = lut.existence_summary(0.10)
        self.assertTrue(summary["passed"])
        self.assertAlmostEqual(summary["safe_bf16_energy_mass_fraction"], 0.89)
        self.assertAlmostEqual(summary["fallback_bf16_energy_mass_fraction"], 0.11)

    def test_lut_round_trip_rejects_bin_drift(self) -> None:
        key = make_key()
        lut = RouteRowLUT(
            key, {"1": make_cell("1")}, True, 30, QUALITY_SHA256
        )
        encoded = lut.to_dict()
        decoded = RouteRowLUT.from_dict(encoded)
        self.assertEqual(decoded.decide(1, key).action, ACTION_FP8)
        encoded["fixed_row_bins"] = ["1", "2"]
        with self.assertRaises(ValueError):
            RouteRowLUT.from_dict(encoded)

    def test_lut_json_quality_boolean_is_strict_and_missing_fails(self) -> None:
        key = make_key()
        valid = RouteRowLUT(
            key, {"1": make_cell("1")}, True, 30, QUALITY_SHA256
        ).to_dict()
        for invalid in ("true", "false", 1, 0, None, [], {}):
            document = dict(valid)
            document["quality_gate_passed"] = invalid
            with self.subTest(invalid=invalid):
                with self.assertRaises((TypeError, ValueError)):
                    RouteRowLUT.from_dict(document)
        missing = dict(valid)
        del missing["quality_gate_passed"]
        with self.assertRaises(ValueError):
            RouteRowLUT.from_dict(missing)

    def test_quality_pass_requires_valid_artifact_sha256(self) -> None:
        key = make_key()
        with self.assertRaises(ValueError):
            RouteRowLUT(key, {"1": make_cell("1")}, True, 30, None)

        valid = RouteRowLUT(
            key, {"1": make_cell("1")}, True, 30, QUALITY_SHA256
        ).to_dict()
        for invalid_hash in (None, "", "a" * 63, "A" * 64, 7):
            document = dict(valid)
            document["quality_artifact_sha256"] = invalid_hash
            with self.subTest(invalid_hash=invalid_hash):
                with self.assertRaises((TypeError, ValueError)):
                    RouteRowLUT.from_dict(document)
        missing_hash = dict(valid)
        del missing_hash["quality_artifact_sha256"]
        with self.assertRaises(ValueError):
            RouteRowLUT.from_dict(missing_hash)

        failed_quality = dict(valid)
        failed_quality["quality_gate_passed"] = False
        failed_quality["quality_artifact_sha256"] = None
        decoded = RouteRowLUT.from_dict(failed_quality)
        self.assertFalse(decoded.quality_gate_passed)


class AccountingAndGateTests(unittest.TestCase):
    def test_frozen_synthetic_energy_identity(self) -> None:
        result = energy_per_completed_token(
            100.0 * 10.0,
            100,
            duration_s=10.0,
            idle_power_w=30.0,
        )
        self.assertEqual(result["board_j_per_completed_output_token"], 10.0)
        self.assertEqual(result["dynamic_j_per_completed_output_token"], 7.0)

    def test_hash_is_named_order_independent_and_content_sensitive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first"
            second = root / "second"
            first.write_bytes(b"a")
            second.write_bytes(b"b")
            digest = code_config_sha256({"a": first, "b": second})
            self.assertEqual(
                digest, code_config_sha256({"b": second, "a": first})
            )
            second.write_bytes(b"changed")
            self.assertNotEqual(
                digest, code_config_sha256({"a": first, "b": second})
            )

    def test_formal_gate_requires_phase_status_and_exact_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            signoff = Path(temporary) / "signoff.json"
            expected_hash = "a" * 64
            signoff.write_text(
                json.dumps(
                    {
                        "status": "SIGNED-OFF",
                        "phase": "Phase4",
                        "code_config_sha256": expected_hash,
                    }
                ),
                encoding="utf-8",
            )
            accepted = verify_formal_signoff(signoff, expected_hash)
            self.assertEqual(accepted["status"], "SIGNED-OFF")
            with self.assertRaises(RuntimeError):
                verify_formal_signoff(signoff, "b" * 64)
            signoff.write_text(
                json.dumps(
                    {
                        "status": "BLOCKED",
                        "phase": "Phase4",
                        "code_config_sha256": expected_hash,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(RuntimeError):
                verify_formal_signoff(signoff, expected_hash)

    def test_config_pins_exact_models_and_target_counts(self) -> None:
        config_path = Path(__file__).parent / "configs" / "route_row_break_even_v1.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        self.assertEqual(tuple(config["fixed_row_bins"]), ROW_BIN_LABELS)
        self.assertEqual(
            config["status"],
            "PHASE3_BLOCKED_PENDING_INTEGRATED_CONTINUOUS_DYNAMIC_EXPERT_HOT_PATH",
        )
        self.assertEqual(
            config["artifact_scope"], "CAPABILITY_AND_CALIBRATION_PROXY_ONLY"
        )
        self.assertEqual(config["models"]["olmoe"]["expected_target_linears"], 3072)
        self.assertEqual(config["models"]["llm_jp"]["expected_target_linears"], 1536)
        self.assertEqual(
            config["models"]["olmoe"]["revision"],
            "6d84c48581ece794365f2b8e9cfb043c68ade9c5",
        )
        self.assertEqual(
            config["models"]["llm_jp"]["revision"],
            "1d5983076dfc67aee4a77ec06a27027f5bab6055",
        )

    def test_formal_hash_manifest_covers_all_energy_phase3_inputs(self) -> None:
        config_path = Path(__file__).parent / "configs" / "route_row_break_even_v1.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        expected = {
            "docs/ideas/energy_slo/route_row_fp8/experiments/continuous_decode_harness.py",
            "docs/ideas/energy_slo/route_row_fp8/experiments/power_accounting.py",
            "docs/ideas/energy_slo/route_row_fp8/experiments/route_row_policy.py",
            "docs/ideas/energy_slo/route_row_fp8/experiments/run_route_row_surface.py",
            "docs/ideas/energy_slo/route_row_fp8/experiments/test_continuous_decode_harness.py",
            "docs/ideas/energy_slo/route_row_fp8/experiments/test_power_accounting.py",
            "docs/ideas/energy_slo/route_row_fp8/experiments/test_route_row_policy.py",
            "docs/ideas/energy_slo/route_row_fp8/experiments/configs/route_row_break_even_v1.json",
        }
        manifest = set(config["formal_gate"]["hash_manifest"])
        self.assertEqual(manifest, expected)
        repository_root = next(
            candidate
            for candidate in Path(__file__).resolve().parents
            if (candidate / "experiments/shared").is_dir()
        )
        named_paths = {name: repository_root / name for name in manifest}
        for path in named_paths.values():
            self.assertTrue(path.is_file(), path)
        self.assertEqual(len(code_config_sha256(named_paths)), 64)


class NativeFp8RefusalTests(unittest.TestCase):
    def test_cpu_is_never_an_fp8_proxy(self) -> None:
        with self.assertRaises(RuntimeError):
            require_cuda_fp8(torch.device("cpu"))

    def test_dual_resident_wrapper_refuses_cpu_weights(self) -> None:
        class CpuExpert(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.gate_proj = nn.Linear(4, 8, bias=False, dtype=torch.bfloat16)
                self.up_proj = nn.Linear(4, 8, bias=False, dtype=torch.bfloat16)
                self.down_proj = nn.Linear(8, 4, bias=False, dtype=torch.bfloat16)
                self.act_fn = nn.SiLU()

        with self.assertRaises(RuntimeError):
            DualResidentExpertMLP(CpuExpert(), RuntimeCounters())


if __name__ == "__main__":
    unittest.main(verbosity=2)
