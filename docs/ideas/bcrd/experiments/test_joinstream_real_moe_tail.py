from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

try:
    from . import joinstream_real_moe_tail_analyzer as analyzer
except ImportError:
    import joinstream_real_moe_tail_analyzer as analyzer  # type: ignore


class RealMoETailAnalyzerTest(unittest.TestCase):
    @staticmethod
    def _calibration() -> dict[str, object]:
        cells = []
        for route in analyzer.ROUTES:
            for scale in analyzer.SCALES:
                cells.append(
                    {
                        "cell_id": f"{route}__{scale}",
                        "route_distribution": route,
                        "problem_scale": scale,
                        "selection_status": "SELECTED",
                        "selected_gate_remaining_ratio": 0.25,
                        "candidates": [
                            {"remaining_ratio": 0.0, "producer_regression_ratio": 0.0},
                            {"remaining_ratio": 0.125, "producer_regression_ratio": 0.02},
                            {"remaining_ratio": 0.25, "producer_regression_ratio": 0.04},
                            {"remaining_ratio": 0.5, "producer_regression_ratio": 0.08},
                        ],
                    }
                )
        return {
            "schema": analyzer.CALIBRATION_SCHEMA,
            "status": "CALIBRATION_COMPLETE_LOCKED",
            "calibration_route_seed": 111,
            "gate_candidates_remaining_ratio": list(analyzer.GATE_CANDIDATES),
            "repetitions": {
                "warmups_per_cell_candidate": 10,
                "measured_per_cell_candidate": 50,
            },
            "cells": cells,
        }

    @staticmethod
    def _run_lock(calibration_sha256: str) -> dict[str, object]:
        cells = []
        for route in analyzer.ROUTES:
            for scale in analyzer.SCALES:
                high = scale == "HIGH_RESIDENCY"
                cells.append(
                    {
                        "cell_id": f"{route}__{scale}",
                        "route_distribution": route,
                        "problem_scale": scale,
                        "gate_selection_status": "SELECTED",
                        "gate_remaining_ratio": 0.25,
                        "total_tokens": 64 if not high else 128,
                        "routed_tokens": 128 if not high else 256,
                        "top_k": 2,
                        "expert_count": 8,
                        "theoretical_flops": 1_000_000 if not high else 2_000_000,
                        "expert_routed_token_counts": (
                            ([16] * 8 if not high else [32] * 8)
                            if route == "BALANCED"
                            else (
                                [64, 16, 12, 10, 8, 8, 6, 4]
                                if not high
                                else [128, 32, 24, 20, 16, 16, 12, 8]
                            )
                        ),
                    }
                )
        return {
            "schema": analyzer.RUN_LOCK_SCHEMA,
            "status": "LOCKED_BEFORE_FORMAL_RUN",
            "calibration_sha256": calibration_sha256,
            "calibration_route_seed": 111,
            "formal_route_seed": 222,
            "repetitions": {
                "warmups_per_cell_mode": 3,
                "correctness": 3,
                "utility": 3,
            },
            "contracts": {
                "synthetic_delay": False,
                "artificial_sm_reservation": False,
                "single_producer_kernel": True,
                "matrix_multiply_like_work": True,
                "representative_grouped_expert": True,
            },
            "cells": cells,
        }

    @staticmethod
    def _environment() -> dict[str, object]:
        return {
            "schema": analyzer.ENVIRONMENT_SCHEMA,
            "status": "CAPTURED",
            "gpu_available": True,
            "hardware": {
                "name": "NVIDIA GeForce RTX 5090",
                "sm_count": 170,
                "compute_capability": "12.0",
            },
            "software": {
                "driver_version": "595.71.05",
                "cuda_toolkit_version": "12.8",
            },
            "timer": {"resolution_ns": 32},
        }

    @staticmethod
    def _row(
        *,
        route: str,
        scale: str,
        mode: str,
        repeat_kind: str,
        repeat_index: int,
        variant: str,
        slot: int,
    ) -> dict[str, object]:
        start = 1_000_000 + repeat_index * 100_000
        join = start + 1_000
        baseline_end = start + 10_000
        if variant == "A_ALL_DONE_SHAM":
            gate = baseline_end
            observe = gate + 100
            consumer_start = observe + 100
            consumer_end = consumer_start + 2_000
            producer_end = baseline_end
            remaining = 0
        elif variant == "B_EAGER_JOINSTREAM":
            gate = join
            observe = gate + 100
            consumer_start = observe + 100
            consumer_end = consumer_start + 2_000
            producer_end = start + 10_600
            remaining = 90
        else:
            gate = join + 500
            observe = gate + 100
            consumer_start = observe + 100
            consumer_end = consumer_start + 2_000
            producer_end = start + 10_200
            remaining = 20
        total_end = max(producer_end, consumer_end)
        high = scale == "HIGH_RESIDENCY"
        return {
            "schema_version": analyzer.RAW_SCHEMA,
            "mode": mode,
            "cell_id": f"{route}__{scale}",
            "route_distribution": route,
            "problem_scale": scale,
            "repeat_kind": repeat_kind,
            "repeat_index": repeat_index,
            "permutation_slot": slot,
            "variant": variant,
            "route_seed": 222,
            "gate_threshold_remaining_ratio": 0.25,
            "total_tokens": 64 if not high else 128,
            "routed_tokens": 128 if not high else 256,
            "top_k": 2,
            "expert_count": 8,
            "theoretical_flops": 1_000_000 if not high else 2_000_000,
            "route_table_hash": f"route-{route}-{scale}",
            "input_hash": f"input-{scale}",
            "producer_work_hash": f"producer-{scale}",
            "consumer_work_hash": "consumer-fixed",
            "progress_instrumentation_hash": "progress-fixed",
            "expert_token_counts_hash": f"counts-{route}-{scale}",
            "producer_launches": 1,
            "consumer_launches": 1,
            "producer_grid_size": 80 if not high else 200,
            "producer_block_size": 256,
            "consumer_grid_size": 1,
            "consumer_block_size": 256,
            "expert_tiles_total": 100,
            "producer_work_expected": 10_000 if not high else 20_000,
            "producer_work_done": 10_000 if not high else 20_000,
            "synthetic_delay_enabled": 0,
            "artificial_sm_reservation_enabled": 0,
            "critical_expert_a": 1,
            "critical_expert_b": 6,
            "critical_contributions_done": 2,
            "critical_contributions_expected": 2,
            "producer_start_ns": start,
            "join_close_ns": join,
            "gate_satisfied_ns": gate,
            "consumer_entry_ns": start - 100,
            "consumer_observe_ns": observe,
            "consumer_start_ns": consumer_start,
            "consumer_end_ns": consumer_end,
            "producer_end_ns": producer_end,
            "total_end_ns": total_end,
            "remaining_producer_work_at_consumer_start": remaining,
            "output_hash": "output-fixed",
            "reference_output_hash": "output-fixed",
            "correctness_pass": 1,
            "stale_read": 0,
            "timestamp_contract_pass": 1,
            "cuda_error": "",
        }

    @classmethod
    def _rows(cls) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for route in analyzer.ROUTES:
            for scale in analyzer.SCALES:
                for mode in analyzer.MODES:
                    for repeat_kind in ("warmup", "measured"):
                        count = 3
                        for repeat in range(count):
                            order = analyzer.PERMUTATIONS[repeat % len(analyzer.PERMUTATIONS)]
                            for slot, variant in enumerate(order):
                                rows.append(
                                    cls._row(
                                        route=route,
                                        scale=scale,
                                        mode=mode,
                                        repeat_kind=repeat_kind,
                                        repeat_index=repeat,
                                        variant=variant,
                                        slot=slot,
                                    )
                                )
        return rows

    @classmethod
    def _fixture(
        cls, directory: Path, *, mutate=None
    ) -> tuple[Path, Path, Path, Path]:
        calibration = directory / "calibration.json"
        run_lock = directory / "run_lock.json"
        formal = directory / "formal_run.csv"
        environment = directory / "environment.json"
        calibration_payload = cls._calibration()
        calibration.write_text(json.dumps(calibration_payload, sort_keys=True), encoding="utf-8")
        digest = hashlib.sha256(calibration.read_bytes()).hexdigest()
        lock_payload = cls._run_lock(digest)
        rows = cls._rows()
        environment_payload = cls._environment()
        if mutate is not None:
            mutate(calibration_payload, lock_payload, rows, environment_payload)
            calibration.write_text(json.dumps(calibration_payload, sort_keys=True), encoding="utf-8")
            lock_payload["calibration_sha256"] = hashlib.sha256(calibration.read_bytes()).hexdigest()
        run_lock.write_text(json.dumps(lock_payload, sort_keys=True), encoding="utf-8")
        environment.write_text(json.dumps(environment_payload, sort_keys=True), encoding="utf-8")
        with formal.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted(analyzer.REQUIRED_COLUMNS))
            writer.writeheader()
            writer.writerows(rows)
        return calibration, run_lock, formal, environment

    def test_support_requires_safe_gated_cells_and_gating_rescue(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._fixture(Path(tmp))
            result = analyzer.analyze(*paths)
        self.assertEqual(result["primary_verdict"], "SUPPORT_REAL_TAIL_ACTION_SPACE")
        self.assertEqual(result["secondary_gating_interpretation"], "GATING_NECESSARY")
        self.assertEqual(result["novelty_positioning"], "SUPPORTS")
        self.assertEqual(result["gates"]["safe_benefit_cells"], 4)
        self.assertTrue(result["gates"]["has_at_least_1us_natural_window"])
        self.assertEqual(result["evaluation_type"], analyzer.EVALUATION_TYPE)
        self.assertIn("not independent numerical ground truth", result["correctness_semantics"])

    def test_calibration_and_formal_seed_overlap_is_invalid(self) -> None:
        def mutate(_calibration, lock, _rows, _environment):
            lock["formal_route_seed"] = 111

        with tempfile.TemporaryDirectory() as tmp:
            paths = self._fixture(Path(tmp), mutate=mutate)
            with self.assertRaisesRegex(analyzer.ContractError, "seed"):
                analyzer.analyze(*paths)

    def test_synthetic_delay_in_formal_row_is_invalid(self) -> None:
        def mutate(_calibration, _lock, rows, _environment):
            rows[0]["synthetic_delay_enabled"] = 1

        with tempfile.TemporaryDirectory() as tmp:
            paths = self._fixture(Path(tmp), mutate=mutate)
            with self.assertRaisesRegex(analyzer.ContractError, "synthetic delay"):
                analyzer.analyze(*paths)

    def test_paired_route_table_mismatch_is_invalid(self) -> None:
        def mutate(_calibration, _lock, rows, _environment):
            target = next(row for row in rows if row["variant"] == "C_PROGRESS_GATED_JOINSTREAM")
            target["route_table_hash"] = "leaked-route"

        with tempfile.TemporaryDirectory() as tmp:
            paths = self._fixture(Path(tmp), mutate=mutate)
            with self.assertRaisesRegex(analyzer.ContractError, "route_table_hash"):
                analyzer.analyze(*paths)

    def test_gated_action_before_locked_remaining_threshold_is_invalid(self) -> None:
        def mutate(_calibration, _lock, rows, _environment):
            target = next(
                row
                for row in rows
                if row["variant"] == "C_PROGRESS_GATED_JOINSTREAM"
            )
            target["remaining_producer_work_at_consumer_start"] = 26

        with tempfile.TemporaryDirectory() as tmp:
            paths = self._fixture(Path(tmp), mutate=mutate)
            with self.assertRaisesRegex(analyzer.ContractError, "remaining-work threshold"):
                analyzer.analyze(*paths)

    def test_progress_gate_before_join_is_legal_when_gated_waits_for_both(self) -> None:
        def mutate(_calibration, _lock, rows, _environment):
            for row in rows:
                if row["variant"] == "C_PROGRESS_GATED_JOINSTREAM":
                    row["gate_satisfied_ns"] = int(row["join_close_ns"]) - 100

        with tempfile.TemporaryDirectory() as tmp:
            paths = self._fixture(Path(tmp), mutate=mutate)
            result = analyzer.analyze(*paths)
        self.assertEqual(result["primary_verdict"], "SUPPORT_REAL_TAIL_ACTION_SPACE")

    def test_eager_may_observe_join_before_independent_gate(self) -> None:
        def mutate(_calibration, _lock, rows, _environment):
            for row in rows:
                if row["variant"] == "B_EAGER_JOINSTREAM":
                    row["gate_satisfied_ns"] = int(row["consumer_observe_ns"]) + 500

        with tempfile.TemporaryDirectory() as tmp:
            paths = self._fixture(Path(tmp), mutate=mutate)
            result = analyzer.analyze(*paths)
        self.assertEqual(result["primary_verdict"], "SUPPORT_REAL_TAIL_ACTION_SPACE")
        self.assertLess(
            result["cells"][0]["metrics"]["eager_notification_latency_us"]["median"],
            0.0,
        )

    def test_balanced_route_label_without_balanced_counts_is_invalid(self) -> None:
        def mutate(_calibration, lock, _rows, _environment):
            target = next(
                cell
                for cell in lock["cells"]
                if cell["route_distribution"] == "BALANCED"
                and cell["problem_scale"] == "MEDIUM_RESIDENCY"
            )
            target["expert_routed_token_counts"] = [64, 16, 12, 10, 8, 8, 6, 4]

        with tempfile.TemporaryDirectory() as tmp:
            paths = self._fixture(Path(tmp), mutate=mutate)
            with self.assertRaisesRegex(analyzer.ContractError, "not balanced"):
                analyzer.analyze(*paths)

    def test_no_overlap_mechanically_weakens_natural_headroom(self) -> None:
        def mutate(_calibration, _lock, rows, _environment):
            for row in rows:
                if row["variant"] == "C_PROGRESS_GATED_JOINSTREAM":
                    row["consumer_start_ns"] = int(row["producer_end_ns"]) + 100
                    row["consumer_end_ns"] = int(row["consumer_start_ns"]) + 2000
                    row["total_end_ns"] = int(row["consumer_end_ns"])

        with tempfile.TemporaryDirectory() as tmp:
            paths = self._fixture(Path(tmp), mutate=mutate)
            result = analyzer.analyze(*paths)
        self.assertEqual(result["primary_verdict"], "WEAKEN_NO_NATURAL_HEADROOM")
        self.assertEqual(result["novelty_positioning"], "WEAKENS")

    def test_clear_gain_too_small_weakens_upper_bound(self) -> None:
        def mutate(_calibration, _lock, rows, _environment):
            for row in rows:
                if row["variant"] == "C_PROGRESS_GATED_JOINSTREAM":
                    baseline_end = int(row["producer_start_ns"]) + 12_200
                    row["consumer_end_ns"] = baseline_end - 10
                    row["total_end_ns"] = max(
                        int(row["consumer_end_ns"]), int(row["producer_end_ns"])
                    )

        with tempfile.TemporaryDirectory() as tmp:
            paths = self._fixture(Path(tmp), mutate=mutate)
            result = analyzer.analyze(*paths)
        self.assertEqual(result["primary_verdict"], "WEAKEN_UPPER_BOUND_TOO_SMALL")

    def test_one_regression_safe_overlap_cell_still_weakens_upper_bound(self) -> None:
        def mutate(_calibration, _lock, rows, _environment):
            keep = "BALANCED__MEDIUM_RESIDENCY"
            for row in rows:
                if (
                    row["variant"] == "C_PROGRESS_GATED_JOINSTREAM"
                    and row["cell_id"] != keep
                ):
                    row["producer_end_ns"] = int(row["producer_start_ns"]) + 11_000
                    row["total_end_ns"] = max(
                        int(row["producer_end_ns"]), int(row["consumer_end_ns"])
                    )

        with tempfile.TemporaryDirectory() as tmp:
            paths = self._fixture(Path(tmp), mutate=mutate)
            result = analyzer.analyze(*paths)
        self.assertEqual(result["primary_verdict"], "WEAKEN_UPPER_BOUND_TOO_SMALL")

    def test_overlap_without_any_regression_safe_cell_weakens_applicability(self) -> None:
        def mutate(_calibration, _lock, rows, _environment):
            for row in rows:
                if row["variant"] == "C_PROGRESS_GATED_JOINSTREAM":
                    row["producer_end_ns"] = int(row["producer_start_ns"]) + 11_000
                    row["total_end_ns"] = max(
                        int(row["producer_end_ns"]), int(row["consumer_end_ns"])
                    )

        with tempfile.TemporaryDirectory() as tmp:
            paths = self._fixture(Path(tmp), mutate=mutate)
            result = analyzer.analyze(*paths)
        self.assertEqual(result["primary_verdict"], "WEAKEN_REAL_MOE_APPLICABILITY")

    def test_gating_not_necessary_requires_no_paired_mad_improvement(self) -> None:
        def mutate(_calibration, _lock, rows, _environment):
            for row in rows:
                if row["variant"] == "B_EAGER_JOINSTREAM":
                    row["producer_end_ns"] = int(row["producer_start_ns"]) + 10_200
                    row["consumer_end_ns"] = int(row["producer_start_ns"]) + 3_700
                    row["total_end_ns"] = max(
                        int(row["producer_end_ns"]), int(row["consumer_end_ns"])
                    )

        with tempfile.TemporaryDirectory() as tmp:
            paths = self._fixture(Path(tmp), mutate=mutate)
            result = analyzer.analyze(*paths)
        self.assertEqual(
            result["secondary_gating_interpretation"], "GATING_NOT_NECESSARY"
        )

    def test_material_gating_improvement_without_safety_rescue_is_insufficient(self) -> None:
        def mutate(_calibration, _lock, rows, _environment):
            for row in rows:
                if row["variant"] == "B_EAGER_JOINSTREAM":
                    row["producer_end_ns"] = int(row["producer_start_ns"]) + 10_300
                    row["consumer_end_ns"] = int(row["producer_start_ns"]) + 4_700
                    row["total_end_ns"] = max(
                        int(row["producer_end_ns"]), int(row["consumer_end_ns"])
                    )

        with tempfile.TemporaryDirectory() as tmp:
            paths = self._fixture(Path(tmp), mutate=mutate)
            result = analyzer.analyze(*paths)
        self.assertEqual(
            result["secondary_gating_interpretation"], "GATING_INSUFFICIENT"
        )
        self.assertGreater(
            result["gates"]["substantive_gating_improvement_cells"], 0
        )

    def test_blocked_no_gpu_does_not_require_other_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            environment = Path(tmp) / "environment.json"
            environment.write_text(
                json.dumps(
                    {
                        "schema": analyzer.ENVIRONMENT_SCHEMA,
                        "status": "BLOCKED_NO_GPU",
                        "gpu_available": False,
                    }
                ),
                encoding="utf-8",
            )
            missing = Path(tmp) / "missing"
            result = analyzer.analyze(missing, missing, missing, environment)
        self.assertEqual(result["primary_verdict"], "BLOCKED_NO_GPU")
        self.assertEqual(result["novelty_positioning"], "DOES_NOT_ADDRESS")


if __name__ == "__main__":
    unittest.main()
