"""Synthetic integrity unit tests only; never a pretrained/GPU acceptance result."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import signal
import tempfile
import time
import unittest
from unittest import mock


MODULE_PATH = Path(__file__).with_name("run_phase0a_5090.py")
SPEC = importlib.util.spec_from_file_location("run_phase0a_5090", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def trace_line(*, m: int, a_rows: int, a_cols: int, d_rows: int, algo: str) -> str:
    return (
        "[x][cublasLt][1][Trace][cublasLtTSTMatmul] "
        f"A=0X1 Adesc=[type=R_16BF rows={a_rows} cols={a_cols} ld={a_rows}] "
        f"B=0X2 Bdesc=[type=R_16BF rows={a_rows} cols={m} ld={a_rows}] "
        f"C=0X3 Cdesc=[type=R_16BF rows={d_rows} cols={m} ld={d_rows}] "
        f"D=0X4 Ddesc=[type=R_16BF rows={d_rows} cols={m} ld={d_rows}] "
        "computeDesc=[computeType=COMPUTE_32F scaleType=R_32F transa=OP_T smCountTarget=170] "
        f"algo=[{algo}] workSpace=0X0 workSpaceSizeInBytes=0 "
        "beta=0 outOfPlace=0 stream=0X0"
    )


class TraceParsingTest(unittest.TestCase):
    def test_parse_and_validate_real_shape_triplet(self) -> None:
        rows = [
            MODULE.parse_cublaslt_trace_line(
                trace_line(m=32, a_rows=2048, a_cols=1024, d_rows=1024, algo="algoId=21 tile=MATMUL_TILE_32x32 numSplitsK=5 reductionScheme=INPLACE")
            ),
            MODULE.parse_cublaslt_trace_line(
                trace_line(m=32, a_rows=2048, a_cols=1024, d_rows=1024, algo="algoId=21 tile=MATMUL_TILE_32x32 numSplitsK=5 reductionScheme=INPLACE")
            ),
            MODULE.parse_cublaslt_trace_line(
                trace_line(m=32, a_rows=1024, a_cols=2048, d_rows=2048, algo="algoId=22 tile=MATMUL_TILE_32x32")
            ),
        ]
        self.assertTrue(all(row is not None for row in rows))
        roles = MODULE.validate_projection_triplet(
            rows, m_value=32, hidden_size=2048, intermediate_size=1024
        )
        self.assertEqual(roles, ["gate_proj", "up_proj", "down_proj"])
        signature = MODULE.algorithm_signature(rows[0], roles[0])
        self.assertEqual(signature["algo"]["algoId"], 21)
        self.assertEqual(signature["algo"]["numSplitsK"], 5)

    def test_m_is_not_itself_an_algorithm_regime(self) -> None:
        line_m2 = trace_line(
            m=2,
            a_rows=2048,
            a_cols=1024,
            d_rows=1024,
            algo="algoId=21 tile=MATMUL_TILE_16x16",
        )
        line_m4 = trace_line(
            m=4,
            a_rows=2048,
            a_cols=1024,
            d_rows=1024,
            algo="algoId=21 tile=MATMUL_TILE_16x16",
        )
        row_m2 = MODULE.parse_cublaslt_trace_line(line_m2)
        row_m4 = MODULE.parse_cublaslt_trace_line(line_m4)
        self.assertNotEqual(row_m2["Bdesc"]["cols"], row_m4["Bdesc"]["cols"])
        self.assertEqual(
            MODULE.algorithm_signature(row_m2, "gate_proj"),
            MODULE.algorithm_signature(row_m4, "gate_proj"),
        )

    def test_workspace_size_alone_is_not_a_regime(self) -> None:
        base = MODULE.parse_cublaslt_trace_line(
            trace_line(
                m=32,
                a_rows=2048,
                a_cols=1024,
                d_rows=1024,
                algo="algoId=21 tile=MATMUL_TILE_32x32",
            )
        )
        changed_workspace = dict(base)
        changed_workspace["workspace_bytes"] = 327680
        self.assertEqual(
            MODULE.algorithm_signature(base, "gate_proj"),
            MODULE.algorithm_signature(changed_workspace, "gate_proj"),
        )

    def test_shape_misalignment_fails_closed(self) -> None:
        rows = [
            MODULE.parse_cublaslt_trace_line(
                trace_line(m=8, a_rows=2048, a_cols=1024, d_rows=1024, algo="algoId=1")
            )
        ] * 3
        with self.assertRaises(MODULE.ProtocolError):
            MODULE.validate_projection_triplet(
                rows, m_value=8, hidden_size=2048, intermediate_size=1024
            )

    def test_non_trace_line_is_ignored(self) -> None:
        self.assertIsNone(MODULE.parse_cublaslt_trace_line("ordinary output"))


class RawEvidenceValidationTest(unittest.TestCase):
    CONFIG = {
        "intervention": {"m_values": [1, 64], "reference_m": 64, "repeats": 10},
        "model": {"hidden_size": 2048, "intermediate_size": 1024},
    }

    @staticmethod
    def _sha(character: str) -> str:
        return character * 64

    def _numeric(self) -> dict:
        changed = self._sha("a")
        reference = self._sha("b")
        return {
            "schema_version": "spectatorroute-phase0a-numeric-cell-v1",
            "cell_id": "victim/L00/E00",
            "victim_id": "victim",
            "document_index": 0,
            "offset": 0,
            "layer": 0,
            "expert_id": 0,
            "hidden_row_sha256": self._sha("c"),
            "all_m_within_bitwise_stable": True,
            "any_cross_m_output_change": True,
            "m_results": [
                {
                    "m": 1,
                    "repeat_count": 10,
                    "repeat_hashes": [changed] * 10,
                    "unique_repeat_hashes": [changed],
                    "within_m_bitwise_stable": True,
                    "representative_sha256": changed,
                    "cross_m_bitwise_equal_to_reference": False,
                    "changed_bf16_elements_to_reference": 1,
                    "max_abs_delta_to_reference": 0.01,
                    "l2_delta_to_reference": 0.01,
                },
                {
                    "m": 64,
                    "repeat_count": 10,
                    "repeat_hashes": [reference] * 10,
                    "unique_repeat_hashes": [reference],
                    "within_m_bitwise_stable": True,
                    "representative_sha256": reference,
                    "cross_m_bitwise_equal_to_reference": True,
                    "changed_bf16_elements_to_reference": 0,
                    "max_abs_delta_to_reference": 0.0,
                    "l2_delta_to_reference": 0.0,
                },
            ],
        }

    def _regime(self, *, m: int, algo_id: int, output_hash: str) -> dict:
        algo = f"algoId={algo_id} customOption=73"
        raw = [
            MODULE.parse_cublaslt_trace_line(
                trace_line(
                    m=m,
                    a_rows=2048,
                    a_cols=1024,
                    d_rows=1024,
                    algo=algo,
                )
            ),
            MODULE.parse_cublaslt_trace_line(
                trace_line(
                    m=m,
                    a_rows=2048,
                    a_cols=1024,
                    d_rows=1024,
                    algo=algo,
                )
            ),
            MODULE.parse_cublaslt_trace_line(
                trace_line(
                    m=m,
                    a_rows=1024,
                    a_cols=2048,
                    d_rows=2048,
                    algo=algo,
                )
            ),
        ]
        roles = ["gate_proj", "up_proj", "down_proj"]
        signatures = [
            MODULE.algorithm_signature(record, role)
            for record, role in zip(raw, roles)
        ]
        return {
            "schema_version": "spectatorroute-phase0a-regime-cell-v1",
            "call_index": 0,
            "cell_id": "victim/L00/E00",
            "victim_id": "victim",
            "layer": 0,
            "expert_id": 0,
            "m": m,
            "trace_row0_output_sha256": output_hash,
            "raw_trace_records": raw,
            "algorithm_signatures": signatures,
            "algorithm_signature_sha256": MODULE.canonical_sha256(signatures),
        }

    def test_raw_numeric_and_trace_form_joint_change(self) -> None:
        numeric_by_m, stable, changed = MODULE.validate_numeric_cell(
            self._numeric(), self.CONFIG
        )
        self.assertTrue(stable and changed)
        regimes = {
            1: self._regime(m=1, algo_id=13, output_hash=self._sha("a")),
            64: self._regime(m=64, algo_id=21, output_hash=self._sha("b")),
        }
        validated = {
            m: regimes[m] | MODULE.validate_regime_cell(regimes[m], self.CONFIG)
            for m in (1, 64)
        }
        result = MODULE.analyze_validated_cell(
            cell_id="victim/L00/E00",
            numeric_by_m=numeric_by_m,
            regime_by_m=validated,
            m_values=[1, 64],
            reference_m=64,
        )
        self.assertEqual(result["joint_changed_ms"], [1])

    def test_missing_repeat_hashes_fail_closed(self) -> None:
        numeric = self._numeric()
        numeric["m_results"][0]["repeat_hashes"] = []
        with self.assertRaises(MODULE.ProtocolError):
            MODULE.validate_numeric_cell(numeric, self.CONFIG)

    def test_fake_stability_flag_fails_closed(self) -> None:
        numeric = self._numeric()
        numeric["m_results"][0]["repeat_hashes"][-1] = self._sha("d")
        with self.assertRaises(MODULE.ProtocolError):
            MODULE.validate_numeric_cell(numeric, self.CONFIG)

    def test_tampered_signature_digest_fails_closed(self) -> None:
        regime = self._regime(m=1, algo_id=13, output_hash=self._sha("a"))
        regime["algorithm_signature_sha256"] = self._sha("e")
        with self.assertRaises(MODULE.ProtocolError):
            MODULE.validate_regime_cell(regime, self.CONFIG)

    def test_cross_process_output_mismatch_fails_closed(self) -> None:
        numeric_by_m, _stable, _changed = MODULE.validate_numeric_cell(
            self._numeric(), self.CONFIG
        )
        regimes = {
            1: self._regime(m=1, algo_id=13, output_hash=self._sha("f")),
            64: self._regime(m=64, algo_id=21, output_hash=self._sha("b")),
        }
        validated = {
            m: regimes[m] | MODULE.validate_regime_cell(regimes[m], self.CONFIG)
            for m in (1, 64)
        }
        with self.assertRaises(MODULE.ProtocolError):
            MODULE.analyze_validated_cell(
                cell_id="victim/L00/E00",
                numeric_by_m=numeric_by_m,
                regime_by_m=validated,
                m_values=[1, 64],
                reference_m=64,
            )

    def test_gate_threshold_cannot_be_edited_to_zero(self) -> None:
        with self.assertRaises(MODULE.ProtocolError):
            MODULE.phase0a_decision(
                unstable_cells=0, positive_victims=0, minimum=0
            )
        self.assertEqual(
            MODULE.phase0a_decision(
                unstable_cells=0, positive_victims=8, minimum=8
            ),
            "PASS_TO_PHASE0B",
        )


class RuntimeIntegrityTest(unittest.TestCase):
    @unittest.skipUnless(
        importlib.util.find_spec("torch") is not None, "torch is unavailable"
    )
    def test_bf16_signed_zero_is_bitwise_different(self) -> None:
        import torch

        positive = torch.tensor([0.0], dtype=torch.bfloat16)
        negative = torch.tensor([-0.0], dtype=torch.bfloat16)
        self.assertTrue(torch.equal(positive, negative))
        self.assertEqual(
            MODULE._bitwise_changed_bf16_elements(positive, negative), 1
        )

    def test_foreign_gpu_process_fails_closed(self) -> None:
        config = {"environment": {"gpu_uuid": "GPU-frozen"}}
        process = {
            "gpu_uuid": "GPU-frozen",
            "pid": 123,
            "process_name": "foreign",
            "used_gpu_memory_mib": "512",
        }
        with mock.patch.object(
            MODULE, "_nvidia_compute_processes", return_value=[process]
        ):
            with self.assertRaises(MODULE.ProtocolError):
                MODULE.assert_no_foreign_gpu_processes(
                    config, allowed_pids=set()
                )
            self.assertEqual(
                MODULE.assert_no_foreign_gpu_processes(
                    config, allowed_pids={123}
                ),
                [process],
            )

    @unittest.skipUnless(
        hasattr(signal, "SIGALRM") and hasattr(signal, "setitimer"),
        "POSIX interval timer is unavailable",
    )
    def test_parent_hard_watchdog_interrupts(self) -> None:
        previous = MODULE._arm_parent_hard_deadline(time.time() + 0.02)
        try:
            with self.assertRaises(TimeoutError):
                time.sleep(0.2)
        finally:
            MODULE._disarm_parent_hard_deadline(previous)

    def test_frozen_semantic_fields_cannot_be_ignored(self) -> None:
        config_path = MODULE_PATH.parent / "configs" / "phase0a_5090_v1.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        MODULE.verify_frozen_semantics(config)
        mutations = (
            ("model", "repo_id", "different/model"),
            ("intervention", "filler", "different_filler"),
            ("gate", "require_actual_regime_signature_change", False),
        )
        for section, field, value in mutations:
            mutated = json.loads(json.dumps(config))
            mutated[section][field] = value
            with self.subTest(section=section, field=field):
                with self.assertRaises(MODULE.ProtocolError):
                    MODULE.verify_frozen_semantics(mutated)

    def test_missing_real_gpu_acceptance_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "REAL_GPU_ACCEPTANCE.json"
            with self.assertRaises(MODULE.ProtocolError):
                MODULE.verify_real_gpu_acceptance_artifact(
                    path=missing,
                    config={},
                    lock_info={},
                    config_path=missing,
                    runner_path=missing,
                )


class FrozenLockTest(unittest.TestCase):
    FILES = [
        "docs/ideas/spectatorroute/N05_PHASE0_FROZEN_PROTOCOL_20260729.md",
        "docs/ideas/spectatorroute/experiments/configs/phase0a_5090_v1.json",
        "docs/ideas/spectatorroute/experiments/run_phase0a_5090.py",
        "docs/ideas/spectatorroute/experiments/test_phase0a_5090.py",
    ]
    CONSTANTS = {
        "victim_count": 64,
        "document_count": 32,
        "token_offsets": [0, 256],
        "window_tokens": 16,
        "victim_position": 15,
        "m_values": [1, 2, 4, 8, 16, 32, 64],
        "reference_m": 64,
        "repeats": 10,
        "minimum_distinct_positive_victims": 8,
        "max_gpu_seconds": 1800,
    }

    def test_edit_after_seal_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, relative in enumerate(self.FILES):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"frozen-{index}\n", encoding="utf-8")
            lock = {
                "schema_version": "spectatorroute-phase0a-frozen-lock-v1",
                "status": "FROZEN_PRE_RUN",
                "files": {
                    relative: MODULE.sha256_file(root / relative)
                    for relative in self.FILES
                },
                "frozen_constants": self.CONSTANTS,
            }
            lock_path = root / "lock.json"
            lock_path.write_text(
                json.dumps(lock, sort_keys=True) + "\n", encoding="utf-8"
            )
            lock_hash = MODULE.sha256_file(lock_path)
            MODULE.verify_frozen_lock(
                lock_path=lock_path,
                expected_lock_sha256=lock_hash,
                repo_root=root,
            )
            edited = root / self.FILES[1]
            edited.write_text("minimum=0\n", encoding="utf-8")
            with self.assertRaises(MODULE.ProtocolError):
                MODULE.verify_frozen_lock(
                    lock_path=lock_path,
                    expected_lock_sha256=lock_hash,
                    repo_root=root,
                )


if __name__ == "__main__":
    unittest.main()
