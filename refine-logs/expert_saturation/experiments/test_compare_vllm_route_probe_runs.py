from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np


MODULE_PATH = Path(__file__).with_name("compare_vllm_route_probe_runs.py")
SPEC = importlib.util.spec_from_file_location("compare_route_probe_hardened", MODULE_PATH)
assert SPEC and SPEC.loader
compare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compare)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True))


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _read_rows(run_dir: Path) -> list[dict]:
    return [json.loads(line) for line in (run_dir / "batches.jsonl").read_text().splitlines()]


def _write_rows(run_dir: Path, rows: list[dict]) -> None:
    (run_dir / "batches.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )


def _seal_top_level(run_dir: Path) -> None:
    seal = {
        "status": "RUN_COMPLETE",
        "config_sha256": hashlib.sha256((run_dir / "config.json").read_bytes()).hexdigest(),
        "raw_sha256": hashlib.sha256((run_dir / "batches.jsonl").read_bytes()).hexdigest(),
        "summary_sha256": hashlib.sha256((run_dir / "summary.json").read_bytes()).hexdigest(),
        "artifact_hashes_sha256": hashlib.sha256(
            (run_dir / "ARTIFACT_HASHES.json").read_bytes()
        ).hexdigest(),
    }
    _write_json(run_dir / "RUN_COMPLETE.json", seal)


def _refresh_artifact_hash(run_dir: Path, relative: str) -> str:
    manifest = _read_json(run_dir / "ARTIFACT_HASHES.json")
    digest = hashlib.sha256((run_dir / relative).read_bytes()).hexdigest()
    manifest[relative] = digest
    _write_json(run_dir / "ARTIFACT_HASHES.json", manifest)
    return digest


def _request_metrics(tpot_ms: float) -> list[dict]:
    return [
        {
            "request_id": str(request),
            "generated_tokens": 4,
            "ttft_ms": 1.0 + request,
            "queue_ms": 0.0,
            "decode_span_ms": 3.0 * tpot_ms,
            "tpot_ms": tpot_ms,
            "token_ids": [10 * (request + 1) + token for token in range(4)],
            "finish_reason": "length",
        }
        for request in range(2)
    ]


def _timing_summary(wall_ms: float, tpot_ms: float) -> dict:
    throughput = 8.0 / (wall_ms / 1000.0) if wall_ms > 0 else 0.0
    return {
        "wall_ms": wall_ms,
        "throughput_tokens_per_s": throughput,
        "request_tpot_p50_ms": tpot_ms,
        "request_tpot_p95_ms": tpot_ms,
        "request_tpot_max_ms": tpot_ms,
        "request_ttft_p50_ms": 1.5,
        "request_ttft_p95_ms": 1.95,
        "request_queue_p95_ms": 0.0,
    }


def _make_bundle(
    root: Path,
    name: str,
    capture_routes: bool,
    wall_ms: float = 10.0,
    tpot_ms: float = 2.0,
) -> Path:
    run_dir = root / name
    (run_dir / "inputs").mkdir(parents=True)
    (run_dir / "routes").mkdir()
    environment = {
        "python": "3.12-test",
        "platform": "linux-test",
        "torch": "test",
        "torch_cuda": "test",
        "vllm": "0.26.0",
        "gpu": "test-gpu",
        "vllm_batch_invariant": "0",
        "vllm_use_flashinfer_sampler": "0",
        "compute_processes_before_engine_init": [],
    }
    runtime_identity = {
        key: environment[key]
        for key in environment
        if key != "compute_processes_before_engine_init"
    }
    model_shape = {"num_experts": 4, "num_layers": 2, "top_k": 2}
    workload = {"requests": [{"prompt": "one"}, {"prompt": "two"}]}
    workload_bytes = json.dumps(workload, sort_keys=True).encode()
    (run_dir / "workload_manifest.json").write_bytes(workload_bytes)
    producer_bytes = Path(__file__).with_name(
        "run_vllm_route_shape_probe.py"
    ).read_bytes()
    producer_hash = hashlib.sha256(producer_bytes).hexdigest()
    (run_dir / "producer_source.py").write_bytes(producer_bytes)
    config = {
        "model": "test/model",
        "revision": "frozen",
        "dtype": "bfloat16",
        "batch_sizes": [2],
        "prompt_lengths": [3],
        "output_tokens": 4,
        "groups": 1,
        "within_process_repeats": 1,
        "process_repeat": 0,
        "seed": 7,
        "order_seed": 8,
        "max_model_len": 16,
        "max_num_seqs": 2,
        "max_num_batched_tokens": 16,
        "gpu_memory_utilization": 0.8,
        "runtime_patch_id": "test-patch",
        "enforce_eager": True,
        "require_exclusive_gpu": True,
        "capture_routes": capture_routes,
        "workload_manifest_sha256": hashlib.sha256(workload_bytes).hexdigest(),
        "probe_script_sha256": producer_hash,
        "producer_source_artifact": "producer_source.py",
        "producer_source_artifact_sha256": producer_hash,
        "runtime_identity": runtime_identity,
        "model_shape": model_shape,
        "claim_ceiling": "NATIVE_OFFLINE_FIXED_BATCH_MEASUREMENT_ONLY",
    }
    _write_json(run_dir / "environment.json", environment)
    _write_json(run_dir / "model_shape.json", model_shape)
    _write_json(run_dir / "config.json", config)

    prompts = np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    input_relative = "inputs/batch.npz"
    np.savez_compressed(run_dir / input_relative, prompt_token_ids=prompts)
    input_hash = hashlib.sha256((run_dir / input_relative).read_bytes()).hexdigest()
    row = {
        "batch_id": "batch",
        "execution_order": 0,
        "process_repeat": 0,
        "within_process_repeat": 0,
        "prompt_length": 3,
        "batch_size": 2,
        "group": 0,
        "prompt_token_ids_sha256": compare._json_hash(prompts.tolist()),
        "input_artifact": input_relative,
        "input_artifact_sha256": input_hash,
        "request_metrics": _request_metrics(tpot_ms),
        "timing": _timing_summary(wall_ms, tpot_ms),
    }
    manifest = {
        relative: hashlib.sha256((run_dir / relative).read_bytes()).hexdigest()
        for relative in (
            "environment.json",
            "model_shape.json",
            "workload_manifest.json",
            "producer_source.py",
            input_relative,
        )
    }
    if capture_routes:
        routes = np.empty((2, 3, 2, 2), dtype=np.int16)
        routes[..., 0] = 0
        routes[..., 1] = 1
        route_relative = "routes/batch.npz"
        np.savez_compressed(run_dir / route_relative, routes=routes)
        route_hash = hashlib.sha256((run_dir / route_relative).read_bytes()).hexdigest()
        manifest[route_relative] = route_hash
        row.update(
            {
                "route_artifact": route_relative,
                "route_artifact_sha256": route_hash,
                "route": {
                    "batch_size": 2,
                    "decode_route_steps": 3,
                    "num_layers": 2,
                    "top_k": 2,
                    "num_experts": 4,
                    "total_assignments": 24,
                },
            }
        )
    _write_rows(run_dir, [row])
    summary = {
        "schema": "vllm-native-route-shape-probe-v1",
        "status": "COMPLETE",
        "claim_ceiling": "NATIVE_OFFLINE_FIXED_BATCH_MEASUREMENT_ONLY",
        "capture_routes": capture_routes,
        "record_count": 1,
        "cell_summaries": [{"prompt_length": 3, "batch_size": 2, "samples": 1}],
    }
    _write_json(run_dir / "summary.json", summary)
    _write_json(run_dir / "ARTIFACT_HASHES.json", manifest)
    _seal_top_level(run_dir)
    return run_dir


class BundleIntegrityTest(unittest.TestCase):
    def test_valid_sealed_on_and_off_bundles(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.assertTrue(compare.verify_bundle(_make_bundle(root, "on", True))["valid"])
            self.assertTrue(compare.verify_bundle(_make_bundle(root, "off", False))["valid"])

    def test_malformed_route_is_rejected_after_resealing(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run_dir = _make_bundle(Path(temp), "on", True)
            malformed = np.zeros((2, 3, 2, 2), dtype=np.int16)
            np.savez_compressed(run_dir / "routes/batch.npz", routes=malformed)
            digest = _refresh_artifact_hash(run_dir, "routes/batch.npz")
            rows = _read_rows(run_dir)
            rows[0]["route_artifact_sha256"] = digest
            _write_rows(run_dir, rows)
            _seal_top_level(run_dir)

            result = compare.verify_bundle(run_dir)
            self.assertFalse(result["valid"])
            self.assertTrue(any("duplicate expert" in error for error in result["errors"]))

    def test_missing_row_artifact_is_rejected_after_resealing(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run_dir = _make_bundle(Path(temp), "on", True)
            (run_dir / "inputs/batch.npz").unlink()
            _seal_top_level(run_dir)

            result = compare.verify_bundle(run_dir)
            self.assertFalse(result["valid"])
            self.assertTrue(any("artifact_missing" in error for error in result["errors"]))

    def test_path_traversal_is_rejected_after_resealing(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run_dir = _make_bundle(Path(temp), "on", True)
            rows = _read_rows(run_dir)
            old_relative = rows[0]["input_artifact"]
            rows[0]["input_artifact"] = "../escape.npz"
            manifest = _read_json(run_dir / "ARTIFACT_HASHES.json")
            manifest["../escape.npz"] = manifest.pop(old_relative)
            _write_rows(run_dir, rows)
            _write_json(run_dir / "ARTIFACT_HASHES.json", manifest)
            _seal_top_level(run_dir)

            result = compare.verify_bundle(run_dir)
            self.assertFalse(result["valid"])
            self.assertTrue(any("unsafe artifact path" in error for error in result["errors"]))

    def test_zero_timing_is_rejected_after_resealing(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run_dir = _make_bundle(Path(temp), "on", True)
            rows = _read_rows(run_dir)
            rows[0]["timing"]["wall_ms"] = 0.0
            _write_rows(run_dir, rows)
            _seal_top_level(run_dir)

            result = compare.verify_bundle(run_dir)
            self.assertFalse(result["valid"])
            self.assertTrue(any("nonpositive" in error for error in result["errors"]))

    def test_route_off_cannot_reference_route_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run_dir = _make_bundle(Path(temp), "off", False)
            rows = _read_rows(run_dir)
            rows[0]["route_artifact"] = "routes/phantom.npz"
            rows[0]["route_artifact_sha256"] = "b" * 64
            _write_rows(run_dir, rows)
            _seal_top_level(run_dir)

            result = compare.verify_bundle(run_dir)
            self.assertFalse(result["valid"])
            self.assertTrue(any("route-OFF" in error for error in result["errors"]))

    def test_runtime_environment_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run_dir = _make_bundle(Path(temp), "on", True)
            config = _read_json(run_dir / "config.json")
            config["runtime_identity"]["vllm"] = "different"
            _write_json(run_dir / "config.json", config)
            _seal_top_level(run_dir)

            result = compare.verify_bundle(run_dir)
            self.assertFalse(result["valid"])
            self.assertIn("runtime_environment:vllm", result["errors"])

    def test_declared_producer_cannot_be_removed_and_resealed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run_dir = _make_bundle(Path(temp), "on", True)
            (run_dir / "producer_source.py").unlink()
            manifest = _read_json(run_dir / "ARTIFACT_HASHES.json")
            manifest.pop("producer_source.py")
            _write_json(run_dir / "ARTIFACT_HASHES.json", manifest)
            _seal_top_level(run_dir)

            result = compare.verify_bundle(run_dir)
            self.assertFalse(result["valid"])
            self.assertEqual(result["producer_source_status"], "PRODUCER_SOURCE_INVALID")
            self.assertFalse(result["producer_source_verified"])

    def test_historical_missing_producer_is_explicitly_unverified(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run_dir = _make_bundle(Path(temp), "on", True)
            config = _read_json(run_dir / "config.json")
            config.pop("producer_source_artifact")
            config.pop("producer_source_artifact_sha256")
            _write_json(run_dir / "config.json", config)
            (run_dir / "producer_source.py").unlink()
            manifest = _read_json(run_dir / "ARTIFACT_HASHES.json")
            manifest.pop("producer_source.py")
            _write_json(run_dir / "ARTIFACT_HASHES.json", manifest)
            _seal_top_level(run_dir)

            result = compare.verify_bundle(run_dir)
            self.assertTrue(result["valid"], result["errors"])
            self.assertEqual(
                result["producer_source_status"],
                "PRODUCER_SOURCE_UNVERIFIED_HISTORICAL",
            )
            self.assertFalse(result["producer_source_verified"])

    def test_resealed_forged_timing_summary_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run_dir = _make_bundle(Path(temp), "on", True)
            rows = _read_rows(run_dir)
            rows[0]["timing"]["request_tpot_p95_ms"] = 0.25
            _write_rows(run_dir, rows)
            _seal_top_level(run_dir)

            result = compare.verify_bundle(run_dir)
            self.assertFalse(result["valid"])
            self.assertFalse(result["timing_evidence_verified"])
            self.assertTrue(
                any("timing summary mismatch" in error for error in result["errors"])
            )

    def test_missing_per_request_timing_field_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run_dir = _make_bundle(Path(temp), "on", True)
            rows = _read_rows(run_dir)
            rows[0]["request_metrics"][0].pop("decode_span_ms")
            _write_rows(run_dir, rows)
            _seal_top_level(run_dir)

            result = compare.verify_bundle(run_dir)
            self.assertFalse(result["valid"])
            self.assertTrue(any("decode_span_ms" in error for error in result["errors"]))

    def test_truncated_npz_returns_structured_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run_dir = _make_bundle(Path(temp), "on", True)
            (run_dir / "routes/batch.npz").write_bytes(b"PK")
            digest = _refresh_artifact_hash(run_dir, "routes/batch.npz")
            rows = _read_rows(run_dir)
            rows[0]["route_artifact_sha256"] = digest
            _write_rows(run_dir, rows)
            _seal_top_level(run_dir)

            result = compare.verify_bundle(run_dir)
            self.assertFalse(result["valid"])
            self.assertTrue(any("invalid NPZ" in error for error in result["errors"]))


class PairGateTest(unittest.TestCase):
    def _pair_payload(self, on_wall: float, off_wall: float) -> tuple[dict, dict, dict, dict]:
        config = {field: "same" for field in compare.MATCHED_CONFIG_FIELDS}
        config.update(
            {
                "prompt_lengths": [3],
                "batch_sizes": [2],
                "groups": 1,
                "within_process_repeats": 1,
                "output_tokens": 2,
                "capture_routes": False,
            }
        )
        row = {
            "prompt_length": 3,
            "batch_size": 2,
            "group": 0,
            "within_process_repeat": 0,
            "prompt_token_ids_sha256": "same",
            "request_metrics": [
                {
                    "request_id": str(request),
                    "generated_tokens": 2,
                    "ttft_ms": 1.0,
                    "queue_ms": 0.0,
                    "decode_span_ms": off_wall,
                    "tpot_ms": off_wall,
                    "token_ids": [1 + request, 2 + request],
                    "finish_reason": "length",
                }
                for request in range(2)
            ],
            "timing": {
                "wall_ms": off_wall,
                "throughput_tokens_per_s": 4.0 / (off_wall / 1000.0)
                if off_wall > 0
                else 0.0,
                "request_tpot_p50_ms": off_wall,
                "request_tpot_p95_ms": off_wall,
                "request_tpot_max_ms": off_wall,
                "request_ttft_p50_ms": 1.0,
                "request_ttft_p95_ms": 1.0,
                "request_queue_p95_ms": 0.0,
            },
        }
        on_row = json.loads(json.dumps(row))
        for request in on_row["request_metrics"]:
            request["decode_span_ms"] = on_wall
            request["tpot_ms"] = on_wall
        on_row["timing"].update(
            {
                "wall_ms": on_wall,
                "throughput_tokens_per_s": 4.0 / (on_wall / 1000.0)
                if on_wall > 0
                else 0.0,
                "request_tpot_p50_ms": on_wall,
                "request_tpot_p95_ms": on_wall,
                "request_tpot_max_ms": on_wall,
            }
        )
        return dict(config, capture_routes=True), config, on_row, row

    def test_large_negative_delta_fails_absolute_deviation_gate(self) -> None:
        on_config, off_config, on_row, off_row = self._pair_payload(1.0, 10.0)
        report = compare.compare_runs(on_config, off_config, [on_row], [off_row], 5.0)
        self.assertEqual(
            report["status"], "ROUTE_EXPORT_TOO_EXPENSIVE_FOR_TIMING_CLAIM"
        )
        self.assertAlmostEqual(report["wall_overhead_p95_pct"], -90.0)
        self.assertAlmostEqual(report["wall_absolute_deviation_p95_pct"], 90.0)

    def test_zero_timing_and_invalid_threshold_fail_closed(self) -> None:
        on_config, off_config, on_row, off_row = self._pair_payload(0.0, 10.0)
        zero = compare.compare_runs(on_config, off_config, [on_row], [off_row], 5.0)
        self.assertEqual(zero["status"], "INVALID_TELEMETRY_PAIR")
        self.assertEqual(zero["timing_errors"], [[3, 2, 0, 0]])

        on_config, off_config, on_row, off_row = self._pair_payload(10.0, 10.0)
        threshold = compare.compare_runs(
            on_config, off_config, [on_row], [off_row], float("nan")
        )
        self.assertEqual(threshold["status"], "INVALID_TELEMETRY_PAIR")
        self.assertFalse(threshold["threshold_valid"])

        noncanonical = compare.compare_runs(
            on_config, off_config, [on_row], [off_row], 100.0
        )
        self.assertEqual(noncanonical["status"], "INVALID_TELEMETRY_PAIR")
        self.assertFalse(noncanonical["threshold_valid"])

    def test_require_exclusive_gpu_is_matched(self) -> None:
        on_config, off_config, on_row, off_row = self._pair_payload(10.0, 10.0)
        on_config["require_exclusive_gpu"] = False
        report = compare.compare_runs(on_config, off_config, [on_row], [off_row], 5.0)
        self.assertEqual(report["status"], "INVALID_TELEMETRY_PAIR")
        self.assertIn("require_exclusive_gpu", report["config_drift"])

    def test_cli_exit_codes_are_semantic(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            off = _make_bundle(root, "off", False, wall_ms=10.0, tpot_ms=2.0)
            for name, wall, tpot, expected in (
                ("qualified", 10.0, 2.0, 0),
                ("nonqualified", 20.0, 4.0, 1),
                ("invalid", 0.0, 2.0, 2),
            ):
                on = _make_bundle(root, name, True, wall_ms=wall, tpot_ms=tpot)
                output = root / f"{name}.json"
                result = subprocess.run(
                    [
                        sys.executable,
                        str(MODULE_PATH),
                        "--route-on",
                        str(on),
                        "--route-off",
                        str(off),
                        "--output",
                        str(output),
                        "--max-p95-overhead-pct",
                        "5",
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, expected, result.stderr)

            invalid_threshold_output = root / "invalid-threshold.json"
            invalid_threshold = subprocess.run(
                [
                    sys.executable,
                    str(MODULE_PATH),
                    "--route-on",
                    str(root / "qualified"),
                    "--route-off",
                    str(off),
                    "--output",
                    str(invalid_threshold_output),
                    "--max-p95-overhead-pct",
                    "nan",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(invalid_threshold.returncode, 2, invalid_threshold.stderr)
            report = _read_json(invalid_threshold_output)
            self.assertIsNone(report["thresholds"]["max_p95_absolute_deviation_pct"])

    def test_cli_never_qualifies_clean_historical_source_unverified_pair(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            on = _make_bundle(root, "historical-on", True)
            off = _make_bundle(root, "historical-off", False)
            for run_dir in (on, off):
                config = _read_json(run_dir / "config.json")
                config.pop("producer_source_artifact")
                config.pop("producer_source_artifact_sha256")
                _write_json(run_dir / "config.json", config)
                (run_dir / "producer_source.py").unlink()
                manifest = _read_json(run_dir / "ARTIFACT_HASHES.json")
                manifest.pop("producer_source.py")
                _write_json(run_dir / "ARTIFACT_HASHES.json", manifest)
                _seal_top_level(run_dir)

            output = root / "historical-comparison.json"
            result = subprocess.run(
                [
                    sys.executable,
                    str(MODULE_PATH),
                    "--route-on",
                    str(on),
                    "--route-off",
                    str(off),
                    "--output",
                    str(output),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            report = _read_json(output)

        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(
            report["status"], "PROVENANCE_OR_ENVIRONMENT_UNQUALIFIED"
        )
        self.assertFalse(
            report["qualification_prerequisites"]["producer_source_verified"]
        )

    def test_cli_never_qualifies_self_consistent_unknown_producer(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            on = _make_bundle(root, "future-on", True)
            off = _make_bundle(root, "future-off", False)
            future_bytes = b"# FUTURE PRODUCER WITH UNKNOWN SEMANTICS\n"
            future_hash = hashlib.sha256(future_bytes).hexdigest()
            for run_dir in (on, off):
                (run_dir / "producer_source.py").write_bytes(future_bytes)
                config = _read_json(run_dir / "config.json")
                config["probe_script_sha256"] = future_hash
                config["producer_source_artifact_sha256"] = future_hash
                _write_json(run_dir / "config.json", config)
                manifest = _read_json(run_dir / "ARTIFACT_HASHES.json")
                manifest["producer_source.py"] = future_hash
                _write_json(run_dir / "ARTIFACT_HASHES.json", manifest)
                _seal_top_level(run_dir)

            self.assertTrue(compare.verify_bundle(on)["producer_source_verified"])
            self.assertFalse(
                compare.verify_bundle(on)["producer_source_semantics_approved"]
            )
            output = root / "future-comparison.json"
            result = subprocess.run(
                [
                    sys.executable,
                    str(MODULE_PATH),
                    "--route-on",
                    str(on),
                    "--route-off",
                    str(off),
                    "--output",
                    str(output),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            report = _read_json(output)

        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(
            report["status"], "PROVENANCE_OR_ENVIRONMENT_UNQUALIFIED"
        )
        self.assertFalse(
            report["qualification_prerequisites"][
                "producer_source_semantics_approved"
            ]
        )


if __name__ == "__main__":
    unittest.main()
