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


MODULE_PATH = Path(__file__).with_name("analyze_vllm_route_probe_bundles.py")
SPEC = importlib.util.spec_from_file_location("route_probe_analysis", MODULE_PATH)
assert SPEC and SPEC.loader
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_bundle(
    root: Path,
    *,
    capture_routes: bool,
    process_repeat: int,
    token_offset: int = 0,
    route_overhead: float = 1.12,
    reverse_on_tpot: bool = False,
    include_producer: bool = True,
    require_exclusive_gpu: bool = True,
) -> Path:
    kind = "on" if capture_routes else "off"
    run_dir = root / f"{kind}-r{process_repeat}"
    run_dir.mkdir()
    (run_dir / "inputs").mkdir()
    if capture_routes:
        (run_dir / "routes").mkdir()

    (run_dir / "workload_manifest.json").write_text(json.dumps({"requests": []}))
    claim_ceiling = "NATIVE_OFFLINE_FIXED_BATCH_MEASUREMENT_ONLY"

    config = {
        "model": "test/olmoe",
        "revision": "fixed",
        "dtype": "bfloat16",
        "batch_sizes": [2, 4],
        "prompt_lengths": [64],
        "output_tokens": 4,
        "groups": 6,
        "within_process_repeats": 1,
        "process_repeat": process_repeat,
        "seed": 7,
        "order_seed": 11,
        "max_model_len": 128,
        "max_num_seqs": 4,
        "max_num_batched_tokens": 512,
        "gpu_memory_utilization": 0.75,
        "enforce_eager": True,
        "capture_routes": capture_routes,
        "claim_ceiling": claim_ceiling,
        "require_exclusive_gpu": require_exclusive_gpu,
        "workload_manifest_sha256": _hash(run_dir / "workload_manifest.json"),
        "probe_script_sha256": "probe-sha",
        "runtime_identity": {"vllm": "0.26.0", "gpu": "test-gpu"},
        "model_shape": {"num_experts": 64, "num_layers": 2, "top_k": 2},
    }
    (run_dir / "config.json").write_text(json.dumps(config, sort_keys=True))
    environment = {
        "vllm": "0.26.0",
        "gpu": "test-gpu",
        "compute_processes_before_engine_init": [],
    }
    (run_dir / "environment.json").write_text(json.dumps(environment))
    (run_dir / "model_shape.json").write_text(json.dumps(config["model_shape"]))

    if include_producer:
        producer_bytes = Path(__file__).with_name(
            "run_vllm_route_shape_probe.py"
        ).read_bytes()
        producer_hash = hashlib.sha256(producer_bytes).hexdigest()
        (run_dir / "producer_source.py").write_bytes(producer_bytes)
        config["probe_script_sha256"] = producer_hash
        config["producer_source_artifact"] = "producer_source.py"
        config["producer_source_artifact_sha256"] = producer_hash
        (run_dir / "config.json").write_text(json.dumps(config, sort_keys=True))

    artifact_hashes = {
        name: _hash(run_dir / name)
        for name in ("environment.json", "model_shape.json", "workload_manifest.json")
    }
    if include_producer:
        artifact_hashes["producer_source.py"] = _hash(
            run_dir / "producer_source.py"
        )
    rows = []
    for batch_size in config["batch_sizes"]:
        for group in range(config["groups"]):
            off_tpot = 10.0 + group + batch_size * 0.1
            if capture_routes and reverse_on_tpot:
                observed_tpot = 30.0 - off_tpot
            else:
                observed_tpot = off_tpot * route_overhead if capture_routes else off_tpot
            batch_id = f"r{process_repeat:02d}-p64-b{batch_size}-g{group:02d}-w00"
            prompt_ids = np.asarray(
                [
                    [1000 * group + 100 * request + token for token in range(64)]
                    for request in range(batch_size)
                ],
                dtype=np.int32,
            )
            input_path = run_dir / "inputs" / f"{batch_id}.npz"
            np.savez_compressed(input_path, prompt_token_ids=prompt_ids)
            input_relative = str(input_path.relative_to(run_dir))
            input_hash = _hash(input_path)
            artifact_hashes[input_relative] = input_hash
            row = {
                "batch_id": batch_id,
                "execution_order": len(rows),
                "process_repeat": process_repeat,
                "within_process_repeat": 0,
                "prompt_length": 64,
                "batch_size": batch_size,
                "group": group,
                "prompt_token_ids_sha256": analysis.PRODUCER._json_hash(
                    prompt_ids.tolist()
                ),
                "input_artifact": input_relative,
                "input_artifact_sha256": input_hash,
                "request_metrics": [
                    {
                        "request_id": f"{batch_id}-{request}",
                        "generated_tokens": 4,
                        "ttft_ms": 2.0 + request,
                        "queue_ms": 0.0,
                        "decode_span_ms": 3.0 * observed_tpot,
                        "tpot_ms": observed_tpot,
                        "token_ids": [batch_size, group, token_offset, request],
                        "finish_reason": "length",
                    }
                    for request in range(batch_size)
                ],
                "timing": {
                    "wall_ms": observed_tpot * batch_size,
                    "throughput_tokens_per_s": 4000.0 / observed_tpot,
                    "request_tpot_p50_ms": observed_tpot,
                    "request_tpot_p95_ms": observed_tpot,
                    "request_tpot_max_ms": observed_tpot,
                    "request_ttft_p50_ms": float(
                        np.quantile([2.0 + request for request in range(batch_size)], 0.50)
                    ),
                    "request_ttft_p95_ms": float(
                        np.quantile([2.0 + request for request in range(batch_size)], 0.95)
                    ),
                    "request_queue_p95_ms": 0.0,
                },
            }
            if capture_routes:
                route_path = run_dir / "routes" / f"{batch_id}.npz"
                # Use a progressively smaller expert pool so raw NPZ-derived
                # concentration increases with group while every top-k row
                # keeps unique expert IDs.
                pool_size = max(2, 2 * batch_size - group)
                routes = np.empty((batch_size, 3, 2, 2), dtype=np.int16)
                for request in range(batch_size):
                    routes[request, ..., 0] = (2 * request) % pool_size
                    routes[request, ..., 1] = (2 * request + 1) % pool_size
                np.savez_compressed(route_path, routes=routes)
                relative = str(route_path.relative_to(run_dir))
                route_hash = _hash(route_path)
                artifact_hashes[relative] = route_hash
                row.update(
                    {
                        "route_artifact": relative,
                        "route_artifact_sha256": route_hash,
                        "route": analysis.PRODUCER.summarize_routes(
                            [routes[index] for index in range(batch_size)], 64
                        ),
                    }
                )
            rows.append(row)

    raw = run_dir / "batches.jsonl"
    raw.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    summary = {
        "schema": "vllm-native-route-shape-probe-v1",
        "status": "COMPLETE",
        "claim_ceiling": claim_ceiling,
        "capture_routes": capture_routes,
        "record_count": len(rows),
        "cell_summaries": [
            {
                "prompt_length": 64,
                "batch_size": batch_size,
                "samples": config["groups"] * config["within_process_repeats"],
            }
            for batch_size in config["batch_sizes"]
        ],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, sort_keys=True))
    (run_dir / "ARTIFACT_HASHES.json").write_text(
        json.dumps(artifact_hashes, sort_keys=True)
    )
    seal = {
        "status": "RUN_COMPLETE",
        "config_sha256": _hash(run_dir / "config.json"),
        "raw_sha256": _hash(raw),
        "summary_sha256": _hash(run_dir / "summary.json"),
        "artifact_hashes_sha256": _hash(run_dir / "ARTIFACT_HASHES.json"),
    }
    (run_dir / "RUN_COMPLETE.json").write_text(json.dumps(seal, sort_keys=True))
    return run_dir


class RouteProbePivotAnalysisTest(unittest.TestCase):
    def test_too_expensive_route_export_cannot_select_action(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            route_on = [
                _write_bundle(
                    root,
                    capture_routes=True,
                    process_repeat=repeat,
                    reverse_on_tpot=True,
                )
                for repeat in range(2)
            ]
            route_off = [
                _write_bundle(root, capture_routes=False, process_repeat=repeat)
                for repeat in range(2)
            ]
            report = analysis.analyze_bundles(route_on, route_off)

        self.assertEqual(report["status"], "COMPLETE")
        self.assertEqual(report["timing_source_for_decision"], "ROUTE_ON_DIAGNOSTIC_ONLY")
        self.assertEqual(report["pivot_verdict"], "WORKING_SET_MEASUREMENT_ONLY")
        self.assertEqual(
            report["failure_category"], "TELEMETRY_TIMING_DEVIATION_FAILED"
        )
        self.assertFalse(report["decision_gates"]["paired_route_OFF_timing"])
        self.assertEqual(
            report["telemetry_transparency"]["status"],
            "FAILED_TIMING_DEVIATION",
        )
        self.assertEqual(
            {pair["status"] for pair in report["telemetry_pairs"]},
            {"ROUTE_EXPORT_TOO_EXPENSIVE_FOR_TIMING_CLAIM"},
        )

    def test_all_qualified_repeats_select_only_next_action_test(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            route_on = [
                _write_bundle(
                    root,
                    capture_routes=True,
                    process_repeat=repeat,
                    route_overhead=1.02,
                )
                for repeat in range(2)
            ]
            route_off = [
                _write_bundle(root, capture_routes=False, process_repeat=repeat)
                for repeat in range(2)
            ]
            report = analysis.analyze_bundles(route_on, route_off)

        self.assertEqual(report["status"], "COMPLETE")
        self.assertEqual(
            report["timing_source_for_decision"], "PAIRED_ROUTE_OFF_ALL_REPEATS"
        )
        self.assertEqual(report["pivot_verdict"], "TEST_MARGINAL_PRESSURE_ACTION")
        self.assertTrue(report["decision_gates"]["composition_association"])
        self.assertTrue(report["decision_gates"]["action_stability"])
        self.assertTrue(
            report["decision_gates"]["producer_source_semantics_approved"]
        )
        self.assertTrue(report["decision_gates"]["exclusive_gpu_verified"])
        self.assertEqual(
            {pair["status"] for pair in report["telemetry_pairs"]},
            {"TELEMETRY_OVERHEAD_QUALIFIED"},
        )

    def test_route_on_timing_cannot_select_action(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            route_on = [
                _write_bundle(root, capture_routes=True, process_repeat=repeat)
                for repeat in range(2)
            ]
            report = analysis.analyze_bundles(route_on)

        self.assertEqual(report["timing_source_for_decision"], "ROUTE_ON_DIAGNOSTIC_ONLY")
        self.assertEqual(report["pivot_verdict"], "WORKING_SET_MEASUREMENT_ONLY")
        self.assertFalse(report["decision_gates"]["paired_route_OFF_timing"])
        self.assertIn("instrumented diagnostic timing", report["anti_claims"][2])

    def test_token_drift_disqualifies_timing_but_preserves_structural_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            route_on = [
                _write_bundle(root, capture_routes=True, process_repeat=repeat)
                for repeat in range(2)
            ]
            route_off = [
                _write_bundle(
                    root,
                    capture_routes=False,
                    process_repeat=repeat,
                    token_offset=1,
                )
                for repeat in range(2)
            ]
            report = analysis.analyze_bundles(route_on, route_off)

        self.assertEqual(report["status"], "COMPLETE")
        self.assertEqual(report["pivot_verdict"], "WORKING_SET_MEASUREMENT_ONLY")
        self.assertEqual(report["failure_category"], "TELEMETRY_TRANSPARENCY_FAILED")
        self.assertEqual(report["timing_source_for_decision"], "ROUTE_ON_DIAGNOSTIC_ONLY")
        self.assertEqual(report["telemetry_transparency"]["status"], "FAILED_TOKEN_DRIFT")
        self.assertEqual(report["telemetry_transparency"]["qualified_route_OFF_process_repeats"], [])
        self.assertEqual(
            {pair["status"] for pair in report["telemetry_pairs"]},
            {"TELEMETRY_TOKEN_DRIFT"},
        )
        self.assertEqual(
            report["structural_evidence_scope"],
            "TELEMETRY_CONDITIONED_ROUTE_STRUCTURE_ONLY",
        )

    def test_one_drift_repeat_disqualifies_every_clean_repeat(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            route_on = [
                _write_bundle(root, capture_routes=True, process_repeat=repeat)
                for repeat in range(3)
            ]
            route_off = [
                _write_bundle(
                    root,
                    capture_routes=False,
                    process_repeat=repeat,
                    token_offset=1 if repeat == 2 else 0,
                )
                for repeat in range(3)
            ]
            report = analysis.analyze_bundles(route_on, route_off)

        self.assertEqual(report["status"], "COMPLETE")
        self.assertEqual(report["pivot_verdict"], "WORKING_SET_MEASUREMENT_ONLY")
        self.assertEqual(report["timing_source_for_decision"], "ROUTE_ON_DIAGNOSTIC_ONLY")
        self.assertFalse(report["decision_gates"]["paired_route_OFF_timing"])
        self.assertEqual(
            report["telemetry_transparency"]["qualified_route_OFF_process_repeats"],
            [],
        )
        self.assertEqual(len(report["telemetry_pairs"]), 3)

    def test_resealed_inconsistent_route_metrics_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            route_on = _write_bundle(root, capture_routes=True, process_repeat=0)
            rows = [json.loads(line) for line in (route_on / "batches.jsonl").read_text().splitlines()]
            rows[0]["route"]["mean_layer_step_concentration"] += 0.1
            (route_on / "batches.jsonl").write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
            )
            seal = json.loads((route_on / "RUN_COMPLETE.json").read_text())
            seal["raw_sha256"] = _hash(route_on / "batches.jsonl")
            (route_on / "RUN_COMPLETE.json").write_text(json.dumps(seal))
            report = analysis.analyze_bundles([route_on])

        self.assertEqual(report["status"], "INVALID_INPUT")
        self.assertTrue(
            any("route metric mismatch" in error for error in report["validation_errors"])
        )

    def test_tampered_seal_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            route_on = _write_bundle(root, capture_routes=True, process_repeat=0)
            with (route_on / "batches.jsonl").open("a") as raw:
                raw.write("{}\n")
            report = analysis.analyze_bundles([route_on])

        self.assertEqual(report["status"], "INVALID_INPUT")
        self.assertEqual(report["pivot_verdict"], "STOP_ROUTE_CONTROL")
        self.assertTrue(any("seal_hash:raw_sha256" in error for error in report["validation_errors"]))

    def test_sealed_but_malformed_config_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            route_on = _write_bundle(root, capture_routes=True, process_repeat=0)
            (route_on / "config.json").write_text("{")
            seal = json.loads((route_on / "RUN_COMPLETE.json").read_text())
            seal["config_sha256"] = _hash(route_on / "config.json")
            (route_on / "RUN_COMPLETE.json").write_text(json.dumps(seal))
            report = analysis.analyze_bundles([route_on])

        self.assertEqual(report["status"], "INVALID_INPUT")
        self.assertEqual(report["pivot_verdict"], "STOP_ROUTE_CONTROL")
        self.assertEqual(
            report["failure_category"], "BUNDLE_INTEGRITY_OR_COMPATIBILITY"
        )

    def test_one_process_repeat_is_insufficient_not_family_no_go(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            route_on = _write_bundle(root, capture_routes=True, process_repeat=0)
            report = analysis.analyze_bundles([route_on])

        self.assertEqual(report["status"], "INSUFFICIENT_EVIDENCE")
        self.assertEqual(report["failure_category"], "INSUFFICIENT_EVIDENCE")
        self.assertEqual(report["pivot_verdict"], "STOP_ROUTE_CONTROL")

    def test_historical_source_unverified_stays_measurement_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            route_on = [
                _write_bundle(
                    root,
                    capture_routes=True,
                    process_repeat=repeat,
                    route_overhead=1.02,
                    include_producer=False,
                )
                for repeat in range(2)
            ]
            route_off = [
                _write_bundle(
                    root,
                    capture_routes=False,
                    process_repeat=repeat,
                    include_producer=False,
                )
                for repeat in range(2)
            ]
            report = analysis.analyze_bundles(route_on, route_off)

        self.assertEqual(report["status"], "COMPLETE")
        self.assertEqual(report["pivot_verdict"], "WORKING_SET_MEASUREMENT_ONLY")
        self.assertEqual(
            report["failure_category"], "PRODUCER_SOURCE_SEMANTICS_UNAPPROVED"
        )
        self.assertEqual(
            report["producer_source_provenance"]["status"],
            "PRODUCER_SOURCE_SEMANTICS_UNAPPROVED",
        )
        self.assertFalse(
            report["decision_gates"]["producer_source_semantics_approved"]
        )

    def test_self_consistent_unknown_producer_cannot_select_action(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundles = [
                _write_bundle(
                    root,
                    capture_routes=capture,
                    process_repeat=repeat,
                    route_overhead=1.02 if capture else 1.12,
                )
                for capture in (True, False)
                for repeat in range(2)
            ]
            future_bytes = b"# FUTURE PRODUCER WITH UNKNOWN SEMANTICS\n"
            future_hash = hashlib.sha256(future_bytes).hexdigest()
            for run_dir in bundles:
                (run_dir / "producer_source.py").write_bytes(future_bytes)
                config = json.loads((run_dir / "config.json").read_text())
                config["probe_script_sha256"] = future_hash
                config["producer_source_artifact_sha256"] = future_hash
                (run_dir / "config.json").write_text(
                    json.dumps(config, sort_keys=True)
                )
                manifest = json.loads(
                    (run_dir / "ARTIFACT_HASHES.json").read_text()
                )
                manifest["producer_source.py"] = future_hash
                (run_dir / "ARTIFACT_HASHES.json").write_text(
                    json.dumps(manifest, sort_keys=True)
                )
                seal = json.loads((run_dir / "RUN_COMPLETE.json").read_text())
                seal["config_sha256"] = _hash(run_dir / "config.json")
                seal["artifact_hashes_sha256"] = _hash(
                    run_dir / "ARTIFACT_HASHES.json"
                )
                (run_dir / "RUN_COMPLETE.json").write_text(json.dumps(seal))
            route_on = [path for path in bundles if "on-" in path.name]
            route_off = [path for path in bundles if "off-" in path.name]
            report = analysis.analyze_bundles(route_on, route_off)

        self.assertEqual(report["status"], "COMPLETE")
        self.assertEqual(report["pivot_verdict"], "WORKING_SET_MEASUREMENT_ONLY")
        self.assertEqual(
            report["failure_category"], "PRODUCER_SOURCE_SEMANTICS_UNAPPROVED"
        )
        self.assertFalse(
            report["decision_gates"]["producer_source_semantics_approved"]
        )

    def test_repeat_gpu_isolation_drift_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            route_on = [
                _write_bundle(
                    root,
                    capture_routes=True,
                    process_repeat=0,
                    route_overhead=1.02,
                    require_exclusive_gpu=True,
                ),
                _write_bundle(
                    root,
                    capture_routes=True,
                    process_repeat=1,
                    route_overhead=1.02,
                    require_exclusive_gpu=False,
                ),
            ]
            report = analysis.analyze_bundles(route_on)

        self.assertEqual(report["status"], "INVALID_INPUT")
        self.assertTrue(
            any("require_exclusive_gpu" in error for error in report["validation_errors"])
        )

    def test_consistently_unisolated_bundles_cannot_select_action(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            route_on = [
                _write_bundle(
                    root,
                    capture_routes=True,
                    process_repeat=repeat,
                    route_overhead=1.02,
                    require_exclusive_gpu=False,
                )
                for repeat in range(2)
            ]
            route_off = [
                _write_bundle(
                    root,
                    capture_routes=False,
                    process_repeat=repeat,
                    require_exclusive_gpu=False,
                )
                for repeat in range(2)
            ]
            report = analysis.analyze_bundles(route_on, route_off)

        self.assertEqual(report["status"], "COMPLETE")
        self.assertEqual(report["pivot_verdict"], "WORKING_SET_MEASUREMENT_ONLY")
        self.assertEqual(report["failure_category"], "GPU_ISOLATION_UNVERIFIED")
        self.assertFalse(report["decision_gates"]["exclusive_gpu_verified"])

    def test_cli_exit_codes_are_semantic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            route_on = [
                _write_bundle(
                    root,
                    capture_routes=True,
                    process_repeat=repeat,
                    route_overhead=1.02,
                )
                for repeat in range(2)
            ]
            route_off = [
                _write_bundle(root, capture_routes=False, process_repeat=repeat)
                for repeat in range(2)
            ]
            positive_output = root / "positive.json"
            positive = subprocess.run(
                [
                    sys.executable,
                    str(MODULE_PATH),
                    "--route-on",
                    *(str(path) for path in route_on),
                    "--route-off",
                    *(str(path) for path in route_off),
                    "--output",
                    str(positive_output),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(positive.returncode, 0, positive.stderr)

            nonpositive_output = root / "nonpositive.json"
            nonpositive = subprocess.run(
                [
                    sys.executable,
                    str(MODULE_PATH),
                    "--route-on",
                    *(str(path) for path in route_on),
                    "--output",
                    str(nonpositive_output),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(nonpositive.returncode, 1, nonpositive.stderr)

            invalid_output = root / "invalid.json"
            invalid = subprocess.run(
                [
                    sys.executable,
                    str(MODULE_PATH),
                    "--route-on",
                    str(root / "missing"),
                    "--output",
                    str(invalid_output),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(invalid.returncode, 2, invalid.stderr)


if __name__ == "__main__":
    unittest.main()
