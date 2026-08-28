from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).with_name("analyze_vllm_decode_cap_branches.py")
SPEC = importlib.util.spec_from_file_location("decode_cap_analysis", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

PRODUCER_SHA = MODULE.APPROVED_BRANCH_RUNNER_SHA256
HELPER_SHA = MODULE.APPROVED_HELPER_SHA256
ROUTE_SHAPE = {
    "num_experts": 4,
    "num_layers": 1,
    "top_k": 2,
    "decode_route_steps": 3,
}


def runtime_identity() -> dict:
    return {
        "vllm": MODULE.EXPECTED_VLLM_VERSION,
        "vllm_actuator_sources": {
            relative: {"sha256": digest, "size_bytes": 1}
            for relative, digest in MODULE.EXPECTED_ACTUATOR_SOURCE_SHA256.items()
        },
        "vllm_runtime_sources": {
            relative: {"sha256": digest, "size_bytes": 1}
            for relative, digest in MODULE.EXPECTED_TELEMETRY_SOURCE_SHA256.items()
        },
        "compute_processes_before_engine_init": [],
    }


def request_rows(arm: str, wall_ms: float, capture: bool) -> list[dict]:
    token_offset = {"low": 0, "mid": 100, "high": 200}[arm]
    tpot_s = 0.0051 if capture else 0.005
    rows: list[dict] = []
    for index in range(16):
        queued = 1.0 + index * 1e-5
        scheduled = queued + 0.002
        first = scheduled + 0.006
        last = first + tpot_s * 3
        rows.append(
            {
                "cohort_index": index,
                "request_id": f"request-{index}",
                "token_ids": [token_offset + index, 2, 3, 4],
                "generated_tokens": 4,
                "decode_intervals": 3,
                "finish_reason": "length",
                "raw_timing_s": {
                    "branch_started_perf_counter": 10.0,
                    "branch_finished_perf_counter": 10.0 + wall_ms / 1000.0,
                    "queued_ts": queued,
                    "scheduled_ts": scheduled,
                    "first_token_ts": first,
                    "last_token_ts": last,
                    "first_token_latency": 0.008,
                },
                "ttft_ms": 8.0,
                "queue_ms": 2.0,
                "prefill_ms": 6.0,
                "decode_span_ms": tpot_s * 3000.0,
                "tpot_ms": tpot_s * 1000.0,
                "e2e_ms": 8.0 + tpot_s * 3000.0,
            }
        )
    return rows


def route_tensor() -> np.ndarray:
    routes = np.empty((16, 3, 1, 2), dtype=np.int32)
    for request in range(16):
        routes[request, :, 0, 0] = request % 4
        routes[request, :, 0, 1] = (request + 1) % 4
    return routes


def branch(arm: str, cap: int, capture: bool, base_wall_ms: float) -> dict:
    wall_ms = base_wall_ms * (1.02 if capture else 1.0)
    identity = {
        "runner_sha256": PRODUCER_SHA,
        "helper_source_sha256": HELPER_SHA,
        "require_exclusive_gpu": True,
        "compute_processes_before_engine_init": [],
        "cohort": "same",
    }
    runtime = runtime_identity()
    config = {
        "schema": MODULE.BRANCH_SCHEMA,
        "claim_ceiling": MODULE.CLAIM_CEILING,
        "experiment_identity": identity,
        "experiment_identity_sha256": MODULE._json_hash(identity),
        "runtime_identity": runtime,
        "runtime_identity_sha256": MODULE._json_hash(runtime),
        "runtime_patch_id": "valid-window-clear-v1",
        "probe_script_sha256": PRODUCER_SHA,
        "producer_source_sha256": PRODUCER_SHA,
        "helper_source_sha256": HELPER_SHA,
        "helper_source_artifact": MODULE.HELPER_SOURCE_ARTIFACT,
        "require_exclusive_gpu": True,
        "budget_arm": arm,
        "decode_cap": cap,
        "capture_routes": capture,
        "cohort_size": 16,
        "output_tokens": 4,
        "route_shape": deepcopy(ROUTE_SHAPE),
    }
    rows = request_rows(arm, wall_ms, capture)
    routes = route_tensor() if capture else None
    timing = MODULE._recompute_timing(config, rows)
    pressure = MODULE._recompute_route_pressure(config, routes)
    summary = {
        "schema": MODULE.BRANCH_SCHEMA,
        "status": "COMPLETE",
        "claim_ceiling": MODULE.CLAIM_CEILING,
        "budget_arm": arm,
        "decode_cap": cap,
        "capture_routes": capture,
        "timing": timing,
    }
    if capture:
        summary["route_shape"] = deepcopy(ROUTE_SHAPE)
        summary["route_pressure"] = pressure
    return {"config": config, "summary": summary, "rows": rows, "routes": routes}


def six_branches() -> dict[str, dict]:
    result = {}
    for arm, cap, wall in (("low", 4, 100.0), ("mid", 8, 70.0), ("high", 16, 80.0)):
        result[f"{arm}_off"] = branch(arm, cap, False, wall)
        result[f"{arm}_on"] = branch(arm, cap, True, wall)
    return result


def reset_wall(branch_value: dict, wall_ms: float) -> None:
    for row in branch_value["rows"]:
        row["raw_timing_s"]["branch_finished_perf_counter"] = 10.0 + wall_ms / 1000.0
    branch_value["summary"]["timing"] = MODULE._recompute_timing(
        branch_value["config"], branch_value["rows"]
    )


class DecodeCapBranchAnalysisTest(unittest.TestCase):
    def analyze(self, branches: dict[str, dict], **kwargs):
        return MODULE.analyze(
            branches,
            tpot_slo_ms=kwargs.get("tpot_slo_ms", 6.0),
            ttft_slo_ms=kwargs.get("ttft_slo_ms", 10.0),
            max_telemetry_overhead_pct=5.0,
            min_headroom_pct=3.0,
        )

    def test_route_off_drives_positive_initial_cap_headroom(self) -> None:
        report = self.analyze(six_branches())
        self.assertEqual(report["status"], MODULE.POSITIVE_STATUS)
        self.assertEqual(report["headroom"]["best_arm"], "mid")
        self.assertTrue(report["gates"]["telemetry_join_qualified"])
        self.assertFalse(report["cross_budget_token_drift"]["mid_vs_low"]["exact"])
        self.assertEqual(
            report["timing_comparison"]["canonical_source"],
            "requests.jsonl route-OFF recomputation",
        )

    def test_route_on_token_drift_fails_all_timing_comparisons(self) -> None:
        branches = six_branches()
        branches["mid_on"]["rows"][0]["token_ids"][0] += 1
        report = self.analyze(branches)
        self.assertEqual(
            report["status"],
            "INITIAL_ACTIVE_SEQUENCE_CAP_HEADROOM_TELEMETRY_INVALID",
        )
        self.assertEqual(report["timing_comparison"]["comparisons"], [])
        self.assertIsNone(report["arm_measurements"]["mid"]["route_pressure_joined"])

    def test_positive_overhead_above_five_percent_fails_join(self) -> None:
        branches = six_branches()
        reset_wall(branches["high_on"], 90.0)
        report = self.analyze(branches)
        self.assertFalse(report["gates"]["telemetry_overhead_qualified"])
        self.assertEqual(
            report["status"],
            "INITIAL_ACTIVE_SEQUENCE_CAP_HEADROOM_TELEMETRY_INVALID",
        )

    def test_negative_drift_above_five_percent_also_fails_join(self) -> None:
        branches = six_branches()
        reset_wall(branches["high_on"], 60.0)
        report = self.analyze(branches)
        self.assertLess(
            report["telemetry_pairs"]["high"]["signed_deviation_pct"]["wall_ms"],
            -5.0,
        )
        self.assertGreater(
            report["telemetry_pairs"]["high"]["absolute_deviation_pct"]["wall_ms"],
            5.0,
        )
        self.assertFalse(report["gates"]["telemetry_overhead_qualified"])

    def test_summary_timing_cannot_override_request_ledger(self) -> None:
        branches = six_branches()
        branches["mid_off"]["summary"]["timing"]["wall_ms"] = 1.0
        with self.assertRaisesRegex(ValueError, "summary.timing.wall_ms"):
            self.analyze(branches)

    def test_summary_pressure_cannot_override_route_tensor(self) -> None:
        branches = six_branches()
        branches["mid_on"]["summary"]["route_pressure"][
            "max_expert_load_across_waves"
        ] += 1
        with self.assertRaisesRegex(ValueError, "summary.route_pressure"):
            self.analyze(branches)

    def test_route_dtype_range_shape_and_topk_are_fail_closed(self) -> None:
        branches = six_branches()
        branches["low_on"]["routes"] = branches["low_on"]["routes"].astype(float)
        with self.assertRaisesRegex(ValueError, "dtype"):
            self.analyze(branches)

        branches = six_branches()
        branches["low_on"]["routes"][0, 0, 0, 0] = 4
        with self.assertRaisesRegex(ValueError, "outside"):
            self.analyze(branches)

        branches = six_branches()
        branches["low_on"]["routes"] = branches["low_on"]["routes"][:, :-1]
        with self.assertRaisesRegex(ValueError, "shape drift"):
            self.analyze(branches)

        branches = six_branches()
        branches["low_on"]["routes"][0, 0, 0] = [1, 1]
        with self.assertRaisesRegex(ValueError, "duplicate top-k"):
            self.analyze(branches)

    def test_nonfinite_and_nonpositive_timing_are_rejected(self) -> None:
        branches = six_branches()
        branches["low_off"]["rows"][0]["raw_timing_s"]["last_token_ts"] = float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            self.analyze(branches)

        branches = six_branches()
        for row in branches["low_off"]["rows"]:
            row["raw_timing_s"]["branch_finished_perf_counter"] = 10.0
        with self.assertRaisesRegex(ValueError, "non-positive"):
            self.analyze(branches)

    def test_unset_slo_is_explicitly_fail_closed(self) -> None:
        report = MODULE.analyze(six_branches(), tpot_slo_ms=None, ttft_slo_ms=None)
        self.assertEqual(
            report["status"], "INITIAL_ACTIVE_SEQUENCE_CAP_HEADROOM_SLO_UNSET"
        )
        self.assertFalse(report["gates"]["slo_thresholds_configured"])

    def test_zero_low_goodput_is_nonpositive_undefined_baseline(self) -> None:
        branches = six_branches()
        for name in ("low_off", "low_on"):
            for row in branches[name]["rows"]:
                row["raw_timing_s"]["first_token_latency"] = 0.012
                row["ttft_ms"] = 12.0
                row["e2e_ms"] = 12.0 + row["decode_span_ms"]
            branches[name]["summary"]["timing"] = MODULE._recompute_timing(
                branches[name]["config"], branches[name]["rows"]
            )
        report = self.analyze(branches)
        self.assertEqual(
            report["status"],
            "INITIAL_ACTIVE_SEQUENCE_CAP_HEADROOM_UNDEFINED_RELATIVE_BASELINE",
        )
        self.assertEqual(report["headroom"]["low_arm_goodput_output_tokens_per_s"], 0.0)
        self.assertIsNone(report["headroom"]["best_vs_low_goodput_pct"])
        self.assertEqual(MODULE.exit_code_for_status(report["status"]), 1)

    def test_gpu_isolation_and_frozen_actuator_sources_are_required(self) -> None:
        branches = six_branches()
        branches["low_on"]["config"]["require_exclusive_gpu"] = False
        with self.assertRaisesRegex(ValueError, "require_exclusive_gpu"):
            self.analyze(branches)

        branches = six_branches()
        branches["low_on"]["config"]["runtime_identity"][
            "compute_processes_before_engine_init"
        ] = ["other-process"]
        branches["low_on"]["config"]["runtime_identity_sha256"] = MODULE._json_hash(
            branches["low_on"]["config"]["runtime_identity"]
        )
        with self.assertRaisesRegex(ValueError, "isolated GPU"):
            self.analyze(branches)

        branches = six_branches()
        relative = next(iter(MODULE.EXPECTED_ACTUATOR_SOURCE_SHA256))
        branches["low_on"]["config"]["runtime_identity"][
            "vllm_actuator_sources"
        ][relative]["sha256"] = "c" * 64
        branches["low_on"]["config"]["runtime_identity_sha256"] = MODULE._json_hash(
            branches["low_on"]["config"]["runtime_identity"]
        )
        with self.assertRaisesRegex(ValueError, "actuator source hash"):
            self.analyze(branches)

        branches = six_branches()
        relative = next(iter(MODULE.EXPECTED_TELEMETRY_SOURCE_SHA256))
        branches["low_on"]["config"]["runtime_identity"][
            "vllm_runtime_sources"
        ][relative]["sha256"] = "d" * 64
        branches["low_on"]["config"]["runtime_identity_sha256"] = MODULE._json_hash(
            branches["low_on"]["config"]["runtime_identity"]
        )
        with self.assertRaisesRegex(ValueError, "telemetry source hash"):
            self.analyze(branches)

    def test_zero_queue_is_valid(self) -> None:
        branches = six_branches()
        for row in branches["low_off"]["rows"]:
            row["raw_timing_s"]["scheduled_ts"] = row["raw_timing_s"]["queued_ts"]
            row["queue_ms"] = 0.0
            row["prefill_ms"] = (
                row["raw_timing_s"]["first_token_ts"]
                - row["raw_timing_s"]["scheduled_ts"]
            ) * 1000.0
        branches["low_off"]["summary"]["timing"] = MODULE._recompute_timing(
            branches["low_off"]["config"], branches["low_off"]["rows"]
        )
        self.assertEqual(
            branches["low_off"]["summary"]["timing"]["request_queue_p95_ms"],
            0.0,
        )

    def test_partial_or_nonpositive_slo_is_invalid(self) -> None:
        with self.assertRaisesRegex(ValueError, "set together"):
            MODULE.analyze(six_branches(), tpot_slo_ms=6.0, ttft_slo_ms=None)
        with self.assertRaisesRegex(ValueError, "non-positive"):
            MODULE.analyze(six_branches(), tpot_slo_ms=0.0, ttft_slo_ms=10.0)

    def test_scientific_gate_thresholds_are_frozen(self) -> None:
        with self.assertRaisesRegex(ValueError, "frozen 5.0"):
            MODULE.analyze(
                six_branches(),
                tpot_slo_ms=6.0,
                ttft_slo_ms=10.0,
                max_telemetry_overhead_pct=100.0,
            )
        with self.assertRaisesRegex(ValueError, "frozen 3.0"):
            MODULE.analyze(
                six_branches(),
                tpot_slo_ms=6.0,
                ttft_slo_ms=10.0,
                min_headroom_pct=0.0,
            )

    def test_input_runtime_or_producer_identity_drift_is_rejected(self) -> None:
        branches = deepcopy(six_branches())
        branches["low_on"]["config"]["runtime_identity_sha256"] = "drift"
        with self.assertRaisesRegex(ValueError, "runtime"):
            self.analyze(branches)

        branches = deepcopy(six_branches())
        branches["low_on"]["config"]["producer_source_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "producer"):
            self.analyze(branches)

    def test_self_consistent_unknown_runner_or_helper_semantics_are_rejected(self) -> None:
        branches = six_branches()
        unknown_runner = "c" * 64
        for branch_value in branches.values():
            config = branch_value["config"]
            config["probe_script_sha256"] = unknown_runner
            config["producer_source_sha256"] = unknown_runner
            config["experiment_identity"]["runner_sha256"] = unknown_runner
            config["experiment_identity_sha256"] = MODULE._json_hash(
                config["experiment_identity"]
            )
        with self.assertRaisesRegex(ValueError, "producer semantics"):
            self.analyze(branches)

        branches = six_branches()
        unknown_helper = "d" * 64
        for branch_value in branches.values():
            config = branch_value["config"]
            config["helper_source_sha256"] = unknown_helper
            config["experiment_identity"]["helper_source_sha256"] = unknown_helper
            config["experiment_identity_sha256"] = MODULE._json_hash(
                config["experiment_identity"]
            )
        with self.assertRaisesRegex(ValueError, "helper semantics"):
            self.analyze(branches)

    def test_exit_code_contract(self) -> None:
        self.assertEqual(MODULE.exit_code_for_status(MODULE.POSITIVE_STATUS), 0)
        for status in MODULE.VALID_NONPOSITIVE_STATUSES:
            self.assertEqual(MODULE.exit_code_for_status(status), 1)
        self.assertEqual(MODULE.exit_code_for_status(MODULE.INVALID_STATUS), 2)
        self.assertEqual(MODULE.exit_code_for_status("unknown"), 2)

    def test_sealed_complete_bundle_loads_and_tamper_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            prompts = [[1, 2], [3, 4]]
            workload_bytes = b"{}"
            producer_bytes = MODULE.APPROVED_BRANCH_RUNNER_SOURCE.read_bytes()
            producer_sha = hashlib.sha256(producer_bytes).hexdigest()
            helper_bytes = MODULE.APPROVED_HELPER_SOURCE.read_bytes()
            helper_sha = hashlib.sha256(helper_bytes).hexdigest()
            prompt_hash = MODULE._json_hash(prompts)
            experiment_identity = {
                "prompt_token_ids_sha256": prompt_hash,
                "workload_manifest_sha256": hashlib.sha256(workload_bytes).hexdigest(),
                "runner_sha256": producer_sha,
                "helper_source_sha256": helper_sha,
                "require_exclusive_gpu": True,
                "compute_processes_before_engine_init": [],
            }
            runtime = runtime_identity()
            config = {
                "schema": MODULE.BRANCH_SCHEMA,
                "claim_ceiling": MODULE.CLAIM_CEILING,
                "cohort_size": 2,
                "output_tokens": 3,
                "prompt_length": 2,
                "capture_routes": False,
                "decode_cap": 1,
                "budget_arm": "low",
                "route_shape": {
                    "num_experts": 4,
                    "num_layers": 1,
                    "top_k": 2,
                    "decode_route_steps": 2,
                },
                "experiment_identity": experiment_identity,
                "experiment_identity_sha256": MODULE._json_hash(experiment_identity),
                "runtime_identity": runtime,
                "runtime_identity_sha256": MODULE._json_hash(runtime),
                "runtime_patch_id": "valid-window-clear-v1",
                "probe_script_sha256": producer_sha,
                "producer_source_sha256": producer_sha,
                "producer_source_artifact": "producer_source.py",
                "helper_source_sha256": helper_sha,
                "helper_source_artifact": MODULE.HELPER_SOURCE_ARTIFACT,
                "require_exclusive_gpu": True,
            }
            rows = []
            for index, prompt in enumerate(prompts):
                rows.append(
                    {
                        "cohort_index": index,
                        "request_id": str(index),
                        "prompt_token_ids_sha256": MODULE._json_hash(prompt),
                        "generated_tokens": 3,
                        "decode_intervals": 2,
                        "token_ids": [1, 2, 3],
                        "finish_reason": "length",
                        "raw_timing_s": {
                            "branch_started_perf_counter": 10.0,
                            "branch_finished_perf_counter": 10.1,
                            "queued_ts": 1.0,
                            "scheduled_ts": 1.002,
                            "first_token_ts": 1.008,
                            "last_token_ts": 1.018,
                            "first_token_latency": 0.008,
                        },
                        "ttft_ms": 8.0,
                        "queue_ms": 2.0,
                        "prefill_ms": 6.0,
                        "decode_span_ms": 10.0,
                        "tpot_ms": 5.0,
                        "e2e_ms": 18.0,
                    }
                )
            summary = {
                "schema": MODULE.BRANCH_SCHEMA,
                "status": "COMPLETE",
                "claim_ceiling": MODULE.CLAIM_CEILING,
                "budget_arm": "low",
                "decode_cap": 1,
                "capture_routes": False,
                "timing": MODULE._recompute_timing(config, rows),
            }
            (run_dir / "config.json").write_text(json.dumps(config))
            (run_dir / "environment.json").write_text(json.dumps(runtime))
            (run_dir / "workload_manifest.json").write_bytes(workload_bytes)
            (run_dir / "summary.json").write_text(json.dumps(summary))
            (run_dir / "producer_source.py").write_bytes(producer_bytes)
            (run_dir / MODULE.HELPER_SOURCE_ARTIFACT).write_bytes(helper_bytes)
            (run_dir / "requests.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows)
            )
            np.savez_compressed(
                run_dir / "input_cohort.npz",
                prompt_token_ids=np.asarray(prompts, dtype=np.int32),
            )
            artifacts = {
                name: hashlib.sha256((run_dir / name).read_bytes()).hexdigest()
                for name in MODULE.BASE_ARTIFACTS
            }
            (run_dir / "ARTIFACT_HASHES.json").write_text(json.dumps(artifacts))
            seal = {
                "status": "RUN_COMPLETE",
                "schema": MODULE.BRANCH_SCHEMA,
                "config_sha256": artifacts["config.json"],
                "requests_sha256": artifacts["requests.jsonl"],
                "summary_sha256": artifacts["summary.json"],
                "producer_source_sha256": artifacts["producer_source.py"],
                "helper_source_sha256": artifacts[MODULE.HELPER_SOURCE_ARTIFACT],
                "artifact_hashes": artifacts,
                "artifact_hashes_sha256": hashlib.sha256(
                    (run_dir / "ARTIFACT_HASHES.json").read_bytes()
                ).hexdigest(),
            }
            (run_dir / "RUN_COMPLETE.json").write_text(json.dumps(seal))
            loaded = MODULE.load_branch(run_dir)
            self.assertEqual(len(loaded["rows"]), 2)
            self.assertAlmostEqual(
                loaded["recomputed"]["timing"]["request_tpot_p95_ms"], 5.0
            )

            helper_path = run_dir / MODULE.HELPER_SOURCE_ARTIFACT
            helper_path.write_text("tampered helper\n")
            self.assertFalse(MODULE.verify_bundle(run_dir)["valid"])
            helper_path.write_bytes(helper_bytes)

            (run_dir / "producer_source.py").write_text("tampered\n")
            self.assertFalse(MODULE.verify_bundle(run_dir)["valid"])


if __name__ == "__main__":
    unittest.main()
