from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, EXPERIMENTS / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


builder = load_module("route_shape_builder", "build_route_windows.py")
analyzer = load_module("route_shape_analyzer", "analyze_incremental_signal.py")
inspector = load_module("route_shape_inspector", "inspect_existing_assets.py")
p2 = load_module("route_shape_p2", "replay_capacity_oracle.py")
p3 = load_module("route_shape_p3", "run_causal_controller.py")


ROUTE_FIELDS = [
    "model",
    "phase",
    "request_id",
    "sample_id",
    "arrival_us",
    "deadline_us",
    "layer",
    "token_position",
    "rank",
    "expert_id",
    "gate_weight",
    "src_replica",
    "input_event_id",
    "token_id",
    "decode_step",
    "layer_id",
    "topk_slot",
    "source_rank",
    "target_replica",
    "document_id",
    "request_arrival_us",
    "layer_ready_us",
    "route_end_us",
    "dispatch_end_us",
    "expert_start_us",
    "expert_end_us",
    "combine_end_us",
    "legal_replica_set",
]


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def write_jsonl(path: Path, values: list[dict]) -> None:
    path.write_text("".join(json.dumps(value) + "\n" for value in values), encoding="utf-8")


def make_capture(
    root: Path,
    *,
    episode: str = "ep0",
    windows: int = 3,
    model: str = "fixture-model",
    model_revision: str = "fixture-rev",
    arrival_regime: str = "steady",
    split: str = "unassigned",
) -> Path:
    capture = root / episode
    capture.mkdir()
    request_names = (f"{episode}-r0", f"{episode}-r1")
    document_names = (f"{episode}-d0", f"{episode}-d1")
    requests = [
        {
            "request_id": request_names[0],
            "document_id": document_names[0],
            "arrival_us": 0.0,
            "deadline_us": 10_000.0,
            "prompt_tokens": 4,
            "steps": [
                {"decode_step": step, "batch_index": step} for step in range(windows)
            ],
        },
        {
            "request_id": request_names[1],
            "document_id": document_names[1],
            "arrival_us": 0.0,
            "deadline_us": 10_000.0,
            "prompt_tokens": 5,
            "steps": [
                {"decode_step": step, "batch_index": step} for step in range(windows)
            ],
        },
    ]
    batches = []
    routes = []
    for step in range(windows):
        start = step * 100.0
        end = start + 50.0 + step
        batches.append(
            {
                "batch_index": step,
                "start_us": start,
                "end_us": end,
                "batch_size": 2,
                "active_request_ids": list(request_names),
                "pending_request_count": 0,
                "request_ids": list(request_names),
                "decode_steps": [step, step],
                "prior_cache_lengths": [4 + step, 5 + step],
                "left_padding": [1, 0],
            }
        )
        for request_index, request_id in enumerate(request_names):
            for layer in (0, 1):
                for slot in (0, 1):
                    expert = (request_index + layer + slot + step) % 4
                    routes.append(
                        {
                            "model": model,
                            "phase": "decode",
                            "request_id": request_id,
                            "sample_id": request_index,
                            "arrival_us": 0.0,
                            "deadline_us": 10_000.0,
                            "layer": layer,
                            "token_position": 4 + step,
                            "rank": slot + 1,
                            "expert_id": expert,
                            "gate_weight": 0.5,
                            "src_replica": 0,
                            "input_event_id": f"{request_id}:decode:{step:06d}",
                            "token_id": 7 + step,
                            "decode_step": step,
                            "layer_id": layer,
                            "topk_slot": slot,
                            "source_rank": 0,
                            "target_replica": -1,
                            "document_id": document_names[request_index],
                            "request_arrival_us": 0.0,
                            "layer_ready_us": start,
                            "route_end_us": end,
                            "dispatch_end_us": -1,
                            "expert_start_us": -1,
                            "expert_end_us": -1,
                            "combine_end_us": -1,
                            "legal_replica_set": "[0]",
                        }
                    )
    with (capture / "routes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROUTE_FIELDS)
        writer.writeheader()
        writer.writerows(routes)
    write_jsonl(capture / "decode_batches.jsonl", batches)
    write_jsonl(capture / "request_ledger.jsonl", requests)
    write_json(capture / "CAPTURE_COMPLETE.json", {"status": "CAPTURE_COMPLETE"})
    write_json(capture / "environment.json", {"cuda_available": False})
    write_json(
        capture / "route_shape_slo_capture.json",
        {
            "episode_id": episode,
            "arrival_regime": arrival_regime,
            "model_revision": model_revision,
            "split": split,
            "num_experts": 4,
            "evidence_type": "[Synthetic fixture]",
            "runtime_representative": False,
            "instrumentation_overhead_measured": False,
            "fresh_holdout_sealed": False,
        },
    )
    return capture


class RouteWindowBuilderTest(unittest.TestCase):
    def test_route_summary_includes_zero_count_experts(self) -> None:
        summary = builder._route_summary(
            {0: {0: 3, 1: 1}, 1: {0: 2, 2: 2}},
            num_experts=4,
            batch_size=2,
        )
        self.assertEqual(summary.top_k, 2)
        self.assertAlmostEqual(summary.route_hhi, (0.625 + 0.5) / 2)

    def test_builds_identity_closed_causal_features(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            capture = make_capture(Path(temp))
            rows, metadata = builder.build_all([capture], {})
        self.assertEqual(len(rows), 3)
        self.assertEqual(metadata["status"], "BLOCKED_RUNTIME_NOT_REPRESENTATIVE")
        first = rows[0]
        self.assertEqual(first["active_tokens"], 2)
        self.assertEqual(first["running_sequences"], 2)
        self.assertEqual(first["queue_depth"], 0)
        self.assertEqual(first["tokens_completed"], 2)
        self.assertEqual(first["route_layer_count"], 2)
        self.assertEqual(first["top_k"], 2)
        self.assertEqual(first["feature_available_at_us"], first["window_end_us"])
        self.assertEqual(metadata["captures"][0]["future_fields_consumed"], [])

    def test_rejects_route_only_legacy_capture(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            capture = make_capture(Path(temp))
            route_path = capture / "routes.csv"
            with route_path.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            with route_path.open("w", newline="", encoding="utf-8") as handle:
                fields = [field for field in ROUTE_FIELDS if field != "document_id"]
                writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(builder.ProtocolError, "legacy/partial"):
                builder.build_all([capture], {})

    def test_rejects_duplicate_window_ownership(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            capture = make_capture(Path(temp))
            batches = builder.read_jsonl(capture / "decode_batches.jsonl")
            duplicate = dict(batches[1])
            duplicate["batch_index"] = 3
            duplicate["start_us"] = batches[0]["start_us"]
            duplicate["end_us"] = batches[0]["end_us"]
            duplicate["decode_steps"] = batches[0]["decode_steps"]
            write_jsonl(capture / "decode_batches.jsonl", [batches[0], duplicate] + batches[1:])
            with self.assertRaisesRegex(
                builder.ProtocolError, "contiguous|more than one batch|time regresses"
            ):
                builder.build_all([capture], {})

    def test_rejects_duplicate_expert_within_token_layer_topk(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            capture = make_capture(Path(temp))
            route_path = capture / "routes.csv"
            with route_path.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            first = rows[0]
            duplicate = next(
                row
                for row in rows
                if row["request_id"] == first["request_id"]
                and row["decode_step"] == first["decode_step"]
                and row["layer_id"] == first["layer_id"]
                and row["topk_slot"] != first["topk_slot"]
            )
            duplicate["expert_id"] = first["expert_id"]
            with route_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=ROUTE_FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(
                builder.ProtocolError, "same expert"
            ):
                builder.build_all([capture], {})


class IncrementalAnalysisTest(unittest.TestCase):
    CONFIG = {
        "action": "next_window_active_token_budget",
        "frozen_models": dict(analyzer.FROZEN_MODELS),
        "required_arrival_regimes": ["steady", "bursty"],
        "ridge_alpha": 1.0,
        "target_quantile": 0.95,
        "pinball_relative_improvement_gate": 0.05,
        "underprediction_relative_reduction_gate": 0.15,
        "stop_below_relative_improvement": 0.03,
        "minimum_test_cell_train_coverage": 0.0,
        "dangerous_underprediction_margin": 0.10,
        "matched_cell_bucket_widths": {
            "active_tokens": 2,
            "running_sequences": 2,
            "queue_depth": 2,
            "mean_kv_length": 32,
            "max_kv_length": 32,
            "prompt_tokens": 16,
            "decode_tokens": 2,
            "batch_size": 2,
            "decode_stage": 4,
        },
    }

    def test_next_window_alignment_keeps_future_route_oracle_separate(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            captures = [make_capture(Path(temp), episode=f"ep{index}") for index in range(5)]
            rows, _ = builder.build_all(captures, {})
        aligned = analyzer.align_next_window(rows)
        self.assertEqual(len(aligned), 10)
        self.assertIn("future_route_cv", aligned[0])
        self.assertNotIn("future_route_cv", analyzer.METHODS["M3_workload_plus_route"])
        self.assertIn("future_route_cv", analyzer.METHODS["M4_future_route_oracle"])

    def test_smoke_metrics_cannot_issue_scientific_p1_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            captures = [make_capture(Path(temp), episode=f"ep{index}") for index in range(5)]
            rows, _ = builder.build_all(captures, {})
        aligned = analyzer.align_next_window(rows)
        metrics, summary = analyzer.analyze(aligned, self.CONFIG)
        self.assertEqual(summary["p1_status"], "SMOKE_ONLY_NOT_SCIENTIFICALLY_ELIGIBLE")
        self.assertEqual(summary["verdict"], "BLOCKED_RUNTIME_NOT_REPRESENTATIVE")
        self.assertFalse(summary["scientific_result_eligible"])
        self.assertEqual(
            {row["method"] for row in metrics if row["scope"] == "aggregate"},
            set(analyzer.METHODS),
        )

    def test_second_action_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            captures = [make_capture(Path(temp), episode=f"ep{index}") for index in range(5)]
            rows, _ = builder.build_all(captures, {})
        aligned = analyzer.align_next_window(rows)
        config = dict(self.CONFIG)
        config["action"] = "next_window_max_running_sequences"
        with self.assertRaisesRegex(
            analyzer.ProtocolError, "next_window_active_token_budget"
        ):
            analyzer.analyze(aligned, config)

    def test_zero_baseline_underprediction_detects_regression(self) -> None:
        self.assertEqual(analyzer._rate_reduction(0.0, 0.0), 0.0)
        self.assertLess(analyzer._rate_reduction(0.0, 0.1), 0.0)

    def test_target_window_eligibility_is_checked(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            captures = [make_capture(Path(temp), episode=f"ep{index}") for index in range(5)]
            rows, _ = builder.build_all(captures, {})
        aligned = analyzer.align_next_window(rows)
        self.assertIn("target_runtime_representative", aligned[0])
        self.assertIn("target_evidence_type", aligned[0])
        self.assertIn("target_arrival_episode_independent", aligned[0])

    def test_alignment_rejects_late_features_and_episode_invariant_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            capture = make_capture(Path(temp))
            rows, _ = builder.build_all([capture], {})
        rows[0]["feature_available_at_us"] = rows[0]["window_end_us"] + 1
        with self.assertRaisesRegex(analyzer.ProtocolError, "after window t"):
            analyzer.align_next_window(rows)

        rows[0]["feature_available_at_us"] = rows[0]["window_end_us"]
        rows[1]["model_revision"] = "drifted-revision"
        with self.assertRaisesRegex(analyzer.ProtocolError, "model_revision changes"):
            analyzer.align_next_window(rows)

    def test_frozen_models_and_required_regimes_are_exactly_bound(self) -> None:
        split_sequence = ("train", "train", "validation", "test", "test")
        with tempfile.TemporaryDirectory() as temp:
            captures = []
            for model_index, (model, revision) in enumerate(
                analyzer.FROZEN_MODELS.items()
            ):
                for episode_index, split in enumerate(split_sequence):
                    captures.append(
                        make_capture(
                            Path(temp),
                            episode=f"m{model_index}-ep{episode_index}",
                            model=model,
                            model_revision=revision,
                            arrival_regime=(
                                "steady" if episode_index % 2 == 0 else "bursty"
                            ),
                            split=split,
                        )
                    )
            rows, _ = builder.build_all(captures, {})
        _, summary = analyzer.analyze(analyzer.align_next_window(rows), self.CONFIG)
        self.assertTrue(summary["eligibility_checks"]["two_frozen_models"])
        self.assertTrue(summary["eligibility_checks"]["arrival_regimes_per_model"])
        self.assertFalse(summary["scientific_result_eligible"])

        drifted = dict(self.CONFIG)
        drifted["frozen_models"] = {"fake/a": "1", "fake/b": "2"}
        with self.assertRaisesRegex(analyzer.ProtocolError, "exact frozen"):
            analyzer.analyze(analyzer.align_next_window(rows), drifted)

        for row in rows:
            row["arrival_episode_independent"] = "true"
            row["evidence_type"] = "[Observed real runtime]"
            row["runtime_representative"] = "true"
            row["instrumentation_overhead_measured"] = "true"
            row["fresh_holdout_sealed"] = "true"
            row["gate_weight_available"] = "true"
        eligible_metrics, eligible_summary = analyzer.analyze(
            analyzer.align_next_window(rows), self.CONFIG
        )
        self.assertTrue(eligible_summary["scientific_result_eligible"])
        self.assertIn(eligible_summary["classification"], {"B", "E"})
        self.assertNotEqual(
            eligible_summary["claim_boundary"],
            "A smoke-only result validates code paths and leakage guards only; it does not measure route-conditioned serving capacity.",
        )
        with tempfile.TemporaryDirectory() as report_temp:
            report = Path(report_temp) / "eligible-report.md"
            analyzer.write_report(report, eligible_summary, eligible_metrics)
            report_text = report.read_text(encoding="utf-8")
        self.assertIn("## Eligible P1 result", report_text)
        self.assertIn(f"Classification is `{eligible_summary['classification']}`", report_text)
        self.assertNotIn("classification is `D`", report_text)
        self.assertNotIn("P0/P1 smoke actually run", report_text)
        self.assertNotIn("one model, one replay", report_text)
        self.assertNotIn("Unknown.", report_text)

    def test_report_contains_required_decision_sections(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            captures = [make_capture(Path(temp), episode=f"ep{index}") for index in range(5)]
            rows, _ = builder.build_all(captures, {})
            aligned = analyzer.align_next_window(rows)
            metrics, summary = analyzer.analyze(aligned, self.CONFIG)
            report = Path(temp) / "report.md"
            analyzer.write_report(report, summary, metrics)
            text = report.read_text(encoding="utf-8")
        for heading in (
            "## Verdict",
            "## Evidence Table",
            "## Measured / Inferred Boundary",
            "## Next Smallest Experiment",
            "## Direct answer",
        ):
            self.assertIn(heading, text)

    def test_request_identity_cannot_cross_analysis_splits(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            captures = [make_capture(Path(temp), episode=f"ep{index}") for index in range(5)]
            rows, _ = builder.build_all(captures, {})
        aligned = analyzer.align_next_window(rows)
        assigned, diagnostics = analyzer.assign_splits(
            analyzer.annotate_matched_cells(aligned, self.CONFIG)
        )
        split_to_row = {}
        for row in assigned:
            split_to_row.setdefault(row["analysis_split"], row)
        self.assertEqual(set(split_to_row), {"train", "validation", "test"})
        split_to_row["test"]["request_ids"] = split_to_row["train"]["request_ids"]
        with self.assertRaisesRegex(analyzer.ProtocolError, "cross analysis splits"):
            analyzer.validate_identity_disjoint_splits(assigned)


class GuardAndInventoryTest(unittest.TestCase):
    def test_p2_and_p3_fail_closed(self) -> None:
        self.assertEqual(
            p2.gate({"scientific_result_eligible": False})["status"],
            "BLOCKED_P1_NOT_ELIGIBLE",
        )
        self.assertEqual(
            p3.gate({"scientific_result_eligible": False})["status"],
            "BLOCKED_P2_NOT_PASSED",
        )

    def test_inventory_discovery_is_empty_for_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            make_capture(root)
            self.assertEqual(inspector.discover_bcrd(root), [])
            self.assertEqual(inspector.discover_stablebatch(root), [])


if __name__ == "__main__":
    unittest.main()
