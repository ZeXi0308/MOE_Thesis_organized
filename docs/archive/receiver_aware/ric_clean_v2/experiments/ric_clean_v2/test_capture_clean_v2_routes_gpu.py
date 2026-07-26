from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

try:
    from . import native_route_core as core
    from . import capture_clean_v2_routes_gpu as routes
except ImportError:
    import native_route_core as core  # type: ignore
    import capture_clean_v2_routes_gpu as routes  # type: ignore


def _request(index: int) -> dict[str, object]:
    text = f"request-{index}-" + ("word " * 130)
    text_hash = hashlib.sha256(text.encode()).hexdigest()
    request_id = f"ric-clean-v2:calibration:{index:04d}:{text_hash[:12]}"
    return {
        "request_id": request_id,
        "text": text,
        "text_sha256": text_hash,
        "token_lengths": {"olmoe": 130, "llmjp": 130},
        "source_row": 100000 + index,
    }


class NativeRouteCoreTests(unittest.TestCase):
    def test_route_blind_placement_is_order_independent(self) -> None:
        requests = [{"request_id": value} for value in ("a", "b", "c", "d")]
        self.assertEqual(
            core.origin_lpt(requests, 2),
            core.origin_lpt(list(reversed(requests)), 2),
        )
        self.assertEqual(
            [core.expert_sender(index, 8, 4) for index in range(8)],
            [0, 0, 1, 1, 2, 2, 3, 3],
        )

    def test_layer_selection_is_outcome_blind(self) -> None:
        first = core.selected_layers(
            [0, 1, 2, 3, 4, 5],
            selection_seed=2026072301,
            model_revision="model@revision",
            count=4,
        )
        second = core.selected_layers(
            [5, 4, 3, 2, 1, 0],
            selection_seed=2026072301,
            model_revision="model@revision",
            count=4,
        )
        self.assertEqual(first, second)
        self.assertIn(core.assigned_layer("request", first), first)

    def test_route_tuple_binds_dtype_and_rejects_nonfinite(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")
        experts = torch.tensor([[1, 0]], dtype=torch.int64)
        bf16 = torch.tensor([[0.5, 0.25]], dtype=torch.bfloat16)
        self.assertNotEqual(
            core.route_tuple_sha256(experts, bf16),
            core.route_tuple_sha256(experts, bf16.float()),
        )
        with self.assertRaisesRegex(core.NativeRouteError, "non-finite"):
            core.route_tuple_sha256(
                experts, torch.tensor([[float("nan"), 0.25]], dtype=torch.float32)
            )

    def test_dispatch_observer_preserves_native_tie_order(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")
        active = {"value": 2}
        capture = core.make_native_topk_capture_mode(
            active_layer=active,
            expected_num_experts=5,
            expected_top_k=3,
            expected_tokens=2,
        )
        probabilities = torch.tensor(
            [[0.4, 0.2, 0.2, 0.1, 0.1], [0.1, 0.1, 0.2, 0.2, 0.4]],
            dtype=torch.float32,
        )
        with capture:
            values, indices = torch.topk(probabilities, 3, dim=-1)
        captured_values, captured_indices = capture.calls[2]
        self.assertTrue(torch.equal(values, captured_values))
        self.assertTrue(torch.equal(indices, captured_indices))

    def test_reconstruction_parity_blocks_perturbed_route(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")

        class Expert(torch.nn.Module):
            def __init__(self, scale: float) -> None:
                super().__init__()
                self.scale = scale

            def forward(self, value):
                return value * self.scale

        moe = type("Moe", (), {"experts": [Expert(1.0), Expert(3.0)]})()
        hidden = torch.ones((1, 2, 1), dtype=torch.float32)
        experts = torch.tensor([[0], [0]], dtype=torch.int64)
        weights = torch.ones((2, 1), dtype=torch.float32)
        native, _, _ = core.reconstruct_native_moe_output(
            moe=moe,
            hidden_states=hidden,
            selected_experts=experts,
            effective_weights=weights,
        )
        perturbed, _, _ = core.reconstruct_native_moe_output(
            moe=moe,
            hidden_states=hidden,
            selected_experts=torch.tensor([[1], [1]], dtype=torch.int64),
            effective_weights=weights,
        )
        tolerance = {
            "rule": "finfo_scaled_source_only",
            "rtol_finfo_eps_multiplier": 4.0,
            "atol_finfo_eps_multiplier": 4.0,
            "topk_indexes_must_match_exactly": True,
            "outcome_tuning_allowed": False,
        }
        with self.assertRaisesRegex(core.NativeRouteError, "output parity failed"):
            core.validate_native_moe_output_parity(
                native, perturbed, tolerance_rule=tolerance
            )


class CleanRouteProducerTests(unittest.TestCase):
    def test_clean_source_does_not_import_old_route_runner(self) -> None:
        source = Path(routes.__file__).read_text(encoding="utf-8")
        self.assertNotIn("capture_routes_gpu import", source)
        self.assertNotIn("formal_provenance", source)
        self.assertNotIn("ric_v1.json", source)

    def test_manifest_validator_binds_all_selected_requests(self) -> None:
        requests = [_request(index) for index in range(64)]
        manifest = {
            "manifest_sha256": routes.CALIBRATION_MANIFEST_SHA256,
            "phase4_signoff_sha256": "a" * 64,
            "requests": requests,
            "selected_request_ids": [row["request_id"] for row in requests],
            "selected_text_sha256": [row["text_sha256"] for row in requests],
        }
        with patch.object(
            routes.data, "validate_calibration_manifest", return_value=manifest
        ):
            verified = routes.validate_calibration_manifest()
        self.assertEqual(len(verified["requests"]), 64)
        changed = dict(manifest)
        changed["selected_text_sha256"] = list(changed["selected_text_sha256"])
        changed["selected_text_sha256"][0] = "b" * 64
        with patch.object(
            routes.data, "validate_calibration_manifest", return_value=changed
        ), self.assertRaisesRegex(routes.CleanRouteError, "selected lists"):
            routes.validate_calibration_manifest()

    def test_foreign_gpu_process_is_a_hard_block(self) -> None:
        routes.validate_compute_apps([{"pid": 7}], producer_pid=7)
        with self.assertRaisesRegex(routes.CleanRouteError, "foreign GPU"):
            routes.validate_compute_apps([{"pid": 8}], producer_pid=7)

    def test_route_signoff_binds_sources_and_data(self) -> None:
        manifest = {
            "manifest_sha256": routes.CALIBRATION_MANIFEST_SHA256,
            "phase4_signoff_sha256": "a" * 64,
        }
        with tempfile.TemporaryDirectory() as directory:
            review = Path(directory).resolve(strict=True)
            calibration_manifest = review / "manifest.json"
            report = review / "RIC_Clean_v2_Route_CodeReview.md"
            test_report_path = review / "RIC_Clean_v2_Route_TestReport.json"
            source_path = review / "reviewed_source_manifest_route.json"
            signoff_path = review / "signoff_route_olmoe.json"
            source_manifest = routes.add_self_hash(
                {
                    "schema_version": "ric-clean-v2-route-reviewed-source-manifest-v1",
                    "status": "REVIEWED",
                    "sources": routes._reviewed_sources(),
                }
            )
            source_path.write_text(json.dumps(source_manifest), encoding="utf-8")
            report.write_text(
                "STATUS: SIGNED-OFF\nOPEN_P0: 0\n"
                f"REVIEWED_SOURCE_MANIFEST_SHA256: {source_manifest['manifest_sha256']}\n",
                encoding="utf-8",
            )
            test_report = routes.add_self_hash(
                {
                    "schema_version": "ric-clean-v2-route-test-report-v1",
                    "status": "PASS",
                    "errors": 0,
                    "failures": 0,
                    "tests_run": 10,
                    "reviewed_source_manifest_sha256": source_manifest["manifest_sha256"],
                    "reviewed_source_manifest_file_sha256": routes.data.file_sha256(source_path),
                }
            )
            test_report_path.write_text(json.dumps(test_report), encoding="utf-8")
            config = routes.load_mapping(routes.DEFAULT_CONFIG, label="config")
            calibration_manifest.write_text("{}\n", encoding="utf-8")
            signoff = routes.data.add_self_hash(
                {
                    "schema_version": "ric-clean-v2-route-phase4-signoff-v1",
                    "status": "SIGNED-OFF",
                    "open_p0": 0,
                    "stage": "capture_calibration_routes",
                    "model_key": "olmoe",
                    "config_sha256": routes.data.file_sha256(routes.DEFAULT_CONFIG),
                    "base_protocol_sha256": routes.data.file_sha256(routes.BASE_PROTOCOL),
                    "route_addendum_sha256": routes.data.file_sha256(routes.ROUTE_ADDENDUM),
                    "route_producer_source_sha256": routes.source_sha256(),
                    "data_manifest_sha256": manifest["manifest_sha256"],
                    "data_manifest_file_sha256": routes.data.file_sha256(calibration_manifest),
                    "data_phase4_signoff_sha256": manifest["phase4_signoff_sha256"],
                    "model_tree_manifest_sha256": config["models"]["olmoe"]["expected_local_model_tree_manifest_sha256"],
                    "review_report_sha256": routes.data.file_sha256(report),
                    "test_report_sha256": routes.data.file_sha256(test_report_path),
                    "reviewed_source_manifest_sha256": source_manifest["manifest_sha256"],
                    "reviewed_source_manifest_file_sha256": routes.data.file_sha256(source_path),
                },
                "signoff_sha256",
            )
            signoff_path.write_text(json.dumps(signoff), encoding="utf-8")
            with patch.multiple(
                routes,
                CLEAN_REVIEW_DIR=review,
                REVIEW_REPORT=report,
                TEST_REPORT=test_report_path,
                REVIEWED_SOURCE_MANIFEST=source_path,
                SIGNOFFS={"olmoe": signoff_path, "llmjp": review / "unused.json"},
                CALIBRATION_MANIFEST=calibration_manifest,
            ):
                verified = routes.validate_route_signoff("olmoe", manifest)
            self.assertEqual(verified["signoff_sha256"], signoff["signoff_sha256"])

    def test_atomic_writer_hashes_every_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(strict=True)
            output = root / "routes/olmoe"
            signoff = routes.data.add_self_hash(
                {"schema_version": "fixture"}, "signoff_sha256"
            )
            signoff_bytes = json.dumps(signoff).encode()
            placement = routes.add_self_hash({"schema_version": "placement"})
            parity = routes.add_self_hash({"schema_version": "parity"})
            metadata = routes._write_artifacts(
                output_dir=output,
                route_rows=[{"schema_version": "ric-clean-v2-route-row-v1"}],
                placement=placement,
                parity=parity,
                metadata_payload={
                    "schema_version": "ric-clean-v2-route-capture-v1",
                    "scientific_result": False,
                },
                signoff_bytes=signoff_bytes,
                expected_signoff_sha256=signoff["signoff_sha256"],
            )
            self.assertTrue((output / "route_trace.jsonl").is_file())
            self.assertEqual(metadata["scientific_result"], False)
            with self.assertRaisesRegex(routes.CleanRouteError, "overwrite"):
                routes._write_artifacts(
                    output_dir=output,
                    route_rows=[],
                    placement=placement,
                    parity=parity,
                    metadata_payload={},
                    signoff_bytes=signoff_bytes,
                    expected_signoff_sha256=signoff["signoff_sha256"],
                )

    def test_route_reservation_is_fixed_and_one_shot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(strict=True)
            state = root / "state"
            ledger = state / "route_calibration_olmoe_consumption.json"
            with patch.multiple(
                routes,
                ROUTE_STATE_DIR=state,
                ROUTE_LEDGERS={"olmoe": ledger},
            ):
                record = routes.reserve_route(
                    "olmoe",
                    manifest={"manifest_sha256": "m" * 64},
                    signoff={"signoff_sha256": "s" * 64},
                )
                self.assertEqual(record["state"], "RESERVED_FAIL_CLOSED")
                with self.assertRaisesRegex(routes.CleanRouteError, "already consumed"):
                    routes.reserve_route(
                        "olmoe",
                        manifest={"manifest_sha256": "m" * 64},
                        signoff={"signoff_sha256": "s" * 64},
                    )

    def test_cartesian_census_rejects_duplicate_or_missing_identity(self) -> None:
        rows = [
            {
                "request_id": "r0",
                "layer_id": 0,
                "token_position": token,
                "topk_slot": slot,
            }
            for token in range(128)
            for slot in range(2)
        ]
        parity = [{"request_id": "r0", "layer_id": 0}]
        routes.validate_route_census(
            rows, parity, request_ids={"r0"}, layer_ids={0}, top_k=2
        )
        broken = list(rows)
        broken[-1] = dict(broken[0])
        with self.assertRaisesRegex(routes.CleanRouteError, "Cartesian"):
            routes.validate_route_census(
                broken, parity, request_ids={"r0"}, layer_ids={0}, top_k=2
            )


if __name__ == "__main__":
    unittest.main()
