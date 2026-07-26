from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path
import unittest
from unittest import mock

try:
    from . import capture_phasemap_lut_gpu as lutmod
    from . import phasemap_instances as instances
    from . import phasemap_oracle_core as core
    from . import run_phasemap_oracle_gate as runner
    from .test_capture_phasemap_lut_gpu import PhaseMapLUTTests
    from .test_phasemap_instances import raw_join
except ImportError:  # pragma: no cover
    import capture_phasemap_lut_gpu as lutmod  # type: ignore
    import phasemap_instances as instances  # type: ignore
    import phasemap_oracle_core as core  # type: ignore
    import run_phasemap_oracle_gate as runner  # type: ignore
    from test_capture_phasemap_lut_gpu import PhaseMapLUTTests  # type: ignore
    from test_phasemap_instances import raw_join  # type: ignore


SERVICES = {
    "sender_pack": 2.0,
    "receiver_unpack": 10.0,
    "canonical_combine": 3.0,
    "analytic_cut": 1.0,
}

TEST_META = {
    "model_key": "model",
    "model_revision": "revision:model",
    "data_manifest_sha256": "d" * 64,
    "placement_manifest_sha256": "e" * 64,
    "top_k": 8,
    "expected_join_candidates_per_request": 2,
}


def pair_manifests(prefix: str = "fixture") -> list[dict]:
    selected = []
    for receiver in range(8):
        for request_index in range(4):
            request = f"{prefix}-r{receiver}-{request_index}"
            join = raw_join(request, receiver, 0)
            for sibling in join["siblings"]:
                sibling["placement_manifest_sha256"] = "e" * 64
            selected.append(instances._normalize_join(join, model="model", metadata=TEST_META))  # type: ignore[attr-defined]
    return [
        instances.build_world_manifest(pair, SERVICES["receiver_unpack"])
        for pair in instances.canonical_perfect_matching(selected)
    ]


def _formal_routes(model: str, lut: dict) -> tuple[list[dict], dict]:
    inputs = lut["model_inputs"][model]
    revision = inputs["model_revision"]
    top_k = int(inputs["top_k"])
    rows = []
    for receiver in range(8):
        for request_index in range(8):
            request = f"formal-{model}-r{receiver}-{request_index}"
            for position in (0, 1):
                row = raw_join(
                    request, receiver, position, senders=tuple(range(top_k)), model=model
                )
                for sibling in row["siblings"]:
                    sibling["model_revision"] = revision
                rows.append(row)
    metadata = {
        "model_key": model,
        "model_revision": revision,
        "data_manifest_sha256": "d" * 64,
        "placement_manifest_sha256": "e" * 64,
        "top_k": top_k,
        "expected_join_candidates_per_request": 2,
        "manifest_sha256": "a" * 64,
        "route_trace_file_sha256": "b" * 64,
        "route_phase4_signoff_sha256": "c" * 64,
        "producer_signoff_file_sha256": "f" * 64,
        "placement_file_sha256": "9" * 64,
    }
    return rows, metadata


def _informative_lut() -> dict:
    """Valid synthetic timing fixture whose frozen grid is informative for both models."""

    value = PhaseMapLUTTests().fixture_artifact()
    value.pop("artifact_sha256")
    for row in value["raw_trials"]:
        row["cuda_event_us"] *= 0.01
    value["summary"] = lutmod.validate_and_summarize(value["raw_trials"])
    return lutmod.add_self_hash(value)


def _formal_manifest(model: str, lut: dict, service: dict) -> dict:
    rows, metadata = _formal_routes(model, lut)
    with mock.patch.object(instances, "load_verified_joins", return_value=(rows, metadata)):
        return instances.build_model_manifests(
            Path("/formal/reviewed-routes"), model, service["receiver_unpack"],
            lut_artifact_sha256=lut["artifact_sha256"],
            lut_model_identity=lut["model_inputs"][model],
        )


def _rehash_pair(pair: dict) -> None:
    pair.pop("manifest_sha256", None)
    pair["manifest_sha256"] = instances.object_sha256(pair)


def _rehash_bundle(bundle: dict) -> None:
    bundle.pop("artifact_sha256", None)
    bundle["artifact_sha256"] = instances.object_sha256(bundle)


def _rehash_manifest(manifest: dict) -> None:
    manifest.pop("manifest_sha256", None)
    manifest["manifest_sha256"] = instances.object_sha256(manifest)


class PhaseMapRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.selection_pairs = pair_manifests("selection")
        cls.holdout_pairs = pair_manifests("holdout")
        cls.pair = cls.selection_pairs[0]
        cls.lut = _informative_lut()
        cls.services = runner.services_from_lut(cls.lut)
        cls.model_manifests = {
            model: _formal_manifest(model, cls.lut, cls.services[model])
            for model in runner.MODELS
        }
        cls.selection_bundles = {
            model: runner.make_split_bundle(
                cls.model_manifests[model], "selection", lut_artifact=cls.lut
            ) for model in runner.MODELS
        }
        cls.holdout_bundles = {
            model: runner.make_split_bundle(
                cls.model_manifests[model], "holdout", lut_artifact=cls.lut
            ) for model in runner.MODELS
        }
        cls.selection = runner.freeze_selection(
            cls.selection_bundles, cls.model_manifests, cls.lut,
            runner.current_source_hashes(),
        )

    def test_pair_manifest_converts_without_losing_full_identity(self):
        scenario = runner.scenario_from_pair(self.pair, "olmoe", SERVICES, 1.5)
        self.assertEqual(len(scenario.worlds), 4)
        self.assertEqual(len(scenario.joins), 2)
        self.assertEqual(len(core.enumerate_actions(scenario)), 4)
        self.assertEqual(
            {arm: len(core.observation_partitions(scenario, arm)) for arm in ("B0", "Q", "J", "R")},
            {"B0": 1, "Q": 2, "J": 2, "R": 4},
        )

    def test_scenario_consumes_recorded_sender_timestamps_and_transit(self):
        scenario = runner.scenario_from_pair(self.pair, "olmoe", SERVICES, 1.5)
        expected = {
            row["full_sibling_key"]: row["timestamp_us"]
            for row in self.pair["worlds"][0]["sender_history"]
            if row["event"] == "send_complete_no_commit_ack"
        }
        self.assertEqual(
            {row.task_id: row.send_complete_us for row in scenario.worlds[0].sender_history},
            expected,
        )

    def test_rehashed_sender_timestamp_drift_is_rejected(self):
        pair = copy.deepcopy(self.pair)
        row = next(row for row in pair["worlds"][0]["sender_history"] if row["event"] == "send_complete_no_commit_ack")
        row["timestamp_us"] -= 1.0
        pair["worlds"][0]["sender_history_hash"] = instances.object_sha256(pair["worlds"][0]["sender_history"])
        _rehash_pair(pair)
        with self.assertRaisesRegex(runner.PhaseMapRunnerError, "transit accounting"):
            runner.scenario_from_pair(pair, "olmoe", SERVICES, 1.5)

    def test_rehashed_transit_drift_is_rejected(self):
        pair = copy.deepcopy(self.pair)
        pair["worlds"][0]["receiver_transit_ledger"][0]["hidden_transit_us"] += 1.0
        _rehash_pair(pair)
        with self.assertRaisesRegex(runner.PhaseMapRunnerError, "transit accounting"):
            runner.scenario_from_pair(pair, "olmoe", SERVICES, 1.5)

    def test_formal_bundle_requires_exact_mother_and_lut(self):
        runner.validate_split_bundle(
            self.selection_bundles["olmoe"], "olmoe", "selection",
            model_manifest=self.model_manifests["olmoe"], lut_artifact=self.lut,
        )
        arbitrary = copy.deepcopy(self.selection_bundles["olmoe"])
        arbitrary["source_model_manifest_sha256"] = "8" * 64
        _rehash_bundle(arbitrary)
        with self.assertRaisesRegex(runner.PhaseMapRunnerError, "exact split"):
            runner.validate_split_bundle(
                arbitrary, "olmoe", "selection",
                model_manifest=self.model_manifests["olmoe"], lut_artifact=self.lut,
            )

    def test_rehashed_fifo_world_is_rejected_against_mother(self):
        bundle = copy.deepcopy(self.selection_bundles["olmoe"])
        fifo = bundle["pairs"][0]["worlds"][0]["fifo_ledgers"]
        fifo[sorted(fifo)[0]][0]["arrival_us"] -= 1.0
        _rehash_pair(bundle["pairs"][0])
        _rehash_bundle(bundle)
        with self.assertRaisesRegex(runner.PhaseMapRunnerError, "exact split"):
            runner.validate_split_bundle(
                bundle, "olmoe", "selection",
                model_manifest=self.model_manifests["olmoe"], lut_artifact=self.lut,
            )

    def test_wrong_model_manifest_is_rejected(self):
        with self.assertRaises(runner.PhaseMapRunnerError):
            runner.validate_split_bundle(
                self.selection_bundles["olmoe"], "olmoe", "selection",
                model_manifest=self.model_manifests["llmjp"], lut_artifact=self.lut,
            )

    def test_wrong_internal_lut_provenance_is_rejected(self):
        mother = copy.deepcopy(self.model_manifests["olmoe"])
        mother["service_provenance"]["lut_artifact_sha256"] = "7" * 64
        _rehash_manifest(mother)
        with self.assertRaisesRegex(runner.PhaseMapRunnerError, "LUT/model identity"):
            runner.validate_split_bundle(
                self.selection_bundles["olmoe"], "olmoe", "selection",
                model_manifest=mother, lut_artifact=self.lut,
            )

    def test_wrong_internal_lut_hidden_is_rejected(self):
        mother = copy.deepcopy(self.model_manifests["olmoe"])
        mother["service_provenance"]["lut_model_identity"]["hidden"] += 1
        _rehash_manifest(mother)
        with self.assertRaisesRegex(
            runner.PhaseMapRunnerError,
            "LUT/model identity|full model manifest validation",
        ):
            runner.validate_split_bundle(
                self.selection_bundles["olmoe"], "olmoe", "selection",
                model_manifest=mother, lut_artifact=self.lut,
            )

    def test_fake_holdout_made_from_selection_pairs_is_rejected(self):
        fake = copy.deepcopy(self.holdout_bundles["olmoe"])
        fake["pairs"] = copy.deepcopy(self.selection_bundles["olmoe"]["pairs"])
        _rehash_bundle(fake)
        with self.assertRaisesRegex(runner.PhaseMapRunnerError, "exact split"):
            runner.validate_split_bundle(
                fake, "olmoe", "holdout",
                model_manifest=self.model_manifests["olmoe"], lut_artifact=self.lut,
            )

    def test_selection_self_report_drift_is_recomputed_and_rejected(self):
        broken = copy.deepcopy(self.selection)
        broken.pop("artifact_sha256")
        index = next(
            index for index, row in enumerate(broken["selection_rows"])
            if row["kappa"] != broken["selected_kappa"]
        )
        row = broken["selection_rows"][index]
        old = float(row["B0_miss_by_model"]["olmoe"])
        delta = 1e-6 if old <= 0.999999 else -1e-6
        row["B0_miss_by_model"]["olmoe"] = old + delta
        row["pooled_B0_miss"] = sum(row["B0_miss_by_model"].values()) / 2.0
        broken = runner._self_hashed(broken)
        with self.assertRaisesRegex(runner.PhaseMapRunnerError, "recomputed"):
            runner.validate_holdout_lineage(
                self.holdout_bundles, self.selection_bundles,
                self.model_manifests, self.lut, broken,
            )

    def test_full_holdout_lineage_closes(self):
        runner.validate_holdout_lineage(
            self.holdout_bundles, self.selection_bundles,
            self.model_manifests, self.lut, self.selection,
        )

    def test_four_worlds_fold_to_two_native_requests(self):
        scenario = runner.scenario_from_pair(self.pair, "olmoe", SERVICES, 1.5)
        report = core.optimize_arm(scenario, "B0")
        self.assertEqual(report["metrics"].request_count, 2)

    def test_runner_control_transformations_are_exact(self):
        primary = runner.scenario_from_pair(self.pair, "olmoe", SERVICES, 1.5)
        for name in ("equal_j", "fanout1", "no_conflict", "shuffled_key"):
            scenario = runner.control_scenario(primary, name)
            report = core.optimize_information_lattice(scenario)
            if name == "no_conflict":
                objectives = [report["arms"][arm]["metrics"].objective for arm in ("B0", "Q", "J", "R")]
                self.assertTrue(all(value == objectives[0] for value in objectives))
            else:
                self.assertEqual(report["arms"]["R"]["metrics"].objective, report["arms"]["Q"]["metrics"].objective)

    def test_raw_ledger_includes_diagnostic_ceiling(self):
        scenario = runner.scenario_from_pair(self.pair, "olmoe", SERVICES, 1.5)
        raw = runner._raw_four_world_ledger(scenario, core.optimize_information_lattice(scenario))
        self.assertEqual(set(raw), {"B0", "Q", "J", "R", "C"})
        self.assertEqual(len(raw["C"]), 4)

    def test_no_overwrite_writer(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.json"
            runner.write_json_no_overwrite(path, {"x": 1})
            with self.assertRaisesRegex(runner.PhaseMapRunnerError, "overwrite"):
                runner.write_json_no_overwrite(path, {"x": 2})

    def test_writer_retries_short_os_write_and_verifies_bytes(self):
        real_write = runner.os.write

        def short_write(descriptor, data):
            return real_write(descriptor, data[: max(1, len(data) // 3)])

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "short-write.json"
            with mock.patch.object(runner.os, "write", side_effect=short_write):
                runner.write_json_no_overwrite(path, {"payload": "x" * 10_000})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"payload": "x" * 10_000})

    def test_git_provenance_is_optional_when_checkout_is_unavailable(self):
        with mock.patch.object(
            runner.subprocess, "run", side_effect=FileNotFoundError("git")
        ):
            provenance = runner.git_provenance()
        self.assertFalse(provenance["git_available"])
        self.assertEqual(provenance["git_head"], "UNAVAILABLE")

    def test_publisher_emits_exact_frozen_artifact_set(self):
        aggregate = core.AggregateMetrics(
            request_count=32,
            expected_miss_count=16.0,
            miss_rate=0.5,
            cvar90_normalized_tardiness=0.25,
            mean_normalized_tardiness=0.125,
            expected_join_close_sum=320.0,
            expected_miss_by_request=tuple((f"request-{i}", 0.5) for i in range(32)),
            expected_tardiness_by_request=tuple((f"request-{i}", 0.125) for i in range(32)),
        )
        holdout = {
            "models": {
                model: {
                    "baseline_metrics": {
                        name: aggregate for name in runner.baselines.BASELINE_NAMES
                    },
                    "baseline_capture": {}, "baseline_pair_ledgers": [],
                    "controls": {}, "control_pair_reports": {},
                    "depth_sensitivity_8_12": {}, "depth_sensitivity_pair_reports": [],
                    "adjacent_kappa_robustness": {}, "adjacent_kappa_pair_reports": {},
                    "milp_crosscheck": [],
                } for model in runner.MODELS
            }
        }
        decision = {"decision": "NO_GO_PHASEMAP_QUEUE_JOIN_INTERACTION", "two_model_AND": False}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "formal"
            with mock.patch.object(
                runner, "_holdout_rows",
                return_value=([{"row": i} for i in range(32)], [{"row": i} for i in range(64)]),
            ):
                runner.publish_holdout_artifacts(
                    output, bundles=self.holdout_bundles,
                    selection_bundles=self.selection_bundles,
                    model_manifests=self.model_manifests, services=self.services,
                    selection=self.selection, lut=self.lut, holdout=holdout,
                    decision=decision, source_hashes=runner.current_source_hashes(),
                )
            self.assertEqual({path.name for path in output.iterdir()}, set(runner.REQUIRED_HOLDOUT_ARTIFACTS))
            environment = json.loads((output / "environment.json").read_text(encoding="utf-8"))
            runner._validate_self(environment)
            self.assertEqual(
                set(environment["numeric_runtime"]),
                {"numpy_version", "scipy_version", "milp_callable", "highs_version"},
            )
            self.assertEqual(len(environment["git"]["git_status_porcelain_sha256"]), 64)
            self.assertIs(type(environment["git"]["git_dirty"]), bool)
            baseline_results = json.loads(
                (output / "baseline_results.json").read_text(encoding="utf-8")
            )
            runner._validate_self(baseline_results)
            self.assertEqual(
                baseline_results["models"]["olmoe"]["metrics"]["edf"]["request_count"],
                32,
            )

    def test_holdout_entry_has_no_call_to_selection_fitter(self):
        self.assertNotIn("freeze_selection", set(runner.evaluate_holdout_primary.__code__.co_names))

    def test_holdout_lineage_replay_does_not_call_selection_fitter(self):
        with mock.patch.object(
            runner.baselines,
            "fit_separable_linear",
            side_effect=AssertionError("holdout called fitter"),
        ):
            runner.validate_holdout_lineage(
                self.holdout_bundles,
                self.selection_bundles,
                self.model_manifests,
                self.lut,
                self.selection,
            )


if __name__ == "__main__":
    unittest.main()
