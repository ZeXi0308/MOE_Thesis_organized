from __future__ import annotations

import copy
import hashlib
import json
import os
import random
from pathlib import Path
import tempfile
import unittest
from unittest import mock

try:
    from .phasemap_instances import (
        PhaseMapInstanceError,
        build_model_manifests,
        build_world_manifest,
        canonical_perfect_matching,
        canonical_split_and_select,
        object_sha256,
        produce_formal_instance_artifacts,
        rebuild_from_pair_identity,
        validate_model_manifest,
    )
except ImportError:  # pragma: no cover
    from phasemap_instances import (  # type: ignore
        PhaseMapInstanceError,
        build_model_manifests,
        build_world_manifest,
        canonical_perfect_matching,
        canonical_split_and_select,
        object_sha256,
        produce_formal_instance_artifacts,
        rebuild_from_pair_identity,
        validate_model_manifest,
    )


META = {
    "model_key": "model",
    "model_revision": "revision:model",
    "data_manifest_sha256": "d" * 64,
    "placement_manifest_sha256": "e" * 64,
    "top_k": 8,
    "expected_join_candidates_per_request": 2,
    "manifest_sha256": "a" * 64,
    "route_trace_file_sha256": "b" * 64,
    "route_phase4_signoff_sha256": "c" * 64,
    "producer_signoff_file_sha256": "f" * 64,
    "placement_file_sha256": "9" * 64,
}


def metadata(model: str = "model") -> dict:
    return {**META, "model_key": model, "model_revision": f"revision:{model}"}


def raw_join(
    request: str,
    receiver: int,
    position: int,
    *,
    senders=tuple(range(8)),
    model: str = "model",
):
    siblings = []
    for slot, sender in enumerate(senders):
        siblings.append(
            {
                "model_key": model,
                "model_revision": f"revision:{model}",
                "data_manifest_sha256": "d" * 64,
                "placement_manifest_sha256": "e" * 64,
                "request_id": request,
                "forward_id": f"forward:{request}",
                "layer_id": 3,
                "token_position": position,
                "epoch": 1,
                "topk_slot": slot,
                "expert_id": slot,
                "sender_rank": sender,
                "receiver_rank": receiver,
                "deadline": -999,
                "service": -999,
                "outcome": "must-not-be-read",
            }
        )
    return {
        "join_id": f"legacy:{request}:{position}",
        "request_id": request,
        "layer_id": 3,
        "token_position": position,
        "receiver_rank": receiver,
        "siblings": siblings,
    }


def route_fixture(model: str = "model"):
    joins = []
    for receiver in range(8):
        for request_index in range(8):
            request = f"req-r{receiver}-{request_index}"
            # More than one candidate verifies the canonical join selector.
            joins.append(raw_join(request, receiver, 0, model=model))
            joins.append(raw_join(request, receiver, 1, model=model))
    return joins


class PhaseMapInstanceTests(unittest.TestCase):
    def test_split_and_join_selection_are_order_invariant(self):
        rows = route_fixture()
        expected = canonical_split_and_select(rows, metadata(), "model")
        random.Random(17).shuffle(rows)
        actual = canonical_split_and_select(rows, metadata(), "model")
        self.assertEqual(
            {
                split: [(row["request_id"], row["full_join_key"]) for row in values]
                for split, values in expected.items()
            },
            {
                split: [(row["request_id"], row["full_join_key"]) for row in values]
                for split, values in actual.items()
            },
        )
        self.assertEqual(len(expected["selection"]), 32)
        self.assertEqual(len(expected["holdout"]), 32)
        self.assertFalse(
            {row["request_id"] for row in expected["selection"]}
            & {row["request_id"] for row in expected["holdout"]}
        )

    def test_pairing_is_a_canonical_perfect_matching(self):
        selected = canonical_split_and_select(route_fixture(), metadata(), "model")["holdout"]
        expected = canonical_perfect_matching(selected)
        shuffled = list(selected)
        random.Random(23).shuffle(shuffled)
        actual = canonical_perfect_matching(shuffled)
        self.assertEqual(
            [row["edge_key"] for row in expected],
            [row["edge_key"] for row in actual],
        )
        requests = [
            request
            for pair in expected
            for request in (pair["request_a"], pair["request_b"])
        ]
        self.assertEqual(len(expected), 16)
        self.assertEqual(len(requests), len(set(requests)))
        self.assertTrue(all(pair["receiver_a"] != pair["receiver_b"] for pair in expected))
        self.assertTrue(all(len(pair["decision_senders"]) == 2 for pair in expected))
        self.assertTrue(
            all(
                len(pair["phase_carriers"][request]) >= 4
                for pair in expected
                for request in (pair["request_a"], pair["request_b"])
            )
        )

    def test_2x2_worlds_have_frozen_information_partitions(self):
        selected = canonical_split_and_select(route_fixture(), metadata(), "model")["selection"]
        pair = canonical_perfect_matching(selected)[0]
        manifest = build_world_manifest(pair, 10.0)
        self.assertEqual(len(manifest["worlds"]), 4)
        certificate = manifest["reachability_certificate"]
        self.assertEqual(
            certificate["observation_class_counts"],
            {"B0": 1, "Q": 2, "J": 2, "R": 4},
        )
        self.assertTrue(certificate["fixed_q_flip_j_q_observation_byte_identical"])
        self.assertTrue(certificate["fixed_j_flip_q_j_observation_byte_identical"])
        self.assertTrue(certificate["sender_history_byte_identical"])
        self.assertTrue(certificate["all_hidden_transit_nonnegative"])
        identity = manifest["pair_identity"]
        self.assertEqual(set(identity["joins"]), set(manifest["request_ids"]))
        for request in manifest["request_ids"]:
            join = identity["joins"][request]
            self.assertEqual(len(join["siblings"]), 8)
            self.assertEqual(
                {row["full_sibling_key"] for row in join["siblings"]},
                {
                    row["full_sibling_key"]
                    for row in identity["phase_carriers"][request]
                }
                | {
                    row["full_sibling_key"]
                    for row in identity["decision_contributions"][request].values()
                },
            )
            for sender, row in identity["decision_contributions"][request].items():
                self.assertEqual(row["identity"]["request_id"], request)
                self.assertEqual(row["identity"]["sender_rank"], int(sender))
                self.assertEqual(row["identity"]["receiver_rank"], join["receiver_rank"])
        for world in manifest["worlds"]:
            self.assertEqual(world["sender_history"], manifest["worlds"][0]["sender_history"])
            self.assertTrue(
                all(row["hidden_transit_us"] >= 0 for row in world["receiver_transit_ledger"])
            )
            depths = sorted(world["depth_by_receiver"].values())
            self.assertEqual(depths, [8, 16])
            for receiver, ledger in world["fifo_ledgers"].items():
                unfinished = [row for row in ledger if "unfinished" in row["kind"]]
                self.assertEqual(len(unfinished), world["depth_by_receiver"][receiver])
                self.assertTrue(all(row["end_us"] > 0 for row in unfinished))

    def test_carrier_and_pairing_ignore_outcome_fields(self):
        original = route_fixture()
        mutated = route_fixture()
        for join in mutated:
            for sibling in join["siblings"]:
                sibling["deadline"] = 123456
                sibling["service"] = 0.0001
                sibling["outcome"] = "opposite"
        first = canonical_perfect_matching(
            canonical_split_and_select(original, metadata(), "model")["selection"]
        )
        second = canonical_perfect_matching(
            canonical_split_and_select(mutated, metadata(), "model")["selection"]
        )
        projection = lambda pairs: [
            (
                pair["edge_key"],
                pair["decision_senders"],
                pair["decision_contributions"],
                {
                    request: [row["full_sibling_key"] for row in rows]
                    for request, rows in pair["phase_carriers"].items()
                },
            )
            for pair in pairs
        ]
        self.assertEqual(projection(first), projection(second))

    def test_receiver_identity_mismatch_is_rejected(self):
        rows = route_fixture()
        rows[0]["siblings"][0]["receiver_rank"] = 7
        with self.assertRaisesRegex(PhaseMapInstanceError, "one native join"):
            canonical_split_and_select(rows, metadata(), "model")

    def test_route_support_without_cross_receiver_edges_is_blocked(self):
        # Four already-normalized records sharing one receiver cannot form an edge.
        joins = canonical_split_and_select(route_fixture(), metadata(), "model")["selection"][:4]
        for row in joins:
            row["receiver_rank"] = 0
        with self.assertRaisesRegex(PhaseMapInstanceError, "BLOCKED_ROUTE_SUPPORT"):
            canonical_perfect_matching(joins)

    def test_manifest_self_hash_excludes_only_its_hash_field(self):
        selected = canonical_split_and_select(route_fixture(), metadata(), "model")["selection"]
        manifest = build_world_manifest(canonical_perfect_matching(selected)[0], 3.0)
        payload = dict(manifest)
        recorded = payload.pop("manifest_sha256")
        self.assertEqual(recorded, object_sha256(payload))

    def test_pair_identity_rebuild_and_equal_q_control(self):
        selected = canonical_split_and_select(route_fixture(), metadata(), "model")["selection"]
        primary = build_world_manifest(canonical_perfect_matching(selected)[0], 3.0)
        rebuilt = rebuild_from_pair_identity(primary["pair_identity"], 3.0)
        self.assertEqual(primary["manifest_sha256"], rebuilt["manifest_sha256"])
        equal = rebuild_from_pair_identity(
            primary["pair_identity"], 3.0, (8, 8), mode="equal_q"
        )
        self.assertEqual(equal["control_mode"], "equal_q")
        self.assertEqual(
            equal["reachability_certificate"]["observation_class_counts"],
            {"B0": 1, "Q": 1, "J": 2, "R": 2},
        )

    def test_sibling_identity_must_match_frozen_metadata_and_topk(self):
        rows = route_fixture()
        rows[0]["siblings"][0]["placement_manifest_sha256"] = "0" * 64
        with self.assertRaisesRegex(PhaseMapInstanceError, "disagrees with route metadata"):
            canonical_split_and_select(rows, metadata(), "model")
        rows = route_fixture()
        rows[0]["siblings"].pop()
        with self.assertRaisesRegex(PhaseMapInstanceError, "full canonical top-k"):
            canonical_split_and_select(rows, metadata(), "model")

    def test_model_manifest_certificate_recomputes_split_join_and_pair(self):
        try:
            from . import phasemap_instances as module
        except ImportError:  # pragma: no cover
            import phasemap_instances as module  # type: ignore
        with mock.patch.object(
            module, "load_verified_joins", return_value=(route_fixture(), metadata())
        ):
            manifest = build_model_manifests(
                Path("/reviewed/routes"),
                "model",
                3.0,
                lut_artifact_sha256="7" * 64,
                lut_model_identity={
                    "model_revision": "revision:model", "top_k": 8, "hidden": 64,
                },
            )
        validate_model_manifest(manifest)
        self.assertEqual(
            manifest["selection_certificate"]["normalized_join_count"], 128
        )
        corrupted = dict(manifest)
        corrupted["selection_certificate"] = dict(manifest["selection_certificate"])
        corrupted["selection_certificate"]["candidate_universe"] = list(
            manifest["selection_certificate"]["candidate_universe"][:-1]
        )
        certificate_payload = dict(corrupted["selection_certificate"])
        certificate_payload.pop("certificate_sha256")
        corrupted["selection_certificate"]["certificate_sha256"] = object_sha256(
            certificate_payload
        )
        manifest_payload = dict(corrupted)
        manifest_payload.pop("manifest_sha256")
        corrupted["manifest_sha256"] = object_sha256(manifest_payload)
        with self.assertRaisesRegex(PhaseMapInstanceError, "universe hash/census"):
            validate_model_manifest(corrupted)

        def rehash_after_pair_tamper(value: dict) -> dict:
            pair = value["splits"]["holdout"]["pairs"][0]
            pair_payload = dict(pair)
            pair_payload.pop("manifest_sha256")
            pair["manifest_sha256"] = object_sha256(pair_payload)
            model_payload = dict(value)
            model_payload.pop("manifest_sha256")
            value["manifest_sha256"] = object_sha256(model_payload)
            return value

        mutations = {
            "arrival": lambda pair: pair["worlds"][0]["fifo_ledgers"][
                next(iter(pair["worlds"][0]["fifo_ledgers"]))
            ][0].__setitem__("arrival_us", 123.0),
            "hidden_transit": lambda pair: pair["worlds"][0][
                "receiver_transit_ledger"
            ][0].__setitem__("hidden_transit_us", 123.0),
            "sender_timestamp": lambda pair: pair["worlds"][0]["sender_history"][
                0
            ].__setitem__("timestamp_us", -123.0),
            "world_census": lambda pair: pair["worlds"].pop(),
        }
        for name, mutate in mutations.items():
            with self.subTest(tamper=name):
                tampered = copy.deepcopy(manifest)
                mutate(tampered["splits"]["holdout"]["pairs"][0])
                rehash_after_pair_tamper(tampered)
                with self.assertRaisesRegex(
                    PhaseMapInstanceError, "canonical reconstruction"
                ):
                    validate_model_manifest(tampered)

        # Rebuilding every derived world and both outer hashes is still not
        # authority to invent a join identity outside the route certificate.
        # A published JSON artifact has no Python object aliases between its
        # certificate and embedded pair copy.
        tampered = json.loads(json.dumps(manifest))
        pair = tampered["splits"]["holdout"]["pairs"][0]
        request = pair["request_ids"][0]
        embedded = pair["pair_identity"]["joins"][request]
        embedded["full_join_identity"]["model_revision"] = "revision:attacker"
        embedded["full_join_key"] = object_sha256(embedded["full_join_identity"])
        tampered["splits"]["holdout"]["pairs"][0] = rebuild_from_pair_identity(
            pair["pair_identity"], 3.0, (8, 16)
        )
        model_payload = dict(tampered)
        model_payload.pop("manifest_sha256")
        tampered["manifest_sha256"] = object_sha256(model_payload)
        with self.assertRaisesRegex(
            PhaseMapInstanceError, "certified candidate universe"
        ):
            validate_model_manifest(tampered)

    def test_formal_producer_emits_both_models_and_refuses_overwrite(self):
        try:
            from . import capture_phasemap_lut_gpu as lut_module
            from . import phasemap_instances as module
        except ImportError:  # pragma: no cover
            import capture_phasemap_lut_gpu as lut_module  # type: ignore
            import phasemap_instances as module  # type: ignore
        lut = {
            "artifact_sha256": "7" * 64,
            "model_inputs": {
                model: {
                    "model_revision": f"revision:{model}",
                    "top_k": 8,
                    "hidden": 2048 if model == "olmoe" else 512,
                }
                for model in ("olmoe", "llmjp")
            },
            "summary": [
                {
                    "model_key": model,
                    "component": "receiver_unpack",
                    "median_cuda_event_us": 3.0,
                }
                for model in ("olmoe", "llmjp")
            ],
        }

        def routes(_root: Path, model: str):
            return route_fixture(model), metadata(model)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "formal"
            with mock.patch.object(lut_module, "validate_artifact", return_value=None), mock.patch.object(
                module, "load_verified_joins", side_effect=routes
            ):
                produced = produce_formal_instance_artifacts(
                    Path("/reviewed/routes"), lut, output
                )
                self.assertEqual(
                    set(produced),
                    {
                        "olmoe_model_manifest", "olmoe_selection", "olmoe_holdout",
                        "llmjp_model_manifest", "llmjp_selection", "llmjp_holdout",
                        "source_manifest",
                    },
                )
                self.assertTrue(all((output / filename).is_file() for filename in produced.values()))
                self.assertFalse((output / ".INCOMPLETE").exists())
                source = json.loads(
                    (output / "source_manifest.json").read_text(encoding="utf-8")
                )
                for row in source["published_files"].values():
                    self.assertEqual(
                        row["sha256"],
                        hashlib.sha256((output / row["path"]).read_bytes()).hexdigest(),
                    )
                with self.assertRaisesRegex(PhaseMapInstanceError, "refusing to overwrite"):
                    produce_formal_instance_artifacts(Path("/reviewed/routes"), lut, output)
                os.chmod(output, 0o755)

    def test_formal_producer_rejects_lut_route_identity_drift(self):
        try:
            from . import capture_phasemap_lut_gpu as lut_module
            from . import phasemap_instances as module
        except ImportError:  # pragma: no cover
            import capture_phasemap_lut_gpu as lut_module  # type: ignore
            import phasemap_instances as module  # type: ignore
        lut = {
            "artifact_sha256": "7" * 64,
            "model_inputs": {
                model: {
                    "model_revision": "wrong",
                    "top_k": 8,
                    "hidden": 2048 if model == "olmoe" else 512,
                }
                for model in ("olmoe", "llmjp")
            },
            "summary": [
                {"model_key": model, "component": "receiver_unpack", "median_cuda_event_us": 3.0}
                for model in ("olmoe", "llmjp")
            ],
        }

        def routes(_root: Path, model: str):
            return route_fixture(model), metadata(model)

        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            lut_module, "validate_artifact", return_value=None
        ), mock.patch.object(module, "load_verified_joins", side_effect=routes):
            with self.assertRaisesRegex(PhaseMapInstanceError, "differs from route identity"):
                produce_formal_instance_artifacts(
                    Path("/reviewed/routes"), lut, Path(directory) / "formal"
                )

    def test_formal_hidden_and_depths_are_frozen(self):
        try:
            from . import capture_phasemap_lut_gpu as lut_module
            from . import phasemap_instances as module
        except ImportError:  # pragma: no cover
            import capture_phasemap_lut_gpu as lut_module  # type: ignore
            import phasemap_instances as module  # type: ignore
        lut = {
            "artifact_sha256": "7" * 64,
            "model_inputs": {
                model: {
                    "model_revision": f"revision:{model}",
                    "top_k": 8,
                    "hidden": 2048 if model == "olmoe" else 512,
                }
                for model in ("olmoe", "llmjp")
            },
            "summary": [
                {
                    "model_key": model,
                    "component": "receiver_unpack",
                    "median_cuda_event_us": 3.0,
                }
                for model in ("olmoe", "llmjp")
            ],
        }

        def routes(_root: Path, model: str):
            return route_fixture(model), metadata(model)

        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            lut_module, "validate_artifact", return_value=None
        ), mock.patch.object(module, "load_verified_joins", side_effect=routes):
            bad_hidden = copy.deepcopy(lut)
            bad_hidden["model_inputs"]["olmoe"]["hidden"] = 2176
            with self.assertRaisesRegex(PhaseMapInstanceError, "hidden"):
                produce_formal_instance_artifacts(
                    Path("/reviewed/routes"), bad_hidden, Path(directory) / "bad-hidden"
                )
            with self.assertRaisesRegex(PhaseMapInstanceError, "depths"):
                produce_formal_instance_artifacts(
                    Path("/reviewed/routes"), lut, Path(directory) / "bad-depths", (7, 9)
                )


if __name__ == "__main__":
    unittest.main()
