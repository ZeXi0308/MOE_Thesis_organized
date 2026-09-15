from __future__ import annotations

import unittest
from dataclasses import replace

try:
    from .build_fixed_replica_instances import (
        assign_cluster_splits,
        build_instances,
        validate_instance_contracts,
        validate_instance_split_disjointness,
    )
    from .census_fragmentation import (
        _connected_component_labels,
        _continuous_policy_assignments,
        _validate_exposure_coverage,
        _waves,
        _work,
        evaluate_gate1_predicate,
    )
    from .core import Contribution, CurvePoint, ProtocolError, ServiceCatalog
    from .policies import LeastLoadPolicy, assign_online
except ImportError:
    from build_fixed_replica_instances import (
        assign_cluster_splits,
        build_instances,
        validate_instance_contracts,
        validate_instance_split_disjointness,
    )
    from census_fragmentation import (
        _connected_component_labels,
        _continuous_policy_assignments,
        _validate_exposure_coverage,
        _waves,
        _work,
        evaluate_gate1_predicate,
    )
    from core import Contribution, CurvePoint, ProtocolError, ServiceCatalog
    from policies import LeastLoadPolicy, assign_online


def route(
    *,
    model: str = "m0",
    request_id: str,
    document_id: str,
    layer: int = 0,
    decode_step: int = 0,
    ready_us: float = 0.0,
    end_us: float | None = None,
) -> Contribution:
    return Contribution(
        model=model,
        phase="decode",
        request_id=request_id,
        sample_id=decode_step,
        arrival_us=0.0,
        deadline_us=1_000.0,
        layer=layer,
        token_position=decode_step,
        rank=1,
        expert_id=0,
        gate_weight=1.0,
        src_replica=0,
        input_event_id=f"{request_id}:decode:{decode_step}",
        token_id=decode_step,
        decode_step=decode_step,
        layer_id=layer,
        topk_slot=0,
        source_rank=0,
        document_id=document_id,
        request_arrival_us=0.0,
        layer_ready_us=ready_us,
        route_end_us=ready_us,
        dispatch_end_us=ready_us if end_us is not None else -1.0,
        expert_start_us=ready_us if end_us is not None else -1.0,
        expert_end_us=end_us if end_us is not None else -1.0,
        combine_end_us=end_us if end_us is not None else -1.0,
    )


def gate_cell(point: float, *, ci_low: float = 0.06, actionable: float = 0.25):
    return {"point": point, "ci_low": ci_low, "actionable_rate": actionable}


class CausalWaveTest(unittest.TestCase):
    @staticmethod
    def catalog() -> ServiceCatalog:
        return ServiceCatalog(
            {("m0", 0): [
                CurvePoint(1, 10, 11),
                CurvePoint(2, 14, 15),
                CurvePoint(4, 20, 22),
            ]}
        )

    def test_overlapping_document_blocks_share_one_independent_cluster(self) -> None:
        labels = _connected_component_labels(
            (("document:d0", "document:d1"), ("document:d0", "document:d2"), ("document:d3",))
        )
        self.assertEqual(labels[0], labels[1])
        self.assertNotEqual(labels[0], labels[2])

    def test_wave_never_contains_two_decode_events_from_one_request(self) -> None:
        rows = [
            route(request_id=request_id, document_id=request_id, decode_step=step, ready_us=step * 10.0)
            for step in range(2)
            for request_id in ("r0", "r1")
        ]

        waves = list(_waves(rows, concurrency=2))

        self.assertEqual(len(waves), 2)
        for _, _, _, _, _, _, wave in waves:
            request_events: dict[str, set[tuple[str, int]]] = {}
            for row in wave:
                request_events.setdefault(row.request_id, set()).add(
                    (row.input_event_id, row.decode_step)
                )
            self.assertEqual(set(request_events), {"r0", "r1"})
            self.assertTrue(all(len(events) == 1 for events in request_events.values()))

    def test_formal_wave_requires_one_common_overlap_interval(self) -> None:
        pairwise_only = [
            route(request_id="r0", document_id="d0", ready_us=0.0, end_us=10.0),
            route(request_id="r1", document_id="d1", ready_us=0.0, end_us=4.0),
            route(request_id="r2", document_id="d2", ready_us=6.0, end_us=10.0),
        ]
        self.assertEqual(list(_waves(pairwise_only, concurrency=3)), [])

        common_overlap = [
            route(request_id="r0", document_id="d0", ready_us=0.0, end_us=10.0),
            route(request_id="r1", document_id="d1", ready_us=2.0, end_us=8.0),
            route(request_id="r2", document_id="d2", ready_us=4.0, end_us=6.0),
        ]
        self.assertEqual(len(list(_waves(common_overlap, concurrency=3))), 1)

    def test_formal_wave_search_does_not_greedily_hide_later_clique(self) -> None:
        rows = [
            route(request_id="a", document_id="da", ready_us=0.0, end_us=10.0),
            route(request_id="b", document_id="db", ready_us=0.0, end_us=4.0),
            route(request_id="c", document_id="dc", ready_us=6.0, end_us=10.0),
            route(request_id="d", document_id="dd", ready_us=6.0, end_us=10.0),
        ]
        waves = list(_waves(rows, concurrency=3))
        self.assertEqual(len(waves), 1)
        self.assertEqual({row.request_id for row in waves[0][-1]}, {"a", "c", "d"})

    def test_back_to_back_half_open_intervals_are_not_concurrent(self) -> None:
        rows = [
            route(request_id="a", document_id="da", ready_us=0.0, end_us=10.0),
            route(request_id="b", document_id="db", ready_us=10.0, end_us=20.0),
        ]
        self.assertEqual(list(_waves(rows, concurrency=2)), [])

    def test_least_load_census_keeps_prefix_state_across_waves(self) -> None:
        rows = [
            route(request_id=f"r{index}", document_id=f"d{index}", ready_us=ready)
            for index, ready in enumerate((0.0, 0.0, 0.0, 5.0))
        ]
        assignments = _continuous_policy_assignments(
            rows,
            self.catalog(),
            2,
            "current_least_load",
            seed=7,
        )
        self.assertEqual(assignments[rows[-1].route_semantic_id], 1)
        self.assertEqual(
            assign_online([rows[-1]], LeastLoadPolicy(), self.catalog(), 2),
            [0],
        )

    def test_exposure_must_cover_every_route_concurrency_coordinate(self) -> None:
        rows = [route(request_id="r0", document_id="d0")]
        with self.assertRaisesRegex(ProtocolError, "exactly cover"):
            _validate_exposure_coverage(
                rows,
                (1, 4),
                {("m0", "decode", 0, 1): 100.0},
            )

    def test_actionability_requires_a_common_legal_replica(self) -> None:
        rows = [
            replace(route(request_id="r0", document_id="d0"), legal_replica_set=(0,)),
            replace(route(request_id="r1", document_id="d1"), legal_replica_set=(1,)),
        ]
        result = _work(self.catalog(), rows, [0, 1], 2)
        self.assertFalse(result["actionable"])
        self.assertEqual(result["actionable_fragmented_work_us"], 0.0)
        self.assertEqual(result["fragmented_work_us"], result["consolidated_work_us"])


class GateOnePredicateTest(unittest.TestCase):
    def test_fifteen_percent_witness_is_a_subset_of_ten_percent_witness(self) -> None:
        ten_only = ("decode", 8, 2, "least_load")
        raw_fifteen_but_bad_lcb = ("decode", 16, 2, "hash")
        common = {
            ten_only: {"m0": gate_cell(0.12), "m1": gate_cell(0.13)},
            raw_fifteen_but_bad_lcb: {
                "m0": gate_cell(0.18, ci_low=0.04),
                "m1": gate_cell(0.17),
            },
        }

        result = evaluate_gate1_predicate(
            common, ("m0", "m1"), (ten_only, raw_fifteen_but_bad_lcb)
        )

        self.assertEqual(result["passing10"], (ten_only,))
        self.assertEqual(result["passing15"], ())
        self.assertLessEqual(set(result["passing15"]), set(result["passing10"]))

    def test_kill_requires_every_preregistered_common_cell_to_be_low(self) -> None:
        low = ("decode", 8, 2, "hash")
        not_low = ("decode", 16, 2, "hash")
        common = {
            low: {"m0": gate_cell(0.01), "m1": gate_cell(0.02)},
            not_low: {"m0": gate_cell(0.04), "m1": gate_cell(0.08)},
        }
        result = evaluate_gate1_predicate(common, ("m0", "m1"), (low, not_low))
        self.assertFalse(result["all_preregistered_common_cells_low"])

        all_low = {
            key: {model: gate_cell(0.01) for model in ("m0", "m1")}
            for key in (low, not_low)
        }
        result = evaluate_gate1_predicate(all_low, ("m0", "m1"), (low, not_low))
        self.assertTrue(result["all_preregistered_common_cells_low"])

        result = evaluate_gate1_predicate({low: all_low[low]}, ("m0", "m1"), (low, not_low))
        self.assertFalse(result["all_preregistered_common_cells_low"])
        self.assertEqual(result["missing_preregistered_cells"], (not_low,))


class FrozenSplitTest(unittest.TestCase):
    @staticmethod
    def corpus() -> list[Contribution]:
        return [
            route(
                model=model,
                request_id=f"r{index:02d}",
                document_id=f"d{index:02d}",
                layer=layer,
            )
            for index in range(32)
            for model in ("m0", "m1")
            for layer in (0, 1)
        ]

    def test_document_split_is_frozen_across_models_and_layers(self) -> None:
        instances, assignments = build_instances(
            self.corpus(),
            replicas=2,
            tokens_per_instance=1,
            phase="decode",
            calibration_fraction=0.5,
            seed=7,
        )

        audit = validate_instance_split_disjointness(instances)
        contract = validate_instance_contracts(instances, require_formal_v3=False)
        self.assertEqual(contract["instances"], len(instances))
        self.assertGreater(audit["calibration_documents"], 0)
        self.assertGreater(audit["evaluation_documents"], 0)
        for instance in instances:
            for document_id in instance["document_ids"]:
                self.assertEqual(
                    assignments[f"document:{document_id}"], instance["split"]
                )

        independent_by_document: dict[str, set[str]] = {}
        for instance in instances:
            for document_id in instance["document_ids"]:
                independent_by_document.setdefault(document_id, set()).add(
                    instance["independent_cluster_id"]
                )
        self.assertTrue(all(len(labels) == 1 for labels in independent_by_document.values()))

        documents = {
            split: {
                document_id
                for instance in instances
                if instance["split"] == split
                for document_id in instance["document_ids"]
            }
            for split in ("calibration", "evaluation")
        }
        self.assertTrue(documents["calibration"].isdisjoint(documents["evaluation"]))

    def test_split_fails_closed_instead_of_relabeling_instances(self) -> None:
        with self.assertRaisesRegex(ProtocolError, "insufficient"):
            assign_cluster_splits(
                [route(request_id="r0", document_id="d0")],
                calibration_fraction=0.5,
                seed=7,
            )

        with self.assertRaisesRegex(ProtocolError, "did not populate both"):
            assign_cluster_splits(
                self.corpus(),
                calibration_fraction=0.000001,
                seed=7,
            )

    def test_validator_rejects_request_or_document_overlap(self) -> None:
        request_overlap = [
            {
                "split": "calibration",
                "request_ids": ["r0"],
                "document_ids": ["d0"],
                "split_cluster_ids": ["document:d0"],
            },
            {
                "split": "evaluation",
                "request_ids": ["r0"],
                "document_ids": ["d1"],
                "split_cluster_ids": ["document:d1"],
            },
        ]
        with self.assertRaisesRegex(ProtocolError, "requests cross"):
            validate_instance_split_disjointness(request_overlap)

        document_overlap = [
            {
                "split": "calibration",
                "request_ids": ["r0"],
                "document_ids": ["d0"],
                "split_cluster_ids": ["request:r0"],
            },
            {
                "split": "evaluation",
                "request_ids": ["r1"],
                "document_ids": ["d0"],
                "split_cluster_ids": ["request:r1"],
            },
        ]
        with self.assertRaisesRegex(ProtocolError, "documents cross"):
            validate_instance_split_disjointness(document_overlap)

    def test_gate_consumers_reject_legacy_or_tampered_instance_schema(self) -> None:
        legacy = [
            {
                "split": "calibration",
                "request_ids": ["r0"],
                "document_ids": ["d0"],
                "split_cluster_ids": ["document:d0"],
            },
            {
                "split": "evaluation",
                "request_ids": ["r1"],
                "document_ids": ["d1"],
                "split_cluster_ids": ["document:d1"],
            },
        ]
        validate_instance_split_disjointness(legacy)
        with self.assertRaisesRegex(ProtocolError, "bcrd-instance-v2"):
            validate_instance_contracts(legacy, require_formal_v3=False)

        instances, _ = build_instances(
            self.corpus(),
            replicas=2,
            tokens_per_instance=1,
            phase="decode",
            calibration_fraction=0.5,
            seed=7,
        )
        instances[0]["legal_targets_by_expert"] = {"0": [0]}
        with self.assertRaisesRegex(ProtocolError, "legal target manifest"):
            validate_instance_contracts(instances, require_formal_v3=False)


if __name__ == "__main__":
    unittest.main()
