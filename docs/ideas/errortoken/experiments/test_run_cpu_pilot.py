from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from run_cpu_pilot import (
    Candidate,
    build_mismatch_onset_risks,
    enumerate_exact_matched_null,
    freeze_policy_selections,
    group_candidate_pairs,
    load_candidates,
    require_manifest_hash,
    retrospective_verdict,
    validate_config,
)


def candidate(
    target_id: str,
    victim_id: str,
    *,
    layer: int,
    rank: int,
    gate: float,
    risk: float | None,
) -> Candidate:
    return Candidate(
        target_id=target_id,
        victim_id=victim_id,
        layer=layer,
        expert_id=7,
        topk_rank_zero_based=rank,
        route_rank_one_based=rank + 1,
        gate_weight=gate,
        mismatch_onset_risk=risk,
        calibration_row_count=3 if risk is not None else 0,
    )


class ErrorTokenCpuSelectorTest(unittest.TestCase):
    def test_mismatch_onset_risk_uses_first_mismatch_and_never_zero(self) -> None:
        calls = [
            {
                "arm": "calibration",
                "call_index": 0,
                "layer": 2,
                "expert_id": 7,
                "m": 2,
                "row_ids": ["a", "b"],
                "row_records": [
                    {"layer": 2, "expert_id": 7, "route_rank": 1},
                    {"layer": 2, "expert_id": 7, "route_rank": 1},
                ],
                "repeat_row_exact": [[True, True], [True, True]],
            },
            {
                "arm": "calibration",
                "call_index": 1,
                "layer": 2,
                "expert_id": 7,
                "m": 4,
                "row_ids": ["a", "b", "c", "d"],
                "row_records": [
                    {"layer": 2, "expert_id": 7, "route_rank": 1},
                    {"layer": 2, "expert_id": 7, "route_rank": 1},
                    {"layer": 2, "expert_id": 7, "route_rank": 2},
                    {"layer": 2, "expert_id": 7, "route_rank": 2},
                ],
                "repeat_row_exact": [[False, True, True, True], [False, True, True, True]],
            },
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "calibration.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in calls), encoding="utf-8")
            risks = build_mismatch_onset_risks(path, [2, 4])
        # Row a first mismatches at M=4 => 2/4; row b never mismatches => 0.
        self.assertAlmostEqual(risks[(2, 7, 1)].mismatch_onset_risk, 0.25)
        self.assertEqual(risks[(2, 7, 1)].calibration_row_count, 2)

    def test_stablebatch_rank_is_converted_to_one_based_key(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "targets.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "victim_id": "v0",
                        "layer": 2,
                        "expert_id": 7,
                        "topk_rank": 0,
                        "gate_weight": 0.2,
                        "next_layer_topk_margin": -999,
                        "selection_score": 999,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            candidates = load_candidates(
                path,
                {(2, 7, 1): type("Risk", (), {"mismatch_onset_risk": 0.75, "calibration_row_count": 4})()},
                1,
            )
        self.assertEqual(candidates[0].route_rank_one_based, 1)
        self.assertEqual(candidates[0].mismatch_onset_risk, 0.75)

    def test_selector_and_two_baselines_are_distinct_and_causally_ordered(self) -> None:
        values = [
            candidate("target-00", "v0", layer=5, rank=0, gate=0.9, risk=0.2),
            candidate("target-01", "v0", layer=2, rank=2, gate=0.1, risk=0.8),
            candidate("target-02", "v1", layer=3, rank=1, gate=0.2, risk=0.9),
            candidate("target-03", "v1", layer=1, rank=0, gate=0.8, risk=0.1),
        ]
        pairs = group_candidate_pairs(values, expected_victims=2, candidates_per_victim=2)
        frozen = freeze_policy_selections(
            pairs,
            {
                "errortoken_mismatch_onset": {"threshold": 0.5},
                "gate_weight_first": {"threshold": 0.5},
                "topk_rank_first": {"threshold": 1},
            },
        )
        self.assertEqual([item.target_id for item in frozen["errortoken_mismatch_onset"]], ["target-01", "target-02"])
        self.assertEqual([item.target_id for item in frozen["gate_weight_first"]], ["target-03", "target-00"])
        self.assertEqual([item.target_id for item in frozen["topk_rank_first"]], ["target-03", "target-00"])
        self.assertEqual([item.layer for item in frozen["errortoken_mismatch_onset"]], [2, 3])

    def test_causal_first_eligible_does_not_replace_with_higher_future_score(self) -> None:
        values = [
            candidate("early", "v0", layer=1, rank=1, gate=0.61, risk=0.51),
            candidate("future", "v0", layer=9, rank=0, gate=0.99, risk=0.99),
        ]
        pairs = group_candidate_pairs(values, expected_victims=1, candidates_per_victim=2)
        frozen = freeze_policy_selections(
            pairs,
            {
                "errortoken_mismatch_onset": {"threshold": 0.5},
                "gate_weight_first": {"threshold": 0.6},
                "topk_rank_first": {"threshold": 1},
            },
        )
        self.assertTrue(all(chosen[0].target_id == "early" for chosen in frozen.values()))

    def test_exact_matched_null_enumerates_every_assignment(self) -> None:
        values = [
            candidate("a0", "v0", layer=0, rank=0, gate=0.1, risk=0.1),
            candidate("a1", "v0", layer=0, rank=1, gate=0.2, risk=0.2),
            candidate("b0", "v1", layer=1, rank=0, gate=0.1, risk=0.1),
            candidate("b1", "v1", layer=1, rank=1, gate=0.2, risk=0.2),
        ]
        pairs = group_candidate_pairs(values, expected_victims=2, candidates_per_victim=2)
        null = enumerate_exact_matched_null(
            pairs, {"a0": False, "a1": True, "b0": False, "b1": True}
        )
        self.assertEqual(null["enumerated_assignments"], 4)
        self.assertEqual(null["hit_count_histogram"], {"0": 1, "1": 2, "2": 1})
        self.assertEqual(null["hit_count_mean"], 1.0)

    def test_config_statically_rejects_leakage_field(self) -> None:
        config_path = Path(__file__).parent / "configs" / "cpu_selector_v1.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        validate_config(config)
        config["selector_contract"]["selector_fields"].append("selection_score")
        with self.assertRaisesRegex(ValueError, "selector fields drifted|forbidden"):
            validate_config(config)

    def test_manifest_hash_check_fails_closed_on_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sealed.jsonl"
            path.write_text("original\n", encoding="utf-8")
            expected = __import__("hashlib").sha256(b"original\n").hexdigest()
            self.assertEqual(require_manifest_hash(path, expected, "fixture"), expected)
            path.write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "sealed hash mismatch"):
                require_manifest_hash(path, expected, "fixture")

    def test_retrospective_verdict_never_claims_runtime_execution(self) -> None:
        verdict, _ = retrospective_verdict(
            {"selected_hits": 3, "exact_one_sided_p_ge": 0.25}, 2.0, 0.1
        )
        self.assertEqual(verdict, "INCONCLUSIVE_RETROSPECTIVE_ENRICHMENT")


if __name__ == "__main__":
    unittest.main()
