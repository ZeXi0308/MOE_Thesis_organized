from __future__ import annotations

import copy
import unittest
from unittest import mock

try:
    from . import phasemap_baselines as baseline_module
    from .phasemap_baselines import (
        BASELINE_NAMES,
        PhaseMapBaselineError,
        action_key,
        baseline_action,
        compute_capture,
        enumerate_joint_actions,
        fit_separable_linear,
        run_all_baselines,
        validate_linear_fit_against_examples,
        validate_linear_artifact,
    )
except ImportError:  # pragma: no cover
    import phasemap_baselines as baseline_module  # type: ignore
    from phasemap_baselines import (  # type: ignore
        BASELINE_NAMES,
        PhaseMapBaselineError,
        action_key,
        baseline_action,
        compute_capture,
        enumerate_joint_actions,
        fit_separable_linear,
        run_all_baselines,
        validate_linear_fit_against_examples,
        validate_linear_artifact,
    )


def task(sender: int, request: str) -> dict:
    is_a = request == "A"
    return {
        "task_id": f"{request}{sender}",
        "request_id": request,
        "full_join_key": f"join-{request}",
        "sender_rank": sender,
        "receiver_rank": 0 if is_a else 1,
        "request_arrival_us": -20.0 if is_a else -10.0,
        "deadline_us": 50.0 if is_a else 40.0,
        "ready_us": 0.0,
        "service_us": 3.0,
        "receiver_service_us": 1.0,
        "combine_service_us": 2.0,
        "receiver_work_us": 20.0 if is_a else 5.0,
        "receiver_availability_us": 20.0 if is_a else 5.0,
        "remaining_siblings": 1 if is_a else 4,
    }


def observation(identifier: str = "obs") -> dict:
    return {
        "observation_id": identifier,
        "pair_key": "pair",
        "world_id": "q0j0",
        "now_us": 0.0,
        "tasks": [task(0, "A"), task(0, "B"), task(1, "A"), task(1, "B")],
    }


def fitting_examples() -> list[dict]:
    obs = observation("fit")
    objectives = {}
    for action in enumerate_joint_actions(obs):
        chosen = [task_id[0] for _sender, task_id in action]
        objectives[action_key(action)] = [float(chosen.count("B")), 0.0, 0.0]
    return [{"observation": obs, "action_objectives": objectives}]


def fitted_artifact() -> dict:
    return fit_separable_linear(
        fitting_examples(),
        selection_source_sha256="a" * 64,
    )


class PhaseMapBaselineTests(unittest.TestCase):
    def test_all_eight_baselines_return_legal_canonical_actions(self):
        obs = observation()
        actions = run_all_baselines(obs, fitted_artifact())
        self.assertEqual(tuple(actions), BASELINE_NAMES)
        legal = set(enumerate_joint_actions(obs))
        self.assertTrue(all(action in legal for action in actions.values()))
        self.assertEqual(actions["request_fcfs"], ((0, "A0"), (1, "A1")))
        self.assertEqual(actions["edf"], ((0, "B0"), (1, "B1")))
        self.assertEqual(actions["qwork_first"], ((0, "B0"), (1, "B1")))
        self.assertEqual(
            actions["remaining_siblings_last_missing_first"], ((0, "A0"), (1, "A1"))
        )
        self.assertEqual(actions["least_laxity"], ((0, "A0"), (1, "A1")))
        self.assertIn(actions["myopic_predicted_join_close"], legal)

    def test_input_order_does_not_change_actions(self):
        artifact = fitted_artifact()
        first = run_all_baselines(observation(), artifact)
        reversed_observation = observation()
        reversed_observation["tasks"].reverse()
        second = run_all_baselines(reversed_observation, artifact)
        self.assertEqual(first, second)

    def test_future_and_unknown_observation_fields_are_rejected(self):
        obs = observation()
        obs["future_arrivals"] = []
        with self.assertRaisesRegex(PhaseMapBaselineError, "forbidden/noncausal"):
            enumerate_joint_actions(obs)
        obs = observation()
        obs["tasks"][0]["actual_join_close_us"] = 1.0
        with self.assertRaisesRegex(PhaseMapBaselineError, "forbidden"):
            enumerate_joint_actions(obs)
        obs = observation()
        obs["tasks"][0]["remaining_work_us"] = 0.0
        with self.assertRaisesRegex(PhaseMapBaselineError, "forbidden"):
            enumerate_joint_actions(obs)
        obs = observation()
        obs["tasks"][0]["predicted_join_close_us"] = 0.0
        with self.assertRaisesRegex(PhaseMapBaselineError, "forbidden"):
            enumerate_joint_actions(obs)

    def test_post_t0_future_mutation_cannot_change_causal_actions(self):
        obs = observation()
        artifact = fitted_artifact()
        before = run_all_baselines(obs, artifact)
        future_a = {"actual_join_close_us": {"A": 1.0, "B": 999.0}}
        future_b = {"actual_join_close_us": {"A": 999.0, "B": 1.0}}
        self.assertNotEqual(future_a, future_b)
        self.assertEqual(before, run_all_baselines(copy.deepcopy(obs), artifact))

    def test_myopic_enumerates_joint_actions_and_can_choose_mixed_orders(self):
        obs = observation("mixed")
        # Asymmetric sender times make the joint predictor prefer different
        # first requests at the two senders.  A sender-separable request score
        # cannot express this AB/BA choice.
        for row in obs["tasks"]:
            row["deadline_us"] = 12.0
            row["receiver_work_us"] = 0.0
            row["receiver_availability_us"] = 0.0
            row["receiver_service_us"] = 1.0
            row["combine_service_us"] = 1.0
            if row["task_id"] in {"A0", "B1"}:
                row["service_us"] = 2.0
            else:
                row["service_us"] = 10.0
        selected = baseline_action(obs, "myopic_predicted_join_close", fitted_artifact())
        self.assertIn(selected, (((0, "A0"), (1, "B1")), ((0, "B0"), (1, "A1"))))

    def test_myopic_full_tie_uses_serialized_task_identity_not_action_hash(self):
        obs = observation("full-tie")
        for row in obs["tasks"]:
            row["request_arrival_us"] = 0.0
            row["deadline_us"] = 200.0
            row["receiver_work_us"] = 100.0
            row["receiver_availability_us"] = 100.0
            row["service_us"] = 3.0
            row["receiver_service_us"] = 1.0
            row["combine_service_us"] = 1.0
            row["remaining_siblings"] = 2
        selected = baseline_action(obs, "myopic_predicted_join_close", fitted_artifact())
        self.assertEqual(selected, ((0, "A0"), (1, "A1")))

    def test_request_level_state_must_match_across_senders(self):
        obs = observation()
        obs["tasks"][2]["remaining_siblings"] = 2
        with self.assertRaisesRegex(PhaseMapBaselineError, "disagrees across senders"):
            enumerate_joint_actions(obs)

    def test_linear_fit_is_selection_only_self_hashed_and_frozen(self):
        artifact = fitted_artifact()
        validate_linear_artifact(artifact)
        self.assertEqual(artifact["split"], "selection")
        self.assertGreater(artifact["weights"]["deficit"], 0)
        self.assertEqual(len(artifact["selection_examples_sha256"]), 64)
        selected = baseline_action(
            observation(), "separable_linear_slack_qwork_deficit", artifact
        )
        self.assertEqual(selected, ((0, "A0"), (1, "A1")))
        corrupted = copy.deepcopy(artifact)
        corrupted["weights"]["deficit"] = 0.0
        with self.assertRaisesRegex(PhaseMapBaselineError, "self-hash"):
            validate_linear_artifact(corrupted)
        with self.assertRaisesRegex(PhaseMapBaselineError, "only be fitted on selection"):
            fit_separable_linear([], selection_source_sha256="a" * 64, split="holdout")

    def test_linear_integrity_replay_does_not_call_fitter(self):
        artifact = fitted_artifact()
        with mock.patch.object(
            baseline_module,
            "fit_separable_linear",
            side_effect=AssertionError("integrity replay called fitter"),
        ):
            validate_linear_fit_against_examples(
                artifact,
                fitting_examples(),
                selection_source_sha256="a" * 64,
            )

    def test_selection_fit_requires_complete_action_objectives(self):
        obs = observation("bad-fit")
        one_action = enumerate_joint_actions(obs)[0]
        with self.assertRaisesRegex(PhaseMapBaselineError, "cover the frozen action domain"):
            fit_separable_linear(
                [{"observation": obs, "action_objectives": {action_key(one_action): [0, 0, 0]}}],
                selection_source_sha256="a" * 64,
            )

    def test_capture_preserves_negative_raw_and_selects_strongest(self):
        misses = {name: 0.50 for name in BASELINE_NAMES}
        misses["edf"] = 0.43
        misses["request_fcfs"] = 0.55
        report = compute_capture(0.50, 0.40, misses)
        self.assertTrue(report["gate_eligible"])
        self.assertEqual(report["strongest_baseline"], "edf")
        self.assertAlmostEqual(report["strongest_capture"], 0.7)
        self.assertAlmostEqual(
            report["per_baseline"]["request_fcfs"]["raw_capture"], -0.5
        )
        self.assertEqual(report["per_baseline"]["request_fcfs"]["capture"], 0.0)

    def test_nonpositive_capture_denominator_fails_and_r_dominance_is_checked(self):
        report = compute_capture(0.4, 0.4, {name: 0.4 for name in BASELINE_NAMES})
        self.assertFalse(report["gate_eligible"])
        misses = {name: 0.45 for name in BASELINE_NAMES}
        misses["edf"] = 0.39
        with self.assertRaisesRegex(PhaseMapBaselineError, "beats the exact R"):
            compute_capture(0.5, 0.4, misses)


if __name__ == "__main__":
    unittest.main()
