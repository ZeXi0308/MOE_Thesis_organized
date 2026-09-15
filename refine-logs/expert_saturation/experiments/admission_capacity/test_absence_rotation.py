#!/usr/bin/env python3
"""Check online pair selection and event bookkeeping in CPU fixtures.

These tests do not establish a bound on absence volume, unchanged p50,
executable native recovery, or performance after the action.
"""

import unittest

from absence_rotation import (AbsenceRotation, RotationConfig, RequestView)


def req(rid, computed, prompt=3072, total=4096):
    return RequestView(request_id=rid, num_computed_tokens=computed,
                       num_prompt_tokens=prompt, max_total_tokens=total)


class TestOnlineOnly(unittest.TestCase):
    """The controller must not need anything beyond present state."""

    def test_decide_signature_takes_only_present_state(self):
        """Whitelist, not a blacklist: any new input must be justified here.

        `blocks_needed` is derived from the KV manager's current allocation for
        an already-preempted request, so it is present state. Nothing in this
        set can encode future routing, future arrivals, or completion times.
        """
        import inspect
        params = set(inspect.signature(AbsenceRotation.decide).parameters)
        self.assertEqual(params, {"self", "step", "running", "waiting_ids",
                                  "free_blocks", "blocks_needed"})

    def test_identical_state_gives_identical_decision(self):
        def run():
            r = AbsenceRotation(RotationConfig(min_absence_steps=10,
                                               min_steps_between_swaps=0))
            r.note_preempted(0, ["A"])
            return r.decide(50, [req("B", 3100), req("C", 3500)], ["A"], 0)
        a, b = run(), run()
        self.assertEqual((a.action, a.resume_id, a.victim_id),
                         (b.action, b.resume_id, b.victim_id))

    def test_no_rotation_before_the_absence_exists(self):
        r = AbsenceRotation(RotationConfig(min_absence_steps=10))
        d = r.decide(5, [req("B", 3100)], ["A"], 0)
        self.assertEqual(d.action, "noop")
        self.assertEqual(d.reason, "no tracked absentee waiting")


class TestProposalPairing(unittest.TestCase):
    """Rotation redistributes absence; it must never shrink the running set."""

    def test_every_rotation_pairs_one_resume_with_one_victim(self):
        r = AbsenceRotation(RotationConfig(min_absence_steps=10,
                                           min_steps_between_swaps=0))
        r.note_preempted(0, ["A"])
        d = r.decide(40, [req("B", 3100), req("C", 3500)], ["A"], 0)
        self.assertEqual(d.action, "rotate")
        self.assertIsNotNone(d.resume_id)
        self.assertIsNotNone(d.victim_id)
        self.assertNotEqual(d.resume_id, d.victim_id)

    def test_resumer_and_victim_are_never_the_same_request(self):
        r = AbsenceRotation(RotationConfig(min_absence_steps=1,
                                           min_steps_between_swaps=0))
        r.note_preempted(0, ["A"])
        d = r.decide(10, [req("A", 3100)], ["A"], 0)
        self.assertEqual(d.action, "noop")
        self.assertEqual(d.reason, "victim is the resumer")


class TestVictimChoice(unittest.TestCase):
    def test_first_output_choice_consumed_only_by_applied_exchange(self):
        tracker = AbsenceRotation(RotationConfig(min_absence_steps=1,
            min_steps_between_swaps=0), victim_order="first_most_then_least")
        tracker.note_preempted(0, ["waiting"])
        rows = [RequestView("low", 3100, 3072, 4096, 29),
                RequestView("high", 3500, 3072, 4096, 429)]
        self.assertEqual(tracker.decide(40, rows, ["waiting"], 0).victim_id, "high")
        # A refused proposal, natural preemption, and resumption do not spend it.
        tracker.note_preempted(41, ["natural"])
        tracker.note_resumed(42, ["natural"])
        self.assertEqual(tracker.decide(50, rows, ["waiting"], 0).victim_id, "high")
        self.assertEqual(tracker.applied_rotations, 0)
        tracker.note_rotation_applied()
        self.assertEqual(tracker.decide(60, rows, ["waiting"], 0).victim_id, "low")
        tracker.note_rotation_applied()
        self.assertEqual(tracker.decide(80, rows, ["waiting"], 0).victim_id, "low")

    def test_output_order_uses_observed_service_not_computed_or_fraction(self):
        cfg = RotationConfig(min_absence_steps=1)
        rows = [RequestView("low", 3100, 3072, 4096, 29),
                RequestView("high", 3500, 3072, 4096, 429)]
        for order, expected in (("least_progress", "low"), ("most_output", "high")):
            tracker = AbsenceRotation(cfg, victim_order=order)
            tracker.note_preempted(0, ["waiting"])
            decision = tracker.decide(40, rows, ["waiting"], 0, {"waiting": 256})
            self.assertEqual((decision.resume_id, decision.victim_id), ("waiting", expected))
        # Unequal output limits distinguish absolute service from progress.
        tracker = AbsenceRotation(cfg, victim_order="most_output")
        tracker.note_preempted(0, ["waiting"])
        rows = [RequestView("fraction", 3572, 3072, 4096, 501),
                RequestView("service", 3772, 3072, 7168, 701)]
        self.assertEqual(tracker.decide(40, rows, ["waiting"], 0).victim_id, "service")

    def test_output_order_keeps_near_completion_residency_and_absence_guards(self):
        tracker = AbsenceRotation(victim_order="most_output")
        tracker.note_preempted(0, ["waiting"])
        tracker.resident_since["just_resumed"] = 39
        tracker.absence_count["at_limit"] = 8
        rows = [RequestView("nearly_done", 4050, 3072, 4096, 979),
                RequestView("just_resumed", 3950, 3072, 4096, 879),
                RequestView("at_limit", 3850, 3072, 4096, 779),
                RequestView("eligible", 3150, 3072, 4096, 79)]
        self.assertEqual(tracker.decide(40, rows, ["waiting"], 0).victim_id, "eligible")

    def test_output_order_ties_and_missing_observation(self):
        tracker = AbsenceRotation(victim_order="most_output")
        tracker.note_preempted(0, ["waiting"])
        rows = [RequestView(rid, 3200, 3072, 4096, 129) for rid in ("b", "a")]
        self.assertEqual(tracker.decide(40, rows, ["waiting"], 0).victim_id, "a")
        tracker = AbsenceRotation(victim_order="most_output")
        tracker.note_preempted(0, ["waiting"])
        with self.assertRaisesRegex(ValueError, "observed output"):
            tracker.decide(40, [req("unknown", 3200)], ["waiting"], 0)
        with self.assertRaisesRegex(ValueError, "unsupported"):
            AbsenceRotation(victim_order="unrecognized")

    def test_least_progressed_is_chosen(self):
        r = AbsenceRotation(RotationConfig(min_absence_steps=10,
                                           min_steps_between_swaps=0))
        r.note_preempted(0, ["A"])
        d = r.decide(40, [req("hi", 4000), req("lo", 3100), req("mid", 3500)],
                     ["A"], 0)
        self.assertEqual(d.victim_id, "lo")

    def test_nearly_finished_request_is_protected(self):
        """Its KV frees soon anyway; evicting it discards the most progress."""
        cfg = RotationConfig(min_absence_steps=10, min_steps_between_swaps=0,
                             protect_progress_fraction=0.90)
        r = AbsenceRotation(cfg)
        r.note_preempted(0, ["A"])
        # 3072 + 0.95*1024 computed -> progress 0.95, above the guard.
        d = r.decide(40, [req("almost", 3072 + 973)], ["A"], 0)
        self.assertEqual(d.action, "noop")
        self.assertEqual(d.reason, "no eligible victim")

    def test_repeatedly_evicted_request_becomes_ineligible(self):
        cfg = RotationConfig(min_absence_steps=10, min_steps_between_swaps=0,
                             max_absences_per_request=2)
        r = AbsenceRotation(cfg)
        r.note_preempted(0, ["A"])
        r.absence_count["victim"] = 2
        d = r.decide(40, [req("victim", 3100)], ["A"], 0)
        self.assertEqual(d.reason, "no eligible victim")


class TestCostGuards(unittest.TestCase):
    def test_no_swap_while_pool_has_room(self):
        """The engine resumes on its own; a forced swap would be pure loss."""
        r = AbsenceRotation(RotationConfig(min_absence_steps=1,
                                           min_steps_between_swaps=0,
                                           free_block_slack=0))
        r.note_preempted(0, ["A"])
        d = r.decide(40, [req("B", 3100)], ["A"], free_blocks=500)
        self.assertEqual(d.action, "noop")
        self.assertEqual(d.reason, "pool can already resume the victim")

    def test_cooldown_bounds_recompute_cost(self):
        cfg = RotationConfig(min_absence_steps=1, min_steps_between_swaps=20)
        r = AbsenceRotation(cfg)
        r.note_preempted(0, ["A", "B"])
        first = r.decide(30, [req("X", 3100), req("Y", 3200)], ["A", "B"], 0)
        self.assertEqual(first.action, "rotate")
        second = r.decide(35, [req("X", 3100), req("Y", 3200)], ["A", "B"], 0)
        self.assertEqual(second.action, "noop")
        self.assertEqual(second.reason, "swap cooldown")
        third = r.decide(60, [req("X", 3100), req("Y", 3200)], ["A", "B"], 0)
        self.assertEqual(third.action, "rotate")

    def test_absence_threshold_blocks_cheap_churn(self):
        r = AbsenceRotation(RotationConfig(min_absence_steps=30,
                                           min_steps_between_swaps=0))
        r.note_preempted(0, ["A"])
        d = r.decide(10, [req("B", 3100)], ["A"], 0)
        self.assertEqual(d.action, "noop")
        self.assertIn("below threshold", d.reason)


class TestLongestWaitingWins(unittest.TestCase):
    def test_oldest_absentee_is_resumed(self):
        r = AbsenceRotation(RotationConfig(min_absence_steps=10,
                                           min_steps_between_swaps=0))
        r.note_preempted(5, ["late"])
        r.note_preempted(1, ["early"])
        d = r.decide(60, [req("B", 3100)], ["late", "early"], 0)
        self.assertEqual(d.resume_id, "early")

    def test_resume_clears_the_absence_record(self):
        r = AbsenceRotation(RotationConfig())
        r.note_preempted(5, ["A"])
        self.assertIn("A", r.absent_since)
        r.note_resumed(40, ["A"])
        self.assertNotIn("A", r.absent_since)
        # The count is history and must survive, so repeat victims are capped.
        self.assertEqual(r.absence_count["A"], 1)


class TestReplayAgainstSealedDynamics(unittest.TestCase):
    """Replay the measured preemption timeline; check the controller would act.

    Sealed facts: preemption at step 809 (victim A) and 931 (victim B); A was
    only resumed at step 1030, 221 steps later, after a COMPLETION. This is a
    decision-level check, not a performance claim: no KV, queue or completion
    dynamics are simulated here.
    """

    def test_controller_would_have_acted_long_before_step_1030(self):
        cfg = RotationConfig(min_absence_steps=30, min_steps_between_swaps=20)
        r = AbsenceRotation(cfg)
        r.note_preempted(809, ["A"])
        survivors = [req(f"s{i}", 3780 + i) for i in range(31)]
        first_rotation = None
        for step in range(810, 1031):
            if step == 931:
                r.note_preempted(931, ["B"])
            waiting = [x for x in ("A", "B") if x in r.absent_since]
            d = r.decide(step, survivors, waiting, free_blocks=0)
            if d.action == "rotate" and first_rotation is None:
                first_rotation = d
        self.assertIsNotNone(first_rotation, "controller never acted")
        self.assertEqual(first_rotation.resume_id, "A")
        self.assertLessEqual(first_rotation.step, 809 + 30 + 1)
        # The measured absence was 221 steps; acting at ~839 is far earlier.
        self.assertLess(first_rotation.step, 1030)

    def test_rotation_count_stays_bounded_over_the_binding_window(self):
        cfg = RotationConfig(min_absence_steps=30, min_steps_between_swaps=20)
        r = AbsenceRotation(cfg)
        r.note_preempted(809, ["A"])
        survivors = [req(f"s{i}", 3780 + i) for i in range(31)]
        for step in range(810, 1031):
            waiting = ["A"] if "A" in r.absent_since else []
            r.decide(step, survivors, waiting, free_blocks=0)
        s = r.summary()
        # 221 steps at a 20-step cooldown bounds swaps at 12; each costs one
        # recompute, so the cost is knowable in advance rather than open-ended.
        self.assertLessEqual(s["n_rotations"], 12)
        self.assertGreater(s["n_rotations"], 0)




class TestReviewFixes(unittest.TestCase):
    """Three defects found by self-review, each of which would bias a run.

    B  the pool check compared free blocks against zero, so the controller was
       silent at exactly the measured moment it exists for (step 931: 245
       blocks free, victim needed 237, victim still waited 94 more steps);
    C  rotations did not increment `absence_count`, so the repeat-victim cap
       was unreachable and one request could be rotated without limit;
    A  nothing stopped a just-resumed request from being evicted again.
    """

    def test_B_pool_check_uses_the_victims_requirement(self):
        cfg = RotationConfig(min_absence_steps=30, min_steps_between_swaps=20)
        r = AbsenceRotation(cfg)
        r.note_preempted(809, ["A"])
        survivors = [req(f"s{i}", 3800 + i) for i in range(31)]
        # 245 free but A needs 237 -> engine could resume it; stay silent.
        d = r.decide(930, survivors, ["A"], free_blocks=245,
                     blocks_needed={"A": 237})
        self.assertEqual(d.action, "noop")
        self.assertEqual(d.reason, "pool can already resume the victim")
        # 200 free and A needs 237 -> engine cannot; this is the real case.
        r.last_swap_step = -10 ** 9
        d = r.decide(930, survivors, ["A"], free_blocks=200,
                     blocks_needed={"A": 237})
        self.assertEqual(d.action, "rotate")

    def test_exact_recovery_requirement_does_not_add_an_eviction(self):
        r = AbsenceRotation(RotationConfig(min_absence_steps=1))
        r.note_preempted(0, ["A"])
        d = r.decide(50, [req("V", 3800)], ["A"], free_blocks=237,
                     blocks_needed={"A": 237})
        self.assertEqual(d.action, "noop")
        self.assertEqual(d.reason, "pool can already resume the victim")

    def test_B_without_requirements_falls_back_to_slack(self):
        """Old behaviour is preserved when the caller cannot supply blocks."""
        cfg = RotationConfig(min_absence_steps=30, min_steps_between_swaps=20)
        r = AbsenceRotation(cfg)
        r.note_preempted(809, ["A"])
        d = r.decide(900, [req("s", 3800)], ["A"], free_blocks=1)
        self.assertEqual(d.action, "noop")

    def test_C_rotation_counts_toward_the_repeat_cap(self):
        cfg = RotationConfig(min_absence_steps=1, min_steps_between_swaps=0,
                             min_residency_steps=0, max_absences_per_request=2)
        r = AbsenceRotation(cfg)
        r.note_preempted(0, ["A"])
        # Drive three swaps against a single eligible victim, feeding the
        # controller's own decisions back as the engine would.
        for i in range(3):
            d = r.decide(10 + i, [req("V", 3100)], ["A"], 0)
            if d.action == "rotate":
                r.note_preempted(10 + i, [d.victim_id])
                r.note_resumed(10 + i, [d.resume_id])
                r.absent_since["A"] = 0        # keep A waiting for the next round
        self.assertEqual(r.absence_count.get("V"), 2,
                         "rotations must count toward the cap")
        last = r.decide(40, [req("V", 3100)], ["A"], 0)
        self.assertEqual(last.reason, "no eligible victim")

    def test_A_just_resumed_request_is_not_evicted_again(self):
        cfg = RotationConfig(min_absence_steps=1, min_steps_between_swaps=0,
                             min_residency_steps=30)
        r = AbsenceRotation(cfg)
        r.note_preempted(0, ["A"])
        r.note_resumed(50, ["A"])              # A is back, resident since 50
        r.note_preempted(50, ["B"])
        d = r.decide(60, [req("A", 3100)], ["B"], 0)
        self.assertEqual(d.action, "noop")
        self.assertEqual(d.reason, "no eligible victim")
        # After the residency window it becomes eligible again.
        d = r.decide(90, [req("A", 3100)], ["B"], 0)
        self.assertEqual(d.action, "rotate")
        self.assertEqual(d.victim_id, "A")

    def test_rotation_count_in_fixed_waiting_fixture(self):
        """Count proposed swaps in this fixture; this does not prove p50 safety."""
        cfg = RotationConfig(min_absence_steps=30, min_steps_between_swaps=20)
        r = AbsenceRotation(cfg)
        r.note_preempted(809, ["A"])
        survivors = [req(f"s{i}", 3800 + i) for i in range(31)]
        for step in range(810, 1031):
            waiting = ["A"] if "A" in r.absent_since else []
            d = r.decide(step, survivors, waiting, free_blocks=0,
                         blocks_needed={"A": 237})
            if d.action == "rotate":
                r.note_resumed(step, [d.resume_id])
                r.note_preempted(step, [d.victim_id])
                r.absent_since["A"] = step
        self.assertLessEqual(r.summary()["n_rotations"], 16,
                             "proposal count exceeded this fixed fixture bound")


if __name__ == "__main__":
    unittest.main()
