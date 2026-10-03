"""CPU regression for the conservative existing-model LTR admission candidate."""
from pathlib import Path
import inspect
import sys
import unittest

sys.dont_write_bytecode = True
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "candidate_g64_perf_r01" / "pkg"))
sys.path.insert(0, str(BASE))

import ltr_style_native_inflight_guard_candidate as candidate
from ltr_style_native_inflight_guard_candidate import (
    guard_waiting_intent, waiting_admission_reason,
)
from ltr_style_selected import LTRStyleSelected, Request
from staged_save_contract import RequestState


class InflightAdmissionGuardTest(unittest.TestCase):
    def setUp(self):
        self.target = RequestState("target", computed=0, prompt=80, output=16,
                                   max_output=1024, status="PREEMPTED", blocks=())
        self.assertEqual(self.target.remaining_blocks, 6)
        self.rows = [Request("target", arrival=0.0, status="PREEMPTED",
                             remaining_blocks=6, output_tokens=16),
                     Request("inflight-prefill", arrival=1.0, status="RUNNING",
                             remaining_blocks=1)]

    def boosted_selector(self):
        selector = LTRStyleSelected(threshold=30, quantum=1)
        for _ in range(30):
            intent = selector.begin_step(self.rows, 6, free_slots=31)
            self.assertEqual(intent.action, "NOOP")
            selector.after_step({})
        return selector

    def test_original_six_six_one_counterexample_defers_without_latch(self):
        selector = self.boosted_selector()
        check = lambda _: waiting_admission_reason(self.target, 6, 1, 31)
        intent = selector.begin_step(self.rows, 6, free_slots=31, executable=check)
        self.assertEqual(intent.action, "DEFER")
        self.assertIsNone(selector.active_target)
        self.assertIn("inflight reservation", selector.skipped[0]["reason"])
        # No target latch means begin() cannot set _rotation_target or reserve/hold.
        self.assertFalse(selector.active_target is not None)
        selector.after_step({})
        for _ in range(3):
            intent = selector.begin_step(self.rows, 6, free_slots=31, executable=check)
            self.assertEqual(intent.action, "DEFER")
            self.assertIsNone(selector.active_target)
            selector.after_step({})

    def test_one_more_free_block_accepts(self):
        self.assertIsNone(waiting_admission_reason(self.target, 7, 1, 31))
        selector = self.boosted_selector()
        intent = selector.begin_step(
            self.rows, 7, free_slots=31,
            executable=lambda _: waiting_admission_reason(self.target, 7, 1, 31))
        self.assertEqual(intent.action, "PRIORITIZE_WAITING")
        selector.accept(intent)
        self.assertEqual(selector.active_target, "target")
        selector.after_step({})

    def test_no_inflight_reservation_accepts_exact_fit(self):
        self.assertIsNone(waiting_admission_reason(self.target, 6, 0, 31))

    def test_observed_diagnostic_resource_numbers_are_rejected(self):
        # The later native diagnostic recorded F=86, N=86, R=132. This test
        # checks only the candidate guard's decision for those numbers.
        target = RequestState("observed-target", computed=0, prompt=1376, output=0,
                              max_output=1024, status="PREEMPTED", blocks=())
        self.assertEqual(target.remaining_blocks, 86)
        self.assertIsNotNone(waiting_admission_reason(target, 86, 132, 31))
        self.assertIsNone(waiting_admission_reason(target, 218, 132, 31))

    def test_active_preempted_target_rechecked_and_released(self):
        selector = self.boosted_selector()
        first = selector.begin_step(self.rows, 6, free_slots=31,
                                    executable=lambda _: None)
        selector.accept(first)
        selector.after_step({})  # No native tokens; Q1 remains active.
        active_intent = selector.begin_step(self.rows, 6, free_slots=31)
        self.assertEqual(active_intent.action, "PRIORITIZE_WAITING")
        self.assertEqual(selector.active_target, "target")
        releases = []

        def release(reason):
            releases.append(reason)
            selector.release_active()

        deferred = guard_waiting_intent(
            active_intent, selector,
            lambda _: waiting_admission_reason(self.target, 6, 1, 31), release)
        self.assertEqual(deferred.action, "DEFER")
        self.assertEqual(len(releases), 1)
        self.assertIsNone(selector.active_target)
        self.assertEqual(selector.counters.states["target"].quantum_remaining, 1)
        selector.after_step({})

    def test_resource_release_is_not_relabelled_quantum_expired(self):
        # Integration ordering matters: selector's own expiry event is emitted
        # before the new resource guard can release an active target.
        source = inspect.getsource(candidate.install)
        selector_begin = source.index("intent=selector.begin_step(")
        selector_expiry_event = source.index("reason='terminal' if prior not in scheduler.requests else 'quantum_expired'", selector_begin)
        guard_recheck = source.index("intent=guard_waiting_intent(intent,selector,executable,release)", selector_begin)
        self.assertLess(selector_expiry_event, guard_recheck)

        selector = self.boosted_selector()
        first = selector.begin_step(self.rows, 6, free_slots=31, executable=lambda _: None)
        selector.accept(first)
        selector.after_step({})
        active_intent = selector.begin_step(self.rows, 6, free_slots=31)
        self.assertIsNotNone(selector.active_target)  # No selector expiry event.
        logged_reasons = []
        def release(reason):
            logged_reasons.append(reason)
            selector.release_active()
        deferred = guard_waiting_intent(
            active_intent, selector,
            lambda _: waiting_admission_reason(self.target, 6, 1, 31), release)
        self.assertEqual(deferred.action, "DEFER")
        self.assertEqual(logged_reasons,
                         ["target plus native inflight reservation exceeds free KV blocks"])
        self.assertNotIn("quantum_expired", logged_reasons)
        selector.after_step({})

    def test_active_target_with_enough_resources_stays_active(self):
        selector = self.boosted_selector()
        first = selector.begin_step(self.rows, 6, free_slots=31,
                                    executable=lambda _: None)
        selector.accept(first)
        selector.after_step({})
        active_intent = selector.begin_step(self.rows, 6, free_slots=31)
        releases = []
        retained = guard_waiting_intent(active_intent, selector,
                                         lambda _: waiting_admission_reason(self.target, 7, 1, 31),
                                         releases.append)
        self.assertEqual(retained.action, "PRIORITIZE_WAITING")
        self.assertEqual(selector.active_target, "target")
        self.assertFalse(releases)
        selector.after_step({})

    def test_unknown_or_failed_native_reads_defer(self):
        for free, reserve, slots in ((None, 0, 31), (6, None, 31),
                                     (True, 0, 31), (6, -1, 31), (6, 0, 0)):
            with self.subTest(free=free, reserve=reserve, slots=slots):
                self.assertIsNotNone(waiting_admission_reason(
                    self.target, free, reserve, slots))
        self.assertIsNotNone(waiting_admission_reason(
            self.target, 6, 0, 31, target_inflight=True))

        class BrokenTarget:
            status = "PREEMPTED"
            @property
            def remaining_blocks(self):
                raise RuntimeError("unreadable allocation state")

        self.assertIn("unavailable", waiting_admission_reason(
            BrokenTarget(), 6, 0, 31))
        selector = self.boosted_selector()
        intent = selector.begin_step(self.rows, 6, free_slots=31,
                                     executable=lambda _: None)
        selector.accept(intent)
        selector.after_step({})
        active_intent = selector.begin_step(self.rows, 6, free_slots=31)
        def broken_check(_):
            raise RuntimeError("native reservation probe failed")
        deferred = guard_waiting_intent(active_intent, selector, broken_check,
                                         lambda _: selector.release_active())
        self.assertEqual(deferred.action, "DEFER")
        self.assertIsNone(selector.active_target)
        selector.after_step({})


if __name__ == "__main__":
    unittest.main()
