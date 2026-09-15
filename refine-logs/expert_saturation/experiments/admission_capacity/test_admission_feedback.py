"""Only causal cutoff and nonpreemptive drain risks for the pilot helper."""
from types import SimpleNamespace as NS
import unittest

from admission_feedback import AdmissionFeedback, apply_nonpreemptive_limit


class AdmissionFeedbackTest(unittest.TestCase):
    def test_completed_history_cutoff_cooldown_and_shadow(self):
        config = dict(policy="feedback", cap=32, tpot_slo_s=0.01)
        feedback = AdmissionFeedback(config, 32)
        shadow = AdmissionFeedback(dict(config, policy="shadow"), 32)
        for step in range(1, 5):
            for controller in (feedback, shadow):
                before = controller.decide(decision_start_s=step - 0.1, active=32, waiting=4)
                self.assertEqual(before["intent_target"], 32)
                controller.observe({"a": 0.011, "b": 0.013}, received_s=step,
                                   available_s=step + 0.01)
        # A timestamp before the fourth observation becomes available cannot use it.
        with self.assertRaisesRegex(ValueError, "future observation"):
            feedback.decide(decision_start_s=4.005, active=32, waiting=4)
        action = feedback.decide(decision_start_s=4.02, active=32, waiting=4)
        sham = shadow.decide(decision_start_s=4.02, active=32, waiting=4)
        self.assertEqual(action["window_completed_steps"], [1, 2, 3, 4])
        self.assertEqual((action["target_cap"], sham["target_cap"]), (16, 32))
        self.assertEqual(action["intent_target"], sham["intent_target"])
        self.assertLessEqual(action["signal_available_s"], action["decision_start_s"])
        for step in range(5, 8):
            feedback.observe({"a": 0.012}, received_s=step, available_s=step)
            self.assertEqual(feedback.decide(decision_start_s=step, active=32, waiting=4)["target_cap"], 16)

    def test_down_clamp_preserves_running_and_only_blocks_new_admission(self):
        running = [NS(request_id=str(i)) for i in range(32)]
        waiting = [NS(request_id="waiting")]
        scheduler = NS(running=running, waiting=waiting, max_num_running_reqs=32)
        original = tuple(running)
        self.assertEqual(apply_nonpreemptive_limit(scheduler, 16, 32), 32)
        self.assertIs(scheduler.running, running)
        self.assertEqual(tuple(running), original)
        # Native scheduler first traverses the entire running list, then admits
        # waiting only below max_num_running_reqs, and finally asserts this bound.
        scheduled = [r.request_id for r in scheduler.running]
        if len(running) < scheduler.max_num_running_reqs:
            running.append(waiting.pop(0))
        self.assertEqual(scheduled, [str(i) for i in range(32)])
        self.assertEqual(len(waiting), 1)
        del running[:17]  # Natural completions: now one slot is free under target16.
        self.assertEqual(apply_nonpreemptive_limit(scheduler, 16, 32), 16)
        if len(running) < scheduler.max_num_running_reqs:
            running.append(waiting.pop(0))
        self.assertEqual((len(running), len(waiting)), (16, 0))
        self.assertEqual([r.request_id for r in running[:-1]], [str(i) for i in range(17, 32)])

    def test_single_action_cutoff_latch_and_matched_shadow(self):
        config = dict(policy="single_down", cap=32, tpot_slo_s=0.01,
                      feedback_caps=[8, 12, 16, 32], single_down_target=16)
        feedback = AdmissionFeedback(config, 32)
        shadow = AdmissionFeedback(dict(config, policy="single_shadow"), 32)
        # Initial headroom must not cause an up action; four high observations
        # then trigger exactly one descent, regardless of subsequent signals.
        for step, itl in enumerate([0.005] * 4 + [0.012] * 12 + [0.005] * 12, 1):
            for controller in (feedback, shadow):
                controller.observe({"a": itl}, received_s=step, available_s=step + 0.01)
                with self.assertRaisesRegex(ValueError, "future observation"):
                    controller.decide(decision_start_s=step + 0.005, active=32, waiting=4)
            action = feedback.decide(decision_start_s=step + 0.02, active=32, waiting=4)
            sham = shadow.decide(decision_start_s=step + 0.02, active=32, waiting=4)
            for key in ("window_completed_steps", "recent_step_median_itl_s", "cooldown_steps",
                        "intent_before", "intent_target", "intent_changed", "reason"):
                self.assertEqual(action[key], sham[key], key)
            self.assertLessEqual(action["signal_available_s"], action["decision_start_s"])
            self.assertEqual(sham["target_cap"], 32)
            self.assertFalse(sham["applied_change"])
            if step <= 4:
                self.assertEqual(action["target_cap"], 32)
                self.assertFalse(action["intent_changed"])
            if feedback.actions:
                self.assertEqual(action["target_cap"], 16)
        self.assertEqual((len(feedback.actions), len(shadow.actions)), (1, 1))
        self.assertEqual(feedback.actions[0]["window_completed_steps"], [4, 5, 6, 7])
        self.assertTrue(feedback.actions[0]["applied_change"])
        self.assertEqual(feedback.decisions[-1]["reason"], "single_action_latched")

    def test_single_action_explicit_target_cannot_raise_or_escape_caps(self):
        config = dict(policy="single_down", cap=32, tpot_slo_s=0.01)
        for target in (32, 64, 0, True, 15):
            with self.subTest(target=target), self.assertRaisesRegex(ValueError, "single_down_target"):
                AdmissionFeedback(dict(config, single_down_target=target), 32)
        # An initial cap below engine capacity still cannot climb before firing.
        controller = AdmissionFeedback(dict(config, cap=16, single_down_target=12), 32)
        for step in range(1, 5):
            controller.observe({"a": 0.005}, received_s=step, available_s=step)
        row = controller.decide(decision_start_s=4, active=16, waiting=4)
        self.assertEqual(row["intent_target"], 16)
        self.assertEqual(controller.actions, [])


if __name__ == "__main__":
    unittest.main()
