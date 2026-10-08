"""CPU lifecycle tests for the independent LTR-style selected-save policy."""

import unittest

from ltr_fair_policy import Intent, LTRFairPolicy, LTRRequestState, Request


def req(rid, status, order, *, arrival=None, output=0, owned=0, need=0,
        victim=False):
    return Request(rid, status, float(order if arrival is None else arrival),
                   order, output, owned, need, victim)


class LTRFairPolicyTests(unittest.TestCase):
    def test_pending_load_rearm_positive_call_quantum_and_first_output(self):
        policy = LTRFairPolicy(threshold=2, quantum=2)
        target = req("t", "PREEMPTED", 0, need=1)
        peer = req("p", "RUNNING", 1, output=9, owned=2, victim=True)
        for _ in range(2):
            policy.observe([target, peer])
            self.assertIsNone(policy.propose(1, 1))
            policy.feedback({"p": 1})
        policy.observe([target, peer])
        intent = policy.propose(1, 1)
        self.assertEqual((intent.action, intent.target_id, intent.victim_id),
                         ("PRIORITIZE_WAITING", "t", None))
        policy.accept(intent)
        self.assertEqual(policy.active_target, "t")
        policy.feedback({"p": 1})
        self.assertEqual(policy.counters.states["t"].quantum_remaining, 2)

        # Loading is pending: two idle scheduler calls re-arm a full Q, rather
        # than spending Q or ending the accepted recovery epoch.
        pending = req("t", "WAITING_FOR_REMOTE_KVS", 0, need=1)
        policy.observe([pending, peer]); policy.feedback({"p": 1})
        policy.observe([pending, peer])
        self.assertEqual(policy.counters.states["t"].quantum_remaining, 2)
        self.assertEqual(policy.counters.states["t"].idle, 0)
        policy.feedback({"p": 1})

        running = req("t", "RUNNING", 0, need=0)
        policy.observe([running, peer])
        policy.feedback({"t": 128, "p": 1})  # positive prefill/recompute call
        self.assertEqual(policy.counters.states["t"].quantum_remaining, 1)
        policy.observe([running, peer])
        self.assertEqual(policy.active_target, "t")
        policy.feedback({"t": 1, "p": 1})  # first new output is still one call
        self.assertEqual(policy.counters.states["t"].quantum_remaining, 0)
        self.assertEqual(policy.active_target, "t")
        policy.observe([running, peer])  # demotion before the next ranking
        self.assertIsNone(policy.active_target)
        self.assertEqual(policy.counters.states["t"].priority, 0)
        policy.feedback({"p": 1})

    def test_running_priority_and_rearm_precedes_quantum_demotion(self):
        policy = LTRFairPolicy(threshold=2, quantum=3)
        older = req("older", "RUNNING", 0)
        later = req("later", "RUNNING", 1)
        policy.counters.states["later"] = LTRRequestState(idle=2, priority=-1,
                                                           quantum_remaining=0)
        policy.observe([older, later])
        self.assertEqual(policy.ordered_running_ids(), ("later", "older"))
        self.assertEqual(policy.counters.states["later"].quantum_remaining, 3)
        policy.feedback({"later": 300})
        self.assertEqual(policy.counters.states["later"].quantum_remaining, 2)
        policy.observe([older, later])
        self.assertEqual(policy.ordered_running_ids(), ("later", "older"))
        policy.feedback({"later": 1})

    def test_first_executable_target_and_backend_rejection_do_not_latch(self):
        policy = LTRFairPolicy(threshold=3, quantum=2)
        old = req("old", "PREEMPTED", 0, need=8)
        later = req("later", "PREEMPTED", 1, need=3)
        rich = req("rich", "RUNNING", 2, output=20, owned=3, victim=True)
        small = req("small", "RUNNING", 3, output=5, owned=3, victim=True)
        policy.counters.states = {"old": LTRRequestState(idle=3),
                                  "later": LTRRequestState(idle=3),
                                  "rich": LTRRequestState(),
                                  "small": LTRRequestState()}
        policy.observe([old, later, rich, small])
        first = policy.propose(0, 0)
        self.assertEqual((first.target_id, first.victim_id), ("later", "rich"))
        self.assertIsNone(policy.active_target)
        self.assertEqual(policy.skipped[0]["target"], "old")
        policy.reject(first, "BACKEND_PREPARE_REJECTED")
        second = policy.propose(0, 0, excluded_victim_ids={"rich"})
        self.assertEqual((second.target_id, second.victim_id), ("later", "small"))
        with self.assertRaises(ValueError):
            policy.accept(first)
        policy.accept(second)
        self.assertEqual(policy.active_target, "later")
        policy.feedback({"rich": 1, "small": 1}, preempted={"small"},
                        forced_preempted={"small"})
        self.assertEqual(policy.active_target, "later")  # forced victim != target
        self.assertEqual(policy.last_forced_preempted, ("small",))
        self.assertEqual(policy.last_natural_preempted, ())
        policy.observe([old, later, rich])
        self.assertIsNone(policy.propose(0, 0))  # active latch prevents another
        policy.feedback({"rich": 1})

    def test_free_capacity_slot_and_single_victim_censoring(self):
        target = req("t", "PREEMPTED", 0, need=4)
        a = req("a", "RUNNING", 1, output=8, owned=2, victim=True)
        b = req("b", "RUNNING", 2, output=7, owned=2, victim=True)
        policy = LTRFairPolicy(threshold=1, quantum=1)
        policy.counters.states["t"] = LTRRequestState(idle=1)
        policy.observe([target, a, b])
        self.assertIsNone(policy.propose(0, 0))
        self.assertEqual(policy.last_noop_reason, "UNFUNDED_SINGLE_VICTIM")
        self.assertTrue(policy.skipped[-1]["aggregate_fundable"])
        self.assertFalse(policy.skipped[-1]["single_victim_fundable"])
        self.assertEqual(policy.censor_counts["UNFUNDED_SINGLE_VICTIM"], 1)
        self.assertIsNone(policy.active_target)
        policy.feedback({})

        slot = LTRFairPolicy(threshold=1, quantum=1)
        slot.counters.states["t"] = LTRRequestState(idle=1)
        slot.observe([target])
        self.assertIsNone(slot.propose(4, 0))
        self.assertEqual(slot.last_noop_reason, "SLOT_BLOCKED")
        slot.feedback({})

    def test_equal_boosted_priority_uses_the_declared_tie_order(self):
        policy = LTRFairPolicy(threshold=1, quantum=2)
        target = req("early", "PREEMPTED", 0, need=2)
        later = req("late", "RUNNING", 1, output=7, owned=2, victim=True)
        policy.counters.states = {"early": LTRRequestState(idle=1),
                                  "late": LTRRequestState(idle=1)}
        policy.observe([target, later])
        intent = policy.propose(0, 0)
        self.assertEqual((intent.target_id, intent.victim_id), ("early", "late"))
        policy.feedback({})

        reverse = LTRFairPolicy(threshold=1, quantum=2)
        older = req("older", "RUNNING", 0, output=7, owned=2, victim=True)
        target = req("later", "PREEMPTED", 1, need=2)
        reverse.counters.states = {"older": LTRRequestState(idle=1),
                                   "later": LTRRequestState(idle=1)}
        reverse.observe([older, target])
        self.assertIsNone(reverse.propose(0, 0))
        self.assertEqual(reverse.last_noop_reason, "NO_ELIGIBLE_VICTIM")
        reverse.feedback({})
        slot.observe([target])
        self.assertEqual(slot.propose(4, 1).action, "PRIORITIZE_WAITING")
        slot.feedback({})

    def test_natural_preemption_terminal_and_status_expiry_release_safely(self):
        policy = LTRFairPolicy(threshold=1, quantum=3)
        target = req("t", "PREEMPTED", 0, need=0)
        policy.counters.states["t"] = LTRRequestState(idle=1)
        policy.observe([target])
        intent = policy.propose(0, 1)
        policy.accept(intent)
        policy.feedback({})
        policy.observe([req("t", "RUNNING", 0)])
        policy.feedback({"t": 1}, preempted={"t"})
        self.assertIsNone(policy.active_target)
        self.assertEqual(policy.last_natural_preempted, ("t",))
        self.assertIn("t", policy.counters.states)
        self.assertEqual(policy.counters.states["t"].quantum_remaining, 2)

        policy.observe([target])
        self.assertEqual(policy.propose(0, 1).quantum_remaining, 2)
        self.assertIsNone(policy.active_target)  # natural preempt released latch
        policy.feedback({}, terminal={"t"})
        self.assertNotIn("t", policy.counters.states)
        policy.observe([req("u", "WAITING", 0)])
        self.assertIsNone(policy.propose(0, 1))
        self.assertEqual(policy.last_noop_reason, "NO_BOOSTED_PREEMPTED")
        policy.feedback({})

        stale = LTRFairPolicy(threshold=1, quantum=2)
        stale.counters.states["t"] = LTRRequestState(idle=1)
        stale.observe([target])
        stale.accept(stale.propose(0, 1))
        stale.feedback({})
        stale.observe([req("t", "CANCELLED", 0)])
        self.assertIsNone(stale.active_target)
        self.assertNotIn("t", stale.counters.states)
        stale.feedback({})

    def test_no_op_and_invalid_call_order(self):
        policy = LTRFairPolicy(threshold=2, quantum=2)
        with self.assertRaises(RuntimeError):
            policy.propose(0, 0)
        policy.observe([req("w", "WAITING", 0)])
        self.assertIsNone(policy.propose(0, 1))
        self.assertEqual(policy.no_op("BACKEND_CENSORED"),
                         Intent("NO_OP", None, reason="BACKEND_CENSORED"))
        with self.assertRaises(RuntimeError):
            policy.observe([])
        policy.feedback({})
        self.assertEqual(policy.counters.states["w"].idle, 1)
        policy.observe([])
        self.assertEqual(policy.counters.states, {})
        policy.feedback({})


if __name__ == "__main__":
    unittest.main()
