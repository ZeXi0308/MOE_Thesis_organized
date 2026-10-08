import unittest

from restore_obligation import RequestView, RestoreObligations

def view(status="PREEMPTED", output=4, pending=20):
    return RequestView(status, output, pending)

def apply(ledger, step, before, resumed, *, enabled=True, preempted=(), opened=False):
    if not opened:
        ledger.begin_schedule(step, before)
    original = {rid: -1 for rid in before}
    effective = ledger.effective_priorities(original, enabled=enabled)
    return ledger.after_schedule(step, before, resumed,
        {rid: 8 for rid in resumed}, preempted, {rid: 0 for rid in before}, 0,
        enabled=enabled, effective_priorities=effective)

class RestoreObligationTest(unittest.TestCase):
    def test_recompute_does_not_release(self):
        ledger = RestoreObligations()
        self.assertEqual(apply(ledger, 0, {"r": view()}, ["r"])["started"], ["r"])
        phase = ledger.begin_schedule(1, {"r": view("RUNNING", 4, 12)})
        self.assertEqual(phase, {"outstanding": ["r"], "released": []})
        self.assertEqual(ledger.effective_priorities({"r": 0}, enabled=True), {"r": -2})
        apply(ledger, 1, {"r": view("RUNNING", 4, 12)}, [], opened=True)
    def test_output_from_prior_call_releases_at_next_begin(self):
        ledger = RestoreObligations()
        apply(ledger, 0, {"r": view()}, ["r"])
        phase = ledger.begin_schedule(1, {"r": view("RUNNING", 5, 1)})
        self.assertEqual(phase["outstanding"], [])
        self.assertEqual(phase["released"][0]["reason"], "new_output")
        apply(ledger, 1, {"r": view("RUNNING", 5, 1)}, [], opened=True)
    def test_multiple_obligations_are_independent(self):
        ledger = RestoreObligations()
        apply(ledger, 0, {"a": view(output=2), "b": view(output=7)}, ["b", "a"])
        phase = ledger.begin_schedule(1, {"a": view("RUNNING", 3, 1), "b": view("RUNNING", 7, 8)})
        self.assertEqual(phase["outstanding"], ["b"])
        self.assertEqual(phase["released"][0]["request_id"], "a")
        apply(ledger, 1, {"a": view("RUNNING", 3, 1), "b": view("RUNNING", 7, 8)}, [], opened=True)
    def test_completion_and_unexplained_disappearance(self):
        ledger = RestoreObligations()
        apply(ledger, 0, {"r": view()}, ["r"])
        self.assertEqual(ledger.finalize(1, {}, {"r"})[0]["reason"], "completed")
        other = RestoreObligations()
        apply(other, 0, {"r": view()}, ["r"])
        with self.assertRaisesRegex(RuntimeError, "disappeared"):
            other.finalize(1, {})
    def test_initial_prefill_and_ordinary_decode_do_not_start(self):
        ledger = RestoreObligations()
        before = {"prefill": view(output=0, pending=100), "decode": view(output=3, pending=1)}
        self.assertEqual(apply(ledger, 0, before, before)["started"], [])
        self.assertEqual(ledger.snapshot()["active"], {})
    def test_interruption_closes_off_and_fails_on(self):
        ledger = RestoreObligations()
        apply(ledger, 0, {"r": view()}, ["r"])
        result = apply(ledger, 1, {"r": view("RUNNING")}, [], enabled=False, preempted={"r"})
        self.assertEqual(result["interrupted"], ["r"])
        self.assertEqual(apply(ledger, 2, {"r": view()}, ["r"])["started"], ["r"])
        ledger.begin_schedule(3, {"r": view("RUNNING")})
        with self.assertRaisesRegex(RuntimeError, "was preempted"):
            ledger.after_schedule(3, {"r": view("RUNNING")}, [], {}, {"r"},
                {"r": 0}, 0, enabled=True, effective_priorities={"r": -2})

if __name__ == "__main__":
    unittest.main()
