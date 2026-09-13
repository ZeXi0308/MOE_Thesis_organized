import unittest
from paging_cost_model import (RequestState, apply_event, admission_slots,
    layer_transfer_bytes, independent_expected_bytes, uniform_union, step_time_s)


class AccountingTests(unittest.TestCase):
    def test_cap_lowering_preserves_existing_work_and_future_is_not_waiting(self):
        active = RequestState('a', 32, 40, 9, 32, 'decode')
        queued = RequestState('b', 32, 0, 0, 32, 'waiting', 1.0)
        future = RequestState('c', 32, 0, 0, 32, 'waiting', 3.0)
        self.assertEqual(admission_slots([active, queued, future], 0, 2), [])
        self.assertEqual(active.blocks(16), 3)
        self.assertEqual(admission_slots([active, queued, future], 3, 2), ['b'])

    def test_recovery_keeps_output_and_completion_releases_blocks(self):
        r = RequestState('a', 32, 40, 9, 32, 'decode')
        victim = apply_event(r, event='preempt')
        self.assertEqual((victim.blocks(16), victim.output_tokens, victim.recovery_tokens), (0, 9, 40))
        partial = apply_event(victim, computed_delta=32)
        self.assertEqual(partial.phase, 'recovering')
        restored = apply_event(partial, computed_delta=9, returned_tokens=1)
        self.assertEqual((restored.phase, restored.output_tokens), ('decode', 10))
        self.assertEqual(apply_event(restored, event='complete').blocks(16), 0)

    def test_batch_shares_load_under_supplied_resident_sets(self):
        sizes = {0:100, 1:100, 2:100}
        self.assertEqual(layer_transfer_bytes([{0,1}, {1,2}], {0}, sizes), 200)
        self.assertEqual(layer_transfer_bytes([{0,1}], {0,1}, sizes), 0)
        self.assertEqual(layer_transfer_bytes([{0,1}], {1,2}, sizes), 100)

    def test_impossible_lifecycle_and_early_output_are_rejected(self):
        waiting = RequestState('a', 32, 0, 0, 32, 'waiting')
        self.assertIs(apply_event(waiting), waiting)
        for event in ('preempt', 'complete'):
            with self.assertRaises(ValueError):
                apply_event(waiting, event=event)
        with self.assertRaises(ValueError):
            apply_event(waiting, computed_delta=16, returned_tokens=1)
        with self.assertRaises(ValueError):
            RequestState('bad', 32, -1, 0, 32, 'prefill')
        done = RequestState('done', 32, 32, 1, 32, 'complete')
        with self.assertRaises(ValueError):
            apply_event(done, event='preempt')

    def test_fcfs_does_not_depend_on_reconstruction_order(self):
        a = RequestState('a', 32, 0, 0, 32, 'waiting', 1)
        b = RequestState('b', 32, 0, 0, 32, 'waiting', 2)
        self.assertEqual(admission_slots([b, a], 1, 3), ['a'])
        with self.assertRaises(ValueError):
            admission_slots([a, a], 1, 3)

    def test_uniform_expectation_is_analytic_and_saturates(self):
        probs = [{i:8/64 for i in range(64)}]*32
        self.assertAlmostEqual(independent_expected_bytes(probs, set(), {i:1 for i in range(64)}), uniform_union(64,8,32))
        self.assertLess(uniform_union(64,8,32), 64)
        self.assertGreater(uniform_union(64,8,32), 63)
        self.assertEqual(step_time_s(nontransfer_s=0.01, exposed_transfer_s=0.002, scheduler_s=0.001), 0.013000000000000001)


if __name__ == '__main__':
    unittest.main()
