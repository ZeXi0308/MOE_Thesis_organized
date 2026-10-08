"""Small CPU-only checks for the read-only allocation observer."""
from collections import defaultdict
from types import SimpleNamespace as NS
import unittest

from observe_tail import _install


class Pool:
    def __init__(self): self.free = 0
    def get_num_free_blocks(self): return self.free


class Manager:
    def __init__(self):
        self.block_pool = Pool()
        self.max_model_len = 4096
        self.coordinator = NS(single_type_managers=[NS(req_to_blocks=defaultdict(list))])
        self.calls, self.result, self.error = [], object(), None

    def allocate_slots(self, *args, **kwargs):
        self.calls.append((args, kwargs.copy()))
        if self.error: raise self.error
        return self.result


class Scheduler:
    def __init__(self):
        self.kv_cache_manager = Manager()
        self.preempt_calls, self.schedule_calls = [], []
        self.preempt_error = self.schedule_error = None
        self.output = NS(num_scheduled_tokens={})

    def _preempt_request(self, *args, **kwargs):
        self.preempt_calls.append((args, kwargs.copy()))
        if self.preempt_error: raise self.preempt_error
        return None

    def schedule(self, *args, **kwargs):
        self.schedule_calls.append((args, kwargs.copy()))
        if self.schedule_error: raise self.schedule_error
        return self.output


class ObserverChecks(unittest.TestCase):
    def setUp(self):
        self.s = Scheduler(); self.m = self.s.kv_cache_manager
        self.req = NS(request_id='r', status=NS(name='WAITING'), num_tokens=48,
                      num_computed_tokens=32, num_in_flight_tokens=0)
        self.data, self.close = _install(self.s, dict(exact_layout=True, block_size=16))
        self.addCleanup(self.close)

    def test_readonly_scope_and_call_identity(self):
        table = self.m.coordinator.single_type_managers[0].req_to_blocks
        result = self.m.allocate_slots(self.req, 16)
        self.assertIs(result, self.m.result)
        self.assertEqual(self.data['events'], [])
        self.s._preempt_request(self.req, timestamp=123.)
        result = self.m.allocate_slots(self.req, 16, reserved_blocks=2)
        self.assertIs(result, self.m.result)
        self.assertEqual(len(self.m.calls), 2)
        self.assertIs(self.m.calls[-1][0][0], self.req)
        self.assertEqual(self.m.calls[-1][0][1:], (16,))
        self.assertEqual(self.m.calls[-1][1], {'reserved_blocks': 2})
        self.assertEqual(dict(table), {})  # .get must not insert a defaultdict key.
        self.assertEqual(self.m.block_pool.free, 0)
        self.assertEqual(self.req.num_computed_tokens, 32)
        self.s.output.num_scheduled_tokens = {'r': 16}
        self.assertIs(self.s.schedule(throttle_prefills=False), self.s.output)
        count = len(self.data['events'])
        self.m.allocate_slots(self.req, 1)
        self.assertEqual(len(self.data['events']), count)
        self.assertEqual(self.data['allocation_attempts'], 1)
        self.close()
        self.assertNotIn('allocate_slots', vars(self.m))
        self.assertNotIn('schedule', vars(self.s))

    def test_held_prefix_tail_shortfall_and_full_fit_is_only_gate(self):
        self.s._preempt_request(self.req, 0.)
        held = [NS(is_null=False, ref_cnt=1), NS(is_null=False, ref_cnt=1)]
        self.m.coordinator.single_type_managers[0].req_to_blocks['r'] = held
        self.m.result = None
        self.m.allocate_slots(self.req, 16, full_sequence_must_fit=True)
        row = self.data['events'][-1]
        self.assertEqual(row['before']['held_gpu_blocks'], [2])
        self.assertEqual(row['descriptor']['slot_extra_blocks'], 1)
        self.assertEqual(row['descriptor']['full_fit_extra_blocks'], 1)
        self.assertEqual(row['descriptor']['shortfall_blocks'], 1)
        self.assertFalse(row['success'])
        self.assertIs(self.m.coordinator.single_type_managers[0].req_to_blocks['r'], held)
        self.req.num_computed_tokens = 0
        self.req.num_tokens = 64
        held.clear()
        self.m.allocate_slots(self.req, 0, num_external_computed_tokens=32, full_sequence_must_fit=True)
        descriptor = self.data['events'][-1]['descriptor']
        self.assertEqual(descriptor['slot_extra_blocks'], 2)
        self.assertEqual(descriptor['full_fit_extra_blocks'], 4)

    def test_unsupported_counts_and_exception_identity(self):
        self.s._preempt_request(self.req, 0.)
        error = ValueError('native sentinel'); self.m.error = error
        with self.assertRaises(ValueError) as caught:
            self.m.allocate_slots(request=self.req, num_new_tokens=16, num_lookahead_tokens=1)
        self.assertIs(caught.exception, error)
        row = self.data['events'][-1]
        self.assertFalse(row['descriptor']['exact'])
        self.assertNotIn('shortfall_blocks', row['descriptor'])
        self.assertEqual(row['exception'], 'ValueError')
        self.assertLessEqual(row['begin_host_perf_s'], row['end_host_perf_s'])
        self.assertEqual(len(self.m.calls), 1)
        self.s.schedule_error = error
        with self.assertRaises(ValueError) as caught: self.s.schedule()
        self.assertIs(caught.exception, error)
        self.s.preempt_error = error
        with self.assertRaises(ValueError) as caught: self.s._preempt_request(self.req, 2.)
        self.assertIs(caught.exception, error)


if __name__ == '__main__':
    unittest.main()
