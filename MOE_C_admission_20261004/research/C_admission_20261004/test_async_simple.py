"""Bounded CPU lifecycle/accounting checks using pinned native allocation ASTs."""
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import async_simple as simple
from async_observer import Observer
from test_admission_probe import Queue, Request, Status, allocation_state, manager


def request(rid, status=Status.WAITING, *, prompt=32, maximum=32, preemptions=0):
    req = Request(rid, status, tokens=prompt, preemptions=preemptions)
    req.max_tokens = maximum
    req.num_output_placeholders = req.async_tokens_to_discard = 0
    req.num_tokens_with_spec = req.num_tokens
    req.next_decode_eligible_step = 0
    return req


def scheduler(requests=(), *, free=100):
    k = manager(free=free)
    k.max_model_len = 4096
    return NS(requests={r.request_id: r for r in requests},
        running=[r for r in requests if r.status == Status.RUNNING],
        kv_cache_manager=k, max_model_len=4096, deferred_frees=[],
        sched_step_seq=0, processed_step_seq=0, current_step=0,
        num_waiting_for_streaming_input=0)


def before(gate, req):
    return gate.before_allocate(req, scheduled={}, token_budget=32,
        num_new_tokens=16, full_sequence_must_fit=True)


class AsyncSimpleTests(unittest.TestCase):
    def test_old_preempted_and_remote_requests_never_query_fit_or_enter_gate(self):
        old = [request('preempted', Status.PREEMPTED, preemptions=1),
               request('remote', Status.WAITING_FOR_REMOTE_KVS)]
        s = scheduler(old, free=16)
        gate = simple.Gate(s, 'fixed', cap=2, budget_blocks=16)
        self.assertIsInstance(gate, Observer)
        gate.admitted.update(r.request_id for r in old)
        gate.begin()
        gate.barrier = 'cap'
        with patch.object(simple, 'pure_fit', side_effect=AssertionError('old request queried fit')) as fit:
            for req in old:
                with self.subTest(status=req.status.name):
                    self.assertIsNone(gate.early(req))
                    check = before(gate, req)
                    self.assertEqual(check, {'fit': None, 'row': None})
                    # Native recovery results are accepted without a fit prediction.
                    gate.after_allocate(req, check, NS(blocks=([],)))
            fit.assert_not_called()
        self.assertEqual(gate.decisions, [])
        self.assertEqual(gate.fit_checks, 0)

    def test_fifo_restores_on_exception_and_scan_break_preserves_nonrunning_work(self):
        for old_status in (Status.RUNNING, Status.WAITING_FOR_REMOTE_KVS):
            with self.subTest(old_status=old_status.name):
                old = request('old', old_status)
                a, b, c, d = [request(rid) for rid in 'abcd']
                s = scheduler([old, a, b, c, d])
                m = s.kv_cache_manager.coordinator.single_type_managers[0]
                m.req_to_blocks['old'] = s.kv_cache_manager.block_pool.get_new_blocks(2)
                gate = simple.Gate(s, 'fixed', cap=1, budget_blocks=100)
                gate.admitted.add('old')
                gate.begin()
                check = before(gate, a)
                self.assertTrue(check['row']['native_fit'])
                self.assertTrue(check['row']['denied'])
                self.assertEqual(check['row']['reason'], 'cap')
                q1, q2 = Queue([a, b]), Queue([c, d])
                with self.assertRaisesRegex(RuntimeError, 'native failure'):
                    try:
                        gate.hold(q1, q1.pop_request())
                        with patch.object(simple, 'pure_fit', side_effect=AssertionError('FIFO queried fit')):
                            self.assertEqual(gate.early(c), 'hold')
                            gate.hold(q2, q2.pop_request())
                            self.assertEqual(gate.early(b), 'hold')
                            gate.hold(q1, q1.pop_request())
                            self.assertIsNone(gate.early(old))
                        raise RuntimeError('native failure')
                    finally:
                        gate.restore()
                gate.restore()  # Cleanup remains idempotent.
                self.assertEqual(q1.items, [a, b])
                self.assertEqual(q2.items, [c, d])
                self.assertEqual(gate.held, [])
                self.assertTrue(all(row['native_fit'] is None for row in gate.decisions[1:]))
                self.assertEqual(gate.stop_new_scan(check), old_status == Status.RUNNING)

    def test_native_fit_failure_retains_native_allocation_none_even_at_cap(self):
        old, new = request('old', Status.RUNNING), request('new')
        s = scheduler([old, new], free=0)
        gate = simple.Gate(s, 'fixed', cap=1, budget_blocks=16)
        gate.admitted.add('old')
        gate.begin()
        original = allocation_state(s.kv_cache_manager)
        check = before(gate, new)
        self.assertFalse(check['row']['native_fit'])
        self.assertFalse(check['row']['denied'])
        self.assertEqual(check['row']['reason'], 'native_capacity')
        self.assertFalse(gate.stop_new_scan(check))
        self.assertEqual(allocation_state(s.kv_cache_manager), original)
        blocks = s.kv_cache_manager.allocate_slots(new, num_new_tokens=16,
                                                 full_sequence_must_fit=True)
        self.assertIsNone(blocks)
        gate.after_allocate(new, check, blocks)
        self.assertFalse(check['row']['native_allocation_succeeded'])
        self.assertEqual(gate.fit_mismatches, 0)
        self.assertEqual(allocation_state(s.kv_cache_manager), original)
        self.assertNotIn('new', gate.admitted)

    def test_deferred_ghost_stays_physical_without_double_charging_live_pages(self):
        old = request('old', Status.RUNNING, prompt=32, maximum=48)  # 5 declared pages.
        new = request('new', prompt=16, maximum=32)  # 3 declared pages.
        s = scheduler([old, new], free=10)
        pool = s.kv_cache_manager.block_pool
        m = s.kv_cache_manager.coordinator.single_type_managers[0]
        m.req_to_blocks['old'] = pool.get_new_blocks(2)
        ghost = pool.get_new_blocks(3)
        # Native pop_blocks_for_free already removed this finished request's
        # table; the pages are held only by deferred_frees and not yet pool-free.
        s.deferred_frees = [('ghost', ghost)]
        self.assertNotIn('ghost', s.requests)
        self.assertNotIn('ghost', m.req_to_blocks)
        gate = simple.Gate(s, 'declared_budget', cap=2, budget_blocks=10)
        gate.admitted.add('old')
        gate.begin()
        check = before(gate, new)
        row = check['row']
        self.assertTrue(row['native_fit'])
        self.assertTrue(row['denied'])
        self.assertEqual(row['reason'], 'declared_budget')
        self.assertEqual({key: row[key] for key in (
            'physical_used_blocks', 'live_allocated_blocks', 'live_remaining_blocks',
            'nonlive_physical_blocks', 'budget_before_blocks', 'budget_required_blocks',
            'budget_after_if_admitted_blocks')}, dict(
                physical_used_blocks=5, live_allocated_blocks=2, live_remaining_blocks=3,
                nonlive_physical_blocks=3, budget_before_blocks=8,
                budget_required_blocks=3, budget_after_if_admitted_blocks=11))
        # Simulate native deferred completion: only now may these pages count free.
        s.deferred_frees.clear()
        pool.free += len(ghost)
        gate.begin()
        allowed = before(gate, new)
        self.assertTrue(allowed['row']['native_fit'])
        self.assertFalse(allowed['row']['denied'])
        self.assertEqual(allowed['row']['reason'], 'allow')
        self.assertEqual(allowed['row']['physical_used_blocks'], 2)
        self.assertEqual(allowed['row']['live_allocated_blocks'], 2)
        self.assertEqual(allowed['row']['live_remaining_blocks'], 3)
        self.assertEqual(allowed['row']['budget_before_blocks'], 5)
        self.assertEqual(allowed['row']['budget_after_if_admitted_blocks'], 8)
        blocks = s.kv_cache_manager.allocate_slots(new, num_new_tokens=16,
                                                 full_sequence_must_fit=True)
        self.assertIsNotNone(blocks)
        gate.after_allocate(new, allowed, blocks)
        self.assertTrue(allowed['row']['native_allocation_succeeded'])
        self.assertIn('new', gate.admitted)


if __name__ == '__main__':
    unittest.main()
