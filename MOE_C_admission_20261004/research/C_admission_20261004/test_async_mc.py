"""Targeted CPU checks for conservative async MC mapping and actual admission.

Uses the existing pinned native allocation AST fixture; no GPU or timing model.
"""
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import async_mc
import async_simple
from mc_budget import peak_envelope
from test_async_simple import request, scheduler
from test_admission_probe import Queue, Status, allocation_state


class AsyncMCTests(unittest.TestCase):
    @staticmethod
    def case(*, budget=5, nonlive=0):
        old = request('old', Status.RUNNING, prompt=16, maximum=18)
        old.num_output_tokens, old.num_tokens = 16, 32
        old.num_tokens_with_spec = 32
        old.num_computed_tokens = 32
        old.num_output_placeholders = old.num_in_flight_tokens = 1
        old.last_sched_seq, old.is_prefill_chunk = 1, False
        old.spec_token_ids = []
        old.sampling_params = NS(min_tokens=0, structured_outputs=None)
        old.structured_output_request = old.pooling_params = None
        old.resumable = False
        new, later = [request(rid, prompt=16, maximum=32) for rid in ('new', 'later')]
        for req in (new, later):
            req.spec_token_ids = []
            req.sampling_params = NS(min_tokens=0, structured_outputs=None)
            req.structured_output_request = req.pooling_params = None
            req.resumable, req.is_prefill_chunk, req.last_sched_seq = False, False, 0
        s = scheduler([old, new, later], free=budget)
        s.sched_step_seq, s.processed_step_seq, s.current_step = 1, 0, 10
        s.max_num_scheduled_tokens = 1024
        manager = s.kv_cache_manager.coordinator.single_type_managers[0]
        # C + scheduled = 33: the current native allocation owns three pages,
        # although prompt + returned output = 32 would misleadingly give two.
        manager.req_to_blocks['old'] = manager.block_pool.get_new_blocks(3)
        if nonlive:
            s.deferred_frees = [('finished', manager.block_pool.get_new_blocks(nonlive))]
        pending = NS(total_num_scheduled_tokens=1, num_scheduled_tokens={'old': 1})
        core = NS(scheduler=s, async_scheduling=True, batch_queue_size=2,
                  batch_queue=[(NS(), pending, NS())])
        gate = async_mc.Gate(s, core, cap=256, budget_blocks=budget)
        gate.admitted.add('old')
        gate.begin()
        gate.scheduled_tokens['old'] = 1
        return gate, old, new, later

    @staticmethod
    def before(gate, req):
        return gate.before_allocate(req, token_budget=1023, num_new_tokens=16,
            full_sequence_must_fit=True, has_scheduled_reqs=True)

    def test_current_allocated_slot_and_remaining_do_not_drop_placeholder(self):
        gate, old, new, _ = self.case()
        physical_before = allocation_state(gate.s.kv_cache_manager)
        check = self.before(gate, new)
        reason, rows, nonlive = gate._old_rows(new, check['row'])
        self.assertEqual((reason, nonlive), (None, 0))
        self.assertEqual(len(rows), 1)
        mapped = rows[0]
        self.assertEqual((mapped['n'], mapped['allocated_blocks']), (33, 3))
        self.assertEqual((old.max_tokens-old.num_output_tokens, mapped['r']), (2, 4))
        self.assertEqual(mapped['placeholders'], 1)
        # Placeholder is an outstanding result, not an observed output to
        # subtract from remaining max_tokens. Such subtraction would give r=3.
        self.assertNotEqual(mapped['r'], old.max_tokens-old.num_output_tokens-1+2)
        self.assertEqual(allocation_state(gate.s.kv_cache_manager), physical_before)
        self.assertEqual((check['row']['ordinary_budget_allowed'],
                          check['row']['final_allowed'], check['row']['mc_peak_blocks']),
                         (False, True, 5))

    def test_release_waits_two_extra_nonempty_steps(self):
        gate, _, new, _ = self.case()
        check = self.before(gate, new)
        reason, rows, _ = gate._old_rows(new, check['row'])
        self.assertIsNone(reason)
        # With two unobserved outputs left, old pages still count at h=3 and
        # disappear only at h=4. This checks the mapping with a small declared
        # new maximum exposing the release checkpoint; no measured duration.
        result = peak_envelope(rows, p=16, B=4)
        self.assertEqual(result['checkpoints'], [dict(h=0, blocks=4),
            dict(h=3, blocks=5), dict(h=4, blocks=2)])

    def test_nonlive_pages_remain_charged_at_every_future_checkpoint(self):
        gate, _, new, _ = self.case(nonlive=1)
        row = self.before(gate, new)['row']
        self.assertTrue(row['mc_eligible'])
        self.assertEqual((row['mc_nonlive_physical_blocks'], row['mc_envelope_peak_blocks'],
                          row['mc_peak_blocks']), (1, 5, 6))
        self.assertTrue(row['denied'])
        self.assertFalse(row['final_allowed'])
        self.assertEqual((gate.barrier, gate.mc['peak_rejections']), ('declared_budget', 1))
        self.assertFalse(row['changed_by_mc_budget'])

    def test_guard_failure_preserves_ordinary_budget_and_fifo(self):
        def remove_current_page(gate, old):
            gate.manager.req_to_blocks[old.request_id].pop()
            gate.manager.block_pool.free += 1

        mutations = (
            ('native_terminal_guard', lambda g,o:setattr(o, 'max_tokens', 17)),
            ('unscheduled_old', lambda g,o:g.scheduled_tokens.update(old=0)),
            ('placeholder_pending_mismatch', lambda g,o:setattr(o, 'num_output_placeholders', 2)),
            ('future_eligibility', lambda g,o:setattr(o, 'next_decode_eligible_step', 11)),
            ('missing_current_page', remove_current_page),
            ('batch_fence_mismatch', lambda g,o:setattr(g.s, 'processed_step_seq', 1)),
        )
        for label, mutate in mutations:
            with self.subTest(guard=label):
                gate, old, new, later = self.case()
                mutate(gate, old)
                row = self.before(gate, new)['row']
                self.assertTrue(row['native_fit'])
                self.assertTrue(row['baseline_allowed'])
                self.assertFalse(row['ordinary_budget_allowed'])
                self.assertTrue(row['mc_attempted'])
                self.assertFalse(row['mc_eligible'])
                self.assertIsNotNone(row['mc_guard_reason'])
                self.assertIsNone(row['mc_peak_blocks'])
                self.assertEqual(gate.mc['peak_evaluations'], 0)
                self.assertTrue(row['denied'])
                self.assertEqual(row['reason'], 'declared_budget')
                self.assertFalse(row['changed_by_mc_budget'])
                self.assertEqual((gate.barrier, gate.barrier_decision_index), ('declared_budget', 0))
                queue = Queue([new, later])
                gate.hold(queue, queue.pop_request())
                self.assertEqual(gate.early(later), 'hold')
                fifo = gate.decisions[-1]
                self.assertEqual(fifo['reason'], 'fifo_declared_budget')
                self.assertIsNone(fifo['native_fit'])
                gate.hold(queue, queue.pop_request())
                gate.restore()
                self.assertEqual(queue.items, [new, later])

    def test_relaxation_clears_only_its_barrier_and_counts_actual_admission(self):
        gate, old, new, later = self.case()
        check = self.before(gate, new)
        row = check['row']
        self.assertTrue(row['changed_by_mc_budget'])
        self.assertFalse(row['denied'])
        self.assertEqual((gate.barrier, gate.barrier_decision_index), (None, None))
        self.assertIsNone(gate.early(later))
        self.assertEqual((gate.mc['requested_relaxations'], gate.mc['actual_allocations'],
                          gate.mc['successful_relaxations']), (1, 0, 0))
        result = gate.s.kv_cache_manager.allocate_slots(new, num_new_tokens=16,
            full_sequence_must_fit=True, has_scheduled_reqs=True)
        self.assertIsNotNone(result)
        gate.after_allocate(new, check, result)
        self.assertTrue(row['native_allocation_result'])
        self.assertEqual((gate.mc['actual_allocations'], gate.mc['successful_relaxations']), (1, 0))
        new.status = Status.RUNNING
        gate.s.running.append(new)
        gate.confirm_admitted(new)
        gate.confirm_admitted(new)  # An admitted request must not be counted twice.
        self.assertTrue(row['mc_admission_confirmed'])
        self.assertEqual(gate.mc['successful_relaxations'], 1)
        report = gate.report()
        self.assertEqual(report['mode'], 'async_mc')
        self.assertEqual(report['budget_overrides'], 1)
        self.assertEqual(report['mc_budget']['release_delay_nonempty_steps'], 2)
        # The newly admitted, still incomplete prefill invalidates the whole
        # cohort. It may not be silently excluded to lend again this step.
        peaks_before = gate.mc['peak_evaluations']
        next_row = self.before(gate, later)['row']
        self.assertTrue(next_row['native_fit'])
        self.assertFalse(next_row['mc_eligible'])
        self.assertTrue(next_row['denied'])
        self.assertEqual(gate.mc['peak_evaluations'], peaks_before)
        self.assertEqual(gate.barrier, 'declared_budget')

    def test_existing_recovery_bypasses_fit_peak_and_budget(self):
        gate, old, _, _ = self.case()
        gate.barrier = 'declared_budget'
        for status in (Status.PREEMPTED, Status.WAITING_FOR_REMOTE_KVS):
            with self.subTest(status=status.name):
                old.status, old.num_preemptions = status, 1
                with patch.object(async_simple, 'pure_fit', side_effect=AssertionError('recovery queried fit')):
                    self.assertIsNone(gate.early(old))
                    check = self.before(gate, old)
                    self.assertEqual(check, dict(fit=None, row=None))
                    gate.after_allocate(old, check, NS(blocks=([],)))
                self.assertEqual(gate.mc['attempted_evaluations'], 0)
                self.assertEqual(gate.decisions, [])
                self.assertEqual(gate.fit_checks, 0)


if __name__ == '__main__':
    unittest.main()
