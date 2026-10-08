"""CPU arithmetic checks for the declared-length maximum-concurrency envelope."""
import inspect
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

import mc_budget
from mc_budget import Gate, peak_envelope
import test_admission_probe as fixture


def brute_blocks(rows, p, budget, horizon, block_size=16):
    old = sum((row['n'] + horizon + block_size - 1) // block_size
        for row in rows if horizon < row['r'])
    new = (p + min(horizon, budget) + block_size - 1) // block_size
    return old + new


class PeakEnvelopeTests(unittest.TestCase):
    def assert_matches_all_integer_horizons(self, rows, p, budget, block_size=16):
        result = peak_envelope(rows, p, budget, block_size=block_size)
        last = max([budget] + [row['r'] for row in rows])
        values = [brute_blocks(rows, p, budget, h, block_size) for h in range(last + 1)]
        checkpoints = sorted({0, budget, *(row['r'] - 1 for row in rows)})
        self.assertEqual([(point['h'], point['blocks']) for point in result['checkpoints']],
            [(h, values[h]) for h in checkpoints])
        self.assertEqual(result['peak_blocks'], max(values))
        self.assertIn(result['peak_h'], checkpoints)
        self.assertEqual(values[result['peak_h']], result['peak_blocks'])
        return result

    def test_small_deterministic_cases_match_brute_force(self):
        cases = [
            ([], 1, 1, 16),
            ([{'n': 1, 'r': 1}], 16, 1, 16),
            ([{'n': 15, 'r': 1}], 1, 16, 16),
            ([{'n': 16, 'r': 2}], 15, 1, 16),
            ([{'n': 17, 'r': 3}, {'n': 31, 'r': 5}], 16, 2, 16),
            ([{'n': 1, 'r': 4}, {'n': 16, 'r': 4}], 17, 4, 16),
            ([{'n': 31, 'r': 2}, {'n': 1, 'r': 6}], 15, 8, 16),
            ([{'n': 3, 'r': 2}, {'n': 1, 'r': 5}], 2, 3, 2),
        ]
        for rows, p, budget, block_size in cases:
            with self.subTest(rows=rows, p=p, budget=budget, block_size=block_size):
                self.assert_matches_all_integer_horizons(rows, p, budget, block_size)

    def test_exact_page_boundary_and_first_token_over_boundary(self):
        exact = self.assert_matches_all_integer_horizons([{'n': 15, 'r': 2}], 15, 1)
        self.assertEqual(exact['checkpoints'], [{'h': 0, 'blocks': 2}, {'h': 1, 'blocks': 2}])
        crossed = self.assert_matches_all_integer_horizons([{'n': 16, 'r': 2}], 16, 1)
        self.assertEqual(crossed['checkpoints'], [{'h': 0, 'blocks': 2}, {'h': 1, 'blocks': 4}])
        self.assertEqual((crossed['peak_blocks'], crossed['peak_h']), (4, 1))

    def test_old_request_counts_last_live_step_and_releases_at_r(self):
        result = self.assert_matches_all_integer_horizons([{'n': 31, 'r': 2}], 1, 2)
        self.assertEqual(result['checkpoints'],
            [{'h': 0, 'blocks': 3}, {'h': 1, 'blocks': 3}, {'h': 2, 'blocks': 1}])
        self.assertEqual(result['peak_blocks'], 3)

    def test_new_request_retains_plateau_after_its_budget(self):
        result = self.assert_matches_all_integer_horizons([{'n': 13, 'r': 5}], 16, 1)
        self.assertEqual(result['checkpoints'],
            [{'h': 0, 'blocks': 2}, {'h': 1, 'blocks': 3}, {'h': 4, 'blocks': 4}])
        self.assertEqual((result['peak_blocks'], result['peak_h']), (4, 4))


class GuardedGateTests(unittest.TestCase):
    setUp = fixture.GateTests.setUp

    @staticmethod
    def request(rid, *, prompt=16, output=0, maximum=16, running=False):
        req = fixture.Request(rid, fixture.Status.RUNNING if running else fixture.Status.WAITING,
            tokens=prompt+output, output=output, computed=prompt+output-1 if running else 0)
        req.max_tokens, req.num_output_placeholders, req.spec_token_ids = maximum, 0, []
        req.next_decode_eligible_step, req.async_tokens_to_discard = 0, 0
        req.sampling_params = NS(min_tokens=0, structured_outputs=None)
        req.structured_output_request = req.pooling_params = None
        req.resumable = False
        return req

    def case(self, *, budget=4, maximum=32, old_output=15, old_maximum=16, cap=256, **gate_options):
        old = self.request('old', output=old_output, maximum=old_maximum, running=True)
        new, later = self.request('new', maximum=maximum), self.request('later')
        scheduler = fixture.scheduler([old, new, later], free=budget)
        scheduler.max_model_len = scheduler.kv_cache_manager.max_model_len = 4096
        scheduler.num_waiting_for_streaming_input, scheduler.current_step = 0, 1
        manager = scheduler.kv_cache_manager.coordinator.single_type_managers[0]
        manager.req_to_blocks[old.request_id] = manager.block_pool.get_new_blocks((old.num_tokens+15)//16)
        gate = Gate(scheduler, cap=cap, budget_blocks=budget, **gate_options)
        gate.admitted(old)
        scheduler.running.append(old)
        gate.begin()
        gate.scheduled_tokens[old.request_id] = 1
        return gate, old, new, later

    @staticmethod
    def before(gate, req):
        return gate.before_allocate(req, token_budget=1023, num_new_tokens=16,
            full_sequence_must_fit=True, has_scheduled_reqs=True)

    def test_real_allocation_authorizes_charge_and_partial_prefill_blocks_next_loan(self):
        gate, old, new, later = self.case()
        physical_before = fixture.allocation_state(gate.s.kv_cache_manager)
        check = self.before(gate, new)
        self.assertEqual(fixture.allocation_state(gate.s.kv_cache_manager), physical_before)
        row = check['row']
        self.assertEqual((row['ordinary_budget_allowed'], row['final_allowed'], row['mc_peak_blocks']),
                         (False, True, 3))
        self.assertEqual(row['reason'], 'mc_budget_allow')
        self.assertTrue(row['changed_by_mc_budget'])
        self.assertIsNone(gate.barrier)
        self.assertEqual(gate.mc['successful_relaxations'], 0)
        result = gate.s.kv_cache_manager.allocate_slots(new, num_new_tokens=16,
            full_sequence_must_fit=True, has_scheduled_reqs=True)
        gate.after_allocate(new, check, result)
        new.status = fixture.Status.RUNNING
        gate.admitted(new)
        gate.s.running.append(new)
        gate.admitted(new)  # Never double charge the same admitted ID.
        self.assertEqual((gate.budget_used_blocks, gate.budget_peak_blocks), (5, 5))
        self.assertEqual(gate.mc['successful_relaxations'], 1)
        next_row = self.before(gate, later)['row']
        self.assertTrue(next_row['native_fit'])
        self.assertTrue(next_row['denied'])
        self.assertEqual(next_row['mc_guard_reason'], 'not_complete_prompt_decode')
        old.status = fixture.Status.FINISHED_STOPPED
        gate.s.running.remove(old)
        gate.begin()
        self.assertEqual(gate.budget_used_blocks, 3)
        report = gate.report()
        self.assertEqual(report['mode'], 'mc_budget')
        self.assertEqual(report['budget_overrides'], 1)

    def test_exact_peak_limit_peak_failure_native_failure_and_cap_are_distinct(self):
        gate, old, new, _ = self.case(budget=3, maximum=16)
        row = self.before(gate, new)['row']
        self.assertEqual(row['mc_peak_blocks'], 3)
        self.assertFalse(row['denied'])
        gate, old, new, _ = self.case(budget=3, maximum=16, old_output=1)
        row = self.before(gate, new)['row']
        self.assertEqual((row['mc_eligible'], row['mc_peak_blocks'], row['denied']), (True, 4, True))
        self.assertEqual(gate.mc['peak_rejections'], 1)
        self.assertTrue(gate.stop_new_scan(dict(row=row)))
        gate, old, new, _ = self.case(budget=3, maximum=16, cap=1)
        row = self.before(gate, new)['row']
        self.assertEqual(row['reason'], 'cap')
        self.assertFalse(row['mc_attempted'])
        gate.s.kv_cache_manager.block_pool.free = 0
        row = self.before(gate, new)['row']
        self.assertEqual(row['reason'], 'native_capacity')
        self.assertFalse(row['denied'])
        self.assertFalse(row['mc_attempted'])

    def test_guard_fallback_fifo_and_existing_recovery_bypass(self):
        mutations = [
            (lambda gate, old:setattr(old, 'num_computed_tokens', old.num_tokens-2), 'not_complete_prompt_decode'),
            (lambda gate, old:setattr(old, 'num_in_flight_tokens', 1), 'not_synchronous_one_token'),
            (lambda gate, old:gate.scheduled_tokens.update(old=2), 'not_synchronous_one_token'),
            (lambda gate, old:setattr(old.sampling_params, 'min_tokens', old.max_tokens+1), 'sampling_minimum'),
            (lambda gate, old:setattr(old, 'structured_output_request', object()), 'non_plain_generation'),
            (lambda gate, old:setattr(gate.s.kv_cache_manager.block_pool, 'free', 1), 'unaccounted_physical_pages'),
        ]
        for mutate, reason in mutations:
            with self.subTest(reason=reason):
                gate, old, new, _ = self.case()
                mutate(gate, old)
                row = self.before(gate, new)['row']
                self.assertEqual((row['mc_guard_reason'], row['denied']), (reason, True))
        gate, old, new, later = self.case()
        old.status, old.num_preemptions = fixture.Status.PREEMPTED, 1
        old.num_computed_tokens = 0
        manager = gate.s.kv_cache_manager.coordinator.single_type_managers[0]
        manager.block_pool.free += len(manager.req_to_blocks.pop(old.request_id))
        gate.s.running.clear()
        check = self.before(gate, new)
        self.assertEqual(check['row']['mc_guard_reason'], 'admitted_not_all_running')
        self.assertFalse(gate.stop_new_scan(check))
        queue = fixture.Queue([new, later, old])
        gate.hold(queue, queue.pop_request())
        self.assertEqual(gate.early(later), 'hold')
        gate.hold(queue, queue.pop_request())
        self.assertIsNone(gate.early(old))
        self.assertIsNone(self.before(gate, old)['row'])
        gate.restore()
        self.assertEqual(queue.items, [new, later, old])
        self.assertEqual(gate.budget_used_blocks, 2)

    def test_capped_options_validate_and_ordinary_allow_stays_unchanged(self):
        for function in (Gate, mc_budget.install):
            parameters = inspect.signature(function).parameters
            for name, default in (('use_future_peak', True), ('maximum_declared_blocks', None),
                                  ('allow_scheduled_prefill', False)):
                self.assertEqual(parameters[name].kind, inspect.Parameter.KEYWORD_ONLY)
                self.assertIs(parameters[name].default, default)
        for options in ({'use_future_peak': 1}, {'maximum_declared_blocks': 4.0},
                        {'maximum_declared_blocks': 2}, {'use_future_peak': False},
                        {'allow_scheduled_prefill': 1},
                        {'allow_scheduled_prefill': True, 'use_future_peak': False,
                         'maximum_declared_blocks': 4}):
            with self.subTest(invalid=options), self.assertRaises(ValueError):
                self.case(budget=3, maximum=16, old_output=1, **options)
        default, _, default_new, _ = self.case()
        self.assertEqual(default.mode, 'mc_budget')
        default_row = self.before(default, default_new)['row']
        self.assertFalse(default_row['allow_scheduled_prefill'])
        self.assertFalse(default_row['changed_by_scheduled_prefill'])
        for future in (True, False):
            with self.subTest(ordinary_allow=True, use_future_peak=future):
                gate, _, new, _ = self.case(budget=4, maximum=16, old_output=1,
                    use_future_peak=future, maximum_declared_blocks=4)
                with patch.object(mc_budget, 'peak_envelope', side_effect=AssertionError('peak must not run')):
                    row = self.before(gate, new)['row']
                self.assertTrue(row['ordinary_budget_allowed'])
                self.assertTrue(row['final_allowed'])
                self.assertFalse(row['denied'])
                self.assertEqual(row['reason'], 'allow')
                self.assertFalse(row['mc_attempted'])
                self.assertFalse(row['mc_peak_computed'])
                self.assertIsNone(row['mc_peak_blocks'])
                self.assertIsNone(row['mc_peak_h'])

    def test_same_ceiling_static_allows_while_future_peak_rejects(self):
        static, _, new, _ = self.case(budget=3, maximum=16, old_output=1,
            use_future_peak=False, maximum_declared_blocks=4)
        self.assertEqual(static.mode, 'mc_budget_static')
        with patch.object(mc_budget, 'peak_envelope', side_effect=AssertionError('static must not compute a peak')):
            row = self.before(static, new)['row']
        self.assertEqual(row['budget_after_if_admitted_blocks'], 4)
        self.assertTrue(row['mc_eligible'])
        self.assertIsNone(row['mc_guard_reason'])
        self.assertFalse(row['ordinary_budget_allowed'])
        self.assertFalse(row['denied'])
        self.assertTrue(row['final_allowed'])
        self.assertEqual(row['reason'], 'mc_budget_static_allow')
        self.assertFalse(row['mc_peak_computed'])
        self.assertIsNone(row['mc_peak_blocks'])
        self.assertIsNone(row['mc_peak_h'])
        capped, _, new, _ = self.case(budget=3, maximum=16, old_output=1,
            use_future_peak=True, maximum_declared_blocks=4)
        self.assertEqual(capped.mode, 'mc_budget_capped')
        row = self.before(capped, new)['row']
        self.assertEqual(row['budget_after_if_admitted_blocks'], 4)
        self.assertTrue(row['mc_eligible'])
        self.assertIsNone(row['mc_guard_reason'])
        self.assertTrue(row['mc_peak_computed'])
        self.assertEqual(row['mc_peak_blocks'], 4)
        self.assertIsNotNone(row['mc_peak_h'])
        self.assertTrue(row['denied'])
        self.assertFalse(row['final_allowed'])
        self.assertEqual(row['reason'], 'declared_budget')

    def test_shared_declared_ceiling_rejects_before_any_peak_computation(self):
        for future in (True, False):
            with self.subTest(use_future_peak=future):
                gate, _, new, _ = self.case(budget=3, maximum=16, old_output=1,
                    use_future_peak=future, maximum_declared_blocks=3)
                with patch.object(mc_budget, 'peak_envelope', side_effect=AssertionError('ceiling must short circuit peak')):
                    row = self.before(gate, new)['row']
                self.assertEqual(row['budget_after_if_admitted_blocks'], 4)
                self.assertTrue(row['declared_ceiling_denied'])
                self.assertEqual(row['mc_limit_reason'], 'declared_ceiling')
                self.assertTrue(row['denied'])
                self.assertFalse(row['final_allowed'])
                self.assertEqual(row['reason'], 'declared_budget')
                self.assertFalse(row['mc_peak_computed'])
                self.assertIsNone(row['mc_peak_blocks'])
                self.assertIsNone(row['mc_peak_h'])

    def test_scheduled_last_prefill_step_relaxes_only_the_phase_guard(self):
        strict, _, new, _ = self.case(budget=4, maximum=32, old_output=0)
        strict_row = self.before(strict, new)['row']
        self.assertTrue(strict_row['denied'])
        self.assertFalse(strict_row['mc_eligible'])
        self.assertFalse(strict_row['changed_by_scheduled_prefill'])
        phase, old, new, _ = self.case(budget=4, maximum=32, old_output=0,
            allow_scheduled_prefill=True)
        self.assertEqual(phase.mode, 'mc_budget_phase')
        self.assertEqual((old.num_computed_tokens, phase.scheduled_tokens['old']), (15, 1))
        row = self.before(phase, new)['row']
        self.assertTrue(row['allow_scheduled_prefill'])
        self.assertTrue(row['mc_eligible'])
        self.assertEqual((row['mc_peak_blocks'], row['mc_scheduled_prefill_count']), (4, 1))
        self.assertFalse(row['denied'])
        self.assertTrue(row['final_allowed'])
        self.assertTrue(row['changed_by_scheduled_prefill'])
        event = phase.mc['first_scheduled_prefill_relaxation']
        self.assertEqual(event['request_id'], new.request_id)
        self.assertEqual((event['mc_scheduled_prefill_count'], event['num_new_tokens']), (1, 16))
        self.assertTrue(event['allow_scheduled_prefill'])
        self.assertEqual(event['envelope']['peak_blocks'], 4)
        self.assertEqual({key: event['old_rows'][0][key] for key in
            ('scheduled_prefill', 'computed', 'output', 'prompt', 'max_tokens', 'scheduled_tokens')},
            dict(scheduled_prefill=True, computed=15, output=0, prompt=16,
                 max_tokens=16, scheduled_tokens=1))
        another = self.request('another', maximum=32)
        phase.s.requests[another.request_id] = another
        self.assertFalse(self.before(phase, another)['row']['denied'])
        self.assertIs(phase.mc['first_scheduled_prefill_relaxation'], event)
        decode, _, new, _ = self.case(allow_scheduled_prefill=True)
        decode_row = self.before(decode, new)['row']
        self.assertFalse(decode_row['denied'])
        self.assertEqual(decode_row['mc_scheduled_prefill_count'], 0)
        self.assertFalse(decode_row['changed_by_scheduled_prefill'])

    def test_phase_guard_still_rejects_partial_or_unscheduled_prefill(self):
        for computed, scheduled in ((14, 1), (15, 0)):
            with self.subTest(computed=computed, scheduled=scheduled):
                gate, old, new, _ = self.case(budget=4, maximum=32, old_output=0,
                    allow_scheduled_prefill=True)
                old.num_computed_tokens = computed
                gate.scheduled_tokens['old'] = scheduled
                with patch.object(mc_budget, 'peak_envelope', side_effect=AssertionError('failed guard must not compute peak')):
                    row = self.before(gate, new)['row']
                self.assertTrue(row['denied'])
                self.assertFalse(row['mc_eligible'])
                self.assertFalse(row['mc_peak_computed'])
                self.assertFalse(row['changed_by_scheduled_prefill'])
                self.assertIsNone(gate.mc['first_scheduled_prefill_relaxation'])

    def test_one_output_prefill_is_live_at_h0_and_released_at_h1(self):
        gate, _, new, _ = self.case(budget=3, maximum=1, old_output=0, old_maximum=1,
            allow_scheduled_prefill=True)
        row = self.before(gate, new)['row']
        self.assertFalse(row['denied'])
        self.assertTrue(row['changed_by_scheduled_prefill'])
        self.assertEqual(row['mc_scheduled_prefill_count'], 1)
        event = gate.mc['first_scheduled_prefill_relaxation']
        self.assertEqual((event['old_rows'][0]['n'], event['old_rows'][0]['r']), (16, 1))
        # h0: old prompt page + new prompt page. h1: old is free while
        # the new prompt-plus-one-token bound occupies two pages.
        self.assertEqual(event['envelope']['checkpoints'],
            [{'h': 0, 'blocks': 2}, {'h': 1, 'blocks': 2}])


if __name__ == '__main__':
    unittest.main()
