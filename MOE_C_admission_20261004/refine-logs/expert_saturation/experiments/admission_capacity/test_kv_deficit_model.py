"""Targeted checks for the KV deficit primitives.

Scope follows the exploration-stage audit budget: identity/alignment, causal
cutoff (no future information), accounting correctness, and action semantics.
No coverage engineering.
"""

import unittest

from kv_deficit_model import (
    BridgeRequirement,
    DeficitLedger,
    DepletionForecast,
    StaggerFeasibility,
    StaticFeasibility,
    decode_drain_rate,
    first_release_step,
    infer_block_size,
    predict_victim_stall,
    preemption_race,
    request_blocks,
)


class TestBlockSizeInference(unittest.TestCase):
    def test_identifies_block_size_from_sealed_observations(self):
        # (computed_tokens, block_counts) read from the two sealed victims.
        self.assertEqual(infer_block_size([(3780, 237), (3905, 245)]), 16)

    def test_rejects_ambiguous_evidence(self):
        # One observation alone does not pin the block size.
        with self.assertRaises(ValueError):
            infer_block_size([(32, 2)])

    def test_rejects_inconsistent_evidence(self):
        with self.assertRaises(ValueError):
            infer_block_size([(3780, 237), (3905, 300)])

    def test_rejects_empty_and_nonpositive(self):
        with self.assertRaises(ValueError):
            infer_block_size([])
        with self.assertRaises(ValueError):
            infer_block_size([(0, 1)])


class TestStaticFeasibility(unittest.TestCase):
    def _cell(self, usable):
        return StaticFeasibility(
            n_requests=32,
            prompt_tokens=3072,
            max_output_tokens=1024,
            block_size=16,
            usable_blocks=usable,
        )

    def test_uses_only_admission_time_information(self):
        # Terminal demand depends on declared bounds, never on realised output.
        self.assertEqual(self._cell(7671).blocks_per_request, 256)
        self.assertEqual(self._cell(7671).terminal_demand_blocks, 8192)

    def test_sign_matches_the_four_sealed_cells(self):
        self.assertTrue(self._cell(7671).infeasible)   # repeat0/1 budget90
        self.assertFalse(self._cell(8425).infeasible)  # repeat0 budget95
        self.assertFalse(self._cell(8473).infeasible)  # repeat1 budget95

    def test_deficit_and_feasible_concurrency_are_consistent(self):
        cell = self._cell(7671)
        self.assertEqual(cell.deficit_blocks, 521)
        self.assertEqual(cell.feasible_concurrency, 29)
        # A cohort trimmed to the feasible concurrency must itself be feasible.
        trimmed = StaticFeasibility(29, 3072, 1024, 16, 7671)
        self.assertFalse(trimmed.infeasible)

    def test_margin_and_deficit_never_double_count(self):
        cell = self._cell(8425)
        self.assertEqual(cell.deficit_blocks, 0)
        self.assertEqual(cell.margin_blocks, 8425 - 8192)

    def test_completion_avoids_simultaneous_terminal_demand_without_holds(self):
        from resource_transition_model import Event, Request, State, reduce_events
        # Natural progress is staggered: A has emitted 2 tokens and B has 1.
        # Both advance on every step they remain active. The terminal native
        # footprint would be 3+3>5, but A completes before B reaches 3 blocks.
        state = State(tuple(Request(rid, prompt_tokens=1, status='decode',
            prompt_processed=1, output_observed=outputs, cached_tokens=outputs,
            cached_blocks=outputs, ever_admitted=True)
            for rid, outputs in [('A', 2), ('B', 1)]),
            usable_kv_blocks=5, block_size=1, cap_target=2)
        state = reduce_events(state, [Event('decode', 'A', 1, cached_tokens_after=3),
            Event('decode', 'B', 1, cached_tokens_after=2)])
        self.assertEqual(state.used_kv_blocks, 5)
        state = reduce_events(state, [Event('complete', 'A', 1),
            Event('decode', 'B', 2, cached_tokens_after=3), Event('complete', 'B', 2)])
        self.assertTrue(all(r.status == 'completed' and r.output_observed == 3 for r in state.requests))
        result = StaticFeasibility(2, 1, 3, 1, 5).as_dict()
        self.assertTrue(result['terminal_co_residency_exceeds_pool'])
        self.assertNotIn('predicted_preemption', result)


class TestDepletionForecast(unittest.TestCase):
    def test_drain_rate_matches_width_over_block_size(self):
        self.assertAlmostEqual(decode_drain_rate(32, 16), 2.0)

    def test_forecast_from_early_state_reaches_sealed_exhaustion_step(self):
        # Observed: free=1412 at step 100, first preemption attempt at step 806.
        f = DepletionForecast(step0=100, free_blocks0=1412, drain_rate=2.0)
        self.assertEqual(f.exhaustion_step, 806)

    def test_lead_time_is_positive_when_action_window_closes_first(self):
        f = DepletionForecast(step0=100, free_blocks0=1412, drain_rate=2.0)
        # Last new admission in the sealed cells happened at step 94.
        self.assertEqual(f.lead_time_steps(94), 712)

    def test_non_draining_pool_has_no_exhaustion(self):
        with self.assertRaises(ValueError):
            DepletionForecast(step0=0, free_blocks0=10, drain_rate=0.0).exhaustion_step


class TestPreemptionRace(unittest.TestCase):
    """Arithmetic fits for supplied growth/release values, not online validation.

    The release step 1025 is observed in the retained trace. Using it explains
    that trace but cannot prove an online prediction of first completion.
    """

    def _exhaustion(self, free_at_119):
        return DepletionForecast(119, free_at_119, 2.0).exhaustion_step

    def test_first_output_is_emitted_by_last_prefill_step(self):
        # Prefill steps 0,1,2 emit output 1 at step 2; another 1023 steps
        # emit outputs 2..1024. No scheduling holds in this conditional model.
        self.assertEqual(first_release_step(3, 1024), 1025)
        self.assertEqual(first_release_step(1, 1, first_service_step=7), 7)

    def test_low_budget_cells_lose_the_race(self):
        exh = self._exhaustion(1375)  # free at step 119, both budget90 cells
        self.assertTrue(preemption_race(exh, 1025))
        self.assertLess(abs(exh - 806), 5)

    def test_high_budget_cells_win_the_race(self):
        for free119 in (2129, 2177):  # budget95, repeat0 / repeat1
            exh = self._exhaustion(free119)
            self.assertFalse(preemption_race(exh, 1025))

    def test_race_is_not_trivially_always_true(self):
        # A cohort that drains long before exhaustion must be predicted safe.
        self.assertFalse(preemption_race(exhaustion_step=5000, release_step=1025))

    def test_release_step_rejects_invalid_bounds(self):
        with self.assertRaises(ValueError):
            first_release_step(3, 0)
        with self.assertRaises(ValueError):
            first_release_step(-1, 1024)
        with self.assertRaises(ValueError):
            first_release_step(0, 1024)


class TestForecastCutoff(unittest.TestCase):
    def test_admission_cutoff_does_not_scan_later_trajectory(self):
        from analyze_kv_deficit import last_admission_step
        steps = [{'step': 4, 'scheduled': [{'internal_request_id': 'A',
            'computed_before': 0, 'prefill_tokens': 1}]},
            {'step': 5, 'scheduled': [{'internal_request_id': []}]}]
        self.assertEqual(last_admission_step(steps, 1), 4)

    def test_forecast_window_ignores_future_width_but_rejects_in_window_changes(self):
        from copy import deepcopy
        from analyze_kv_deficit import validate_fit_window
        raw = dict(scheduler_steps=[], engine_steps=[], memory_trace=[],
                   output_events=[], preemption_events=[])
        for k in range(3):
            raw['scheduler_steps'].append(dict(step=k, actual_active=1,
                decode_requests=1, preempted_request_ids=[], scheduled=[dict(
                    internal_request_id='A', prefill_tokens=0, recompute_tokens=0,
                    scheduled_tokens=1, decode_tokens=1)]))
            raw['engine_steps'].append(dict(scheduler_step_start=k,
                scheduler_step_end=k+1, completed=True, returned_s=k+0.5))
            raw['memory_trace'].append(dict(attempted_step=k, schedule_completed=True,
                before={'pool': {'usable_blocks': 4}}, after={'pool': {'usable_blocks': 4}}))
        expected = validate_fit_window(raw, 0, 1, 4)
        changed = deepcopy(raw)
        changed['scheduler_steps'][2]['actual_active'] = 999
        changed['preemption_events'].append({'attempted_step': 2})
        self.assertEqual(validate_fit_window(changed, 0, 1, 4), expected)
        changed['scheduler_steps'][1]['scheduled'][0]['recompute_tokens'] = 1
        with self.assertRaises(ValueError):
            validate_fit_window(changed, 0, 1, 4)


class TestBridgeRequirement(unittest.TestCase):
    def test_fixed_path_victim_estimate_matches_sealed_observation(self):
        b = BridgeRequirement(
            exhaustion_step=807,
            first_release_step=1025,
            drain_rate=2.0,
            victim_yield_blocks=237,
        )
        self.assertEqual(b.bridge_steps, 218)
        self.assertEqual(b.bridge_blocks, 436)
        self.assertEqual(b.fixed_path_victim_estimate, 2)

    def test_no_bridge_needed_when_release_precedes_exhaustion(self):
        b = BridgeRequirement(1025, 807, 2.0, 237)
        self.assertEqual(b.bridge_steps, 0)
        self.assertEqual(b.fixed_path_victim_estimate, 0)

    def test_legal_growth_change_breaks_cross_policy_victim_bound(self):
        # Structural counterexample, not a runtime/performance simulation.
        # Both policies begin with the same two requests: 1 cached prompt
        # token and 1 emitted token each. Pool=4; each owes 2 more outputs.
        from resource_transition_model import Event, Request, State, reduce_events
        initial = State(tuple(Request(rid, prompt_tokens=1, status='decode',
            prompt_processed=1, output_observed=1, cached_tokens=1,
            cached_blocks=1, ever_admitted=True) for rid in ('A', 'B')),
            usable_kv_blocks=4, block_size=1, cap_target=2)
        grow_all = reduce_events(initial, [Event('decode', rid, 1, cached_tokens_after=2)
                                          for rid in ('A', 'B')])
        with self.assertRaises(ValueError):
            reduce_events(grow_all, [Event('decode', 'A', 2, cached_tokens_after=3)])
        # Hold B's KV while advancing A; then let B finish. No eviction or
        # recomputation, same initial state and declared work, first release=2.
        state, peak, releases = initial, initial.used_kv_blocks, []
        for event in (Event('decode', 'A', 1, cached_tokens_after=2),
                      Event('decode', 'A', 2, cached_tokens_after=3),
                      Event('complete', 'A', 2),
                      Event('decode', 'B', 3, cached_tokens_after=2),
                      Event('decode', 'B', 4, cached_tokens_after=3),
                      Event('complete', 'B', 4)):
            state = reduce_events(state, [event])
            peak = max(peak, state.used_kv_blocks)
            if event.kind == 'complete': releases.append(event.at_ms)
        self.assertEqual(peak, 4)
        self.assertEqual(releases[0], 2)
        self.assertTrue(all(r.status == 'completed' and r.output_observed == 3 for r in state.requests))
        self.assertEqual(BridgeRequirement(1, 2, 2, 2).fixed_path_victim_estimate, 1)
        self.assertTrue(StaticFeasibility(2, 1, 3, 1, 4).as_dict()['terminal_co_residency_exceeds_pool'])


class TestVictimStallLaw(unittest.TestCase):
    # Sealed completion times of the first two natural completions.
    RELEASES = [20.912855259142816, 21.03345991577953, 21.12306389864534]

    # Tolerance is stated in absolute milliseconds, not relative error: the
    # preemption timestamp is the victim's last output receipt, which precedes
    # the engine's preempt decision by a sub-millisecond amount.  Claiming
    # relative exactness would overstate what the ledger resolves.
    TOL_S = 1e-3

    def test_first_restored_victim_waits_for_first_completion(self):
        # article-0003571: preempted at 19.000084, observed wait 1.9132248 s.
        pred = predict_victim_stall(19.000083608552814, self.RELEASES, 0, 0.0)
        self.assertLess(abs(pred - 1.9132247641682625), self.TOL_S)

    def test_second_restored_victim_waits_for_second_completion(self):
        # article-0003640: preempted at 16.560300, observed wait 4.4736173 s.
        # Restore rank 1: the first freed slot went to the later-preempted
        # victim, so this one waits for the *second* natural completion.
        pred = predict_victim_stall(16.560300201177597, self.RELEASES, 1, 0.0)
        self.assertLess(abs(pred - 4.473617320880294), self.TOL_S)

    def test_wrong_restore_rank_is_off_by_a_whole_completion_gap(self):
        # Guards against the law fitting by accident: rank 0 for this victim
        # mispredicts by more than 100 ms.
        pred = predict_victim_stall(16.560300201177597, self.RELEASES, 0, 0.0)
        self.assertGreater(abs(pred - 4.473617320880294), 0.1)

    def test_recompute_is_added_not_folded_into_wait(self):
        wait_only = predict_victim_stall(16.560300201177597, self.RELEASES, 1, 0.0)
        with_rc = predict_victim_stall(16.560300201177597, self.RELEASES, 1, 0.117)
        self.assertAlmostEqual(with_rc - wait_only, 0.117)

    def test_rejects_release_before_preemption(self):
        # A completion that happened before the preemption cannot restore it.
        with self.assertRaises(ValueError):
            predict_victim_stall(25.0, self.RELEASES, 0, 0.0)


class TestStaggerFeasibility(unittest.TestCase):
    SEALED = StaggerFeasibility(32, 3072, 1024, 16, 7671)

    def test_dynamic_range_is_only_the_decode_share(self):
        self.assertEqual(self.SEALED.prefill_blocks, 192)
        self.assertEqual(self.SEALED.full_blocks, 256)
        self.assertEqual(self.SEALED.decode_blocks, 64)
        self.assertAlmostEqual(self.SEALED.prefill_share, 0.75)

    def test_small_deferral_sets_cannot_cover_the_sealed_deficit(self):
        # 521 blocks of deficit, at most 64 blocks recoverable per deferred
        # request: three deferred requests are arithmetically insufficient.
        self.assertIsNone(self.SEALED.min_delay_steps(3))
        self.assertEqual(self.SEALED.min_deferred_requests, 9)

    def test_required_delay_is_most_of_the_decode_horizon(self):
        # Nine deferred requests still need ~926 steps of delay each, which is
        # the same order as the 1024-step output length.
        self.assertEqual(self.SEALED.min_delay_steps(9), 928)

    def test_decode_dominated_workload_has_real_stagger_leverage(self):
        # Short prompt, long output: the footprint is created during decode,
        # so deferring admission genuinely lowers peak occupancy.
        s = StaggerFeasibility(32, 128, 4096, 16, 7671)
        self.assertLess(s.prefill_share, 0.05)
        self.assertTrue(s.stagger_helps)
        self.assertLess(s.min_deferred_requests, self.SEALED.min_deferred_requests)

    def test_feasible_cohort_needs_no_deferral(self):
        s = StaggerFeasibility(32, 3072, 1024, 16, 8425)
        self.assertEqual(s.deficit_blocks, 0)
        self.assertEqual(s.min_deferred_requests, 0)
        self.assertTrue(s.stagger_helps)


class TestDeficitLedger(unittest.TestCase):
    def _sealed_ledger(self):
        led = DeficitLedger()
        led.add_victim("article-0003640", 16.5603, 4.473617320880294, 0.11702534556388855, 237)
        led.add_victim("article-0003571", 19.0001, 1.9132247641682625, 0.12015154305845499, 245)
        return led

    def test_buckets_are_mutually_exclusive_and_sum_to_stall(self):
        led = self._sealed_ledger()
        self.assertAlmostEqual(
            led.total_stall_s, led.total_wait_s + led.total_recompute_s
        )

    def test_wait_dominates_recompute_in_the_sealed_episode(self):
        led = self._sealed_ledger()
        self.assertGreater(led.wait_share, 0.96)

    def test_released_blocks_cover_the_bridge_requirement(self):
        led = self._sealed_ledger()
        bridge = BridgeRequirement(807, 1025, 2.0, 237)
        self.assertGreaterEqual(led.released_blocks, bridge.bridge_blocks)

    def test_duplicate_victim_is_rejected(self):
        led = self._sealed_ledger()
        with self.assertRaises(ValueError):
            led.add_victim("article-0003640", 16.5603, 1.0, 0.1, 237)

    def test_negative_components_are_rejected(self):
        led = DeficitLedger()
        with self.assertRaises(ValueError):
            led.add_victim("x", 1.0, -0.1, 0.1, 10)
        with self.assertRaises(ValueError):
            led.add_victim("x", 1.0, 0.1, 0.1, 0)


class TestRequestBlocks(unittest.TestCase):
    def test_ceiling_semantics(self):
        self.assertEqual(request_blocks(3072, 16), 192)
        self.assertEqual(request_blocks(3073, 16), 193)
        self.assertEqual(request_blocks(0, 16), 0)

    def test_rejects_bad_arguments(self):
        with self.assertRaises(ValueError):
            request_blocks(-1, 16)
        with self.assertRaises(ValueError):
            request_blocks(10, 0)


if __name__ == "__main__":
    unittest.main()
