"""Synthetic CPU fixtures: state, causal cutoff and accounting; no performance evidence."""
from dataclasses import FrozenInstanceError, replace
import unittest

from resource_transition_model import (
    EstimatedDemand, Event, GroupedSlotState, Request, State, blocks_for, enumerate_actions,
    estimated_unique_load_bytes, exposed_transfer_seconds, lru_dry_run,
    grouped_slot_dry_run, reduce_events, uniform_expected_union, unique_load_bytes,
)


class ResourceTransitions(unittest.TestCase):
    def initial(self, pool=8, cap=2):
        return State((Request("a", 5), Request("b", 3), Request("future", 2, 100)),
                     usable_kv_blocks=pool, block_size=4, cap_target=cap)

    def decoding(self):
        return reduce_events(self.initial(), (
            Event("admit", "a", 0), Event("prefill", "a", 1, tokens=5),
            Event("decode", "a", 1, cached_tokens_after=5),
        ))

    def test_prefill_rounding_and_first_output_are_separate(self):
        state = self.decoding()
        self.assertEqual((state.requests[0].prompt_processed, state.requests[0].output_observed), (5, 1))
        self.assertEqual((state.requests[0].cached_tokens, state.used_kv_blocks), (5, 2))
        self.assertEqual([blocks_for(n, 4) for n in (0, 1, 4, 5)], [0, 1, 1, 2])

    def test_cap_lowering_never_evicts_or_skips_existing_decode(self):
        before = self.decoding()
        lowered = replace(before, cap_target=0)
        actions = enumerate_actions(lowered, chunk_options=(0, 2))
        self.assertEqual(lowered.requests, before.requests)
        self.assertTrue(actions)
        self.assertTrue(all(a.admissions == () and a.decode_requests == ("a",) for a in actions))

    def test_actions_use_actual_pool_and_preserve_all_decodes(self):
        state = replace(self.decoding(), usable_kv_blocks=2)
        actions = enumerate_actions(state, chunk_options=(0, 2))
        self.assertTrue(actions)
        self.assertTrue(all(not a.admissions for a in actions))
        self.assertTrue(all(a.decode_requests == ("a",) for a in actions))
        with self.assertRaisesRegex(ValueError, "actual KV pool"):
            reduce_events(state, (Event("admit", "b", 2), Event("prefill", "b", 3, tokens=1)))

    def test_infeasible_decode_requires_native_resolution(self):
        request = Request("a", 4, status="decode", prompt_processed=4,
                          output_observed=1, cached_tokens=4, cached_blocks=1, ever_admitted=True)
        state = State((request,), usable_kv_blocks=1, block_size=4, cap_target=0)
        self.assertEqual(enumerate_actions(state), ())

    def test_prefills_share_actual_token_budget_with_existing_decodes(self):
        # All chunks fit KV, but existing prefill b and newcomer c share a
        # three-token budget with the already decoding a.
        state = self.decoding()
        state = replace(state, requests=state.requests + (Request("c", 3),),
                        cap_target=3, token_budget=3)
        state = reduce_events(state, (Event("admit", "b", 2),))
        actions = enumerate_actions(state, chunk_options=(0, 1, 2))
        self.assertTrue(any(a.admissions == ("c",) and dict(a.prefill_chunks) == {"b": 1, "c": 1}
                            for a in actions))
        self.assertTrue(any(dict(a.prefill_chunks) == {"b": 2} for a in actions))
        self.assertTrue(all(len(a.decode_requests) + sum(c for _, c in a.prefill_chunks) <= 3
                            for a in actions))
        self.assertFalse(any(dict(a.prefill_chunks) == {"b": 2, "c": 2} for a in actions))

    def test_native_preemption_retains_history_and_recompute_does_not_emit(self):
        original = self.decoding()
        preempted = reduce_events(original, (Event("preempt", "a", 2),))
        r = preempted.requests[0]
        self.assertEqual((r.status, r.prompt_processed, r.output_observed, r.cached_blocks),
                         ("recovering", 5, 1, 0))
        self.assertTrue(all("a" not in a.admissions and "a" not in a.decode_requests
                            for a in enumerate_actions(preempted, (0, 2))))
        with self.assertRaises(ValueError):
            reduce_events(preempted, (Event("admit", "a", 3),))
        with self.assertRaises(ValueError):
            reduce_events(preempted, (Event("decode", "a", 3, cached_tokens_after=1),))
        partial = reduce_events(preempted, (Event("recompute", "a", 3, tokens=4),))
        self.assertEqual(partial.requests[0].status, "recovering")
        restored = reduce_events(partial, (Event("recompute", "a", 4, tokens=1, resumed=True),))
        self.assertEqual(restored.requests[0].status, "decode")
        self.assertEqual(restored.requests[0].output_observed, 1)
        self.assertEqual(restored.requests[0].last_token_ms, 1)
        advanced = reduce_events(restored, (Event("decode", "a", 5, cached_tokens_after=6),))
        self.assertEqual(advanced.requests[0].output_observed, 2)
        self.assertEqual(original.requests[0].cached_tokens, 5)

    def test_recovery_requires_full_known_prefix(self):
        state = reduce_events(self.decoding(), (Event("preempt", "a", 2),))
        with self.assertRaisesRegex(ValueError, "required history"):
            reduce_events(state, (Event("recompute", "a", 3, tokens=1, resumed=True),))

    def test_completion_releases_capacity_without_erasing_outputs(self):
        full = replace(self.decoding(), usable_kv_blocks=2)
        completed = reduce_events(full, (Event("complete", "a", 2),))
        self.assertEqual(completed.used_kv_blocks, 0)
        self.assertEqual(completed.requests[0].output_observed, 1)
        self.assertTrue(any(a.admissions == ("b",) for a in enumerate_actions(completed, (0, 3))))

    def test_causal_arrival_order_identity_and_branch_independence(self):
        source = self.initial()
        self.assertFalse(any("future" in a.admissions for a in enumerate_actions(source)))
        for event in (Event("admit", "future", 1), Event("admit", "missing", 1)):
            with self.assertRaises(ValueError):
                reduce_events(source, (event,))
        a = reduce_events(source, (Event("admit", "a", 1), Event("prefill", "a", 2, tokens=2)))
        b = reduce_events(source, (Event("admit", "b", 1), Event("prefill", "b", 2, tokens=3)))
        self.assertEqual(source.requests[0].status, "waiting")
        self.assertEqual(a.requests[1].status, "waiting")
        self.assertEqual(b.requests[0].status, "waiting")
        with self.assertRaises(FrozenInstanceError):
            a.requests[0].cached_tokens = 100
        with self.assertRaises(ValueError):
            reduce_events(a, (Event("complete", "a", 1),))


class ExpertAccounting(unittest.TestCase):
    def test_group_needed_protection_avoids_sequential_lru_false_reload(self):
        initial = GroupedSlotState((0, 1), (1, 2), 2)
        grouped = grouped_slot_dry_run(initial, ((2, 0),), {0: 10, 1: 10, 2: 10})
        sequential = lru_dry_run(((0, 2), (0, 0)), ((0, 0), (0, 1)), 20,
                                 {(0, 0): 10, (0, 1): 10, (0, 2): 10})
        self.assertEqual(grouped.groups[0].loaded_experts, (2,))
        self.assertEqual(grouped.groups[0].evicted_experts, (1,))
        self.assertEqual((grouped.total_load_bytes, sequential.total_load_bytes), (10, 20))
        self.assertEqual(grouped.final_state, GroupedSlotState((0, 2), (3, 3), 3))
        self.assertEqual(initial, GroupedSlotState((0, 1), (1, 2), 2))

    def test_group_equal_lru_ticks_evict_by_slot_then_pop_free_slots(self):
        initial = GroupedSlotState((9, 8, 7), (4, 4, 4), 4)
        trace = grouped_slot_dry_run(initial, ((5, 6), (5, 6)), {5: 10, 6: 20})
        self.assertEqual(trace.groups[0].evicted_experts, (9, 8))
        self.assertEqual(trace.groups[0].loaded_experts, (5, 6))
        self.assertEqual(trace.groups[0].loaded_slots, (1, 0))
        self.assertEqual(trace.groups[1].load_bytes, 0)
        self.assertEqual(trace.final_state, GroupedSlotState((6, 5, 7), (6, 6, 4), 6))

    def test_group_manual_load_bytes_close_with_free_slots_hits_and_reload(self):
        initial = GroupedSlotState((-1, -1), (0, 0), 0)
        trace = grouped_slot_dry_run(initial, ((0, 1), (1,), (2, 1), (0, 2)),
                                     {0: 10, 1: 20, 2: 30})
        self.assertEqual([g.load_bytes for g in trace.groups], [30, 0, 30, 10])
        self.assertEqual(trace.groups[0].loaded_slots, (1, 0))
        self.assertEqual([g.evicted_experts for g in trace.groups], [(), (), (0,), (1,)])
        self.assertEqual(trace.total_load_bytes, 70)
        self.assertEqual(trace.final_state, GroupedSlotState((0, 2), (4, 4), 4))

    def test_shared_union_deduplicates_requests_but_not_layers(self):
        weights = {(0, 0): 10, (0, 1): 20, (1, 0): 30}
        routes = (((0, 0), (0, 1)), ((0, 0), (1, 0)))
        self.assertEqual(unique_load_bytes(routes, ((0, 1),), weights), 40)
        self.assertEqual(unique_load_bytes((), (), weights), 0)

    def test_eviction_externality_separates_reload_from_unique_bound(self):
        a, b, c = (0, 0), (0, 1), (0, 2)
        weights = {a: 10, b: 10, c: 10}
        # b and c arriving before existing work for a evict an initially resident a.
        trace = lru_dry_run((b, c, a, b), (a,), 20, weights)
        self.assertEqual((trace.unique_lower_bound_bytes, trace.reload_bytes, trace.total_load_bytes),
                         (20, 20, 40))
        self.assertEqual(trace.final_lru, (a, b))
        self.assertEqual(lru_dry_run((a, a), (a,), 20, weights).total_load_bytes, 0)

    def test_estimate_cutoff_and_uniform_saturation(self):
        demand = EstimatedDemand(frozenset({(0, 1)}), 10)
        self.assertEqual(estimated_unique_load_bytes(demand, 10, (), {(0, 1): 12}), 12)
        with self.assertRaisesRegex(ValueError, "future information"):
            estimated_unique_load_bytes(demand, 9, (), {(0, 1): 12})
        self.assertAlmostEqual(uniform_expected_union(8, 2, 1), 2)
        self.assertAlmostEqual(uniform_expected_union(8, 2, 2), 3.5)
        self.assertEqual(uniform_expected_union(8, 8, 0), 0)
        self.assertEqual(uniform_expected_union(8, 8, 2), 8)
        self.assertAlmostEqual(uniform_expected_union(8, 2, 1000), 8)

    def test_only_exposed_transfer_is_counted(self):
        self.assertAlmostEqual(exposed_transfer_seconds(0.100, 0.075), 0.025)
        self.assertEqual(exposed_transfer_seconds(0.100, 0.100), 0)
        with self.assertRaisesRegex(ValueError, "hidden intersection"):
            exposed_transfer_seconds(0.100, 0.200)
        with self.assertRaises(ValueError):
            exposed_transfer_seconds(float("nan"), 0)


if __name__ == "__main__":
    unittest.main()
