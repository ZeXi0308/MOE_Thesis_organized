"""Checks for execution-specific state, output boundary and resource accounting."""
import unittest
from dataclasses import replace
from functools import cache
from itertools import product
from recovery_execution_share import (Protection, Request, State, allocate,
    decode_release_envelope, allocate_completion_bridge, decode_service_envelope)


class ExecutionShareTest(unittest.TestCase):
    def test_service_bound_against_independent_output_enumeration(self):
        for cs, horizon, free in product(product(range(1, 5), repeat=3), range(1, 5), range(5)):
            owned = tuple((c+1)//2 for c in cs)
            rows = tuple(Request(str(i), c+1, c, a, 1, horizon)
                         for i, (c, a) in enumerate(zip(cs, owned)))
            state = State(rows[0], rows[1:], free, block_size=2, token_budget=3)
            result = decode_service_envelope(state, horizon, ['0'])
            # Enumerate final output vectors independently of marginal-block ranking.
            feasible = []
            for other in product(range(horizon+1), repeat=2):
                ys = (horizon,) + other
                need = sum(max(0, (c+y+1)//2-a) for c,y,a in zip(cs,ys,owned))
                if need <= free:
                    feasible.append((sum(ys), sum(y == 0 for y in other)))
            if not feasible:
                self.assertEqual(result['status'], 'UNFUNDED_REQUIRED_SERVICE')
            else:
                self.assertEqual(result['maximum_total_output_opportunities'], max(x[0] for x in feasible))
                self.assertEqual(result['minimum_zero_service_peers'], min(x[1] for x in feasible))

    def test_service_bound_rejects_early_cap_and_compute_contention(self):
        state = State(Request('t', 2, 1, 1, 1, 4),
                      (Request('p', 2, 1, 1, 1, 1),), 3, block_size=2)
        with self.assertRaises(ValueError):
            decode_service_envelope(state, 2, ['t'])
        with self.assertRaises(ValueError):
            decode_service_envelope(replace(state, token_budget=1), 1, ['t'])

    def test_release_bound_against_all_tiny_legal_orders(self):
        @cache
        def explore(cs, owned, caps, free):
            reachable, longest = False, 0
            for i in range(2):
                need = max(0, (cs[i]+2)//2-owned[i])
                if need > free:
                    continue
                if caps[i] == 1:
                    reachable = True
                    continue
                c, a, u = list(cs), list(owned), list(caps)
                c[i] += 1; a[i] += need; u[i] -= 1
                can, more = explore(tuple(c), tuple(a), tuple(u), free-need)
                reachable |= can
                longest = max(longest, 1+more)
            return reachable, longest
        for c1, c2, u1, u2, free in product(range(1, 5), range(1, 5),
                                          range(1, 5), range(1, 5), range(2)):
            owned = ((c1+1)//2, (c2+1)//2)
            s = State(Request('t', c1+1, c1, owned[0], 1, u1),
                      (Request('p', c2+1, c2, owned[1], 1, u2),), free, block_size=2)
            bound = decode_release_envelope(s)
            can, longest = explore((c1,c2), owned, (u1,u2), free)
            self.assertEqual(bound['no_early_eos_first_completion_possible'], can)
            if not can:
                self.assertEqual(bound['no_early_eos_total_output_opportunities_before_exhaustion'], longest)

    def test_bridge_reserves_both_without_borrowing_completion(self):
        s = State(Request('t', 2, 1, 1, 1, 10),
                  (Request('busy', 3, 2, 1, 1, 10),
                   Request('end', 2, 1, 1, 1, 2)), 2, block_size=2)
        p = allocate_completion_bridge(s)
        self.assertEqual(p['finisher_id'], 'end')
        self.assertEqual(p['held'], {'busy':'COMPLETION_BRIDGE_BLOCKS'})
        self.assertEqual(p['reserved_blocks'], 2)
        self.assertEqual(p['free_after_allocations'], 2)
        self.assertIn('busy', allocate(s)['scheduled'])
        next_state = replace(s, target=replace(s.target, history=3, computed=2, outputs=2, remaining_cap=9),
            peers=(s.peers[0], replace(s.peers[1], history=3, computed=2, outputs=2, remaining_cap=1)))
        last = allocate_completion_bridge(next_state, 'end')
        self.assertEqual(last['scheduled'], {'t':1, 'end':1})
        self.assertEqual(last['free_after_allocations'], 0)
        self.assertEqual(last['remaining_cap_calls'], 1)  # No completion blocks credited yet.

    def test_integer_boundary_and_peer_cost(self):
        t = Request('t', 3070, 0, 0, 510, 514)
        peers = tuple(Request(str(i), 33, 32, 3, 1, 1023) for i in range(30))
        s = State(t, peers, 192)
        old, new = allocate(s), allocate(s, 'preserve_calls')
        self.assertEqual((old['target_positions'], new['target_positions']), (994, 1022))
        self.assertEqual(new['peer_output_opportunities'], 2)
        self.assertEqual(len(new['held']), 28)  # Explicit service transferred.
        self.assertEqual(new['compute_only_min_calls'], 3)
        last = replace(s, target=replace(t, computed=2800, allocated=175), free_blocks=17)
        self.assertEqual(allocate(last, 'restore_first')['peer_output_opportunities'], 30)

    def test_escrow_never_borrows_later_completion(self):
        s = State(Request('t', 17, 0, 0, 1, 5),
                  (Request('p', 17, 16, 1, 1, 1),), 2, token_budget=8)
        p = allocate(s)
        self.assertEqual(p['held'], {'p': 'KV_ESCROW'})
        self.assertGreaterEqual(p['free_after_allocations'], p['remaining_escrow_blocks'])
        self.assertEqual(allocate(replace(s, free_blocks=1))['status'],
                         'UNFUNDED_RETURN_TO_RESOURCE_LAYER')

    def test_internal_work_does_not_release_output_obligation(self):
        s = State(Request('t', 17, 17, 2, 1, 5), (), 0)
        self.assertEqual(allocate(s)['status'], 'WAIT_OUTPUT_RETURN')
        obligation = Protection('t', 1)
        self.assertFalse(obligation.released(observed_outputs=1))
        self.assertTrue(obligation.released(observed_outputs=2))
        self.assertTrue(obligation.released(observed_outputs=1, completed=True))
        self.assertEqual(allocate(replace(s, target=replace(s.target,
                         remote_pending=True)))['status'], 'WAIT_NATIVE_READY')

    def test_budget_and_kv_over_boundary_states(self):
        # Covers ceil boundaries and unavailable peer KV, not future workloads.
        for R in (1, 7, 8, 9, 15, 16, 17, 23, 24):
            t = Request('t', R + 5, 5, 1, 1, 100)
            peers = (Request('p', 17, 16, 1, 1, 100),)
            for free in range(5):
                s = State(t, peers, free, token_budget=8)
                p = allocate(s, 'preserve_calls')
                if p['status'] == 'READY':
                    self.assertLessEqual(sum(p['scheduled'].values()), 8)
                    self.assertGreaterEqual(p['free_after_allocations'], p['remaining_escrow_blocks'])
                    remaining = R - p['target_positions']
                    self.assertLessEqual((remaining + 7) // 8, p['compute_only_min_calls'] - 1)


if __name__ == '__main__':
    unittest.main()
