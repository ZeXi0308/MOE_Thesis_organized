import unittest
from dataclasses import replace

from service_window_model import (Action, Node, Request, State, critical_span,
                                  maximum_age, qualify, retention_efficiency)


def support():
    state = State(Request('target', 32, 16, 1, 3),
                  (Request('peer', 31, 30, 2, 4),), 16, 4, 100, 120)
    mixed = Action('mixed-4', 4, (Node('queue', 1), Node('restore', 5, ('queue',))),
                   1, ('peer',), (('peer', (2, 4)),), recompute_tokens=16,
                   prospective_tax_ms=4)
    return state, mixed


def check(state, action, gap=10):
    return qualify(state, action, gap_budget_ms=gap, tax_budget_ms_per_output=1)


class ServiceWindowTests(unittest.TestCase):
    def test_actions_separate_amortization_and_peer_urgency(self):
        s, a = support()
        self.assertTrue(check(s, a)['eligible'])
        self.assertFalse(check(s, replace(a, tokens=1))['amortized'])
        unknown = check(s, replace(a, prospective_tax_ms=None))
        self.assertIsNone(unknown['amortized'])
        self.assertFalse(unknown['eligible'])
        solo = replace(a, co_batch=(), recovery_peer_outputs=())
        self.assertTrue(check(s, solo)['resource_ok'])
        self.assertFalse(check(s, solo)['urgency_ok'])

    def test_incremental_kv_caps_window_even_when_first_output_fits(self):
        s, a = support()
        s = replace(s, free_gpu_blocks=2)
        a = replace(a, recovery_peer_outputs=())
        self.assertTrue(check(s, replace(a, tokens=3))['resource_ok'])
        self.assertFalse(check(s, a)['resource_ok'])
        self.assertFalse(check(s, replace(a, tokens=3))['amortized'])

    def test_recovery_peak_and_host_validity_are_real_constraints(self):
        s, a = support()
        self.assertFalse(check(s, replace(a, restore_scratch_blocks=4))['resource_ok'])
        offload = replace(a, transfer_prefix_tokens=31, recompute_tokens=1)
        self.assertFalse(check(s, offload)['source_ok'])
        s = replace(s, target=replace(s.target, host_valid=True, host_prefix_tokens=31))
        self.assertTrue(check(s, offload)['source_ok'])
        self.assertFalse(check(s, replace(offload, staging_bytes=21))['resource_ok'])
        self.assertFalse(check(s, replace(offload, recompute_tokens=0))['source_ok'])

    def test_overlap_uses_dependency_path_not_sum(self):
        nodes = (Node('queue', 1), Node('dma', 4, ('queue',)),
                 Node('independent_compute', 5, ('queue',)),
                 Node('join', 1, ('dma', 'independent_compute')))
        self.assertEqual(critical_span(nodes), 7)
        with self.assertRaises(ValueError):
            critical_span((Node('future', 1, ('unknown',)),))

    def test_unrestored_peer_or_no_pending_target_cannot_mint_output(self):
        s, a = support()
        s = replace(s, peers=(replace(s.peers[0], computed_tokens=16),))
        with self.assertRaises(ValueError):
            check(s, a)
        with self.assertRaises(ValueError):
            check(s, replace(a, recovery_peer_outputs=()))
        s, a = support()
        s = replace(s, target=replace(s.target, computed_tokens=32, gpu_blocks=2))
        with self.assertRaises(ValueError):
            check(s, a)

    def test_internal_progress_does_not_reset_age_or_force_retention(self):
        self.assertEqual(maximum_age(8, [], 3), 11)
        self.assertEqual(maximum_age(8, [2], 3), 10)
        s, a = support()
        # Same paid history, different *remaining* work. No spent-cost input exists.
        s = replace(s, target=replace(s.target, computed_tokens=31, gpu_blocks=2))
        a = replace(a, tokens=1, recompute_tokens=1, prospective_tax_ms=0)
        self.assertEqual(check(s, a)['min_opportunity_tokens'], 1)
        self.assertEqual(retention_efficiency(3, 1, (0, 0), (10, 10))[
            'efficiency_preference'], 'SWITCH')
        self.assertEqual(retention_efficiency(3, 1, (1, 1), (10, 10))[
            'efficiency_preference'], 'KEEP')
        self.assertEqual(retention_efficiency(3, 1, (0, 1), (10, 10))[
            'efficiency_preference'], 'UNRESOLVED')

    def test_pending_native_load_keeps_blocks_but_does_not_certify_output(self):
        # Measured request3571: history3307, staged prefix3296, held206 blocks.
        # Times below are synthetic; no transfer latency or marginal tax is inferred.
        r = Request('target', 3307, 0, 206, 0, 3296, True, True)
        s = State(r, (), 16, 144, 432013312, 16*2**30)
        a = Action('pending-load', 1, (Node('hypothetical-ready-and-output', 1),),
                   1, transfer_prefix_tokens=3296, recompute_tokens=11)
        pending = check(s, a)
        self.assertTrue(pending['resource_ok'])
        self.assertEqual(pending['peak_extra_blocks'], 1)
        self.assertFalse(pending['readiness_ok'])
        self.assertFalse(pending['eligible'])
        self.assertEqual(pending['deferred_reason'], 'WAIT_FOR_NATIVE_LOAD_PROMOTION')
        # Same numeric native count becomes usable only after native promotion.
        s = replace(s, target=replace(r, computed_tokens=3296, waiting_for_remote_kv=False))
        ready = check(s, replace(a, transfer_prefix_tokens=0))
        self.assertTrue(ready['readiness_ok'] and ready['source_ok'])
        self.assertIsNone(ready['amortized'])
        self.assertFalse(ready['eligible'])  # Ready does not supply a marginal cost.

    def test_pending_native_counter_cannot_be_imported_as_valid_prefix(self):
        s, a = support()
        s = replace(s, target=replace(s.target, waiting_for_remote_kv=True))
        with self.assertRaisesRegex(ValueError, 'not a usable GPU prefix'):
            check(s, a)
        s, a = support()
        s = replace(s, peers=(Request('peer', 1, 0, 1, 4, waiting_for_remote_kv=True),))
        with self.assertRaisesRegex(ValueError, 'only serves resident peers'):
            check(s, a)

    def test_declared_caps_cannot_mint_target_or_peer_outputs(self):
        s, a = support()
        capped = replace(s, target=replace(s.target, remaining_output_cap=3))
        with self.assertRaisesRegex(ValueError, 'remaining output cap'):
            check(capped, a)
        # Two recovery outputs plus three later peer outputs exceed its cap4.
        capped = replace(s, peers=(replace(s.peers[0], remaining_output_cap=4),))
        with self.assertRaisesRegex(ValueError, 'remaining output cap'):
            check(capped, a)
        with self.assertRaisesRegex(ValueError, 'context limit'):
            check(replace(s, max_context_tokens=33), a)
        # KV for the returned token is still pending, but native check_stop
        # counts that token toward the length cap at the engine-return boundary.
        only = State(Request('target', 32, 31, 2, 0), (), 16, 2, 0, 0,
                     max_context_tokens=33)
        one = Action('last-output', 1, (Node('decode', 1),), 1,
                     recompute_tokens=1, prospective_tax_ms=0)
        self.assertTrue(check(only, one)['eligible'])
        with self.assertRaisesRegex(ValueError, 'context limit'):
            check(replace(only, max_context_tokens=32), one)

    def test_peer_cap_completion_stops_age_but_does_not_lend_its_blocks(self):
        s, a = support()
        s = replace(s, peers=(replace(s.peers[0], remaining_output_cap=1),),
                    free_gpu_blocks=1)
        a = replace(a, co_batch=(), recovery_peer_outputs=(('peer', (2,)),), batch_ms=5)
        result = check(s, a)
        self.assertEqual(result['maximum_ages_ms']['peer'], 6)
        # Target needs two new blocks. Peer completion may enable a different
        # phased action, but this conservative no-release action cannot borrow
        # its two blocks before the corresponding engine return.
        self.assertEqual(result['peak_extra_blocks'], 2)
        self.assertFalse(result['resource_ok'])


if __name__ == '__main__':
    unittest.main()
