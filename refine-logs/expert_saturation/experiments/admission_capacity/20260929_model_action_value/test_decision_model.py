import sys
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from service_window_model import Request as WindowRequest
from staged_save_contract import RequestState, prepare
from decision_model import (StateSnapshot, count_only_prediction, propose,
                            validate_for_commit)


def snapshot(free=193, need=95, victim_blocks=233):
    owned = tuple(range(1, victim_blocks + 1))
    # These counters have the native pure-decode relation before and after the
    # one preparation decode.  IDs and resources are synthetic CPU fixtures.
    before = RequestState('V', victim_blocks * 16 - 1, 3072,
                          victim_blocks * 16 - 3072, 1024, 'RUNNING', owned)
    target = RequestState('T', 0, 1024,
                          need * 16 - 1024, 1024, 'PREEMPTED', ())
    plan = prepare(858, before, target, free)
    after = replace(before, computed=before.computed + 1, output=before.output + 1)
    return StateSnapshot(
        state_version=17, schedule_step=859, observed_at_ms=1000,
        plan=plan, target=target, victim=after,
        target_window=WindowRequest('T', target.prompt + target.output, 0, 0, 900,
                                    remaining_output_cap=target.max_output - target.output),
        victim_window=WindowRequest('V', after.prompt + after.output,
                                    after.computed, len(owned), 20,
                                    remaining_output_cap=after.max_output - after.output),
        free_gpu_blocks=free, free_sequence_slots=1, block_size_tokens=16,
        running_order=('V',), waiting_order=('T',),
        request_identity_stable=True, all_running_ownership_valid=True,
        victim_physical_ids=owned, victim_refcounts=(1,) * len(owned),
        queue_clear=True, connector_retire_safe=True, target_transfer_clear=True,
        pending_load_ids=(),
        host_used_bytes=None, host_budget_bytes=16 * 2**30,
        new_host_staging_bytes=0,
    )


class CommitDecisionTests(unittest.TestCase):
    def test_direct_action_and_resource_cost_boundary(self):
        s = snapshot()
        p = propose(s)
        self.assertEqual((p.action, p.victim_ids), ('DIRECT_RESUME', ()))
        self.assertEqual((p.cost.target_need_blocks, p.cost.free_after_direct_blocks,
                          p.cost.planned_victim_held_blocks), (95, 98, 233))
        self.assertEqual(p.cost.target_output_age_ms, 900)
        self.assertIsNone(p.cost.exposed_marginal_ms)
        self.assertEqual(validate_for_commit(p, s).action, 'DIRECT_RESUME')
        aged = replace(s, observed_at_ms=1001,
                       target_window=replace(s.target_window, output_age_ms=901),
                       victim_window=replace(s.victim_window, output_age_ms=21))
        self.assertEqual(validate_for_commit(p, aged).action, 'DIRECT_RESUME')

    def test_no_slot_or_insufficient_free_keeps_qualified_old_action(self):
        s = snapshot()
        self.assertEqual(propose(replace(s, free_sequence_slots=0)).action,
                         'KEEP_PLANNED_SWAP')
        constrained = replace(s, free_gpu_blocks=94)
        self.assertEqual(propose(constrained).reason, 'TARGET_NEEDS_PLANNED_VICTIM')
        # No all-peer future-growth reservation: a legal present action stays
        # available even though their next allocation is not forecast here.
        self.assertEqual(propose(s).action, 'DIRECT_RESUME')

    def test_changed_ownership_and_shared_blocks_fall_back(self):
        s = snapshot()
        self.assertEqual(propose(replace(s, victim_physical_ids=s.victim_physical_ids[:-1])).reason,
                         'SHARED_OR_CHANGED_BLOCK_OWNERSHIP')
        refs = (2,) + s.victim_refcounts[1:]
        self.assertEqual(propose(replace(s, victim_refcounts=refs)).action,
                         'KEEP_PLANNED_SWAP')
        self.assertEqual(propose(replace(s, victim_physical_ids=None)).action,
                         'KEEP_PLANNED_SWAP')
        self.assertEqual(propose(replace(s, all_running_ownership_valid=None)).action,
                         'KEEP_PLANNED_SWAP')

    def test_partial_native_load_never_counts_as_executable_prefix(self):
        s = snapshot()
        pending = replace(s.target_window, computed_tokens=0, gpu_blocks=12,
                          waiting_for_remote_kv=True)
        self.assertEqual(propose(replace(s, target_window=pending)).action,
                         'KEEP_PLANNED_SWAP')
        self.assertEqual(propose(replace(s, pending_load_ids=('T',))).action,
                         'KEEP_PLANNED_SWAP')
        self.assertEqual(propose(replace(s, target_transfer_clear=False)).action,
                         'KEEP_PLANNED_SWAP')
        native_pending = replace(s.target, status='WAITING_FOR_REMOTE_KVS')
        self.assertEqual(propose(replace(s, target=native_pending)).action,
                         'CANCEL_PLAN')

    def test_stale_request_eos_and_cancel(self):
        s = snapshot()
        p = propose(s)
        self.assertEqual(validate_for_commit(p, replace(s, free_gpu_blocks=94)).reason,
                         'STALE_SNAPSHOT')
        self.assertEqual(validate_for_commit(p, replace(s, free_gpu_blocks=94)).victim_ids, ())
        self.assertEqual(validate_for_commit(p, replace(s, state_version=18)).reason,
                         'STALE_SNAPSHOT')
        self.assertEqual(propose(replace(s, schedule_step=860)).reason,
                         'CANCEL_STEP_CHANGED')
        self.assertEqual(propose(replace(s, target=None)).victim_ids, ())
        terminal = replace(s.victim, max_output=s.victim.output)
        self.assertEqual(propose(replace(s, victim=terminal)).reason,
                         'REQUEST_AT_DECLARED_CAP')

    def test_unknown_queue_connector_and_host_staging_do_not_mint_savings(self):
        s = snapshot()
        self.assertEqual(propose(replace(s, queue_clear=None)).action,
                         'KEEP_PLANNED_SWAP')
        self.assertEqual(propose(replace(s, connector_retire_safe=False)).action,
                         'KEEP_PLANNED_SWAP')
        self.assertEqual(propose(replace(s, new_host_staging_bytes=None)).reason,
                         'UNKNOWN_NEW_HOST_STAGING')
        self.assertEqual(propose(replace(s, block_size_tokens=None)).action,
                         'KEEP_PLANNED_SWAP')
        self.assertEqual(propose(replace(s, waiting_order=('other', 'T'))).action,
                         'DIRECT_RESUME')  # A may promote the same accepted target.
        self.assertEqual(propose(replace(s, waiting_order=('other',))).action,
                         'KEEP_PLANNED_SWAP')
        self.assertEqual(propose(replace(s, new_host_staging_bytes=1)).reason,
                         'HOST_STAGING_UNFUNDED')
        self.assertEqual(propose(replace(s, host_used_bytes=20,
                                         host_budget_bytes=20,
                                         new_host_staging_bytes=1)).action,
                         'KEEP_PLANNED_SWAP')
        self.assertIsNone(propose(s).cost.exposed_marginal_ms)

    def test_internal_progress_does_not_reset_output_age(self):
        s = snapshot()
        progressed = replace(s, victim=replace(s.victim, computed=s.victim.computed + 1))
        # Without a new receipt, age stays 20 ms.  Native counter drift also
        # invalidates the old one-step plan, rather than buying service credit.
        result = propose(progressed)
        self.assertEqual(result.cost.victim_output_age_ms, 20)
        self.assertEqual(result.action, 'CANCEL_PLAN')

    def test_old_log_counts_are_conditional_not_legal_action_receipts(self):
        d = count_only_prediction(193, 95, 233)
        e = count_only_prediction(277, 106, 184)
        self.assertEqual((d['free_after_direct'], e['free_after_direct']), (98, 171))
        self.assertEqual((d['actual_direct_legal'], e['actual_direct_legal']),
                         ('UNKNOWN', 'UNKNOWN'))
        self.assertEqual(count_only_prediction(94, 95, 233)['direct_kv_funded'], False)


if __name__ == '__main__':
    unittest.main()
