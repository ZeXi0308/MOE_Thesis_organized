"""One targeted regression for the existing release shadow made online."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'pkg'))
import staged_store_rotation as adapter
from test_equal_release_budget_cpu import real_pick
from test_partial_restore_once_cpu import native_suffix_action


def fixture(rule):
    values=real_pick(rule);ns=values[1]
    state=lambda mode:dict(mode=mode,decision_count=0,proposal_count=0,
                          action_request_count=0,fallback_counts={},selector_wall_s=0.)
    ns['remaining_budget_state']=state('ACTIVE' if rule=='remaining_budget' else 'SHADOW')
    ns['max_release_state']=state('ACTIVE' if rule=='max_release' else 'SHADOW')
    return values


class MaxReleaseTests(unittest.TestCase):
    def test_refcounts_full_suffix_guards_native_lifecycle_and_abba(self):
        blocks=[NS(block_id=i,ref_cnt=ref,is_null=False) for i,ref in enumerate((1,2,3))]
        before=deepcopy([vars(b) for b in blocks])
        observed=adapter._native_release_state(blocks,NS(blocks=blocks))
        self.assertEqual((observed['immediate_releasable_blocks'],observed['shared_blocks']),(1,2))
        self.assertEqual([vars(b) for b in blocks],before)
        choose=adapter._max_release_shadow_choice
        rows=[dict(index=1,held_blocks=2,immediate_releasable_blocks=2,qualified=False),
              dict(index=2,held_blocks=3,**observed)]
        self.assertEqual(choose(rows,False),(1,None))  # More held pages do not imply more release.
        self.assertEqual(choose(rows,True),(2,'ACTIVE_PROTECTION_OR_PHASE'))
        rows[0]['immediate_releasable_blocks']=None
        self.assertEqual(choose(rows,False),(2,'UNKNOWN_PHYSICAL_RELEASE'))
        rows[0]['immediate_releasable_blocks']=1
        self.assertEqual(choose(rows,False),(2,None))  # Latest/tail tie.

        execute=native_suffix_action()
        for candidate_index in (1,2):
            for rule in ('max_release','remaining_budget'):
                scheduler,ns,requests,calls,waiting,freed,encoder,discarded=fixture(rule)
                for req in requests[1:]:
                    for block in ns['owned'][req.request_id]:block.ref_cnt=2
                candidate=requests[candidate_index]
                candidate.num_output_tokens=1;candidate.prompt_tokens=100;candidate.num_computed_tokens=50
                ns['owned'][candidate.request_id]=ns['owned'][candidate.request_id][:100]
                for block in ns['owned'][candidate.request_id]:block.ref_cnt=1
                budget=requests[3]
                budget.num_output_tokens=0;budget.prompt_tokens=100;budget.num_computed_tokens=50
                ns['owned'][budget.request_id]=ns['owned'][budget.request_id][:67]
                for block in ns['owned'][budget.request_id]:block.ref_cnt=1
                ns['cs']._req_status[candidate.request_id].transfer_jobs={'store-job'}
                ns['cs']._jobs['store-job']=NS(req_id=candidate.request_id,is_store=True,pending_count=1)
                prefix=[requests[0]];stamp=object()
                selected=execute(scheduler,1,prefix,stamp)
                self.assertIs(selected,candidate if rule=='max_release' else budget)
                self.assertEqual(prefix,[requests[0]]);self.assertNotIn(selected,scheduler.running)
                self.assertEqual(calls,[(selected,stamp)])
                self.assertEqual((selected.status,selected.num_computed_tokens,
                                  selected.spec_token_ids,selected.num_preemptions),('PREEMPTED',0,[],1))
                self.assertIs(waiting[0],selected);self.assertEqual(freed,[selected])
                self.assertEqual(encoder,[selected]);self.assertEqual(discarded,[selected])
                self.assertIn(selected.request_id,scheduler.reset_preempted_req_ids)
                self.assertNotIn(selected.request_id,ns['residence'])
                decision=ns['data']['victim_decisions'][0];probe=decision['max_release_shadow']
                self.assertEqual([r['index'] for r in decision['candidates']],[1,2,3,4])
                self.assertEqual(decision['failed_request'],requests[1].request_id)
                self.assertTrue(decision['fallback_unknown'])  # Legacy decode qualification does not gate release.
                self.assertEqual(probe['proposed_request'],candidate.request_id)
                self.assertEqual(probe['returned_request'],selected.request_id)
                self.assertEqual((probe['tail_releasable_blocks'],probe['proposed_releasable_blocks']),(0,100))
                self.assertEqual(probe['mode'],'ACTIVE' if rule=='max_release' else 'SHADOW')
                self.assertEqual(probe['action_requested'],rule=='max_release')
                self.assertEqual(ns['max_release_state']['action_request_count'],int(rule=='max_release'))
                self.assertEqual((ns['max_release_state']['decision_count'],ns['max_release_state']['proposal_count']),(1,1))
                remaining=decision['remaining_budget']
                self.assertEqual(remaining['proposed_request'],budget.request_id)
                self.assertEqual(remaining['mode'],'ACTIVE' if rule=='remaining_budget' else 'SHADOW')
                self.assertEqual(remaining['action_requested'],rule=='remaining_budget')
                for shadow in ['equal_held_once','partial_restore_once','equal_release_host_once','equal_release_budget']:
                    self.assertIsNotNone(decision[shadow])
        for guard,reason in [('phase','ACTIVE_PROTECTION_OR_PHASE'),('unknown','UNKNOWN_PHYSICAL_RELEASE')]:
            scheduler,ns,requests,*_=fixture('max_release')
            if guard=='phase':ns['phase']=object()
            else:ns['owned'][requests[1].request_id][0].ref_cnt=None
            self.assertEqual(scheduler._rotation_pick_victim(1,[requests[0]]),4)
            self.assertEqual(ns['data']['victim_decisions'][0]['max_release_shadow']['fallback'],reason)
            self.assertEqual(ns['max_release_state']['fallback_counts'],{reason:1})
        for name in ['rotation_native.py','request_measurement.py']:
            self.assertEqual((HERE/'pkg'/name).read_bytes(),
                (HERE.parent/'candidate_native_remaining_budget_r01/pkg'/name).read_bytes())

        # Exercise only pure configuration; no lock, process, CUDA, or files are created.
        sys.path.insert(0,str(HERE.parent))
        spec=importlib.util.spec_from_file_location('max_release_controller',HERE.parent/'run_native_max_release_group.py')
        group=importlib.util.module_from_spec(spec);spec.loader.exec_module(group)
        plan=json.loads((HERE.parent/'plan-native-remaining-budget-westd53005-20261008-r02.json').read_text())
        plan.update(kind='PRO6000_NATIVE_MAX_RELEASE',experiment_role='MAX_RELEASE_INTERVENTION')
        rules=['remaining_budget','max_release','max_release','remaining_budget']
        for cell,rule in zip(plan['cells'],rules):cell['native_victim_rule']=rule
        original_env=group.base.private_env
        try:
            group.configure(plan)
            wrong=deepcopy(plan);wrong['cells'][0]['native_victim_rule']='tail'
            with self.assertRaises(ValueError):group.configure(wrong)
        finally:group.base.private_env=original_env


if __name__=='__main__':unittest.main()
