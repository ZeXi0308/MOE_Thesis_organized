"""One targeted selection/native-lifecycle check for the full-suffix baseline."""
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'pkg'))
import staged_store_rotation as adapter
from test_equal_release_budget_cpu import real_pick
from test_partial_restore_once_cpu import native_suffix_action


class RemainingBudgetTests(unittest.TestCase):
    def test_complete_native_suffix_budget_choice_and_original_lifecycle(self):
        choose=adapter._remaining_budget_choice
        rows=[dict(index=4,max_tokens=1024,output_tokens=0),
              dict(index=5,max_tokens=1024,output_tokens=0),
              dict(index=6,max_tokens=128,output_tokens=80)]
        # No required physical/progress/host/qualification fields; latest tie.
        for _ in range(2):self.assertEqual(choose(rows,False),(5,None,True,3,2))
        extras=deepcopy(rows)
        for row in extras:
            row.update(qualified=False,computed_tokens=None,shared_blocks=99,
                immediate_releasable_blocks=None,pending_native_store_dependencies=7,
                host_state_error='unknown',request_status='RUNNING')
        self.assertEqual(choose(extras,False,None),(5,None,True,3,2))
        tied=deepcopy(rows);tied[-1].update(max_tokens=1024,output_tokens=0)
        self.assertEqual(choose(tied,False),(6,'NO_REMAINING_BUDGET_IMPROVEMENT',False,3,0))
        self.assertEqual(choose(rows,True),(6,'ACTIVE_PROTECTION_OR_PHASE',False,0,0))
        for key,value in [('max_tokens',None),('max_tokens',0),('output_tokens',None),
                          ('output_tokens',-1),('output_tokens',1025)]:
            unknown=deepcopy(rows);unknown[0][key]=value
            self.assertEqual(choose(unknown,False),(6,'UNKNOWN_OUTPUT_BUDGET',False,0,0))

        execute=native_suffix_action()
        for candidate_index in (1,2):
            for rule in ('remaining_budget','tail'):
                scheduler,ns,requests,calls,waiting,freed,encoder,discarded=real_pick(rule)
                ns['remaining_budget_state']=dict(mode='ACTIVE' if rule=='remaining_budget' else 'SHADOW',
                    decision_count=0,proposal_count=0,action_request_count=0,
                    fallback_counts={},selector_wall_s=0.)
                requests[0].max_tokens=4096  # Already-processed prefix is excluded.
                candidate=requests[candidate_index]
                candidate.num_output_tokens=0;candidate.prompt_tokens=100;candidate.num_computed_tokens=50
                ns['owned'][candidate.request_id]=ns['owned'][candidate.request_id][:67]
                ns['cs']._req_status[candidate.request_id].transfer_jobs={'store-job'}
                ns['cs']._jobs['store-job']=NS(req_id=candidate.request_id,is_store=True,pending_count=1)
                prefix=[requests[0]];stamp=object()
                selected=execute(scheduler,1,prefix,stamp)
                expected=candidate if rule=='remaining_budget' else requests[-1]
                self.assertIs(selected,expected);self.assertEqual(prefix,[requests[0]])
                self.assertNotIn(selected,scheduler.running);self.assertEqual(calls,[(selected,stamp)])
                self.assertEqual((selected.status,selected.num_computed_tokens,
                                  selected.spec_token_ids,selected.num_preemptions),('PREEMPTED',0,[],1))
                self.assertIs(waiting[0],selected);self.assertEqual(freed,[selected])
                self.assertEqual(encoder,[selected]);self.assertEqual(discarded,[selected])
                self.assertIn(selected.request_id,scheduler.reset_preempted_req_ids)
                self.assertNotIn(selected.request_id,ns['residence'])
                decision=ns['data']['victim_decisions'][0];probe=decision['remaining_budget']
                self.assertEqual(decision['failed_request'],requests[1].request_id)
                self.assertEqual(decision['unprocessed_suffix_start'],1)
                self.assertEqual([row['index'] for row in decision['candidates']],[1,2,3,4])
                self.assertTrue(decision['fallback_unknown'])  # Legacy qualification observation only.
                self.assertIsNone(probe['fallback'])
                self.assertEqual(probe['proposed_request'],candidate.request_id)
                self.assertEqual(probe['returned_request'],selected.request_id)
                self.assertEqual(probe['action_requested'],rule=='remaining_budget')
                self.assertEqual((probe['tail_releasable_blocks'],probe['proposed_releasable_blocks']),(205,67))
                self.assertEqual((probe['tail_remaining_output_budget'],probe['proposed_remaining_output_budget']),(48,1024))
                self.assertEqual(probe['matching_candidate_count'],4)
                self.assertGreaterEqual(probe['selector_wall_s'],0.)
                for shadow in ['equal_held_once','partial_restore_once','equal_release_host_once',
                               'max_release_shadow','equal_release_budget']:
                    self.assertIsNotNone(decision[shadow])
                state=ns['remaining_budget_state']
                self.assertEqual((state['decision_count'],state['proposal_count'],state['action_request_count']),
                                 (1,1,int(rule=='remaining_budget')))
        for name in ['rotation_native.py','request_measurement.py']:
            self.assertEqual((HERE/'pkg'/name).read_bytes(),
                (HERE.parent/'candidate_native_equal_release_budget_r01/pkg'/name).read_bytes())


if __name__=='__main__':unittest.main()
