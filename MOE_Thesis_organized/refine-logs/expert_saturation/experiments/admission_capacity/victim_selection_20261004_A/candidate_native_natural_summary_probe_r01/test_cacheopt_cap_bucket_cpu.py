"""One targeted CPU regression for the fixed cap-bucket component adapter."""
import ast
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'pkg'))
import staged_store_rotation as adapter
from test_max_release_cpu import fixture
from test_partial_restore_once_cpu import native_suffix_action


def row(index,remaining,held):
    return dict(index=index,max_tokens=1024,output_tokens=1024-remaining,
                held_blocks=held,request_status='RUNNING')


class CacheoptCapBucketTests(unittest.TestCase):
    def test_fixed_boundaries_held_metric_native_candidates_and_lifecycle(self):
        choose=adapter._cacheopt_cap_bucket_choice
        # Remaining bucket boundary precedes any capacity preference.
        self.assertEqual(choose([row(1,128,8),row(2,129,64)],False,16),(2,None))
        # Ceil boundaries: 8 pages=128 tokens, 9 pages=144; no exact-value tie break.
        self.assertEqual(choose([row(1,129,8),row(2,256,9)],False,16),(1,None))
        self.assertEqual(choose([row(1,129,9),row(2,256,16)],False,16),(2,None))
        self.assertEqual(choose([row(1,255,8),row(2,129,8)],False,16),(2,None))
        self.assertEqual(choose([row(1,0,0),row(2,0,1)],False,16),(1,None))
        self.assertEqual(choose([row(1,0,0),row(2,1,100)],False,16),(2,None))
        rows=[row(1,982,8),row(2,1011,16)]
        rows[0].update(immediate_releasable_blocks=8,shared_blocks=0)
        rows[1].update(immediate_releasable_blocks=0,shared_blocks=16)
        for value in rows:value.update(qualified=False,computed_tokens=None,pending_native_store_dependencies=7)
        for _ in range(2):self.assertEqual(choose(rows,False,16),(1,None))
        self.assertEqual(choose(rows,True,16),(2,'ACTIVE_PROTECTION_OR_PHASE'))
        for invalid in (None,0,True):self.assertEqual(choose(rows,False,invalid),(2,'UNKNOWN_BLOCK_SIZE'))
        for key,value in [('max_tokens',None),('output_tokens',-1),('output_tokens',1025),
                          ('held_blocks',None),('held_blocks',-1),('request_status',None)]:
            bad=deepcopy(rows);bad[0][key]=value
            self.assertEqual(choose(bad,False,16),(2,'UNKNOWN_BUDGET_CAPACITY_OR_NATIVE_STATUS'))

        execute=native_suffix_action()
        for candidate_index in (1,2):
            for rule in ('cacheopt_cap_bucket','remaining_budget'):
                scheduler,ns,requests,calls,waiting,freed,encoder,discarded=fixture(rule)
                ns['cacheopt_bucket_state']=dict(mode='ACTIVE' if rule=='cacheopt_cap_bucket' else 'SHADOW',
                    decision_count=0,proposal_count=0,action_request_count=0,fallback_counts={},selector_wall_s=0.)
                candidate=requests[candidate_index]
                candidate.num_output_tokens=42;candidate.prompt_tokens=100;candidate.num_computed_tokens=50
                ns['owned'][candidate.request_id]=ns['owned'][candidate.request_id][:50]
                budget=requests[3]
                budget.num_output_tokens=13;budget.prompt_tokens=100;budget.num_computed_tokens=50
                ns['owned'][budget.request_id]=ns['owned'][budget.request_id][:83]
                ns['cs']._req_status[candidate.request_id].transfer_jobs={'store-job'}
                ns['cs']._jobs['store-job']=NS(req_id=candidate.request_id,is_store=True,pending_count=1)
                prefix=[requests[0]];stamp=object()
                selected=execute(scheduler,1,prefix,stamp)
                self.assertIs(selected,candidate if rule=='cacheopt_cap_bucket' else budget)
                self.assertEqual(prefix,[requests[0]]);self.assertNotIn(selected,scheduler.running)
                self.assertEqual(calls,[(selected,stamp)])
                self.assertEqual((selected.status,selected.num_computed_tokens,
                                  selected.spec_token_ids,selected.num_preemptions),('PREEMPTED',0,[],1))
                self.assertIs(waiting[0],selected);self.assertEqual(freed,[selected])
                self.assertEqual(encoder,[selected]);self.assertEqual(discarded,[selected])
                self.assertIn(selected.request_id,scheduler.reset_preempted_req_ids)
                self.assertNotIn(selected.request_id,ns['residence'])
                event=ns['data']['victim_decisions'][0];probe=event['cacheopt_cap_bucket']
                self.assertEqual([r['index'] for r in event['candidates']],[1,2,3,4])
                self.assertEqual(event['failed_request'],requests[1].request_id)
                self.assertTrue(event['fallback_unknown'])  # Legacy decode qualifier does not gate this component.
                self.assertEqual(probe['proposed_request'],candidate.request_id)
                self.assertEqual(probe['returned_request'],selected.request_id)
                self.assertEqual(probe['action_requested'],rule=='cacheopt_cap_bucket')
                self.assertEqual(probe['mode'],'ACTIVE' if rule=='cacheopt_cap_bucket' else 'SHADOW')
                self.assertTrue(probe['changed_from_tail']);self.assertIsNone(probe['fallback'])
                state=ns['cacheopt_bucket_state']
                self.assertEqual((state['decision_count'],state['proposal_count'],state['action_request_count']),
                                 (1,1,int(rule=='cacheopt_cap_bucket')))
                for name in ['remaining_budget','max_release_shadow']:
                    self.assertEqual(event[name]['returned_request'],selected.request_id)
                self.assertEqual(event['remaining_budget']['action_requested'],rule=='remaining_budget')
                self.assertFalse(event['max_release_shadow']['action_requested'])
                for name in ['equal_held_once','partial_restore_once','equal_release_host_once','equal_release_budget']:
                    self.assertIsNotNone(event[name])
        for guard,reason in [('phase','ACTIVE_PROTECTION_OR_PHASE'),
                             ('unknown','UNKNOWN_BUDGET_CAPACITY_OR_NATIVE_STATUS')]:
            scheduler,ns,requests,*_=fixture('cacheopt_cap_bucket')
            ns['cacheopt_bucket_state']=dict(mode='ACTIVE',decision_count=0,proposal_count=0,
                action_request_count=0,fallback_counts={},selector_wall_s=0.)
            if guard=='phase':ns['phase']=object()
            else:requests[1].max_tokens=None
            self.assertEqual(scheduler._rotation_pick_victim(1,[requests[0]]),4)
            self.assertEqual(ns['data']['victim_decisions'][0]['cacheopt_cap_bucket']['fallback'],reason)
            self.assertEqual(ns['cacheopt_bucket_state']['fallback_counts'],{reason:1})
        parent=HERE.parent/'candidate_native_max_release_r01/pkg'
        source=(HERE/'pkg/staged_store_rotation.py').read_text();old=(parent/'staged_store_rotation.py').read_text()
        for name in ['_remaining_budget_choice','_max_release_shadow_choice']:
            snippets=[]
            for text in (source,old):
                node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name==name)
                snippets.append(ast.get_source_segment(text,node))
            self.assertEqual(*snippets)
        for name in ['rotation_native.py','request_measurement.py']:
            self.assertEqual((HERE/'pkg'/name).read_bytes(),(parent/name).read_bytes())


if __name__=='__main__':unittest.main()
