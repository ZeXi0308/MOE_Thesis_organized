"""Causal accounting regressions for observed recovery epochs."""
from copy import deepcopy
import unittest
from analyze_effective_recovery_service import analyze, summarize


def fixture():
    # Two recoveries: first pays six historical positions and yields one token,
    # then loses state; second pays seven and completes with one more token.
    work = [(0,2,2,3),(2,4,2,3),(4,7,2,3),(0,7,7,4),(7,8,0,4)]
    steps, calls, mem = [], [], []
    for k,(before,after,recompute,outputs) in enumerate(work):
        x = dict(request_id='q', computed_before=before, computed_after=after,
                 computed_adjustment=0, scheduled_start_computed=before,
                 scheduled_tokens=after-before, recompute_tokens=recompute,
                 output_tokens_before=outputs)
        steps.append(dict(step=k,start_s=k+.1,end_s=k+.2,scheduled=[x],recompute_tokens=recompute))
        calls.append(dict(scheduler_step_start=k,scheduler_step_end=k+1,
                          completed=True,start_s=float(k),returned_s=k+.9))
        mem.append(dict(schedule_completed=True,model_execution_confirmed=None))
    events = []
    for k,computed,outputs in [(0,6,3),(3,7,4)]:
        events.append(dict(victim_internal_request_id='internal',attempted_step=k,
                           original_preemption_returned=True,
                           victim_state=dict(computed_tokens=computed,output_tokens=outputs),
                           victim_state_after=dict(computed_tokens=0,block_counts=[0]),
                           output_token_ids_before=list(range(outputs)),output_token_ids_after=list(range(outputs))))
    q = dict(request_id='q',status='completed',output_token_ids=list(range(5)),token_times_s=[-3.,-2.,-1.,2.9,4.9])
    return dict(status='COMPLETE',error=None,requests=[q],internal_to_source={'internal':'q'},
                scheduler_steps=steps,engine_steps=calls,memory_trace=mem,preemption_events=events)


class LifecycleTests(unittest.TestCase):
    def test_served_discard_is_not_zero_value(self):
        result = analyze(fixture())
        a,b = result['residencies']
        self.assertEqual((a['new_outputs'],a['recompute_positions'],a['recompute_prefix_reexecuted_next_residency']),(1,6,6))
        self.assertTrue(a['partial_prefix_reused_in_later_call'])
        self.assertEqual(a['first_output_step'],2)
        self.assertEqual(result['summary']['recompute_positions'],13)
        self.assertEqual(result['summary']['served_then_discarded_recompute_positions'],6)
        self.assertEqual(result['summary']['served_to_completion_recompute_positions'],7)
        self.assertNotIn('discard_before_output_recompute_positions',result['summary'])

    def test_internal_reuse_and_terminal_loss_are_separate(self):
        r = dict(resumed=True,new_outputs=0,end_reason='repreempted',recompute_positions=4,
                 recompute_only_calls=2,partial_prefix_reused_in_later_call=True)
        result = summarize([r])
        self.assertEqual(result['discard_before_output_recompute_positions'],4)
        self.assertEqual(result['partial_recovery_chains_with_retained_prefix'],1)

    def test_schedule_success_cannot_override_execution_failure(self):
        raw = fixture(); raw['memory_trace'][0]['model_execution_confirmed']=False
        with self.assertRaisesRegex(ValueError,'unconfirmed execution'):
            analyze(raw)

    def test_no_future_output_or_hidden_prefix_load(self):
        for key,value,message in [('output_tokens_before',4,'future output'),('computed_adjustment',1,'cache loading')]:
            raw=deepcopy(fixture());raw['scheduler_steps'][0]['scheduled'][0][key]=value
            with self.assertRaisesRegex(ValueError,message):
                analyze(raw)


if __name__ == '__main__':
    unittest.main()
