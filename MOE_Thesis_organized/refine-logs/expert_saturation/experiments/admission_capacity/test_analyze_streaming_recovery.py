import copy
import unittest
import analyze_streaming_recovery as a


def fixture(specs, execution):
    requests=[];events=[];steps=[];calls=[];memory=[];decisions=[]
    for rid,arrival,times,tokens,terminal,reason in specs:
        requests.append(dict(request_id=rid,arrival_s=arrival,admission_s=arrival,
            prompt_tokens=1,prompt_token_ids_sha256='synthetic-prompt',max_output_tokens=2,output_token_ids=tokens,token_times_s=times,
            completion_s=terminal,status='completed',stop_reason=reason,native_stop_reason=None))
        for n,t in enumerate(times,1):
            events.append(dict(request_id=rid,received_s=t,cumulative_tokens=n,cumulative_token_ids=tokens[:n],
                new_token_ids=[tokens[n-1]],chunk_size=1,prefix_valid=True,
                finished=t==terminal,finish_reason=reason if t==terminal else None))
        if not times or times[-1]!=terminal:
            events.append(dict(request_id=rid,received_s=terminal,cumulative_tokens=len(tokens),
                cumulative_token_ids=tokens,new_token_ids=[],chunk_size=0,prefix_valid=True,
                finished=True,finish_reason=reason))
    for k,(rid,before,after,out,recompute) in enumerate(execution):
        item=dict(request_id=rid,internal_request_id='i-'+rid,computed_adjustment=0,
            scheduled_start_computed=before,computed_before=before,computed_after=after,
            scheduled_tokens=after-before,output_tokens_before=out,recompute_tokens=recompute)
        steps.append(dict(step=k,start_s=k+.1,end_s=k+.2,running_before=1,scheduled=[item],recompute_tokens=recompute))
        calls.append(dict(start_s=k,returned_s=k+.9,scheduler_step_start=k,scheduler_step_end=k+1,completed=True))
        memory.append(dict(schedule_completed=True,model_execution_confirmed=True))
        decisions.append(dict(step=k,status='APPLIED',active=True,population_mode='open',mode='native',
            actual_scheduled={'i-'+rid:after-before},forced_preempted=[],natural_preempted=[],preempted=[]))
    raw=dict(status='COMPLETE',error=None,output_mode='eos',requests=requests,
        output_events=sorted(events,key=lambda e:e['received_s']),scheduler_steps=steps,
        engine_steps=calls,memory_trace=memory,preemption_events=[],
        internal_to_source={'i-'+q['request_id']:q['request_id'] for q in requests},observation_end_s=len(steps))
    workload=dict(source_requests=[dict(request_id=q['request_id'],prompt_token_count=1,
        prompt_token_ids_sha256='synthetic-prompt',max_output_tokens=2) for q in requests],
        arrival_traces_s=dict(steady=[q['arrival_s'] for q in requests]))
    return raw,workload,decisions


class StreamingTests(unittest.TestCase):
    def test_zero_one_output_and_terminal_after_last_output(self):
        raw,w,d=fixture([('zero',0,[],[],.9,'stop'),('one',1,[1.9],[7],2.9,'stop'),
            ('two',2,[3.9,4.9],[8,9],4.9,'length')],
            [('zero',0,1,0,0),('one',0,1,0,0),('one',1,2,1,0),('two',0,1,0,0),('two',1,2,1,0)])
        result=a.analyze(raw,w,d,dict(generation_config=None),'native');m=result['metrics']
        self.assertEqual(m['completed_requests'],3)
        self.assertEqual(m['output_tokens'],3)
        self.assertEqual(m['ttft_defined_requests'],2)
        self.assertEqual(m['gap_defined_requests'],1)
        self.assertAlmostEqual(m['mean_completion_s'],1.9)
        self.assertAlmostEqual(m['per_request'][1]['terminal_after_last_output_s'],1)
        self.assertIsNone(m['per_request'][0]['ttft_s'])
        self.assertEqual(m['finish_reason_counts'],dict(stop=2,length=1))

    def test_zero_output_terminal_recovery_is_not_censored(self):
        raw,w,d=fixture([('q',0,[],[],1.9,'stop')],[('q',0,1,0,0),('q',0,2,0,1)])
        raw['preemption_events']=[dict(attempted_step=1,victim_internal_request_id='i-q',
            original_preemption_returned=True,victim_state=dict(output_tokens=0,computed_tokens=1),
            victim_state_after=dict(computed_tokens=0,block_counts=[0]),
            output_token_ids_before=[],output_token_ids_after=[])]
        d[1].update(natural_preempted=['i-q'],preempted=['i-q'])
        r=a.analyze(raw,w,d,{},'native')['lifecycle']
        self.assertEqual(r['summary']['completed_without_new_output_residencies'],1)
        self.assertEqual(r['summary']['censored_no_output_residencies'],0)
        self.assertEqual(r['residencies'][0]['terminal_s'],1.9)
        self.assertIsNone(r['residencies'][0]['first_new_output_engine_return_s'])

    def test_silent_adapter_bypass_and_fake_terminal_are_rejected(self):
        raw,w,d=fixture([('q',0,[.9],[7],.9,'stop')],[('q',0,1,0,0)])
        bad=copy.deepcopy(d);bad[0]['active']=False
        with self.assertRaisesRegex(ValueError,'bypassed'):
            a.analyze(raw,w,bad,{},'native')
        raw['requests'][0]['completion_s']=1
        with self.assertRaisesRegex(ValueError,'terminal event differs'):
            a.analyze(raw,w,d,{},'native')


if __name__=='__main__':unittest.main()
