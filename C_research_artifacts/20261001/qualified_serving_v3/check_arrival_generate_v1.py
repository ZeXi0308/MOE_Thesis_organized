#!/usr/bin/env python3
"""Small CPU clock fixtures; no vLLM installation or GPU needed."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
sys.dont_write_bytecode=True
import arrival_generate_v1 as helper


class Clock:
    def __init__(self): self.t=100.0; self.origin=self.t; self.epoch=1700000000.0; self.sleeps=[]
    def perf_counter(self): return self.t
    def time(self): return self.epoch
    def sleep(self,duration):
        assert 0<=duration<=.010000001
        self.sleeps.append(duration); self.t+=duration


class Params:
    def __init__(self,**kwargs):
        assert set(kwargs)=={'n','temperature','max_tokens','min_tokens','ignore_eos','stop','stop_token_ids','detokenize','output_kind'}
        assert kwargs['n']==1 and kwargs['temperature']==0 and kwargs['ignore_eos'] is False
        assert kwargs['min_tokens']==0 and kwargs['stop']==kwargs['stop_token_ids']==[]
        self.__dict__.update(kwargs); self.repetition_penalty=1.05
        self._eos_token_id=2; self._all_stop_token_ids={2}


class Engine:
    def __init__(self,clock,step_duration=.1,fail=False):
        self.clock,self.duration,self.fail=clock,step_duration,fail
        self.active={}; self.added=[]
        pool=NS(num_gpu_blocks=101,get_num_free_blocks=lambda:100)
        self.s=NS(requests={},current_step=32,kv_cache_manager=NS(block_pool=pool))
        def schedule(): self.s.current_step+=1; return NS(preempted_req_ids=[])
        self.s.schedule=schedule; self.original=schedule
        self.engine_core=NS(engine_core=NS(scheduler=self.s))
    def add_request(self,rid,prompt,sampling,arrival_time):
        offered=0.0 if int(rid.rsplit('r',1)[1])<2 else 4.0
        assert self.clock.t-self.clock.origin>=offered, 'early second-batch submission rejected'
        assert arrival_time==self.clock.epoch+offered
        assert prompt.keys()=={'prompt_token_ids'}
        nid=rid+'-abcdef01'; self.active[rid]=(nid,0); self.s.requests[nid]=NS(sampling_params=sampling)
        self.added.append((rid,self.clock.t-self.clock.origin,arrival_time)); self.clock.t+=.001
    def has_unfinished_requests(self): return bool(self.active)
    def step(self):
        self.s.schedule()
        if self.fail: raise RuntimeError('fixture engine failure')
        self.clock.t+=self.duration; result=[]
        for rid,(nid,n) in list(self.active.items()):
            n+=1; done=n==3
            completion=NS(token_ids=list(range(10,10+n)),text='fixture',finish_reason='stop' if done else None,stop_reason=None)
            result.append(NS(request_id=rid,outputs=[completion],finished=done))
            if done: del self.active[rid]; del self.s.requests[nid]
            else: self.active[rid]=(nid,n)
        return result


def self_test():
    old_time=helper.time; saved={k:sys.modules.get(k) for k in ('vllm','vllm.sampling_params')}
    sys.modules['vllm']=NS(SamplingParams=Params)
    sys.modules['vllm.sampling_params']=NS(RequestOutputKind=NS(CUMULATIVE='cumulative'))
    requests=[dict(request_id='r'+str(i),prompt_token_ids=[i+1]) for i in range(4)]
    try:
        with tempfile.TemporaryDirectory(prefix='arrival-generate-check-') as tmp:
            for label,duration,fail in [('idle',.1,False),('busy',2.5,False),('failure',.1,True)]:
                out=Path(tmp)/label;out.mkdir(); clock=Clock();helper.time=clock;engine=Engine(clock,duration,fail)
                try: result=helper.generate(engine,requests,4,'measured',out,[0.,0.,4.,4.])
                except RuntimeError as exc: assert fail and str(exc)=='fixture engine failure'
                else: assert not fail and result['status']=='COMPLETE'
                assert engine.s.schedule is engine.original
                rows=json.loads((out/'measured-outputs.json').read_text())
                receipt=json.loads((out/'measured-arrival-receipt.json').read_text())
                sampling=json.loads((out/'measured-native-sampling.json').read_text())
                assert [r['arrival_s'] for r in rows]==[0.,0.,4.,4.]
                assert receipt['clock']==dict(perf_counter_origin_s=100.0,epoch_origin_unix_s=1700000000.0,
                    basis='All row/journal/step timestamps are seconds from the same perf_counter origin; offered arrivals are unchanged.')
                if fail:
                    assert receipt['status']=='INCOMPLETE' and receipt['submitted_requests']==2
                    assert not rows[2]['submitted'] and rows[2]['add_request_s'] is None
                    continue
                assert len(sampling)==4 and len(receipt['sampling_scans'])==2
                assert [len(r['new_native_request_ids']) for r in receipt['sampling_scans']]==[2,2]
                assert all(r['repetition_penalty']==1.05 for r in sampling.values())
                for r in rows:
                    assert r['add_request_return_s']>r['add_request_s']>=r['arrival_s']
                    assert r['host_elapsed_s']==r['host_returns'][-1]['return_s']==r['token_times_s'][-1]
                    assert r['host_elapsed_s']>r['arrival_s'] and len(r['output_token_ids'])==3
                steps=json.loads((out/'measured-steps.json').read_text())['steps']
                for step in steps:
                    assert all(r['submitted'] and r['add_request_return_s']<=step['start_s']
                               for r in rows if r['arrival_s']<=step['start_s'])
                if label=='idle':
                    assert clock.sleeps and rows[0]['host_elapsed_s']<4 and rows[2]['host_elapsed_s']>4
                else:
                    assert not clock.sleeps and rows[2]['add_request_s']>=5
        return dict(status='PASS',checks=['no_early_submission_and_native_epoch','idle_pending_arrival',
            'both_batches_native_sampling_no_per_step_rescan','absolute_completion_clocks',
            'busy_step_arrival_lag','failure_receipts_and_hook_restore'],gpu_used=False)
    finally:
        helper.time=old_time
        for k,value in saved.items():
            if value is None: sys.modules.pop(k,None)
            else: sys.modules[k]=value


if __name__=='__main__': print(json.dumps(self_test()))
