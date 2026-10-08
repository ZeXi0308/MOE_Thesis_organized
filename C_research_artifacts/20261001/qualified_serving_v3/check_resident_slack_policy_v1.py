#!/usr/bin/env python3
"""CPU fixtures executing the exact frozen native RUNNING loop; no GPU imports."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import textwrap
from types import SimpleNamespace as NS
sys.dont_write_bytecode = True
import resident_slack_policy_v1 as policy


class FullAttentionSpec: pass
class FullAttentionManager:
    def __init__(self): self.kv_cache_spec=FullAttentionSpec(); self.req_to_blocks={}


class Manager:
    def __init__(self, requests, free=0, fail_id=None):
        self.single=FullAttentionManager(); self.coordinator=NS(single_type_managers=[self.single])
        self.block_pool=NS(free=free); self.block_pool.get_num_free_blocks=lambda:self.block_pool.free
        self.steps=0; self.attempts=[]; self.fail_id=fail_id
        for r in requests: self.single.req_to_blocks[r.request_id]=[object() for _ in range((r.num_computed_tokens+15)//16)]
    def new_step_starts(self): self.steps+=1
    def allocate_slots(self, r, new_tokens, num_lookahead_tokens=0):
        assert num_lookahead_tokens==0
        if r.request_id==self.fail_id: raise RuntimeError('fixture allocate exception')
        held=self.single.req_to_blocks[r.request_id]; need=max((r.num_computed_tokens+new_tokens+15)//16-len(held),0)
        success=need<=self.block_pool.free; self.attempts.append((r.request_id,success))
        if not success: return None
        fresh=[object() for _ in range(need)]; held.extend(fresh); self.block_pool.free-=need
        return fresh  # Empty blocks is a successful zero-growth allocation.
    def free(self,r): self.block_pool.free+=len(self.single.req_to_blocks.pop(r.request_id))


def request(rid, computed, known=None, prompt=16):
    known=computed+1 if known is None else known
    return NS(request_id=rid,status='RUNNING',num_prompt_tokens=prompt,num_computed_tokens=computed,
        num_tokens=known,num_tokens_with_spec=known,num_output_placeholders=0,next_decode_eligible_step=0,
        max_tokens=1024,has_encoder_inputs=False,spec_token_ids=[],is_prefill_chunk=computed<prompt,
        priority=0,arrival_time=0.0,num_preemptions=0)


def self_test():
    source_path=Path(__file__).parent/'vllm_pf_source_v1/v1/core/sched/scheduler.py'
    source=source_path.read_text(); assert hashlib.sha256(source.encode()).hexdigest()==policy.SCHEDULER_SHA256
    original=next(n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.FunctionDef) and n.name=='schedule')
    outer=next(n for n in original.body if isinstance(n,ast.While))
    patched=policy.patched_schedule_tree(source).body[0]
    envelope=next(n for n in patched.body if isinstance(n,ast.For) and ast.unparse(n.target)=='_slack_pass')
    restored=copy.deepcopy(envelope.body[1]); del restored.body[0]
    inner=next(n for n in ast.walk(restored) if isinstance(n,ast.While) and ast.unparse(n.test)=='True')
    del inner.body[2]
    exit_branch=next(n for n in restored.body if isinstance(n,ast.If) and ast.unparse(n.test)=='new_blocks is None')
    del exit_branch.body[0]
    assert ast.dump(restored)==ast.dump(outer), 'native loop modified outside explicit skip insertion points'
    assert sum(isinstance(n,ast.Call) and ast.unparse(n.func)=='self.kv_cache_manager.new_step_starts' for n in ast.walk(patched))==1
    assert sum(isinstance(n,ast.AugAssign) and ast.unparse(n.target)=='self.current_step' for n in ast.walk(patched))==1
    # Compile a small scheduler with the entire exact native RUNNING loop.
    lines=['from types import SimpleNamespace as NS','from contextlib import nullcontext',
        'def record_function_or_nullcontext(*a): return nullcontext()',
        'class SchedulingPolicy: PRIORITY="priority"','class PauseState: UNPAUSED="UNPAUSED"','class Scheduler:',
        '    def schedule(self):','        self.current_step += 1','        self.kv_cache_manager.new_step_starts()',
        '        scheduled_running_reqs=[]; preempted_reqs=[]; req_to_new_blocks={}; num_scheduled_tokens={}',
        '        scheduled_spec_decode_tokens={}; scheduled_encoder_inputs={}; encoder_compute_budget=0',
        '        token_budget=self.max_num_scheduled_tokens; prefill_scheduled=False; defer_prefills=False; scheduled_timestamp=0.',
        '        req_index=0',textwrap.indent(ast.unparse(outer),'        '),
        '        if not preempted_reqs and self._pause_state == PauseState.UNPAUSED:',
        '            self.waiting_visits += 1',
        '        scheduler_output=NS(num_scheduled_tokens=num_scheduled_tokens,preempted_req_ids=[r.request_id for r in preempted_reqs])',
        '        for r in self.running:',
        '            r.num_computed_tokens += num_scheduled_tokens.get(r.request_id,0)',
        '        return scheduler_output',
        '    def _preempt_request(self,r,timestamp):',
        '        assert r not in self.running',
        '        self.kv_cache_manager.free(r); r.status="PREEMPTED"; r.num_computed_tokens=0; r.num_preemptions+=1',
        '        self.waiting.insert(0,r)']
    old_pin=policy.SCHEDULER_SHA256
    with tempfile.TemporaryDirectory(prefix='resident-slack-check-') as tmp:
        root=Path(tmp); fake=root/'fixture_scheduler.py'; fake.write_text('\n'.join(lines)+'\n')
        spec=importlib.util.spec_from_file_location('fixture_slack_scheduler',fake); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        policy.SCHEDULER_SHA256=hashlib.sha256(fake.read_bytes()).hexdigest()
        def setup(requests,fail_id=None):
            m=Manager(requests,fail_id=fail_id); s=mod.Scheduler(); s.kv_cache_manager=m
            s.policy='fcfs'; s.scheduler_config=NS(async_scheduling=False,long_prefill_token_threshold=0)
            s.use_eagle=False; s.num_lookahead_tokens=0; s.num_sampled_tokens_per_step=1
            s.is_encoder_decoder=False; s.connector=None; s.ec_connector=None; s.lora_config=None
            s.running=[]; s.waiting=[]; s.skipped_waiting=[]; s.current_step=32; s.max_num_scheduled_tokens=4096
            s.max_model_len=4096; s.need_mamba_block_aligned_split=False; s._pause_state='UNPAUSED'; s.waiting_visits=0
            return s,m,NS(engine_core=NS(engine_core=NS(scheduler=s)))
        try:
            cases=[('suffix',[request('a',16),request('b',18)],{'b':1},[],False),
                ('prefix',[request('a',18),request('b',16),request('c',19)],{'a':1,'c':1},[],False),
                ('all-boundary',[request('a',16),request('b',16)],{'a':1},['b'],True),
                ('native-self-break',[request('a',16)],{},['a'],True),
                ('prefill-none',[request('a',1,known=32,prompt=32),request('b',18)],{'a':31},['b'],False)]
            for label,requests,scheduled,preempted,fallback in cases:
                s,m,engine=setup(requests); original_schedule=s.schedule
                owned={k:tuple(v) for k,v in m.single.req_to_blocks.items()}
                with policy.resident_slack(engine,root/(label+'.json')) as report:
                    s.running.extend(requests); s.waiting.append(NS(request_id='fresh'))
                    result=s.schedule()
                    assert result.num_scheduled_tokens==scheduled and result.preempted_req_ids==preempted
                    assert s.current_step==33 and m.steps==1, 'fallback advanced scheduler clock/new_step twice'
                    assert s.waiting_visits==0, 'affected call admitted WAITING'
                    if label in ('suffix','prefix'):
                        assert all(tuple(m.single.req_to_blocks[k])==v for k,v in owned.items())
                        assert len(m.attempts)==len(requests), 'successful prefix was rescanned'
                        for r in requests: assert r.num_computed_tokens==({'a':16,'b':18} if label=='suffix' else {'a':18,'b':16,'c':19})[r.request_id]+scheduled.get(r.request_id,0)
                    if label=='all-boundary': assert m.attempts==[('a',False),('b',False),('a',False),('a',True)]
                    if label=='prefill-none': assert not report['calls']
                    else:
                        event=report['calls'][0]
                        assert event['fallback']==fallback and event['completed'] and event['scheduled_tokens']==scheduled
                        assert all(e['free_blocks']==0 and e['held_blocks']>0 for e in event['skips'])
                assert report['status']=='COMPLETE' and report['hooks_restored'] and s.schedule==original_schedule
                assert not any(k.startswith('_resident_slack') for k in vars(s))
                assert 'schedule' not in vars(s)
            requests=[request('a',16),request('b',18)];s,m,engine=setup(requests,fail_id='b'); original_schedule=s.schedule
            try:
                with policy.resident_slack(engine,root/'failure.json') as report:
                    s.running.extend(requests); s.schedule()
            except RuntimeError as exc: assert str(exc)=='fixture allocate exception'
            else: raise AssertionError('fixture exception swallowed')
            assert report['status']=='INCOMPLETE' and report['hooks_restored'] and not report['calls'][0]['completed']
            assert s.schedule==original_schedule and requests[0].num_computed_tokens==16
            assert len(m.single.req_to_blocks['a'])==1 and requests[0].status=='RUNNING'
        finally: policy.SCHEDULER_SHA256=old_pin
    return dict(status='PASS',checks=['native_loop_preserved_except_declared_skip_points','skip_KV_progress_and_suffix_execution',
        'successful_prefix_never_rescanned','empty_pass_native_fallback_once_same_step','native_self_preempt_break',
        'prefill_None_native_path','WAITING_closed_after_skip','exception_receipt_and_hook_restore'],gpu_used=False)


if __name__=='__main__': print(json.dumps(self_test()))
