#!/usr/bin/env python3
"""Small CPU-only AST rollback, physical-reference and hook restoration fixtures."""
import argparse
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
sys.dont_write_bytecode=True
import victim_choice_policy_v1 as policy


class FullAttentionSpec: pass
class FullAttentionManager:
    def __init__(self): self.kv_cache_spec=FullAttentionSpec(); self.req_to_blocks={}


class Manager:
    def __init__(self, counts, failures=1):
        self.single=FullAttentionManager(); self.coordinator=NS(single_type_managers=[self.single])
        self.block_pool=NS(free=100); self.block_pool.get_num_free_blocks=lambda:self.block_pool.free
        self.failures=failures; self.next_id=0; self.native_attempts=[]
        for rid,n in counts.items():
            self.single.req_to_blocks[rid]=[self.new_block() for _ in range(n)]
    def new_block(self):
        self.next_id+=1; self.block_pool.free-=1
        return NS(block_id=self.next_id,ref_cnt=1,is_null=False)
    def allocate_slots(self, request, num_new_tokens):
        failed=request.request_id=='b' and self.failures>0
        self.native_attempts.append((request.request_id,failed))
        if failed: self.failures-=1; return None
        self.single.req_to_blocks.setdefault(request.request_id,[]).append(self.new_block())
        return object()
    def free(self, request):
        for block in self.single.req_to_blocks.pop(request.request_id,[]):
            block.ref_cnt-=1
            if block.ref_cnt==0 and not block.is_null:self.block_pool.free+=1


def self_test(source_path):
    source=source_path.read_text()
    assert hashlib.sha256(source.encode()).hexdigest()==policy.SCHEDULER_SHA256
    module=ast.parse(source); cls=next(n for n in module.body if isinstance(n,ast.ClassDef) and n.name=='Scheduler')
    original=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='schedule')
    branch=next(n for n in ast.walk(original) if isinstance(n,ast.If) and ast.unparse(n.test)=='self.policy == SchedulingPolicy.PRIORITY')
    patched=policy.patched_schedule_tree(source).body[0]
    replacement=next(n for n in ast.walk(patched) if isinstance(n,ast.If) and ast.unparse(n.test)=='self.policy == SchedulingPolicy.PRIORITY')
    assert [ast.dump(n) for n in replacement.orelse[1:-1]]==[ast.dump(n) for n in branch.body[1:]]
    replacement.orelse=copy.deepcopy(branch.orelse)
    assert ast.dump(patched)==ast.dump(original)  # No other native statement changed.
    single=FullAttentionManager(); duplicate=NS(block_id=1,ref_cnt=2,is_null=False)
    shared=NS(block_id=2,ref_cnt=3,is_null=False); exclusive=NS(block_id=3,ref_cnt=1,is_null=False)
    null=NS(block_id=0,ref_cnt=-7,is_null=True)
    single.req_to_blocks['r']=[duplicate,duplicate,shared,shared,exclusive,null,null]
    assert policy.releasable_blocks(single,'r')==2 and duplicate.ref_cnt==2 and shared.ref_cnt==3
    assert policy.releasable_blocks(single,'missing')==0
    # Insert the exact pinned PRIORITY branch into a minimal synchronous CPU scheduler.
    lines=['from types import SimpleNamespace as NS','class SchedulingPolicy: PRIORITY="priority"','class Scheduler:',
        '    def schedule(self):','        self.current_step += 1',
        '        scheduled_running_reqs=[]; preempted_reqs=[]; num_scheduled_tokens={}; req_to_new_blocks={}',
        '        scheduled_spec_decode_tokens={}; scheduled_encoder_inputs={}; encoder_compute_budget=0',
        '        token_budget=self.max_num_scheduled_tokens; req_index=0',
        '        while req_index < len(self.running) and token_budget > 0:',
        '            request=self.running[req_index]',
        '            if request.skip: req_index+=1; continue',
        '            num_new_tokens=1','            while True:',
        '                new_blocks=self.kv_cache_manager.allocate_slots(request,num_new_tokens)',
        '                if new_blocks is not None: break',textwrap.indent(ast.unparse(branch),'                '),
        '                self._preempt_request(preempted_req,0.0)',
        '                preempted_reqs.append(preempted_req)',
        '                if preempted_req == request: break',
        '            if new_blocks is None: break',
        '            scheduled_running_reqs.append(request)',
        '            num_scheduled_tokens[request.request_id]=num_new_tokens',
        '            req_to_new_blocks[request.request_id]=new_blocks',
        '            token_budget-=num_new_tokens; req_index+=1',
        '        return NS(num_scheduled_tokens=num_scheduled_tokens,new_blocks=req_to_new_blocks,',
        '                  preempted=[r.request_id for r in preempted_reqs],budget=token_budget)',
        '    def _preempt_request(self,request,timestamp):',
        '        assert request not in self.running',
        '        self.kv_cache_manager.free(request)',
        '        request.status="PREEMPTED"; request.num_computed_tokens=0; request.num_preemptions+=1',
        '        self.waiting.insert(0,request)']
    cases=[('prefix-retry','max-free-full',{'a':3,'b':1,'c':3},2,False,{'b'},['a','c']),
           ('tail-tie','max-free-full',{'a':0,'b':1,'c':1},1,False,{'a','b'},['c']),
           ('current-break','bidkv-full',{'a':0,'b':1,'c':1},1,False,{'a'},['b']),
           ('bidkv-tie-prefix','bidkv-full',{'a':0,'b':1,'c':1},1,False,{'b','c'},['a']),
           ('skip-prefix','max-free-full',{'a':8,'b':1,'c':2},1,True,{'b'},['c'])]
    old_pin=policy.SCHEDULER_SHA256
    with tempfile.TemporaryDirectory(prefix='victim-choice-check-') as tmp:
        root=Path(tmp); fake=root/'fixture_scheduler.py'; fake.write_text('\n'.join(lines)+'\n')
        spec=importlib.util.spec_from_file_location('fixture_scheduler',fake); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        policy.SCHEDULER_SHA256=hashlib.sha256(fake.read_bytes()).hexdigest()
        try:
            for label,mode,counts,failures,skip,scheduled,victims in cases:
                m=Manager(counts,failures); s=mod.Scheduler()
                s.kv_cache_manager=m; s.policy='fcfs'; s.scheduler_config=NS(async_scheduling=False)
                s.use_eagle=False; s.num_lookahead_tokens=0; s.num_sampled_tokens_per_step=1
                s.is_encoder_decoder=False; s.connector=None; s.lora_config=None
                s.running=[]; s.waiting=[]; s.skipped_waiting=[]; s.current_step=32; s.max_num_scheduled_tokens=4
                requests=[NS(request_id=rid,status='RUNNING',arrival_time=0.0,priority=0,
                    num_computed_tokens=100 if label=='current-break' and rid=='b' else 1,
                    num_prompt_tokens=1, num_tokens=101 if label=='current-break' and rid=='b' else 2,
                    num_preemptions=0,output_token_ids=[1]* (100 if label=='current-break' and rid=='b' else 1),
                    max_tokens=1024,skip=skip and rid=='a') for rid in ('a','b','c')]
                engine=NS(engine_core=NS(engine_core=NS(scheduler=s)))
                before=dict(vars(s)); original_schedule=s.schedule
                with policy.victim_choice(engine,root/(label+'.json'),mode) as report:
                    s.running.extend(requests); result=s.schedule()
                    assert set(result.num_scheduled_tokens)==set(result.new_blocks)==scheduled
                    assert result.preempted==victims and result.budget+sum(result.num_scheduled_tokens.values())==4
                    assert report['initial_scheduler_step']==32
                    assert [e['trigger_index_in_call'] for e in report['decisions']]==list(range(len(victims)))
                    assert all(e['preemption']['predicted_free_delta_matches'] for e in report['decisions'])
                    if label=='prefix-retry':
                        assert report['decisions'][0]['rollback']['refunded_tokens']==1
                        assert report['decisions'][0]['rollback']['req_index_after']==0
                        assert report['decisions'][0]['selected_kind']=='scheduled-prefix'
                    if skip: assert report['decisions'][0]['excluded_unscheduled_prefix']==1
                assert s.schedule==original_schedule and report['hooks_restored'] and report['status']=='COMPLETE'
                assert not any(k.startswith('_victim_choice') for k in vars(s))
                assert ('schedule' in vars(s))==('schedule' in before) and ('_preempt_request' in vars(s))==('_preempt_request' in before)
            # A selected multi-token catchup victim must fail before remove/preempt.
            s.running=[]; s.waiting=[]; m.failures=1
            for r in requests: r.status='RUNNING'; r.num_computed_tokens=1; r.num_tokens=2; r.skip=False
            bad=requests[0]; bad.num_tokens=bad.num_computed_tokens+2
            m.single.req_to_blocks['a']=[m.new_block() for _ in range(8)]
            try:
                with policy.victim_choice(engine,root/'unsupported-catchup.json','max-free-full') as report:
                    s.running.extend(requests); s.schedule()
            except RuntimeError as exc: assert 'UNSUPPORTED_SELECTED_APC_DOMAIN' in str(exc)
            else: raise AssertionError('multi-token victim accepted')
            assert report['status']=='INCOMPLETE' and report['actual_preemptions']==0
            assert bad in s.running and report['hooks_restored']
            s.running=[]; s.waiting=[]; saved_schedule=s.schedule
            try:
                with policy.victim_choice(engine,root/'interrupted.json','max-free-full') as report:
                    raise InterruptedError('CPU fixture')
            except InterruptedError: pass
            assert report['status']=='INCOMPLETE' and report['hooks_restored'] and s.schedule==saved_schedule
        finally: policy.SCHEDULER_SHA256=old_pin
    return dict(status='PASS',checks=['one_AST_else_only_and_exact_native_rollback','duplicate_shared_null_physical_release',
        'prefix_refund_and_repeated_None','tail_tie','BidKV_current_native_break','BidKV_tie_prefix',
        'unscheduled_prefix_exclusion','normal_and_exception_hook_restore','APC_catchup_victim_fail_closed'],gpu_used=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scheduler-source',type=Path,default=Path(__file__).parent/'vllm_pf_source_v1/v1/core/sched/scheduler.py')
    print(json.dumps(self_test(parser.parse_args().scheduler_source)))
