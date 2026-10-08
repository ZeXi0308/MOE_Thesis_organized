"""Actual adapter closures/native scheduling loop on CPU fixtures; no GPU/KV transfer claim."""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS,MethodType
from unittest.mock import patch
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[4]
sys.path[:0]=[str(ROOT/'pkg'),str(REPO/'refine-logs/expert_saturation/experiments/admission_capacity')]
import ltr_style_native as adapter
from ltr_style_selected import LTRStyleSelected,Request
from recovery_service_components import LTRRequestState
from verify_native_recovery_execution import create,returned
from verify_rotation_native import Queue
import verify_native_recovery_execution as harness


def fixture(scheduler,native,cs,threshold=30,quantum=10):
    tree=ast.parse(Path(adapter.__file__).read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='install')
    start=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='step' for t in n.targets))
    body=ast.parse('def build(scheduler,native,owned,pool,cs,threshold,quantum):\n diagnostic=True\n oldcalc=cs._calc_num_offloadable_tokens\n hadcalc=True\n').body[0]
    body.body+=deepcopy(fn.body[start:]);env=dict(vars(adapter))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[body],type_ignores=[])),'<actual-ltr-closures>','exec'),env)
    pool=scheduler.kv_cache_manager.block_pool
    owned=scheduler.kv_cache_manager.coordinator.single_type_managers[0].req_to_blocks
    data,undo=env['build'](scheduler,native,owned,pool,cs,threshold,quantum)
    cells=dict(zip(scheduler.schedule.__code__.co_freevars,scheduler.schedule.__closure__))
    return data,undo,cells['selector'].cell_contents


def selector_cases():
    rows=[Request('old',0,'PREEMPTED',10),Request('later',1,'PREEMPTED',2),
          Request('peer',2,'RUNNING',0,8,2,True)]
    s=LTRStyleSelected(30,2)
    s.counters.states={r.request_id:LTRRequestState(idle=30 if r.status=='PREEMPTED' else 0) for r in rows}
    intent=s.begin_step(rows,0);assert intent.target_id=='later' and s.active_target is None
    assert s.skipped[0]['target']=='old'
    s.accept(intent);s.after_step({'peer':1})
    for status,scheduled,q in [('WAITING_FOR_REMOTE_KVS',{},2),('RUNNING',{'later':128},2),('RUNNING',{'later':1},1)]:
        now=[rows[0],Request('later',1,status,0,1 if q==1 else 0),rows[2]]
        assert s.begin_step(now,0).quantum_remaining==q;s.after_step(scheduled)
    intent=s.begin_step(now,0);assert s.active_target is None;s.after_step({})
    free=LTRStyleSelected(30,10);free.counters.states={'later':LTRRequestState(idle=30)}
    intent=free.begin_step([rows[1]],2);assert intent.action=='PRIORITIZE_WAITING' and intent.victim_id is None
    free.accept(intent);free.after_step({});free.release_active()
    slots=LTRStyleSelected(30,10);slots.counters.states={'later':LTRRequestState(idle=30),'peer':LTRRequestState()}
    intent=slots.begin_step(rows[1:],2,free_slots=0)
    assert intent.action=='PREPARE_SELECTED' and intent.victim_id=='peer';slots.after_step({})
    assert free.counters.states['later'].quantum_remaining==10
    return ['HOL_first_executable_accept_only','pending_zero_quantum','first_output_not_release',
            'positive_call_not_token_count','free_capacity_no_eviction','release_retains_counters','slot_full_uses_legal_single_victim']


def native_growth(source,fail_target=False):
    # Target needs its third block, peer owns one, free=0. Existing fake allocator;
    # actual pinned native schedule/preempt/cache payload/post-schedule methods.
    before=dict(pool=dict(total_blocks=4,free_blocks=0),running_ids=['peer','target'],waiting_count=0,
        requests={'peer':dict(computed_tokens=15,prompt_tokens=8,output_tokens=8,num_preemptions=0,block_counts=[1]),
                  'target':dict(computed_tokens=32,prompt_tokens=16,output_tokens=17,num_preemptions=1,block_counts=[2])})
    with patch.object(harness.adapter,'install',return_value=([],lambda:None)):
        scheduler,_,_=create(source,before,[],False,fail_ids=['target'] if fail_target else [])
    for i,r in enumerate(scheduler.requests.values()):r.arrival_time=float(i)
    ns=dict(scheduler.schedule.__func__.__globals__)
    exec(compile(adapter.patched_schedule_tree(source.read_text()),str(source),'exec'),ns)
    native_method=MethodType(ns['schedule'],scheduler)
    def native():
        result=native_method();result.kv_connector_metadata=NS(store_jobs={},load_jobs={},jobs_to_flush=set())
        return result
    cs=NS(_calc_num_offloadable_tokens=lambda rs,n:n,_req_status={},_jobs={})
    data,undo,selector=fixture(scheduler,native,cs)
    selector.counters.states={'target':LTRRequestState(priority=-1,quantum_remaining=10),'peer':LTRRequestState()}
    selector.active_target='target'
    result=scheduler.schedule()
    assert any(e['event']=='intent' and not e['reserve_enabled'] for e in data['events'])
    if fail_target:
        assert 'target' in result.preempted_req_ids and selector.active_target is None
        assert selector.counters.states['target'].quantum_remaining==10
    else:
        assert result.num_scheduled_tokens.get('target')==1 and result.preempted_req_ids=={'peer'}
        assert selector.active_target=='target' and selector.counters.states['target'].quantum_remaining==9
    out=dict(scheduled=result.num_scheduled_tokens,preempted=sorted(result.preempted_req_ids),
        quantum_remaining=selector.counters.states['target'].quantum_remaining,active=selector.active_target)
    undo();assert not hasattr(scheduler,'_rotation_begin')
    return out


def staged_case(omit_flush=False,changed=False,foreign=False,active=False):
    blocks={i:NS(block_id=i,is_null=False) for i in (1,2)};owned={'v':[blocks[1],blocks[2]],'t':[]};free=[1]
    req=lambda rid,p,o,c,status:NS(request_id=rid,num_prompt_tokens=p,num_output_tokens=o,
        num_computed_tokens=c,max_tokens=1024,status=NS(name=status),arrival_time=0 if rid=='t' else 1)
    v,t=req('v',16,16,31,'RUNNING'),req('t',32,1,0,'PREEMPTED')
    pool=NS(blocks=blocks,get_num_free_blocks=lambda:free[0])
    scheduler=NS(requests={'t':t,'v':v},running=[v],waiting=Queue([t]),skipped_waiting=Queue(),max_num_running_reqs=32,
        kv_cache_manager=NS(block_pool=pool,coordinator=NS(single_type_managers=[NS(req_to_blocks=owned)])))
    rs=NS(req=v,transfer_jobs=set(),group_states=[NS(block_ids=[1,2],offload_keys=['k1','k2'])])
    cs=NS(_calc_num_offloadable_tokens=lambda rs,n:n,_req_status={'v':rs},_jobs={})
    def preempt(r,ts):free[0]+=len(owned[r.request_id]);owned[r.request_id]=[];r.status=NS(name='PREEMPTED');r.num_computed_tokens=0;scheduler.waiting.prepend_request(r)
    scheduler._preempt_request=preempt
    def native():
        pre=[];scheduler._rotation_begin(pre,0.);meta=NS(store_jobs={},load_jobs={},jobs_to_flush=set())
        scheduled={r.request_id:1 for r in scheduler.running}
        if pre:
            if not omit_flush:meta.jobs_to_flush={7}
            t.status=NS(name='WAITING_FOR_REMOTE_KVS')
            scheduler.waiting.remove(t);scheduler.skipped_waiting.append(t)
        elif cs._calc_num_offloadable_tokens(rs,32):
            cs._jobs[7]=NS(req_id='v',is_store=True,keys={'k1'});rs.transfer_jobs.add(7)
            meta.store_jobs[7]=NS(req_id='v',src_spec=NS(block_ids=[1]))
        return NS(num_scheduled_tokens=scheduled,preempted_req_ids={r.request_id for r in pre},kv_connector_metadata=meta)
    data,undo,selector=fixture(scheduler,native,cs)
    selector.counters.states={'t':LTRRequestState(idle=30),'v':LTRRequestState()}
    if foreign:
        u=req('u',16,1,0,'WAITING_FOR_REMOTE_KVS');scheduler.requests['u']=u
        scheduler.skipped_waiting.append(u);free[0]=3
        if active:
            selector.active_target='t';selector.counters.states['t']=LTRRequestState(priority=-1,quantum_remaining=10)
        scheduler.schedule()
        assert scheduler._rotation_target is None and selector.active_target is None
        assert data['applied_rotations']==0 and not any(e['event']=='accept' for e in data['events'])
        assert selector.counters.states['t'].quantum_remaining==10 and scheduler.skipped_waiting==[u]
        undo();return 'FOREIGN_LOAD_CENSORED_ACTIVE_GATE_RELEASED' if active else 'FREE_FOREIGN_LOAD_NOT_LATCHED'
    scheduler.schedule();assert any(e['event']=='store_delta' for e in data['events'])
    v.num_computed_tokens+=1;v.num_output_tokens+=1
    if changed:t.num_output_tokens+=1
    try:scheduler.schedule()
    except RuntimeError as exc:
        assert omit_flush and 'flush omitted' in str(exc);outcome='MISSING_FLUSH_REJECTED'
    else:
        if changed:assert data['applied_rotations']==0 and selector.active_target is None;outcome='CHANGED_TARGET_CANCELLED'
        else:
            assert data['applied_rotations']==1 and selector.counters.states['t'].quantum_remaining==10
            scheduler.schedule();assert selector.counters.states['t'].quantum_remaining==10;outcome='SAVE_COMMIT_PENDING_LOAD'
    undo();return outcome


def main():
    captures=json.loads((REPO/'refine-logs/expert_saturation/outputs/admission_capacity/20260914_load_ready_contract_r01/native_source.json').read_text())
    value=captures['v1/core/sched/scheduler.py'];source_text=value['source'] if isinstance(value,dict) else value
    source=Path('/private/tmp/ltr-native-cpu-scheduler.py');source.write_text(source_text)
    assert hashlib.sha256(source.read_bytes()).hexdigest()==adapter.SCHEDULER_SHA256
    result=dict(status='PASS_CPU_ONLY',selector=selector_cases(),native_growth=native_growth(source),
        native_target_censored=native_growth(source,True),staged=[staged_case(),staged_case(changed=True),staged_case(omit_flush=True),staged_case(foreign=True),staged_case(foreign=True,active=True)],
        scope='Actual private selector/adapter closures and pinned native scheduling/preemption/allocation loop; allocator, transfers and request outputs are CPU fixtures. Install-time vLLM types, native load completion, GPU tensor ownership and EOS remain GPU UNRUN.')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
