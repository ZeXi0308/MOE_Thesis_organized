"""Execute a given-target bridge through pinned native methods on CPU fakes.

Full native schedule executes with the existing AST hold/admission hooks.
Allocation, returned tokens, store generation, hashes and DMA are fake; selected
native finish/reuse/worker barrier methods execute unchanged. No GPU timing,
KV fidelity, autonomous trigger, or full staged-controller qualification.
"""
import argparse
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import MethodType, SimpleNamespace as NS
from unittest.mock import patch

import recovery_execution_share as share
import rotation_native as adapter
import verify_native_recovery_execution as fixture


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract(source, names, namespace, filename):
    methods=[n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.FunctionDef) and n.name in names]
    assert {n.name for n in methods} == set(names) and len(methods)==len(names)
    tree=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+methods,type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree),filename,'exec'),namespace)
    return {name:namespace[name] for name in names}


def execute(source, offload_sources, before, target, expected, pending_store):
    # Reuse only fixture state construction. This test installs its own hooks;
    # do not require the unmerged protect_native_recovery adapter API.
    with patch.object(adapter, 'install', return_value=([], lambda: None)):
        s,_,_=fixture.create(source,deepcopy(before),[],False)
    s.current_step=1376
    fixture.Request.is_finished=lambda r:r.status==fixture.Status.FINISHED_STOPPED
    ns=dict(s.schedule.__func__.__globals__)
    ns['SupportsHMA']=type('SupportsHMA',(),{})
    methods=extract(source.read_text(),['_free_request','_free_blocks','_free_request_blocks',
        '_connector_finished','_build_kv_connector_meta'],ns,str(source))
    for name,method in methods.items():setattr(s,name,MethodType(method,s))
    ns2=dict(s.schedule.__func__.__globals__)
    exec(compile(adapter.patched_schedule_tree(source.read_text()),str(source),'exec'),ns2)
    s.schedule=MethodType(ns2['schedule'],s)
    s.finished_req_ids_dict=None
    m=s.kv_cache_manager
    m.remove_skipped_blocks=lambda **_:None  # Qualified full attention has no out-of-window blocks.
    m.get_block_ids_for_computed_tokens=lambda request_id,**_:([b.block_id for b in s.owned[request_id]],)
    frees=[]
    def free(request):
        ids=[b.block_id for b in s.owned.pop(request.request_id)]
        s.available.extend(ids);frees.append(dict(request=request.request_id,blocks=len(ids)))
    m.free=free

    ons=dict(OffloadingConnectorMetadata=lambda **kw:NS(**kw),ScheduleEndContext=lambda **kw:NS(**kw))
    om=extract(offload_sources['scheduler'],['request_finished','build_connector_meta'],ons,'native-offload-scheduler')
    cs=NS(_req_status={},_jobs={},_block_id_to_pending_jobs={},_current_batch_jobs_to_flush=set(),
          _current_batch_allocated_block_ids=set(),_current_batch_load_jobs={},
          manager=NS(on_request_finished=lambda _:None,on_schedule_end=lambda _:None),
          _maybe_observe_lookup_async_delay=lambda _:None,_update_req_states=lambda _:None,
          _build_store_jobs=lambda _:{})
    for rid,r in s.requests.items():
        cs._req_status[rid]=NS(req=r,req_context=rid,transfer_jobs=set(),update_offload_keys=lambda:None)
    s.connector=NS(request_finished=lambda r,ids:om['request_finished'](cs,r),
                   build_connector_meta=lambda output:om['build_connector_meta'](cs,output))
    alloc=m.allocate_slots
    def allocate_slots(request,amount,**kw):
        result=alloc(request,amount,**kw)
        if result is not None:cs._current_batch_allocated_block_ids.update(b.block_id for b in result.blocks)
        return result
    m.allocate_slots=allocate_slots

    def view(r):
        return share.Request(r.request_id,r.num_tokens,r.num_computed_tokens,len(s.owned[r.request_id]),
                             r.num_output_tokens,r.max_tokens-r.num_output_tokens)
    def state():
        return share.State(view(s.requests[target]),tuple(view(r) for r in s.running if r.request_id!=target),len(s.available))
    current={};finisher=None
    def begin(preempted,timestamp):
        nonlocal current,finisher
        current=share.allocate_completion_bridge(state(),finisher)
        assert current['status']=='READY'
        finisher=current['finisher_id'];s._rotation_forced_count=0;s._rotation_target=target
    s._rotation_begin=begin
    s._rotation_hold=lambda r:r.request_id in current['held']
    s._rotation_target=target;s._rotation_forced_count=0
    trace=[];initial_outputs={rid:r.num_output_tokens for rid,r in s.requests.items()}
    for index in range(16):
        result=s.schedule()
        assert not result.preempted_req_ids and result.num_scheduled_tokens==current['scheduled']
        assert sum(map(len,s.owned.values()))+len(s.available)==before['pool']['usable_blocks']
        planned=expected['calls'][index]
        short=lambda rid:rid.split('/')[1].rsplit('-',1)[0]
        assert {short(k):v for k,v in result.num_scheduled_tokens.items()}==planned['scheduled']
        assert len(s.available)==planned['free_after']
        scheduled_free=len(s.available);completed=[]
        for rid,n in result.num_scheduled_tokens.items():
            r=s.requests[rid];r.num_in_flight_tokens=0
            assert n==1 and r.num_computed_tokens==r.num_tokens
            r.num_output_tokens+=1;r.num_tokens+=1;r.all_token_ids.append(0)
            if r.num_output_tokens==r.max_tokens:
                if pending_store:
                    # Deliberate uncompleted store witnesses the native reuse barrier;
                    # this is not claimed to be the actual alternative GPU transfer.
                    job=900001;ids=[b.block_id for b in s.owned[rid]]
                    cs._jobs[job]=NS(is_store=True,non_sliding_window_block_ids=ids)
                    cs._req_status[rid].transfer_jobs.add(job)
                r.status=fixture.Status.FINISHED_STOPPED;s.running.remove(r)
                s._free_request(r);completed.append(rid)
        trace.append(dict(call=index+1,scheduled=planned['scheduled'],free_after_schedule=scheduled_free,
                          free_after_native_output=len(s.available),completed=completed))
        if completed:
            assert index==15 and completed==[finisher]
    assert frees==[dict(request=finisher,blocks=134)] and finisher not in s.requests
    assert len(s.available)==134
    # One following resident-only native schedule demonstrates allocation reuse.
    s._rotation_begin=lambda preempted,timestamp:None
    s._rotation_hold=lambda request:False
    reuse=s.schedule();meta=reuse.kv_connector_metadata
    assert bool(meta.jobs_to_flush)==pending_store
    assert meta.jobs_to_flush==({900001} if pending_store else set())
    dma=[]
    Spec=type('GPULoadStoreSpec',(),{})
    wm=extract(offload_sources['worker'],['handle_preemptions'],dict(GPULoadStoreSpec=Spec),'native-offload-worker')
    worker=NS(worker=NS(submit_store=lambda job,src,dst:dma.append(['submit_store',job]) or True,
                        wait=lambda jobs:dma.append(['wait',sorted(jobs)])),
              _unsubmitted_store_jobs=[(900001,Spec(),None)] if pending_store else [])
    wm['handle_preemptions'](worker,meta)
    dma.append(['next_gpu_write_boundary_NOT_EXECUTED'])
    if pending_store:assert dma[:2]==[['submit_store',900001],['wait',[900001]]]
    return dict(pending_store_fixture=pending_store,trace=trace,freed=frees,
        target_new_outputs=s.requests[target].num_output_tokens-initial_outputs[target],
        allocation_pool_after_completion=134,reuse_scheduled_requests=len(reuse.num_scheduled_tokens),
        free_after_reuse_allocations=len(s.available),flush_jobs=sorted(meta.jobs_to_flush),dma_boundary_order=dma,
        conditional_source_methods=['Scheduler.schedule','Scheduler._free_request','Scheduler._connector_finished',
          'Scheduler._free_blocks','Scheduler._free_request_blocks','OffloadingScheduler.request_finished',
          'OffloadingScheduler.build_connector_meta','OffloadingWorker.handle_preemptions'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--scheduler-source',type=Path,default=Path('/private/tmp/moe-native-v026-recovery-source/scheduler.py'))
    a=p.parse_args();assert not a.output.exists()
    assert sha(a.scheduler_source)==adapter.SCHEDULER_SHA256
    base=a.repo/'refine-logs/expert_saturation/outputs/admission_capacity'
    raw_path=base/'20260915_natural_native_full_gate_r01/execution_weste_26862/readback/results/diagnostic-native-full/raw.json'
    r=json.loads(raw_path.read_text());before=r['memory_trace'][1377]['before']
    expected_path=base/'20260915_recovery_execution_share_r01/native_full_release_boundary/analysis.json'
    expected=next(v for v in json.loads(expected_path.read_text())['states'] if v['step']==1377)['cap_bridge']
    target=next(rid for rid in before['running_ids'] if '0002680-' in rid)
    offpath=base/'20260914_kv_roundtrip_feasibility_r01/native_offload_source.json'
    off=json.loads(offpath.read_text());sources={name:off['distributed/kv_transfer/kv_connector/v1/offloading/'+name+'.py'] for name in ('scheduler','worker')}
    rows=[execute(a.scheduler_source,sources,before,target,expected,pending) for pending in (False,True)]
    result=dict(status='PASS',evidence='CPU_NATIVE_METHOD_EXECUTION',cases=rows,
        sources={str(p):sha(p) for p in [a.scheduler_source,offpath,raw_path,expected_path,
            Path(__file__),Path(share.__file__),Path(fixture.__file__),Path(adapter.__file__),
            Path(fixture.__file__).with_name('verify_headroom_fast.py'),
            Path(fixture.__file__).with_name('verify_rotation_native.py')]},
        limitations='CPU fake blocks/outputs/hash/store generation/DMA; given post-hoc target and prestate, '
        'upper-layer actions/new admissions held for the slice; not a full staged policy or GPU alternative. '
        'Pending store injected only to test the native flush-before-reuse branch. No waiting time/quality guarantee.')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(status=result['status'],cases=[{k:v for k,v in x.items() if k not in ('trace','conditional_source_methods')} for x in rows]),ensure_ascii=False))


if __name__=='__main__':main()
