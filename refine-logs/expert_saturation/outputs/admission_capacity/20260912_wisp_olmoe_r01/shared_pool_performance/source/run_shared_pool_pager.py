"""Single-stream shared scratch with ordinary cap21 LRU end state; native runner wrapper."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
from types import MethodType
from threading import Lock

from shared_pool_plan import plan_shared_pool


def memory(torch):
    return dict(allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved(),
                peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())


def initialize(runtime):
    if hasattr(runtime,'shared_weights'):return
    torch=runtime.torch
    entries=sorted(runtime.layers.values(),key=lambda e:int(e['layer_name'].split('.')[2]))
    if [e['layer_name'] for e in entries]!=[f'model.layers.{i}.mlp.experts' for i in range(16)]:
        raise RuntimeError('all 16 layers must load before shared scratch initialization')
    if any(e['state'].cap_experts!=21 or e['state'].expert_to_slot for e in entries):
        raise RuntimeError('shared scratch requires empty private21 states')
    sizes={runtime.expert_bytes(e['state']) for e in entries}
    if sizes!={12582912}:raise RuntimeError('unsupported expert row size')
    first=entries[0]['state'];device=first.scratch_w13.device
    shapes=[(384,*t.shape[1:]) for t in (first.scratch_w13,first.scratch_w2)]
    event=dict(before_release=memory(torch),pool_shapes=shapes)
    torch.cuda.synchronize()
    for e in entries:e['state'].scratch_w13=None;e['state'].scratch_w2=None
    torch.cuda.empty_cache();event['after_release']=memory(torch)
    runtime.shared_weights=tuple(torch.empty(s,dtype=torch.bfloat16,device=device) for s in shapes)
    if runtime.shared_qualification:
        for t in runtime.shared_weights:t.fill_(float('nan'))
    for i,e in enumerate(entries):
        e['state'].scratch_w13=runtime.shared_weights[0][i*21:(i+1)*21]
        e['state'].scratch_w2=runtime.shared_weights[1][i*21:(i+1)*21]
    runtime.shared_stream=torch.cuda.current_stream().cuda_stream
    runtime.shared_last_event=torch.cuda.Event();runtime.shared_last_event.record()
    runtime.shared_call_lock=Lock();runtime.shared_transitions=[]
    event.update(after_allocate=memory(torch),stream=runtime.shared_stream)
    runtime.shared_initialization=event


def allocation(runtime):
    initialize(runtime);entries=list(runtime.layers.values());pools=runtime.shared_weights
    storages={(str(t.device),t.untyped_storage().data_ptr()):t.untyped_storage().nbytes() for t in pools}
    if len(storages)!=2 or sum(storages.values())!=4831838208:raise RuntimeError('384-slot unique storage mismatch')
    caps={}
    for e in entries:
        state=e['state'];i=int(e['layer_name'].split('.')[2]);caps[e['layer_name']]=state.cap_experts
        for t,pool in zip((state.scratch_w13,state.scratch_w2),pools):
            if (t.shape[0]!=21 or not t.is_contiguous() or t.untyped_storage().data_ptr()!=pool.untyped_storage().data_ptr()
                or t.storage_offset()!=i*21*pool[0].numel()):raise RuntimeError('private slice alias/offset mismatch')
    if len(caps)!=16 or set(caps.values())!={21}:raise RuntimeError('private capacity mismatch')
    return dict(layer_caps=caps,expert_slots_total=384,private_slots_total=336,shared_slots=48,
        expert_scratch_bytes=sum(storages.values()),private_scratch_view_bytes=4227858432,shared_staging_bytes=603979776,
        legacy_scratch_bytes_scope='private views only; expert_scratch_bytes is unique384 total',shared_pool_mode=runtime.shared_mode,
        expert_scratch_unique_storages=[dict(device=d,pointer=p,bytes=n) for (d,p),n in storages.items()])


def metadata(state):
    return dict(slot_to_expert=list(state.slot_to_expert),expert_to_slot=dict(state.expert_to_slot),
                lru_tick=list(state.lru_tick),lru_clock=state.lru_clock)


def verify(runtime,method,layer,fixed,actual,record):
    torch=runtime.torch;state=runtime.layers[id(layer)]['state'];x,weights,ids=fixed
    refs=tuple(t.to(device=x.device) for t in (state.cpu_w13,state.cpu_w2))
    common=dict(activation=layer.activation,quant_config=method.moe_quant_config,
                apply_router_weight_on_input=layer.apply_router_weight_on_input)
    reference=runtime.kernel(hidden_states=x,w1=refs[0],w2=refs[1],topk_weights=weights,
                            topk_ids=ids,global_num_experts=64,expert_map=None,**common)
    finite=bool(torch.isfinite(actual).all() & torch.isfinite(reference).all())
    delta=actual.float()-reference.float();den=float(torch.linalg.vector_norm(reference.float())) if finite else None
    bit_equal=bool(torch.equal(actual.view(torch.uint8),reference.view(torch.uint8)))
    checks=[]
    resident=sorted(state.expert_to_slot)
    logical=torch.tensor(resident,dtype=torch.long,device=x.device)
    private=torch.tensor([state.expert_to_slot[e] for e in resident],dtype=torch.long,device=x.device)
    for t,ref in zip((state.scratch_w13,state.scratch_w2),refs):
        checks.append(bool(torch.equal(t.index_select(0,private).view(torch.uint8),ref.index_select(0,logical).view(torch.uint8))))
    if runtime.shared_mode=='oneshot':
        active=record['active_experts'];logical=torch.tensor(active,dtype=torch.long,device=x.device)
        physical=torch.tensor([record['shared_plan']['expert_map_device'][e] for e in active],dtype=torch.long,device=x.device)
        for pool,ref in zip(runtime.shared_weights,refs):
            checks.append(bool(torch.equal(pool.index_select(0,physical).view(torch.uint8),ref.index_select(0,logical).view(torch.uint8))))
    row=dict(layer_name=layer.layer_name,call_id=record['call_id'],context=record['context'],rows=len(x),
        actual_group_ranges=[[g['start'],g['stop']] for g in record['groups']],allfinite=finite,
        allclose=bool(torch.allclose(actual,reference,atol=.01,rtol=.01)),bit_equal=bit_equal,
        maxabs=float(delta.abs().max()) if finite else None,relative_l2=float(torch.linalg.vector_norm(delta))/den if den else None,
        reference_l2=den,weight_byte_checks=checks,mode=runtime.shared_mode,
        rtol=.01,atol=.01,scope='Every nonempty measurement call, same pre-call full rows/top-k; private final and current active physical weights checked. Qualification overhead excluded from performance.')
    runtime.validation_results.append(row)
    if not finite or not row['allclose'] or not all(checks) or (runtime.shared_mode=='oneshot' and not bit_equal):
        raise RuntimeError('shared lifecycle numerical/weight qualification failed')


def oneshot(runtime,method,layer,x,weights,ids):
    torch=runtime.torch;entry=runtime.layers[id(layer)];state=entry['state'];started=time.perf_counter()
    if (runtime.summary is not None or runtime._flush_error is not None or x.dtype!=torch.bfloat16
        or x.ndim!=2 or ids.ndim!=2 or weights.shape!=ids.shape or len(x)!=len(ids)):
        raise RuntimeError('invalid lifecycle/tensor state')
    record=dict(call_id=runtime.next_call_id,layer_name=entry['layer_name'],context=dict(runtime.context),
        execution='shared_pool_oneshot',grouping_axis='expert',measurement=runtime.measurement and not runtime.validation_enabled,
        validation_run=runtime.validation_enabled,rows=len(x),groups=[],status='started')
    runtime.records.append(record)
    try:
        t=time.perf_counter();rows=ids.to(device='cpu',dtype=torch.long).tolist()
        active=sorted({e for row in rows for e in row});resident=set(state.expert_to_slot)
        record.update(row_topk_experts=rows,active_experts=active,entry_resident_experts=sorted(resident),
                      route_to_host_ms=(time.perf_counter()-t)*1000,missing_experts=sorted(set(active)-resident))
        t=time.perf_counter();plan=plan_shared_pool(active,metadata(state),int(entry['layer_name'].split('.')[2]))
        record.update(shared_plan=plan,host_plan_ms=(time.perf_counter()-t)*1000)
        validate=runtime.validation_enabled and runtime.measurement and len(x)>0
        fixed=tuple(t.clone() for t in (x,weights,ids)) if validate else None
        state.stats_forward+=1
        if not active:
            record['status']='complete';return torch.empty_like(x)
        group=dict(start=0,stop=len(x),unique_experts=len(active),required_experts=active,
            ensure_experts=active,protected_after=[],loaded_experts=[],reloaded_experts=[],evicted_experts=[],
            miss=0,evict=0,weight_copy_bytes=0,d2d_experts=[],d2d_copy_bytes=0,load_cuda_span_ms=None,
            copy_scope='All D2D before H2D before same-stream kernel; payload counters count completed copy submissions, no overlap credit')
        record['groups'].append(group)
        events=(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True));events[0].record()
        t=time.perf_counter()
        try:
            for c in plan['d2d']:
                for pool in runtime.shared_weights:pool[c['dst']].copy_(pool[c['src']],non_blocking=True)
                group['d2d_experts'].append(c['expert']);group['d2d_copy_bytes']+=runtime.expert_bytes(state)
            for c in plan['h2d']:
                for pool,master in zip(runtime.shared_weights,(state.cpu_w13,state.cpu_w2)):
                    pool[c['dst']].copy_(master[c['expert']],non_blocking=True)
                group['loaded_experts'].append(c['expert']);group['miss']+=1
                group['weight_copy_bytes']+=runtime.expert_bytes(state);state.stats_miss+=1
            final=plan['final_state']
            state.slot_to_expert[:]=final['slot_to_expert'];state.expert_to_slot.clear()
            state.expert_to_slot.update({int(e):s for e,s in final['expert_to_slot'].items()})
            state.lru_tick[:]=final['lru_tick'];state.lru_clock=final['lru_clock']
            state.expert_map_device.copy_(torch.tensor(final['expert_map_device'],dtype=torch.int32,device=x.device))
            mapping=torch.tensor(plan['expert_map_device'],dtype=torch.int32,device=x.device)
            group.update(evicted_experts=plan['entry_evicted_experts'],evict=plan['entry_evict'],
                reloaded_experts=sorted(set(group['loaded_experts'])&entry['ever_loaded']))
            state.stats_evict+=plan['entry_evict'];state.stats_hits+=len(set(active)&resident)
            entry['ever_loaded'].update(group['loaded_experts'])
        finally:
            events[1].record();runtime.events.append((group,events));group['host_ensure_ms']=(time.perf_counter()-t)*1000
        y=runtime.kernel(hidden_states=x.contiguous(),w1=runtime.shared_weights[0],w2=runtime.shared_weights[1],
            topk_weights=weights,topk_ids=ids,global_num_experts=384,expert_map=mapping,
            activation=layer.activation,quant_config=method.moe_quant_config,
            apply_router_weight_on_input=layer.apply_router_weight_on_input)
        record.update(status='complete',loaded_experts=group['loaded_experts'],
            final_resident_experts=sorted(state.expert_to_slot),d2d_copy_bytes=group['d2d_copy_bytes'],
            canonical_group_count=plan['canonical_group_count'],canonical_evict=plan['canonical_evict'])
        if validate:runtime.verify_current_call(method,layer,fixed,y,record)
        return y
    except BaseException as exc:
        record.update(status='failed',error=f'{type(exc).__name__}: {exc}');raise
    finally:
        record.update(host_apply_ms=(time.perf_counter()-started)*1000,group_count=len(record['groups']))
        for key in ('miss','evict','weight_copy_bytes'):record[key]=sum(g.get(key,0) for g in record['groups'])



def ordered_dispatch(runtime,ordinary,method,layer,x,weights,ids):
    initialize(runtime)
    if not runtime.shared_call_lock.acquire(blocking=False):raise RuntimeError('concurrent shared-pool host call')
    current=None
    try:
        current=runtime.torch.cuda.current_stream()
        if current.cuda_stream!=runtime.shared_stream:
            current.wait_event(runtime.shared_last_event)
            runtime.shared_transitions.append(dict(previous=runtime.shared_stream,current=current.cuda_stream,
                next_call_id=runtime.next_call_id,context=dict(runtime.context),layer=layer.layer_name,
                ordering='current waits for previous shared-call event before any shared access'))
            runtime.shared_stream=current.cuda_stream
        protected_warmup=runtime.group_retention!='none'
        if protected_warmup and (runtime.measurement or '/warmup_injection_' not in '/'+str(runtime.context.get('phase',''))):
            raise RuntimeError('protection only allowed on declared ordinary warmup path')
        if runtime.shared_qualification and runtime.measurement:runtime.layers[id(layer)]['validated']=False
        return oneshot(runtime,method,layer,x,weights,ids) if runtime.shared_mode=='oneshot' and not protected_warmup else ordinary(method,layer,x,weights,ids)
    finally:
        try:
            if current is not None:runtime.shared_last_event.record(current)
        finally:runtime.shared_call_lock.release()


def main():
    parser=argparse.ArgumentParser(description=__doc__,add_help=False)
    parser.add_argument('--pool-execution',choices=['split','oneshot'],required=True)
    parser.add_argument('--pool-qualification',action='store_true')
    args,rest=parser.parse_known_args()
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--output',type=Path,required=True)
    output,_=p.parse_known_args(rest);out=output.output.resolve()
    if out.exists():raise FileExistsError(out)
    if '--layer-caps' in rest or '--warmup-executions' in rest or '--execution' not in rest or rest[rest.index('--execution')+1]!='expert':
        raise ValueError('shared pool requires fixed expert execution without layer-caps or token warmup')
    if args.pool_qualification!=('--verify-kernel' in rest):raise ValueError('qualification and verify-kernel must agree')
    import run_native_pager as runner
    import wisp_v026_adapter as pager
    import wisp_expert_groups as groups
    original_install=pager.install;original_groups=groups.install_expert_groups;saved=sys.argv
    sources=Path(__file__).resolve().parent
    names=['run_shared_pool_pager.py','shared_pool_plan.py','analyze_layer_budget.py','run_native_pager.py',
           'wisp_v026_adapter.py','wisp_expert_groups.py','native_pager_context.py','native_capture.py','admission_feedback.py','runtime_variation_observer.py']
    report=dict(status='STARTED',mode=args.pool_execution,qualification=args.pool_qualification,
        sources={n:hashlib.sha256((sources/n).read_bytes()).hexdigest() for n in names},invocation=[sys.executable,*sys.argv],
        warmup_scope='Protected common warmups use unchanged ordinary private21 execution in both pool modes. Measurement forbids protection; none warmups use the selected pool mode.',
        counter_scope='Legacy weight_copy_bytes is H2D only; d2d_copy_bytes is separate in call/group. Canonical evictions are simulated ordinary operations; entry evictions are actual old resident removals.',
        scope='384 total expert slots, 336 private and 48 reusable staging; ordinary cap21 LRU end-state on own actual rows. Each call uses its current stream; stream changes wait for the previous shared-call CUDA event. Concurrent host calls forbidden, no prediction or overlap. Qualifier uses extra references/readback.')
    def install(cap,outdir,layer_caps=None):
        if cap!=24 or layer_caps is not None:raise ValueError('shared pool aggregate requires expert-cap24/no layer override')
        runtime=original_install(cap,outdir,layer_caps)
        runtime.shared_mode=args.pool_execution;runtime.shared_qualification=args.pool_qualification
        runtime.cap_for_layer=MethodType(lambda self,layer:21,runtime)
        runtime.validate_layer_allocations=MethodType(allocation,runtime)
        if args.pool_qualification:runtime.verify_current_call=MethodType(verify,runtime)
        return runtime
    def install_groups(runtime):
        ordinary=original_groups(runtime)
        def dispatch(self,method,layer,x,weights,ids):
            return ordered_dispatch(self,ordinary,method,layer,x,weights,ids)
        runtime.apply=MethodType(dispatch,runtime);return runtime.apply
    pager.install=install;groups.install_expert_groups=install_groups
    sys.argv=[str(sources/'run_native_pager.py'),*rest]
    failure=None
    try:runner.main()
    except BaseException as exc:failure=exc;report['error']=traceback.format_exc()
    finally:
        pager.install=original_install;groups.install_expert_groups=original_groups;sys.argv=saved
        runtime=pager._runtime
        if runtime is not None:
            report.update(initialization=getattr(runtime,'shared_initialization',None),validation=runtime.validation_results,
                          shared_stream=getattr(runtime,'shared_stream',None),shared_stream_transitions=getattr(runtime,'shared_transitions',[]))
            if hasattr(runtime,'shared_weights'):
                try:report['allocation']=allocation(runtime)
                except BaseException as exc:
                    report['allocation_error']=traceback.format_exc();failure=failure or exc
        if args.pool_qualification and (runtime is None or {r['layer_name'] for r in runtime.validation_results}!={f'model.layers.{i}.mlp.experts' for i in range(16)}):
            report['coverage_error']='qualification did not cover every layer';failure=failure or RuntimeError(report['coverage_error'])
        report['status']='FAILED' if failure else ('QUALIFIED_LIFECYCLE_ONLY' if args.pool_qualification else 'COMPLETE')
        out.mkdir(parents=True,exist_ok=True)
        with (out/'shared_pool.json').open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    if failure:raise failure


if __name__=='__main__':main()
