"""Real pre-action resource state + native store-builder CPU interface fixture."""
import ast, hashlib, json, math
from itertools import chain
from pathlib import Path
from types import SimpleNamespace as NS


BASE=Path('refine-logs/expert_saturation/outputs/admission_capacity')
sources=json.loads((BASE/'20260914_kv_roundtrip_feasibility_r01/native_offload_source.json').read_text())
source=sources['distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py']
tree=ast.parse(source);names={'_build_store_jobs','_calc_num_offloadable_tokens'}
methods=[m for c in tree.body if isinstance(c,ast.ClassDef) for m in c.body
         if isinstance(m,ast.FunctionDef) and m.name in names]
class GPUSpec:
    def __init__(self, block_ids, **kwargs):self.block_ids=block_ids;self.layout=kwargs
ns=dict(chain=chain,GPULoadStoreSpec=GPUSpec,TransferJobStatus=NS,TransferJob=NS,
        logger=NS(debug=lambda *a:None))
module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),
    ast.ClassDef(name='Builder',bases=[],keywords=[],body=methods,decorator_list=[])],type_ignores=[])
exec(compile(ast.fix_missing_locations(module),'<sealed-native-store-builder>','exec'),ns)
rows=[]
for block in [0,1]:
    path=BASE/f'20260914_d6_action_branches_r01/readback/results/block{block}-least/raw.json'
    raw=json.loads(path.read_text());before=raw['memory_trace'][328]['before']
    ids=raw['internal_to_source']
    # Select with step-328 visible output only, not the future victim event.
    victim=min(before['running_ids'],key=lambda rid:before['requests'][rid]['output_tokens'])
    state=before['requests'][victim]
    executed=next(r for r in raw['scheduler_steps'][328]['scheduled'] if r['request_id']==ids[victim])
    assert executed['scheduled_tokens']==1
    after_compute=state['computed_tokens']+1
    # Block IDs are synthetic interface inputs; counts, lengths and resource
    # accounting come from raw. This does not instantiate the GPU block map.
    blocks=list(range(1,state['block_counts'][0]+1))
    req=NS(num_computed_tokens=state['computed_tokens'],num_tokens=state['prompt_tokens']+state['output_tokens'],
           num_prompt_tokens=state['prompt_tokens'],is_finished=lambda:False)
    results={}
    for prompt_only in [True,False]:
        group=NS(block_ids=blocks,offload_keys=list(range(len(blocks))),next_stored_chunk_idx=0)
        rs=NS(req=req,max_offload_tokens=None,group_states=[group],transfer_jobs=set(),req_context=NS(),
              storable_chunks=lambda cfg,n:n//16)
        obj=ns['Builder']();obj.config=NS(blocks_per_chunk=1,num_workers=1,offload_prompt_only=prompt_only,
            kv_group_configs=[NS(alignment_chunk_count=None,sliding_window_size_in_chunks=None)])
        obj._req_status={victim:rs};obj._jobs={};obj._events_tracker=NS(record_store=lambda *a:None)
        obj._touch=lambda r:None;obj._generate_job_id=lambda:1
        obj.manager=NS(prepare_store=lambda keys,ctx:NS(keys_to_store=keys,store_spec='FAKE_HOST_ALLOCATION'))
        jobs=obj._build_store_jobs(NS(num_scheduled_tokens={victim:1},finished_req_ids=set()))
        assert len(jobs)==1
        saved=len(jobs[1].src_spec.block_ids)*16
        results[str(prompt_only)]=dict(saved_tokens=saved,unsaved_computed_tail=after_compute-saved,
            store_bytes=saved*131072,job_registered=1 in rs.transfer_jobs)
        # A victim absent from scheduled and finished sets creates no job.
        assert not obj._build_store_jobs(NS(num_scheduled_tokens={},finished_req_ids=set()))
    event=next(e for e in raw['preemption_events'] if e['attempted_step']==329)
    assert event['victim_internal_request_id']==victim
    waiting=[rid for rid in before['requests'] if rid not in before['running_ids']]
    assert len(waiting)==1
    target=before['requests'][waiting[0]]
    need=math.ceil((target['prompt_tokens']+target['output_tokens'])/16)-target['block_counts'][0]
    free_after=raw['memory_trace'][328]['after']['pool']['free_blocks']
    rows.append(dict(block=block,victim=ids[victim],selection_uses_step=328,
        matches_later_observed_victim=True,free_before=before['pool']['free_blocks'],
        free_after=free_after,victim_blocks=len(blocks),
        target_full_history_need=need,free_plus_victim_minus_target=free_after+len(blocks)-need,
        saved=results,raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
out=BASE/'20260914_native_store_contract_r01'
(out/'selective_boundary.json').write_text(json.dumps(dict(status='CPU_CONDITIONAL_FEASIBILITY',rows=rows,
    source_sha256=hashlib.sha256(source.encode()).hexdigest(),scope='Native builder methods; fake host allocator, offload keys and GPU block IDs. Raw establishes resource counts only. No connector model execution, physical transfers, host availability or policy-specific future validated.'),indent=2)+'\n')
print(json.dumps(rows,indent=2))
