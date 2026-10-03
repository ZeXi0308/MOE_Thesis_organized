"""Shadow qualification of the 384-slot expert-map interface; no shared pager lifecycle."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time
import traceback


def fingerprint(tensor):
    data=tensor.detach()
    import torch
    array=data.contiguous().view(torch.uint8).cpu().numpy()
    return dict(shape=list(tensor.shape),dtype=str(tensor.dtype),sha256=hashlib.sha256(memoryview(array)).hexdigest())


def physical_slots(layer_index):
    if type(layer_index) is not int or not 0<=layer_index<16: raise ValueError('invalid OLMoE layer index')
    return list(range(layer_index*21,layer_index*21+21))+list(range(336,379))


def errors(actual, reference, torch):
    finite=bool(torch.isfinite(actual).all() & torch.isfinite(reference).all())
    delta=actual.float()-reference.float();den=float(torch.linalg.vector_norm(reference.float())) if finite else None
    num=float(torch.linalg.vector_norm(delta)) if finite else None
    return dict(allfinite=finite,maxabs=float(delta.abs().max()) if finite else None,reference_l2=den,
        relative_l2=num/den if den else (0.0 if num==0 else None),
        bit_equal=bool(torch.equal(actual.contiguous().view(torch.uint8),reference.contiguous().view(torch.uint8))),
        allclose=bool(torch.allclose(actual,reference,rtol=.01,atol=.01)),rtol=.01,atol=.01)


def memory(torch, device):
    return dict(allocated_bytes=torch.cuda.memory_allocated(device),reserved_bytes=torch.cuda.memory_reserved(device),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device),peak_reserved_bytes=torch.cuda.max_memory_reserved(device))


def shadow(runtime, method, layer, fixed, record, item):
    torch=runtime.torch;state=runtime.layers[id(layer)]['state'];x,weights,ids=fixed
    index=int(layer.layer_name.split('.')[2]);slots=physical_slots(index)
    assert state.num_experts==64 and len(x)>0 and ids.shape==weights.shape and ids.shape[1]==8
    masters=(state.cpu_w13,state.cpu_w2)
    assert [list(t.shape) for t in masters]==[[64,2048,2048],[64,2048,1024]]
    item.update(layer_index=index,context=record['context'],input_rows=len(x),logical_to_physical=slots,
        private_slots=slots[:21],shared_slots=slots[21:],unused_slots='All other slots contain NaN',
        cpu_master_sources=[fingerprint(t) for t in masters],cases=[],memory_before=memory(torch,x.device))
    references=tuple(t.to(device=x.device) for t in masters)
    pools=tuple(torch.full((384,*t.shape[1:]),float('nan'),dtype=t.dtype,device=x.device) for t in masters)
    mapping=torch.tensor(slots+[-1]*320,dtype=torch.int32,device=x.device)
    selection=torch.tensor(slots,dtype=torch.long,device=x.device)
    for master,pool in zip(masters,pools):
        for logical,physical in enumerate(slots):pool[physical].copy_(master[logical],non_blocking=True)
    def same_rows():
        return [bool(torch.equal(pool.index_select(0,selection).view(torch.uint8),ref.view(torch.uint8)))
                for pool,ref in zip(pools,references)]
    item.update(weight_rows_byte_equal_before=same_rows(),memory_with_pools=memory(torch,x.device),
        pool_shapes=[list(t.shape) for t in pools],pool_strides=[list(t.stride()) for t in pools],
        pool_storage_bytes=sum(t.untyped_storage().nbytes() for t in pools),
        reference_storage_bytes=sum(t.untyped_storage().nbytes() for t in references),
        map_storage_bytes=mapping.untyped_storage().nbytes(),selection_storage_bytes=selection.untyped_storage().nbytes())
    common=dict(activation=layer.activation,quant_config=method.moe_quant_config,
                apply_router_weight_on_input=layer.apply_router_weight_on_input)
    widths=sorted({min(n,len(x)) for n in (2,10,64)})
    item['expected_widths']=widths
    for width in widths:
        inputs=tuple(t[:width].contiguous() for t in (x,weights,ids));xx,ww,ii=inputs
        case=dict(rows=width,input_sources={k:fingerprint(t) for k,t in zip(('x','topk_weights','topk_ids'),inputs)},
                  topk_ids=ii.cpu().tolist(),status='started');item['cases'].append(case)
        reference=runtime.kernel(hidden_states=xx,w1=references[0],w2=references[1],topk_weights=ww,
                                 topk_ids=ii,global_num_experts=64,expert_map=None,**common)
        actual=runtime.kernel(hidden_states=xx,w1=pools[0],w2=pools[1],topk_weights=ww,
                              topk_ids=ii,global_num_experts=384,expert_map=mapping,**common)
        case.update(status='complete',**errors(actual,reference,torch))
        if index==0 and width==10:
            active=sorted({e for row in case['topk_ids'] for e in row});assert len(active)>=2
            a,b=active[:2];wrong=mapping.clone();wrong[a]=slots[b];wrong[b]=slots[a]
            negative=runtime.kernel(hidden_states=xx,w1=pools[0],w2=pools[1],topk_weights=ww,
                                    topk_ids=ii,global_num_experts=384,expert_map=wrong,**common)
            item['negative_control']=dict(logical_experts=[a,b],original_physical_slots=[slots[a],slots[b]],
                rows=width,pair_rule='two smallest active logical IDs; chosen before observing negative output',
                expected='finite and not allclose',**errors(negative,reference,torch))
            del negative,wrong
        case['memory_after']=memory(torch,x.device)
        del reference,actual,inputs,xx,ww,ii
    item.update(weight_rows_byte_equal_after=same_rows(),memory_before_release=memory(torch,x.device))
    item['allocated_with_pools_minus_entry_bytes']=item['memory_with_pools']['allocated_bytes']-item['memory_before']['allocated_bytes']
    del pools,references,mapping,selection,pool,master
    item['memory_after_release']=memory(torch,x.device)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('output','prepared','model'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve()
    if out.exists():parser.error('output directory must not already exist')
    import run_native_pager as runner
    import wisp_v026_adapter as pager
    source=Path(__file__).resolve().parent
    names=('qualify_shared_expert_pool.py','run_native_pager.py','wisp_v026_adapter.py','wisp_expert_groups.py',
           'native_pager_context.py','native_capture.py','admission_feedback.py')
    report=dict(schema='shared_expert_pool_kernel_qualification_v1',status='STARTED',started_unix_s=time.time(),
        invocation=[sys.executable,*sys.argv],sources={n:hashlib.sha256((source/n).read_bytes()).hexdigest() for n in names},
        layers=[],issues=[],scope=[
            'Independent shadow calls on actual current input/top-k and real CPU weights; original verification executes first and original model output is returned unchanged.',
            'Per-layer 21 private plus 43 shared physical slots; all 64 experts mapped into two 384-slot shadow tensors. global_num_experts=384 describes the operator indexing domain, not a 384-expert model.',
            'Qualification allocates additional pool/reference/sorting/workspaces and performs synchronization/readback. No same-budget performance, cross-layer shared-pool lifecycle, request-quality or bitwise guarantee is established.',
            'Allocator peaks retain the runner high-water scope and are never reset here. Allocated deltas are boundary observations, not isolated peak attribution.'])
    original=pager._Runtime.verify_current_call;saved_argv=sys.argv;failure=None
    def verify(runtime, method, layer, fixed, actual, record):
        item=dict(layer_name=layer.layer_name,call_id=record['call_id'],status='started');report['layers'].append(item)
        try:
            original(runtime,method,layer,fixed,actual,record)
            if 'runtime_sources' not in report:
                report['runtime_sources']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                          (Path(runtime.upstream.__file__),Path(inspect.getfile(runtime.kernel)))}
            with runtime.torch.inference_mode():shadow(runtime,method,layer,fixed,record,item)
            item['status']='complete'
        except BaseException:
            item.update(status='failed',error=traceback.format_exc());raise
    pager._Runtime.verify_current_call=verify
    sys.argv=[str(source/'run_native_pager.py'),'--output',str(out),'--prepared',str(args.prepared),'--model',str(args.model),
              '--expert-cap','24','--execution','expert','--verify-kernel','--requests','4','--prompt-tokens','64',
              '--output-tokens','8','--token-budget','160','--kv-bytes','1073741824']
    report['runner_argv']=list(sys.argv)
    try:runner.main()
    except BaseException as exc:failure=exc;report['execution_error']=traceback.format_exc()
    finally:
        pager._Runtime.verify_current_call=original;sys.argv=saved_argv
        native=pager._runtime.validation_results if pager._runtime is not None else []
        report['native_validation']=native
        if len(native)!=16 or not all(r['allfinite'] and r['allclose'] for r in native):report['issues'].append('original kernel qualification incomplete/failed')
        if {r['layer_name'] for r in report['layers']}!={f'model.layers.{i}.mlp.experts' for i in range(16)} or len(report['layers'])!=16:report['issues'].append('16-layer shadow coverage incomplete')
        negatives=[]
        for row in report['layers']:
            if (row['status']!='complete' or row.get('weight_rows_byte_equal_before')!=[True,True] or
                row.get('weight_rows_byte_equal_after')!=[True,True] or [c['rows'] for c in row.get('cases',[])]!=row.get('expected_widths') or
                not all(c.get('allfinite') and c.get('allclose') for c in row.get('cases',[]))):report['issues'].append(row['layer_name']+': shadow/weight qualification failed')
            if 'negative_control' in row:negatives.append(row['negative_control'])
        if len(negatives)!=1 or not all(n['allfinite'] and not n['allclose'] for n in negatives):report['issues'].append('predeclared layer0/M10 negative control missing/failed')
        if failure is not None:report['issues'].append('runner or shadow execution failed')
        report.update(status='FAILED' if report['issues'] else 'PASSED_KERNEL_INTERFACE_ONLY',finished_unix_s=time.time())
        out.mkdir(parents=True,exist_ok=True)
        if (out/'status.json').exists():report['runner_status']=json.loads((out/'status.json').read_text())
        with (out/'shared_pool_qualification.json').open('x') as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
    if failure is not None:raise failure
    if report['issues']:raise SystemExit(1)


if __name__=='__main__':main()
