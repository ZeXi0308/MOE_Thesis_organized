"""Execute sealed native worker methods with a fake transfer device, CPU only."""
import ast, hashlib, json
from pathlib import Path
from types import SimpleNamespace as NS


def check(source):
    tree=ast.parse(source)
    wanted={'handle_preemptions','start_kv_transfers','prepare_store_kv'}
    methods=[n for c in tree.body if isinstance(c,ast.ClassDef)
             for n in c.body if isinstance(n,ast.FunctionDef) and n.name in wanted]
    assert {m.name for m in methods}==wanted
    cls=ast.ClassDef(name='NativeMethods',bases=[],keywords=[],body=methods,decorator_list=[])
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),cls],type_ignores=[])
    class GPUSpec:pass
    ns={'GPULoadStoreSpec':GPUSpec}
    exec(compile(ast.fix_missing_locations(module),'<sealed-native-worker-methods>','exec'),ns)
    records=[]
    for pending in [False,True]:
        obj=ns['NativeMethods']();log=[]
        def store(j,s,d):log.append(['submit_store',j]);return True
        def wait(js):log.append(['wait',sorted(js)])
        def load(j,s,d):log.append(['submit_load',j]);return True
        obj.worker=NS(submit_store=store,wait=wait,submit_load=load)
        obj._load_jobs={};obj._unsubmitted_store_jobs=[]
        entry=lambda: NS(src_spec=GPUSpec(),dst_spec=GPUSpec(),req_id='victim')
        # Previous iteration records a store but does not submit it yet.
        obj.prepare_store_kv(NS(store_jobs={11:entry()} if pending else {}))
        assert not log
        meta=NS(jobs_to_flush={11} if pending else set(),store_jobs={},load_jobs={21:entry()})
        obj.handle_preemptions(meta)
        log.append(['reuse_blocks_boundary'])
        obj.start_kv_transfers(meta)
        expected=([['submit_store',11],['wait',[11]]] if pending else [])
        assert log==expected+[['reuse_blocks_boundary'],['submit_load',21]]
        records.append(dict(pending=pending,events=log))
    obj=ns['NativeMethods']();log=[];obj._unsubmitted_store_jobs=[]
    obj.worker=NS(submit_store=lambda j,s,d:log.append(['submit_store',j]) or True,
                  wait=lambda js:log.append(['wait',sorted(js)]))
    meta=NS(jobs_to_flush={31},store_jobs={31:NS(src_spec=GPUSpec(),dst_spec=GPUSpec())})
    obj.handle_preemptions(meta)
    assert log==[['submit_store',31],['wait',[31]]] and not meta.store_jobs
    records.append(dict(same_batch_forced_store=log))
    return dict(status='PASS',native_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        cases=records,scope='Real sealed Python methods with synchronous fake transfer backend. Reuse marker is a fixture boundary, not real GPU execution. No data correctness, scheduler integration or physical transfer timing validated.')


if __name__=='__main__':
    base=Path('refine-logs/expert_saturation/outputs/admission_capacity')
    sources=json.loads((base/'20260914_kv_roundtrip_feasibility_r01/native_offload_source.json').read_text())
    result=check(sources['distributed/kv_transfer/kv_connector/v1/offloading/worker.py'])
    out=base/'20260914_native_store_contract_r01';out.mkdir(exist_ok=True)
    (out/'cpu_fence_check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result['status'])
