"""Exercise wrapper against exact native handle_preemptions, without CUDA imports."""
import ast
import itertools
import json
from pathlib import Path
import sys
import types
sys.dont_write_bytecode = True
root = Path(__file__).resolve().parent
sys.path.insert(0, str(root/'pkg'))
from recovery_order import install, order_stores
N = types.SimpleNamespace
class GPULoadStoreSpec:
    pass
src = GPULoadStoreSpec()
tree = ast.parse((root/'native_sources/offloading_worker.py').read_text())
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'OffloadingConnectorWorker')
method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'handle_preemptions')
method.args.args[1].annotation = None
ns = {'GPULoadStoreSpec': GPULoadStoreSpec}
exec(compile(ast.Module(body=[method], type_ignores=[]), 'native_handle', 'exec'), ns)
checks = 0
for mode in ('native', 'age', 'flush_first'):
    submitted, waited = [], []
    worker = type('CPUOffloadingWorker', (), {})()
    worker.submit_store = lambda jid, s, d: submitted.append(jid) is None
    worker.submit_load = lambda *a: True
    worker.get_finished = lambda: []
    worker.wait = lambda jids: waited.append(set(jids))
    cw = N(worker=worker, _unsubmitted_store_jobs=[(1, src, N(block_ids=[1])), (2, src, N(block_ids=[2]))], start_kv_transfers=lambda m: None)
    cw.handle_preemptions = types.MethodType(ns['handle_preemptions'], cw)
    cs = N(_jobs={i: N(req_id=f'r{i}', is_store=True) for i in (1, 2, 3)}, build_connector_meta=lambda out: None,
           update_connector_output=lambda out: None, get_num_new_matched_tokens=lambda r, c: (0, False))
    scheduler = N(connector=N(connector_scheduler=cs), _preempt_request=lambda *a: None,
                  schedule=lambda: None, kv_cache_manager=N(allocate_slots=lambda *a: None))
    sys.modules['vllm.distributed.kv_transfer.kv_transfer_state'] = N(_KV_CONNECTOR_AGENT=N(connector_worker=cw))
    data, undo = install(scheduler, mode)
    third = N(src_spec=src, dst_spec=N(block_ids=[3]))
    unrelated = N(src_spec=src, dst_spec=N(block_ids=[4]))
    meta = N(jobs_to_flush={3}, store_jobs={3: third, 4: unrelated})
    cw.handle_preemptions(meta)
    assert submitted == ([3, 1, 2] if mode == 'flush_first' else [1, 2, 3])
    assert waited == [{3}] and not cw._unsubmitted_store_jobs
    assert meta.store_jobs == {4: unrelated} and meta.jobs_to_flush == {3}
    assert len([e for e in data['events'] if e['kind'] == 'ready']) == 3
    undo()
    checks += 1
for perm in itertools.permutations(range(4)):
    es = [(i, src, N(block_ids=[i])) for i in perm]
    for mask in range(16):
        critical = {i for i in range(4) if mask >> i & 1}
        got, why = order_stores(es, critical, 'flush_first')
        assert [x[0] for x in got] == [i for i in perm if i in critical] + [i for i in perm if i not in critical]
        assert {id(e) for e in got} == {id(e) for e in es} and why is None
        checks += 1
collision = [(1, src, N(block_ids=[4])), (2, src, N(block_ids=[4]))]
assert order_stores(collision, {2}, 'flush_first') == (collision, 'overlapping_destination')
unknown = [(1, src, N()), (2, src, N())]
assert order_stores(unknown, {2}, 'age') == (unknown, 'unknown_destination')
print(json.dumps(dict(status='PASS', checks=checks+2, scope='CPU native wrapper lifecycle + immutable tasks + alias fallback; GPU UNRUN')))
