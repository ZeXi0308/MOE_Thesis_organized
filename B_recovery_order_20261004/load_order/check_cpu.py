#!/usr/bin/env python3
"""CPU-only ordering/lifecycle checks against the pinned native submission loop."""
import ast
import hashlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import patch

import load_order as candidate

ROOT = Path(__file__).resolve().parent.parent


class Array(list):
    ndim = 1
    dtype = NS(kind='i')


class CPULoadStoreSpec:
    def __init__(self, ids):
        self.block_ids = Array(ids)


class GPULoadStoreSpec(CPULoadStoreSpec):
    def __init__(self, ids, offset=0):
        super().__init__(ids)
        self.group_sizes, self.block_indices = [len(ids)], [offset]


class FullAttentionSpec:
    sliding_window = attention_chunk_size = None


class Tensor:
    dtype, ndim = 'torch.int8', 2

    def __init__(self, device, rows, columns, pointer):
        self.device, self.shape, self.pointer = NS(type=device), (rows, columns), pointer

    def element_size(self): return 1
    def stride(self, axis): return self.shape[1] if axis == 0 else 1
    def data_ptr(self): return self.pointer


def main():
    for name, sha in [('offloading_worker.py', candidate.WORKER_SHA), ('gpu_worker.py', candidate.HANDLER_SHA)]:
        assert hashlib.sha256((ROOT/'native_sources'/name).read_bytes()).hexdigest() == sha
    tree = ast.parse((ROOT/'native_sources/offloading_worker.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'OffloadingConnectorWorker')
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'start_kv_transfers']
    body = ast.parse('from __future__ import annotations').body + [cls]
    ns = dict(GPULoadStoreSpec=GPULoadStoreSpec)
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), 'pinned-native-start', 'exec'), ns)
    NativeWorker = ns['OffloadingConnectorWorker']

    class CPUOffloadingWorker:
        def __init__(self):
            self.calls, self.inflight, self.fail_id = [], ['prior-inflight-load'], None
            self._load_handler = type('SingleDirectionOffloadingHandler', (), {})()
            h = self._load_handler
            h.gpu_to_cpu, h.src_blocks_per_chunk, h.dst_blocks_per_chunk = False, 2, 1
            h.kv_cache_groups_data_refs = [[NS(tensor_idx=0, page_size_bytes=1024), NS(tensor_idx=1, page_size_bytes=2048)]]
            h.dst_tensors = [Tensor('cuda', 32, 1024, 1000000), Tensor('cuda', 32, 2048, 2000000)]
            h.src_tensors = [Tensor('cpu', 64, 2048, 3000000), Tensor('cpu', 64, 4096, 4000000)]

        def submit_store(self, jid, src, dst):
            self.calls.append(('store', jid, src, dst))
            return True

        def submit_load(self, jid, src, dst):
            self.calls.append(('load', jid, src, dst))
            if jid == self.fail_id:
                raise RuntimeError('native submission error')
            self.inflight.append(jid)
            return True

    package_names = ('numpy', 'vllm', 'vllm.v1', 'vllm.v1.kv_cache_interface', 'vllm.v1.kv_offload',
        'vllm.v1.kv_offload.base', 'vllm.v1.kv_offload.cpu', 'vllm.v1.kv_offload.cpu.common',
        'vllm.distributed', 'vllm.distributed.kv_transfer', 'vllm.distributed.kv_transfer.kv_transfer_state')
    packages = {name: ModuleType(name) for name in package_names}
    packages['numpy'].ndarray = Array
    packages['vllm.v1.kv_cache_interface'].FullAttentionSpec = FullAttentionSpec
    packages['vllm.v1.kv_offload.base'].GPULoadStoreSpec = GPULoadStoreSpec
    packages['vllm.v1.kv_offload.cpu.common'].CPULoadStoreSpec = CPULoadStoreSpec
    state = packages['vllm.distributed.kv_transfer.kv_transfer_state']

    def setup(mode='short_load', change=None, owned=False):
        cw = NativeWorker()
        cw.worker, cw._unsubmitted_store_jobs, cw._load_jobs = CPUOffloadingWorker(), [(99, object(), object())], {}
        cw.spec = type('CPUOffloadingSpec', (), {})()
        connector = type('OffloadingConnector', (), {})()
        connector.connector_worker = cw
        scheduler = NS(connector=connector, kv_cache_config=NS(kv_cache_groups=[NS(
            kv_cache_spec=FullAttentionSpec(), is_eagle_group=False)]))
        state._KV_CONNECTOR_AGENT = connector
        if change: change(scheduler, cw.worker._load_handler)
        count = [0]
        previous = cw.start_kv_transfers
        if owned:
            def previous_wrapper(meta):
                count[0] += 1
                return previous(meta)
            cw.start_kv_transfers = previous_wrapper
        old = cw.start_kv_transfers
        data, undo = candidate.install(scheduler, mode)
        return cw, data, undo, old, count

    def metadata():
        jobs = {30:NS(req_id='a', src_spec=CPULoadStoreSpec([0, 1]), dst_spec=GPULoadStoreSpec([1, 2, 3])),
                31:NS(req_id='b', src_spec=CPULoadStoreSpec([2]), dst_spec=GPULoadStoreSpec([4])),
                32:NS(req_id='c', src_spec=CPULoadStoreSpec([3]), dst_spec=GPULoadStoreSpec([5])),
                33:NS(req_id='d', src_spec=CPULoadStoreSpec([4]), dst_spec=GPULoadStoreSpec([6, 7]))}
        return NS(load_jobs=jobs, store_jobs={'untouched':object()}, jobs_to_flush={99})

    def source(cls):
        return str(ROOT/'native_sources'/('offloading_worker.py' if cls is NativeWorker else 'gpu_worker.py'))

    with patch.dict(sys.modules, packages), patch.object(candidate.inspect, 'getsourcefile', side_effect=source):
        for mode in ('native', 'short_load'):
            cw, data, undo, old, count = setup(mode, owned=True)
            meta = metadata(); original = meta.load_jobs; entries = dict(original); stores = meta.store_jobs
            cw.start_kv_transfers(meta)
            expected = [30,31,32,33] if mode == 'native' else [31,32,33,30]
            assert [x[1] for x in cw.worker.calls] == [99, *expected]
            assert cw.worker.inflight == ['prior-inflight-load', *expected]
            assert cw._load_jobs == {jid:entries[jid].req_id for jid in expected}
            assert count[0] == 1 and meta.load_jobs is original and list(original) == [30,31,32,33]
            assert meta.store_jobs is stores and meta.jobs_to_flush == {99}
            assert all(src is entries[jid].src_spec and dst is entries[jid].dst_spec
                       for kind,jid,src,dst in cw.worker.calls if kind == 'load')
            event = data['events'][0]
            assert event['after'] == expected and event['bytes'] == {30:9216,31:3072,32:3072,33:6144}
            assert event['fallback'] is None and event['decision_s'] >= 0
            undo(); undo(); assert cw.start_kv_transfers is old

        mutations = [lambda m:setattr(m.load_jobs[31].dst_spec, 'block_ids', Array([2])),
            lambda m:setattr(m.load_jobs[30].dst_spec, 'block_ids', Array([1,1,3])),
            lambda m:setattr(m.load_jobs[30], 'src_spec', object()),
            lambda m:setattr(m.load_jobs[30].dst_spec, 'group_sizes', [2]),
            lambda m:setattr(m.load_jobs[31].dst_spec, 'block_ids', Array([32])),
            lambda m:setattr(m.load_jobs[30].src_spec, 'block_ids', Array([0]))]
        for mutate in mutations:
            cw,data,undo,_,_=setup(); meta=metadata(); mutate(meta); original=meta.load_jobs
            cw.start_kv_transfers(meta)
            assert data['events'][0]['fallback'] and data['events'][0]['after'] == [30,31,32,33]
            assert meta.load_jobs is original and [x[1] for x in cw.worker.calls] == [99,30,31,32,33]
            undo(); assert 'start_kv_transfers' not in vars(cw)

        for change in [lambda s,h:s.kv_cache_config.kv_cache_groups.append(s.kv_cache_config.kv_cache_groups[0]),
                       lambda s,h:setattr(h.dst_tensors[1], 'pointer', 1001024),
                       lambda s,h:setattr(h.kv_cache_groups_data_refs[0][1], 'tensor_idx', 0)]:
            cw,data,undo,_,_=setup(change=change);meta=metadata();cw.start_kv_transfers(meta)
            assert data['layout_fallback'] and data['events'][0]['fallback']
            assert [x[1] for x in cw.worker.calls] == [99,30,31,32,33]
            undo()

        for n in (0,1):
            cw,data,undo,_,count=setup(owned=True);meta=metadata();meta.load_jobs=dict(list(meta.load_jobs.items())[:n])
            original=meta.load_jobs;cw.start_kv_transfers(meta)
            assert count[0] == 1 and meta.load_jobs is original and not data['events']
            undo()
        cw,data,undo,_,count=setup(owned=True);meta=metadata();original=meta.load_jobs;cw.worker.fail_id=32
        try: cw.start_kv_transfers(meta)
        except RuntimeError as error: assert str(error) == 'native submission error'
        else: raise AssertionError('Native error was swallowed')
        assert count[0] == 1 and meta.load_jobs is original
        assert [x[1] for x in cw.worker.calls] == [99,31,32]
        undo()
        with patch.object(candidate.inspect, 'getsourcefile', return_value=str(Path(__file__))):
            try: setup()
            except RuntimeError as error: assert 'source changed' in str(error)
            else: raise AssertionError('Unknown source accepted')
    print('CPU_PASS: pinned native submit loop, stable/equal/partial bytes, alias/unknown fallback, exact-once STORE/LOAD, mapping identity/error restore, source guard; GPU_UNRUN')


if __name__ == '__main__':
    main()
