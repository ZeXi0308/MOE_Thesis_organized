#!/usr/bin/env python3
"""Targeted lifecycle fixture using the exact pinned native context-manager AST."""
import ast
from contextlib import contextmanager, nullcontext, AbstractContextManager
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace as NS, ModuleType
from typing import Generator
from unittest.mock import patch

import completion_finalize as candidate

ROOT = Path(__file__).resolve().parent.parent


def main():
    for name, sha in [('gpu_model_runner.py', candidate.RUNNER_SHA),
                      ('kv_connector_model_runner_mixin.py', candidate.MIXIN_SHA)]:
        assert hashlib.sha256((ROOT/'native_sources'/name).read_bytes()).hexdigest() == sha
    source = (ROOT/'native_sources/kv_connector_model_runner_mixin.py').read_text()
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef))
    keep = {'_get_kv_connector_output', 'maybe_get_kv_connector_output',
            'finalize_kv_connector', 'kv_connector_no_forward'}
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in keep]
    log, active = [], [False]

    class Output:
        @staticmethod
        def with_kv_conn_output_only(value):
            return value

    class Group:
        def bind_connector_metadata(self, metadata): log.append('bind')
        def start_load_kv(self, context): log.append('load')
        def wait_for_save(self): log.append('wait')
        def get_finished(self, ids):
            log.append('query')
            return set(), {'load'} if 'sample_sync' in log else set()
        def get_block_ids_with_load_errors(self): log.append('errors'); return set()
        def get_kv_connector_stats(self): log.append('stats'); return None
        def get_kv_connector_kv_cache_events(self): log.append('events'); return None
        def build_connector_worker_meta(self): log.append('build'); return {'original': True}
        def clear_connector_metadata(self): log.append('clear')

    group = Group()

    @contextmanager
    def forward(*args):
        active[0] = True
        try: yield
        finally: active[0] = False

    def forward_context():
        assert active[0], 'Native start_load must remain inside forward context'
        return object()

    ns = dict(contextmanager=contextmanager, nullcontext=nullcontext,
        Generator=Generator, AbstractContextManager=AbstractContextManager,
        KVConnectorOutput=Output, ModelRunnerOutput=Output, KVConnectorBase=Group,
        VllmConfig=object, get_kv_transfer_group=lambda:group, has_kv_transfer_group=lambda:True,
        get_forward_context=forward_context, set_forward_context=forward)
    body = ast.parse('from __future__ import annotations').body + [cls]
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])),
                 'pinned-native-context', 'exec'), ns)
    Mixin = ns['KVConnectorModelRunnerMixin']

    packages = {name: ModuleType(name) for name in ('vllm','vllm.v1','vllm.v1.worker',
        'vllm.v1.worker.kv_connector_model_runner_mixin','vllm.distributed','vllm.distributed.kv_transfer')}
    packages['vllm'].__version__ = '0.26.0'
    packages['vllm.v1.worker.kv_connector_model_runner_mixin'].KVConnectorModelRunnerMixin = Mixin
    connector = type('OffloadingConnector', (), {})()
    connector.connector_worker = NS(spec=type('CPUOffloadingSpec', (), {})(), worker=type('CPUOffloadingWorker', (), {})())
    packages['vllm.distributed.kv_transfer'].get_kv_transfer_group = lambda:connector
    checked = type('GPUModelRunner', (), {})()
    checked.model_config = NS(model='allenai/OLMoE-1B-7B-0924', revision='6d84c48581ece794365f2b8e9cfb043c68ade9c5')
    checked.parallel_config = NS(tensor_parallel_size=1,pipeline_parallel_size=1,data_parallel_size=1)
    checked.scheduler_config = NS(async_scheduling=False)
    checked.use_async_scheduling=False;checked.speculative_config=None;checked.is_pooling_model=False
    checked.broadcast_pp_output=False;checked.supports_mm_inputs=False;checked.dtype='torch.bfloat16'
    with patch.dict(sys.modules, packages), patch.object(candidate.inspect,'getsourcefile',
            side_effect=lambda cls:str(ROOT/'native_sources'/('gpu_model_runner.py' if cls is type(checked) else 'kv_connector_model_runner_mixin.py'))):
        assert candidate._guard(checked)['GPUModelRunner']==candidate.RUNNER_SHA
        checked.model_config.model='/root/autodl-tmp/moe-a-20261002/hf/hub/models--allenai--OLMoE-1B-7B-0924/snapshots/6d84c48581ece794365f2b8e9cfb043c68ade9c5'
        assert candidate._guard(checked)['GPUModelRunner']==candidate.RUNNER_SHA
        checked.model_config.model='/another/model'
        try:candidate._guard(checked)
        except RuntimeError:pass
        else:raise AssertionError('Unauthorized model path accepted')
        checked.model_config.model='allenai/OLMoE-1B-7B-0924'
        for field in ('use_async_scheduling','is_pooling_model','broadcast_pp_output','supports_mm_inputs'):
            setattr(checked,field,True)
            try:candidate._guard(checked)
            except RuntimeError:pass
            else:raise AssertionError('Unsupported runtime accepted: '+field)
            setattr(checked,field,False)
        connector.connector_worker.spec=object()
        try:candidate._guard(checked)
        except RuntimeError:pass
        else:raise AssertionError('Non-CPU offload spec accepted')

    class Runner(Mixin):
        def execute_model(self, fault=None, count=1, defer_finalize=False):
            self.fault, self.defer = fault, defer_finalize
            schedule = NS(total_num_scheduled_tokens=count, kv_connector_metadata=object(), finished_req_ids=set())
            if count == 0:
                return self.kv_connector_no_forward(schedule, None)
            with forward(), self.maybe_get_kv_connector_output(schedule, defer_finalize=defer_finalize) as self.output:
                log.append('forward')
                if fault == 'forward': raise RuntimeError(fault)
            if fault == 'execute': raise RuntimeError(fault)
            return self.output if fault == 'unexpected_return' else None
        def _bookkeeping_sync(self):
            if self.fault == 'bookkeeping': raise RuntimeError(self.fault)
            log.append('sample_sync')
            return 7
        def sample_tokens(self):
            if self.fault == 'sample_before': raise RuntimeError(self.fault)
            if self.fault != 'skip_bookkeeping': self._bookkeeping_sync()
            if self.defer: self.finalize_kv_connector()
            if self.fault == 'sample_after': raise RuntimeError(self.fault)
            return self.output

    def installed(mode='after_sample'):
        log.clear(); runner=Runner()
        data, uninstall=candidate._install_runner(runner, mode, {'fixture':'pinned native context'})
        return runner,data,uninstall

    def once():
        assert all(log.count(x)==1 for x in ('bind','load','query','build','clear')), log

    for mode in ('native','after_sample'):
        r,data,undo=installed(mode);r.execute_model();original=r.output
        assert log.count('query')==(mode=='native')
        assert r.sample_tokens() is original
        once();row=data['events'][0]
        assert data['instrumented_context_enters']==data['instrumented_native_finalizations']==1
        assert (log.index('query')>log.index('sample_sync'))==(mode=='after_sample')
        assert (row['finalize_begin_perf_s']>=row['bookkeeping_return_perf_s'])==(mode=='after_sample')
        undo();undo();once()
        assert not any(name in vars(r) for name in ('execute_model','sample_tokens','_bookkeeping_sync','maybe_get_kv_connector_output'))
    r,_,undo=installed();assert r.execute_model(count=0) is not None;once();undo()
    r,data,undo=installed();r.execute_model(defer_finalize=True);assert log.count('query')==1
    assert 'clear' not in log;r.sample_tokens();once();assert not data['events'][0]['deferred'];undo()
    for fault in ('forward','execute','bookkeeping','sample_before','sample_after','unexpected_return','skip_bookkeeping'):
        r,_,undo=installed()
        try:r.execute_model(fault=fault);r.sample_tokens()
        except RuntimeError:pass
        else:raise AssertionError(f'Expected failure: {fault}')
        once();undo();once()
    for cleanup in ('next_iteration','overlapping_context','uninstall'):
        r,_,undo=installed();r.execute_model()
        try:
            if cleanup=='next_iteration':r.execute_model()
            elif cleanup=='overlapping_context':
                with r.maybe_get_kv_connector_output(NS(total_num_scheduled_tokens=1)):pass
            else:undo()
        except RuntimeError:assert cleanup!='uninstall'
        else:assert cleanup=='uninstall'
        once();undo();once()
    print('CPU_PASS: source/runtime guards, pinned context identity/order, native/after_sample, no-forward, defer-finalize, 7 exception/return paths, pending cleanup; GPU_UNRUN')


if __name__ == '__main__':
    main()
