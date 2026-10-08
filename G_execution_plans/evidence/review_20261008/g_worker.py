"""Startup-only instrumentation for archived vLLM 0.26 Worker APIs.

Imported only after startup_probe owns the shared GPU lock. Never changes the
router, precision, scheduling, graph dispatch, allocator policy or KV budget.
GPU path is untested; measurements remain native estimates and observations.
"""
from __future__ import annotations

import functools
import json
import os
from pathlib import Path
import time

import torch
from vllm.v1.worker.gpu_worker import Worker


def storage_bytes(value) -> dict:
    """Count backing storages once, including aliases across layer KV views."""
    storages = {}
    tensor_count = 0

    def visit(item):
        nonlocal tensor_count
        if isinstance(item, torch.Tensor):
            tensor_count += 1
            storage = item.untyped_storage()
            key = (str(item.device), storage.data_ptr(), storage.nbytes())
            storages[key] = storage.nbytes()
        elif isinstance(item, dict):
            for child in item.values():
                visit(child)
        elif isinstance(item, (tuple, list)):
            for child in item:
                visit(child)

    visit(value)
    return dict(tensor_views=tensor_count, unique_storages=len(storages),
                unique_storage_bytes=sum(storages.values()))


class GWorker(Worker):
    def _emit(self, event: str, **fields):
        path = Path(os.environ['G_STARTUP_STAGE_FILE'])
        row = dict(event=event, pid=os.getpid(), rank=getattr(self, 'rank', None),
                   monotonic_s=time.perf_counter(), **fields)
        with path.open('a') as handle:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')

    def _snapshot(self) -> dict:
        # Startup boundaries only; do not reset peaks or empty caches, which
        # would interfere with native profiling and subsequent KV budgeting.
        torch.cuda.synchronize(self.device)
        free, total = torch.cuda.mem_get_info(self.device)
        data = dict(device=str(self.device), free_bytes=free, total_bytes=total,
                    allocated_bytes=torch.cuda.memory_allocated(self.device),
                    reserved_bytes=torch.cuda.memory_reserved(self.device),
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(self.device),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(self.device),
                    peak_scope='native allocator history; native code may reset counters')
        pools = {}
        try:
            for segment in torch.cuda.memory_snapshot():
                if segment.get('device') != self.device.index:
                    continue
                pool_id = segment.get('segment_pool_id', segment.get('pool_id', 'unknown'))
                key = json.dumps(pool_id)
                pool = pools.setdefault(key, dict(segments=0, total_bytes=0,
                                                  allocated_bytes=0, active_bytes=0,
                                                  inactive_block_bytes=0))
                pool['segments'] += 1
                pool['total_bytes'] += segment.get('total_size', 0)
                pool['allocated_bytes'] += segment.get('allocated_size', 0)
                pool['active_bytes'] += segment.get('active_size', 0)
                pool['inactive_block_bytes'] += sum(b['size'] for b in segment.get('blocks', [])
                                                     if b.get('state') == 'inactive')
            data['torch_allocator_pools'] = pools
            data['pool_attribution'] = 'Segment pool IDs only; non-Torch graph data is not attributed. Unknown means API omitted pool identity.'
        except Exception as exc:
            data['pool_snapshot_error'] = repr(exc)
        try:
            from vllm.v1.worker.workspace import current_workspace_manager, is_workspace_manager_initialized
            if is_workspace_manager_initialized():
                workspace = current_workspace_manager()
                data['shared_workspace'] = dict(locked=workspace.is_locked(),
                    **storage_bytes(workspace._current_workspaces))
        except Exception as exc:
            data['workspace_observation_error'] = repr(exc)
        return data

    def _observe_runner_call(self, method_name: str):
        original = getattr(self.model_runner, method_name)

        @functools.wraps(original)
        def observed(*args, **kwargs):
            self._emit(method_name + '_before', memory=self._snapshot())
            if method_name == 'capture_model':
                self._emit('live_capture_order', groups=[
                    dict(mode=mode.name, token_buckets=[int(desc.num_tokens) for desc in descs])
                    for mode, descs in self.model_runner.cudagraph_dispatcher.get_capture_descs()
                ])
            start = time.perf_counter()
            returned = original(*args, **kwargs)
            duration = time.perf_counter() - start
            self._emit(method_name + '_after', elapsed_s=duration,
                       native_return_bytes=int(returned), memory=self._snapshot(),
                       interpretation=('Representative-capture extrapolation, not exact set cost'
                         if method_name == 'profile_cudagraph_memory' else
                         'Native whole live capture device-free-memory delta after KV allocation'))
            return returned

        setattr(self.model_runner, method_name, observed)

    def init_device(self):
        start = time.perf_counter()
        value = super().init_device()
        self._emit('init_device_after', elapsed_s=time.perf_counter() - start,
                   memory=self._snapshot())
        return value

    def load_model(self, *, load_dummy_weights=False):
        start = time.perf_counter()
        value = super().load_model(load_dummy_weights=load_dummy_weights)
        self._emit('load_model_after', elapsed_s=time.perf_counter() - start,
                   model_memory_usage_bytes=int(self.model_runner.model_memory_usage),
                   memory=self._snapshot())
        actual_backends = set()
        for module in self.model_runner.model.modules():
            method = getattr(module, 'quant_method', None)
            if hasattr(method, 'unquantized_backend'):
                actual_backends.add((str(method.unquantized_backend),
                                     getattr(method.experts_cls, '__name__', str(method.experts_cls))))
        self._emit('actual_moe_backends', backend_expert_classes=sorted(actual_backends))
        self._observe_runner_call('profile_cudagraph_memory')
        self._observe_runner_call('capture_model')
        return value

    def determine_available_memory(self):
        self._emit('determine_available_memory_before', memory=self._snapshot())
        start = time.perf_counter()
        value = super().determine_available_memory()
        self._emit('determine_available_memory_after', elapsed_s=time.perf_counter() - start,
                   native_available_kv_bytes=int(value),
                   native_cudagraph_estimate_bytes=int(getattr(self, 'cudagraph_memory_estimate', 0)),
                   native_peak_activation_bytes=int(getattr(self, 'peak_activation_memory', 0)),
                   native_non_torch_increase_bytes=int(getattr(self, 'non_torch_memory', 0)),
                   memory=self._snapshot())
        return value

    def initialize_from_config(self, kv_cache_config):
        self._emit('kv_allocate_before', num_blocks=int(kv_cache_config.num_blocks),
                   configured_kv_tensor_bytes=sum(t.size for t in kv_cache_config.kv_cache_tensors),
                   memory=self._snapshot())
        start = time.perf_counter()
        value = super().initialize_from_config(kv_cache_config)
        self._emit('kv_allocate_after', elapsed_s=time.perf_counter() - start,
                   num_blocks=int(kv_cache_config.num_blocks),
                   allocated_kv=storage_bytes([self.model_runner.kv_caches,
                                  getattr(self.model_runner, 'cross_layers_kv_cache', None)]),
                   memory=self._snapshot())
        return value

    def compile_or_warm_up_model(self):
        self._emit('compile_or_warm_up_before', memory=self._snapshot())
        start = time.perf_counter()
        value = super().compile_or_warm_up_model()
        self._emit('startup_ready', elapsed_s=time.perf_counter() - start,
                   num_blocks=int(self.cache_config.num_gpu_blocks), memory=self._snapshot())
        # Native capture's return was sampled before its own empty_cache;
        # report an additional, explicitly labelled final cleanup separately.
        # This cannot increase the KV blocks already allocated above.
        self._emit('startup_explicit_empty_cache_before', memory=self._snapshot())
        cleanup_start = time.perf_counter()
        torch.cuda.empty_cache()
        self._emit('startup_explicit_empty_cache_after',
                   elapsed_s=time.perf_counter() - cleanup_start,
                   num_blocks=int(self.cache_config.num_gpu_blocks), memory=self._snapshot())
        return value
