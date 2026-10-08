"""UNEXECUTED BASELINE SCAFFOLD: live 24->23 expert / 512->608 KV, vLLM 0.26.

Research stopped before execution after the directly overlapping VAMP result
was identified. Syntax only was checked; runtime correctness, allocator release,
the storage bound, and continuation with live requests have NOT been validated.
Do not treat this retained scaffold as a tested runtime or a novel mechanism.

Caller owns the synchronous after-engine.step boundary and its episode clock.
Only uniform cross-layer HND BF16 OLMoE, AR, TP1, CPU offload is supported.
The 5.5 GiB bound is UNIQUE EXPERT SCRATCH + GPU KV STORAGE, not total CUDA
reserved memory. Old allocations must actually leave the active allocator
before any replacement is allocated. Failure leaves engine.step disabled and
retains the CPU snapshot on both the exception and runtime for inspection.
No rollback, scheduling loop, allocator replacement, or controller is provided.
"""
from contextlib import contextmanager
import gc
import hashlib
from pathlib import Path
import time


POOL_LIMIT = 5905580032
OLD_BLOCKS, NEW_BLOCKS, OLD_CAP, NEW_CAP = 512, 608, 24, 23
OLD_KV_BYTES, NEW_KV_BYTES = 1073741824, 1275068416


class LiveKVGrowError(RuntimeError):
    def __init__(self, receipt, snapshot=None):
        self.receipt, self.snapshot = receipt, snapshot
        super().__init__("live KV grow stopped service: " + receipt.get("error", "in progress"))


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _unique_objects(objects):
    return list({id(obj): obj for obj in objects if obj is not None}.values())


def _storage_records(tensors):
    records = {}
    for tensor in tensors:
        if tensor is None or tensor.numel() == 0:
            continue
        storage = tensor.untyped_storage()
        key = (str(tensor.device), storage.data_ptr())
        records[key] = dict(device=key[0], pointer=key[1], bytes=storage.nbytes())
    return sorted(records.values(), key=lambda item: (item["device"], item["pointer"]))


def _pool_records(runner, runtime, gpu_views):
    tensors = [getattr(entry["state"], name) for entry in runtime.layers.values()
               for name in ("scratch_w13", "scratch_w2")]
    tensors.extend(runner.kv_caches)
    tensors.extend(gpu_views)
    tensors.append(runner.cross_layers_kv_cache)
    tensors.extend(getattr(layer, "kv_cache", None)
                   for layer in runner.compilation_config.static_forward_context.values()
                   if hasattr(layer, "kv_cache"))
    return _storage_records(tensors)


def _active_allocation_owners(snapshot, records):
    """Integer-only allocator inspection; never retain a storage or GPU view."""
    result = []
    for segment in snapshot:
        address = segment["address"]
        for block in segment["blocks"]:
            start = block.get("address", address)
            end = start + block["size"]
            if block["state"] != "inactive":
                for record in records:
                    if start <= record["pointer"] < end:
                        result.append(dict(pointer=record["pointer"], allocation=start,
                                           bytes=block["size"], state=block["state"]))
            address = end
    return result


def _resize_configs(configs, old, new):
    """Update aliases exactly once, including shared KVCacheTensor descriptors."""
    configs = _unique_objects(configs)
    tensors = _unique_objects(t for config in configs for t in config.kv_cache_tensors)
    for config in configs:
        _require(config.num_blocks == old, "inconsistent initial KV config block count")
    for tensor in tensors:
        _require(tensor.size % old == 0, "KV descriptor not integral pages")
    for tensor in tensors:
        tensor.size = tensor.size // old * new
    for config in configs:
        config.num_blocks = new


def _publish_blocks(pool, new_count, block_class):
    old = pool.num_gpu_blocks
    _require(len(pool.blocks) == old and new_count > old, "only append-only block growth")
    new = [block_class(block_id=i) for i in range(old, new_count)]
    pool.blocks.extend(new)
    pool.free_block_queue.prepend_n(new)
    pool.num_gpu_blocks = new_count


def _request_state(scheduler, runner):
    managers = scheduler.kv_cache_manager.coordinator.single_type_managers
    return dict(input_batch=id(runner.input_batch),
        requests={key: (id(req), str(req.status), req.num_computed_tokens,
                        req.num_tokens, tuple(req.all_token_ids))
                  for key, req in scheduler.requests.items()},
        req_blocks=[{key: tuple((id(block), block.block_id) for block in blocks)
                     for key, blocks in manager.req_to_blocks.items()} for manager in managers],
        finished_recving=sorted(scheduler.finished_recving_kv_req_ids))


def _block_state(pool):
    return [(id(block), block.block_id, block.ref_cnt, block.block_hash, block.is_null)
            for block in pool.blocks]


def _barrier(scheduler, connector_worker, metadata_class, output_class):
    cw, cs = connector_worker, scheduler.connector.connector_scheduler
    _require(not cs._current_batch_load_jobs and not cs._current_batch_jobs_to_flush,
             "step has scheduler metadata that has not reached the worker")
    _require(cs.config.num_workers == 1, "barrier requires a single worker")
    pending = sorted(cs._jobs)
    deferred = len(cw._unsubmitted_store_jobs)
    cw.start_kv_transfers(metadata_class(load_jobs={}, store_jobs={}, jobs_to_flush=set()))
    cw.worker.wait(set(pending))
    sent, recv = cw.get_finished(set())  # NOT outer connector.get_finished/prepare_store_kv.
    meta = cw.build_connector_worker_meta()
    scheduler._update_from_kv_xfer_finished(output_class(
        finished_sending=sent, finished_recving=recv, kv_connector_worker_meta=meta))
    _require(not cs._jobs and not cw._load_jobs and not cw._unsubmitted_store_jobs,
             "offload jobs did not settle at the step boundary")
    _require(not getattr(cs, "_chunks_being_loaded", None), "load-key bookkeeping remains")
    _require(not cs.manager.has_pending_work(), "unsupported background manager work")
    for handler in (cw.worker._store_handler, cw.worker._load_handler):
        _require(not handler._transfers and not handler._transfer_events,
                 "native transfer descriptors still hold GPU pointers")
    return dict(pending_jobs=pending, deferred_stores=deferred,
                finished_sending=sorted(sent), finished_recving=sorted(recv),
                completed_jobs=dict(meta.completed_jobs) if meta is not None else {})


def _cpu_cache_identity(cw, cs):
    store, load = cw.worker._store_handler, cw.worker._load_handler
    _require(store.dst_tensors is load.src_tensors, "CPU transfer lists do not alias")
    return dict(worker=id(cw.worker), cached_worker=id(cw.spec._worker),
                manager=id(cs.manager), policy=id(cs.manager._policy),
                cpu_list=id(store.dst_tensors), cpu_storage=_storage_records(store.dst_tensors),
                mmap=id(store._mmap_region))


def _snapshot_cross_layer(runner, torch):
    # This view aliases old GPU bytes, and must die on return; return CPU only.
    raw = runner.cross_layers_kv_cache.view(torch.uint8).reshape(-1)
    snapshot = torch.empty(raw.numel(), dtype=torch.uint8, device="cpu", pin_memory=True)
    snapshot.copy_(raw, non_blocking=False)
    return snapshot


def _release_pool(runner, runtime, gpu_views, layer_names):
    gpu_views.clear()  # Store/load handlers share this list; CPU lists stay untouched.
    runner.kv_caches.clear()
    runner.cross_layers_kv_cache = None
    for name in layer_names:
        runner.compilation_config.static_forward_context[name].kv_cache = None
    if hasattr(runner, "_kv_block_zeroer"):
        del runner._kv_block_zeroer
    for entry in runtime.layers.values():
        state = entry["state"]
        state.scratch_w13 = state.scratch_w2 = None
        state.expert_to_slot.clear()
        state.slot_to_expert[:] = [-1] * NEW_CAP
        state.lru_tick[:] = [0] * NEW_CAP
        state.expert_map_device.fill_(-1)
        state.lru_clock, state.last_topk_ids_cpu = 0, None
        state.last_unique_experts.clear()
        state.cap_experts = NEW_CAP


def _preflight(engine, runner, runtime, worker_connector, target_cap, target_kv_bytes):
    from vllm.forward_context import is_forward_context_available
    from vllm.v1.attention.backends.utils import get_kv_cache_layout

    torch, scheduler = runtime.torch, engine.engine_core.engine_core.scheduler
    cw, cs = worker_connector.connector_worker, scheduler.connector.connector_scheduler
    _require((target_cap, target_kv_bytes) == (NEW_CAP, NEW_KV_BYTES), "only 24->23 / 512->608")
    _require(type(runner.model).__name__ == "OlmoeForCausalLM", "only OLMoE")
    _require(runner.model_config.enforce_eager and not runner.use_async_scheduling,
             "only synchronous eager execution")
    _require(runner.speculative_config is None and not is_forward_context_available(),
             "grow only after an AR step outside ForwardContext")
    _require(not runner.cache_config.enable_prefix_caching and runtime.layer_caps is None,
             "only uniform expert caps with GPU prefix caching disabled")
    _require(not getattr(runner.model_config, "is_hybrid", False), "no hybrid state")
    _require(all(getattr(runner.parallel_config, name, 1) == 1 for name in
                 ("tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size",
                  "decode_context_parallel_size", "prefill_context_parallel_size")), "only TP/PP/DP/CP1")
    _require(not runner.parallel_config.enable_dbo and get_kv_cache_layout() == "HND",
             "only non-DBO HND")
    _require(runtime.cap == OLD_CAP and len(runtime.layers) == 16, "initial expert cap/layer count")
    _require(not any(row["status"] == "started" for row in runtime.records), "MoE call in progress")
    _require(type(worker_connector).__name__ == type(scheduler.connector).__name__ == "OffloadingConnector",
             "only OffloadingConnector")
    _require(type(cw.worker).__name__ == "CPUOffloadingWorker" and
             type(cs.manager).__name__ == "CPUOffloadingManager", "only native CPU offload")
    _require(cw.spec._worker is cw.worker, "cached CPU worker mismatch")
    store, load = cw.worker._store_handler, cw.worker._load_handler
    _require(store.src_tensors is load.dst_tensors and len(store.src_tensors) == 1,
             "only one shared canonical GPU tensor")
    _require(store.kv_cache_groups_data_refs == load.kv_cache_groups_data_refs,
             "offload group mapping mismatch")
    cross = runner.cross_layers_kv_cache
    _require(cross is not None and cross.is_cuda and cross.is_contiguous()
             and cross.storage_offset() == 0 and cross.dtype == torch.bfloat16
             and cross.shape[0] == OLD_BLOCKS, "only contiguous blocks-first cross-layer BF16 KV")
    _require(cross.numel() * cross.element_size() == OLD_KV_BYTES == cross.untyped_storage().nbytes(),
             "old cross-layer KV size/storage mismatch")
    _require(_storage_records(store.src_tensors) == _storage_records([cross]), "connector GPU view mismatch")
    _require(list(store.src_tensors[0].shape) == [OLD_BLOCKS, OLD_KV_BYTES // OLD_BLOCKS],
             "unexpected canonical KV page shape")
    _require(runner.use_uniform_kv_cache(runner.attn_groups), "uniform native allocator unavailable")
    _require(not runner.shared_kv_cache_layers and not runner.runner_only_attn_layers, "no shared/extra layers")
    pool = scheduler.kv_cache_manager.block_pool
    _require(pool.num_gpu_blocks == OLD_BLOCKS and len(pool.blocks) == OLD_BLOCKS, "old pool count")
    _require(scheduler.kv_cache_manager.watermark_blocks == 0, "nonzero watermark unsupported")
    _require(len(runner.kv_cache_config.kv_cache_groups) == 1, "one full-attention KV group required")
    names = list(runner.kv_cache_config.kv_cache_groups[0].layer_names)
    _require(len(names) == 16, "expected 16 KV layers")
    for entry in runtime.layers.values():
        state = entry["state"]
        _require(state.cap_experts == OLD_CAP and entry["layer"]._wisp_state is state,
                 "expert state identity mismatch")
        _require(state.cpu_w13.device.type == state.cpu_w2.device.type == "cpu", "missing CPU master")
        _require(state.cpu_w13.dtype == state.cpu_w2.dtype == torch.bfloat16, "only BF16 experts")
        _require(entry["layer"].w13_weight.numel() == entry["layer"].w2_weight.numel() == 0,
                 "expert parameters still own GPU weight tensors")
    return scheduler, cw, cs, store.src_tensors, names, list(cross.shape)


def grow(engine, runner, runtime, target_cap=NEW_CAP, target_kv_bytes=NEW_KV_BYTES):
    """Perform the sole supported transaction; caller must stop on any failure.

    Receipt is JSON-safe. Failed snapshot is runtime._live_kv_grow_snapshot and
    LiveKVGrowError.snapshot. Successful snapshots are released before return.
    """
    _require(not hasattr(runtime, "_live_kv_grow_receipt"), "live grow may be attempted only once")
    receipt = dict(status="started", stages=[], memory=[], storage_limit_bytes=POOL_LIMIT,
        old_cap=OLD_CAP, new_cap=target_cap, old_blocks=OLD_BLOCKS, new_blocks=NEW_BLOCKS,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        memory_scope="unique expert scratch + GPU KV storages; excludes other CUDA allocations",
        snapshot_bytes=0, snapshot_d2h_bytes=0, restore_h2d_bytes=0,
        expert_reheat="cold; future native pager misses/copies remain in episode timing")
    runtime._live_kv_grow_receipt = receipt
    runtime._live_kv_grow_snapshot = None
    started, snapshot, destructive = time.perf_counter(), None, False
    torch = runtime.torch
    had_step, original_step = "step" in vars(engine), engine.step

    def blocked_step(*args, **kwargs):
        raise LiveKVGrowError(receipt, runtime._live_kv_grow_snapshot)

    engine.step = blocked_step

    @contextmanager
    def stage(name):
        row = dict(name=name, start_s=time.perf_counter() - started, status="started")
        receipt["stages"].append(row)
        try:
            yield
            row["status"] = "complete"
        finally:
            row["elapsed_s"] = time.perf_counter() - started - row["start_s"]

    def record_memory(label):
        records = _pool_records(runner, runtime, gpu_views)
        total = sum(row["bytes"] for row in records)
        receipt["memory"].append(dict(label=label, unique_pool_bytes=total,
            storage_count=len(records), cuda_allocated=torch.cuda.memory_allocated(),
            cuda_reserved=torch.cuda.memory_reserved()))
        receipt["peak_unique_pool_bytes"] = max(total, receipt.get("peak_unique_pool_bytes", 0))
        _require(total <= POOL_LIMIT, "unique pool storage exceeded 5.5 GiB")
        return records

    try:
        import vllm
        from vllm.distributed.kv_transfer import get_kv_transfer_group
        from vllm.config import set_current_vllm_config
        from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import OffloadingConnectorMetadata
        from vllm.v1.outputs import KVConnectorOutput
        from vllm.v1.core.kv_cache_utils import KVCacheBlock

        _require(vllm.__version__ == "0.26.0", "only pinned vLLM 0.26.0")
        connector = get_kv_transfer_group()
        with set_current_vllm_config(runner.vllm_config), torch.inference_mode():
            with stage("preflight"):
                scheduler, cw, cs, gpu_views, old_shape = None, None, None, None, None
                scheduler, cw, cs, gpu_views, names, old_shape = _preflight(
                    engine, runner, runtime, connector, target_cap, target_kv_bytes)
                pool = scheduler.kv_cache_manager.block_pool
                old_records = record_memory("initial")
                _require(sum(row["bytes"] for row in old_records) == POOL_LIMIT,
                         "initial actual pool storage is not exactly 5.5 GiB")
                owners = _active_allocation_owners(torch.cuda.memory_snapshot(), old_records)
                _require({row["pointer"] for row in owners} == {row["pointer"] for row in old_records},
                         "allocator snapshot cannot account for all old pool storages")
                receipt["old_storage_allocations"] = owners
            with stage("native_transfer_barrier"):
                torch.cuda.synchronize()
                receipt["barrier"] = _barrier(scheduler, cw, OffloadingConnectorMetadata, KVConnectorOutput)
                torch.cuda.synchronize()
                requests_before, blocks_before = _request_state(scheduler, runner), _block_state(pool)
                free_before = [block.block_id for block in pool.free_block_queue.get_all_free_blocks()]
                cpu_before = _cpu_cache_identity(cw, cs)
                pool_identity = (id(pool.blocks), id(pool.free_block_queue), id(pool.null_block),
                                 id(pool.cached_block_hash_to_block), id(pool.cached_block_hashes_by_block))
                had_zeroer = hasattr(runner, "_kv_block_zeroer")
                receipt["barrier_requests"] = len(requests_before["requests"])
            with stage("cpu_full_kv_snapshot"):
                snapshot = _snapshot_cross_layer(runner, torch)
                runtime._live_kv_grow_snapshot = snapshot
                receipt.update(snapshot_bytes=snapshot.numel(), snapshot_d2h_bytes=snapshot.numel(),
                               snapshot_pinned=snapshot.is_pinned())
            with stage("release_old_gpu_pool"):
                destructive = True
                _release_pool(runner, runtime, gpu_views, names)
                torch.cuda.synchronize()
                gc.collect()
                torch.cuda.empty_cache()
                leaked = _active_allocation_owners(torch.cuda.memory_snapshot(), old_records)
                receipt["old_allocations_still_active_after_release"] = leaked
                _require(not leaked, "old GPU allocation is retained by an external tensor/view")
                _require(not record_memory("released"), "known GPU pool references survived release")
            with stage("rebuild_cold_experts"):
                for entry in runtime.layers.values():
                    state = entry["state"]
                    state.scratch_w13 = torch.empty((NEW_CAP, *state.cpu_w13.shape[1:]),
                                                    dtype=state.cpu_w13.dtype, device=runner.device)
                    record_memory(entry["layer_name"] + ":w13")
                    state.scratch_w2 = torch.empty((NEW_CAP, *state.cpu_w2.shape[1:]),
                                                   dtype=state.cpu_w2.dtype, device=runner.device)
                    record_memory(entry["layer_name"] + ":w2")
                runtime.cap = NEW_CAP
                runtime.upstream._WISP_RUNTIME_CONFIG["cap_experts_override"] = NEW_CAP
                _require(sum(row["bytes"] for row in record_memory("experts_ready"))
                         == POOL_LIMIT - NEW_KV_BYTES, "new expert storage byte mismatch")
            with stage("native_kv_reallocation_and_restore"):
                configs = [runner.kv_cache_config, scheduler.kv_cache_config,
                           scheduler.kv_cache_manager.kv_cache_config,
                           scheduler.kv_cache_manager.coordinator.kv_cache_config, cw.kv_cache_config]
                _resize_configs(configs, OLD_BLOCKS, NEW_BLOCKS)
                runner.initialize_kv_cache_tensors(runner.kv_cache_config, runner._kernel_block_sizes)
                _require(list(runner.cross_layers_kv_cache.shape) == [NEW_BLOCKS, *old_shape[1:]],
                         "native cross-layer allocation changed page layout")
                record_memory("new_kv_allocated")
                runner.cross_layers_kv_cache.view(torch.uint8).reshape(-1)[:OLD_KV_BYTES].copy_(
                    snapshot, non_blocking=False)
                torch.cuda.synchronize()
                receipt["restore_h2d_bytes"] = OLD_KV_BYTES
            with stage("rebind_existing_cpu_offload_worker"):
                # Reuse native canonicalization, replacing only its terminal init hook.
                had_init, old_init = "_init_worker" in vars(cw), cw._init_worker

                def rebind(canonical):
                    _require(canonical.group_data_refs == cw.worker._store_handler.kv_cache_groups_data_refs,
                             "canonical group/page references changed")
                    _require(len(canonical.tensors) == 1 and not gpu_views, "unexpected canonical tensors")
                    value = canonical.tensors[0]
                    _require(value.page_size_bytes == OLD_KV_BYTES // OLD_BLOCKS, "canonical page changed")
                    gpu_views.append(value.tensor.view(torch.int8).view(NEW_BLOCKS, value.page_size_bytes))

                cw._init_worker = rebind
                try:
                    connector.register_cross_layers_kv_cache(
                        runner.cross_layers_kv_cache, runner.cross_layers_attn_backend)
                finally:
                    if had_init:
                        cw._init_worker = old_init
                    else:
                        del cw._init_worker
                if had_zeroer:
                    runner._init_kv_zero_meta()
                _require(_cpu_cache_identity(cw, cs) == cpu_before, "CPU worker/cache identity changed")
                _require(_request_state(scheduler, runner) == requests_before, "live request/input state changed")
                _require(_block_state(pool) == blocks_before, "old block metadata changed before publish")
                final_records = record_memory("rebound")
                _require(sum(row["bytes"] for row in final_records) == POOL_LIMIT,
                         "final pool storage is not exactly 5.5 GiB")
            with stage("publish_new_blocks"):
                cache_configs = _unique_objects([engine.vllm_config.cache_config, runner.cache_config,
                                                 scheduler.cache_config])
                for config in cache_configs:
                    _require(config.num_gpu_blocks == OLD_BLOCKS, "cache config count changed unexpectedly")
                    config.num_gpu_blocks = NEW_BLOCKS
                    config.kv_cache_memory_bytes = NEW_KV_BYTES
                    if config.num_gpu_blocks_override is not None:
                        config.num_gpu_blocks_override = NEW_BLOCKS
                _publish_blocks(pool, NEW_BLOCKS, KVCacheBlock)
                _require(_block_state(pool)[:OLD_BLOCKS] == blocks_before, "old blocks not preserved")
                _require([block.block_id for block in pool.free_block_queue.get_all_free_blocks()]
                         == list(range(OLD_BLOCKS, NEW_BLOCKS)) + free_before, "free queue order changed")
                _require(pool_identity == (id(pool.blocks), id(pool.free_block_queue), id(pool.null_block),
                    id(pool.cached_block_hash_to_block), id(pool.cached_block_hashes_by_block)), "pool replaced")
                _require(_request_state(scheduler, runner) == requests_before, "request state changed on publish")
                receipt.update(old_free_blocks=len(free_before), new_free_blocks=pool.get_num_free_blocks(),
                               old_block_objects_preserved=True, cpu_cache_preserved=True,
                               input_batch_preserved=True, cold_expert_states=16)
            with stage("release_cpu_snapshot"):
                runtime._live_kv_grow_snapshot = None
                snapshot = None
            receipt["status"] = "complete"
        if had_step:
            engine.step = original_step
        else:
            del engine.step
        return receipt
    except BaseException as exc:
        receipt.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                       destructive_started=destructive, service_disabled=True,
                       snapshot_retained=runtime._live_kv_grow_snapshot is not None)
        raise LiveKVGrowError(receipt, runtime._live_kv_grow_snapshot) from exc
    finally:
        receipt["elapsed_s"] = time.perf_counter() - started
