"""Reuse one drained vLLM 0.11.2/WiSP engine across common-warmup episodes.

Reset metadata only. Pending finished-request notifications remain native;
the next warmup consumes them before adding its requests. No request data is
returned here: records contain counts, allocation identities and hashes.
"""
import hashlib
import json
import os
import sys


def _require(ok, message):
    if not ok:
        raise RuntimeError(message)


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _worker_reset(worker, finished_ids, expected_hash):
    import torch
    torch.cuda.synchronize()
    runner = worker.model_runner
    pending = set(finished_ids)
    cached, batched = set(runner.requests), set(runner.input_batch.req_id_to_index)
    _require(cached <= pending and batched <= cached, "worker has non-finished requests")
    _require(runner.execute_model_state is None and runner.kv_connector_output is None,
             "worker has pending execution or KV transfer")
    module = sys.modules["wisp.integrations.vllm.fused_moe"]
    states = module._LAYER_STATES
    _require(len(states) == 16 and runner.kv_caches, "expected OLMoE16 pager and KV tensors")

    def tensor_id(tensor):
        storage = tensor.untyped_storage()
        return dict(object_id=id(tensor), data_ptr=tensor.data_ptr(),
                    storage_ptr=storage.data_ptr(), storage_bytes=storage.nbytes(),
                    shape=list(tensor.shape), stride=list(tensor.stride()), dtype=str(tensor.dtype))

    def identities():
        return dict(pid=os.getpid(), runner=id(runner), registry=id(states),
            layers=[dict(state=id(s), maps=[id(s.slot_to_expert), id(s.expert_to_slot), id(s.lru_tick)],
                stream=id(s.copy_stream), event=id(s.copy_event),
                tensors=[tensor_id(t) for t in (s.cpu_w13, s.cpu_w2, s.scratch_w13,
                                                s.scratch_w2, s.expert_map_device)]) for s in states],
            kv=[tensor_id(t) for t in runner.kv_caches])

    before = identities()
    _require(expected_hash is None or _hash(before) == expected_hash, "allocation identity changed between episodes")
    _require(all(s.mode == "paged" and s.cap_experts == 16 and s.cpu_w13.is_pinned()
                 and s.cpu_w2.is_pinned() for s in states), "unsupported pager configuration")
    for state in states:
        state.slot_to_expert[:] = [-1] * state.cap_experts
        state.expert_to_slot.clear()
        state.expert_map_device.fill_(-1)
        state.lru_tick[:] = [0] * state.cap_experts
        state.lru_clock = 0
        state.last_topk_ids_cpu = None
        state.last_unique_experts.clear()
        for key in state.__slots__:
            if key.startswith("stats_"):
                setattr(state, key, 0)
    module._CURRENT_STEP_OBS.clear()
    torch.cuda.synchronize()
    after = identities()
    _require(before == after, "metadata reset changed allocation identities")
    _require(all(bool((s.expert_map_device == -1).all().item()) for s in states), "device map reset failed")
    return dict(pid=os.getpid(), allocation_identity=after, allocation_sha256=_hash(after),
                allocation_unchanged=True, finished_notification_count=len(pending),
                cached_finished_count=len(cached), batched_finished_count=len(batched),
                worker_pending_only_finished=True, pager_empty=True)


class ReusingLLM:
    def __init__(self, original_factory):
        self.original_factory = original_factory
        self.llm = self.config = self.native_schedule = self.worker_hash = None
        self.records = []

    def __call__(self, **llm_config):
        row = dict(invocation=len(self.records), pid=os.getpid(), status="RESETTING")
        self.records.append(row)
        try:
            _require(self.config is None or self.config == llm_config, "LLM configuration changed")
            _require(all(os.environ.get(key, "0") == "0" for key in
                         ("WISP_PREFETCH", "WISP_DYNAMIC", "VLLM_ENABLE_V1_MULTIPROCESSING")),
                     "requires prefetch/dynamic/multiprocessing disabled")
            if self.llm is None:
                self.config = dict(llm_config)
                self.llm = self.original_factory(**llm_config)
                self.native_schedule = self.llm.llm_engine.engine_core.engine_core.scheduler.schedule
            engine = self.llm.llm_engine
            scheduler = engine.engine_core.engine_core.scheduler
            cfg = engine.vllm_config
            _require(not cfg.scheduler_config.async_scheduling and cfg.speculative_config is None
                     and not cfg.cache_config.enable_prefix_caching and scheduler.connector is None
                     and not scheduler.num_lookahead_tokens and not scheduler.num_spec_tokens,
                     "unsupported async/spec/prefix/connector state")
            _require(not engine.has_unfinished_requests() and not scheduler.requests
                     and not scheduler.running and not scheduler.waiting, "engine is not fully drained")
            pool = scheduler.kv_cache_manager.block_pool
            queue = pool.free_block_queue
            blocks = queue.get_all_free_blocks()
            expected = [b for b in pool.blocks if not b.is_null]
            _require(pool.null_block is pool.blocks[0] and pool.null_block.is_null
                     and len(expected) == pool.num_gpu_blocks - 1
                     and queue.num_free_blocks == len(expected)
                     and {id(b) for b in blocks} == {id(b) for b in expected}
                     and all(b.ref_cnt == 0 and b.block_hash is None for b in expected),
                     "KV pool is not entirely free with one reserved null block")
            finished = sorted(scheduler.finished_req_ids)  # Preserve, never consume or clear.
            workers = engine.engine_core.collective_rpc(_worker_reset, args=(finished, self.worker_hash))
            _require(len(workers) == 1, "requires one worker")
            self.worker_hash = workers[0]["allocation_sha256"]
            before_order = [b.block_id for b in blocks]
            queue.append_n(sorted(queue.popleft_n(len(expected)), key=lambda b: b.block_id))
            after_order = [b.block_id for b in queue.get_all_free_blocks()]
            _require(after_order == sorted(b.block_id for b in expected), "KV order reset failed")
            scheduler.schedule = self.native_schedule
            scheduler.scheduler_config.long_prefill_token_threshold = 32
            row.update(status="READY", worker=workers[0], kv_free_blocks=len(expected),
                       kv_order_before_sha256=_hash(before_order), kv_order_after_sha256=_hash(after_order),
                       llm_id=id(self.llm), scheduler_id=id(scheduler), pool_id=id(pool), queue_id=id(queue),
                       native_schedule_restored=True, threshold=32)
            return self.llm
        except Exception as exc:
            row.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
            raise
