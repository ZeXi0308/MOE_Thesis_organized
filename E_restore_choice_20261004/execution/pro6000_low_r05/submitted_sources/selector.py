"""Thin vLLM 0.26 native lookup adapter; never executes both branches.

Both arms keep the same native STORE, FCFS, victim, quantum and admission.
Only ready Host hits on preempted requests with full GPU capacity are eligible.
Unsupported/unready/capacity-limited cases retain the native lookup result.
"""
import hashlib
import json
import math
import time


def token_hash(tokens):
    return hashlib.sha256(json.dumps(list(tokens)).encode()).hexdigest()


class RestoreSelector:
    def __init__(self, scheduler, policy, threshold=0):
        self.scheduler = scheduler
        self.cs = scheduler.connector.connector_scheduler
        self.policy, self.threshold = policy, threshold
        self.events, self.commits, self.transfers, self.preemptions = [], [], [], []
        self.last_step_s = 0.0
        self.origin = time.perf_counter()
        self.pending = {}
        self.enabled = True
        self._lookup = self.cs.get_num_new_matched_tokens
        self._alloc = self.cs.update_state_after_alloc
        self._update = self.cs.update_connector_output
        self._preempt = scheduler._preempt_request
        self.cs.get_num_new_matched_tokens = self.lookup
        self.cs.update_state_after_alloc = self.allocated
        self.cs.update_connector_output = self.update
        scheduler._preempt_request = self.preempt
        assert scheduler.scheduler_reserve_full_isl
        assert len(self.cs.config.kv_group_configs) == 1
        assert self.cs.config.kv_group_configs[0].sliding_window_size_in_chunks is None
        assert type(self.cs.manager).__name__ == 'CPUOffloadingManager'
        assert scheduler.kv_cache_manager.enable_caching is False

    def now(self):
        return time.perf_counter() - self.origin

    def clear(self):
        self.events.clear(); self.commits.clear(); self.transfers.clear(); self.preemptions.clear()
        self.pending.clear()
        self.origin = time.perf_counter()
        self.last_step_s = 0.0

    def lookup(self, request, local_tokens):
        start = self.now()
        native = self._lookup(request, local_tokens)
        lookup_end = self.now()
        if not self.enabled or request.num_preemptions == 0:
            return native
        assert local_tokens == 0, 'APC must remain disabled'
        s, cs = self.scheduler, self.cs
        size = cs.config.kv_group_configs[0].tokens_per_block
        free = s.kv_cache_manager.block_pool.get_num_free_blocks()
        required = math.ceil(request.num_tokens / size)
        reserved = s._inflight_prefill_reserved_blocks()
        watermark = s.kv_cache_manager.watermark_blocks if s.running else 0
        ready = native[0] is not None and native[0] > 0
        capacity = required + reserved + watermark <= free
        eligible = ready and capacity
        pending_loads = sum(not j.is_store for j in cs._jobs.values())
        action = 'native'
        fallback = 'none'
        if not ready:
            fallback = 'pending_native' if native[0] is None else 'host_miss_native_recompute'
        elif not capacity:
            fallback = 'full_capacity_not_jointly_available_native'
        elif self.policy == 'host':
            action = 'host'
        elif self.policy == 'recompute':
            action = 'recompute'
        elif self.policy == 'length':
            action = 'recompute' if request.num_tokens <= self.threshold else 'host'
        else:
            raise ValueError(self.policy)
        row = dict(event=len(self.events), request_id=request.request_id,
            preemptions=request.num_preemptions, decision_s=start,
            known_tokens=request.num_tokens, prompt_tokens=request.num_prompt_tokens,
            generated_tokens=request.num_output_tokens, host_hit_tokens=native[0],
            free_blocks=free, full_required_blocks=required, reserved_blocks=reserved,
            watermark_blocks=watermark, joint_capacity=capacity, eligible=eligible,
            pending_load_jobs=pending_loads, pending_transfer_jobs=len(cs._jobs),
            running=len(s.running), waiting=len(s.waiting), recent_step_s=self.last_step_s,
            policy=self.policy, action=action, fallback=fallback)
        result = native
        if eligible:
            before = (id(request), tuple(request.all_token_ids), id(request.sampling_params),
                      request.num_output_tokens, request.status, request.stop_reason)
        if action == 'recompute':
            # Native lookup has touched LRU and populated chunk-hit metadata, but
            # has NOT allocated GPU memory, prepared a load, or executed compute.
            cs._req_status[request.request_id].update_num_hit_chunks(local_tokens)
            result = (0, False)
        if eligible:
            after = (id(request), tuple(request.all_token_ids), id(request.sampling_params),
                     request.num_output_tokens, request.status, request.stop_reason)
            assert before == after
            row['prefix_sha256'] = token_hash(request.all_token_ids)
            row['request_state_preserved'] = True
        row['lookup_seconds'] = lookup_end - start
        row['adapter_seconds'] = self.now() - lookup_end
        self.events.append(row)
        self.pending[request.request_id] = row
        return result

    def allocated(self, request, blocks, external_tokens):
        result = self._alloc(request, blocks, external_tokens)
        event = self.pending.pop(request.request_id, None)
        if self.enabled and event is not None:
            row = dict(event, allocation_s=self.now(), external_tokens=external_tokens,
                       actual_action='host' if external_tokens else 'recompute',
                       allocated_blocks=sum(len(g) for g in blocks.blocks))
            if event['action'] == 'recompute':
                assert external_tokens == 0 and event['joint_capacity']
            if event['action'] == 'host':
                assert external_tokens > 0 and event['joint_capacity']
            self.commits.append(row)
        return result

    def preempt(self, request, timestamp):
        if self.enabled:
            self.preemptions.append(dict(request_id=request.request_id, time_s=self.now(),
                known_tokens=request.num_tokens, generated_tokens=request.num_output_tokens,
                free_blocks_before_preempt=self.scheduler.kv_cache_manager.block_pool.get_num_free_blocks(),
                computed_tokens=request.num_computed_tokens))
        return self._preempt(request, timestamp)

    def update(self, output):
        meta = output.kv_connector_worker_meta
        if self.enabled and meta is not None and not meta.transfer_stats.is_empty():
            row = {'time_s': self.now()}
            for name in ('load', 'store'):
                stat = getattr(meta.transfer_stats, name)
                row[name] = dict(bytes=stat.bytes, time=stat.time, sizes=list(stat.sizes))
            self.transfers.append(row)
        return self._update(output)
