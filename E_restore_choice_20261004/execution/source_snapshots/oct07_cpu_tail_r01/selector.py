"""Thin vLLM 0.26 native lookup adapter; never executes both branches.

Both arms keep the same native STORE, FCFS, victim, quantum and admission.
Only ready Host hits on preempted requests with full GPU capacity are eligible.
Unsupported/unready/capacity-limited cases retain the native lookup result.
"""
import hashlib
import json
import math
import time
from collections import Counter


def token_hash(tokens):
    return hashlib.sha256(json.dumps(list(tokens)).encode()).hexdigest()


def validate_target_spec(spec):
    if spec is None:
        return
    assert spec['schema'] == 'E.one_event_target.v1'
    target = spec['target']
    assert isinstance(target['external_id'], str) and target['external_id']
    for field in ('num_preemptions', 'known_tokens', 'generated_tokens'):
        assert type(target[field]) is int and target[field] > 0
    assert target['known_tokens'] > target['generated_tokens']
    value = target['prefix_sha256']
    assert isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


class RestoreSelector:
    def __init__(self, scheduler, policy, threshold=0, target_spec=None):
        validate_target_spec(target_spec)
        assert policy in ('host', 'recompute', 'length', 'headroom')
        assert target_spec is None or policy in ('host', 'recompute')
        self.scheduler = scheduler
        self.cs = scheduler.connector.connector_scheduler
        self.policy, self.threshold = policy, threshold
        self.events, self.commits, self.transfers, self.preemptions = [], [], [], []
        self.last_step_s = 0.0
        self.origin = time.perf_counter()
        self.pending = {}
        self.enabled = True
        self.target_spec = target_spec
        self.external_ids = {}
        self.target_commit_count = 0
        self.target_identity_lookups = self.target_state_lookups = self.target_eligible_lookups = 0
        self.target_mismatch_counts = Counter()
        self.target_effective_action = None
        self._validate_headroom_scope()
        assert scheduler.scheduler_reserve_full_isl
        assert len(self.cs.config.kv_group_configs) == 1
        assert self.cs.config.kv_group_configs[0].sliding_window_size_in_chunks is None
        assert type(self.cs.manager).__name__ == 'CPUOffloadingManager'
        self._lookup = self.cs.get_num_new_matched_tokens
        self._alloc = self.cs.update_state_after_alloc
        self._update = self.cs.update_connector_output
        self._preempt = scheduler._preempt_request
        self.cs.get_num_new_matched_tokens = self.lookup
        self.cs.update_state_after_alloc = self.allocated
        self.cs.update_connector_output = self.update
        scheduler._preempt_request = self.preempt

    def _validate_headroom_scope(self):
        """Fail before installing hooks outside the audited native stack."""
        s = self.scheduler
        assert s.policy.value == 'fcfs', 'headroom observation requires native FCFS'
        assert s.kv_cache_manager.enable_caching is False, 'APC must remain disabled'
        assert s.scheduler_config.async_scheduling is False
        assert s.scheduler_config.long_prefill_token_threshold == 0
        assert s.vllm_config.speculative_config is None
        assert s.num_spec_tokens == 0 and s.num_lookahead_tokens == 0
        assert not s.use_eagle and s.dynamic_sd_lookup is None
        assert s.num_sampled_tokens_per_step == 1
        assert not s.is_encoder_decoder and s.max_num_encoder_input_tokens == 0
        assert s.ec_connector is None and s.lora_config is None
        assert not s.has_mamba_layers and not s.need_mamba_block_aligned_split
        assert not s.use_pp and s.dcp_world_size == s.pcp_world_size == 1
        assert s.parallel_config.data_parallel_size == 1

    def headroom_feature(self, request, size, slack):
        """Snapshot the current decode set, never future EOS or token demand.

        Native computed counts advance only at _update_after_schedule. At a
        waiting lookup, supported running requests have already received this
        step's one-token allocation. This is not a next-step safety guarantee:
        later waiting allocations and requests outside this set can change KV.
        """
        s = self.scheduler
        self._validate_headroom_scope()
        assert request.num_computed_tokens == 0, 'expected native waiting lookup'
        assert size == s.block_size and size > 0
        running, growth, unsupported, reasons = [], [], [], {}
        for r in [request, *s.running]:
            assert not r.has_encoder_inputs and r.lora_request is None
            assert not r.spec_token_ids and r.num_output_placeholders == 0
            assert r.num_in_flight_tokens == 0
            if r is request:
                continue
            row = dict(request_id=r.request_id, known_tokens=r.num_tokens,
                       computed_tokens=r.num_computed_tokens)
            running.append(dict(row, is_prefill_chunk=r.is_prefill_chunk,
                                next_decode_eligible_step=r.next_decode_eligible_step))
            why = []
            if r.num_tokens != r.num_computed_tokens + 1:
                why.append('not_one_token_ready')
            if s.current_step < r.next_decode_eligible_step:
                why.append('decode_cadence_skip')
            if r.num_tokens >= s.max_model_len:
                why.append('model_length_skip')
            # throttle_prefills is a native local argument, not observed here.
            # Exclude its possible skip set instead of wrapping schedule.
            if r.is_prefill_chunk:
                why.append('prefill_chunk')
            if why:
                unsupported.append(dict(row))
                reasons[r.request_id] = why
            elif r.num_tokens % size == 0:
                growth.append(dict(row))
        target_growth = int(request.num_tokens % size == 0)
        return dict(schema='E.headroom.v1', block_tokens=size,
            current_slack_blocks=slack, supported=not unsupported,
            background_next_decode_growth_blocks=len(growth),
            target_next_decode_growth_blocks=target_growth,
            margin_blocks=None if unsupported else slack - len(growth) - target_growth,
            running_requests=running, background_requests=growth,
            unsupported_background_requests=unsupported,
            unsupported_background_reasons=reasons)

    def now(self):
        return time.perf_counter() - self.origin

    def clear(self):
        self.events.clear(); self.commits.clear(); self.transfers.clear(); self.preemptions.clear()
        self.pending.clear()
        self.origin = time.perf_counter()
        self.last_step_s = 0.0
        self.external_ids.clear()
        self.target_commit_count = 0
        self.target_identity_lookups = self.target_state_lookups = self.target_eligible_lookups = 0
        self.target_mismatch_counts.clear()
        self.target_effective_action = None

    def register_request(self, internal_id, external_id):
        """Use the engine's returned identity, never parse its random suffix."""
        assert self.target_spec is not None
        assert internal_id not in self.external_ids
        self.external_ids[internal_id] = external_id

    def target_summary(self):
        if self.target_spec is None:
            return dict(selection_mode='full_policy', target_commit_count=0, target_reached=None)
        reasons = dict(self.target_mismatch_counts)
        if not self.target_commit_count:
            if not self.target_identity_lookups:
                reasons['target_recovery_not_observed'] = 1
            elif self.target_eligible_lookups:
                reasons['target_allocation_not_committed'] = 1
        return dict(selection_mode='one_event', target_external_id=self.target_spec['target']['external_id'],
            target_commit_count=self.target_commit_count, target_reached=self.target_commit_count == 1,
            target_result='DISABLED' if not self.enabled else
                ('TARGET_COMMITTED' if self.target_commit_count else 'TARGET_NOT_REACHED'),
            target_effective_action=self.target_effective_action,
            target_identity_lookups=self.target_identity_lookups,
            target_state_lookups=self.target_state_lookups,
            target_eligible_lookups=self.target_eligible_lookups,
            target_mismatch_counts=reasons,
            target_comparison_eligible=self.enabled and self.target_commit_count == 1)

    def target_match(self, request, ready, capacity):
        """Only current, already-known request state enters this predicate."""
        target = self.target_spec['target']
        external = self.external_ids.get(request.request_id)
        result = dict(selection_mode='one_event', external_id=external,
            target_external_id=target['external_id'], target_selected=False,
            target_state_match=False, target_commit_count_before=self.target_commit_count)
        if external != target['external_id']:
            result['target_reason'] = 'other_request_native_host' if external else 'external_id_unregistered_native_host'
            return result
        self.target_identity_lookups += 1
        observed = dict(num_preemptions=request.num_preemptions, known_tokens=request.num_tokens,
                        generated_tokens=request.num_output_tokens, prefix_sha256=token_hash(request.all_token_ids))
        result['prefix_sha256'] = observed['prefix_sha256']
        mismatches = [key + '_mismatch' for key, value in observed.items() if value != target[key]]
        result['target_state_match'] = not mismatches
        if not mismatches:
            self.target_state_lookups += 1
        if not ready:
            mismatches.append('native_host_not_ready')
        if not capacity:
            mismatches.append('joint_capacity_unavailable')
        if not mismatches:
            self.target_eligible_lookups += 1
        if self.target_commit_count:
            mismatches.append('target_already_committed')
        self.target_mismatch_counts.update(mismatches)
        result['target_selected'] = not mismatches
        result['target_reason'] = 'matched_legal_target' if not mismatches else ','.join(mismatches)
        return result

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
        headroom = self.headroom_feature(request, size, free - required - reserved - watermark) if eligible else None
        pending_loads = sum(not j.is_store for j in cs._jobs.values())
        target = self.target_match(request, ready, capacity) if self.target_spec is not None else None
        if target is not None:
            assert self.policy in ('host', 'recompute')
        effective_policy = self.policy if target is None or target['target_selected'] else 'host'
        action = 'native'
        fallback = 'none'
        policy_reason = 'native_fallback'
        if not ready:
            fallback = 'pending_native' if native[0] is None else 'host_miss_native_recompute'
        elif not capacity:
            fallback = 'full_capacity_not_jointly_available_native'
        elif effective_policy == 'host':
            action = 'host'
            policy_reason = 'fixed_host'
        elif effective_policy == 'recompute':
            action = 'recompute'
            policy_reason = 'fixed_recompute'
        elif effective_policy == 'length':
            action = 'recompute' if request.num_tokens <= self.threshold else 'host'
            policy_reason = 'length_within_threshold' if action == 'recompute' else 'length_above_threshold'
        elif effective_policy == 'headroom':
            if not headroom['supported']:
                action, policy_reason = 'host', 'headroom_background_unsupported'
            elif request.num_tokens > self.threshold:
                action, policy_reason = 'host', 'headroom_length_above_threshold'
            elif headroom['margin_blocks'] < 0:
                action, policy_reason = 'host', 'headroom_negative_margin'
            else:
                action, policy_reason = 'recompute', 'headroom_nonnegative_margin'
        else:
            raise ValueError(self.policy)
        row = dict(event=len(self.events), request_id=request.request_id,
            preemptions=request.num_preemptions, decision_s=start,
            known_tokens=request.num_tokens, prompt_tokens=request.num_prompt_tokens,
            generated_tokens=request.num_output_tokens, host_hit_tokens=native[0],
            free_blocks=free, full_required_blocks=required, reserved_blocks=reserved,
            watermark_blocks=watermark, joint_capacity=capacity, eligible=eligible,
            headroom=headroom, headroom_supported=headroom['supported'] if eligible else None,
            scheduler_step=s.current_step, policy_reason=policy_reason,
            pending_load_jobs=pending_loads, pending_transfer_jobs=len(cs._jobs),
            running=len(s.running), waiting=len(s.waiting), recent_step_s=self.last_step_s,
            policy=self.policy, action=action, fallback=fallback)
        if target is not None:
            row.update(target, effective_policy=effective_policy, target_committed=False)
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
            if event.get('target_selected'):
                # A lookup that fails to allocate never consumes the one-event
                # budget; this native callback is the successful commit point.
                assert self.target_spec is not None and self.target_commit_count == 0
                assert event['eligible'] and event['target_state_match']
                assert row['actual_action'] == event['action']
                assert token_hash(request.all_token_ids) == event['prefix_sha256']
                assert request.num_preemptions == event['preemptions']
                assert request.num_tokens == event['known_tokens']
                assert request.num_output_tokens == event['generated_tokens']
                self.target_commit_count = 1
                self.target_effective_action = row['actual_action']
                row.update(target_committed=True, target_commit_count_after=1)
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
