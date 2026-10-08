"""Thin vLLM 0.26 native lookup adapter; never executes both branches.

Both arms keep the same native STORE, FCFS, victim, quantum and admission.
Only ready Host hits on preempted requests with full GPU capacity are eligible.
Unsupported/unready/capacity-limited cases retain the native lookup result.
"""
import hashlib
import json
import math
import time
from copy import deepcopy
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
        assert policy in ('host', 'recompute', 'length', 'budget')
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
        self.budget_steps = []
        self._budget_context = None
        self._validate_budget_scope()
        assert scheduler.scheduler_reserve_full_isl
        assert len(self.cs.config.kv_group_configs) == 1
        assert self.cs.config.kv_group_configs[0].sliding_window_size_in_chunks is None
        assert type(self.cs.manager).__name__ == 'CPUOffloadingManager'
        self._lookup = self.cs.get_num_new_matched_tokens
        self._alloc = self.cs.update_state_after_alloc
        self._update = self.cs.update_connector_output
        self._preempt = scheduler._preempt_request
        self._schedule = scheduler.schedule
        self._allocate_slots = scheduler.kv_cache_manager.allocate_slots
        self.cs.get_num_new_matched_tokens = self.lookup
        self.cs.update_state_after_alloc = self.allocated
        self.cs.update_connector_output = self.update
        scheduler._preempt_request = self.preempt
        scheduler.schedule = self.schedule
        scheduler.kv_cache_manager.allocate_slots = self.allocate_slots

    def _validate_budget_scope(self):
        """Fail explicitly outside the audited native scheduling path."""
        s = self.scheduler
        assert s.policy.value == 'fcfs', 'budget observation requires native FCFS'
        assert s.kv_cache_manager.enable_caching is False, 'APC must remain disabled'
        assert s.vllm_config.speculative_config is None, 'speculation is unsupported'
        assert s.num_spec_tokens == 0 and s.num_lookahead_tokens == 0
        assert not s.use_eagle and s.dynamic_sd_lookup is None
        assert s.num_sampled_tokens_per_step == 1
        assert not s.is_encoder_decoder and s.max_num_encoder_input_tokens == 0
        assert s.ec_connector is None and s.lora_config is None
        assert not s.has_mamba_layers and not s.need_mamba_block_aligned_split
        assert s.scheduler_config.async_scheduling is False
        assert s.scheduler_config.long_prefill_token_threshold == 0, 'prefill threshold must be disabled'
        assert type(s.max_num_scheduled_tokens) is int and s.max_num_scheduled_tokens > 0

    def schedule(self, *args, **kwargs):
        """Observe the native call; never reproduce its scheduling algorithm."""
        assert self._budget_context is None, 'nested schedule is unsupported'
        self._validate_budget_scope()
        pause = self.scheduler._pause_state.name
        assert pause in ('UNPAUSED', 'PAUSED_NEW', 'PAUSED_ALL')
        initial = 0 if pause == 'PAUSED_ALL' else self.scheduler.max_num_scheduled_tokens
        row = dict(step=len(self.budget_steps), start_s=self.now(), pause_state=pause,
                   initial_budget_tokens=initial, allocation_calls=0,
                   failed_allocations=0, zero_token_allocations=0,
                   checked=False, status='RUNNING')
        self.budget_steps.append(row)
        context = dict(initial=initial, remaining=initial, positive_allocations={}, record=row)
        self._budget_context = context
        try:
            result = self._schedule(*args, **kwargs)
            observed = context['positive_allocations']
            expected = dict(result.num_scheduled_tokens)
            if observed != expected:
                row.update(observed_allocations=dict(observed), native_scheduled_tokens=expected)
                raise AssertionError('successful allocation tokens differ from native SchedulerOutput: '
                                     f'observed={observed!r}, native={expected!r}')
            total = sum(expected.values())
            assert total == result.total_num_scheduled_tokens
            assert context['remaining'] == initial - total >= 0
            row.update(checked=True, status='CHECKED', native_scheduled_tokens_total=total)
            return result
        except BaseException as exc:
            row.update(status='ERROR', error=repr(exc))
            raise
        finally:
            row.update(end_s=self.now(), remaining_budget_tokens=context['remaining'],
                       positive_allocation_requests=len(context['positive_allocations']),
                       allocated_compute_tokens=sum(context['positive_allocations'].values()))
            self._budget_context = None

    def allocate_slots(self, *args, **kwargs):
        context = self._budget_context
        if context is None:
            return self._allocate_slots(*args, **kwargs)
        request = kwargs.get('request', args[0] if args else None)
        count = kwargs.get('num_new_tokens', args[1] if len(args) > 1 else None)
        assert request is not None and type(count) is int and count >= 0
        context['record']['allocation_calls'] += 1
        result = self._allocate_slots(*args, **kwargs)
        if result is None:
            context['record']['failed_allocations'] += 1
        elif count == 0:
            context['record']['zero_token_allocations'] += 1
        else:
            rid = request.request_id
            assert rid not in context['positive_allocations'], 'duplicate positive allocation in one schedule'
            context['positive_allocations'][rid] = count
            context['remaining'] -= count
            assert context['remaining'] >= 0, 'successful allocations exceeded native budget'
        return result

    def budget_summary(self):
        return dict(signal='native_remaining_scheduled_token_budget',
                    checked_steps=sum(row['checked'] for row in self.budget_steps),
                    error_steps=sum(row['status'] == 'ERROR' for row in self.budget_steps),
                    steps=deepcopy(self.budget_steps))

    def now(self):
        return time.perf_counter() - self.origin

    def clear(self):
        assert self._budget_context is None, 'cannot clear during schedule'
        self.events.clear(); self.commits.clear(); self.transfers.clear(); self.preemptions.clear()
        self.budget_steps.clear()
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
            return dict(selection_mode='full_policy', target_commit_count=0, target_reached=None,
                        budget_observation=self.budget_summary())
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
            target_comparison_eligible=self.enabled and self.target_commit_count == 1,
            budget_observation=self.budget_summary())

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
        context = self._budget_context
        if self.enabled and request.num_preemptions:
            assert context is not None, 'recovery choice requires an active schedule budget'
        remaining = context['remaining'] if context is not None else None
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
        target = self.target_match(request, ready, capacity) if self.target_spec is not None else None
        if target is not None:
            assert self.policy in ('host', 'recompute')
        effective_policy = self.policy if target is None or target['target_selected'] else 'host'
        action = 'native'
        fallback = 'none'
        if not ready:
            fallback = 'pending_native' if native[0] is None else 'host_miss_native_recompute'
        elif not capacity:
            fallback = 'full_capacity_not_jointly_available_native'
        elif effective_policy == 'host':
            action = 'host'
        elif effective_policy == 'recompute':
            action = 'recompute'
        elif effective_policy == 'length':
            action = 'recompute' if request.num_tokens <= self.threshold else 'host'
        elif effective_policy == 'budget':
            assert request.num_computed_tokens == 0 and remaining is not None
            action = 'recompute' if request.num_tokens <= remaining else 'host'
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
            remaining_budget_tokens=remaining, budget_step=context['record']['step'],
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
        if self._budget_context is not None:
            assert request.request_id not in self._budget_context['positive_allocations'], (
                'FCFS unexpectedly preempted an already scheduled request; budget refund unsupported')
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
