"""Pinned full-attention native-fit observer and one bounded admission probe.

No allocation is attempted speculatively. The two capacity queries are native
read-only methods; exact FullAttentionManager makes remove_skipped_blocks a no-op.
"""
import hashlib
import inspect
import math
from pathlib import Path
import textwrap
import time
import types

SOURCE_SHA = {
    'scheduler.py': '2ed2a550b6558b2495eda845a97ae38bcf0225027b9e25fbf00fc3880c1d3941',
    'kv_cache_manager.py': '3f4af8d247f3fe9570b0132818b832b66ae6a2ac12942588828f899f6ff77ccf',
    'kv_cache_coordinator.py': '4c8fbb341f0bd3714eff54ce633f534c1475220decb17c0caed40cbd02b352a2',
    'single_type_kv_cache_manager.py': 'bcb27e38895332bf6a4c55608f2917eb9fd941ec5a629c14b88358f00aadeba8',
}


def pure_fit(k, request, *, num_new_tokens, num_new_computed_tokens=0,
             new_computed_blocks=None, num_lookahead_tokens=0,
             num_external_computed_tokens=0, delay_cache_blocks=False,
             num_encoder_tokens=0, full_sequence_must_fit=True,
             reserved_blocks=0, has_scheduled_reqs=True):
    """Exact None-branch prediction ONLY after install's pinned-type assertions."""
    assert full_sequence_must_fit and not request.has_encoder_inputs
    assert num_new_computed_tokens == num_lookahead_tokens == num_encoder_tokens == 0
    assert num_new_tokens > 0 or num_external_computed_tokens > 0
    blocks = (new_computed_blocks if new_computed_blocks is not None else k.empty_kv_cache_blocks).blocks
    assert len(blocks) == 1 and not blocks[0]
    local = request.num_computed_tokens + num_new_computed_tokens
    assert 0 <= local + num_external_computed_tokens <= request.num_tokens <= k.max_model_len
    assert 0 <= num_new_tokens <= request.num_tokens-local-num_external_computed_tokens
    total = min(local + num_external_computed_tokens, k.max_model_len)
    full = min(request.num_tokens, k.max_model_len)
    free = k.block_pool.get_num_free_blocks()
    watermark = k.watermark_blocks if has_scheduled_reqs and request.status.name in ('WAITING', 'PREEMPTED') else 0
    common = dict(request_id=request.request_id, new_computed_blocks=blocks,
                  num_encoder_tokens=num_encoder_tokens, num_local_computed_tokens=local)
    full_blocks = k.coordinator.get_num_blocks_to_allocate(**common,
        num_tokens=full, total_computed_tokens=total, num_tokens_main_model=full, apply_admission_cap=True)
    chunk_blocks = None
    if full_blocks + watermark <= free:
        main = total + num_new_tokens
        chunk_blocks = k.coordinator.get_num_blocks_to_allocate(**common,
            num_tokens=min(main + num_lookahead_tokens, k.max_model_len),
            total_computed_tokens=local + num_external_computed_tokens, num_tokens_main_model=main)
    return dict(fits=chunk_blocks is not None and chunk_blocks + watermark <= free-reserved_blocks,
        free_blocks=free, full_required_blocks=full_blocks, chunk_required_blocks=chunk_blocks,
        native_watermark_blocks=watermark, native_reserved_blocks=reserved_blocks,
        num_new_tokens=num_new_tokens, external_computed_tokens=num_external_computed_tokens,
        load_kv_async=delay_cache_blocks)


def qualify(s):
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.kv_cache_coordinator import KVCacheCoordinatorNoPrefixCache
    from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
    from vllm.v1.kv_cache_interface import FullAttentionSpec
    k = s.kv_cache_manager
    assert type(k) is KVCacheManager and type(k.coordinator) is KVCacheCoordinatorNoPrefixCache
    assert k.num_kv_cache_groups == len(k.coordinator.single_type_managers) == 1
    m = k.coordinator.single_type_managers[0]
    assert type(m) is FullAttentionManager and type(m.kv_cache_spec) is FullAttentionSpec
    for obj, filename in ((s, 'scheduler.py'), (k, 'kv_cache_manager.py'),
                          (k.coordinator, 'kv_cache_coordinator.py'), (m, 'single_type_kv_cache_manager.py')):
        assert hashlib.sha256(Path(inspect.getsourcefile(type(obj))).read_bytes()).hexdigest() == SOURCE_SHA[filename]
    assert not k.enable_caching and not k.coordinator.enable_caching and not m.enable_caching
    assert not k.use_eagle and not k.coordinator.eagle_group_ids
    assert m.dcp_world_size == m.pcp_world_size == 1 and m._max_admission_blocks_per_request is None
    assert k.block_pool is k.coordinator.block_pool is m.block_pool
    assert s.scheduler_config.async_scheduling is False and s.scheduler_reserve_full_isl
    assert s.policy.name == 'FCFS'
    assert s.parallel_config.tensor_parallel_size == s.parallel_config.pipeline_parallel_size == s.parallel_config.data_parallel_size == 1
    assert s.max_model_len == k.max_model_len == 4096 and m.block_size == 16
    assert s.vllm_config.model_config.hf_config.model_type == 'olmoe'
    assert str(s.vllm_config.model_config.dtype) == 'torch.bfloat16'
    assert not s.num_spec_tokens and not s.num_lookahead_tokens and not s.use_eagle
    assert not s.has_mamba_layers and not s.need_mamba_block_aligned_split and not s.is_encoder_decoder
    assert s.lora_config is None and s.ec_connector is None
    assert not any(not r.is_finished() for r in s.requests.values()), 'Install between drained cells only'


class Gate:
    def __init__(self, scheduler, mode='kv', cap=256, kv_floor=3277,
                 probe_enabled=False, delay_s=.1, max_extra_s=.25, release_mode='timer'):
        assert mode == 'kv' and type(probe_enabled) is bool and cap > 0 and kv_floor >= 0
        assert all(math.isfinite(v) and v >= 0 for v in (delay_s, max_extra_s))
        assert release_mode in ('timer', 'output_progress')
        assert release_mode != 'output_progress' or max_extra_s >= .1
        self.s, self.mode, self.cap, self.kv_floor = scheduler, mode, cap, kv_floor
        self.probe_enabled, self.delay_s, self.max_extra_s = probe_enabled, delay_s, max_extra_s
        self.release_mode = release_mode
        self.origin = time.perf_counter()
        self.admitted_ids, self.started_ids = set(), set()
        self.held, self.snapshots, self.decisions, self.starts, self.allocations = [], [], [], [], []
        self.scheduled_tokens, self.epochs, self.pending_since, self.previous_pending = {}, {}, {}, set()
        self.previous_computed, self.previous_observation, self.sample_at = {}, None, None
        self.event, self.release_at, self.hard_deadline = None, None, None
        self.progress_min_release_at = None
        self.trigger_requests, self.recovery_progress = {}, {}
        self.release_observations = dict(timer=None, output_progress=None, actual=None)
        self.barrier, self.barrier_deadline = None, None
        self.cpu_ns = self.wall_ns = self.calls = self.fit_checks = self.fit_mismatches = 0
        self.idle_sleep_s = 0.

    def timed(self, fn, *args, **kwargs):
        wall, cpu = time.perf_counter_ns(), time.process_time_ns()
        try:
            return fn(*args, **kwargs)
        finally:
            self.wall_ns += time.perf_counter_ns()-wall
            self.cpu_ns += time.process_time_ns()-cpu

    def begin(self):
        assert not self.held
        self.calls += 1
        self.observe_trigger_progress(time.perf_counter())
        self.release_status(time.perf_counter())
        self.barrier = self.barrier_deadline = None
        self.scheduled_tokens = {}
        self.admitted_ids = {rid for rid in self.admitted_ids
            if rid in self.s.requests and not self.s.requests[rid].is_finished()}
        self.epochs = {rid: epoch for rid, epoch in self.epochs.items() if rid in self.admitted_ids}
        # A native preemption suppresses the waiting loop that step. Capture its
        # fixed replay target on the next begin, even between 100 ms snapshots.
        for rid in self.admitted_ids:
            req = self.s.requests[rid]
            if req.num_preemptions and self.epochs.get(rid, (None,))[0] != req.num_preemptions:
                self.epochs[rid] = (req.num_preemptions, req.num_tokens)
        now = time.perf_counter()
        if self.sample_at is None or now-self.sample_at >= .1:
            self.snapshots.append(dict(t=now-self.origin, **self.state()))
            self.sample_at = now

    def is_new(self, req):
        return (req.status.name == 'WAITING' and req.request_id not in self.admitted_ids
                and req.request_id not in self.started_ids and req.num_preemptions == 0
                and req.num_computed_tokens == 0 and req.num_output_tokens == 0)

    def state(self):
        now = time.perf_counter()
        groups = {key: [] for key in ('waiting', 'remote_inflight', 'ready', 'running_recompute', 'scheduled_compute')}
        computed = {}
        finished_remote = self.s.finished_recving_kv_req_ids
        for rid in self.admitted_ids:
            r = self.s.requests.get(rid)
            if r is None or r.is_finished() or not r.num_preemptions:
                continue
            epoch = self.epochs.get(rid)
            if epoch is None or epoch[0] != r.num_preemptions:
                epoch = self.epochs[rid] = (r.num_preemptions, r.num_tokens)
            status = r.status.name
            # Async setup writes num_computed_tokens before the receive finishes.
            # Do not call that progress; also stop counting once replay is done.
            valid_prefix = (0 if status == 'WAITING_FOR_REMOTE_KVS' and rid not in finished_remote
                            else min(r.num_computed_tokens, epoch[1]))
            computed[rid] = (r.num_preemptions, valid_prefix)
            if status == 'WAITING_FOR_REMOTE_KVS':
                groups['ready' if rid in finished_remote else 'remote_inflight'].append(rid)
            elif status in ('PREEMPTED', 'WAITING'):
                groups['ready' if r.num_computed_tokens > 0 else 'waiting'].append(rid)
            elif status == 'RUNNING' and r.num_computed_tokens < epoch[1]:
                groups['running_recompute'].append(rid)
                if self.scheduled_tokens.get(rid, 0) > 0:
                    groups['scheduled_compute'].append(rid)
        pending = set(groups['waiting'] + groups['remote_inflight'] + groups['ready'])
        self.pending_since = {rid: self.pending_since.get(rid, now) for rid in pending}
        progress = sum(max(0, n-old[1]) for rid, (epoch, n) in computed.items()
                       if (old := self.previous_computed.get(rid)) is not None and old[0] == epoch)
        st = dict(free_blocks=self.s.kv_cache_manager.block_pool.get_num_free_blocks(),
            running=len(self.s.running), active=len(self.admitted_ids), admitted_inflight=len(self.admitted_ids),
            recovery_count=len(pending), latch=False,
            oldest_recovery_wait_s=max((now-t for t in self.pending_since.values()), default=0.),
            recovery_entered=len(pending-self.previous_pending), recovery_departed=len(self.previous_pending-pending),
            recovery_computed_advance_tokens=progress,
            observation_interval_s=None if self.previous_observation is None else now-self.previous_observation)
        st.update({f'recovery_{key}_count': len(ids) for key, ids in groups.items()})
        self.previous_pending, self.previous_computed, self.previous_observation = pending, computed, now
        return st

    def base(self, req, st):
        age = max(0., time.time()-req.arrival_time)
        return age, st['active'] >= self.cap, st['free_blocks'] < self.kv_floor and age < 10.

    def progress_observation(self, req, now, **extra):
        return dict(t=now-self.origin, perf_s=now, status=req.status.name,
                    computed=req.num_computed_tokens, output=req.num_output_tokens,
                    preemptions=req.num_preemptions, **extra)

    def observe_trigger_progress(self, now):
        # Hold only the one frozen cohort's original objects. A request removed
        # from the live dict is not evidence of completion; is_finished() is.
        for rid, req in self.trigger_requests.items():
            progress = self.recovery_progress[rid]
            if self.s.requests.get(rid) is not req and progress['first_missing'] is None:
                progress['first_missing'] = self.progress_observation(req, now)
            if req.num_output_tokens > progress['initial_output'] and progress['first_output'] is None:
                progress['first_output'] = self.progress_observation(req, now)
            if req.is_finished() and progress['first_finished'] is None:
                progress['first_finished'] = self.progress_observation(req, now)

    def release_status(self, now):
        """Both observed release rules, independent of native capacity/actual admission."""
        if self.event is None:
            return False, False
        timer_block = now < self.release_at and now < self.hard_deadline
        progressed = all(p['first_output'] is not None or p['first_finished'] is not None
                         for p in self.recovery_progress.values())
        progress_block = now < self.hard_deadline and (now < self.progress_min_release_at or not progressed)
        for mode, blocked in (('timer', timer_block), ('output_progress', progress_block)):
            if not blocked and self.release_observations[mode] is None:
                reason = ('hard_deadline' if now >= self.hard_deadline else
                          'timer_elapsed' if mode == 'timer' else 'output_progress_or_finished')
                self.release_observations[mode] = dict(t=now-self.origin, perf_s=now, reason=reason)
        if self.probe_enabled and self.release_observations['actual'] is None:
            observed = self.release_observations[self.release_mode]
            if observed is not None:
                self.release_observations['actual'] = dict(observed, release_mode=self.release_mode)
        return timer_block, progress_block

    def selected_blocks(self, windows):
        return windows[0 if self.release_mode == 'timer' else 1]

    def early(self, req):
        # Reinsert the older held request before permitting any later new one
        # if the probe window expires during this very same native schedule.
        windows = self.release_status(time.perf_counter())
        if self.barrier == 'probe' and not self.selected_blocks(windows):
            self.restore()
            self.barrier = self.barrier_deadline = None
            return 'restart'
        if self.barrier is None or not self.is_new(req):
            return None
        st = self.state()
        age, cap_block, kv_block = self.base(req, st)
        windows = self.release_status(time.perf_counter())
        if self.barrier == 'probe' and not self.selected_blocks(windows):
            self.restore()
            self.barrier = self.barrier_deadline = None
            return 'restart'
        self.decisions.append(dict(t=time.perf_counter()-self.origin, request_id=req.request_id, **st,
            age_s=age, signal_wait_limit_bypass=age >= 10., denied=True, reason='fifo_'+self.barrier,
            base_allowed=None, kv_only_denied=cap_block or kv_block, changed_by_recovery=False,
            timer_would_block=self.barrier == 'probe' and windows[0],
            progress_would_block=self.barrier == 'probe' and windows[1],
            changed_by_progress=False, progress_additional_opportunity=False,
            probe_fifo_held=self.barrier == 'probe', native_fit=None, native_allocation_result=None))
        return 'hold'

    def before_allocate(self, req, *, token_budget, **kwargs):
        fit = pure_fit(self.s.kv_cache_manager, req, **kwargs)
        check = dict(fit=fit, row=None, status_before=req.status.name, preemptions=req.num_preemptions)
        if not self.is_new(req):
            return check
        now = time.perf_counter()
        st = self.state()
        age, cap_block, kv_block = self.base(req, st)
        base_allowed = fit['fits'] and not cap_block and not kv_block
        local_prefill = kwargs['num_new_tokens'] > 0 and not kwargs.get('delay_cache_blocks', False)
        opportunity = base_allowed and local_prefill and st['recovery_count'] > 0
        if opportunity and self.event is None:
            # state() has just classified true pending recovery. Running replay
            # is deliberately excluded from this frozen release cohort.
            self.trigger_requests = {rid: self.s.requests[rid] for rid in sorted(self.pending_since)}
            assert self.trigger_requests
            self.recovery_progress = {rid: dict(initial_output=r.num_output_tokens,
                initial_status=r.status.name, initial_preemptions=r.num_preemptions,
                first_scheduled=None, first_output=None, first_finished=None, first_missing=None)
                for rid, r in self.trigger_requests.items()}
            self.event = dict(kind='action' if self.probe_enabled else 'shadow', request_id=req.request_id,
                t=now-self.origin, first_block_perf_s=now if self.probe_enabled else None,
                state=dict(st), fit=dict(fit), token_budget=token_budget,
                scheduled_tokens=dict(self.scheduled_tokens),
                pending_targets=[dict(request_id=rid, status=r.status.name, computed=r.num_computed_tokens,
                    output=r.num_output_tokens, preemptions=r.num_preemptions)
                    for rid, r in self.trigger_requests.items()],
                requests=[dict(request_id=r.request_id, status=r.status.name, computed=r.num_computed_tokens,
                    output=r.num_output_tokens, preemptions=r.num_preemptions,
                    num_tokens=r.num_tokens, admitted=r.request_id in self.admitted_ids)
                    for r in self.s.requests.values() if not r.is_finished()])
            blocked_at = time.perf_counter()
            if self.probe_enabled:
                self.event['first_block_perf_s'] = blocked_at
            self.release_at = blocked_at + min(self.delay_s, self.max_extra_s)
            self.progress_min_release_at = blocked_at + min(max(.1, self.delay_s), self.max_extra_s)
            self.hard_deadline = blocked_at + self.max_extra_s
            self.event.update(release_perf_s=self.release_at, hard_deadline_perf_s=self.hard_deadline,
                progress_min_release_perf_s=self.progress_min_release_at, observation_anchor_perf_s=blocked_at)
        decision_now = time.perf_counter()
        windows = self.release_status(decision_now)
        target = self.event is not None and req.request_id == self.event['request_id']
        timer_would_block, progress_would_block = target and windows[0], target and windows[1]
        progress_additional_opportunity = bool(target and base_allowed and not windows[0] and windows[1])
        probe_block = (self.probe_enabled and self.event is not None
            and req.request_id == self.event['request_id'] and base_allowed
            and self.selected_blocks(windows))
        # Preserve the native failed-allocation break, including for a baseline-denied request.
        denied = fit['fits'] and (cap_block or kv_block or probe_block)
        reason = ('native_capacity' if not fit['fits'] else 'cap' if cap_block else
                  'kv' if kv_block else 'probe_progress' if probe_block and not timer_would_block else
                  'probe' if probe_block else 'allow')
        row = dict(t=decision_now-self.origin, request_id=req.request_id, **st, **{k:v for k,v in fit.items() if k != 'free_blocks'},
            age_s=age, signal_wait_limit_bypass=age >= 10., token_budget=token_budget,
            native_fit=fit['fits'], native_allocation_result=None,
            base_allowed=base_allowed, kv_only_denied=cap_block or kv_block,
            timer_would_block=timer_would_block, progress_would_block=progress_would_block,
            progress_additional_opportunity=progress_additional_opportunity,
            changed_by_progress=bool(probe_block and progress_additional_opportunity),
            opportunity=opportunity, denied=denied, reason=reason, changed_by_recovery=bool(probe_block))
        self.decisions.append(row)
        check['row'] = row
        if denied:
            self.barrier = 'probe' if probe_block else reason
            self.barrier_deadline = (self.release_at if self.release_mode == 'timer' else self.hard_deadline) if probe_block else None
        return check

    def after_allocate(self, req, check, result):
        actual = result is not None
        self.fit_checks += 1
        if check['row'] is not None:
            check['row']['native_allocation_result'] = actual
        self.allocations.append(dict(t=time.perf_counter()-self.origin, request_id=req.request_id,
            status_before=check['status_before'], preemptions=check['preemptions'],
            predicted=check['fit']['fits'], actual=actual, **check['fit']))
        if actual != check['fit']['fits']:
            self.fit_mismatches += 1
            raise AssertionError('Native fit prediction mismatch: '+req.request_id)

    def admitted(self, req):
        assert req.status.name in ('RUNNING', 'WAITING_FOR_REMOTE_KVS') and not req.is_finished()
        self.admitted_ids.add(req.request_id)

    def hold(self, queue, req):
        self.held.append((queue, req))

    def restore(self):
        for queue, req in reversed(self.held):
            queue.prepend_request(req)
        self.held.clear()

    def end(self, result):
        now = time.perf_counter()
        for rid, req in self.trigger_requests.items():
            tokens = result.num_scheduled_tokens.get(rid, 0)
            if tokens > 0 and self.recovery_progress[rid]['first_scheduled'] is None:
                self.recovery_progress[rid]['first_scheduled'] = self.progress_observation(req, now,
                    scheduled_tokens=tokens)
        for rid in result.num_scheduled_tokens:
            if rid not in self.started_ids:
                self.starts.append(dict(request_id=rid, first_prefill_perf_s=now, first_prefill_unix_s=time.time()))
                self.started_ids.add(rid)
        # A no-token probe step must not spin until its deadline. This short
        # bounded sleep also leaves repeated native connector polling possible.
        if not result.num_scheduled_tokens and self.probe_enabled and self.event is not None:
            now = time.perf_counter()
            windows = self.release_status(now)
            deadline = self.release_at if self.release_mode == 'timer' else self.hard_deadline
            delay = min(.001, deadline-now) if self.selected_blocks(windows) else 0.
            if delay > 0:
                before = time.perf_counter()
                time.sleep(delay)
                self.idle_sleep_s += time.perf_counter()-before

    def report(self):
        return dict(mode=self.mode, cap=self.cap, kv_floor=self.kv_floor, max_signal_wait_s=10,
            origin_perf_s=self.origin, calls=self.calls, controller_cpu_s=self.cpu_ns/1e9,
            controller_wall_s=self.wall_ns/1e9, controller_idle_sleep_s=self.idle_sleep_s,
            controller_wall_excluding_idle_sleep_s=self.wall_ns/1e9-self.idle_sleep_s,
            snapshots=self.snapshots, decisions=self.decisions, starts=self.starts,
            native_allocations=self.allocations, fit_checks=self.fit_checks, fit_mismatches=self.fit_mismatches,
            first_opportunity=self.event, probe_enabled=self.probe_enabled,
            release_mode=self.release_mode, recovery_progress=self.recovery_progress,
            release_observations=self.release_observations,
            delay_s=self.delay_s, max_extra_s=self.max_extra_s, source_sha256=SOURCE_SHA,
            active_definition='unique successfully admitted and not finished; preemption does not release cap',
            semantics='Active counts unique successfully admitted unfinished requests including all recovery states. '
                'Fit uses pinned native read-only full/chunk capacity queries; actual native allocations verify it. '
                'Recovery categories are action-time host states; scheduled_compute is native scheduled recompute, '
                'not GPU completion. Only one first legal pending-recovery opportunity is shadowed/probed. '
                'Both release rules observe the same frozen pending cohort; output_progress requires at least '
                '100 ms and each retained request output increase or known finish, with the same hard deadline. '
                'First scheduled is a native schedule return; first output/finish and release times are host '
                'observations, not GPU execution or actual new admission. Decision window suggestions precede '
                'native/cap/KV constraints; only progress_additional_opportunity checks all those constraints. '
                'After the probe deadline there is no extra probe denial; native/baseline waiting can persist. '
                'All latency uses unchanged external arrivals; no promised bound on resulting prefill latency.')


def patch_source(source):
    def replace(old, new):
        nonlocal source
        assert source.count(old) == 1, 'Pinned scheduler anchor changed'
        source = source.replace(old, new)
    anchor = '    num_scheduled_tokens: dict[str, int] = {}\n'
    replace(anchor, anchor+'    self._c_probe.scheduled_tokens = num_scheduled_tokens\n')
    anchor = '            request_id = request.request_id\n\n            # try to promote blocked statuses'
    replace(anchor, '            request_id = request.request_id\n'
        "            _probe_early = self._c_probe.timed(self._c_probe.early, request)\n"
        "            if _probe_early == 'restart':\n                continue\n"
        "            if _probe_early == 'hold':\n"
        '                self._c_probe.timed(self._c_probe.hold, request_queue, request_queue.pop_request())\n'
        '                continue\n\n            # try to promote blocked statuses')
    anchor = '            new_blocks = self.kv_cache_manager.allocate_slots(\n                request,\n                num_new_tokens,\n                num_new_computed_tokens='
    before = '''            _probe_check = self._c_probe.timed(self._c_probe.before_allocate, request,
                token_budget=token_budget, num_new_tokens=num_new_tokens,
                num_new_computed_tokens=num_new_local_computed_tokens, new_computed_blocks=new_computed_blocks,
                num_lookahead_tokens=effective_lookahead_tokens, num_external_computed_tokens=num_external_computed_tokens,
                delay_cache_blocks=load_kv_async, num_encoder_tokens=num_encoder_tokens,
                full_sequence_must_fit=self.scheduler_reserve_full_isl, reserved_blocks=reserved_blocks,
                has_scheduled_reqs=bool(self.running))
            if _probe_check['row'] is not None and _probe_check['row']['denied']:
                self._c_probe.timed(self._c_probe.hold, request_queue, request_queue.pop_request())
                continue
'''
    replace(anchor, before+anchor)
    anchor = '            if new_blocks is None:\n                # The request cannot be scheduled.\n'
    replace(anchor, '            self._c_probe.timed(self._c_probe.after_allocate, request, _probe_check, new_blocks)\n'+anchor)
    anchor = '                request.status = RequestStatus.WAITING_FOR_REMOTE_KVS\n'
    replace(anchor, anchor+'                self._c_probe.timed(self._c_probe.admitted, request)\n')
    anchor = '            request.status = RequestStatus.RUNNING\n'
    replace(anchor, anchor+'            self._c_probe.timed(self._c_probe.admitted, request)\n')
    return source


def install(scheduler, mode='kv', cap=256, kv_floor=3277, probe_enabled=False, delay_s=.1, max_extra_s=.25,
            release_mode='timer'):
    qualify(scheduler)
    assert 'schedule' not in vars(scheduler) and '_c_probe' not in vars(scheduler)
    module = inspect.getmodule(type(scheduler))
    source = patch_source(textwrap.dedent(inspect.getsource(type(scheduler).schedule)))
    ns = dict(vars(module))
    exec(compile(source, '<C_native_fit_probe>', 'exec'), ns)
    patched = types.MethodType(ns['schedule'], scheduler)
    gate = Gate(scheduler, mode, cap, kv_floor, probe_enabled, delay_s, max_extra_s, release_mode)
    scheduler._c_probe = gate
    def schedule(*args, **kwargs):
        gate.timed(gate.begin)
        try:
            result = patched(*args, **kwargs)
            return result
        finally:
            gate.timed(gate.restore)
            if 'result' in locals():
                gate.timed(gate.end, result)
    scheduler.schedule = schedule
    def uninstall():
        gate.timed(gate.restore)
        del scheduler.schedule
        del scheduler._c_probe
        return gate.report()
    return gate, uninstall
