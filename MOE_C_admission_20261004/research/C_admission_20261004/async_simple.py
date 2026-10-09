"""Ordinary first-prefill controls for the pinned native async scheduler.

Declared accounting charges physical usage plus each live admission's unallocated
declared remainder. Deferred/finished blocks remain physical usage until native
freeing returns them; no future release, output prediction, or recovery credit.
"""
import hashlib
import inspect
from pathlib import Path
import textwrap
import time
import types

import admission_probe as native
from admission_probe import pure_fit
from async_observer import ASYNC_SHA, Observer


class Gate(Observer):
    timed = native.Gate.timed
    hold = native.Gate.hold
    restore = native.Gate.restore

    def __init__(self, scheduler, mode, cap, budget_blocks=32768):
        if mode not in ('fixed', 'declared_budget'):
            raise ValueError('mode must be fixed or declared_budget')
        if type(cap) is not int or not 1 <= cap <= 256:
            raise ValueError('cap must be an integer in 1..256')
        if type(budget_blocks) is not int or budget_blocks <= 0:
            raise ValueError('budget_blocks must be a positive integer')
        super().__init__(scheduler)
        self.mode, self.cap, self.budget_blocks = mode, cap, budget_blocks
        self.manager = scheduler.kv_cache_manager.coordinator.single_type_managers[0]
        assert self.manager.block_size == 16
        assert scheduler.max_model_len == scheduler.kv_cache_manager.max_model_len == 4096
        self.held, self.allocations = [], []
        self.barrier = self.barrier_decision_index = None
        self.scheduled_tokens = {}
        self.fit_checks = self.fit_mismatches = self.native_waiting_scan_breaks = 0
        self.observed_budget_peak_blocks = 0

    def is_new(self, req):
        return (req.status.name == 'WAITING' and req.request_id not in self.admitted
            and req.request_id not in self.started and req.num_preemptions == 0
            and req.num_computed_tokens == 0 and req.num_output_tokens == 0)

    def declared_pages(self, req):
        prompt, maximum = req.num_prompt_tokens, req.max_tokens
        if (type(prompt) is not int or prompt <= 0 or type(maximum) is not int
                or maximum <= 0 or prompt + maximum > self.s.max_model_len):
            raise ValueError('Request requires positive prompt/max_tokens with sum <= 4096')
        pages = (prompt + maximum + 15) // 16
        if pages > self.budget_blocks:
            raise ValueError('One declared request exceeds the entire physical budget')
        return pages

    def budget_state(self):
        free = self.s.kv_cache_manager.block_pool.get_num_free_blocks()
        assert 0 <= free <= self.budget_blocks
        allocated = remaining = declared = 0
        for rid in self.admitted:
            req = self.s.requests.get(rid)
            if req is None or req.is_finished():
                continue
            charge = self.declared_pages(req)
            # Exact FullAttention/APC-disabled dense tables: all entries are
            # physical pages. pop_blocks_for_free removes a deferred table before
            # pool.free_blocks; those pages remain in physical_used below.
            resident = len(self.manager.req_to_blocks.get(rid, ()))
            allocated += resident
            declared += charge
            remaining += max(0, charge - resident)
        physical = self.budget_blocks - free
        assert allocated <= physical, 'Live page tables exceed physical used pages'
        total = physical + remaining
        self.observed_budget_peak_blocks = max(self.observed_budget_peak_blocks, total)
        return dict(physical_used_blocks=physical, live_allocated_blocks=allocated,
            live_declared_blocks=declared, live_remaining_blocks=remaining,
            nonlive_physical_blocks=physical-allocated, budget_before_blocks=total,
            budget_limit_blocks=self.budget_blocks)

    def state(self, scheduled=None):
        return dict(super().state(scheduled), **self.budget_state())

    def begin(self):
        assert not self.held
        self.barrier = self.barrier_decision_index = None
        self.scheduled_tokens = {}
        super().begin()
        assert self.s.num_waiting_for_streaming_input == 0
        assert len(self.admitted) <= self.cap
        assert all(r.request_id in self.admitted for r in self.s.running)

    def early(self, req):
        if self.barrier is None or not self.is_new(req):
            return None
        # The earlier new request is still blocked. No fit, table traversal or
        # hypothetical permission is attributed to this unevaluated FIFO row.
        self.decisions.append(dict(t=time.perf_counter()-self.origin,
            request_id=req.request_id, status=req.status.name, schedule_call=self.calls-1,
            active=len(self.admitted), free_blocks=self.s.kv_cache_manager.block_pool.get_num_free_blocks(),
            denied=True, reason='fifo_'+self.barrier, fifo_held=True,
            fifo_barrier_decision_index=self.barrier_decision_index,
            native_fit=None, native_allocation_succeeded=None, native_allocation_result=None,
            baseline_allowed=None, final_allowed=False, candidate_action='hold_for_fifo',
            final_action='deferred_fifo', changed_by_recovery=False,
            changed_by_declared_budget=False, native_waiting_scan_break=False))
        return 'hold'

    def before_allocate(self, req, *, token_budget, scheduled=None, **kwargs):
        if not self.is_new(req):
            return dict(fit=None, row=None)  # Recovery never undergoes a new fit query or gate.
        assert not self.manager.req_to_blocks.get(req.request_id), 'New request already owns KV pages'
        required = self.declared_pages(req)
        fit = pure_fit(self.s.kv_cache_manager, req, **kwargs)
        row = super().before_allocate(req, self.scheduled_tokens if scheduled is None else scheduled,
                                      token_budget, kwargs['num_new_tokens'])
        assert row is not None
        cap_block = row['active'] >= self.cap
        budget_block = (self.mode == 'declared_budget'
            and row['budget_before_blocks'] + required > self.budget_blocks)
        native_allowed = fit['fits']
        baseline_allowed = native_allowed and not cap_block
        denied = native_allowed and (cap_block or budget_block)
        reason = ('native_capacity' if not native_allowed else 'cap' if cap_block else
                  'declared_budget' if budget_block else 'allow')
        row.update({k: v for k, v in fit.items() if k != 'free_blocks'})
        row.update(native_fit=native_allowed, native_allocation_result=None,
            baseline_allowed=baseline_allowed, base_allowed=baseline_allowed,
            final_allowed=baseline_allowed and not budget_block, denied=denied, reason=reason,
            baseline_action='native_allocate', candidate_action='defer' if denied else 'native_allocate',
            final_action='deferred_by_gate' if denied else None,
            declared_prompt_tokens=req.num_prompt_tokens, declared_max_tokens=req.max_tokens,
            budget_required_blocks=required,
            budget_after_if_admitted_blocks=row['budget_before_blocks']+required,
            changed_by_declared_budget=bool(baseline_allowed and budget_block),
            changed_by_complete_cap=bool(native_allowed and cap_block),
            age_s=max(0., time.time()-req.arrival_time), signal_wait_limit_bypass=False,
            fifo_held=False, native_waiting_scan_break=False)
        if denied:
            self.barrier, self.barrier_decision_index = reason, len(self.decisions)-1
        return dict(fit=fit, row=row, status_before=req.status.name, preemptions=req.num_preemptions)

    def stop_new_scan(self, check):
        row = check['row']
        if row is None or not row['denied']:
            return False
        if (self.s.num_waiting_for_streaming_input or len(self.s.running) != len(self.admitted)
                or {r.request_id for r in self.s.running} != self.admitted):
            return False
        row.update(native_waiting_scan_break=True, suffix_new_requests_not_evaluated=True)
        self.native_waiting_scan_breaks += 1
        return True

    def after_allocate(self, req, check, result):
        row = check['row']
        super().after_allocate(req, row, result)
        if row is None:
            return
        actual = result is not None
        row['native_allocation_result'] = actual
        self.fit_checks += 1
        self.allocations.append(dict(t=time.perf_counter()-self.origin, request_id=req.request_id,
            status_before=check['status_before'], preemptions=check['preemptions'],
            actual=actual, predicted=check['fit']['fits'], **check['fit']))
        if actual != check['fit']['fits']:
            self.fit_mismatches += 1
            raise AssertionError('Native first-prefill fit mismatch: '+req.request_id)

    def confirm_admitted(self, req):
        assert req.status.name in ('RUNNING', 'WAITING_FOR_REMOTE_KVS') and not req.is_finished()
        self.admitted.add(req.request_id)
        assert len(self.admitted) <= self.cap

    def report(self):
        report = super().report()
        report.update(mode=self.mode, cap=self.cap, kv_floor=0, async_scheduling=True,
            policy_intervention=True, admission_count='complete_unique_unfinished',
            budget_blocks=self.budget_blocks, block_size=16, max_signal_wait_s=None,
            probe_enabled=False, native_allocations=self.allocations,
            native_fit_checks=self.fit_checks, native_fit_mismatches=self.fit_mismatches,
            native_waiting_scan_breaks=self.native_waiting_scan_breaks,
            observed_budget_peak_blocks=self.observed_budget_peak_blocks,
            denied_request_ids=sorted({r['request_id'] for r in self.decisions if r['denied']}),
            source_sha256=dict(native.SOURCE_SHA, **{'async_scheduler.py': ASYNC_SHA}),
            semantics='Ordinary async first-prefill admission, not a recovery signal. Fixed mode bounds '
                'all uniquely admitted unfinished requests. Declared-budget mode charges physical used '
                'pages plus max(0, full declared pages - resident pages) for each live admission, plus '
                'the new request full declaration. Finished/deferred pages remain physical usage; live '
                'resident pages are not added twice. No future output, release credit or age bypass. '
                'Only new requests are queried with pure_fit; native failed allocations retain their '
                'original break, and recovery allocation/updaters are unchanged. FIFO rows and safely '
                'skipped suffixes have no observed native permission. A denied row is an actual delay, '
                'not another policy service counterfactual. Starts are host schedule-return boundaries, '
                'not GPU completion; all observation and control cost remains in service time.')
        return report


def qualify(s, budget_blocks):
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.core.sched.async_scheduler import AsyncScheduler
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.kv_cache_coordinator import KVCacheCoordinatorNoPrefixCache
    from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
    from vllm.v1.kv_cache_interface import FullAttentionSpec
    assert type(s) is AsyncScheduler and AsyncScheduler.schedule is Scheduler.schedule
    k = s.kv_cache_manager
    assert type(k) is KVCacheManager and type(k.coordinator) is KVCacheCoordinatorNoPrefixCache
    assert k.num_kv_cache_groups == len(k.coordinator.single_type_managers) == 1
    m = k.coordinator.single_type_managers[0]
    assert type(m) is FullAttentionManager and type(m.kv_cache_spec) is FullAttentionSpec
    classes = ((Scheduler, 'scheduler.py', native.SOURCE_SHA['scheduler.py']),
        (AsyncScheduler, 'async_scheduler.py', ASYNC_SHA),
        (type(k), 'kv_cache_manager.py', native.SOURCE_SHA['kv_cache_manager.py']),
        (type(k.coordinator), 'kv_cache_coordinator.py', native.SOURCE_SHA['kv_cache_coordinator.py']),
        (type(m), 'single_type_kv_cache_manager.py', native.SOURCE_SHA['single_type_kv_cache_manager.py']))
    for cls, filename, digest in classes:
        assert hashlib.sha256(Path(inspect.getsourcefile(cls)).read_bytes()).hexdigest() == digest, filename
    assert not k.enable_caching and not k.coordinator.enable_caching and not m.enable_caching
    assert not k.use_eagle and not k.coordinator.eagle_group_ids
    assert m.dcp_world_size == m.pcp_world_size == 1 and m._max_admission_blocks_per_request is None
    assert k.block_pool is k.coordinator.block_pool is m.block_pool
    assert s.scheduler_config.async_scheduling is True and s.scheduler_reserve_full_isl
    assert s.policy.name == 'FCFS' and s.max_num_running_reqs == 256
    assert s.parallel_config.tensor_parallel_size == s.parallel_config.pipeline_parallel_size == s.parallel_config.data_parallel_size == 1
    assert s.max_model_len == k.max_model_len == 4096 and m.block_size == 16
    assert s.vllm_config.model_config.hf_config.model_type == 'olmoe'
    assert str(s.vllm_config.model_config.dtype) == 'torch.bfloat16'
    assert not s.num_spec_tokens and not s.num_lookahead_tokens and not s.use_eagle
    assert not s.has_mamba_layers and not s.need_mamba_block_aligned_split and not s.is_encoder_decoder
    assert s.lora_config is None and s.ec_connector is None
    assert not s.num_waiting_for_streaming_input
    assert not s.has_requests() and not s.deferred_frees
    assert not any(not r.is_finished() for r in s.requests.values())
    assert k.block_pool.get_num_free_blocks() == budget_blocks
    return Scheduler


def patch_source(source):
    source = native.patch_source(source)
    source = source.replace('self._c_probe.admitted, request', 'self._c_probe.confirm_admitted, request')
    anchor = "            if _probe_check['row'] is not None and _probe_check['row']['denied']:\n"
    assert source.count(anchor) == 1
    return source.replace(anchor, anchor+
        '                if self._c_probe.timed(self._c_probe.stop_new_scan, _probe_check):\n'
        '                    break\n')


def install(scheduler, mode, cap, budget_blocks=32768):
    Scheduler = qualify(scheduler, budget_blocks)
    assert 'schedule' not in vars(scheduler) and '_c_probe' not in vars(scheduler)
    assert '_c_async_observer' not in vars(scheduler)
    source = patch_source(textwrap.dedent(inspect.getsource(Scheduler.schedule)))
    namespace = dict(Scheduler.schedule.__globals__)
    exec(compile(source, '<C_simple_async_admission>', 'exec'), namespace)
    patched = types.MethodType(namespace['schedule'], scheduler)
    gate = scheduler._c_probe = Gate(scheduler, mode, cap, budget_blocks)

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
