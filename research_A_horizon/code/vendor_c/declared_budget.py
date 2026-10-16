"""Ordinary admission by declared prompt-plus-output capacity, not a recovery signal.

One virtual charge covers an admitted request's entire declared bound. Physical
usage is checked independently by native allocation and is never added to this
virtual total. The pinned Request API exposes num_prompt_tokens and max_tokens
(scheduler.py uses both at its output-limit check); no realized output is read
to determine a charge.
"""
import inspect
import textwrap
import time
import types

import admission_probe as native


class Gate(native.Gate):
    def __init__(self, scheduler, cap=256, budget_blocks=32768):
        if type(budget_blocks) is not int or budget_blocks <= 0:
            raise ValueError('budget_blocks must be a positive integer')
        assert scheduler.max_model_len == scheduler.kv_cache_manager.max_model_len == 4096
        assert scheduler.kv_cache_manager.coordinator.single_type_managers[0].block_size == 16
        super().__init__(scheduler, mode='kv', cap=cap, kv_floor=0, probe_enabled=False)
        self.mode = 'declared_budget'
        self.budget_blocks, self.budget_used_blocks = budget_blocks, 0
        self.budget_peak_blocks = 0
        self.charges, self.budget_events = {}, []
        self.native_waiting_scan_breaks = 0
        self.barrier_decision_index = None

    def declared_pages(self, req):
        prompt, maximum = req.num_prompt_tokens, req.max_tokens
        if type(prompt) is not int or prompt <= 0 or type(maximum) is not int or maximum <= 0:
            raise ValueError('Request must have positive declared prompt and max_tokens bounds')
        if prompt + maximum > self.s.max_model_len:
            raise ValueError('Declared prompt plus max_tokens exceeds max_model_len=4096')
        pages = (prompt + maximum + 15) // 16
        if pages > self.budget_blocks:
            raise ValueError('One request declared bound exceeds the entire admission budget')
        return pages

    def _release_finished(self):
        for rid in list(self.charges):
            req = self.s.requests.get(rid)
            if req is not None and not req.is_finished():
                continue
            charge = self.charges.pop(rid)
            before = self.budget_used_blocks
            self.budget_used_blocks -= charge
            self.admitted_ids.remove(rid)
            self.budget_events.append(dict(t=time.perf_counter()-self.origin, request_id=rid,
                action='release', cause='finished' if req is not None else 'removed_after_admission',
                budget_required_blocks=charge, budget_before_blocks=before,
                budget_after_blocks=self.budget_used_blocks))

    def begin(self):
        assert not self.held
        self.calls += 1
        self.barrier = self.barrier_deadline = self.barrier_decision_index = None
        self.scheduled_tokens = {}
        self._release_finished()
        assert self.s.num_waiting_for_streaming_input == 0
        assert len(self.admitted_ids) <= self.cap
        assert all(r.request_id in self.admitted_ids for r in self.s.running)
        now = time.perf_counter()
        if self.sample_at is None or now-self.sample_at >= .1:
            self.snapshots.append(dict(t=now-self.origin, **self.state()))
            self.sample_at = now

    def state(self):
        return dict(free_blocks=self.s.kv_cache_manager.block_pool.get_num_free_blocks(),
            running=len(self.s.running), active=len(self.admitted_ids), admitted_inflight=len(self.admitted_ids),
            budget_before_blocks=self.budget_used_blocks, budget_limit_blocks=self.budget_blocks,
            budget_available_blocks=self.budget_blocks-self.budget_used_blocks)

    def early(self, req):
        if self.barrier is None or not self.is_new(req):
            return None
        # A cached FIFO barrier requires no repeated fit or recovery-set scan.
        self.decisions.append(dict(t=time.perf_counter()-self.origin, request_id=req.request_id,
            **self.state(), budget_required_blocks=self.declared_pages(req),
            age_s=max(0., time.time()-req.arrival_time), signal_wait_limit_bypass=False,
            denied=True, reason='fifo_'+self.barrier, baseline_allowed=None, base_allowed=None,
            native_fit=None, native_allocation_result=None, final_allowed=False,
            changed_by_declared_budget=False, changed_by_recovery=False,
            fifo_held=True, fifo_barrier_decision_index=self.barrier_decision_index))
        return 'hold'

    def before_allocate(self, req, *, token_budget, **kwargs):
        is_new = self.is_new(req)
        required = self.declared_pages(req) if is_new else None
        fit = native.pure_fit(self.s.kv_cache_manager, req, **kwargs)
        check = dict(fit=fit, row=None, status_before=req.status.name, preemptions=req.num_preemptions)
        if not is_new:
            return check
        cap_block = len(self.admitted_ids) >= self.cap
        budget_block = self.budget_used_blocks + required > self.budget_blocks
        baseline_allowed = fit['fits'] and not cap_block
        # Native failure still executes allocate_slots -> None and its native break.
        denied = fit['fits'] and (cap_block or budget_block)
        reason = ('native_capacity' if not fit['fits'] else 'cap' if cap_block else
                  'declared_budget' if budget_block else 'allow')
        row = dict(t=time.perf_counter()-self.origin, request_id=req.request_id, **self.state(),
            **{k: v for k, v in fit.items() if k != 'free_blocks'},
            budget_required_blocks=required, budget_after_if_admitted_blocks=self.budget_used_blocks+required,
            declared_prompt_tokens=req.num_prompt_tokens, declared_max_tokens=req.max_tokens,
            age_s=max(0., time.time()-req.arrival_time), signal_wait_limit_bypass=False,
            token_budget=token_budget, native_fit=fit['fits'], native_allocation_result=None,
            baseline_allowed=baseline_allowed, base_allowed=baseline_allowed,
            final_allowed=baseline_allowed and not budget_block, denied=denied, reason=reason,
            changed_by_declared_budget=baseline_allowed and budget_block, changed_by_recovery=False,
            fifo_held=False, native_waiting_scan_break=False)
        self.decisions.append(row)
        check['row'] = row
        if denied:
            self.barrier, self.barrier_decision_index = reason, len(self.decisions)-1
        return check

    def stop_new_scan(self, check):
        """Skip only a suffix that cannot contain any admitted non-running work."""
        if check['row'] is None or not check['row']['denied']:
            return False
        if (self.s.num_waiting_for_streaming_input != 0
                or len(self.s.running) != len(self.admitted_ids)
                or any(r.request_id not in self.admitted_ids for r in self.s.running)):
            return False
        check['row'].update(native_waiting_scan_break=True, suffix_new_requests_not_evaluated=True)
        self.native_waiting_scan_breaks += 1
        return True

    def admitted(self, req):
        super().admitted(req)
        if req.request_id in self.charges:
            return
        required = self.declared_pages(req)
        before = self.budget_used_blocks
        assert before + required <= self.budget_blocks
        self.charges[req.request_id] = required
        self.budget_used_blocks += required
        self.budget_peak_blocks = max(self.budget_peak_blocks, self.budget_used_blocks)
        self.budget_events.append(dict(t=time.perf_counter()-self.origin, request_id=req.request_id,
            action='charge', budget_required_blocks=required, budget_before_blocks=before,
            budget_after_blocks=self.budget_used_blocks))

    def report(self):
        self.timed(self._release_finished)
        result = super().report()
        result.update(max_signal_wait_s=None, admission_count='complete_unique_unfinished',
            budget_blocks=self.budget_blocks, budget_used_blocks=self.budget_used_blocks,
            budget_peak_blocks=self.budget_peak_blocks, block_size=16, budget_overrides=0,
            delay_s=None, max_extra_s=None, release_mode=None,
            budget_live_charges=dict(self.charges), budget_events=self.budget_events,
            native_waiting_scan_breaks=self.native_waiting_scan_breaks,
            denied_request_ids=sorted({row['request_id'] for row in self.decisions if row['denied']}),
            semantics='Ordinary fixed declared-budget admission: ceil((num_prompt_tokens+max_tokens)/16) '
                'per uniquely admitted unfinished request; preemption/remote waiting retains its charge. '
                'Only finish or native removal releases it. No future output length, recovery signal, '
                'or addition of physical used pages to this virtual total. Native allocation and cap '
                'remain independent checks; native failures retain their break. No age bypass or '
                'promised wait bound; the common runtime limit accounts for unfinished requests. '
                'FIFO rows have unknown native fit/baseline permission. Safe scan breaks require all '
                'admitted requests running and no streaming; skipped suffixes are not observed opportunities. '
                'Starts are host schedule-return boundaries, not GPU completion.')
        return result


def install(scheduler, cap=256, budget_blocks=32768, *, gate_factory=Gate):
    native.qualify(scheduler)
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == budget_blocks
    assert 'schedule' not in vars(scheduler) and '_c_probe' not in vars(scheduler)
    module = inspect.getmodule(type(scheduler))
    source = native.patch_source(textwrap.dedent(inspect.getsource(type(scheduler).schedule)))
    anchor = "            if _probe_check['row'] is not None and _probe_check['row']['denied']:\n"
    assert source.count(anchor) == 1
    source = source.replace(anchor, anchor+
        '                if self._c_probe.timed(self._c_probe.stop_new_scan, _probe_check):\n'
        '                    break\n')
    namespace = dict(vars(module))
    exec(compile(source, '<C_declared_budget>', 'exec'), namespace)
    patched = types.MethodType(namespace['schedule'], scheduler)
    gate = gate_factory(scheduler, cap, budget_blocks)
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
