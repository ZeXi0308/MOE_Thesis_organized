"""Guarded async adaptation of MC-Benchmark's existing future-round peak.

This ordinary component grants no selective execution credit. Every live old
request must be a resident one-token decode with consistent native batch state.
The unchanged peak routine delays old releases by two nonempty schedule steps;
its continued virtual growth during that delay is deliberately conservative.
"""
import hashlib
import inspect
from pathlib import Path
import textwrap
import time
import types

import async_simple
from mc_budget import EXTRA_SOURCE_SHA, Gate as SynchronousMCGate, peak_envelope


ENGINE_CORE_SHA = '3ae1381a6af841e21058c825702382dc66faae45c950ac5acb8495d2d3d05aad'
RELEASE_DELAY = 2


class Gate(async_simple.Gate):
    def __init__(self, scheduler, engine_core, cap=256, budget_blocks=32768):
        super().__init__(scheduler, 'declared_budget', cap, budget_blocks)
        if cap != 256:
            raise ValueError('This async MC adaptation requires complete cap=256')
        self.core = engine_core
        self.mc_authorizations = {}
        self.mc = dict(attempted_evaluations=0, eligible_evaluations=0,
            guard_fallback_counts={}, peak_evaluations=0, peak_rejections=0,
            requested_relaxations=0, actual_allocations=0, successful_relaxations=0,
            first_eligible=None)
        self.queue_observation = None

    def _pending_batch(self):
        core, s = self.core, self.s
        self.queue_observation = None
        if (core.scheduler is not s or core.batch_queue_size != 2
                or core.async_scheduling is not True or core.batch_queue is None):
            return 'batch_queue_configuration', None
        queue = list(core.batch_queue)
        # The current schedule has not yet been enqueued. Native depth two
        # therefore permits at most one older tuple at this admission hook.
        if len(queue) > 1:
            return 'batch_queue_depth', None
        pending, nonempty, metadata = {}, 0, []
        for item in queue:
            if not isinstance(item, (tuple, list)) or len(item) != 3:
                return 'batch_queue_entry', None
            output = item[1]
            counts, total = output.num_scheduled_tokens, output.total_num_scheduled_tokens
            if (type(total) is not int or total < 0 or not isinstance(counts, dict)
                    or any(type(n) is not int or n <= 0 for n in counts.values())
                    or sum(counts.values()) != total):
                return 'batch_queue_token_counts', None
            if total:
                nonempty += 1
                pending.update(counts)
            metadata.append(dict(total_num_scheduled_tokens=total,
                                 num_scheduled_tokens=dict(counts)))
        if (type(s.sched_step_seq) is not int or type(s.processed_step_seq) is not int
                or s.processed_step_seq < 0
                or s.sched_step_seq-s.processed_step_seq != nonempty):
            return 'batch_fence_sequence', None
        self.queue_observation = dict(batch_queue_size=2, queued_batches=len(queue),
            pending_nonempty_batches=nonempty, scheduled_step=s.sched_step_seq,
            processed_step=s.processed_step_seq, pending_batches=metadata)
        return None, pending

    def _old_rows(self, req, row, *, scheduled=None):
        s = self.s
        if s.max_num_scheduled_tokens != 1024 or s.num_waiting_for_streaming_input:
            return 'quantum_or_streaming', [], None
        failure = SynchronousMCGate._request_guard(req)
        if failure:
            return failure, [], None
        if (len(s.running) != len(self.admitted)
                or {r.request_id for r in s.running} != self.admitted):
            return 'admitted_not_all_running', [], None
        failure, pending = self._pending_batch()
        if failure:
            return failure, [], None
        scheduled = self.scheduled_tokens if scheduled is None else scheduled
        rows, physical = [], 0
        for old in s.running:
            failure = SynchronousMCGate._request_guard(old)
            if failure:
                return failure, rows, None
            output, prompt = old.num_output_tokens, old.num_prompt_tokens
            n, computed = old.num_tokens, old.num_computed_tokens
            q, inflight = old.num_output_placeholders, old.num_in_flight_tokens
            current = scheduled.get(old.request_id, 0)
            previous = pending.get(old.request_id, 0)
            remaining = old.max_tokens-output
            if (old.status.name != 'RUNNING' or old.is_finished() or remaining <= 0
                    or n != prompt+output or computed < prompt or old.is_prefill_chunk):
                return 'not_complete_resident_decode', rows, None
            if (old.spec_token_ids or old.async_tokens_to_discard
                    or old.num_tokens_with_spec != n
                    or old.next_decode_eligible_step > s.current_step or current != 1):
                return 'not_plain_one_token_decode', rows, None
            if q > 0 and computed+2-q >= prompt+old.max_tokens:
                return 'terminal_inflight', rows, None
            # A queued non-final/full prefill is not given a guessed output
            # credit. The conservative first implementation accepts only a
            # queued one-token frame, or no frame for this old request.
            if previous not in (0, 1):
                return 'pending_frame_not_one_token', rows, None
            if q != previous or inflight != previous or computed != n+q-1:
                return 'placeholder_inflight_relation', rows, None
            if ((previous and old.last_sched_seq != s.sched_step_seq)
                    or (not previous and old.last_sched_seq > s.processed_step_seq)):
                return 'request_fence_sequence', rows, None
            a = computed+current  # Current running allocation already happened.
            pages = len(self.manager.req_to_blocks.get(old.request_id, ()))
            if pages != (a+15)//16:
                return 'physical_page_geometry', rows, None
            physical += pages
            rows.append(dict(request_id=old.request_id, n=a, r=remaining+RELEASE_DELAY,
                logical_n=n, remaining_declared_outputs=remaining,
                prompt=prompt, output=output, max_tokens=old.max_tokens,
                computed=computed, scheduled_tokens=current, placeholders=q,
                inflight=inflight, pending_scheduled_tokens=previous,
                last_sched_seq=old.last_sched_seq, allocated_blocks=pages,
                status=old.status.name, preemptions=old.num_preemptions))
        free = s.kv_cache_manager.block_pool.get_num_free_blocks()
        if free != row['free_blocks']:
            return 'physical_free_changed', rows, None
        nonlive = self.budget_blocks-free-physical
        if (nonlive < 0 or physical != row['live_allocated_blocks']
                or nonlive != row['nonlive_physical_blocks']):
            return 'physical_accounting', rows, None
        return None, rows, nonlive

    def before_allocate(self, req, *, token_budget, scheduled=None, **kwargs):
        check = super().before_allocate(req, token_budget=token_budget, scheduled=scheduled, **kwargs)
        row = check['row']
        if row is None:
            return check
        row.update(ordinary_budget_allowed=row['final_allowed'], mc_attempted=False,
            mc_eligible=None, mc_guard_reason=None, mc_peak_blocks=None,
            mc_envelope_peak_blocks=None, mc_nonlive_physical_blocks=None, mc_peak_h=None,
            changed_by_mc_budget=False, mc_admission_confirmed=False,
            release_delay_nonempty_steps=RELEASE_DELAY)
        if not (row['reason'] == 'declared_budget' and row['baseline_allowed']):
            return check
        row['mc_attempted'] = True
        self.mc['attempted_evaluations'] += 1
        if (kwargs['num_new_tokens'] <= 0 or kwargs.get('delay_cache_blocks', False)
                or kwargs.get('num_external_computed_tokens', 0)):
            failure, rows, nonlive = 'not_new_local_prefill', [], None
        else:
            failure, rows, nonlive = self._old_rows(req, row, scheduled=scheduled)
        row.update(mc_eligible=failure is None, mc_guard_reason=failure,
                   mc_nonlive_physical_blocks=nonlive)
        if failure:
            counts = self.mc['guard_fallback_counts']
            counts[failure] = counts.get(failure, 0)+1
            return check
        self.mc['eligible_evaluations'] += 1
        envelope = peak_envelope(rows, req.num_prompt_tokens, req.max_tokens)
        self.mc['peak_evaluations'] += 1
        peak = nonlive+envelope['peak_blocks']
        row.update(mc_peak_blocks=peak, mc_envelope_peak_blocks=envelope['peak_blocks'],
                   mc_peak_h=envelope['peak_h'])
        if self.mc['first_eligible'] is None:
            self.mc['first_eligible'] = dict(decision_index=len(self.decisions)-1,
                request_id=req.request_id, t=row['t'], old_rows=rows,
                free_blocks=row['free_blocks'], nonlive_physical_blocks=nonlive,
                queue=dict(self.queue_observation), current_step=self.s.current_step,
                new_prompt_tokens=req.num_prompt_tokens, new_max_tokens=req.max_tokens,
                token_budget=token_budget, num_new_tokens=kwargs['num_new_tokens'],
                envelope=envelope, peak_blocks=peak, release_delay_nonempty_steps=RELEASE_DELAY)
        if peak > self.budget_blocks:
            self.mc['peak_rejections'] += 1
            return check
        self.mc['requested_relaxations'] += 1
        row.update(denied=False, reason='async_mc_budget_allow', final_allowed=True,
            candidate_action='native_allocate', final_action=None,
            changed_by_declared_budget=False, changed_by_mc_budget=True)
        # Undo only this head request's ordinary-budget barrier. No restoration
        # changes to old requests, running order, native victims or async updater.
        assert self.barrier == 'declared_budget' and self.barrier_decision_index == len(self.decisions)-1
        self.barrier = self.barrier_decision_index = None
        return check

    def after_allocate(self, req, check, result):
        super().after_allocate(req, check, result)
        row = check['row']
        if result is not None and row is not None and row['changed_by_mc_budget']:
            self.mc['actual_allocations'] += 1
            self.mc_authorizations[req.request_id] = row

    def confirm_admitted(self, req):
        super().confirm_admitted(req)
        authorization = self.mc_authorizations.pop(req.request_id, None)
        if authorization is not None:
            assert authorization['native_allocation_succeeded'] is True
            authorization['mc_admission_confirmed'] = True
            self.mc['successful_relaxations'] += 1

    def report(self):
        report = super().report()
        report.update(mode='async_mc', budget_overrides=self.mc['successful_relaxations'],
            mc_budget=dict(self.mc, release_delay_nonempty_steps=RELEASE_DELAY,
                source_sha256=dict(EXTRA_SOURCE_SHA, **{'engine/core.py': ENGINE_CORE_SHA}),
                prior_work='MC-Benchmark arXiv:2502.07115v5; ordinary guarded async component adaptation',
                semantics='Only an ordinary physical-plus-unallocated-declaration denial may be relaxed. '
                    'baseline_allowed means native fit plus complete cap only; ordinary_budget_allowed '
                    'is the full-budget decision. Successful borrowing may exceed the full declared '
                    'platform; capacity is checked by the conservative peak plus constant nonlive pages. '
                    'Every admitted old request must be resident RUNNING and scheduled for one decode; '
                    'real queued batches, placeholders, in-flight counts, fence indices and dense pages '
                    'must agree. No placeholder is subtracted from remaining declared output. Existing '
                    'peak_envelope uses n=computed+current scheduled and r=max-output+2. Continuing '
                    'virtual growth during the two-step release delay is conservative. Nonlive physical '
                    'pages stay constant, and the new request plateau is never released. h counts '
                    'nonempty schedule steps, not empty host calls or time. The release argument is '
                    'conditional on pinned depth-two FIFO execution and the checked plain-generation '
                    'domain; it is neither a wall-time bound nor evidence of service benefit. All-old '
                    'qualification failure retains ordinary FIFO denial. Requested, allocated and '
                    'confirmed admissions are separate counts; authorization is matched by request ID.'))
        report['semantics'] = report['mc_budget']['semantics']
        return report


def qualify(scheduler, engine_core, cap, budget_blocks):
    Scheduler = async_simple.qualify(scheduler, budget_blocks)
    from vllm.v1.engine.core import EngineCore
    from vllm.v1.request import Request
    from vllm.v1.core.sched.utils import check_stop
    assert type(engine_core) is EngineCore and engine_core.scheduler is scheduler
    assert hashlib.sha256(Path(inspect.getsourcefile(EngineCore)).read_bytes()).hexdigest() == ENGINE_CORE_SHA
    assert engine_core.step_fn.__func__ is EngineCore.step_with_batch_queue
    assert engine_core.async_scheduling is True and engine_core.batch_queue_size == 2
    assert engine_core.batch_queue is not None and engine_core.batch_queue.maxlen == 2
    assert not engine_core.batch_queue and engine_core.is_ec_consumer and not engine_core.is_pooling_model
    assert cap == 256 and scheduler.max_num_scheduled_tokens == 1024
    assert scheduler.vllm_config.max_concurrent_batches == 2
    assert scheduler.defer_block_free and not scheduler.use_pp
    assert scheduler.sched_step_seq == scheduler.processed_step_seq
    connector_scheduler = scheduler.connector.connector_scheduler
    for obj, name in ((Request, 'request.py'), (check_stop, 'utils.py'),
                      (type(connector_scheduler), 'offloading/scheduler.py')):
        assert hashlib.sha256(Path(inspect.getsourcefile(obj)).read_bytes()).hexdigest() == EXTRA_SOURCE_SHA[name]
    return Scheduler


def install(scheduler, engine_core, cap=256, budget_blocks=32768):
    Scheduler = qualify(scheduler, engine_core, cap, budget_blocks)
    assert 'schedule' not in vars(scheduler) and '_c_probe' not in vars(scheduler)
    assert '_c_async_observer' not in vars(scheduler)
    source = async_simple.patch_source(textwrap.dedent(inspect.getsource(Scheduler.schedule)))
    namespace = dict(Scheduler.schedule.__globals__)
    exec(compile(source, '<C_MC_async_admission>', 'exec'), namespace)
    patched = types.MethodType(namespace['schedule'], scheduler)
    gate = scheduler._c_probe = Gate(scheduler, engine_core, cap, budget_blocks)

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
