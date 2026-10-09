"""Passive native async admission observations; no MC or new admission policy.

Read state immediately before the real waiting allocation and record its result.
Placeholders / terminal in-flight work are distinct from non-running recovery.
The installed scheduler and its native async update methods remain authoritative.
"""
from collections import Counter
import hashlib
import inspect
from pathlib import Path
import textwrap
import time
import types

from admission_probe import SOURCE_SHA

ASYNC_SHA = 'da6343d7e7c394a1738cf72905cbecc208003ffa461ccb441268333a3eb9f884'


def progress_class(req, scheduled, current_step):
    if req.status.name != 'RUNNING':
        return req.status.name
    if scheduled:
        return 'scheduled_prefill' if req.num_computed_tokens < req.num_prompt_tokens else 'scheduled_decode'
    if (req.num_output_placeholders > 0 and req.num_computed_tokens + 2 -
            req.num_output_placeholders >= req.num_prompt_tokens + req.max_tokens):
        return 'terminal_inflight'  # Native max-token guard; not recovery debt.
    if current_step < req.next_decode_eligible_step:
        return 'decode_ineligible'
    if req.num_tokens_with_spec + req.num_output_placeholders <= req.num_computed_tokens:
        return 'no_new_work_needed'
    return 'unscheduled_work'  # Observation, not an inferred reason or lost GPU time.


class Observer:
    def __init__(self, scheduler):
        self.s = scheduler
        self.origin = time.perf_counter()
        self.started, self.admitted = set(), set()
        self.starts, self.decisions, self.snapshots = [], [], []
        self.calls = self.cpu_ns = self.wall_ns = 0
        self.sample_at = None

    def timed(self, fn, *args):
        wall, cpu = time.perf_counter_ns(), time.process_time_ns()
        try:
            return fn(*args)
        finally:
            self.cpu_ns += time.process_time_ns()-cpu
            self.wall_ns += time.perf_counter_ns()-wall

    def state(self, scheduled=None):
        s = self.s
        old = [s.requests[rid] for rid in self.admitted
               if rid in s.requests and not s.requests[rid].is_finished()]
        statuses = Counter(r.status.name for r in old)
        row = dict(t=time.perf_counter()-self.origin,
            free_blocks=s.kv_cache_manager.block_pool.get_num_free_blocks(),
            active=len(old), running=statuses['RUNNING'], old_status_counts=dict(statuses),
            nonrunning=len(old)-statuses['RUNNING'],
            placeholders=sum(r.num_output_placeholders > 0 for r in old),
            async_discard=sum(r.async_tokens_to_discard > 0 for r in old),
            deferred_free_entries=len(s.deferred_frees),
            deferred_free_blocks=sum(len(blocks) for _, blocks in s.deferred_frees),
            scheduled_step=s.sched_step_seq, processed_step=s.processed_step_seq,
            current_step=s.current_step)
        if scheduled is not None:
            row['old_progress_counts'] = dict(Counter(progress_class(r,
                scheduled.get(r.request_id, 0), s.current_step) for r in old))
            row['mixed_running_nonrunning'] = bool(row['nonrunning'] and row['running'])
        return row

    def begin(self):
        self.calls += 1
        self.admitted = {rid for rid in self.admitted
            if rid in self.s.requests and not self.s.requests[rid].is_finished()}
        now = time.perf_counter()
        if self.sample_at is None or now-self.sample_at >= .1:
            self.snapshots.append(self.state())
            self.sample_at = now

    def before_allocate(self, request, scheduled, token_budget, num_new_tokens):
        if (request.request_id in self.started or request.request_id in self.admitted or
                request.num_preemptions or request.num_computed_tokens or request.num_output_tokens):
            return None  # Already admitted recovery is never a new admission.
        row = dict(self.state(scheduled), request_id=request.request_id,
            status=request.status.name, token_budget=token_budget, num_new_tokens=num_new_tokens,
            native_allocation_succeeded=None, denied=False, changed_by_recovery=False,
            candidate_action=None, baseline_action='native_allocate',
            reason='passive_native_async_observation', schedule_call=self.calls-1)
        self.decisions.append(row)
        return row

    def after_allocate(self, request, row, new_blocks):
        if new_blocks is not None:
            self.admitted.add(request.request_id)
        if row is not None:
            row['native_allocation_succeeded'] = new_blocks is not None
            row['final_action'] = 'allocated' if new_blocks is not None else 'native_capacity_refusal'
            row['free_blocks_after_allocation'] = self.s.kv_cache_manager.block_pool.get_num_free_blocks()

    def end(self, output):
        now = time.perf_counter()
        for rid in output.num_scheduled_tokens:
            self.admitted.add(rid)
            if rid not in self.started:
                self.starts.append(dict(request_id=rid, first_prefill_perf_s=now))
                self.started.add(rid)

    def report(self):
        return dict(mode='passive_native_async', cap=None, native_running_cap=256, kv_floor=None,
            policy_intervention=False, origin_perf_s=self.origin, calls=self.calls,
            controller_cpu_s=self.cpu_ns/1e9, controller_wall_s=self.wall_ns/1e9,
            starts=self.starts, decisions=self.decisions, snapshots=self.snapshots,
            source_sha256={'scheduler.py': SOURCE_SHA['scheduler.py'], 'async_scheduler.py': ASYNC_SHA},
            semantics='No added gate or hypothetical MC result. Native max-running256 is not a complete '
                'inflight cap. States precede actual waiting allocation; outcomes follow that allocation. '
                'Scheduled means work placed in this native schedule, not completed GPU execution. '
                'Placeholders and terminal-inflight work do not imply execution failure. Snapshots are '
                'sampled at100ms; extrema are observed, not continuous. All observer cost stays in service time.')


def patch_source(source):
    before = '            new_blocks = self.kv_cache_manager.allocate_slots(\n                request,\n                num_new_tokens,\n                num_new_computed_tokens='
    after = '            if new_blocks is None:\n                # The request cannot be scheduled.\n'
    assert source.count(before) == source.count(after) == 1
    source = source.replace(before,
        '            _c_async_row = self._c_async_observer.timed(\n'
        '                self._c_async_observer.before_allocate, request, num_scheduled_tokens,\n'
        '                token_budget, num_new_tokens)\n'+before)
    return source.replace(after,
        '            self._c_async_observer.timed(self._c_async_observer.after_allocate,\n'
        '                request, _c_async_row, new_blocks)\n'+after)


def install(scheduler):
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.core.sched.async_scheduler import AsyncScheduler
    assert type(scheduler) is AsyncScheduler and AsyncScheduler.schedule is Scheduler.schedule
    for cls, digest in ((Scheduler, SOURCE_SHA['scheduler.py']), (AsyncScheduler, ASYNC_SHA)):
        assert hashlib.sha256(Path(inspect.getsourcefile(cls)).read_bytes()).hexdigest() == digest
    assert scheduler.scheduler_config.async_scheduling is True
    assert scheduler.parallel_config.pipeline_parallel_size == 1
    assert scheduler.max_num_running_reqs == 256 and not scheduler.num_spec_tokens
    assert not scheduler.has_requests() and not scheduler.deferred_frees
    assert 'schedule' not in vars(scheduler) and '_c_async_observer' not in vars(scheduler)
    source = patch_source(textwrap.dedent(inspect.getsource(Scheduler.schedule)))
    namespace = dict(Scheduler.schedule.__globals__)  # Inherited method uses base module globals.
    exec(compile(source, '<C_passive_async_admission>', 'exec'), namespace)
    patched = types.MethodType(namespace['schedule'], scheduler)
    observer = scheduler._c_async_observer = Observer(scheduler)

    def schedule(*args, **kwargs):
        observer.timed(observer.begin)
        result = patched(*args, **kwargs)
        observer.timed(observer.end, result)
        return result

    scheduler.schedule = schedule

    def uninstall():
        del scheduler.schedule
        del scheduler._c_async_observer
        return observer.report()

    return observer, uninstall
