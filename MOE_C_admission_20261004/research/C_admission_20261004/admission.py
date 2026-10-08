"""New-prefill-only admission on the pinned native scheduler. No recovery edits."""
import hashlib
import inspect
import textwrap
import time
import types
from pathlib import Path

SCHEDULER_SHA = '2ed2a550b6558b2495eda845a97ae38bcf0225027b9e25fbf00fc3880c1d3941'


class Gate:
    def __init__(self, scheduler, mode, cap, kv_floor=410, max_hold_s=10):
        assert mode in ('native', 'fixed', 'kv', 'recovery')
        self.s = scheduler
        self.mode, self.cap = mode, cap
        self.kv_floor, self.max_hold_s = kv_floor, max_hold_s
        self.held = []
        self.snapshots, self.decisions, self.starts = [], [], []
        self.recovery_since = {}
        self.previous_ids = set()
        self.sample_at = None
        self.latch = False
        self.cpu_ns = self.wall_ns = self.calls = 0
        self.origin = time.perf_counter()
        self.started_ids = set()

    def state(self):
        free = self.s.kv_cache_manager.block_pool.get_num_free_blocks()
        running = len(self.s.running)
        # Pinned synchronous scheduler tracks all async loads in this smaller set.
        # Keep the status filter and ignore stale/deleted request identities.
        active = running + sum(r.status.name == 'WAITING_FOR_REMOTE_KVS'
                               and self.s.requests.get(r.request_id) is r
                               for r in self.s._inflight_prefills)
        return dict(free_blocks=free, running=running, active=active,
                    recovery_count=len(self.recovery_since), latch=self.latch)

    def begin(self):
        assert not self.held
        now = time.perf_counter()
        recovery = {r.request_id for r in self.s.requests.values()
                    if r.num_preemptions > 0 and r.status.name in
                    ('PREEMPTED', 'WAITING_FOR_REMOTE_KVS', 'WAITING')}
        self.recovery_since = {rid: self.recovery_since.get(rid, now) for rid in recovery}
        if self.sample_at is None or now - self.sample_at >= .1:
            entered, departed = recovery-self.previous_ids, self.previous_ids-recovery
            if len(recovery) >= 2 and len(entered) > len(departed):
                self.latch = True
            if not recovery:
                self.latch = False
            snap = dict(t=now-self.origin, **self.state(), entered=len(entered), departed=len(departed),
                        recovery_ids=sorted(recovery),
                        oldest_recovery_wait_s=max((now-v for v in self.recovery_since.values()), default=0))
            self.snapshots.append(snap)
            self.previous_ids, self.sample_at = recovery, now
        self.calls += 1

    def defer(self, req):
        # A request that has started computation, output or recovery never waits here.
        if (req.status.name != 'WAITING' or req.num_preemptions > 0
                or req.num_computed_tokens > 0 or req.num_output_tokens > 0
                or req.request_id in self.started_ids):
            return False
        st = self.state()
        age = max(0., time.time()-req.arrival_time)
        expired = age >= self.max_hold_s
        cap_block = self.mode != 'native' and st['active'] >= self.cap
        kv_block = self.mode in ('kv', 'recovery') and st['free_blocks'] < self.kv_floor and not expired
        recovery_block = self.mode == 'recovery' and self.latch and not expired
        base_block = cap_block or kv_block
        denied = base_block or recovery_block
        self.decisions.append(dict(t=time.perf_counter()-self.origin, request_id=req.request_id,
            **st, age_s=age, signal_wait_limit_bypass=expired,
            denied=denied, kv_only_denied=base_block,
            changed_by_recovery=bool(recovery_block and not base_block),
            reason='cap' if cap_block else 'kv' if kv_block else 'recovery' if recovery_block else 'allow'))
        return denied

    def hold(self, queue, req):
        w, c = time.perf_counter_ns(), time.process_time_ns()
        self.held.append((queue, req))
        self.wall_ns += time.perf_counter_ns()-w
        self.cpu_ns += time.process_time_ns()-c

    def restore(self):
        for queue, req in reversed(self.held):
            queue.prepend_request(req)
        self.held.clear()

    def end(self, result):
        now = time.perf_counter()
        for rid in result.num_scheduled_tokens:
            if rid not in self.started_ids:
                self.starts.append(dict(request_id=rid, first_prefill_perf_s=now,
                                        first_prefill_unix_s=time.time()))
                self.started_ids.add(rid)

    def report(self):
        return dict(mode=self.mode, cap=self.cap, kv_floor=self.kv_floor,
                    origin_perf_s=self.origin,
                    max_signal_wait_s=self.max_hold_s, calls=self.calls,
                    controller_cpu_s=self.cpu_ns/1e9, controller_wall_s=self.wall_ns/1e9,
                    snapshots=self.snapshots, decisions=self.decisions, starts=self.starts,
                    semantics='Only never-started WAITING may be held; original queues restored after every native schedule. '
                    'Recovery signal sampled each 100ms: latch when count>=2 and entries>departures; clear at zero. '
                    'After 10s from external arrival bypass KV/recovery signals, retain fixed cap. '
                    'CPU and wall include observations/logging and exclude native schedule work.')


def install(scheduler, mode, cap, kv_floor=410, max_hold_s=10):
    original = scheduler.schedule
    module = inspect.getmodule(type(scheduler))
    source_file = inspect.getsourcefile(type(scheduler))
    assert hashlib.sha256(Path(source_file).read_bytes()).hexdigest() == SCHEDULER_SHA
    assert 'schedule' not in vars(scheduler), 'Require unmodified native scheduler'
    source = textwrap.dedent(inspect.getsource(type(scheduler).schedule))
    anchor = '            request = request_queue.peek_request()\n            request_id = request.request_id\n'
    assert source.count(anchor) == 1
    addition = ('            if self._c_gate.defer(request):\n'
                '                self._c_gate.hold(request_queue, request_queue.pop_request())\n'
                '                continue\n')
    # Dedenting removes four spaces; the waiting-loop anchor is exact and version-pinned.
    source = source.replace(anchor, anchor+addition)
    ns = dict(vars(module))
    exec(compile(source, '<C_new_prefill_only>', 'exec'), ns)
    patched = types.MethodType(ns['schedule'], scheduler)
    gate = Gate(scheduler, mode, cap, kv_floor, max_hold_s)
    scheduler._c_gate = gate

    def schedule(*args, **kwargs):
        w, c = time.perf_counter_ns(), time.process_time_ns()
        gate.begin()
        gate.wall_ns += time.perf_counter_ns()-w
        gate.cpu_ns += time.process_time_ns()-c
        # Only defer() and hold() work inside native scheduling is charged separately.
        try:
            result = patched(*args, **kwargs)
            return result
        finally:
            w, c = time.perf_counter_ns(), time.process_time_ns()
            gate.restore()
            if 'result' in locals():
                gate.end(result)
            gate.wall_ns += time.perf_counter_ns()-w
            gate.cpu_ns += time.process_time_ns()-c

    original_defer = gate.defer
    def timed_defer(req):
        w, c = time.perf_counter_ns(), time.process_time_ns()
        try:
            return original_defer(req)
        finally:
            gate.wall_ns += time.perf_counter_ns()-w
            gate.cpu_ns += time.process_time_ns()-c
    gate.defer = timed_defer
    scheduler.schedule = schedule
    def uninstall():
        gate.restore()
        del scheduler.schedule
        del scheduler._c_gate
        return gate.report()
    return gate, uninstall
