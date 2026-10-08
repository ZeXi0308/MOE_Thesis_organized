"""Simple FIFO-within-restorations action baseline, not a novel scheduler.

Pinned native API: vLLM 0.26.0 FCFSRequestQueue is a deque;
Scheduler._preempt_request(request, timestamp) sets PREEMPTED then prepends.
Install AFTER observe_native_pressure and BEFORE health_native.generate:

    with observe_native_pressure(engine, pressure_path):
        with restore_fifo(engine, policy_path):
            common.generate(...)

Only PREEMPTED slots in the existing waiting queue are permuted. The native
allocation, victim selection, execution and sampling paths remain untouched.
No future output lengths, EOS, predictions or cache observations are consulted.
"""
from collections import deque
from contextlib import contextmanager
import inspect
import json
from pathlib import Path
import time


def _require(ok, message):
    if not ok:
        raise RuntimeError(message)


def _name(status):
    return getattr(status, 'name', str(status))


def _same_objects(a, b):
    return len(a) == len(b) and all(x is y for x, y in zip(a, b))


@contextmanager
def restore_fifo(engine, output_path):
    """Record native preemption episodes and restore them in episode FIFO order.

    Installation requires a cold, drained synchronous FCFS scheduler. The caller
    owns the GPU lock and timeout. No skipped-waiting requests are supported in
    this task. Hooks are restored on every exit, including generation failure.
    """
    s = engine.engine_core.engine_core.scheduler
    queue = s.waiting
    _require(isinstance(queue, deque) and type(queue).__name__ == 'FCFSRequestQueue',
             'restore FIFO requires the pinned native deque FCFSRequestQueue')
    _require(not queue and not s.running and not s.skipped_waiting,
             'restore FIFO must be installed after cold drain, before requests')
    original_schedule, original_preempt = s.schedule, s._preempt_request
    signature = inspect.signature(original_preempt)
    _require(tuple(signature.parameters) == ('request', 'timestamp'),
             'native bound _preempt_request signature differs')
    latest, events, changes = {}, [], []
    dirty, active_call, call_count = False, None, 0
    decision_wall, preempt_hook_wall = 0.0, 0.0
    origin = time.perf_counter()
    report = dict(schema='c-restore-fifo-policy-v1', status='INCOMPLETE',
        scope='Simple queue-order action baseline only; no novelty or performance claim.',
        policy='Ascending native preemption-call sequence within PREEMPTED slots; latest episode replaces earlier episode for the same request.',
        native_queue_class=type(queue).__module__ + '.' + type(queue).__qualname__,
        preempt_signature=str(signature), preempt_events=events, queue_changes=changes,
        observer_order='Policy reorders before calling the outer native pressure observer schedule wrapper.',
        cost_scope='Hook wall time excludes native calls; receipt serialization is included in caller measurement phase, not hook sums.')

    def rows(requests):
        return [dict(request_id=r.request_id, status=_name(r.status),
                     preempt_event_sequence=latest.get(r.request_id)
                     if _name(r.status) == 'PREEMPTED' else None)
                for r in requests]

    def reorder():
        nonlocal dirty
        _require(s.waiting is queue, 'native waiting queue object changed')
        _require(not s.skipped_waiting, 'unexpected skipped-waiting requests')
        if not dirty:
            return
        before = list(queue)
        _require(len({id(r) for r in before}) == len(before) and
                 len({r.request_id for r in before}) == len(before),
                 'duplicate request in native waiting queue')
        _require(all(_name(r.status) in ('WAITING', 'PREEMPTED') for r in before),
                 'unexpected waiting request status')
        restored = [r for r in before if _name(r.status) == 'PREEMPTED']
        _require(all(r.request_id in latest for r in restored),
                 'PREEMPTED request has no observed native preemption episode')
        ordered = iter(sorted(restored, key=lambda r: latest[r.request_id]))
        after = [next(ordered) if _name(r.status) == 'PREEMPTED' else r for r in before]
        _require(sorted(map(id, before)) == sorted(map(id, after)),
                 'FIFO reorder changes request membership')
        _require(all(a is b for a, b in zip(before, after)
                     if _name(a.status) != 'PREEMPTED'),
                 'FIFO reorder changes a non-PREEMPTED queue position')
        if not _same_objects(before, after):
            change = dict(schedule_call=active_call, host_s=time.perf_counter() - origin,
                          before=rows(before), after=rows(after))
            # Use the observed queue's inherited deque API, preserving identity.
            try:
                deque.clear(queue)
                deque.extend(queue, after)
                _require(_same_objects(list(queue), after), 'queue rewrite differs')
            except BaseException:
                deque.clear(queue)
                deque.extend(queue, before)
                raise
            changes.append(change)
        dirty = False

    def schedule(*args, **kwargs):
        nonlocal active_call, call_count, decision_wall
        started, previous = time.perf_counter(), active_call
        active_call = call_count
        call_count += 1
        try:
            try:
                reorder()
            finally:
                decision_wall += time.perf_counter() - started
            return original_schedule(*args, **kwargs)
        finally:
            active_call = previous

    def preempt(*args, **kwargs):
        nonlocal dirty, preempt_hook_wall
        started = time.perf_counter()
        bound = signature.bind(*args, **kwargs)
        request, timestamp = bound.arguments['request'], bound.arguments['timestamp']
        _require(_name(request.status) == 'RUNNING', 'native preemption input is not RUNNING')
        sequence = len(events)
        event = dict(sequence=sequence, schedule_call=active_call,
            host_s=started - origin, request_id=request.request_id,
            previous_episode_sequence=latest.get(request.request_id),
            native_timestamp=timestamp, status_before=_name(request.status),
            num_preemptions_before=request.num_preemptions, status='INCOMPLETE')
        events.append(event)
        native_started = time.perf_counter()
        native_wall = 0.0
        try:
            try:
                result = original_preempt(*args, **kwargs)
            finally:
                native_wall = time.perf_counter() - native_started
            _require(_name(request.status) == 'PREEMPTED' and
                     request.num_preemptions == event['num_preemptions_before'] + 1 and
                     sum(r is request for r in queue) == 1,
                     'native preemption did not create exactly one waiting restoration')
            latest[request.request_id] = sequence
            dirty = True
            event.update(status='COMPLETE', status_after=_name(request.status),
                         num_preemptions_after=request.num_preemptions)
            return result
        except BaseException as exc:
            event['error'] = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            event['native_call_wall_s'] = native_wall
            event['policy_hook_wall_s'] = time.perf_counter() - started - native_wall
            preempt_hook_wall += event['policy_hook_wall_s']

    with Path(output_path).open('x', encoding='utf-8') as stream:
        try:
            s.schedule, s._preempt_request = schedule, preempt
            yield report
            report['status'] = 'COMPLETE'
        except BaseException as exc:
            report['error'] = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            s.schedule, s._preempt_request = original_schedule, original_preempt
            report.update(observation_end_s=time.perf_counter() - origin,
                schedule_calls=call_count, preempt_calls=len(events),
                queue_change_calls=len(changes), decision_hook_wall_s=decision_wall,
                preempt_hook_wall_s=preempt_hook_wall,
                policy_hook_wall_s_total=decision_wall + preempt_hook_wall,
                final_pending_reorder=dirty, hooks_restored=True)
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')


def self_test():
    """Three bounded CPU fixtures; no vLLM import or GPU access."""
    import tempfile
    from types import SimpleNamespace as NS

    class FCFSRequestQueue(deque):
        pass

    def request(rid):
        return NS(request_id=rid, status='RUNNING', num_preemptions=0)

    class Scheduler:
        def __init__(self):
            self.waiting, self.running, self.skipped_waiting = FCFSRequestQueue(), [], []
            self.snapshots, self.raise_next = [], False
        def _preempt_request(self, request, timestamp):
            request.status = 'PREEMPTED'
            request.num_preemptions += 1
            self.waiting.appendleft(request)
        def schedule(self):
            self.snapshots.append([r.request_id for r in self.waiting])
            if self.raise_next:
                raise ValueError('fixture failure')
            return tuple(self.snapshots[-1])

    def engine(s):
        return NS(engine_core=NS(engine_core=NS(scheduler=s)))

    with tempfile.TemporaryDirectory(prefix='restore-fifo-fixtures-') as tmp:
        base = Path(tmp)
        # 1. Two native prepends reverse episode order. Reorder only restoration
        # slots even with a new request between them; observer sees policy order.
        s = Scheduler()
        native_schedule = s.schedule
        observed = []
        def observer():
            observed.append([r.request_id for r in s.waiting])
            return native_schedule()
        s.schedule = observer
        before_schedule, before_preempt = s.schedule, s._preempt_request
        with restore_fifo(engine(s), base / 'mixed.json') as report:
            a, b = request('a'), request('b')
            n1, n2 = request('new1'), request('new2')
            n1.status = n2.status = 'WAITING'
            s._preempt_request(a, 1.0)
            s._preempt_request(b, 1.0)
            s.waiting.clear(); s.waiting.extend([b, n1, a, n2])
            assert s.schedule() == ('a', 'new1', 'b', 'new2')
            assert observed[-1] == ['a', 'new1', 'b', 'new2']
            assert list(s.waiting) == [a, n1, b, n2]
            assert report['queue_changes'][0]['before'][0]['request_id'] == 'b'
            assert s.schedule() == ('a', 'new1', 'b', 'new2')
            assert len(report['queue_changes']) == 1
        assert s.schedule is before_schedule and s._preempt_request == before_preempt
        assert report['status'] == 'COMPLETE' and report['hooks_restored']

        # 2. A request's second preemption is a new episode; it waits behind a
        # request whose first episode is still waiting, irrespective of old age.
        s = Scheduler()
        with restore_fifo(engine(s), base / 'repeat.json') as report:
            a, b = request('a'), request('b')
            s._preempt_request(a, 1.0); s._preempt_request(b, 1.0)
            assert s.schedule() == ('a', 'b')
            s.waiting.popleft(); a.status = 'RUNNING'
            s._preempt_request(a, 2.0)
            assert s.schedule() == ('b', 'a')
            assert report['preempt_events'][-1]['sequence'] == 2
            assert report['preempt_events'][-1]['previous_episode_sequence'] == 0

        # 3. Native schedule failure restores both wrappers and writes failure
        # evidence without swallowing the exception or modifying native work.
        s = Scheduler()
        before_schedule, before_preempt = s.schedule, s._preempt_request
        try:
            with restore_fifo(engine(s), base / 'failure.json'):
                s.raise_next = True
                s.schedule()
        except ValueError as exc:
            assert str(exc) == 'fixture failure'
        else:
            raise AssertionError('native failure was swallowed')
        assert s.schedule == before_schedule and s._preempt_request == before_preempt
        failed = json.loads((base / 'failure.json').read_text())
        assert failed['status'] == 'INCOMPLETE' and failed['hooks_restored']
    return dict(status='PASS', fixtures=3, gpu_used=False)


if __name__ == '__main__':
    print(json.dumps(self_test()))
