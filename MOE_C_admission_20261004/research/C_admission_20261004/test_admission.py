"""Small CPU lifecycle checks for the new-prefill gate; no vLLM/GPU import."""
import hashlib
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import admission


class Request:
    """Identity-hashable request stub, like the native scheduler's set members."""
    def __init__(self, **fields):
        self.__dict__.update(fields)


def request(rid, status='WAITING', preemptions=0, computed=0, output=0,
            arrival=100., prompt=32):
    return Request(request_id=rid, status=NS(name=status), num_preemptions=preemptions,
                   num_computed_tokens=computed, num_output_tokens=output,
                   arrival_time=arrival, num_prompt_tokens=prompt)


class Queue:
    def __init__(self, requests=()):
        self.items = list(requests)

    def __bool__(self):
        return bool(self.items)

    def peek_request(self):
        return self.items[0]

    def pop_request(self):
        return self.items.pop(0)

    def prepend_request(self, req):
        self.items.insert(0, req)


class FakeScheduler:
    def __init__(self, requests=(), free=500):
        self.requests = {r.request_id: r for r in requests}
        self.running = []
        self.waiting = Queue(requests)
        self._inflight_prefills = {r for r in requests if r.status.name == 'WAITING_FOR_REMOTE_KVS'}
        self.free = free
        self.kv_cache_manager = NS(block_pool=NS(get_num_free_blocks=lambda: self.free))
        self.fail_after_pop = False

    def schedule(self):
        result = NS(num_scheduled_tokens={})
        if True:
            while self.waiting:
                request_queue = self.waiting
                request = request_queue.peek_request()
                request_id = request.request_id
                request_queue.pop_request()
                if self.fail_after_pop:
                    raise RuntimeError('injected native failure')
                self.running.append(request)
                result.num_scheduled_tokens[request_id] = 1
        return result


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.clock = patch.object(admission.time, 'time', return_value=101.)
        self.clock.start()
        self.addCleanup(self.clock.stop)

    def test_native_never_blocks(self):
        scheduler = FakeScheduler(free=0)
        scheduler.running = [request('running')]
        gate = admission.Gate(scheduler, 'native', cap=1)
        gate.latch = True
        self.assertFalse(gate.defer(request('new')))

    def test_existing_work_and_recovery_never_gate(self):
        scheduler = FakeScheduler(free=0)
        scheduler.running = [request('running')]
        gate = admission.Gate(scheduler, 'recovery', cap=1)
        gate.latch = True
        existing = [request('preempted', 'PREEMPTED', 1),
                    request('remote', 'WAITING_FOR_REMOTE_KVS', 1),
                    request('promoted', preemptions=1),
                    request('chunk', computed=8), request('output', output=1),
                    request('running', 'RUNNING'), request('started')]
        gate.started_ids.add('started')
        for req in existing:
            with self.subTest(req=req.request_id):
                self.assertFalse(gate.defer(req))

    def test_kv_threshold_and_wait_bypass_preserve_cap(self):
        scheduler = FakeScheduler(free=409)
        gate = admission.Gate(scheduler, 'kv', cap=1, kv_floor=410, max_hold_s=10)
        self.assertTrue(gate.defer(request('below')))
        scheduler.free = 410
        self.assertFalse(gate.defer(request('at_threshold')))
        scheduler.free = 0
        self.assertFalse(gate.defer(request('expired', arrival=91.)))
        scheduler.running = [request('running')]
        self.assertTrue(gate.defer(request('expired_cap', arrival=91.)))

    def test_remote_load_counts_as_active_once(self):
        remote = request('remote', 'WAITING_FOR_REMOTE_KVS', 1)
        scheduler = FakeScheduler([remote])
        gate = admission.Gate(scheduler, 'fixed', cap=1)
        self.assertEqual(gate.state()['active'], 1)
        self.assertTrue(gate.defer(request('new')))
        self.assertFalse(gate.defer(remote))

    def test_inflight_remote_count_matches_live_full_scan(self):
        decode, prefill = request('decode', 'RUNNING'), request('prefill', 'RUNNING')
        first = request('first', 'WAITING_FOR_REMOTE_KVS', 1)
        second = request('second', 'WAITING_FOR_REMOTE_KVS', 1)
        paused, waiting = request('paused', 'PREEMPTED', 1), request('new')
        finished = request('finished', 'FINISHED_ABORTED')
        scheduler = FakeScheduler([decode, prefill, first, second, paused, waiting, finished])
        scheduler.running = [decode, prefill]
        scheduler._inflight_prefills.update([prefill, paused, waiting, finished])
        gates = [admission.Gate(scheduler, mode, cap=8)
                 for mode in ('native', 'fixed', 'kv', 'recovery')]

        def check(expected):
            full_scan = len(scheduler.running) + sum(
                r.status.name == 'WAITING_FOR_REMOTE_KVS' for r in scheduler.requests.values())
            self.assertEqual(full_scan, expected)
            for gate in gates:
                with self.subTest(mode=gate.mode, expected=expected):
                    self.assertEqual(gate.state()['active'], full_scan)

        check(4)  # Ordinary prefill is already in running; count remote loads once.
        first.status.name = 'PREEMPTED'  # Completed receive remains in the set until rescheduled.
        check(3)
        del scheduler.requests['second']  # Defensive cleanup boundary: stale set member.
        check(2)
        replacement = request('second', 'WAITING_FOR_REMOTE_KVS')
        scheduler.requests['second'] = replacement
        scheduler._inflight_prefills.add(replacement)
        check(3)  # A stale object with the same ID must not double-count its replacement.
        replacement.status.name = 'FINISHED_ABORTED'
        scheduler._inflight_prefills.clear()
        check(2)
        scheduler.running.clear()
        scheduler.requests.clear()
        check(0)

    def test_recovery_signal_hysteresis_and_incremental_decision(self):
        scheduler = FakeScheduler()
        gate = admission.Gate(scheduler, 'recovery', cap=3)
        with patch.object(admission.time, 'perf_counter', return_value=1.):
            gate.begin()
        scheduler.requests = {r.request_id: r for r in
                              [request('r1', 'PREEMPTED', 1),
                               request('r2', 'WAITING_FOR_REMOTE_KVS', 1)]}
        scheduler._inflight_prefills = {scheduler.requests['r2']}
        with patch.object(admission.time, 'perf_counter', return_value=1.2):
            gate.begin()
        self.assertTrue(gate.latch)
        self.assertTrue(gate.defer(request('new')))
        self.assertTrue(gate.decisions[-1]['changed_by_recovery'])
        self.assertFalse(gate.defer(request('expired', arrival=90.)))
        scheduler.requests.pop('r2')
        with patch.object(admission.time, 'perf_counter', return_value=1.4):
            gate.begin()
        self.assertTrue(gate.latch)
        scheduler.requests.clear()
        with patch.object(admission.time, 'perf_counter', return_value=1.6):
            gate.begin()
        self.assertFalse(gate.latch)
        self.assertFalse(gate.defer(request('after_drain')))

    def test_restore_preserves_each_queue_fifo(self):
        a, b, c, d = [request(rid) for rid in 'abcd']
        q1, q2 = Queue([a, b]), Queue([c, d])
        gate = admission.Gate(FakeScheduler(), 'kv', cap=4)
        gate.hold(q1, q1.pop_request())
        gate.hold(q2, q2.pop_request())
        gate.hold(q1, q1.pop_request())
        gate.restore()
        self.assertEqual(q1.items, [a, b])
        self.assertEqual(q2.items, [c, d])
        self.assertEqual(gate.held, [])
        gate.restore()  # Idempotent cleanup.
        self.assertEqual(q1.items, [a, b])

    def test_installed_wrapper_restores_after_native_failure(self):
        first, second = request('a'), request('b')
        recovering = request('resume', 'PREEMPTED', 1)
        scheduler = FakeScheduler([first, second, recovering], free=0)
        scheduler.fail_after_pop = True
        # Exercise the actual source splice/wrapper on this tiny scheduler,
        # substituting only its source identity for the production hash.
        fake_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        with patch.object(admission, 'SCHEDULER_SHA', fake_hash):
            gate, uninstall = admission.install(scheduler, 'kv', cap=3)
        try:
            with self.assertRaisesRegex(RuntimeError, 'injected native failure'):
                scheduler.schedule()
            self.assertEqual(scheduler.waiting.items, [first, second])
            self.assertEqual(gate.held, [])
            self.assertEqual(gate.starts, [])
        finally:
            uninstall()
        self.assertNotIn('schedule', vars(scheduler))
        self.assertNotIn('_c_gate', vars(scheduler))


if __name__ == '__main__':
    unittest.main()
