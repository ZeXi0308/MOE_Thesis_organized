"""CPU checks against pinned native methods; no vLLM import or GPU required."""
import ast
import copy
from collections import defaultdict
from enum import Enum, auto
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import admission_probe as probe


VENDOR = Path(__file__).with_name('vendor')


class Status(Enum):
    WAITING = auto()
    RUNNING = auto()
    PREEMPTED = auto()
    WAITING_FOR_REMOTE_KVS = auto()
    FINISHED_STOPPED = auto()


def native_class(filename, name, methods, namespace):
    """Execute the actual checked-in method ASTs, retaining their decorators."""
    module = ast.parse((VENDOR / filename).read_text())
    cls = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == name)
    selected = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in methods]
    assert {n.name for n in selected} == set(methods)
    minimal = ast.ClassDef(name=name, bases=[], keywords=[], body=selected, decorator_list=[])
    tree = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0),
                           minimal], type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree), str(VENDOR / filename), 'exec'), namespace)
    return namespace[name]


NS_NATIVE = {'cdiv': lambda a, b: (a + b - 1) // b, 'RequestStatus': Status,
             'CrossAttentionManager': type('CrossAttentionManager', (), {})}
NativeSingle = native_class('single_type_kv_cache_manager.py', 'SingleTypeKVCacheManager',
    ('get_num_blocks_to_allocate', '_get_num_evictable_blocks', '_has_partial_local_hit',
     'get_num_skipped_tokens', 'remove_skipped_blocks', 'add_local_computed_blocks',
     'allocate_external_computed_blocks', 'allocate_new_blocks'), NS_NATIVE)
NativeCoordinator = native_class('kv_cache_coordinator.py', 'KVCacheCoordinator',
    ('get_num_blocks_to_allocate', 'remove_skipped_blocks', 'allocate_new_computed_blocks',
     'allocate_new_blocks'), NS_NATIVE)
NativeManager = native_class('kv_cache_manager.py', 'KVCacheManager', ('allocate_slots',), NS_NATIVE)


class Pool:
    def __init__(self, free):
        self.free, self.next_id = free, 1

    def get_num_free_blocks(self):
        return self.free

    def get_new_blocks(self, count):
        assert 0 <= count <= self.free
        result = [NS(block_id=i, ref_cnt=1, is_null=False)
                  for i in range(self.next_id, self.next_id + count)]
        self.free -= count
        self.next_id += count
        return result


def manager(free=100, watermark=0, held=0, rid='new', cached=False):
    pool = Pool(free + held)
    m = NativeSingle()
    m.block_size, m.block_pool = 16, pool
    m.req_to_blocks, m.num_cached_block = defaultdict(list), {}
    m._max_admission_blocks_per_request = None
    m.enable_caching = m._record_new_block_ids = False
    m._partial_hit_reqs, m._pending_cow_copies, m.new_block_ids = {}, [], []
    m._null_block = NS(block_id=0, ref_cnt=1, is_null=True)
    if held:
        m.req_to_blocks[rid] = pool.get_new_blocks(held)
    if cached:
        m.num_cached_block[rid] = held
    c = NativeCoordinator()
    c.single_type_managers, c.block_pool = (m,), pool
    k = NativeManager()
    k.coordinator, k.block_pool = c, pool
    k.max_model_len, k.watermark_blocks, k.enable_caching = 64, watermark, False
    k.empty_kv_cache_blocks = NS(blocks=((),))
    k.create_kv_cache_blocks = lambda blocks: NS(blocks=blocks)
    return k


class Request:
    def __init__(self, rid='new', status=Status.WAITING, *, preemptions=0,
                 computed=0, output=0, tokens=32, arrival=999.):
        self.request_id, self.status = rid, status
        self.num_preemptions, self.num_computed_tokens = preemptions, computed
        self.num_output_tokens, self.num_tokens = output, tokens
        self.num_prompt_tokens, self.num_in_flight_tokens = tokens - output, 0
        self.arrival_time, self.has_encoder_inputs = arrival, False

    def is_finished(self):
        return self.status == Status.FINISHED_STOPPED


class Queue:
    def __init__(self, requests=()):
        self.items = list(requests)

    def pop_request(self):
        return self.items.pop(0)

    def prepend_request(self, req):
        self.items.insert(0, req)


def scheduler(requests=(), free=100):
    return NS(requests={r.request_id: r for r in requests}, running=[],
              finished_recving_kv_req_ids=set(), kv_cache_manager=manager(free))


def allocation_state(k):
    m = k.coordinator.single_type_managers[0]
    return copy.deepcopy((vars(k.block_pool), dict(m.req_to_blocks), m.num_cached_block,
                          m._partial_hit_reqs, m._pending_cow_copies, m.new_block_ids))


class PureFitTests(unittest.TestCase):
    def test_matches_native_none_branches_without_query_mutation(self):
        cases = [
            ('full_fail', dict(free=1), {}, {}, False),
            ('reserve_chunk_fail', dict(free=2), {}, dict(reserved_blocks=2), False),
            ('full_check_ignores_reserve', dict(free=2), {}, dict(reserved_blocks=1), True),
            ('watermark_fail', dict(free=2, watermark=1), {}, {}, False),
            ('watermark_not_scheduled', dict(free=2, watermark=1), {},
             dict(has_scheduled_reqs=False), True),
            ('watermark_running_ignored', dict(free=2, watermark=1),
             dict(status=Status.RUNNING), {}, True),
            ('exact_threshold', dict(free=3, watermark=1), {}, {}, True),
            ('remote_external', dict(free=2), {},
             dict(num_new_tokens=0, num_external_computed_tokens=32, delay_cache_blocks=True), True),
            ('empty_allocation_success', dict(free=0, held=1, cached=True),
             dict(tokens=16, computed=15), dict(num_new_tokens=1), True),
            ('cached_append_success', dict(free=1, held=1, cached=True),
             dict(tokens=32, computed=16), dict(num_new_tokens=16), True),
        ]
        for name, config, request_args, overrides, expected in cases:
            with self.subTest(case=name):
                k, req = manager(**config), Request(**request_args)
                kwargs = dict(num_new_tokens=16, full_sequence_must_fit=True)
                kwargs.update(overrides)
                before = allocation_state(k)
                predicted = probe.pure_fit(k, req, **kwargs)
                self.assertEqual(allocation_state(k), before)
                actual = k.allocate_slots(req, **kwargs)
                self.assertEqual(predicted['fits'], actual is not None)
                self.assertEqual(predicted['fits'], expected)
                if not expected:
                    self.assertEqual(allocation_state(k), before)
                if name == 'empty_allocation_success':
                    self.assertEqual(actual.blocks, ([],))
                if name == 'full_fail':
                    self.assertIsNone(predicted['chunk_required_blocks'])

    def test_rejects_outside_supported_call_path(self):
        for kwargs in (dict(full_sequence_must_fit=False), dict(num_lookahead_tokens=1),
                       dict(num_encoder_tokens=1), dict(num_new_computed_tokens=1),
                       dict(num_external_computed_tokens=33),
                       dict(new_computed_blocks=NS(blocks=((object(),),)))):
            with self.subTest(kwargs=kwargs), self.assertRaises(AssertionError):
                probe.pure_fit(manager(), Request(), num_new_tokens=1, **kwargs)


class GateTests(unittest.TestCase):
    def setUp(self):
        self.now, self.wall = 100., 1000.
        for name, value in [('perf_counter', lambda: self.now), ('time', lambda: self.wall)]:
            mocked = patch.object(probe.time, name, side_effect=value)
            mocked.start()
            self.addCleanup(mocked.stop)

    def make_gate(self, *, enabled=False, **kwargs):
        recovery = Request('recover', Status.PREEMPTED, preemptions=1)
        req = Request()
        s = scheduler([recovery, req])
        gate = probe.Gate(s, cap=4, kv_floor=10, probe_enabled=enabled, **kwargs)
        gate.admitted_ids.add('recover')
        return gate, req, recovery

    @staticmethod
    def before(gate, req, **kwargs):
        return gate.before_allocate(req, token_budget=32, num_new_tokens=16, **kwargs)

    def test_unique_admitted_lifecycle_and_same_step_cap(self):
        a, b = Request('a'), Request('b')
        s = scheduler([a, b])
        gate = probe.Gate(s, cap=1, kv_floor=0)
        gate.begin()
        a.status = Status.WAITING_FOR_REMOTE_KVS
        gate.admitted(a)
        gate.admitted(a)
        self.assertEqual(gate.state()['active'], 1)
        self.assertEqual(self.before(gate, b)['row']['reason'], 'cap')
        for status in (Status.PREEMPTED, Status.WAITING, Status.RUNNING):
            a.status, a.num_preemptions = status, 1
            gate.begin()
            self.assertEqual(gate.state()['active'], 1)
            self.assertFalse(gate.is_new(a))
        a.status = Status.FINISHED_STOPPED  # Connector may retain it in requests.
        gate.begin()
        self.assertIn('a', s.requests)
        self.assertEqual(gate.admitted_ids, set())
        self.assertTrue(self.before(gate, b)['row']['base_allowed'])

    def test_missing_admitted_request_is_pruned(self):
        gate, _, _ = self.make_gate()
        del gate.s.requests['recover']
        gate.begin()
        self.assertEqual(gate.admitted_ids, set())

    def test_shadow_once_does_not_deny(self):
        gate, req, _ = self.make_gate()
        first = self.before(gate, req)['row']
        self.assertTrue(first['opportunity'])
        self.assertFalse(first['denied'])
        event = copy.deepcopy(gate.event)
        self.assertEqual(event['kind'], 'shadow')
        later = Request('later')
        gate.s.requests['later'] = later
        self.before(gate, later)
        self.assertEqual(gate.event, event)

    def test_probe_100ms_once_and_250ms_configuration_cap(self):
        for delay, expected in ((.1, .1), (1., .25)):
            with self.subTest(delay=delay):
                self.now = 100.
                gate, req, _ = self.make_gate(enabled=True, delay_s=delay)
                self.assertEqual(self.before(gate, req)['row']['reason'], 'probe')
                event = copy.deepcopy(gate.event)
                self.assertAlmostEqual(gate.release_at - self.now, expected)
                self.now = gate.release_at - .000001
                self.assertTrue(self.before(gate, req)['row']['denied'])
                self.now = gate.release_at
                self.assertFalse(self.before(gate, req)['row']['denied'])
                self.assertEqual(gate.event, event)
                later = Request('later')
                gate.s.requests['later'] = later
                self.assertFalse(self.before(gate, later)['row']['denied'])
                self.assertEqual(gate.event, event)

    def test_external_age_bypass_is_independent_of_probe_deadline(self):
        gate, req, _ = self.make_gate(enabled=True)
        gate.kv_floor = 101
        req.arrival_time = self.wall - 9.999
        self.assertEqual(self.before(gate, req)['row']['reason'], 'kv')
        self.assertIsNone(gate.event)
        gate.begin()
        req.arrival_time = self.wall - 10.
        row = self.before(gate, req)['row']
        self.assertTrue(row['signal_wait_limit_bypass'])
        self.assertEqual(row['reason'], 'probe')
        self.now = gate.release_at
        self.assertEqual(self.before(gate, req)['row']['reason'], 'allow')
        self.assertEqual(req.arrival_time, 990.)

    def test_fifo_expiry_restarts_before_even_existing_recovery(self):
        gate, req, recovering = self.make_gate(enabled=True)
        later = Request('later')
        q = Queue([req, later, recovering])
        self.assertTrue(self.before(gate, req)['row']['denied'])
        gate.hold(q, q.pop_request())
        self.assertEqual(gate.early(later), 'hold')
        gate.hold(q, q.pop_request())
        self.assertIsNone(gate.early(recovering))
        self.now = gate.release_at
        self.assertEqual(gate.early(recovering), 'restart')
        self.assertEqual(q.items, [req, later, recovering])
        self.assertEqual(gate.held, [])
        self.assertIsNone(gate.barrier)
        self.assertFalse(self.before(gate, req)['row']['denied'])

    def test_recovery_bypasses_new_gate_and_categories_are_disjoint(self):
        gate, _, recovering = self.make_gate(enabled=True)
        gate.cap, gate.kv_floor, gate.barrier = 1, 1000, 'cap'
        for status in (Status.PREEMPTED, Status.WAITING_FOR_REMOTE_KVS, Status.WAITING, Status.RUNNING):
            recovering.status = status
            self.assertIsNone(gate.early(recovering))
            self.assertIsNone(self.before(gate, recovering)['row'])
        recovering.status = Status.WAITING_FOR_REMOTE_KVS
        self.assertEqual(gate.state()['recovery_remote_inflight_count'], 1)
        gate.s.finished_recving_kv_req_ids.add(recovering.request_id)
        state = gate.state()
        self.assertEqual((state['recovery_remote_inflight_count'], state['recovery_ready_count']), (0, 1))
        self.assertIsNone(gate.event)

    def test_fit_false_allows_native_none_and_mismatch_is_visible(self):
        gate, req, _ = self.make_gate(enabled=True)
        gate.s.kv_cache_manager.block_pool.free = 0
        gate.cap = 1
        check = self.before(gate, req)
        self.assertEqual(check['row']['reason'], 'native_capacity')
        self.assertFalse(check['row']['denied'])
        self.assertIsNone(gate.barrier)
        gate.after_allocate(req, check, None)
        self.assertFalse(check['row']['native_allocation_result'])
        with self.assertRaisesRegex(AssertionError, 'prediction mismatch'):
            gate.after_allocate(req, check, NS(blocks=([],)))
        self.assertEqual(gate.fit_mismatches, 1)

    def test_empty_allocation_is_success_and_queue_cleanup_is_idempotent(self):
        gate, req, _ = self.make_gate()
        check = self.before(gate, req)
        gate.after_allocate(req, check, NS(blocks=([],)))
        self.assertTrue(check['row']['native_allocation_result'])
        a, b, c, d = [Request(rid) for rid in 'abcd']
        q1, q2 = Queue([a, b]), Queue([c, d])
        for q in (q1, q2, q1):
            gate.hold(q, q.pop_request())
        gate.restore()
        gate.restore()
        self.assertEqual(q1.items, [a, b])
        self.assertEqual(q2.items, [c, d])

    def test_progress_freezes_pending_cohort_and_waits_for_every_original_object(self):
        gate, req, recovering = self.make_gate(enabled=True, release_mode='output_progress')
        recovering.num_output_tokens = 2
        remote = Request('remote', Status.WAITING_FOR_REMOTE_KVS, preemptions=2,
                         computed=16, output=4)
        replay = Request('replay', Status.RUNNING, preemptions=1, computed=1)
        gate.s.requests.update(remote=remote, replay=replay)
        gate.admitted_ids.update(('remote', 'replay'))
        self.assertTrue(self.before(gate, req)['row']['denied'])
        frozen_event = copy.deepcopy(gate.event)
        self.assertEqual({r['request_id'] for r in gate.event['pending_targets']}, {'recover', 'remote'})
        self.now = 100.02
        gate.end(NS(num_scheduled_tokens={'recover': 1, 'replay': 1}))
        progress = gate.report()['recovery_progress']
        self.assertEqual(set(progress), {'recover', 'remote'})
        self.assertEqual(progress['recover']['initial_output'], 2)
        self.assertEqual(progress['remote']['initial_status'], 'WAITING_FOR_REMOTE_KVS')
        self.assertEqual(progress['remote']['initial_preemptions'], 2)
        self.assertEqual(progress['recover']['first_scheduled']['perf_s'], self.now)
        self.assertIsNone(progress['remote']['first_scheduled'])
        self.assertIsNone(progress['recover']['first_output'])
        recovering.num_output_tokens = 3
        self.now = 100.11
        gate.begin()
        self.assertEqual(self.before(gate, req)['row']['reason'], 'probe_progress')
        first_output = copy.deepcopy(progress['recover']['first_output'])
        self.assertEqual(first_output['perf_s'], self.now)
        del gate.s.requests['remote']  # Disappearance alone must not release it.
        self.now = 100.12
        gate.begin()
        self.assertTrue(self.before(gate, req)['row']['denied'])
        self.assertEqual(progress['remote']['first_missing']['perf_s'], self.now)
        self.assertIsNone(progress['remote']['first_finished'])
        remote.status = Status.FINISHED_STOPPED  # Retained original object proves finish.
        self.now = 100.13
        gate.begin()
        self.assertFalse(self.before(gate, req)['row']['denied'])
        self.assertEqual(progress['remote']['first_finished']['perf_s'], self.now)
        self.assertIsNone(progress['remote']['first_output'])
        self.assertEqual(progress['recover']['first_output'], first_output)
        self.assertEqual(gate.report()['release_observations']['actual']['reason'],
                         'output_progress_or_finished')
        self.assertEqual(gate.event, frozen_event)

    def test_progress_keeps_100ms_minimum_even_after_output(self):
        gate, req, recovering = self.make_gate(enabled=True, release_mode='output_progress', delay_s=.01)
        self.assertTrue(self.before(gate, req)['row']['denied'])
        recovering.num_output_tokens += 1
        self.now = 100.05
        gate.begin()
        row = self.before(gate, req)['row']
        self.assertFalse(row['timer_would_block'])
        self.assertTrue(row['progress_would_block'])
        self.assertTrue(row['changed_by_progress'])
        self.assertTrue(row['denied'])
        self.now = 100.1
        gate.begin()
        self.assertFalse(self.before(gate, req)['row']['denied'])
        self.assertEqual(gate.report()['release_observations']['actual']['perf_s'], self.now)
        with self.assertRaises(AssertionError):
            self.make_gate(enabled=True, release_mode='output_progress', max_extra_s=.09)

    def test_timer_shadow_and_progress_action_share_opportunity_and_250ms_deadline(self):
        for mode in ('timer', 'output_progress'):
            with self.subTest(mode=mode):
                self.now = 100.
                gate, req, _ = self.make_gate(enabled=True, release_mode=mode)
                self.before(gate, req)
                self.now = 100.1
                gate.begin()
                row = self.before(gate, req)['row']
                self.assertFalse(row['timer_would_block'])
                self.assertTrue(row['progress_would_block'])
                self.assertTrue(row['progress_additional_opportunity'])
                self.assertEqual(row['changed_by_progress'], mode == 'output_progress')
                self.assertEqual(row['denied'], mode == 'output_progress')
                self.assertEqual(row['reason'], 'probe_progress' if mode == 'output_progress' else 'allow')
                gate.cap = 1  # The same signal is not an extra action if the cap already denies.
                self.assertFalse(self.before(gate, req)['row']['progress_additional_opportunity'])
                gate.cap = 4
                self.now = 100.249999
                gate.begin()
                self.assertEqual(self.before(gate, req)['row']['denied'], mode == 'output_progress')
                self.now = 100.25
                gate.begin()
                row = self.before(gate, req)['row']
                self.assertFalse(row['progress_would_block'])
                self.assertFalse(row['denied'])
                report = gate.report()['release_observations']
                self.assertEqual(report['timer']['perf_s'], 100.1)
                self.assertEqual(report['output_progress']['perf_s'], 100.25)
                self.assertEqual(report['output_progress']['reason'], 'hard_deadline')
                self.assertEqual(report['actual']['release_mode'], mode)
                self.assertEqual(report['actual']['perf_s'], 100.1 if mode == 'timer' else 100.25)

    def test_progress_fifo_uses_progress_window_and_still_bypasses_old_recovery(self):
        gate, req, recovering = self.make_gate(enabled=True, release_mode='output_progress')
        later = Request('later')
        q = Queue([req, later, recovering])
        self.assertTrue(self.before(gate, req)['row']['denied'])
        gate.hold(q, q.pop_request())
        self.now = 100.1
        self.assertEqual(gate.early(later), 'hold')  # Timer expiry cannot reopen this FIFO.
        gate.hold(q, q.pop_request())
        self.assertIsNone(gate.early(recovering))
        self.assertIsNone(self.before(gate, recovering)['row'])
        self.now = 100.25
        self.assertEqual(gate.early(recovering), 'restart')
        self.assertEqual(q.items, [req, later, recovering])
        self.assertEqual(gate.held, [])
        self.assertIsNone(gate.barrier)
        self.assertFalse(self.before(gate, req)['row']['denied'])


class PatchTests(unittest.TestCase):
    def test_real_scheduler_splice_compiles_and_preserves_native_break(self):
        source = (VENDOR / 'scheduler.py').read_text()
        cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'Scheduler')
        schedule = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'schedule')
        import textwrap
        function = textwrap.dedent('\n'.join(source.splitlines()[schedule.lineno-1:schedule.end_lineno])) + '\n'
        patched = probe.patch_source(function)
        tree = ast.parse(patched)
        compile(tree, '<tested C schedule splice>', 'exec')
        self.assertEqual(patched.count('self._c_probe.before_allocate'), 1)
        self.assertEqual(patched.count('self._c_probe.after_allocate'), 1)
        self.assertEqual(patched.count('self._c_probe.admitted'), 2)
        self.assertLess(patched.index('self._c_probe.after_allocate'), patched.index('if new_blocks is None:', patched.index('self._c_probe.before_allocate')))
        with self.assertRaisesRegex(AssertionError, 'anchor changed'):
            probe.patch_source(patched)

    def test_install_finally_restores_on_native_exception(self):
        # Exercise the actual installation wrapper while substituting its already
        # separately compiled source splice with a tiny failing schedule body.
        class Fake:
            def schedule(self):
                raise AssertionError('unpatched')
        s = Fake()
        s.__dict__.update(vars(scheduler()))
        req = Request('held')
        s.queue = Queue([req])
        injected = "def schedule(self):\n    self._c_probe.hold(self.queue, self.queue.pop_request())\n    raise RuntimeError('native failure')\n"
        with patch.object(probe, 'qualify'), patch.object(probe, 'patch_source', return_value=injected):
            gate, uninstall = probe.install(s)
        try:
            with self.assertRaisesRegex(RuntimeError, 'native failure'):
                s.schedule()
            self.assertEqual(s.queue.items, [req])
            self.assertEqual(gate.held, [])
        finally:
            uninstall()
        self.assertNotIn('schedule', vars(s))
        self.assertNotIn('_c_probe', vars(s))


if __name__ == '__main__':
    unittest.main()
