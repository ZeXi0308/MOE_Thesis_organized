"""Bounded CPU checks for admission by declared maximum KV footprint."""
import ast
import inspect
import textwrap
import unittest

import declared_budget as budget
import test_admission_probe as fixture


class DeclaredBudgetTests(unittest.TestCase):
    setUp = fixture.GateTests.setUp
    before = staticmethod(fixture.GateTests.before)

    @staticmethod
    def request(rid, *, prompt=16, max_tokens=16, **kwargs):
        req = fixture.Request(rid, tokens=prompt, **kwargs)
        req.max_tokens = max_tokens
        return req

    def make_gate(self, requests, *, blocks=4, cap=256, free=100):
        scheduler = fixture.scheduler(requests, free=free)
        scheduler.max_model_len = scheduler.kv_cache_manager.max_model_len = 4096
        scheduler.num_waiting_for_streaming_input = 0
        gate = budget.Gate(scheduler, cap=cap, budget_blocks=blocks)
        gate.begin()
        return gate

    def admit(self, gate, req):
        check = self.before(gate, req)
        self.assertFalse(check['row']['denied'])
        result = gate.s.kv_cache_manager.allocate_slots(req, num_new_tokens=16,
            full_sequence_must_fit=True)
        self.assertIsNotNone(result)
        gate.after_allocate(req, check, result)
        req.status = fixture.Status.RUNNING
        gate.admitted(req)
        gate.s.running.append(req)

    def test_charge_only_on_admission_and_never_twice_for_recovery(self):
        req, later = self.request('a', prompt=17), self.request('later')
        gate = self.make_gate([req, later], blocks=8)
        self.assertEqual(gate.declared_pages(req), 3)  # ceil((17 + 16) / 16)
        row = self.before(gate, req)['row']
        self.assertEqual((row['budget_before_blocks'], row['budget_required_blocks'],
            row['budget_limit_blocks']), (0, 3, 8))
        self.assertTrue(row['baseline_allowed'])
        self.assertTrue(row['base_allowed'])
        self.assertEqual(row['reason'], 'allow')
        self.assertEqual(self.before(gate, later)['row']['budget_before_blocks'], 0)
        self.admit(gate, req)
        gate.admitted(req)
        req.num_preemptions = 1
        for status in (fixture.Status.PREEMPTED, fixture.Status.WAITING_FOR_REMOTE_KVS,
                       fixture.Status.RUNNING):
            req.status = status
            gate.begin()
            self.assertIsNone(self.before(gate, req)['row'])
            if status != fixture.Status.PREEMPTED:
                gate.admitted(req)
            self.assertEqual(self.before(gate, later)['row']['budget_before_blocks'], 3)
        self.assertEqual(gate.admitted_ids, {'a'})
        report = gate.report()
        self.assertEqual((report['budget_used_blocks'], report['budget_peak_blocks']), (3, 3))
        self.assertEqual([e['action'] for e in report['budget_events']], ['charge'])
        self.assertEqual((report['block_size'], report['budget_overrides']), (16, 0))

    def test_begin_releases_finished_and_missing_requests(self):
        a, b, later = [self.request(rid) for rid in ('a', 'b', 'later')]
        gate = self.make_gate([a, b, later], blocks=4)
        self.admit(gate, a)
        self.admit(gate, b)
        self.assertEqual(self.before(gate, later)['row']['budget_before_blocks'], 4)
        a.status = fixture.Status.FINISHED_STOPPED
        gate.s.running.remove(a)
        gate.begin()
        self.assertIn('a', gate.s.requests)  # A connector may retain the object.
        self.assertEqual(self.before(gate, later)['row']['budget_before_blocks'], 2)
        del gate.s.requests['b']
        gate.s.running.remove(b)
        gate.begin()
        self.assertEqual(self.before(gate, later)['row']['budget_before_blocks'], 0)
        self.assertEqual(gate.admitted_ids, set())
        report = gate.report()
        self.assertEqual((report['budget_used_blocks'], report['budget_peak_blocks']), (0, 4))
        self.assertEqual([e['cause'] for e in report['budget_events'] if e['action'] == 'release'],
            ['finished', 'removed_after_admission'])

    def test_exact_budget_edge_invalid_single_bounds_and_cap(self):
        a, later = self.request('a'), self.request('later')
        gate = self.make_gate([a, later], blocks=2)
        self.assertEqual(gate.declared_pages(a), 2)
        self.assertEqual(self.before(gate, a)['row']['reason'], 'allow')
        self.admit(gate, a)
        row = self.before(gate, later)['row']
        self.assertEqual((row['budget_before_blocks'], row['budget_required_blocks'],
            row['budget_limit_blocks']), (2, 2, 2))
        self.assertEqual(row['reason'], 'declared_budget')
        self.assertTrue(row['denied'])
        self.assertTrue(row['baseline_allowed'])
        self.assertTrue(row['base_allowed'])
        self.assertFalse(row['final_allowed'])
        self.assertTrue(row['changed_by_declared_budget'])
        with self.assertRaises(ValueError):
            gate.declared_pages(self.request('too_large', prompt=17))
        with self.assertRaises(ValueError):
            gate.declared_pages(self.request('beyond_context', prompt=4096, max_tokens=1))
        a, later = self.request('cap_a'), self.request('cap_later')
        cap_gate = self.make_gate([a, later], blocks=4, cap=1)
        self.admit(cap_gate, a)
        cap_row = self.before(cap_gate, later)['row']
        self.assertEqual(cap_row['reason'], 'cap')
        self.assertTrue(cap_row['denied'])
        self.assertFalse(cap_row['baseline_allowed'])
        self.assertFalse(cap_row['changed_by_declared_budget'])

    def test_fifo_restore_recovery_bypass_and_no_age_bypass(self):
        old, first, later = [self.request(rid) for rid in ('old', 'first', 'later')]
        gate = self.make_gate([old, first, later], blocks=2)
        self.admit(gate, old)
        old.num_preemptions = 1
        old.status = fixture.Status.PREEMPTED
        gate.s.running.clear()
        self.wall = 1000000.  # An old external arrival never cancels this budget.
        row = self.before(gate, first)['row']
        self.assertEqual(row['reason'], 'declared_budget')
        self.assertTrue(row['denied'])
        queue = fixture.Queue([first, later, old])
        gate.hold(queue, queue.pop_request())
        self.assertEqual(gate.early(later), 'hold')
        fifo = gate.decisions[-1]
        self.assertEqual(fifo['reason'], 'fifo_declared_budget')
        self.assertIsNone(fifo['native_fit'])
        self.assertIsNone(fifo['base_allowed'])
        gate.hold(queue, queue.pop_request())
        for status in (fixture.Status.PREEMPTED, fixture.Status.WAITING_FOR_REMOTE_KVS):
            old.status = status
            self.assertIsNone(gate.early(old))
            self.assertIsNone(self.before(gate, old)['row'])
        gate.restore()
        gate.restore()
        self.assertEqual(queue.items, [first, later, old])
        self.assertEqual(gate.held, [])

    def test_native_capacity_and_stop_scan_preserve_old_recovery(self):
        req = self.request('no_native_capacity')
        empty = self.make_gate([req], blocks=2, free=0)
        check = self.before(empty, req)
        self.assertEqual(check['row']['reason'], 'native_capacity')
        self.assertFalse(check['row']['denied'])
        self.assertFalse(empty.stop_new_scan(check))
        empty.after_allocate(req, check, None)
        old, req = self.request('old'), self.request('new')
        gate = self.make_gate([old, req], blocks=2)
        self.admit(gate, old)
        check = self.before(gate, req)
        self.assertTrue(check['row']['denied'])
        self.assertTrue(gate.stop_new_scan(check))
        self.assertTrue(check['row']['native_waiting_scan_break'])
        old.status, old.num_preemptions = fixture.Status.WAITING_FOR_REMOTE_KVS, 1
        gate.s.running.clear()
        gate.begin()
        check = self.before(gate, req)
        self.assertTrue(check['row']['denied'])
        self.assertFalse(gate.stop_new_scan(check))
        old.status = fixture.Status.RUNNING
        gate.s.running.append(old)
        gate.s.num_waiting_for_streaming_input = 1
        self.assertFalse(gate.stop_new_scan(self.before(gate, req)))

        # Execute just the installer's source replacement, then inspect the
        # real native schedule AST. No scheduler body or engine runs here.
        source = (fixture.VENDOR / 'scheduler.py').read_text()
        cls = next(n for n in ast.parse(source).body
            if isinstance(n, ast.ClassDef) and n.name == 'Scheduler')
        schedule = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'schedule')
        function = textwrap.dedent('\n'.join(source.splitlines()[schedule.lineno-1:schedule.end_lineno])) + '\n'
        installer = ast.parse(inspect.getsource(budget.install)).body[0].body
        start = next(i for i, n in enumerate(installer)
            if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'anchor' for t in n.targets))
        namespace = {'source': fixture.probe.patch_source(function)}
        exec(compile(ast.fix_missing_locations(ast.Module(body=installer[start:start+3],
            type_ignores=[])), '<actual declared-budget source splice>', 'exec'), namespace)
        patched = ast.parse(namespace['source'])
        compile(patched, '<declared-budget native schedule>', 'exec')
        denied = next(n for n in ast.walk(patched) if isinstance(n, ast.If)
            and ast.unparse(n.test) == "_probe_check['row'] is not None and _probe_check['row']['denied']")
        self.assertEqual(ast.unparse(denied.body[0].test),
            'self._c_probe.timed(self._c_probe.stop_new_scan, _probe_check)')
        self.assertIsInstance(denied.body[0].body[0], ast.Break)
        self.assertIn('self._c_probe.hold', ast.unparse(denied.body[1]))


if __name__ == '__main__':
    unittest.main()
