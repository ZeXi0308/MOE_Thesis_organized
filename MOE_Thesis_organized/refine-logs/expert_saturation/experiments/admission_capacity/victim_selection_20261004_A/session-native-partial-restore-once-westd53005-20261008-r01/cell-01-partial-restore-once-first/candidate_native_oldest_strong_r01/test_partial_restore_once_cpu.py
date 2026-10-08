"""Targeted CPU checks for the partial-restoration one-shot native probe."""
import ast
from copy import deepcopy
import os
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'pkg'))
import staged_store_rotation as adapter
from rotation_native import patched_schedule_tree

PINNED = HERE.parent.parent / '20260929_commit_recheck/liveness_pinned_sources_20260930/scheduler.py'


def row(index, release, *, computed=1613, prompt=956, output=658, qualified=True, shared=0):
    return dict(index=index, immediate_releasable_blocks=release, shared_blocks=shared,
        held_blocks=release+shared, release_state_error=None, computed_tokens=computed,
        prompt_tokens=prompt, output_tokens=output, qualified=qualified, request_status='RUNNING')


def rows():
    return [row(7, 99), row(8, 101), row(9, 101),
            row(10, 100, computed=1592, prompt=1477, output=402, qualified=False)]


def real_pick(rule='partial_restore_once', active=False):
    install = next(n for n in ast.parse((HERE/'pkg/staged_store_rotation.py').read_text()).body
                   if isinstance(n, ast.FunctionDef) and n.name == 'install')
    pick = deepcopy(next(n for n in install.body if isinstance(n, ast.FunctionDef) and n.name == 'pick_victim'))
    specs = [(16, 4, 19, 2), (2000, 400, 2399, 150),
             (956, 658, 1613, 101), (1477, 402, 1592, 100)]
    requests = [NS(request_id=str(i), arrival_time=float(i), num_preemptions=1,
                   num_output_tokens=spec[1]) for i, spec in enumerate(specs)]
    owned = {}; blocks = []; states = {}
    for req, (prompt, output, computed, held) in zip(requests, specs):
        own = [NS(block_id=len(blocks)+i, ref_cnt=1, is_null=False) for i in range(held)]
        blocks.extend(own); owned[req.request_id] = own
        states[req.request_id] = adapter.RequestState(req.request_id, computed, prompt, output,
            1024, 'RUNNING', tuple(b.block_id for b in own))
    scheduler = NS(running=requests.copy(), _rotation_full_running_enabled=False)
    state = lambda mode: dict(mode=mode, consumed=False, trigger_count=0,
                               proposal_count=0, action_request_count=0)
    namespace = dict(vars(adapter), scheduler=scheduler, victim_rule=rule, full_running_mode='off',
        protected=requests[0] if active else None, phase=None, current_guard='off', block_size=16,
        pool=NS(blocks=blocks, get_num_free_blocks=lambda: 0), owned=owned,
        residence={r.request_id: (1, r.num_output_tokens) for r in requests}, step=12,
        data=dict(victim_decisions=[]), equal_once_state=state('SHADOW'),
        partial_once_state=state('ACTIVE' if rule == 'partial_restore_once' else 'SHADOW'),
        cs=NS(_req_status={r.request_id: NS(transfer_jobs=set()) for r in requests}, _jobs={}),
        view=lambda r: states[r.request_id], bidkv_score=lambda r: {}, host_prefix=lambda *args: {})
    exec(compile(ast.fix_missing_locations(ast.Module(body=[pick], type_ignores=[])), '<actual-pick-victim>', 'exec'), namespace)
    scheduler._rotation_pick_victim = namespace['pick_victim']
    calls = []
    scheduler._preempt_request = lambda *args: calls.append(args)
    return scheduler, namespace, requests, calls


def native_suffix_action():
    tree = patched_schedule_tree(PINNED.read_text())
    branch = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                  and ast.unparse(n.test) == 'self.policy == SchedulingPolicy.PRIORITY')
    preempt = next(n for n in ast.walk(tree) if isinstance(n, ast.Expr)
                   and ast.unparse(n) == 'self._preempt_request(preempted_req, scheduled_timestamp)')
    function = ast.parse("def execute(self, req_index, scheduled_running_reqs, scheduled_timestamp):\n    return None\n").body[0]
    function.body = deepcopy(branch.orelse) + [deepcopy(preempt), ast.Return(value=ast.Name(id='preempted_req', ctx=ast.Load()))]
    namespace = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), '<native-fcfs-preempt>', 'exec'), namespace)
    return namespace['execute']


class PartialRestoreOnceTests(unittest.TestCase):
    def test_minimum_sufficient_release_nearest_tie_and_unqualified_rows(self):
        choose = adapter._partial_restore_once_choice
        observed = rows()
        # The partial tail is unqualified by the old selector, yet native legal.
        self.assertEqual(choose(observed, 7, False, False), (9, None, True, 2))
        observed[0]['qualified'] = False
        self.assertEqual(choose(observed, 7, False, False), (9, None, True, 2))
        observed[2]['shared_blocks'] = 1; observed[2]['held_blocks'] += 1
        self.assertEqual(choose(observed, 7, False, False), (8, None, True, 1))
        observed[1]['immediate_releasable_blocks'] = observed[1]['held_blocks'] = 99
        self.assertEqual(choose(observed, 7, False, False), (10, 'NO_UNSHARED_SUFFICIENT_DECODE', True, 0))

    def test_once_phase_self_unknown_and_progress_guards(self):
        choose = adapter._partial_restore_once_choice
        self.assertEqual(choose(rows(), 7, False, True), (10, 'ALREADY_CONSUMED', False, 0))
        self.assertEqual(choose(rows(), 7, True, False), (10, 'ACTIVE_PROTECTION_OR_PHASE', False, 0))
        self.assertEqual(choose(rows(), 10, False, False), (10, 'SELF_OR_SINGLETON_TAIL', False, 0))
        self.assertEqual(choose(rows()[-1:], 10, False, False), (10, 'SELF_OR_SINGLETON_TAIL', False, 0))
        for key, value in [('immediate_releasable_blocks', None), ('shared_blocks', None),
                           ('held_blocks', 999), ('release_state_error', 'unknown')]:
            observed = rows(); observed[0][key] = value
            self.assertEqual(choose(observed, 7, False, False), (10, 'UNKNOWN_PHYSICAL_RELEASE', False, 0))
        for key, value, reason in [('prompt_tokens', None, 'UNKNOWN_TAIL_PROGRESS'),
                ('output_tokens', 0, 'TAIL_NOT_PARTIAL_RESTORE'),
                ('computed_tokens', 1878, 'TAIL_NOT_PARTIAL_RESTORE'),
                ('request_status', 'PREEMPTED', 'TAIL_NOT_PARTIAL_RESTORE')]:
            observed = rows(); observed[-1][key] = value
            self.assertEqual(choose(observed, 7, False, False), (10, reason, False, 0))
        observed = rows(); observed[-1]['shared_blocks'] = 1; observed[-1]['held_blocks'] += 1
        self.assertEqual(choose(observed, 7, False, False), (10, 'SHARED_TAIL', False, 0))

    def test_actual_hook_suffix_original_preempt_once_and_shadow_consumption(self):
        execute = native_suffix_action()
        for rule, first_index in [('partial_restore_once', 2), ('tail', 3)]:
            scheduler, namespace, requests, calls = real_pick(rule)
            stamp = object()
            picked = execute(scheduler, 1, [requests[0]], stamp)
            self.assertIs(picked, requests[first_index])
            self.assertNotIn(picked, scheduler.running)
            self.assertEqual(len(calls), 1)
            self.assertIs(calls[0][0], picked); self.assertIs(calls[0][1], stamp)
            decision = namespace['data']['victim_decisions'][0]
            self.assertEqual([r['index'] for r in decision['candidates']], [1, 2, 3])
            self.assertTrue(decision['fallback_unknown'])
            probe = decision['partial_restore_once']
            self.assertEqual(probe['proposed_request'], requests[2].request_id)
            self.assertEqual(probe['returned_request'], picked.request_id)
            self.assertEqual(probe['extra_releasable_blocks'], 1)
            self.assertEqual(probe['action_requested'], rule == 'partial_restore_once')
            self.assertEqual((probe['consumed_before'], probe['consumed_after']), (False, True))
            last = scheduler.running[-1]
            execute(scheduler, 1, [requests[0]], stamp)
            self.assertEqual(len(calls), 2); self.assertIs(calls[-1][0], last)
            self.assertEqual(namespace['partial_once_state']['proposal_count'], 1)
            self.assertEqual(namespace['partial_once_state']['action_request_count'], int(rule == 'partial_restore_once'))
            self.assertEqual(namespace['data']['victim_decisions'][-1]['partial_restore_once']['fallback'], 'ALREADY_CONSUMED')
        scheduler, namespace, requests, _ = real_pick(active=True)
        self.assertIs(execute(scheduler, 1, [requests[0]], 0.), requests[-1])
        self.assertFalse(namespace['partial_once_state']['consumed'])

    def test_actual_combination_guards_keep_other_controllers_off(self):
        install = next(n for n in ast.parse((HERE/'pkg/staged_store_rotation.py').read_text()).body
                       if isinstance(n, ast.FunctionDef) and n.name == 'install')
        start = next(i for i, n in enumerate(install.body) if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == 'victim_rule' for t in n.targets))
        end = next(i for i, n in enumerate(install.body) if isinstance(n, ast.Assign)
                   and ast.unparse(n.targets[0]) == "data['capacity_deferral_mode']")
        code = compile(ast.Module(body=deepcopy(install.body[start:end]), type_ignores=[]), '<actual-probe-guards>', 'exec')
        baseline = dict(A_NATIVE_VICTIM_RULE='partial_restore_once', A_NATIVE_VICTIM_FULL_RUNNING='off',
            A_NATIVE_VICTIM_CURRENT_GUARD='off', A_SELF_PREEMPT_CONTINUE='off',
            A_NATIVE_CAPACITY_DEFERRAL='off', A_FUNDING_VICTIM_RULE='tail')
        for key, value in [(None, None), ('A_NATIVE_VICTIM_FULL_RUNNING', 'on'),
                ('A_NATIVE_VICTIM_CURRENT_GUARD', 'on'), ('A_SELF_PREEMPT_CONTINUE', 'on'),
                ('A_NATIVE_CAPACITY_DEFERRAL', 'prefix_work')]:
            environment = dict(baseline)
            if key: environment[key] = value
            namespace = dict(vars(adapter), data={}, oldest_admission_mode='queue_fund', prefix_caching=True,
                allow_forced_rotations=True, store_scope='native_full', ordinary_backfill=False)
            with patch.dict(os.environ, environment):
                if key:
                    with self.assertRaises(ValueError): exec(code, namespace)
                else:
                    exec(code, namespace)
                    self.assertFalse(namespace['partial_once_state']['consumed'])


if __name__ == '__main__':
    unittest.main()
