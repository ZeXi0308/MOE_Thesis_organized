"""Risk-directed helper, real suffix hook and unchanged native preempt-call checks."""
import ast
from copy import deepcopy
import os
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/'pkg'))
import staged_store_rotation as adapter
from rotation_native import patched_schedule_tree

PINNED = HERE.parent.parent/'20260929_commit_recheck/liveness_pinned_sources_20260930/scheduler.py'


def row(index, arrival, qualified=True):
    # Deliberately no held/output/host/request identity/recovery-history signal.
    return dict(index=index, arrival_time=arrival, qualified=qualified)


def real_pick(rule='arrival_once', active=False):
    install = next(n for n in ast.parse((HERE/'pkg/staged_store_rotation.py').read_text()).body
                   if isinstance(n, ast.FunctionDef) and n.name == 'install')
    pick = deepcopy(next(n for n in install.body if isinstance(n, ast.FunctionDef) and n.name == 'pick_victim'))
    requests = [NS(request_id=str(i), arrival_time=arrival, num_preemptions=0,
                   num_output_tokens=4) for i, arrival in enumerate((99., 2., 5., 3.))]
    scheduler = NS(running=requests.copy(), _rotation_full_running_enabled=False)
    residence = {r.request_id: (0, 0) for r in requests}
    state = lambda mode: dict(mode=mode, consumed=False, proposal_count=0, action_request_count=0)
    namespace = dict(vars(adapter), scheduler=scheduler, victim_rule=rule, full_running_mode='off',
        protected=requests[0] if active else None, phase=None, current_guard='off', block_size=16,
        pool=NS(get_num_free_blocks=lambda: 0), residence=residence, step=12,
        data=dict(victim_decisions=[]), equal_once_state=state('SHADOW'),
        arrival_once_state=state('ACTIVE' if rule == 'arrival_once' else 'SHADOW'),
        cs=NS(_req_status={r.request_id: NS(transfer_jobs=set()) for r in requests}, _jobs={}),
        view=lambda r: NS(pure_decode=True, status='RUNNING', blocks=(1,2), computed=19, output=4, max_output=100),
        bidkv_score=lambda r: {}, host_prefix=lambda *args: {})
    exec(compile(ast.fix_missing_locations(ast.Module(body=[pick], type_ignores=[])), '<actual-pick-victim>', 'exec'), namespace)
    scheduler._rotation_pick_victim = namespace['pick_victim']
    calls = []
    scheduler._preempt_request = lambda *args: calls.append(args)
    return scheduler, namespace, requests, calls


def native_suffix_action():
    """Actual patched FCFS pop branch followed by the original native call AST."""
    tree = patched_schedule_tree(PINNED.read_text())
    branch = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                  and ast.unparse(n.test) == 'self.policy == SchedulingPolicy.PRIORITY')
    preempt = next(n for n in ast.walk(tree) if isinstance(n, ast.Expr)
                   and ast.unparse(n) == 'self._preempt_request(preempted_req, scheduled_timestamp)')
    function = ast.parse('''def execute(self, req_index, scheduled_running_reqs, scheduled_timestamp):
    return None
''').body[0]
    function.body = deepcopy(branch.orelse) + [deepcopy(preempt), ast.Return(value=ast.Name(id='preempted_req', ctx=ast.Load()))]
    namespace = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), '<native-fcfs-preempt>', 'exec'), namespace)
    return namespace['execute']


class ArrivalOnceTests(unittest.TestCase):
    def test_once_phase_unknown_finite_and_tail_ties(self):
        choose = adapter._arrival_once_choice
        rows = [row(4, 5.), row(5, 5.), row(6, 3.)]
        self.assertEqual(choose(rows, False, False, False), (5, None, True))
        self.assertEqual(choose(rows, False, False, True), (6, 'ALREADY_CONSUMED', True))
        self.assertEqual(choose(rows, False, True, False), (6, 'ACTIVE_PROTECTION_OR_PHASE', False))
        self.assertEqual(choose(rows, True, False, False), (6, 'UNKNOWN_SUFFIX_STATE', False))
        bad = deepcopy(rows); bad[0]['qualified'] = False
        self.assertEqual(choose(bad, False, False, False), (6, 'UNKNOWN_SUFFIX_STATE', False))
        for value in (None, float('nan'), float('inf'), -float('inf'), True, '5'):
            bad = deepcopy(rows); bad[0]['arrival_time'] = value
            self.assertEqual(choose(bad, False, False, False), (6, 'UNKNOWN_ORIGINAL_ARRIVAL', False))
        rows[-1]['arrival_time'] = 5.
        self.assertEqual(choose(rows, False, False, False), (6, 'TAIL_IS_LATEST_ORIGINAL_ARRIVAL', False))
        self.assertEqual(choose([rows[-1]], False, False, False), (6, 'TAIL_IS_LATEST_ORIGINAL_ARRIVAL', False))

    def test_actual_hook_preserves_suffix_native_call_and_shadow_semantics(self):
        execute = native_suffix_action()
        for rule, first_index in (('arrival_once', 2), ('tail', 3)):
            scheduler, namespace, requests, calls = real_pick(rule)
            stamp = object()
            picked = execute(scheduler, 1, [requests[0]], stamp)
            self.assertIs(picked, requests[first_index])
            self.assertEqual(len(calls), 1)
            self.assertIs(calls[0][0], picked)
            self.assertIs(calls[0][1], stamp)
            decision = namespace['data']['victim_decisions'][0]
            self.assertEqual([r['index'] for r in decision['candidates']], [1,2,3])
            self.assertEqual(decision['arrival_once']['proposed_request'], requests[2].request_id)
            self.assertEqual(decision['arrival_once']['action_requested'], rule == 'arrival_once')
            self.assertEqual((decision['arrival_once']['consumed_before'], decision['arrival_once']['consumed_after']), (False, True))
            # A later differently ranked suffix cannot request a second action.
            last = scheduler.running[-1]
            execute(scheduler, 1, [requests[0]], stamp)
            self.assertIs(calls[-1][0], last)
            self.assertEqual(namespace['arrival_once_state']['proposal_count'], 1)
            self.assertEqual(namespace['arrival_once_state']['action_request_count'], int(rule == 'arrival_once'))
            self.assertEqual(namespace['data']['victim_decisions'][-1]['arrival_once']['fallback'], 'ALREADY_CONSUMED')
        scheduler, namespace, requests, _ = real_pick(active=True)
        self.assertIs(execute(scheduler, 1, [requests[0]], 0.), requests[-1])
        self.assertFalse(namespace['arrival_once_state']['consumed'])

    def test_real_combination_guards_reject_other_controllers(self):
        install = next(n for n in ast.parse((HERE/'pkg/staged_store_rotation.py').read_text()).body
                       if isinstance(n, ast.FunctionDef) and n.name == 'install')
        start = next(i for i,n in enumerate(install.body) if isinstance(n, ast.Assign)
                     and any(isinstance(t,ast.Name) and t.id == 'victim_rule' for t in n.targets))
        end = next(i for i,n in enumerate(install.body) if isinstance(n, ast.Assign)
                   and ast.unparse(n.targets[0]) == "data['capacity_deferral_mode']")
        code = compile(ast.Module(body=deepcopy(install.body[start:end]), type_ignores=[]), '<actual-probe-guards>', 'exec')
        baseline = dict(A_NATIVE_VICTIM_RULE='arrival_once', A_NATIVE_VICTIM_FULL_RUNNING='off',
            A_NATIVE_VICTIM_CURRENT_GUARD='off', A_SELF_PREEMPT_CONTINUE='off',
            A_NATIVE_CAPACITY_DEFERRAL='off', A_FUNDING_VICTIM_RULE='tail')
        for key, value in [(None,None), ('A_NATIVE_VICTIM_FULL_RUNNING','on'),
                ('A_NATIVE_VICTIM_CURRENT_GUARD','on'), ('A_SELF_PREEMPT_CONTINUE','on'),
                ('A_NATIVE_CAPACITY_DEFERRAL','prefix_work')]:
            environment = dict(baseline)
            if key: environment[key] = value
            namespace = dict(vars(adapter), data={}, oldest_admission_mode='queue_fund',
                allow_forced_rotations=True, store_scope='native_full', ordinary_backfill=False)
            with patch.dict(os.environ, environment):
                if key:
                    with self.assertRaises(ValueError): exec(code, namespace)
                else:
                    exec(code, namespace)
                    self.assertFalse(namespace['arrival_once_state']['consumed'])


if __name__ == '__main__':
    unittest.main()
