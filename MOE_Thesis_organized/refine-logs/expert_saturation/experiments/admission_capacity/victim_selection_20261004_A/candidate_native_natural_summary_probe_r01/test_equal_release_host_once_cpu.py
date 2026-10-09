"""Small CPU fixtures for the equal-release host-prefix one-shot choice."""
import ast
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/'pkg'))
import staged_store_rotation as adapter
from test_partial_restore_once_cpu import native_suffix_action


def row(index, missing, *, computed=3272, output=592, release=205):
    return dict(index=index, immediate_releasable_blocks=release, held_blocks=release,
        shared_blocks=0, release_state_error=None, computed_tokens=computed,
        prompt_tokens=computed-output+1, output_tokens=output, qualified=True,
        request_status='RUNNING', host_ready_prefix_blocks=computed//16-missing,
        host_missing_suffix_blocks=missing, host_state_error=None,
        pending_native_store_dependencies=0)


def rows():
    return [row(7, 38), row(8, 38), row(9, 188, computed=3270, output=485)]


def real_pick(rule, active=False):
    install = next(n for n in ast.parse((HERE/'pkg/staged_store_rotation.py').read_text()).body
                   if isinstance(n, ast.FunctionDef) and n.name == 'install')
    pick = deepcopy(next(n for n in install.body if isinstance(n, ast.FunctionDef) and n.name == 'pick_victim'))
    specs = [(16, 4, 19, 2, 0), (2000, 400, 2399, 150, 0),
             (2681, 592, 3272, 205, 166), (2786, 485, 3270, 205, 16)]
    requests = [NS(request_id=str(i), arrival_time=float(i), num_preemptions=0,
                   num_output_tokens=spec[1]) for i, spec in enumerate(specs)]
    owned = {}; blocks = []; states = {}; host = {}
    for req, (prompt, output, computed, held, ready) in zip(requests, specs):
        own = [NS(block_id=len(blocks)+i, ref_cnt=1, is_null=False) for i in range(held)]
        blocks.extend(own); owned[req.request_id] = own
        states[req.request_id] = adapter.RequestState(req.request_id, computed, prompt, output,
            1024, 'RUNNING', tuple(b.block_id for b in own))
        host[req.request_id] = dict(host_ready_prefix_blocks=ready,
            host_missing_suffix_blocks=computed//16-ready, host_state_error=None)
    scheduler = NS(running=requests.copy(), _rotation_full_running_enabled=False)
    state = lambda mode: dict(mode=mode, consumed=False, trigger_count=0,
                               proposal_count=0, action_request_count=0)
    namespace = dict(vars(adapter), scheduler=scheduler, victim_rule=rule, full_running_mode='off',
        protected=requests[0] if active else None, phase=None, current_guard='off', block_size=16,
        pool=NS(blocks=blocks, get_num_free_blocks=lambda: 0), owned=owned,
        residence={r.request_id: (0, 0) for r in requests}, step=12,
        data=dict(victim_decisions=[]), equal_once_state=state('SHADOW'), partial_once_state=state('SHADOW'),
        host_equal_once_state=state('ACTIVE' if rule == 'equal_release_host_once' else 'SHADOW'),
        cs=NS(_req_status={r.request_id: NS(transfer_jobs=set()) for r in requests}, _jobs={}),
        view=lambda r: states[r.request_id], bidkv_score=lambda r: {},
        host_prefix=lambda s,*args: host[s.request_id])
    exec(compile(ast.fix_missing_locations(ast.Module(body=[pick], type_ignores=[])), '<actual-pick-victim>', 'exec'), namespace)
    scheduler._rotation_pick_victim = namespace['pick_victim']
    calls = []
    scheduler._preempt_request = lambda *args: calls.append(args)
    return scheduler, namespace, requests, calls


class EqualReleaseHostOnceTests(unittest.TestCase):
    def test_equal_capacity_floor_strict_improvement_and_nearest_tie(self):
        choose = adapter._equal_release_host_once_choice
        self.assertEqual(choose(rows(), False, False, 16), (8, None, True, 2, 2))
        observed = rows(); observed[0]['host_missing_suffix_blocks'] = 30; observed[0]['host_ready_prefix_blocks'] = 174
        self.assertEqual(choose(observed, False, False, 16), (7, None, True, 2, 2))
        for alt in [row(7, 38, release=206), row(7, 38, computed=3280)]:
            self.assertEqual(choose([alt, rows()[-1]], False, False, 16),
                             (9, 'NO_EQUAL_RELEASE_AND_COMPUTED_PAGES', False, 0, 0))
        self.assertEqual(choose([row(7, 188), rows()[-1]], False, False, 16),
                         (9, 'NO_HOST_PREFIX_IMPROVEMENT', False, 1, 0))

    def test_unknown_phase_pending_and_once(self):
        choose = adapter._equal_release_host_once_choice
        self.assertEqual(choose(rows(), False, True, 16), (9, 'ALREADY_CONSUMED', False, 0, 0))
        self.assertEqual(choose(rows(), True, False, 16), (9, 'ACTIVE_PROTECTION_OR_PHASE', False, 0, 0))
        self.assertEqual(choose(rows(), False, False, None), (9, 'UNKNOWN_BLOCK_SIZE', False, 0, 0))
        for key, reason in [('immediate_releasable_blocks', 'UNKNOWN_PHYSICAL_RELEASE'),
                ('computed_tokens', 'UNKNOWN_PROGRESS'), ('host_missing_suffix_blocks', 'UNKNOWN_HOST_OR_PENDING_STATE'),
                ('pending_native_store_dependencies', 'UNKNOWN_HOST_OR_PENDING_STATE')]:
            observed = rows(); observed[0][key] = None
            self.assertEqual(choose(observed, False, False, 16), (9, reason, False, 0, 0))
        observed = rows(); observed[-1]['pending_native_store_dependencies'] = 1
        self.assertEqual(choose(observed, False, False, 16),
                         (9, 'TAIL_NOT_PRIVATE_DECODE_WITHOUT_PENDING_STORE', False, 0, 0))
        observed = rows(); observed[0]['pending_native_store_dependencies'] = 1
        self.assertEqual(choose(observed, False, False, 16), (8, None, True, 1, 1))
        observed[1]['pending_native_store_dependencies'] = 1
        self.assertEqual(choose(observed, False, False, 16),
                         (9, 'NO_EQUAL_RELEASE_AND_COMPUTED_PAGES', False, 0, 0))
        for change in [dict(qualified=False, prompt_tokens=2700), dict(shared_blocks=1, held_blocks=206)]:
            observed = rows(); observed[0].update(change)
            self.assertEqual(choose(observed, False, False, 16), (8, None, True, 1, 1))

    def test_actual_suffix_pop_calls_original_once_and_consumes_independently(self):
        execute = native_suffix_action()
        for rule, first_index in [('equal_release_host_once', 2), ('tail', 3)]:
            scheduler, namespace, requests, calls = real_pick(rule)
            timestamp = object()
            selected = execute(scheduler, 1, [requests[0]], timestamp)
            self.assertIs(selected, requests[first_index]); self.assertNotIn(selected, scheduler.running)
            self.assertEqual(len(calls), 1); self.assertIs(calls[0][0], selected); self.assertIs(calls[0][1], timestamp)
            decision = namespace['data']['victim_decisions'][0]
            self.assertEqual([r['index'] for r in decision['candidates']], [1, 2, 3])
            proposal = decision['equal_release_host_once']
            self.assertEqual(proposal['proposed_request'], requests[2].request_id)
            self.assertEqual(proposal['returned_request'], selected.request_id)
            self.assertEqual((proposal['tail_releasable_blocks'], proposal['proposed_releasable_blocks']), (205, 205))
            self.assertEqual((proposal['tail_host_missing_blocks'], proposal['proposed_host_missing_blocks']), (188, 38))
            self.assertEqual(proposal['action_requested'], rule == 'equal_release_host_once')
            self.assertEqual((proposal['consumed_before'], proposal['consumed_after']), (False, True))
            self.assertFalse(namespace['partial_once_state']['consumed'])
            tail = scheduler.running[-1]
            execute(scheduler, 1, [requests[0]], timestamp)
            self.assertEqual(len(calls), 2); self.assertIs(calls[-1][0], tail)
            self.assertEqual(namespace['host_equal_once_state']['proposal_count'], 1)
            self.assertEqual(namespace['host_equal_once_state']['action_request_count'], int(rule == 'equal_release_host_once'))
            self.assertEqual(namespace['data']['victim_decisions'][-1]['equal_release_host_once']['fallback'], 'ALREADY_CONSUMED')
        scheduler, namespace, requests, calls = real_pick('equal_release_host_once', active=True)
        self.assertIs(execute(scheduler, 1, [requests[0]], 0.), requests[-1])
        self.assertFalse(namespace['host_equal_once_state']['consumed'])


if __name__ == '__main__':
    unittest.main()
