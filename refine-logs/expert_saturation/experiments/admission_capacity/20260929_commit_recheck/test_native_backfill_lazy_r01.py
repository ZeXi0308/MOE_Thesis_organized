#!/usr/bin/env python3
"""Narrow CPU closure for lazy running views in the timed native-full probe."""

import ast
from pathlib import Path
from types import SimpleNamespace
import sys
import textwrap


HERE = Path(__file__).resolve().parent
ORIGINAL = HERE / "candidate_native_backfill_cpu_r01" / "pkg"
LAZY = HERE / "candidate_native_backfill_lazy_r01" / "pkg"
sys.path.insert(0, str(LAZY))
import staged_store_rotation as staged  # noqa: E402


def exercise_begin(package, *, diagnostic=False, forced=False,
                   open_population=True, ordinary_action=True):
    tree = ast.parse((package / "staged_store_rotation.py").read_text())
    install = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                   and n.name == "install")
    begin = next(n for n in ast.walk(install) if isinstance(n, ast.FunctionDef)
                 and n.name == "begin")
    source = """def exercise():
    step=40
    plan=plan_request_refs=protected=output_start=phase=capacity_plan_choice=None
    followup_pending=followup_choice=ordinary_choice=None
    pending_flush=set();cancelled=False
    cohort=set() if open_population else None
    calls=[];attempts=[];decisions=[]
    req=SimpleNamespace(request_id='running',status=SimpleNamespace(name='RUNNING'))
    scheduler=SimpleNamespace(running=[req],requests={'running':req},
        waiting=[],skipped_waiting=[],_rotation_forced_count=0,_rotation_target=None)
    tracker=SimpleNamespace(absent_since={},absence_count={},resident_since={},
        last_swap_step=0,decide=lambda *args,**kwargs: decisions.append(1) or SimpleNamespace(action='noop'))
    data={'gate_observations':[],'eligibility_snapshots':[],'selector_decisions':[]}
    pool=SimpleNamespace(get_num_free_blocks=lambda:10)
    expected_requests=1
    ordinary_backfill=True
    capacity_victim=False
    def view(request):
        calls.append(request.request_id)
        return SimpleNamespace(request_id=request.request_id,pure_decode=True,
            computed=16,prompt=16,max_output=16,output=1,blocks=())
    def attempt_followup(context):
        raise AssertionError('unexpected followup')
    def attempt_ordinary_backfill():
        attempts.append(1)
        return ordinary_action
    def _eligibility_snapshot(*args):return {}
"""
    source += textwrap.indent(ast.unparse(begin), "    ") + "\n"
    source += "    begin([],0)\n"
    source += "    return dict(views=len(calls),attempts=len(attempts),decisions=len(decisions),phase=phase,cohort_active=cohort is not None,gate_logs=len(data['gate_observations']),snapshots=len(data['eligibility_snapshots']))\n"
    namespace = {"SimpleNamespace": SimpleNamespace,
                 "diagnostic": diagnostic, "allow_forced_rotations": forced,
                 "open_population": open_population,
                 "ordinary_action": ordinary_action,
                 "RequestView": staged.RequestView}
    exec(source, namespace)
    return namespace["exercise"]()


def test_lazy_ordinary_same_action_without_full_view():
    old = exercise_begin(ORIGINAL)
    new = exercise_begin(LAZY)
    assert old == dict(views=1, attempts=1, decisions=0, phase='ordinary_backfill',
                       cohort_active=True, gate_logs=0, snapshots=0)
    assert new == {**old, "views": 0}


def test_existing_fallbacks_keep_rows():
    diagnosed = exercise_begin(LAZY, diagnostic=True)
    assert diagnosed["views"] == 1 and diagnosed["gate_logs"] == 1
    assert diagnosed["snapshots"] == 1
    selected = exercise_begin(LAZY, forced=True, ordinary_action=False)
    assert selected["views"] == 1 and selected["decisions"] == 1
    closed = exercise_begin(LAZY, open_population=False)
    assert closed["views"] == 1 and closed["cohort_active"]


def test_invalid_running_ownership_still_rejected_at_action_boundary():
    target = SimpleNamespace(request_id='target', status=SimpleNamespace(name='PREEMPTED'),
                             num_prompt_tokens=16, num_output_tokens=0,
                             is_finished=lambda: False)
    running = SimpleNamespace(request_id='running')
    scheduler = SimpleNamespace(waiting=[target], skipped_waiting=[], running=[running],
                                max_num_running_reqs=32, num_waiting_for_streaming_input=0)
    block = SimpleNamespace(block_id=7, is_null=False, ref_cnt=1)
    pool = SimpleNamespace(get_num_free_blocks=lambda: 100, blocks={7: object()})
    manager = SimpleNamespace(get_blocks=lambda _rid: SimpleNamespace(
        get_block_ids=lambda: ([7],)))
    cs = SimpleNamespace(_req_status={'target': SimpleNamespace(transfer_jobs=[])})
    reason = staged._direct_resume_reason(
        scheduler, manager, pool, {'running': [block], 'target': []}, cs, target)
    assert reason == 'KEEP_SHARED_OR_INVALID_BLOCK'


if __name__ == '__main__':
    test_lazy_ordinary_same_action_without_full_view()
    test_existing_fallbacks_keep_rows()
    test_invalid_running_ownership_still_rejected_at_action_boundary()
    print('lazy native-full ordinary CPU closure: 3 checks passed')
