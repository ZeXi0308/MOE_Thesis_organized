"""Branch real resident cold-recovery states through native CPU schedule methods.

Current ready-first versus the existing preserve-calls component, for the same
compute-only minimum-call window. Native allocation/output objects are fakes;
store/DMA, new admissions, upper-layer swaps and early EOS are not simulated.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import MethodType
from unittest.mock import patch

import recovery_execution_share as share
import rotation_native as adapter
import verify_native_recovery_execution as fixture
import recovery_compute_share as compute_hook


def run(source, before, target, rule, horizon, runtime_hook=False):
    with patch.object(adapter, 'install', return_value=([], lambda: None)):
        s, _, _ = fixture.create(source, deepcopy(before), [], False)
    ns = dict(s.schedule.__func__.__globals__)
    exec(compile(adapter.patched_schedule_tree(source.read_text()), str(source), 'exec'), ns)
    s.schedule = MethodType(ns['schedule'], s)
    plan = {}; expected = {}
    def view(r):
        return share.Request(r.request_id, r.num_tokens, r.num_computed_tokens,
            len(s.owned[r.request_id]), r.num_output_tokens, r.max_tokens-r.num_output_tokens)
    def begin(preempted, timestamp):
        nonlocal plan, expected
        state = share.State(view(s.requests[target]), tuple(view(r) for r in s.running
            if r.request_id != target), len(s.available))
        plan = share.allocate(state, 'ready_first' if runtime_hook else rule)
        expected = share.allocate(state, rule)
        assert plan['status'] == 'READY'
        s._rotation_target, s._rotation_forced_count = target, 0
    s._rotation_begin, s._rotation_hold = begin, lambda r: r.request_id in plan['held']
    s._rotation_target = target
    hook_data, uninstall = compute_hook.install(s, enabled=runtime_hook and rule=='preserve_calls', diagnostic=True)
    totals = {rid: 0 for rid in before['running_ids']}
    trace = []
    for k in range(horizon):
        result = s.schedule()
        assert not result.preempted_req_ids and result.num_scheduled_tokens == expected['scheduled']
        assert len(s.available) == expected['free_after_allocations']
        assert sum(map(len, s.owned.values()))+len(s.available) == before['pool']['usable_blocks']
        outputs, completed = fixture.returned(s, result)
        assert not completed  # This exact conditional slice stops before any cap/EOS.
        for rid in outputs: totals[rid] += 1
        trace.append(dict(call=k+1, scheduled=result.num_scheduled_tokens,
            held=expected['held'], returned_output_ids=outputs, free_after=len(s.available)))
        if target in outputs:
            assert k == horizon-1  # Boundary only; no unmodelled post-release steps.
    uninstall()
    return dict(rule=rule, calls=trace, output_opportunities=totals, hook=hook_data,
        end_state=dict(free_blocks=len(s.available), requests={rid: dict(
            history=r.num_tokens, computed=r.num_computed_tokens,
            outputs=r.num_output_tokens, allocated=len(s.owned[rid]), status=r.status.name)
            for rid, r in s.requests.items()}),
        target_first_output_call=next((r['call'] for r in trace if target in r['returned_output_ids']), None),
        target_pending_at_end=s.requests[target].num_tokens-s.requests[target].num_computed_tokens)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--diagnosis', type=Path, required=True)
    p.add_argument('--service-bundle', type=Path,
        default=Path('refine-logs/expert_saturation/outputs/admission_capacity/20260915_natural_saved_recovery_gate_r01'))
    p.add_argument('--label', default='diagnostic-current')
    p.add_argument('--scheduler-source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--runtime-hook', action='store_true')
    p.add_argument('--include-restore-first', action='store_true',
        help='Also execute the existing restore-first plan through native hold; no new runtime hook')
    p.add_argument('--branch-start', choices=('changed_call', 'first_resident'), default='changed_call',
        help='First resident recompute includes unchanged initial calls for fair component comparison')
    a = p.parse_args()
    assert hashlib.sha256(a.scheduler_source.read_bytes()).hexdigest() == adapter.SCHEDULER_SHA256
    rawpath = a.repo/a.service_bundle/'execution_weste_26862/readback/results'/a.label/'raw.json'
    raw, diagnosis = json.loads(rawpath.read_text()), json.loads(a.diagnosis.read_text())
    inverse = {alias: rid for rid, alias in raw['internal_to_source'].items()}
    cases, selection = [], []
    for episode in diagnosis['cells'][0]['recoveries']:
        if a.branch_start == 'first_resident':
            rows = [row for row in episode['calls'] if row.get('resident_model_exact')
                    and row['remaining_positions'] > 1][:1]
        else:
            rows = [row for row in episode['calls'] if row.get('preserve_calls_changes_map')]
        selection.append(dict(target=episode['target'], commit_step=episode['commit_step'],
                              selected_steps=[row['step'] for row in rows]))
        for row in rows:
            step, target, horizon = row['step'], inverse[episode['target']], row['compute_only_min_calls']
            before = raw['memory_trace'][step]['before']
            rules = ('ready_first', 'preserve_calls') + (('restore_first',) if a.include_restore_first else ())
            branches = {rule: run(a.scheduler_source, before, target, rule, horizon,
                                 a.runtime_hook and rule != 'restore_first') for rule in rules}
            old, new = branches['ready_first'], branches['preserve_calls']
            # Factual prefix check only; candidate never reads this continuation.
            for k, call in enumerate(old['calls']):
                factual = {r['internal_request_id']: r['scheduled_tokens']
                           for r in raw['scheduler_steps'][step+k]['scheduled']}
                assert call['scheduled'] == factual
            peer_loss = sum(old['output_opportunities'].values())-old['output_opportunities'][target]-(
                        sum(new['output_opportunities'].values())-new['output_opportunities'][target])
            lower = max(0, row['remaining_positions']+horizon*row['peer_count']-horizon*row['budget'])
            assert peer_loss == lower and new['target_first_output_call'] == horizon
            if a.branch_start == 'changed_call':
                assert old['target_first_output_call'] is None
            comparison = None
            if a.include_restore_first:
                simple = branches['restore_first']
                comparison = dict(
                    same_target_first_output_call=simple['target_first_output_call']==new['target_first_output_call'],
                    same_output_counts=simple['output_opportunities']==new['output_opportunities'],
                    same_progress_and_block_counts=simple['end_state']==new['end_state'],
                    same_scheduled_maps=[c['scheduled'] for c in simple['calls']]==[
                        c['scheduled'] for c in new['calls']],
                    peer_output_ids_by_call={rule: [[rid for rid in c['returned_output_ids'] if rid != target]
                        for c in branches[rule]['calls']] for rule in ('preserve_calls', 'restore_first')},
                    same_wall_time_or_kv_contents='NOT_MEASURED')
            cases.append(dict(step=step, target=target, horizon=horizon,
                minimum_peer_opportunity_cost=lower, achieved_peer_opportunity_cost=peer_loss,
                per_request_delta={rid: new['output_opportunities'][rid]-count
                    for rid, count in old['output_opportunities'].items()},
                restore_first_comparison=comparison, branches=branches))
    paths = [rawpath, a.diagnosis, a.scheduler_source, Path(__file__), Path(share.__file__),
             Path(adapter.__file__), Path(fixture.__file__), Path(compute_hook.__file__)]
    result = dict(status='CPU_NATIVE_METHOD_BRANCHES', branch_start=a.branch_start,
        selection=selection, cases=cases,
        sources={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        scope='All selected resident calls in the supplied diagnosis, using the explicit branch-start rule. '
        'Selection includes every episode and its eligible start or empty list; no filtering by branch outcomes. '
        'Repeated recoveries and reuse of raw are not independent replicates. '
        'No future EOS/route, transfer timing, alternative native-full saving, or full-service gain. '
        'Actual baseline scheduled maps match; candidate has its own C/A/output evolution and CPU placeholder outputs.')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x') as f: json.dump(result, f, indent=2); f.write('\n')
    print(json.dumps([dict(step=c['step'], horizon=c['horizon'], peer_cost=c['achieved_peer_opportunity_cost']) for c in cases]))


if __name__ == '__main__':
    main()
