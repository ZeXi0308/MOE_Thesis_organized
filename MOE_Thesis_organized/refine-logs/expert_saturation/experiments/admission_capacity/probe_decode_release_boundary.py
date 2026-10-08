"""Decode allocation alternatives from every native re-preemption after recovery.

Reuse an owner's full analysis; no repeated lifecycle/performance calculation.
The branches generate their own C/A/output state, with no early EOS, arrivals,
block reclamation, or external releases. Stop before crediting completion frees.
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

from recovery_execution_share import (Request, State, blocks, allocate,
    decode_release_envelope, allocate_completion_bridge)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def branch(initial, rule):
    state = initial
    calls = []
    totals = {r.request_id: 0 for r in (state.target,) + state.peers}
    initial_total = state.free_blocks + sum(r.allocated for r in (state.target,) + state.peers)
    finisher = None
    for _ in range(initial.target.remaining_cap+1):
        if rule == 'cap_bridge':
            plan = allocate_completion_bridge(state, finisher)
            finisher = plan.get('finisher_id', finisher)
        else:
            plan = allocate(state)
        if plan['status'] != 'READY':
            return dict(rule=rule, status=plan['status'], calls=calls,
                        new_output_opportunities=totals, completed_at_bound=[], final_free=state.free_blocks)
        F = state.free_blocks
        updated, complete = [], []
        for r in (state.target,) + state.peers:
            n = plan['scheduled'].get(r.request_id, 0)
            assert n in (0, 1)
            need = max(0, blocks(r.computed+n, state.block_size)-r.allocated)
            F -= need
            nr = replace(r, computed=r.computed+n, history=r.history+n,
                         outputs=r.outputs+n, remaining_cap=r.remaining_cap-n,
                         allocated=r.allocated+need)
            updated.append(nr); totals[r.request_id] += n
            if not nr.remaining_cap:
                complete.append(nr.request_id)
        assert F >= 0 and F+sum(r.allocated for r in updated) == initial_total
        assert sum(plan['scheduled'].values()) <= state.token_budget
        calls.append(dict(call=len(calls)+1, scheduled=plan['scheduled'], held=plan['held'],
                          allocated_blocks=state.free_blocks-F, free_after=F,
                          reserve_after=plan.get('remaining_reserved_blocks', plan.get('remaining_escrow_blocks'))))
        if complete:
            return dict(rule=rule, status='CAP_OUTPUT_BOUNDARY_NOT_PHYSICAL_RELEASE', calls=calls,
                        new_output_opportunities=totals, completed_at_bound=complete, final_free=F,
                        finishing_owned_blocks={r.request_id:r.allocated for r in updated if r.request_id in complete})
        state = replace(state, target=updated[0], peers=tuple(updated[1:]), free_blocks=F)
    raise AssertionError('Target made one output per call but failed to reach its declared cap')


def inspect(repo, bundle, label):
    folder = repo/bundle/'execution_weste_26862'
    analysis_path = folder/'analysis.json'
    cell = folder/'readback/results'/label
    analysis = json.loads(analysis_path.read_text())
    assert analysis['status'] == 'COMPLETE'
    assert analysis['native_full_qualification'] == 'NATIVE_FULL_INCREMENTAL_SCOPE_QUALIFIED'
    raw = json.loads((cell/'raw.json').read_text())
    selective = json.loads((cell/'selective-store.json').read_text())
    assert raw['status'] == 'COMPLETE'
    alias = raw['internal_to_source']
    requests = {r['request_id']:r for r in raw['requests']}
    snaps = {r['step']:r for r in selective['eligibility_snapshots']}
    forced = {(r['step'],r['victim']) for r in selective['events']
              if r['event']=='commit_check' and r['reason']=='READY'}
    rows, excluded = [], []
    for segment in analysis['diagnostic']['segments']:
        if segment['end'] != 'repreempted':
            continue
        step, target = segment['next_preempt_step'], segment['internal_request_id']
        if (step,target) in forced:
            excluded.append(dict(step=step, target=alias[target], reason='UPPER_LAYER_FORCED_VICTIM'))
            continue
        before = raw['memory_trace'][step]['before']
        active = before['running_ids']
        if target not in active or any(before['requests'][rid]['computed_tokens'] !=
                before['requests'][rid]['prompt_tokens']+before['requests'][rid]['output_tokens']-1
                or not before['requests'][rid]['output_tokens'] for rid in active):
            excluded.append(dict(step=step, target=alias[target], reason='OUTSIDE_PURE_RESIDENT_DECODE'))
            continue
        assert snaps[step]['protected_id'] is None and snaps[step]['plan_target'] is None
        assert all(snaps[step]['requests'][rid]['status']=='RUNNING' for rid in active)
        def view(rid):
            r = before['requests'][rid]; request = requests[alias[rid]]
            # No model-length termination earlier than the declared cap in this input.
            assert r['prompt_tokens']+request['max_output_tokens'] <= analysis['config']['max_model_len']
            return Request(alias[rid], r['prompt_tokens']+r['output_tokens'], r['computed_tokens'],
                sum(r['block_counts']), r['output_tokens'], request['max_output_tokens']-r['output_tokens'])
        state = State(view(target), tuple(view(rid) for rid in active if rid != target), before['pool']['free_blocks'])
        assert sum(r.allocated for r in (state.target,)+state.peers)+state.free_blocks == before['pool']['usable_blocks']
        envelope = decode_release_envelope(state)
        old, proposed = branch(state, 'existing_history_escrow'), branch(state, 'cap_bridge')
        actual = raw['scheduler_steps'][step]
        events = [p for p in raw['preemption_events'] if p['attempted_step']==step
                  and p['victim_internal_request_id']==target]
        assert len(events)==1 and events[0]['original_preemption_returned']
        rows.append(dict(step=step,target=alias[target], prior_service_outputs=segment['useful_outputs'],
            free_before=state.free_blocks,resident_count=len(active),
            target_stop_reason_observed=requests[alias[target]]['stop_reason'],
            actual_next_call=dict(target_preempted=True,
                scheduled={r['request_id']:r['scheduled_tokens'] for r in actual['scheduled']}),
            envelope=envelope, existing_history_escrow=old, cap_bridge=proposed))
    summary=dict(eligible_states=len(rows),excluded_states=len(excluded),
        own_next_decode_feasible=sum(r['envelope']['target_next_decode_feasible'] for r in rows),
        whole_resident_call_feasible=sum(r['envelope']['all_resident_next_decode_feasible'] for r in rows),
        no_early_eos_first_completion_impossible=sum(not r['envelope']['no_early_eos_first_completion_possible'] for r in rows),
        cap_bridge_reaches_cap=sum(r['cap_bridge']['status']=='CAP_OUTPUT_BOUNDARY_NOT_PHYSICAL_RELEASE' for r in rows))
    paths=[analysis_path,cell/'raw.json',cell/'selective-store.json',Path(__file__),Path(__file__).with_name('recovery_execution_share.py')]
    sources={str(p.relative_to(repo)) if p.is_relative_to(repo) else 'local_source/'+p.name:digest(p) for p in paths}
    return dict(status='STRUCTURAL_FROM_NATIVE_STATES',summary=summary,states=rows,excluded=excluded,sources=sources,
        scope='All native re-preemptions of previously recovered requests in the supplied owner analysis. '
              'No service-output threshold or identity selection. Existing held blocks are not reclaimed before completion. '
              'Independent geometry branches; no borrowed future EOS, trace continuation or physical release. '
              'Cap is a valid upper bound, not known actual remaining length. No latency/quality/performance claim; '
              'native full-store pin/flush and output return must retire before completed blocks are reused.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--label',default='diagnostic-native-full')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    result=inspect(args.repo,args.bundle,args.label)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:
        json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(result['summary'],ensure_ascii=False))


if __name__=='__main__':
    main()
