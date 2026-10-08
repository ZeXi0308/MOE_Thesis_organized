"""Reuse sealed data for a new execution-share question; no GPU or victim search."""
import argparse
from collections import Counter
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
from recovery_execution_share import Request, State, Protection, allocate, blocks


def extract(before, target_id, *, cap=1024):
    def row(rid):
        r = before['requests'][rid]
        return Request(rid, r['prompt_tokens'] + r['output_tokens'],
                       r['computed_tokens'], sum(r['block_counts']),
                       r['output_tokens'], cap - r['output_tokens'])
    return State(row(target_id), tuple(row(r) for r in before['running_ids'] if r != target_id),
                 before['pool']['free_blocks'])


def conditional_prefix(initial, rule):
    """Independent scalar evolution only to target's first new output.

    No admissions, extra victims or early EOS. A successful call is ASSUMED to
    return one ID for each complete pending input; declared-cap completions
    release owned blocks only AFTER the complete batch's allocation succeeds.
    No observed future raw state supplies any of these transitions.
    """
    state, trace, done = initial, [], False
    protection = Protection(initial.target.request_id, initial.target.outputs)
    lost = Counter()
    for call in range(1, 9):
        plan = allocate(state, rule)
        if plan['status'] != 'READY':
            trace.append(dict(call=call, plan=plan))
            break
        F = plan['free_after_allocations']
        updated, completed = {}, []
        for r in (state.target,) + state.peers:
            n = plan['scheduled'].get(r.request_id, 0)
            computed = r.computed + n
            owned = max(r.allocated, blocks(computed, state.block_size))
            output = int(n > 0 and computed == r.history)
            new = replace(r, computed=computed, allocated=owned, history=r.history + output,
                          outputs=r.outputs + output, remaining_cap=r.remaining_cap - output)
            if new.remaining_cap == 0:
                completed.append(r.request_id)
                F += owned
            else:
                updated[r.request_id] = new
        lost.update(plan['held'].keys())
        before_capacity = state.free_blocks + sum(r.allocated for r in (state.target,) + state.peers)
        assert F + sum(r.allocated for r in updated.values()) == before_capacity
        target = updated.get(state.target.request_id)
        done = protection.released(observed_outputs=target.outputs if target else state.target.outputs + 1,
                                   completed=target is None)
        trace.append(dict(call=call, plan=plan, completed_after_return=completed,
                          free_after_return=F, target_output_observed_assumed=done))
        if done:
            break
        state = replace(state, target=target, peers=tuple(updated[r.request_id]
                        for r in state.peers if r.request_id in updated), free_blocks=F)
    return dict(rule=rule, trace=trace, calls_to_assumed_output=len(trace) if done else None,
                held_peer_opportunities=dict(lost), total_held_peer_opportunities=sum(lost.values()),
                boundary='Conditional scalar execution, no early EOS/admissions/extra victims or latency.')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--funding-bundle', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    all_rows, certificate, skipped = [], None, Counter()
    for folder in sorted((args.funding_bundle/'execution/readback/results').iterdir()):
        rawpath = folder/'raw.json'
        if not rawpath.exists():
            continue
        data = rawpath.read_bytes()
        raw = json.loads(data)
        decisions = json.loads((folder/'headroom-decisions.json').read_text())
        selected, changed, matched = [], [], 0
        for d in decisions:
            target_id = d.get('recovery_target')
            if not target_id:
                continue
            if d.get('forced_preempted'):
                skipped['first_forced_call_pre_release_state'] += 1
                continue
            before = raw['memory_trace'][d['step']]['before']
            state = extract(before, target_id)
            old, new = allocate(state), allocate(state, 'preserve_calls')
            # Actual map is compared only after both current-state plans exist.
            actual = d['actual_scheduled']
            matched += int(old['scheduled'] == actual)
            row = dict(step=d['step'], target_source=raw['internal_to_source'][target_id],
                       pending=old['target_pending'], min_calls=old['compute_only_min_calls'],
                       ready_first_positions=old['target_positions'],
                       preserve_calls_positions=new['target_positions'],
                       peer_opportunities_before=old['peer_output_opportunities'],
                       peer_opportunities_after=new['peer_output_opportunities'],
                       actual_ready_first_matches=old['scheduled'] == actual)
            selected.append(row)
            if old['scheduled'] != new['scheduled']:
                # One local current-state counterfactual, not a repeated selector search.
                row['conditional_ready_first'] = conditional_prefix(state, 'ready_first')
                row['conditional_preserve_calls'] = conditional_prefix(state, 'preserve_calls')
                changed.append(row)
        all_rows.append(dict(cell=folder.name, raw_sha256=hashlib.sha256(data).hexdigest(),
                             observed_protected_calls=len(selected), ready_first_matches=matched,
                             changed_calls=len(changed), changes=changed))
        if folder.name == 'funding-block0-least_feasible':
            target = next(r for r, src in raw['internal_to_source'].items() if src.endswith('0020902'))
            state = extract(raw['memory_trace'][1026]['before'], target)
            certificate = dict(causal_cutoff='memory_trace[1026].before', input_state=asdict(state),
                               chosen_target_source=raw['internal_to_source'][target],
                               modes=[conditional_prefix(state, rule) for rule in
                                      ('restore_first', 'ready_first', 'preserve_calls')])
    result = dict(status='CPU_EXECUTION_MODEL_ONLY', cells=all_rows, skipped=dict(skipped),
                  fixed_target_1026=certificate,
                  totals=dict(observed_protected_calls=sum(r['observed_protected_calls'] for r in all_rows),
                              exact_existing_plan_matches=sum(r['ready_first_matches'] for r in all_rows),
                              changed_calls=sum(r['changed_calls'] for r in all_rows)),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  allocator_sha256=hashlib.sha256(Path(__file__).with_name('recovery_execution_share.py').read_bytes()).hexdigest(),
                  scope='Reused six sealed executions. No new GPU, timestamps, performance, victim choice or EOS prediction.')
    args.output_dir.mkdir(parents=True)
    (args.output_dir/'analysis.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'], totals=result['totals'], cells=[{
        k:v for k,v in r.items() if k != 'changes'} for r in all_rows], fixed_target=[{
        'rule':r['rule'], 'calls':r['calls_to_assumed_output'], 'held':r['total_held_peer_opportunities']
        } for r in certificate['modes']])))


if __name__ == '__main__':
    main()
