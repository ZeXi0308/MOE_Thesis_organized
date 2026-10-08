#!/usr/bin/env python3
"""Observed recovery work -> engine-returned output -> actual state invalidation.

Offline accounting only. No replayed policy, future EOS input, or time saving
estimate. A discarded KV prefix may already have supported useful output.
"""
import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path


def read(path):
    with (gzip.open(path, 'rt') if path.suffix == '.gz' else path.open()) as f:
        return json.load(f)


def check(ok, message):
    if not ok:
        raise ValueError(message)


def summarize(rows):
    """Disjoint outcome buckets; repeated work is an overlapping diagnostic."""
    totals = Counter(recompute_positions=0)
    for r in rows:
        if not r['resumed']:
            bucket = 'not_resumed'
        elif r['new_outputs'] == 0:
            bucket = 'discard_before_output' if r['end_reason'] == 'repreempted' else 'censored_no_output'
        elif r['end_reason'] == 'repreempted':
            bucket = 'served_then_discarded'
        elif r['end_reason'] == 'completed':
            bucket = 'served_to_completion'
        else:
            bucket = 'censored_after_output'
        r['outcome_bucket'] = bucket
        totals[bucket + '_residencies'] += 1
        totals[bucket + '_recompute_positions'] += r['recompute_positions']
        totals['recompute_positions'] += r['recompute_positions']
        totals['new_outputs'] += r['new_outputs']
        totals['recompute_only_calls'] += r['recompute_only_calls']
        totals['partial_recovery_chains_with_retained_prefix'] += r['partial_prefix_reused_in_later_call']
    totals['residencies'] = len(rows)
    totals['short_served_then_discarded'] = sum(r['resumed'] and r['end_reason'] == 'repreempted' and 0 < r['new_outputs'] <= 2 for r in rows)
    totals['recompute_for_short_served_then_discarded'] = sum(r['recompute_positions'] for r in rows if r['resumed'] and r['end_reason'] == 'repreempted' and 0 < r['new_outputs'] <= 2)
    totals['observed_recompute_prefix_reexecuted_next_residency'] = sum(r.get('recompute_prefix_reexecuted_next_residency', 0) for r in rows)
    return dict(totals)


def analyze(raw):
    check(raw['status'] == 'COMPLETE' and raw['error'] is None, 'requires a complete raw capture')
    requests = {r['request_id']: r for r in raw['requests']}
    aliases = raw['internal_to_source']
    steps, calls, memory = raw['scheduler_steps'], raw['engine_steps'], raw['memory_trace']
    check(len(steps) == len(calls) == len(memory), 'one native call per scheduler step required')
    for k, (s, c, m) in enumerate(zip(steps, calls, memory)):
        # Older scheduler-only telemetry leaves this field None. The paired
        # successful synchronous engine return is the execution receipt.
        check(s['step'] == k == c['scheduler_step_start'] and c['scheduler_step_end'] == k+1
              and c['completed'] and m['schedule_completed']
              and m['model_execution_confirmed'] is not False, 'unconfirmed execution or call alignment')
        check(c['start_s'] <= s['start_s'] <= s['end_s'] <= c['returned_s'], 'invalid clock nesting')
    selections, events = defaultdict(list), defaultdict(list)
    for s in steps:
        for x in s['scheduled']:
            check(x['computed_adjustment'] == 0 and x['scheduled_start_computed'] == x['computed_before'], 'cache loading needs a different lifecycle adapter')
            check(x['computed_after'] - x['computed_before'] == x['scheduled_tokens'], 'computed work does not conserve')
            selections[x['request_id']].append((s['step'], x))
    for e in raw['preemption_events']:
        check(e['original_preemption_returned'] and e['victim_state_after']['computed_tokens'] == 0
              and e['victim_state_after']['block_counts'] == [0], 'expected actual recompute-only invalidation')
        check(e['output_token_ids_before'] == e['output_token_ids_after'], 'preemption changed returned history')
        events[aliases[e['victim_internal_request_id']]].append(e)
    rows = []
    for rid, evs in events.items():
        q = requests[rid]
        check(evs == sorted(evs, key=lambda e:e['attempted_step']), 'preemptions out of order')
        per = []
        for index, e in enumerate(evs):
            following = evs[index+1] if index+1 < len(evs) else None
            end = following['attempted_step'] if following else len(steps)
            selected = [(k,x) for k,x in selections[rid] if e['attempted_step'] <= k < end]
            n = e['victim_state']['output_tokens']
            end_n = following['victim_state']['output_tokens'] if following else len(q['output_token_ids'])
            check(end_n >= n and (selected or n == end_n), 'output without executed residency')
            recompute = sum(x['recompute_tokens'] for _,x in selected)
            # APC and connector are disabled in this adapter: every recovery is
            # a contiguous prefix starting at zero, retained across calls.
            previous = 0
            carried = False
            for k,x in selected:
                check(x['computed_before'] == previous, 'unobserved loss/load within residency')
                carried |= previous > 0 and x['recompute_tokens'] > 0
                previous = x['computed_after']
                check(x['output_tokens_before'] == bisect_right(q['token_times_s'], calls[k]['start_s']), 'future output leaked into pre-call state')
            if following and selected:
                check(previous == following['victim_state']['computed_tokens'], 'discard receipt differs from executed state')
            output_s = q['token_times_s'][n] if end_n > n else None
            first_call = next((k for k,x in selected if calls[k]['returned_s'] == output_s), None)
            check(output_s is None or first_call is not None, 'new output lacks matching engine return')
            last_output_s = q['token_times_s'][n-1] if n else None
            recovery_calls = [k for k,x in selected if x['recompute_tokens'] > 0]
            row = dict(request_id=rid, preempt_step=e['attempted_step'],
                       end_step=end if following else None,
                       end_reason='repreempted' if following else ('completed' if q['status']=='completed' else 'capture_end'),
                       resumed=bool(selected), first_service_step=selected[0][0] if selected else None,
                       first_output_step=first_call, outputs_before=n, new_outputs=end_n-n,
                       recompute_positions=recompute, executed_positions=sum(x['scheduled_tokens'] for _,x in selected),
                       recompute_only_calls=sum(x['recompute_tokens']==x['scheduled_tokens'] for _,x in selected),
                       partial_prefix_reused_in_later_call=bool(carried),
                       invalidated_computed_prefix_positions=following['victim_state']['computed_tokens'] if following and selected else 0,
                       first_new_output_engine_return_s=output_s,
                       output_age_at_recovery_start_s=calls[selected[0][0]]['start_s']-last_output_s if selected and last_output_s is not None else None,
                       last_output_to_first_new_output_s=output_s-last_output_s if output_s is not None and last_output_s is not None else None,
                       recovery_start_to_first_output_s=output_s-calls[selected[0][0]]['start_s'] if output_s is not None else None,
                       recovery_call_indices=recovery_calls,
                       recovery_call_inclusive_s=sum(calls[k]['returned_s']-calls[k]['start_s'] for k in recovery_calls),
                       recompute_positions_per_new_output=recompute/(end_n-n) if end_n>n else None)
            per.append(row)
        for a,b in zip(per,per[1:]):
            # These are positions of this residency's recovery prefix actually
            # re-executed in the next residency, not all future wasted work.
            a['recompute_prefix_reexecuted_next_residency'] = min(a['recompute_positions'], b['recompute_positions'])
        rows.extend(per)
    total_recompute = sum(s['recompute_tokens'] for s in steps)
    summary = summarize(rows)
    check(summary['recompute_positions'] == total_recompute, 'residencies fail to partition all recomputation')
    summary.update(all_output_tokens=sum(len(q['output_token_ids']) for q in requests.values()),
                   requests=len(requests), completed_requests=sum(q['status']=='completed' for q in requests.values()),
                   preemptions=len(raw['preemption_events']))
    hist = Counter(str(r['new_outputs']) for r in rows if r['resumed'] and r['end_reason']=='repreempted')
    return dict(summary=summary, closed_service_histogram=dict(hist), residencies=rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--campaigns', nargs='+', default=['20260914_ltr_packing_r01', '20260914_restore_completion_r01'])
    p.add_argument('--expected-cells', type=int, default=8)
    args = p.parse_args()
    check(not args.output_dir.exists(), 'output exists; do not overwrite analysis')
    base = args.workspace/'refine-logs/expert_saturation/outputs/admission_capacity'
    cells = []
    for campaign in args.campaigns:
        folder = base/campaign/'execution/readback/results'
        for cell in sorted(folder.iterdir()):
            if not cell.is_dir():
                continue
            paths = [f for f in (cell/'raw.json',cell/'raw.json.gz') if f.exists()]
            if not paths:
                continue
            path = paths[0]
            raw = read(path)
            result = analyze(raw)
            result.update(campaign=campaign, label=cell.name, raw_path=str(path), raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            cells.append(result)
    check(len(cells)==args.expected_cells, 'complete raw cell count differs from explicit campaign size')
    result = dict(status='MEASUREMENT_ONLY', evidence_type='REANALYSIS_OF_NATIVE_SERVING',
                  question='Which recomputation led to output before invalidation, and which was reused only internally?',
                  time_boundary='Host receipt immediately after synchronous LLMEngine.step returns; no client/network receipt.',
                  resource_scope='APC off, native recompute, 6656 usable GPU blocks, no host KV offload; host peak unmeasured.',
                  nonadditive_diagnostics=['recovery_call_inclusive_s contains shared batch work and scheduler/host time; do not sum across requests or call it recompute tax',
                      'invalidated prefix includes work that already supported outputs; not synonymous with wasted work',
                      'reexecuted prefix overlaps outcome buckets; never add it to total recompute'], cells=cells)
    args.output_dir.mkdir(parents=True)
    (args.output_dir/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps([dict(label=c['label'],**c['summary']) for c in cells],indent=2))


if __name__ == '__main__':
    main()
