"""Partition measured output gaps by matched observed preemption positions."""
import argparse
from collections import defaultdict
import importlib.util
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / '.git').exists())
SERVICE = HERE.parent.parent / '20260915_repeated_kv_service_r01'
RESULTS = HERE.parent / 'execution_weste_26862/readback/results'
spec = importlib.util.spec_from_file_location('prior_peer_cost', SERVICE / 'analysis/peer_cost.py')
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)


def diagnostic(name, results):
    prior.RESULTS = results
    raw, _, signatures = prior.read_cell(name)
    step_calls = {step: call['call_index'] for call in raw['engine_steps'] if call['completed']
                  for step in range(call['scheduler_step_start'], call['scheduler_step_end'])}
    preempts = defaultdict(lambda: defaultdict(list))
    before_first = []
    for event in raw['preemption_events']:
        if not event.get('original_preemption_returned', True):
            raise ValueError('Incomplete preemption is not a confirmed gap annotation')
        rid = raw['internal_to_source'][event['victim_internal_request_id']]
        count = event['victim_state']['output_tokens']
        row = dict(step=event['attempted_step'], output_count=count)
        if count == 0:
            before_first.append(dict(request_id=rid, **row))
        else:
            if not 0 < count < len(signatures[rid]):
                raise ValueError('Preemption outside the measured output history')
            a, b = signatures[rid][count - 1:count + 1]
            if not a[0] < step_calls[event['attempted_step']] <= b[0]:
                raise ValueError('Preemption not enclosed by matched output calls')
            preempts[rid][count].append(row)
    return dict(signatures=signatures, preempts=preempts, before_first=before_first)


def measure(name, diag):
    prior.RESULTS = RESULTS
    raw, _, signatures = prior.read_cell(name)
    if signatures != diag['signatures']:
        raise ValueError(f'{name}: diagnostic output path mismatch; do not project annotations')
    rows = []
    for request in raw['requests']:
        rid = request['request_id']
        times = request['token_times_s']
        intervals = []
        for k, events in sorted(diag['preempts'][rid].items()):
            intervals.append(dict(before_output_count=k, after_output_count=k + 1,
                before_call=signatures[rid][k-1][0], after_call=signatures[rid][k][0],
                duration_s=times[k] - times[k-1], preemptions=events))
        recovery = sum(x['duration_s'] for x in intervals)
        other = sum(b - a for k, (a, b) in enumerate(zip(times, times[1:]), 1)
                    if k not in diag['preempts'][rid])
        ttft = times[0] - request['arrival_s']
        tail = request['completion_s'] - times[-1]
        completion = request['completion_s'] - request['arrival_s']
        residual = completion - ttft - recovery - other - tail
        if not math.isclose(residual, 0, abs_tol=1e-10):
            raise ValueError('Request timeline partition does not conserve time')
        rows.append(dict(request_id=rid, recovery_gap_count=len(intervals),
            diagnostic_preemptions_in_annotated_gaps=sum(len(x['preemptions']) for x in intervals),
            recovery_gap_sum_s=recovery, other_gap_sum_s=other, ttft_s=ttft,
            completion_tail_s=tail, completion_latency_s=completion,
            max_recovery_gap_s=max((x['duration_s'] for x in intervals), default=0),
            accounting_residual_s=residual, recovery_gaps=intervals))
    fields = ('recovery_gap_sum_s', 'other_gap_sum_s', 'ttft_s',
              'completion_tail_s', 'completion_latency_s', 'recovery_gap_count',
              'diagnostic_preemptions_in_annotated_gaps')
    return dict(cell=name, matching_diagnostic_requests=len(signatures),
        performance_preemption_events_observed=False,
        annotation_basis='Diagnostic preemption positions projected onto identical observed output paths', requests=rows,
        aggregate={key: sum(x[key] for x in rows) for key in fields})


def analyze():
    old_results = SERVICE / 'execution_weste_26862/readback/results'
    diags = {'current': diagnostic('diag-on', old_results),
             'eager': diagnostic('diagnostic-eager', RESULTS)}
    cells = {f'block{block}-{arm}': measure(f'block{block}-{arm}', diags[arm])
             for block in (0, 1) for arm in ('current', 'eager')}
    pairs = []
    for block in (0, 1):
        a, b = (cells[f'block{block}-{arm}'] for arm in ('current', 'eager'))
        previous = {r['request_id']: r for r in a['requests']}
        pairs.append(dict(block=block,
            aggregate_eager_minus_current={k: b['aggregate'][k] - v for k, v in a['aggregate'].items()},
            requests=[dict(request_id=r['request_id'],
                recovery_gap_sum_delta_s=r['recovery_gap_sum_s']-previous[r['request_id']]['recovery_gap_sum_s'],
                recovery_gap_count_delta=r['recovery_gap_count']-previous[r['request_id']]['recovery_gap_count'],
                max_recovery_gap_delta_s=r['max_recovery_gap_s']-previous[r['request_id']]['max_recovery_gap_s'])
                for r in b['requests']]))
    sources = [old_results/'diag-on/raw.json', RESULTS/'diagnostic-eager/raw.json']
    sources += [RESULTS/name/'raw.json' for name in cells]
    return dict(sources=[str(p.relative_to(REPO)) for p in sources],
        reused_helper=str((SERVICE/'analysis/peer_cost.py').relative_to(REPO)),
        cells=cells, pairs=pairs,
        diagnostic_before_first_output_preemptions={k: v['before_first'] for k,v in diags.items()},
        semantics=[
            'Durations come only from the four current lightweight runs; diagnostics annotate output positions.',
            'Exact request/call/output-count/new-token matching is required before projection.',
            'Lightweight preemption events were not measured; matching outputs do not prove hidden schedule/preemption identity.',
            'Recovery-named fields contain diagnostic-annotated gaps, not directly observed lightweight preemption counts.',
            'A gap may cross multiple preemptions; deduplicate by last returned output count.',
            'Recovery gap includes preemption-adjacent execution and recovery; it is not pure queue or transfer time.',
            'TTFT + recovery gaps + other generation gaps + completion tail equals completion latency per request.',
            'Sums across requests overlap in wall time and are request-seconds, not episode wall time.',
            'Old diagnostic current is reused for annotation only, not an independent performance repeat.',
            'This is observed policy-specific attribution, not a same-state counterfactual or hidden-KV equivalence.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = analyze()
    with args.output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps([dict(block=x['block'], deltas=x['aggregate_eager_minus_current'])
                      for x in result['pairs']], indent=2))
