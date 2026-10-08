#!/usr/bin/env python3
"""Describe recovery footprint and exact max-gap window associations; no causality."""
import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def fraction(part, whole):
    return part/whole if whole else None


def group(rows, denominator):
    return dict(request_count=len(rows), fraction_of_all_requests=fraction(len(rows), denominator),
        request_ids=[r['request'] for r in rows],
        recovery_marked_count=sum(r['preemptions'] > 0 for r in rows),
        maxgap_overlaps_own_recovery_count=sum(r['own_recovery_overlap_s'] > 0 for r in rows),
        maxgap_overlaps_any_recovery_count=sum(r['any_recovery_overlap_s'] > 0 for r in rows),
        directly_selected_count=sum(r['directly_selected'] for r in rows))


def analyze_cell(cell, intervals):
    path = Path(cell['directory'])/'raw.json'
    payload = path.read_bytes(); raw = json.loads(payload)
    raw_rows = {r['request_id']: r for r in raw['requests']}
    full = {r['request']: r for r in cell['run_summary']['per_request']}
    metrics = {r['request']: r for r in cell['per_request']}
    if set(raw_rows) != set(metrics) or set(full) != set(metrics):
        raise ValueError('Request mappings incomplete; cannot state full-population footprint: '+str(path))
    windows = defaultdict(list); missing_windows = []
    for index, event in enumerate(cell['recovery_events']):
        start, end = event.get('demand_host_s'), event.get('next_output_s')
        if finite(start) and finite(end) and end >= start:
            windows[event['request']].append([start, end])
        else:
            missing_windows.append(index)
    all_windows = intervals.merged([window for values in windows.values() for window in values])
    actions = cell['recovery_start_gate_actions']
    selected = {r['target']['source_request'] for r in actions['rows']
                if r.get('actual_gate_executed') and r.get('executed_breaks', 0) > 0}
    p95 = cell['observed']['maxgap_s']['p95']
    result_rows = []
    for rid, metric in metrics.items():
        request, recovery = raw_rows[rid], full[rid]
        times = request['token_times_s']
        positive = [(a, b) for a, b in zip(times, times[1:]) if b > a]
        maximum = max((b-a for a, b in positive), default=None)
        longest = [[a, b] for a, b in positive if b-a == maximum]
        own = intervals.overlap(longest, windows[rid])
        any_recovery = intervals.overlap(longest, all_windows)
        result_rows.append(dict(request=rid, internal_request_id=request.get('internal_request_id'),
            external_request_id=request.get('external_request_id'), status=metric['status'],
            flow_s=metric['flow_s'], maxgap_s=metric['maxgap_s'], slo_pass=metric['slo_pass'],
            ttft_failed=metric['ttft_s'] > cell['joint_slo']['ttft_limit_s'],
            maxgap_failed=metric['maxgap_s'] > cell['joint_slo']['maxgap_limit_s'],
            preemptions=recovery['preemptions'],
            repeat_preemptions_after_prior_recovery_output=recovery['repeat_preemptions_after_prior_recovery_output'],
            recovery_union_s=recovery['cumulative_recovery_union_lower_bound_s'],
            recovery_union_exact=recovery['cumulative_recovery_union_exact'],
            recovery_windows_s=intervals.merged(windows[rid]),
            reconstructed_recovery_union_s=intervals.length(windows[rid]),
            maxgap_matches_closed_raw_gap=maximum == metric['maxgap_s'],
            longest_closed_gap_windows_s=longest,
            own_recovery_overlap_s=own['intersection_s'],
            own_recovery_intersections_s=own['intersection_intervals_s'],
            any_recovery_overlap_s=any_recovery['intersection_s'],
            any_recovery_intersections_s=any_recovery['intersection_intervals_s'],
            directly_selected=rid in selected))
    count = cell['planned']
    recover = [r for r in result_rows if r['preemptions'] > 0]
    repeats = [r for r in result_rows if r['preemptions'] > 1]
    after_output = [r for r in result_rows if r['repeat_preemptions_after_prior_recovery_output'] > 0]
    failures = [r for r in result_rows if r['slo_pass'] is False]
    tail = [r for r in result_rows if r['maxgap_s'] > p95]
    targets = [r for r in result_rows if r['directly_selected']]
    flow_known = [r['flow_s'] for r in result_rows if finite(r['flow_s'])]
    union_known = [r['recovery_union_s'] for r in result_rows if finite(r['recovery_union_s'])]
    flow_sum, union_sum = math.fsum(flow_known), math.fsum(union_known)
    queue_context = []
    mapping = raw['internal_to_source']
    for row in actions['rows']:
        decision = row['raw_decision']; waiting = decision.get('waiting_order')
        target = row['target']['internal_request']
        suffix = waiting[waiting.index(target)+1:] if isinstance(waiting, list) and target in waiting else None
        queue_context.append(dict(target=row['target'], actual_gate_executed=row['actual_gate_executed'],
            executed_breaks=row['executed_breaks'], waiting_order=waiting,
            local_skipped_order=decision.get('local_skipped_order'),
            initial_suffix_after_selected=suffix,
            mapped_initial_suffix=[mapping.get(r) for r in suffix] if suffix is not None else None,
            native_other_head_pass_through=row.get('native_other_head_pass_through')))
    return dict(directory=cell['directory'], mode=cell['mode'], status=cell['status'],
        raw_path=str(path), raw_sha256=hashlib.sha256(payload).hexdigest(),
        planned=count, arrived=sum(r.get('arrived_at_observation_end') is True for r in raw_rows.values()),
        completed=cell['completed'], failed=cell['failed'], unfinished=cell['unfinished'],
        recovery_episodes=len(cell['recovery_events']), recovery_requests=group(recover, count),
        repeatedly_preempted_requests=group(repeats, count),
        repeat_preemption_events=sum(max(0, r['preemptions']-1) for r in result_rows),
        re_preempted_after_recovery_output_requests=group(after_output, count),
        re_preemption_events_after_recovery_output=sum(r['repeat_preemptions_after_prior_recovery_output'] for r in result_rows),
        maxgap_tail=dict(definition='Strictly greater than the existing canonical P95; boundary ties reported separately, not an exactly5% population.',
            p95_s=p95, tied_at_p95_count=sum(r['maxgap_s'] == p95 for r in result_rows), **group(tail, count)),
        joint_slo_failures=dict(ttft_limit_s=cell['joint_slo']['ttft_limit_s'],
            maxgap_limit_s=cell['joint_slo']['maxgap_limit_s'],
            ttft_failure_count=sum(r['ttft_failed'] for r in result_rows),
            maxgap_failure_count=sum(r['maxgap_failed'] for r in result_rows), **group(failures, count)),
        aggregate_exposure=dict(flow_sum_s=flow_sum, flow_known_requests=len(flow_known),
            recovery_union_sum_s=union_sum, recovery_union_known_requests=len(union_known),
            recovery_union_exact_requests=sum(r['recovery_union_exact'] for r in result_rows),
            recovery_union_over_flow_sum=fraction(union_sum, flow_sum),
            missing_recovery_event_windows=missing_windows,
            maxgap_raw_mismatch_ids=[r['request'] for r in result_rows if not r['maxgap_matches_closed_raw_gap']],
            interval_union_mismatch_ids=[r['request'] for r in result_rows
                if abs(r['recovery_union_s']-r['reconstructed_recovery_union_s']) > 1e-9]),
        direct_selected_requests=dict(**group(targets, count),
            fraction_of_recovery_requests=fraction(len(targets), len(recover)),
            recovery_union_sum_s=math.fsum(r['recovery_union_s'] for r in targets),
            fraction_of_all_recovery_union=fraction(math.fsum(r['recovery_union_s'] for r in targets), union_sum),
            slo_failure_count=sum(r['slo_pass'] is False for r in targets)),
        recorded_queue_context=queue_context, per_request=result_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('metrics', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or args.metrics.parent/'problem-scope.json'
    if output.exists():
        raise FileExistsError('Refusing to overwrite '+str(output))
    payload = args.metrics.read_bytes(); metrics = json.loads(payload)
    helper = BASE/'recovery_gc_diag/analyze.py'
    helper_sha = hashlib.sha256(helper.read_bytes()).hexdigest()
    if helper_sha != metrics['analyzer_sources_sha256']['recovery_gc_diag/analyze.py']:
        raise RuntimeError('Existing interval helper differs from canonical source pin')
    spec = importlib.util.spec_from_file_location('scope_existing_intervals', helper)
    intervals = importlib.util.module_from_spec(spec); spec.loader.exec_module(intervals)
    result = dict(input_metrics=str(args.metrics.resolve()), input_metrics_sha256=hashlib.sha256(payload).hexdigest(),
        analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), interval_helper_sha256=helper_sha,
        cells=[analyze_cell(c, intervals) for c in metrics['cells']],
        semantics={
            'recovery': 'Reuse canonical preemption-to-next-output episodes and per-request recovery unions; this includes native compute/capacity/observation waits, not just avoidable transfer delay.',
            'association': 'A recovery-marked request and a positive time intersection are separate. Intersect exact consecutive raw token-time endpoints of all tied longest closed gaps with same-request or any-request recovery windows. Boundary-only contact is not overlap. Temporal overlap is not causality.',
            'denominator': 'Request fractions use all planned requests; arrival/status counts are retained. Flow and recovery exposure are sums across requests, not wall time; the ratio is descriptive and is not an optimization upper bound.',
            'coverage': 'Direct means explicitly selected by an actually executed gate. A head break can also stall its waiting suffix and change later computation/transfer state; selected1/256 is neither total impact coverage nor an upper bound. Initial queue records do not enumerate every indirect effect.',
            'claim_limit': 'One workload and two candidate runs, with pre-action drift and changed actual work. This does not establish the residual loss under a tuned strong simple baseline or the importance/novelty of single-event gating.'})
    encoded = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n'
    with output.open('x') as stream:
        stream.write(encoded)
    print(json.dumps(dict(output=str(output), sha256=hashlib.sha256(encoded.encode()).hexdigest())))


if __name__ == '__main__':
    main()
