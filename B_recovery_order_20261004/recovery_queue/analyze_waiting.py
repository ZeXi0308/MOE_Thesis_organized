#!/usr/bin/env python3
"""Summarize native waiting snapshots and unapplied shadow choices for one anchor."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from analyze import read, summary


def selected_chain(source, metric, raw=None):
    rid = source.get('selected_request'); action = (metric or {}).get('source_actions', {})
    episode = action.get('selected_recovery_episode'); origin = None
    if episode is not None and episode.get('request') != rid: episode = None
    if episode is not None:
        origin = episode['begin_host_perf_s']-episode['begin_s']
    elif raw is not None:
        origin = raw.get('measurement_origin_perf_counter_s')
    selected = next((e for e in source.get('events', []) if e.get('kind') == 'selected'), None)
    releases = [e for e in source.get('events', []) if e.get('kind') == 'release']
    release = releases[0] if releases else None
    recovery = episode.get('request_recovery') if episode else None
    successes = [a for a in (episode or {}).get('attempts', []) if a.get('outcome') == 'success'
        and selected is not None and a['record']['begin_host_perf_s'] >= selected['host_perf_s']]
    first = min(successes, key=lambda a:a['record']['begin_host_perf_s']) if successes else None
    allocated = first['record']['begin_host_perf_s']-origin if first is not None and origin is not None else None
    chosen = selected['host_perf_s']-origin if selected is not None and origin is not None else None
    demand = recovery.get('demand_host_s') if recovery else None
    output = recovery.get('next_output_s') if recovery else None
    source_id = action.get('selected_source_request') or (raw or {}).get('internal_to_source', {}).get(rid)
    return dict(status='NO_SELECTION' if rid is None else 'AVAILABLE' if recovery else 'UNVERIFIED',
        internal_request=rid, source_request=source_id, selection_host_s=chosen,
        demand_host_s=demand, next_output_s=output, demand_to_next_output_s=recovery.get('demand_to_next_output_s') if recovery else None,
        first_successful_allocate_begin_s=allocated,
        selection_to_first_successful_allocate_s=allocated-chosen if allocated is not None and chosen is not None else None,
        demand_to_first_successful_allocate_s=allocated-demand if allocated is not None and demand is not None else None,
        first_successful_allocate_to_next_output_s=output-allocated if output is not None and allocated is not None else None,
        first_scheduled_plan_s=recovery.get('first_scheduled_plan_s') if recovery else None,
        source_release_reason=release.get('reason') if release else None, original_recovery=recovery,
        allocation_success_record=first, clock_origin_perf_s=origin,
        selection_perf_s=selected.get('host_perf_s') if selected else None,
        release_perf_s=release.get('host_perf_s') if release else None)


def analyze_records(waiting, source, metric=None, raw=None):
    issues = []; events = waiting.get('events', []); chain = selected_chain(source, metric, raw)
    rid, origin = source.get('selected_request'), chain['clock_origin_perf_s']
    for key in ('events', 'status', 'mode', 'native_code_verified', 'observation_calls', 'observation_errors'):
        if key not in waiting: issues.append('missing '+key)
    if waiting.get('mode') != 'observe_only' or source.get('mode') != 'native': issues.append('non-native observation mode')
    if waiting.get('native_code_verified') is not True: issues.append('native code verification unavailable')
    if waiting.get('status') != 'UNINSTALLED': issues.append('observer not cleanly uninstalled')
    if not isinstance(events, list): events = []; issues.append('events not a list')
    required = ('kind', 'host_perf_s', 'observer_return_perf_s', 'selected_request', 'entry_gate',
        'candidates', 'native_head', 'oldest_arrival_fit', 'shadow_gate_open')
    valid = [(i, e) for i, e in enumerate(events) if isinstance(e, dict) and all(k in e for k in required)]
    if len(valid) != len(events): issues.append('incomplete event fields')
    if waiting.get('observation_calls') != len(events): issues.append('observation call count differs from event count')
    errors = waiting.get('observation_errors', [])
    if errors: issues.append('observation callback errors')
    if any(e['selected_request'] != rid for _, e in valid): issues.append('anchor identity mismatch')
    if any(a[1]['host_perf_s'] > b[1]['host_perf_s'] for a, b in zip(valid, valid[1:])): issues.append('host snapshots out of order')
    if any(e['kind'] not in ('waiting_entry', 'queue_decision') for _, e in valid): issues.append('unknown observation kind')
    if rid is not None and chain['status'] == 'UNVERIFIED': issues.append('selected recovery unavailable in existing metrics')
    durations = [e['observer_return_perf_s']-e['host_perf_s'] for _, e in valid]
    if any(t < 0 for t in durations): issues.append('negative callback interval')
    entries = [(i, e) for i, e in valid if e['kind'] == 'waiting_entry']
    decisions = [(i, e) for i, e in valid if e['kind'] == 'queue_decision']
    duration_by_gate = defaultdict(float); intervals = []
    def when(stamp): return stamp-origin if stamp is not None and origin is not None else None
    for (index, event), (_, following) in zip(entries, entries[1:]):
        delta = following['host_perf_s']-event['host_perf_s']
        if delta < 0: continue
        duration_by_gate[event['entry_gate']] += delta
        intervals.append(dict(event_index=index, gate=event['entry_gate'], duration_s=delta,
            start_perf_s=event['host_perf_s'], end_perf_s=following['host_perf_s'],
            start_s=when(event['host_perf_s']), end_s=when(following['host_perf_s'])))
    first_stamp = entries[0][1]['host_perf_s'] if entries else None
    last_stamp = entries[-1][1]['host_perf_s'] if entries else chain['selection_perf_s']
    end = chain['release_perf_s']
    unknown_last = dict(label='UNKNOWN_LAST_SEGMENT', start_perf_s=last_stamp, end_perf_s=end,
        duration_s=end-last_stamp if end is not None and last_stamp is not None and end >= last_stamp else None)
    labels = ('native_head_is_selected', 'native_head_differs_selected', 'native_head_missing',
        'selected_is_candidate', 'head_is_preempted_candidate', 'shadow_present', 'shadow_absent',
        'shadow_differs_head', 'shadow_is_selected', 'shadow_differs_selected', 'selected_conservative_fit', 'selected_eligible')
    counts = Counter({key: 0 for key in labels}); categories = defaultdict(list); scored = []
    for index, event in valid:
        candidates = event['candidates']; selected = next((c for c in candidates if c.get('request') == rid), None)
        head = next((c for c in candidates if c.get('request') == event['native_head']), None)
        shadow = event['oldest_arrival_fit']; shadow_row = next((c for c in candidates if c.get('request') == shadow), None)
        closed = event['entry_gate'] != 'MAY_ENTER_WAITING'; different = shadow is not None and shadow != event['native_head']
        tests = dict(selected_memory_fit_gate_closed=closed and bool(selected and selected.get('memory_fit')),
            any_memory_fit_gate_closed=closed and any(c.get('memory_fit') for c in candidates),
            conservative_shadow_eligible_head_differs=bool(event['shadow_gate_open'] and different
                and shadow_row and shadow_row.get('conservative_fit') and shadow_row.get('eligible')),
            selected_eligible_head_differs=bool(selected and selected.get('eligible') and event['native_head'] != rid))
        for label, passed in tests.items():
            if passed: categories[label].append(index)
        score = (8*tests['selected_eligible_head_differs'] + 4*tests['conservative_shadow_eligible_head_differs']
                 + 2*tests['selected_memory_fit_gate_closed'] + tests['any_memory_fit_gate_closed'])
        if score: scored.append((score, index))
        if event['kind'] == 'queue_decision':
            for label, passed in dict(native_head_is_selected=event['native_head'] == rid,
                native_head_differs_selected=event['native_head'] is not None and event['native_head'] != rid,
                native_head_missing=event['native_head'] is None,
                selected_is_candidate=selected is not None, head_is_preempted_candidate=head is not None,
                shadow_present=shadow is not None, shadow_absent=shadow is None, shadow_differs_head=different,
                shadow_is_selected=shadow is not None and shadow == rid,
                shadow_differs_selected=shadow is not None and shadow != rid,
                selected_conservative_fit=bool(selected and selected.get('conservative_fit')),
                selected_eligible=bool(selected and selected.get('eligible'))).items():
                counts[label] += int(passed)
    sample_indices = []
    for label in categories:
        sample_indices.extend(categories[label][:2])
    sample_indices.extend(index for _, index in sorted(scored, key=lambda pair:(-pair[0], pair[1])))
    sample_indices = list(dict.fromkeys(sample_indices))[:8]
    selected_indices = {i for i, _ in valid}
    if not sample_indices and valid: sample_indices = [valid[0][0]]
    return dict(status='UNVERIFIED' if issues else 'ANALYZED', issues=issues,
        actual_task_order_changes=0, shadow_applied=False, selected=chain,
        observer=dict(status=waiting.get('status'), native_code_verified=waiting.get('native_code_verified'),
            declared_calls=waiting.get('observation_calls'), event_count=len(events), valid_events=len(valid),
            errors=errors, error_count=len(errors), callback_preparation_s=summary(durations),
            incomplete_event_durations=len(events)-len(valid), error_durations_unavailable=len(errors),
            measured_callback_preparation_sum_s=sum(durations),
            callback_scope='start to observer_return; excludes pre-start selection scan, final append/counter and unobserved calls'),
        waiting_entry_count=len(entries), waiting_entry_gates=dict(Counter(e['entry_gate'] for _, e in entries)),
        approximate_snapshot_label_durations_s=dict(duration_by_gate), adjacent_entry_intervals=intervals,
        unknown_prefix_s=first_stamp-chain['selection_perf_s'] if first_stamp is not None and chain['selection_perf_s'] is not None else None,
        unknown_last_segment=unknown_last, queue_decision_count=len(decisions), queue_decision_comparisons=dict(counts),
        condition_counts={label: len(categories[label]) for label in (
            'selected_memory_fit_gate_closed', 'any_memory_fit_gate_closed',
            'conservative_shadow_eligible_head_differs', 'selected_eligible_head_differs')},
        condition_counts_by_phase={label: dict(Counter(events[i]['kind'] for i in indices)) for label, indices in categories.items()},
        condition_count_scope='Snapshot counts, not unique scheduling opportunities; entry and decision may describe the same step',
        condition_event_indices=dict(categories),
        discriminating_full_rows=[dict(event_index=i, state=events[i]) for i in sample_indices if i in selected_indices],
        unrecognized_or_incomplete_rows=[dict(event_index=i, state=e) for i, e in enumerate(events) if i not in selected_indices])


def analyze_session(session, metrics_path=None):
    if metrics_path is None:
        metrics_path = next((session/name for name in ('source-metrics.json', 'source-handoff-metrics.json', 'metrics.json')
            if (session/name).exists()), None)
    metrics = read(metrics_path) if metrics_path else {}; cells = []
    by_name = {Path(c['directory']).parent.name: c for c in metrics.get('cells', [])}
    for directory in sorted(p for p in session.glob('cell-*') if p.is_dir()):
        output = directory/'output'; wp = output/'waiting-decisions.json'; sp = output/'source-handoff.json'
        if not wp.exists() or not sp.exists():
            cells.append(dict(directory=str(output), status='UNVERIFIED',
                missing=[str(p) for p in (wp, sp) if not p.exists()], actual_task_order_changes=0, shadow_applied=False)); continue
        waiting, source = read(wp), read(sp); metric = by_name.get(directory.name); raw = None
        episode = (metric or {}).get('source_actions', {}).get('selected_recovery_episode')
        if source.get('selected_request') is not None and (not episode or episode.get('request') != source['selected_request']):
            rawpath = next((output/n for n in ('raw.json', 'raw.json.gz') if (output/n).exists()), None)
            raw = read(rawpath) if rawpath else None
        result = analyze_records(waiting, source, metric, raw); result['directory'] = str(output)
        result['all_request_context'] = {k: (metric or {}).get(k) for k in ('status', 'planned', 'statuses', 'failed', 'unfinished', 'outputs')}
        cells.append(result)
    return dict(metrics=str(metrics_path) if metrics_path else None, cells=cells,
        semantics='Read-only observer; actual task-order changes are fixed at zero and shadows were never applied. '
            'queue_decision records native peek, not allocation success or execution. Conservative eligibility is '
            'the observer predicate, not an oracle proving every native condition. Adjacent waiting_entry intervals '
            'carry the left snapshot label only as an approximation; gates may change inside them. Prefix and final '
            'segments remain unknown and are not assigned to a gate. These intervals and callback durations cannot '
            'be added into request benefit. Observation stops at source release; later recovery-to-output uses the '
            'existing canonical metrics. Missing chains, failed requests and observation errors are retained.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, required=True); parser.add_argument('--metrics', type=Path)
    parser.add_argument('--output', type=Path, required=True); args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    result = analyze_session(args.session, args.metrics)
    with args.output.open('x') as stream: json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(output=str(args.output), cells=len(result['cells']), statuses=[c['status'] for c in result['cells']])))


if __name__ == '__main__':
    main()
