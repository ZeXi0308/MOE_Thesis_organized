#!/usr/bin/env python3
"""Join admission decisions and actual first-prefill assignments to external arrivals.

Usage: python3 decision_summary.py CELL [CELL ...] --output NEW.json
Also writes one NEW.INDEX-CELL.starts.csv per cell; refuses existing outputs.
This reads traces only and does not reconstruct counterfactual native allocation.
"""
import argparse
import collections
import csv
import itertools
import json
import math
from pathlib import Path


STATE_FIELDS = ('free_blocks', 'running', 'active', 'recovery_count')


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def ranges(rows):
    result = {}
    for field in STATE_FIELDS:
        values = [row[field] for row in rows if finite(row.get(field))]
        result[field] = dict(min=min(values) if values else None,
                             max=max(values) if values else None, observations=len(values))
    return result


def opportunity_diagnostic(admission, raw, source_id):
    """Describe states actually inspected by the gate, never native counterfactuals."""
    decisions, snapshots = admission.get('decisions', []), admission.get('snapshots', [])
    cap, floor, limit = (admission.get(k) for k in ('cap', 'kv_floor', 'max_signal_wait_s'))
    gate_origin, raw_origin = admission.get('origin_perf_s'), raw.get('measurement_origin_perf_counter_s')
    offset = gate_origin-raw_origin if finite(gate_origin) and finite(raw_origin) else None

    def boundary(rows):
        stamps = [row['t'] for row in rows if finite(row.get('t'))]
        first, last = min(stamps, default=None), max(stamps, default=None)
        return dict(observations=len(rows), timestamped_observations=len(stamps),
            first_gate_relative_s=first, last_gate_relative_s=last,
            first_external_origin_s=first+offset if first is not None and offset is not None else None,
            last_external_origin_s=last+offset if last is not None and offset is not None else None)

    def tally(rows):
        return dict(evaluations=len(rows), unique_requests=len({source_id(row) for row in rows}),
                    time_boundaries=boundary(rows))

    latched = [row for row in decisions if row.get('latch') is True]

    def conditions(row):
        age, active, free = (row.get(k) for k in ('age_s', 'active', 'free_blocks'))
        bypass = row.get('signal_wait_limit_bypass')
        if isinstance(bypass, bool):
            unexpired = not bypass
        else:
            unexpired = age < limit if finite(age) and finite(limit) else None
        return dict(age_lt_10_s=age < 10 if finite(age) else None,
            signal_wait_limit_bypass=bypass if isinstance(bypass, bool) else None,
            signal_unexpired=unexpired,
            active_below_cap=active < cap if finite(active) and finite(cap) else None,
            free_at_least_kv_floor=free >= floor if finite(free) and finite(floor) else None)

    evaluated = [(row, conditions(row)) for row in latched]
    names = ('age_lt_10_s', 'signal_wait_limit_bypass', 'signal_unexpired',
             'active_below_cap', 'free_at_least_kv_floor')
    breakdown = {name: {label: tally([row for row, flags in evaluated if flags[name] is value])
                        for label, value in (('true', True), ('false', False), ('unknown', None))}
                 for name in names}
    joint = collections.defaultdict(list)
    combinations = collections.defaultdict(list)
    for row, flags in evaluated:
        parts = [flags[name] for name in ('signal_unexpired', 'active_below_cap', 'free_at_least_kv_floor')]
        status = 'false' if False in parts else 'unknown' if None in parts else 'true'
        joint[status].append(row)
        key = '|'.join('unknown' if part is None else str(part).lower() for part in parts)
        combinations[key].append(row)
    arrivals = [row['arrival_s'] for row in raw.get('requests', []) if finite(row.get('arrival_s'))]
    last_arrival = max(arrivals, default=None)
    all_expired = last_arrival+limit if last_arrival is not None and finite(limit) else None
    expiration_counts = None
    if all_expired is not None and offset is not None:
        expiration_counts = dict(
            before=tally([row for row in latched if finite(row.get('t')) and row['t']+offset < all_expired]),
            at_or_after=tally([row for row in latched if finite(row.get('t')) and row['t']+offset >= all_expired]),
            unknown_time=tally([row for row in latched if not finite(row.get('t'))]))
    mismatch = [row for row, _ in evaluated
                if finite(row.get('age_s')) and finite(limit)
                and isinstance(row.get('signal_wait_limit_bypass'), bool)
                and row['signal_wait_limit_bypass'] != (row['age_s'] >= limit)]
    return dict(
        role='OBSERVED_STATE_OPPORTUNITY_DIAGNOSTIC', mode=admission.get('mode'),
        semantics='Only never-started requests actually evaluated by Gate.defer are counted. '
            'In fixed/native mode, a latched state is a shadow diagnostic, not an executed recovery decision. '
            'The joint predicate describes signal-unexpired, active<recorded cap, free>=recorded KV floor. '
            'It does not prove native full-ISL allocation or scheduling would succeed; requests hidden behind '
            'a native queue-head break are not observed. Evaluations and unique requests are distinct; '
            'neither represents independent run repetitions.',
        thresholds=dict(cap=cap, kv_floor=floor, max_signal_wait_s=limit, explicit_age_boundary_s=10),
        signal_age_semantics='Prefer the recorded bypass boolean; when absent, infer unexpired only '
            'from a finite age and recorded max_signal_wait_s. Missing values remain unknown.',
        gate_to_external_origin_offset_s=offset,
        time_semantics='First/last recorded observations are sample boundaries, not continuous latch/backlog '
            'durations. External-origin times are unavailable if origin_perf_s is absent.',
        latch_true_evaluations=tally(latched),
        decision_latch_unknown_evaluations=tally([row for row in decisions if not isinstance(row.get('latch'), bool)]),
        conditions_within_latch=breakdown,
        all_three_conditions_within_latch={key: tally(joint[key]) for key in ('true', 'false', 'unknown')},
        joint_condition_order=['signal_unexpired', 'active_below_cap', 'free_at_least_kv_floor'],
        joint_condition_combinations={key: tally(rows) for key, rows in sorted(combinations.items())},
        latch_true_sample_boundaries=boundary([row for row in snapshots if row.get('latch') is True]),
        backlog_nonzero_sample_boundaries=boundary([row for row in snapshots
            if finite(row.get('recovery_count')) and row['recovery_count'] > 0]),
        snapshot_latch_unknown=sum(not isinstance(row.get('latch'), bool) for row in snapshots),
        snapshot_backlog_unknown=sum(not finite(row.get('recovery_count')) for row in snapshots),
        last_planned_arrival_s=last_arrival,
        all_planned_requests_reach_signal_age_limit_s=all_expired,
        latch_evaluations_relative_to_all_expired=expiration_counts,
        recorded_bypass_age_disagreement=tally(mismatch))


def summarize(cell):
    raw = json.loads((cell/'raw.json').read_text())
    admission = json.loads((cell/'admission.json').read_text())
    sources = {r['request_id']: r for r in raw['requests']}
    if len(sources) != len(raw['requests']):
        raise ValueError('Duplicate source request IDs: '+str(cell))
    identity = {}
    for rid, row in sources.items():
        for key in ('request_id', 'internal_request_id', 'external_request_id'):
            if row.get(key) is not None:
                identity[row[key]] = rid

    def source_id(record):
        rid = record['request_id']
        if rid not in identity:
            raise ValueError(f'Unknown trace request ID {rid} in {cell}')
        return identity[rid]

    decisions = admission.get('decisions', [])
    per_request = collections.defaultdict(list)
    for decision in decisions:
        per_request[source_id(decision)].append(decision)
    origin = raw.get('measurement_origin_perf_counter_s')
    if not finite(origin):
        raise ValueError('Measurement performance-clock origin missing: '+str(cell))
    starts, seen, groups = [], set(), []
    for start in admission.get('starts', []):
        rid = source_id(start)
        if rid in seen:
            raise ValueError('Duplicate actual first-prefill start: '+rid)
        seen.add(rid)
        stamp = start['first_prefill_perf_s']
        if not finite(stamp):
            raise ValueError('Nonfinite first-prefill timestamp: '+rid)
        t = stamp-origin
        arrival = sources[rid]['arrival_s']
        if t < arrival or t > raw['observation_end_s'] + 1e-6:
            raise ValueError('First-prefill assignment outside request observation: '+rid)
        if starts and t < starts[-1]['first_prefill_schedule_return_s']:
            raise ValueError('First-prefill starts are not in chronological order')
        if not groups or groups[-1]['first_prefill_schedule_return_s'] != t:
            groups.append(dict(batch_index=len(groups), first_prefill_schedule_return_s=t, request_ids=[]))
        groups[-1]['request_ids'].append(rid)
        related = per_request[rid]
        starts.append(dict(sequence_index=len(starts), batch_index=len(groups)-1, request_id=rid,
            arrival_s=arrival, first_prefill_schedule_return_s=t, new_prefill_wait_s=t-arrival,
            changed_by_recovery_evaluations=sum(bool(d.get('changed_by_recovery')) for d in related),
            signal_bypass_evaluations=sum(bool(d.get('signal_wait_limit_bypass')) for d in related),
            outcome=sources[rid].get('status')))
    actual = {s['request_id']: s for s in starts}
    changed = [d for d in decisions if d.get('changed_by_recovery')]
    bypass = [d for d in decisions if d.get('signal_wait_limit_bypass')]
    changed_rows = []
    for rid in sources:
        affected = [d for d in per_request[rid] if d.get('changed_by_recovery')]
        if not affected:
            continue
        ages = [d['age_s'] for d in affected if finite(d.get('age_s'))]
        all_related = per_request[rid]
        start = actual.get(rid, {})
        changed_rows.append(dict(request_id=rid, changed_evaluations=len(affected),
            arrival_s=sources[rid]['arrival_s'], state_ranges=ranges(affected),
            first_changed_age_s=min(ages) if ages else None,
            last_changed_age_s=max(ages) if ages else None,
            first_prefill_schedule_return_s=start.get('first_prefill_schedule_return_s'),
            new_prefill_wait_s=start.get('new_prefill_wait_s'), actual_started=rid in actual,
            signal_bypass_evaluations=sum(bool(d.get('signal_wait_limit_bypass')) for d in all_related),
            signal_bypass_still_denied_evaluations=sum(bool(d.get('signal_wait_limit_bypass')
                and d.get('denied')) for d in all_related), outcome=sources[rid].get('status')))
    checks = dict(changed_event_denied=all(d.get('denied') for d in changed),
        changed_event_kv_only_allowed=all(d.get('kv_only_denied') is False for d in changed))
    workload = [dict(request_id=r['request_id'], arrival_s=r['arrival_s'],
        prompt_sha256=r.get('prompt_token_ids_sha256'), max_output_tokens=r.get('max_output_tokens'))
        for r in raw['requests']]
    return dict(cell=str(cell.resolve()), mode=admission.get('mode'), cap=admission.get('cap'),
        planned_requests=len(sources), changed_evaluations=len(changed),
        changed_unique_requests=len(changed_rows), changed_request_ids=[r['request_id'] for r in changed_rows],
        changed_state_ranges=ranges(changed), changed_requests=changed_rows,
        same_state_gate_checks=checks, signal_bypass_evaluations=len(bypass),
        signal_bypass_unique_requests=len({source_id(d) for d in bypass}),
        signal_bypass_still_denied_evaluations=sum(bool(d.get('denied')) for d in bypass),
        signal_bypass_reasons=dict(collections.Counter(d.get('reason') for d in bypass)),
        observed_state_opportunity=opportunity_diagnostic(admission, raw, source_id),
        actually_started_requests=len(starts), never_started_request_ids=[r for r in sources if r not in seen],
        actual_first_prefill_sequence=[s['request_id'] for s in starts],
        actual_first_prefill_batches=groups, actual_first_prefill_starts=starts, workload_identity=workload)


def compare_sequences(a, b):
    same_workload = a['workload_identity'] == b['workload_identity']
    result = dict(cell_a=a['cell'], cell_b=b['cell'], same_workload=same_workload)
    if not same_workload:
        result['comparison_status'] = 'INCOMPARABLE_WORKLOAD'
        return result
    sa, sb = a['actual_first_prefill_sequence'], b['actual_first_prefill_sequence']
    # Inside one schedule call, dict iteration order is not a distinct admission time.
    ba = [sorted(batch['request_ids']) for batch in a['actual_first_prefill_batches']]
    bb = [sorted(batch['request_ids']) for batch in b['actual_first_prefill_batches']]
    ta = {s['request_id']: s['first_prefill_schedule_return_s'] for s in a['actual_first_prefill_starts']}
    tb = {s['request_id']: s['first_prefill_schedule_return_s'] for s in b['actual_first_prefill_starts']}
    differences = [dict(request_id=rid, start_a_s=ta[rid], start_b_s=tb[rid],
                        start_b_minus_a_s=tb[rid]-ta[rid]) for rid in sa if rid in tb]
    absolute = [abs(d['start_b_minus_a_s']) for d in differences]
    first = next((i for i, (x, y) in enumerate(zip(sa, sb)) if x != y), None)
    if first is None and len(sa) != len(sb):
        first = min(len(sa), len(sb))
    result.update(comparison_status='DESCRIPTIVE', same_started_request_set=set(sa) == set(sb),
        identical_sequence_ignoring_timestamps=sa == sb,
        identical_batches_ignoring_timestamps=ba == bb, first_sequence_difference_index=first,
        identical_relative_start_timestamps=ta == tb,
        common_started_requests=len(differences), max_absolute_start_delta_s=max(absolute, default=None),
        mean_absolute_start_delta_s=sum(absolute)/len(absolute) if absolute else None,
        per_request_start_times=differences)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cells', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    csv_paths = [args.output.with_name(f'{args.output.stem}.{i:02d}-{cell.name}.starts.csv')
                 for i, cell in enumerate(args.cells)]
    if any(p.exists() for p in [args.output, *csv_paths]):
        parser.error('Refusing to overwrite an existing output; choose a new --output path.')
    cells = [summarize(cell) for cell in args.cells]
    pairs = [compare_sequences(a, b) for a, b in itertools.combinations(cells, 2)]
    result = dict(schema_version=2,
        decision_semantics='changed_by_recovery is a different gate decision relative to the KV-only gate '
            'at the same observed state. It does not prove an unexecuted native allocation would succeed. '
            'Repeated gate evaluations are not independent requests or experiments.',
        start_semantics='Actual first-prefill permission is the return of the first scheduler call assigning '
            'tokens to the request, not GPU execution or first output. Waiting starts at external arrival. '
            'Equal timestamps identify one scheduler call; within-call sequence order is also reported separately.',
        sequence_semantics='Order equality alone does not imply equal admission timing. Timing differences '
            'between runs also include runtime noise; this summary makes no statistical or benefit claim.',
        signal_bypass_semantics='The age limit bypasses KV/recovery gates; a fixed cap can still deny admission.',
        cells=cells, sequence_comparisons=pairs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for cell, path in zip(cells, csv_paths):
        rows = cell['actual_first_prefill_starts']
        with path.open('x', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ['request_id'])
            writer.writeheader()
            writer.writerows(rows)
        cell['actual_first_prefill_csv'] = str(path.resolve())
    with args.output.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    print('| Cell | Changed evaluations / requests | Actual starts / planned | Bypass evaluations / requests |')
    print('|---|---:|---:|---:|')
    for c in cells:
        print(f"| {Path(c['cell']).name} | {c['changed_evaluations']} / {c['changed_unique_requests']} | "
              f"{c['actually_started_requests']} / {c['planned_requests']} | "
              f"{c['signal_bypass_evaluations']} / {c['signal_bypass_unique_requests']} |")
    print('Observed-state opportunity diagnostic (not native-allocation counterfactual):')
    for c in cells:
        diagnostic = c['observed_state_opportunity']
        latched = diagnostic['latch_true_evaluations']
        joint = diagnostic['all_three_conditions_within_latch']['true']
        print(f"{Path(c['cell']).name}: latch evaluations/requests="
              f"{latched['evaluations']}/{latched['unique_requests']}; "
              f"unexpired + active<cap + free>=floor="
              f"{joint['evaluations']}/{joint['unique_requests']}; "
              f"all-arrivals age-limit time="
              f"{diagnostic['all_planned_requests_reach_signal_age_limit_s']} s")
    for p in pairs:
        if p['same_workload']:
            print(f"{Path(p['cell_a']).name} vs {Path(p['cell_b']).name}: "
                f"same order={p['identical_sequence_ignoring_timestamps']}, "
                f"same batches={p['identical_batches_ignoring_timestamps']}, "
                f"max |start delta|={p['max_absolute_start_delta_s']} s")
    print(args.output.resolve())


if __name__ == '__main__':
    main()
