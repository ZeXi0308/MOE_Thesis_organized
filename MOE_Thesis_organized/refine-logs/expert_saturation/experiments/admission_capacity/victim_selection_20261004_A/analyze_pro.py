#!/usr/bin/env python3
"""Describe normal-capacity PRO runs; shadow rankings are not policy outcomes."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

from analyze import SEMANTICS, distribution, optional, read_cell


SEMANTICS = dict(SEMANTICS, **dict(
    causality='Different input cases are separate workload characterizations; no cross-cell policy or causal comparison.',
    repeats='One run per input case (repeats=1). Events are dependent observations, not independent repetitions.',
    shadow='Read-only rankings of the recorded legal candidates. Changed shadow choices are opportunities, not executed interventions or measured gains.',
    host_signal='Materialized whole pages after the ready contiguous host prefix; not actual absent-key count, bytes to save, or measured recomputation time.',
    unknown='A shadow rule is UNKNOWN if any required candidate signal is missing/nonfinite; unknown values are never filled with zero.',
    pressure='Recorded schedule-boundary extrema, not continuous physical peaks. Pending transfer jobs are metadata, not device completion.',
    cost='Successful-selector time from the common analyzer; observation and boundary-probe timing are retained separately when recorded. No complete-controller-cost claim.',
    host_size='Compare host ranking with minimum floor(computed_tokens/16) and minimum held_blocks, using the same latest-running-index tie-break. Equal-size signal variation is observational evidence of a distinct signal, not causal benefit.',
))


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def shadow(event, rule):
    """Keep the native tie-break and decline rankings with unknown state."""
    rows = event.get('candidates', [])
    result = dict(status='NO_LEGAL_CANDIDATES', changed=None,
                  selected_victim=None, capacity_delta_blocks=None,
                  exact_capacity_opportunity=None)
    if not rows:
        return result
    if (any(not isinstance(row, dict) or not row.get('request')
            or not finite(row.get('running_index')) for row in rows)
            or len({row['request'] for row in rows}) != len(rows)):
        return dict(result, status='UNKNOWN_CANDIDATE_IDENTITY_OR_ORDER')
    tail = max(rows, key=lambda row: row['running_index'])
    recorded = event.get('baseline_victim')
    if recorded is not None and recorded != tail['request']:
        return dict(result, status='INVALID_RECORDED_TAIL', recorded_tail=recorded,
                    observed_tail=tail['request'])
    result['tail_victim'] = tail['request']
    if any(not finite(row.get('held_blocks')) for row in rows):
        return dict(result, status='UNKNOWN_CAPACITY')
    signal = 'host_missing_suffix_blocks' if rule == 'host_missing' else 'original_arrival_time'
    if any(not finite(row.get(signal)) for row in rows):
        return dict(result, status='UNKNOWN_HOST_STATE' if rule == 'host_missing'
                    else 'UNKNOWN_ORIGINAL_ARRIVAL')
    key = ((lambda row: (row[signal], -row['running_index'])) if rule == 'host_missing'
           else (lambda row: (-row[signal], -row['running_index'])))
    selected = min(rows, key=key)
    better = [row for row in rows if key(row) < key(tail)]
    return dict(result, status='KNOWN', changed=selected['request'] != tail['request'],
        selected_victim=selected['request'],
        tail_held_blocks=tail['held_blocks'], selected_held_blocks=selected['held_blocks'],
        capacity_delta_blocks=selected['held_blocks']-tail['held_blocks'],
        tail_signal=tail[signal], selected_signal=selected[signal],
        signal_delta=selected[signal]-tail[signal],
        exact_capacity_opportunity=any(row['held_blocks'] == tail['held_blocks'] for row in better),
        selected_exact_capacity=selected['held_blocks'] == tail['held_blocks'])


def host_size_diagnostics(event, host_shadow):
    rows = event.get('candidates', [])
    result = dict(block_size_tokens=16, all_known_ready_prefix_zero=None,
                  uniform_ready_prefix=None, size_winners={}, equal_size_strata={})
    valid = bool(rows) and all(isinstance(row, dict) and row.get('request')
        and finite(row.get('running_index')) for row in rows)
    valid = valid and len({row['request'] for row in rows}) == len(rows)
    if not valid:
        return dict(result, status='NO_LEGAL_CANDIDATES' if not rows else 'UNKNOWN_CANDIDATE_IDENTITY_OR_ORDER')
    result['status'] = 'OBSERVED'
    prefixes = [row.get('host_ready_prefix_blocks') for row in rows]
    if all(finite(value) and value >= 0 for value in prefixes):
        result['all_known_ready_prefix_zero'] = all(value == 0 for value in prefixes)
        result['uniform_ready_prefix'] = len(set(prefixes)) == 1
    sizes = {}
    for label, field in (('computed_whole_pages', 'computed_tokens'), ('held_pages', 'held_blocks')):
        values = [row.get(field) for row in rows]
        if any(not finite(value) or value < 0 for value in values):
            result['size_winners'][label] = dict(status='UNKNOWN_SIZE', selected_victim=None,
                                                host_different=None)
            continue
        values = [int(value // 16) for value in values] if field == 'computed_tokens' else values
        sizes[label] = values
        winner = min(range(len(rows)), key=lambda i: (values[i], -rows[i]['running_index']))
        result['size_winners'][label] = dict(status='KNOWN', selected_victim=rows[winner]['request'],
            host_different=(rows[winner]['request'] != host_shadow['selected_victim']
                            if host_shadow['status'] == 'KNOWN' else None))
    strata_sizes = dict(sizes)
    if len(sizes) == 2:
        strata_sizes['held_and_computed_whole_pages'] = list(zip(sizes['held_pages'], sizes['computed_whole_pages']))
    for label in ('held_pages', 'computed_whole_pages', 'held_and_computed_whole_pages'):
        if label not in strata_sizes or host_shadow['status'] != 'KNOWN':
            result['equal_size_strata'][label] = dict(status='UNKNOWN_SIZE_OR_HOST',
                multiple_candidate_strata=None, different_host_missing_strata=None)
            continue
        groups = defaultdict(list)
        for size, row in zip(strata_sizes[label], rows):
            groups[size].append(row['host_missing_suffix_blocks'])
        repeated = [values for values in groups.values() if len(values) > 1]
        result['equal_size_strata'][label] = dict(status='KNOWN',
            multiple_candidate_strata=len(repeated),
            different_host_missing_strata=sum(len(set(values)) > 1 for values in repeated))
    return result


def summarize_host_size(rows):
    result = dict(observations=len(rows), status_counts=dict(Counter(row['status'] for row in rows)),
        ready_prefix_known_decisions=sum(row['uniform_ready_prefix'] is not None for row in rows),
        all_known_ready_prefix_zero_decisions=sum(row['all_known_ready_prefix_zero'] is True for row in rows),
        uniform_ready_prefix_decisions=sum(row['uniform_ready_prefix'] is True for row in rows),
        size_comparisons={}, equal_size_strata={})
    for label in ('computed_whole_pages', 'held_pages'):
        choices = [row['size_winners'].get(label, {}) for row in rows]
        result['size_comparisons'][label] = dict(
            size_known_decisions=sum(row.get('status') == 'KNOWN' for row in choices),
            host_comparable_decisions=sum(row.get('host_different') is not None for row in choices),
            host_vs_size_different_decisions=sum(row.get('host_different') is True for row in choices))
    for label in ('held_pages', 'computed_whole_pages', 'held_and_computed_whole_pages'):
        known = [row['equal_size_strata'][label] for row in rows
                 if row['equal_size_strata'].get(label, {}).get('status') == 'KNOWN']
        result['equal_size_strata'][label] = dict(known_decisions=len(known),
            decisions_with_multiple_candidate_strata=sum(row['multiple_candidate_strata'] > 0 for row in known),
            decisions_with_different_host_missing=sum(row['different_host_missing_strata'] > 0 for row in known),
            multiple_candidate_strata=sum(row['multiple_candidate_strata'] for row in known),
            different_host_missing_strata=sum(row['different_host_missing_strata'] for row in known))
    return result


def shadow_summary(events):
    raw_shadows = []
    for index, event in enumerate(events):
        host = shadow(event, 'host_missing')
        raw_shadows.append(dict(observation_index=index, step=event.get('step'),
            target=event.get('target'),
            host_missing=host, max_original_arrival=shadow(event, 'arrival'),
            host_size=host_size_diagnostics(event, host)))
    summaries = {}
    for rule in ('host_missing', 'max_original_arrival'):
        rows = [event[rule] for event in raw_shadows]
        known = [row for row in rows if row['status'] == 'KNOWN']
        changed = [row for row in known if row['changed']]
        summaries[rule] = dict(status_counts=dict(Counter(row['status'] for row in rows)),
            known_observations=len(known), changed_vs_tail=len(changed),
            changed_fraction_known=len(changed)/len(known) if known else None,
            exact_capacity_opportunities=sum(row['exact_capacity_opportunity'] for row in known),
            changed_exact_capacity=sum(row['selected_exact_capacity'] for row in changed),
            changed_capacity_delta_blocks=distribution([row['capacity_delta_blocks'] for row in changed]),
            changed_signal_delta=distribution([row['signal_delta'] for row in changed]))
    return dict(rules=summaries, raw_shadows=raw_shadows,
                host_size=summarize_host_size([row['host_size'] for row in raw_shadows]))


def archive_dir(directory):
    archive = directory / 'archive'
    return archive if archive.exists() else directory / 'output'


def store_observations(store, raw_recorded_elsewhere=False):
    observations = store.get('funding_victim_decisions')
    if observations is None:
        return dict(status='NOT_RECORDED', observations=None, shadows=None)
    if not isinstance(observations, list) or any(not isinstance(row, dict) for row in observations):
        return dict(status='INVALID_OBSERVATION_SCHEMA', observations=observations, shadows=None)
    result = dict(status='RECORDED', source_key='funding_victim_decisions',
        coverage='All recorded successful funding-selector calls; gates rejected before selection have no candidate snapshot.',
        observation_count=len(observations),
        candidate_count=distribution([len(row.get('candidates', [])) for row in observations]),
        host_state_errors=dict(Counter(row.get('host_state_error')
            for event in observations for row in event.get('candidates', [])
            if row.get('host_state_error'))),
        shadows=shadow_summary(observations))
    if raw_recorded_elsewhere:
        result['raw_observation_location'] = 'funding_selector.raw_decisions'
    else:
        result['observations'] = observations
    return result


def policy_observations(store):
    gates = {key: value for key, value in store.items()
             if key.endswith('_gate_counts') or key == 'native_reservation_gate'}
    counters = {key: value for key, value in store.items()
                if type(value) in (int, float) and any(word in key for word in
                    ('admission', 'commit', 'cancel', 'reject', 'rotation', 'episode', 'anchor', 'retired'))}
    rejected = [row for row in store.get('events', []) if row.get('reason') and
                ('reject' in str(row.get('event', '')).lower()
                 or 'cancel' in str(row.get('event', '')).lower()
                 or str(row['reason']).startswith('CANCEL_'))]
    return dict(recorded_gate_counts=gates, recorded_counters=counters,
        rejection_event_counts=dict(Counter(f"{row.get('event')}:{row['reason']}" for row in rejected)),
        rejection_events=rejected,
        semantics='Absent counters are unrecorded. Gate counts and rejection events can overlap; they are not summed.')


def analyze(session, timeout_s=600):
    plan = optional(session / 'plan.json', {})
    if plan.get('kind') != 'PRO6000_NORMAL_CAPACITY':
        raise ValueError('Requires a PRO6000_NORMAL_CAPACITY plan')
    specs = plan.get('cells', [])
    labels = [cell['label'] for cell in specs]
    if not specs or len(labels) != len(set(labels)) or plan.get('arms') != labels:
        raise ValueError('Missing cells, duplicate labels, or cell order differs from arms')
    if not finite(timeout_s) or timeout_s <= 0:
        raise ValueError('Invalid timeout')
    cells, profiles = {}, {}
    for index, spec in enumerate(specs):
        name = f"cell-{index:02d}-{spec['label']}"
        directory = session / name
        if Path(name).name != name:
            raise ValueError('Invalid cell label')
        archive = archive_dir(directory)
        cell = None
        try:
            profile = optional(archive / 'normal_capacity_profile.json')
            if spec.get('profile_only'):
                engine_status = optional(archive / 'status.json')
                profiles[name] = dict(directory=str(directory), plan_cell=spec,
                    status=('PROFILE_COMPLETE' if profile and profile.get('status') == 'PROFILE_COMPLETE'
                            and (engine_status or {}).get('status') == 'PROFILE_COMPLETE'
                            else 'NOT_RUN' if not directory.exists() else 'PROFILE_INCOMPLETE'),
                    engine_status=engine_status, capacity_profile=profile,
                    request_measurement='SKIPPED_PROFILE_ONLY')
                continue
            expected = spec.get('requests')
            if type(expected) is not int or expected <= 0:
                raise ValueError('Measurement cell requires a positive requests count')
            cell, raw = read_cell(directory, spec['victim_rule'], expected, timeout_s)
            cell.update(plan_cell=spec, input_case=spec['input_case'],
                        capacity_profile=profile, repeats=1)
            store_path = archive / 'selective-store.json'
            store = optional(store_path, {})
            cell['store_observation_status'] = 'RECORDED' if store_path.exists() else 'NOT_RECORDED'
            cell['pressure'] = dict(status='RECORDED' if 'resource_boundary_summary' in store else 'NOT_RECORDED',
                resource_boundary_summary=store.get('resource_boundary_summary'))
            cell['policy_observations'] = policy_observations(store)
            cell['funding_candidate_observation'] = store_observations(store, 'funding_selector' in cell)
            cells[name] = cell
        except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
            result = cell if cell is not None else dict(directory=str(directory), plan_cell=spec)
            result.update(status='OBSERVATION_ANALYSIS_ERROR' if cell is not None else 'ARTIFACT_READ_ERROR',
                          error=f'{type(error).__name__}: {error}')
            (profiles if spec.get('profile_only') else cells)[name] = result
    complete = bool(cells) and all(row['status'] == 'COMPLETE' for row in cells.values())
    complete &= all(row['status'] == 'PROFILE_COMPLETE' for row in profiles.values())
    return dict(schema_version=1, session=str(session), semantics=SEMANTICS,
        analysis_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        shared_analysis_code_sha256=hashlib.sha256(Path(__file__).with_name('analyze.py').read_bytes()).hexdigest(),
        receipt=optional(session / 'receipt.json'), plan=plan, timeout_s=timeout_s,
        status='CHARACTERIZATION_COMPLETE' if complete else 'PARTIAL_OR_FAILED',
        profiles=profiles, cells=cells)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--timeout-s', type=float, default=600)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.session, args.timeout_s)
    with args.output.open('x') as output:
        json.dump(result, output, indent=2, ensure_ascii=False, allow_nan=False)
        output.write('\n')
