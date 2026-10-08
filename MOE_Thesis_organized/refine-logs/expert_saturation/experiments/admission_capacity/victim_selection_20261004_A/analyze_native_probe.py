#!/usr/bin/env python3
"""Single-run native victim observations, with all-request outcomes preserved."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import analyze_pro as shared
from analyze import distribution, optional, read


def count_known(value):
    return type(value) is int and value >= 0


def adapt_native(event):
    """Explicit schema mapping only; never infer host state or original arrival."""
    return dict(step=event.get('step'), target=event.get('failed_request'),
        baseline_victim=event.get('native_tail'), selected_victim=event.get('selected'),
        candidates=[dict(row, running_index=row.get('index'),
                         original_arrival_time=row.get('arrival_time'))
                    for row in event.get('candidates', [])])


def tail_equal_held(event, match_computed_pages=False):
    rows = event.get('candidates', [])
    tails = [row for row in rows if row.get('request') == event.get('native_tail')]
    result = dict(status='UNKNOWN_TAIL_OR_CAPACITY', alternatives=None,
        different_host_missing=None, lower_host_missing_than_tail=None,
        different_ready_prefix=None, different_pending_dependencies=None)
    if len(tails) != 1 or any(not count_known(row.get('held_blocks')) for row in rows):
        return result
    tail = tails[0]
    same = [row for row in rows if row['held_blocks'] == tail['held_blocks']]
    if match_computed_pages:
        if any(not count_known(row.get('computed_tokens')) for row in same):
            return dict(result, status='UNKNOWN_COMPUTED_PAGES')
        same = [row for row in same
                if row['computed_tokens']//16 == tail['computed_tokens']//16]
    result.update(status='KNOWN_CAPACITY', held_blocks=tail['held_blocks'], alternatives=len(same)-1)
    for field, output in (('host_missing_suffix_blocks', 'different_host_missing'),
                          ('host_ready_prefix_blocks', 'different_ready_prefix'),
                          ('pending_native_store_dependencies', 'different_pending_dependencies')):
        if all(count_known(row.get(field)) for row in same):
            result[output] = any(row[field] != tail[field] for row in same)
            if field == 'host_missing_suffix_blocks':
                result['lower_host_missing_than_tail'] = any(row[field] < tail[field] for row in same)
    return result


def summarize_equal_held(rows):
    result = dict(status_counts=dict(Counter(row['status'] for row in rows)),
        known_capacity_decisions=sum(row['alternatives'] is not None for row in rows),
        decisions_with_alternative=sum(row['alternatives'] is not None and row['alternatives'] > 0 for row in rows))
    for field in ('different_host_missing', 'lower_host_missing_than_tail',
                  'different_ready_prefix', 'different_pending_dependencies'):
        eligible = [row for row in rows if row['alternatives'] is not None and row['alternatives'] > 0]
        result[field] = dict(known_with_alternative=sum(row[field] is not None for row in eligible),
            unknown_with_alternative=sum(row[field] is None for row in eligible),
            decisions=sum(row[field] is True for row in eligible))
    return result


def pending_summary(decisions):
    values = [row.get('pending_native_store_dependencies')
              for event in decisions for row in event.get('candidates', [])]
    known = [value for value in values if count_known(value)]
    complete = []
    for event in decisions:
        counts = [row.get('pending_native_store_dependencies') for row in event.get('candidates', [])]
        if counts and all(count_known(value) for value in counts):
            complete.append(counts)
    return dict(candidate_observations=len(values), known_candidate_observations=len(known),
        unknown_candidate_observations=len(values)-len(known),
        zero_dependencies=sum(value == 0 for value in known),
        positive_dependencies=sum(value > 0 for value in known),
        dependencies=distribution(known), all_candidates_known_decisions=len(complete),
        varying_dependencies_decisions=sum(len(set(values)) > 1 for values in complete),
        semantics='Registered native STORE dependencies only; None includes missing/invalid mapping or non-STORE jobs. Repeated candidate observations are not independent. No DMA duration or exposed wait is measured.')


def timing_summary(decisions, duration):
    result = dict(semantics='candidate_observation_wall_s includes host, pending-dependency and progress-field reads and is contained in selector_wall_s. Do not sum the two. Separate host/pending timing is NOT_RECORDED.')
    for field in ('selector_wall_s', 'candidate_observation_wall_s'):
        values = [event.get(field) for event in decisions]
        known = [value for value in values if shared.finite(value) and value >= 0]
        total = sum(known) if known else None
        result[field] = dict(recorded_decisions=len(known), unknown_decisions=len(values)-len(known),
            total_recorded_s=total, distribution=distribution(known),
            fraction_capture=total/duration if total is not None and duration and duration > 0 else None)
    return result


def release_joins(decisions, raw):
    if raw is None:
        return dict(status='RAW_UNAVAILABLE')
    actual = [event for event in raw.get('preemption_events', [])
              if event.get('original_preemption_called') is True
              and event.get('original_preemption_returned') is True]
    by_key = defaultdict(list)
    for event in actual:
        by_key[(event.get('engine_call_index'), event.get('internal_request_id'))].append(event)
    keys = [(event.get('step'), event.get('selected')) for event in decisions]
    key_counts = Counter(keys)
    joined, used = [], set()
    for index, (decision, key) in enumerate(zip(decisions, keys)):
        matches = by_key.get(key, [])
        result = dict(decision_index=index, step=key[0], selected=key[1],
                      matching_actual_preemptions=len(matches), matching_decisions=key_counts[key])
        if key[0] is None or key[1] is None:
            result['join_status'] = 'UNKNOWN_DECISION_IDENTITY'
        elif len(matches) != 1 or key_counts[key] != 1:
            result['join_status'] = 'MISSING_OR_AMBIGUOUS_JOIN'
        else:
            event = matches[0]
            used.add(key)
            candidates = [row for row in decision.get('candidates', []) if row.get('request') == key[1]]
            held = candidates[0].get('held_blocks') if len(candidates) == 1 else None
            released = event.get('actual_released_blocks')
            before, after = event.get('free_blocks_before_preempt'), event.get('free_blocks_after_preempt')
            free_known = count_known(before) and count_known(after)
            delta_consistent = after-before == released if free_known and count_known(released) else None
            comparable = count_known(held) and count_known(released) and delta_consistent is not False
            result.update(join_status='UNIQUE', actual_preemption=event,
                selected_held_blocks=held, actual_released_blocks=released,
                free_counter_delta_consistent=delta_consistent,
                capacity_status='INCONSISTENT_FREE_COUNTERS' if delta_consistent is False
                    else 'KNOWN' if comparable else 'UNKNOWN_CAPACITY',
                released_minus_held_blocks=released-held if comparable else None)
        joined.append(result)
    known = [row for row in joined if row.get('capacity_status') == 'KNOWN']
    return dict(status='ANALYZED', actual_preemption_events=len(actual),
        excluded_unsuccessful_or_unconfirmed_events=len(raw.get('preemption_events', []))-len(actual),
        join_status_counts=dict(Counter(row['join_status'] for row in joined)),
        capacity_status_counts=dict(Counter(row.get('capacity_status', 'NO_UNIQUE_JOIN') for row in joined)),
        capacity_comparable_decisions=len(known),
        released_equals_held=sum(row['released_minus_held_blocks'] == 0 for row in known),
        released_less_than_held=sum(row['released_minus_held_blocks'] < 0 for row in known),
        released_more_than_held=sum(row['released_minus_held_blocks'] > 0 for row in known),
        actual_released_blocks=distribution([row['actual_released_blocks'] for row in known]),
        released_minus_held_blocks=distribution([row['released_minus_held_blocks'] for row in known]),
        actual_without_unique_native_decision=[event for event in actual
            if (event.get('engine_call_index'), event.get('internal_request_id')) not in used],
        joins=joined,
        semantics='Exact step + internal request ID joins only. Released capacity is the free-pool counter difference around the native preempt call; its method duration excludes later native flush and is not DMA wait.')


def native_summary(store, raw):
    decisions = store.get('victim_decisions')
    if decisions is None:
        return dict(status='NOT_RECORDED')
    if not isinstance(decisions, list) or any(not isinstance(event, dict) for event in decisions):
        return dict(status='INVALID_DECISION_SCHEMA', raw_decisions=decisions)
    normalized = [adapt_native(event) for event in decisions]
    shadows = shared.shadow_summary(normalized)
    equal = [tail_equal_held(event) for event in decisions]
    equal_computed = [tail_equal_held(event, match_computed_pages=True) for event in decisions]
    for shadow, values, computed_values in zip(shadows['raw_shadows'], equal, equal_computed):
        shadow['tail_equal_held'] = values
        shadow['tail_equal_held_and_computed_pages'] = computed_values
    candidates = [row for event in decisions for row in event.get('candidates', [])]
    valid_pairs = [event for event in decisions if event.get('selected') is not None and event.get('native_tail') is not None]
    return dict(status='ANALYZED', decisions=len(decisions),
        recorded_rules=dict(Counter(event.get('rule', 'UNKNOWN') for event in decisions)),
        selected_tail_known_decisions=len(valid_pairs),
        actual_selected_equals_tail=sum(event['selected'] == event['native_tail'] for event in valid_pairs),
        candidate_count=distribution([len(event.get('candidates', [])) for event in decisions]),
        host_state_errors=dict(Counter(row['host_state_error'] for row in candidates if row.get('host_state_error'))),
        field_mapping={'index': 'running_index', 'arrival_time': 'original_arrival_time',
                       'native_tail': 'baseline_victim', 'selected': 'selected_victim',
                       'failed_request': 'target (label only; this is the allocation-failed request)'},
        mapping_semantics='computed_tokens and both host fields come directly from candidate rows; absent values stay UNKNOWN. No fallback to output limits or later events.',
        raw_decisions=decisions, shadows=shadows,
        tail_equal_held=summarize_equal_held(equal),
        tail_equal_held_and_computed_pages=summarize_equal_held(equal_computed),
        pending_native_store_dependencies=pending_summary(decisions),
        observed_cpu_cost=timing_summary(decisions, raw.get('observation_end_s') if raw else None),
        actual_release=release_joins(decisions, raw))


def analyze(session, timeout_s=600):
    result = shared.analyze(session, timeout_s)
    if len(result['cells']) != 1:
        raise ValueError('Requires exactly one measured native observation cell')
    result['semantics'] = dict(result['semantics'],
        causality='Single native-tail observation run, no policy comparison or measured strategy gain.',
        repeats='One run. Candidate rows and preemption events are dependent observations, not independent repeats.',
        cost='Native selector wall includes candidate observation wall; host and pending reads are not timed separately. Metadata dependencies are not exposed DMA time.')
    result['shared_pro_analysis_code_sha256'] = result['analysis_code_sha256']
    result['analysis_code_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for cell in result['cells'].values():
        try:
            archive = shared.archive_dir(Path(cell['directory']))
            store = optional(archive / 'selective-store.json', {})
            path = next((archive / name for name in ('raw.json', 'raw.json.gz') if (archive / name).exists()), None)
            cell['native_victim_observation'] = native_summary(store, read(path) if path else None)
        except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
            cell['native_victim_observation'] = dict(status='ANALYSIS_ERROR', error=f'{type(error).__name__}: {error}')
        if cell['native_victim_observation']['status'] != 'ANALYZED':
            result['status'] = 'PARTIAL_OR_FAILED'
    return result


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
