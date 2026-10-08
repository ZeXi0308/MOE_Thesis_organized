"""Analyze measured run_cell.py episodes without executing the GPU.

Usage: python analyze.py GROUP_DIRECTORY [--out GROUP_DIRECTORY/summary.json]

All latency units are seconds. Completion latency uses the fixed external
arrival, never the time engine.add_request happened. Token gaps exclude TTFT.
Transfer times are sums of worker-reported CUDA-event durations, not wall time.
Every committed decision independently matches its request's first token time
strictly after decision_s; multiple decisions may therefore share one output.
"""
import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import sys


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def stats(values):
    """Descriptive sample statistics; percentiles use linear interpolation."""
    values = sorted(float(x) for x in values if x is not None)
    if any(not math.isfinite(x) for x in values):
        raise ValueError('nonfinite metric')
    def percentile(q):
        if not values:
            return None
        index = (len(values) - 1) * q
        lo, hi = math.floor(index), math.ceil(index)
        return values[lo] + (values[hi] - values[lo]) * (index - lo)
    return dict(count=len(values), sum=sum(values),
                mean=sum(values) / len(values) if values else None,
                min=values[0] if values else None, p50=percentile(.50),
                p95=percentile(.95), p99=percentile(.99),
                max=values[-1] if values else None)


def counts(rows, key, include_none=True):
    return dict(sorted(Counter(str(row.get(key, 'missing')) for row in rows
                               if include_none or row.get(key) not in (None, 'none')).items()))


def event_key(event):
    return (event.get('request_id'), event.get('event'))


def scheduled_position_counts(entries, requests):
    """Count repeated native compute positions; do not infer GPU durations."""
    seen = defaultdict(set)
    result = {}
    for entry in entries:
        rid = entry['request_id']
        start, end, count = entry['start_computed'], entry['end_computed'], entry['count']
        if start < 0 or end - start != count:
            raise ValueError(f'{rid}: inconsistent scheduled token range')
        current = set(range(start, end))
        repeated = current & seen[rid]
        historical = {position for position in repeated if position < entry['known_tokens']}
        prompt_tokens = requests.get(rid, {}).get('prompt_tokens')
        prompt_repeated = sum(position < prompt_tokens for position in historical) if prompt_tokens is not None else 0
        row = result.setdefault(rid, dict(request_id=rid,
            external_id=requests.get(rid, {}).get('external_id'), scheduled_tokens=0,
            unique_scheduled_positions=0, repeated_scheduled_positions=0,
            known_history_recomputed_tokens=0, repeated_prompt_positions=0,
            repeated_generated_history_positions=0))
        row['scheduled_tokens'] += count
        row['repeated_scheduled_positions'] += len(repeated)
        row['known_history_recomputed_tokens'] += len(historical)
        row['repeated_prompt_positions'] += prompt_repeated
        row['repeated_generated_history_positions'] += len(historical) - prompt_repeated
        seen[rid].update(current)
        row['unique_scheduled_positions'] = len(seen[rid])
    fields = ('scheduled_tokens', 'unique_scheduled_positions', 'repeated_scheduled_positions',
              'known_history_recomputed_tokens', 'repeated_prompt_positions',
              'repeated_generated_history_positions')
    return dict(totals={field: sum(row[field] for row in result.values()) for field in fields},
                by_request=list(result.values()))


def capacity_pressure(raw, resources):
    resources = resources or {}
    steps = raw.get('steps', [])
    sample_sources = dict(
        after_step=[x['free_blocks'] for x in steps if x.get('free_blocks') is not None],
        after_schedule=[x['free_blocks_after_schedule'] for x in raw.get('scheduler_steps', [])
                        if x.get('free_blocks_after_schedule') is not None],
        before_preempt=[x['free_blocks_before_preempt'] for x in raw.get('preemptions', [])
                        if x.get('free_blocks_before_preempt') is not None])
    free = [value for values in sample_sources.values() for value in values]
    gpu_blocks, host_blocks = resources.get('gpu_blocks'), resources.get('host_blocks')
    minimum_free = min(free) if free else None
    peak_used = gpu_blocks - minimum_free if gpu_blocks is not None and minimum_free is not None else None
    resident = [x['host_resident_blocks'] for x in steps if x.get('host_resident_blocks') is not None]
    host_peak = max(resident) if resident else None
    decisions, commits = raw.get('decisions', []), raw.get('commits', [])
    preempted = {x['request_id'] for x in raw.get('preemptions', [])}
    ready = [x for x in decisions if x.get('host_hit_tokens') is not None and x['host_hit_tokens'] > 0]
    eligible = [x for x in decisions if x.get('eligible')]
    eligible_commits = [x for x in commits if x.get('eligible')]
    host_commits = [x for x in commits if x.get('actual_action') == 'host']
    def domain(events):
        request_ids = {x['request_id'] for x in events}
        return dict(events=len(events), distinct_requests=len(request_ids),
                    fraction_of_preempted_requests=len(request_ids & preempted) / len(preempted) if preempted else None)
    return dict(gpu_capacity_blocks=gpu_blocks, gpu_capacity_bytes=resources.get('gpu_bytes'),
        capacity_mode=resources.get('capacity_mode'), initial_free_gpu_blocks=resources.get('free_gpu_blocks'),
        minimum_observed_free_gpu_blocks=minimum_free, peak_observed_used_gpu_blocks=peak_used,
        peak_observed_used_gpu_fraction=peak_used / gpu_blocks if peak_used is not None and gpu_blocks else None,
        free_gpu_samples_by_boundary={name: dict(count=len(values), minimum=min(values) if values else None)
                                      for name, values in sample_sources.items()},
        host_capacity_blocks=host_blocks, host_capacity_bytes=resources.get('host_bytes'),
        peak_observed_host_resident_blocks=host_peak,
        peak_observed_host_resident_fraction=host_peak / host_blocks if host_peak is not None and host_blocks else None,
        peak_observed_running=max((x['running'] for x in steps if 'running' in x), default=None),
        peak_observed_waiting=max((x['waiting'] for x in steps if 'waiting' in x), default=None),
        distinct_preempted_requests=len(preempted),
        fraction_of_requests_preempted=len(preempted) / len(raw['requests']) if raw['requests'] else None,
        host_ready_decision_domain=domain(ready), jointly_eligible_decision_domain=domain(eligible),
        jointly_eligible_commit_domain=domain(eligible_commits), actual_host_commit_domain=domain(host_commits),
        interpretation='Observed GPU occupancy at schedule/step/preempt boundaries; Host residency and running/waiting at step boundaries. This is neither a cap-based demand upper bound nor a complete GPU allocator high-water mark.')


def analyze_cell(raw_path, group):
    raw = read_json(raw_path)
    cell = raw_path.parent
    config = read_json(cell / 'config.json', {})
    inputs = read_json(cell / 'inputs.json')
    engine_args = read_json(cell / 'engine_args.json')
    resources = read_json(cell / 'resources.json')
    warmup = read_json(cell / 'warmup.json')
    rows = raw['requests']
    decisions, commits = raw.get('decisions', []), raw.get('commits', [])
    warnings = []
    by_internal = {}
    by_external = {}
    request_metrics = []
    all_gaps = []
    completion_latencies, ttfts, admission_lags = [], [], []
    caps = []
    for row_index, row in enumerate(rows):
        rid, external = row['request_id'], row['external_id']
        if rid in by_internal or external in by_external:
            raise ValueError('duplicate internal or external request ID')
        by_internal[rid], by_external[external] = row, row
        times = row['token_times_s']
        if len(times) != len(row['output_token_ids']):
            raise ValueError(f'{external}: token/time length mismatch')
        if any(a > b for a, b in zip(times, times[1:])):
            raise ValueError(f'{external}: nonmonotonic token times')
        arrival = row['arrival_s']
        completed = bool(row.get('completed'))
        completion = row.get('completion_s') if completed else None
        if completed and completion is None:
            raise ValueError(f'{external}: completed without completion_s')
        if completion is not None and (completion < arrival or (times and completion < times[-1])):
            raise ValueError(f'{external}: invalid completion timestamp')
        if times and times[0] < arrival:
            raise ValueError(f'{external}: output precedes external arrival')
        latency = completion - arrival if completion is not None else None
        ttft = times[0] - arrival if times else None
        admission_lag = row.get('admitted_s', arrival) - arrival
        gaps = [b - a for a, b in zip(times, times[1:])]
        all_gaps.extend(gaps)
        completion_latencies.append(latency)
        ttfts.append(ttft)
        admission_lags.append(admission_lag)
        source = inputs[row_index] if inputs is not None and row_index < len(inputs) else {}
        cap = row.get('max_output_tokens', source.get('output_tokens', config.get('output_tokens')))
        if cap is not None:
            cap = int(cap)
            if cap <= 0 or len(times) > cap:
                raise ValueError(f'{external}: invalid output cap or observed output exceeds cap')
        caps.append(cap)
        request_metrics.append(dict(request_id=rid, external_id=external,
            source_index=row.get('source_index'), arrival_s=arrival,
            admitted_s=row.get('admitted_s'), completion_s=completion,
            completed=completed, completion_latency_s=latency, ttft_s=ttft,
            output_tokens=len(times), max_output_tokens=cap, token_gap_s=stats(gaps),
            finish_reason=row.get('finish_reason'), stop_reason=row.get('stop_reason'),
            output_token_sha256=fingerprint(row['output_token_ids'])))

    matched_commits = []
    shared_output_counts = Counter()
    for commit in commits:
        item = dict(commit)
        row = by_internal.get(commit.get('request_id'))
        item.update(external_id=row.get('external_id') if row else None,
                    next_output_s=None, next_output_token_index=None,
                    decision_to_next_output_s=None, allocation_to_next_output_s=None,
                    censored=True, censor_reason=None)
        if row is None:
            item['censor_reason'] = 'internal_request_id_not_found'
        elif commit.get('decision_s') is None:
            item['censor_reason'] = 'missing_decision_timestamp'
        else:
            index = bisect_right(row['token_times_s'], commit['decision_s'])
            if index == len(row['token_times_s']):
                item['censor_reason'] = 'no_strictly_later_output_observed'
            else:
                next_time = row['token_times_s'][index]
                allocated = commit.get('allocation_s')
                if allocated is not None and next_time < allocated:
                    item['censor_reason'] = 'next_output_precedes_allocation'
                    warnings.append(f'event {commit.get("event")}: next output precedes allocation')
                else:
                    item.update(next_output_s=next_time, next_output_token_index=index,
                        decision_to_next_output_s=next_time-commit['decision_s'],
                        allocation_to_next_output_s=next_time-allocated if allocated is not None else None,
                        censored=False)
                    shared_output_counts[(commit['request_id'], index)] += 1
        matched_commits.append(item)
    for item in matched_commits:
        item['commits_sharing_next_output'] = shared_output_counts.get(
            (item.get('request_id'), item['next_output_token_index']), 0)
        item['later_commit_before_next_output'] = False
    per_request_commits = defaultdict(list)
    for item in matched_commits:
        per_request_commits[item.get('request_id')].append(item)
    for request_commits in per_request_commits.values():
        ordered = sorted(request_commits, key=lambda x: x.get('decision_s', math.inf))
        for left, right in zip(ordered, ordered[1:]):
            if not left['censored'] and right.get('decision_s', math.inf) < left['next_output_s']:
                left['later_commit_before_next_output'] = True

    commit_index = defaultdict(list)
    for commit in matched_commits:
        commit_index[event_key(commit)].append(commit)
    if any(len(items) > 1 for items in commit_index.values()):
        warnings.append('multiple commits for the same internal request/event key')
    event_diagnostics = []
    for event in decisions:
        matching = commit_index.get(event_key(event), [])
        item = dict(event, committed=bool(matching), commit_count=len(matching))
        item['external_id'] = by_internal.get(event.get('request_id'), {}).get('external_id')
        if len(matching) == 1:
            item.update(matching[0])
        event_diagnostics.append(item)
    decision_keys = {event_key(event) for event in decisions}
    for commit in matched_commits:
        if event_key(commit) not in decision_keys:
            warnings.append(f'commit {commit.get("event")} has no decision record')
            event_diagnostics.append(dict(commit, committed=True, commit_count=1))

    grouped_events = defaultdict(list)
    for commit in matched_commits:
        key = (commit.get('known_tokens'), commit.get('pending_load_jobs'),
               commit.get('running'), commit.get('actual_action'), bool(commit.get('eligible')))
        grouped_events[key].append(commit)
    state_diagnostics = []
    for key, events in sorted(grouped_events.items(), key=lambda item: repr(item[0])):
        known, pending, running, action, eligible = key
        state_diagnostics.append(dict(known_tokens=known, pending_load_jobs=pending,
            running=running, actual_action=action, eligible=eligible, commits=len(events),
            recent_step_s=stats(x.get('recent_step_s') for x in events),
            decision_to_next_output_s=stats(x['decision_to_next_output_s'] for x in events),
            censored=sum(x['censored'] for x in events)))

    recovery_by_action = {}
    for action in sorted({x.get('actual_action', 'missing') for x in matched_commits}):
        action_events = [x for x in matched_commits if x.get('actual_action', 'missing') == action]
        eligible_events = [x for x in action_events if x.get('eligible')]
        recovery_by_action[action] = dict(commits=len(action_events),
            eligible_commits=len(eligible_events), censored=sum(x['censored'] for x in action_events),
            decision_to_next_output_s=stats(x['decision_to_next_output_s'] for x in action_events),
            eligible_decision_to_next_output_s=stats(x['decision_to_next_output_s'] for x in eligible_events),
            allocation_to_next_output_s=stats(x['allocation_to_next_output_s'] for x in action_events))

    transfers = {}
    for direction in ('store', 'load'):
        reported = [x[direction] for x in raw.get('transfers', []) if direction in x]
        sizes = [size for x in reported for size in x.get('sizes', [])]
        transfers[direction] = dict(bytes=sum(x.get('bytes', 0) for x in reported),
            reported_transfer_time_s=sum(x.get('time', 0) for x in reported),
            observed_transfer_count=len(sizes), reported_sizes_bytes=stats(sizes),
            reporting_records=len(reported))

    arrivals = [x['arrival_s'] for x in rows]
    completions = [x['completion_s'] for x in rows if x.get('completed') and 'completion_s' in x]
    all_completed = bool(rows) and len(completions) == len(rows)
    makespan = max(completions) - min(arrivals) if all_completed else None
    total_tokens = sum(len(x['output_token_ids']) for x in rows)
    preemptions = raw.get('preemptions', [])
    per_request_preempts = Counter(x['request_id'] for x in preemptions)
    scheduler_entries = [item for step in raw.get('scheduler_steps', []) for item in step.get('scheduled', [])]
    # This is scheduled native work, not a decomposition into measured GPU time.
    scheduled_tokens = sum(x['count'] for x in scheduler_entries)
    position_counts = scheduled_position_counts(scheduler_entries, by_internal)
    eligible = [x for x in matched_commits if x.get('eligible')]
    fixed_config = {k: v for k, v in config.items() if k not in ('out', 'policy', 'threshold', 'workload', 'max_seconds')}
    workload_identity = [dict(external_id=x['external_id'], arrival_s=x['arrival_s'],
        prompt_tokens=x.get('prompt_tokens'), source_index=x.get('source_index'),
        max_output_tokens=cap) for x, cap in zip(rows, caps)]
    if inputs is not None:
        input_identity = [x.get('prompt_token_ids') for x in inputs]
    else:
        input_identity = None
        warnings.append('inputs.json unavailable: prompt identity cannot be verified across cells')
    warm_identity = None if warmup is None else [dict(external_id=x['external_id'],
        arrival_s=x['arrival_s'], prompt_tokens=x.get('prompt_tokens'),
        output_tokens=len(x['output_token_ids'])) for x in warmup.get('requests', [])]
    summary = dict(cell=str(cell.relative_to(group)), policy=config.get('policy', decisions[0].get('policy') if decisions else 'unknown'),
        threshold=config.get('threshold'), config=config, status=read_json(cell / 'status.json', {}),
        raw_path=str(raw_path), requests=len(rows), completed_requests=len(completions),
        all_requests_completed=all_completed, output_tokens=total_tokens,
        output_work=dict(natural_stopping=config.get('natural'),
            total_max_output_tokens=sum(caps) if all(cap is not None for cap in caps) else None,
            requests_with_known_output_cap=sum(cap is not None for cap in caps),
            output_caps=stats(caps), actual_output_tokens=total_tokens,
            requests_reaching_output_cap=sum(cap is not None and len(row['output_token_ids']) == cap
                                            for row, cap in zip(rows, caps)),
            length_finished_requests=sum(x.get('finish_reason') == 'length' for x in rows),
            interpretation='Caps are requested maxima, not observed work. Natural stopping may generate different token counts across arms; elapsed-time changes then do not establish an equal-output-work speedup.'),
        capacity_pressure=capacity_pressure(raw, resources),
        makespan_s=makespan, all_complete_s=raw.get('all_complete_s'),
        service_and_drain_s=raw.get('service_and_drain_s'),
        output_tokens_per_s=total_tokens / makespan if makespan and makespan > 0 else None,
        completion_latency_s=stats(completion_latencies), ttft_s=stats(ttfts),
        admission_lag_s=stats(admission_lags), token_gap_s=stats(all_gaps),
        steps_s=stats(x['end_s'] - x['start_s'] for x in raw.get('steps', [])),
        scheduled_native_tokens=scheduled_tokens,
        scheduled_position_counts=position_counts,
        decision_count=len(decisions), decision_actions=counts(decisions, 'action'),
        decision_fallbacks=counts(decisions, 'fallback', include_none=False),
        decision_fallback_count=sum(x.get('fallback') not in (None, 'none') for x in decisions),
        eligible_decisions=sum(bool(x.get('eligible')) for x in decisions),
        committed_count=len(commits), committed_actions=counts(commits, 'actual_action'),
        committed_chosen_actions=counts(commits, 'action'),
        committed_fallbacks=counts(commits, 'fallback', include_none=False),
        committed_fallback_count=sum(x.get('fallback') not in (None, 'none') for x in commits),
        eligible_commits=len(eligible),
        censored_commits=sum(x['censored'] for x in matched_commits),
        censor_reasons=counts([x for x in matched_commits if x['censored']], 'censor_reason'),
        decision_to_next_output_s=stats(x['decision_to_next_output_s'] for x in matched_commits),
        eligible_decision_to_next_output_s=stats(x['decision_to_next_output_s'] for x in eligible),
        recovery_by_action=recovery_by_action,
        commits_sharing_output_count=sum(n for n in shared_output_counts.values() if n > 1),
        distinct_next_outputs_matched=len(shared_output_counts),
        preemptions=len(preemptions), preempted_requests=len(per_request_preempts),
        preemptions_by_request=dict(per_request_preempts), transfers=transfers,
        lookup_overhead_s=stats(x.get('lookup_seconds') for x in decisions),
        adapter_overhead_s=stats(x.get('adapter_seconds') for x in decisions),
        prefix_consistency_reported=raw.get('prefix_consistency'),
        eligible_request_state_preserved_count=sum(x.get('request_state_preserved') is True for x in eligible),
        fingerprints=dict(external_workload=fingerprint(workload_identity),
            prompt_token_ids=fingerprint(input_identity) if input_identity is not None else None,
            fixed_config=fingerprint(fixed_config), engine_args=fingerprint(engine_args) if engine_args is not None else None,
            resources=fingerprint(resources) if resources is not None else None,
            warmup_shape=fingerprint(warm_identity) if warm_identity is not None else None),
        request_metrics=request_metrics, event_diagnostics=event_diagnostics,
        committed_event_diagnostics=matched_commits, state_diagnostics=state_diagnostics,
        warnings=warnings)
    return summary, by_external


def compare_tokens(left, right, left_rows, right_rows):
    common = sorted(left_rows.keys() & right_rows.keys())
    differences = []
    metadata_differences = []
    for external in common:
        a, b = left_rows[external], right_rows[external]
        left_tokens, right_tokens = a['output_token_ids'], b['output_token_ids']
        if left_tokens != right_tokens:
            prefix = next((i for i, pair in enumerate(zip(left_tokens, right_tokens)) if pair[0] != pair[1]),
                          min(len(left_tokens), len(right_tokens)))
            differences.append(dict(external_id=external, left_tokens=len(left_tokens), right_tokens=len(right_tokens),
                first_difference_index=prefix,
                left_token=left_tokens[prefix] if prefix < len(left_tokens) else None,
                right_token=right_tokens[prefix] if prefix < len(right_tokens) else None,
                unequal_token_positions=sum(x != y for x, y in zip(left_tokens, right_tokens)),
                unmatched_tail_tokens=abs(len(left_tokens)-len(right_tokens))))
        differing_fields = [k for k in ('finish_reason', 'stop_reason', 'completed') if a.get(k) != b.get(k)]
        if differing_fields:
            metadata_differences.append(dict(external_id=external, fields=differing_fields))
    comparable = {}
    for key in left['fingerprints']:
        a, b = left['fingerprints'][key], right['fingerprints'][key]
        comparable[key] = None if a is None or b is None else a == b
    same_requests = left_rows.keys() == right_rows.keys()
    same_lengths = same_requests and all(len(left_rows[key]['output_token_ids']) == len(right_rows[key]['output_token_ids'])
                                        for key in common)
    left_caps = {x['external_id']: x['max_output_tokens'] for x in left['request_metrics']}
    right_caps = {x['external_id']: x['max_output_tokens'] for x in right['request_metrics']}
    caps_known = all(value is not None for value in list(left_caps.values()) + list(right_caps.values()))
    return dict(left_cell=left['cell'], right_cell=right['cell'],
        left_policy=left['policy'], right_policy=right['policy'],
        matching_metadata=comparable, compared_requests=len(common),
        identical_output_requests=len(common)-len(differences), different_output_requests=len(differences),
        missing_from_left=sorted(right_rows.keys()-left_rows.keys()),
        missing_from_right=sorted(left_rows.keys()-right_rows.keys()),
        differences=differences, completion_metadata_differences=metadata_differences,
        output_work_comparison=dict(same_request_set=same_requests,
            same_output_caps=left_caps == right_caps if caps_known else None,
            same_actual_output_tokens_per_request=same_lengths,
            left_actual_output_tokens=left['output_tokens'], right_actual_output_tokens=right['output_tokens'],
            interpretation='Same observed output counts per request; this alone does not establish quality or equal internal compute.' if same_lengths else
                           'Observed output work differs; do not interpret elapsed-time differences as an equal-output-work speedup.'),
        interpretation='Greedy output-sequence consistency diagnostic only; neither equality nor difference establishes task quality.')


def aggregate_policies(cells):
    grouped = defaultdict(list)
    for cell in cells:
        grouped[(cell['policy'], cell['threshold'])].append(cell)
    result = []
    for (policy, threshold), members in sorted(grouped.items(), key=lambda item: repr(item[0])):
        def values(field, subfield=None):
            return [x[field][subfield] if subfield else x[field] for x in members]
        metrics = dict(makespan_s=values('makespan_s'), output_tokens_per_s=values('output_tokens_per_s'),
            mean_completion_latency_s=values('completion_latency_s', 'mean'),
            p95_completion_latency_s=values('completion_latency_s', 'p95'),
            p95_token_gap_s=values('token_gap_s', 'p95'), p99_token_gap_s=values('token_gap_s', 'p99'),
            max_token_gap_s=values('token_gap_s', 'max'), sum_token_gap_s=values('token_gap_s', 'sum'),
            mean_eligible_decision_to_next_output_s=values('eligible_decision_to_next_output_s', 'mean'))
        result.append(dict(policy=policy, threshold=threshold, cells=[x['cell'] for x in members],
            runs=len(members), eligible_commits=sum(x['eligible_commits'] for x in members),
            per_cell_values=metrics, descriptive_across_cell_statistics={k: stats(v) for k, v in metrics.items()}))
    return result


def analyze_group(group):
    group = Path(group).resolve()
    files = sorted(group.rglob('raw.json'))
    cells, rows_by_cell, errors, invalid = [], {}, [], []
    for path in files:
        try:
            status = read_json(path.parent / 'status.json', {})
            if status.get('status') != 'COMPLETE':
                invalid.append(dict(cell=str(path.parent.relative_to(group)), raw_path=str(path),
                    status=status, reason='cell_status_not_COMPLETE',
                    partial_raw_available=(path.parent / 'partial_raw.json').is_file()))
                continue
            summary, rows = analyze_cell(path, group)
            if not summary['all_requests_completed']:
                invalid.append(dict(cell=summary['cell'], raw_path=str(path), status=status,
                    reason='raw_has_incomplete_requests', completed_requests=summary['completed_requests'],
                    requests=summary['requests']))
                continue
        except (KeyError, TypeError, ValueError, OSError) as error:
            errors.append(dict(raw_path=str(path), error=f'{type(error).__name__}: {error}'))
            continue
        cells.append(summary)
        rows_by_cell[summary['cell']] = rows
    incomplete = []
    for path in sorted(group.rglob('config.json')):
        if not (path.parent / 'raw.json').exists():
            # The same-engine parent owns config/status but is not an episode.
            if any(path.parent in raw_path.parents for raw_path in files):
                continue
            incomplete.append(dict(cell=str(path.parent.relative_to(group)),
                partial_raw_available=(path.parent / 'partial_raw.json').is_file(),
                status=read_json(path.parent / 'status.json', {'status': 'NO_RAW_JSON'})))
    consistency = [compare_tokens(a, b, rows_by_cell[a['cell']], rows_by_cell[b['cell']])
                   for a, b in combinations(cells, 2)]
    return dict(schema_version=2, group=str(group), group_status=read_json(group / 'status.json'),
        definitions=dict(time_unit='seconds',
            completion_latency='completion_s minus fixed external arrival_s, including delayed admission',
            makespan='last completion minus first fixed external arrival; null unless every request completed',
            throughput='all observed output tokens / makespan',
            output_caps='per-request max_output_tokens, then inputs.json output_tokens, then config output_tokens; requested maxima do not represent actual natural-output work',
            capacity_pressure='actual resource capacities and observed occupancy at schedule/step/preempt boundaries, not a cap-based demand upper bound or complete GPU allocator high-water mark',
            successful_cells='only raw.json episodes with cell status COMPLETE and all requests completed enter cells, policy summaries, and cross-arm comparisons',
            token_gaps='consecutive output-token timestamp differences within each request; excludes TTFT; zero gaps retained',
            token_gap_sum='sum over all requests, may exceed wall time because requests coexist',
            committed_recovery_latency='first token timestamp strictly greater than decision_s minus decision_s, matched by internal request_id',
            repeated_decisions='each commit independently matches the next strictly later output; shared matches are counted and flagged, never treated as independent causal trials',
            censoring='missing internal request, missing timestamp, no later output, or inconsistent allocation timing',
            transfer_time='sum of worker-reported CUDA-event durations in seconds; can overlap compute/other transfers and is NOT additive wall time',
            store_cost='all observed STORE bytes/time retained, including drain; no previously paid STORE cost is subtracted from recompute',
            scheduler_work='scheduled native token count is work accounting, not measured compute duration',
            repeated_positions='scheduled positions already scheduled earlier for the same internal request; repeated occurrences are counted, known-history counts restrict positions below known_tokens at that step',
            state_diagnostics='descriptive observations only; across-run action outcomes are not an online oracle',
            policy_aggregation='descriptive statistics across per-cell metrics; no confidence interval or pooled percentile claim',
            percentiles='linear interpolation at (n - 1) * q',
            quality='greedy output differences are consistency diagnostics only; no task-quality claim'),
        cells=cells, policy_summary=aggregate_policies(cells),
        greedy_token_consistency=consistency, incomplete_cells=incomplete,
        invalid_cells=invalid, analysis_errors=errors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('group', type=Path)
    parser.add_argument('--out', type=Path, help='Default: GROUP/summary.json')
    args = parser.parse_args()
    if not args.group.is_dir():
        parser.error(f'group directory does not exist: {args.group}')
    result = analyze_group(args.group)
    output = args.out or args.group / 'summary.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(result, indent=2, allow_nan=False, ensure_ascii=False) + '\n'
    output.write_text(serialized)
    sys.stdout.write(serialized)
    if result['analysis_errors'] or not result['cells']:
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
