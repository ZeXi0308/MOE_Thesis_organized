"""CPU-only checks for a preassigned single-recovery experiment.

Usage: python analyze_one_event.py LOCAL_GROUP [--target-spec FROZEN_SPEC]
Writes LOCAL_GROUP/one_event_summary.json; stdout is a compact receipt.
Imports analyze.py for all service/transfer/work metrics and full-token pairs.
The extra checks below do not execute a selector or load torch/vLLM. Missing,
duplicate, incomplete, or inconsistent targets stay visible and fail validation.
Exit 0 means the recorded comparison checks passed, not a performance benefit,
KV/logit correctness, semantic quality, or an online adaptive-policy result.
"""
import argparse
from bisect import bisect_left, bisect_right
import hashlib
from itertools import combinations
import json
from pathlib import Path

import analyze as base


def difference(left, right):
    """Compact exact sequence comparison, retaining the first discrepancy."""
    index = next((i for i, pair in enumerate(zip(left, right)) if pair[0] != pair[1]),
                 min(len(left), len(right)))
    equal = left == right
    return dict(equal=equal, left_count=len(left), right_count=len(right),
        left_sha256=base.fingerprint(left), right_sha256=base.fingerprint(right),
        first_difference=None if equal else dict(index=index,
            left=left[index] if index < len(left) else None,
            right=right[index] if index < len(right) else None))


def audit_store_replay(config, raw, warmup, commits):
    """Check the optional single-entry engineering intervention, not a selector."""
    present = ('store_replay_enabled' in config or 'store_replay_modes' in config or
               'store_replay' in raw or 'store_replay' in (warmup or {}))
    report = dict(enabled=config.get('store_replay_enabled', False),
                  legacy_off=not present, failures=[], records=[])
    errors = report['failures']
    if not present:
        report.update(executed_count=0, valid=True)
        return report
    enabled = config.get('store_replay_enabled')
    if type(enabled) is not bool:
        errors.append('config_enabled_not_explicit_boolean')
    modes = config.get('store_replay_modes')
    if (enabled is True or isinstance(modes, str) and 'on' in modes.split(',')) and config.get('policy') != 'recompute':
        errors.append('replay_probe_requires_recompute_in_both_arms')
    replay = raw.get('store_replay')
    if not isinstance(replay, dict):
        errors.append('formal_record_missing')
        report['valid'] = False
        return report
    report.update(records=replay.get('records'), executed_count=replay.get('executed_count'),
                  recorded_errors=replay.get('errors'))
    if type(replay.get('enabled')) is not bool or replay.get('enabled') != enabled:
        errors.append('config_raw_enabled_mismatch')
    if replay.get('errors') != []:
        errors.append('recorded_errors_or_missing_error_list')
    warm = (warmup or {}).get('store_replay')
    if (not isinstance(warm, dict) or warm.get('enabled') is not False or
            warm.get('records') != [] or warm.get('errors') != [] or
            type(warm.get('executed_count')) is not int or warm['executed_count'] != 0):
        errors.append('warmup_not_verified_replay_off')
    records, count = replay.get('records'), replay.get('executed_count')
    if not isinstance(records, list) or type(count) is not int:
        errors.append('invalid_records_or_execution_count')
    elif enabled is not True:
        if records or count != 0:
            errors.append('disabled_replay_must_have_no_records_or_executions')
    else:
        executed = [r for r in records if isinstance(r, dict) and r.get('executed') is True]
        targets = [r for r in records if isinstance(r, dict) and
                   (r.get('target_selected') or r.get('requested') or r.get('target_commit_observed'))]
        report.update(executed_records=len(executed), target_records=len(targets))
        if (count != 1 or len(executed) != 1 or len(targets) != 1 or executed != targets or
                any(not isinstance(r, dict) or r.get('status') not in ('executed', 'skipped') or
                    (r.get('status') == 'executed') != (r.get('executed') is True) for r in records)):
            errors.append('enabled_replay_requires_exactly_one_executed_record')
        target = [c for c in commits if c.get('target_committed')]
        if len(target) != 1:
            errors.append('replay_requires_unique_target_commit')
        elif len(targets) == 1:
            record, commit = targets[0], target[0]
            fields = ('event', 'request_id', 'decision_s', 'preemptions', 'known_tokens',
                      'generated_tokens', 'prefix_sha256', 'host_hit_tokens', 'action')
            if any(record.get(k) != commit.get(k) or k not in record for k in fields):
                errors.append('replay_target_event_or_history_mismatch')
            if (record.get('native_allocation_s') != commit.get('allocation_s') or
                    record.get('phase') != 'service'):
                errors.append('replay_allocation_time_or_service_phase_mismatch')
            flags = ('requested', 'target_selected', 'eligible', 'joint_capacity', 'target_commit_observed',
                     'request_state_preserved', 'known_prefix_matches_record')
            if not all(record.get(k) is True for k in flags):
                errors.append('replay_state_or_target_flags_not_true')
            if (record.get('action') != 'recompute' or record.get('native_external_tokens') != 0 or
                    record.get('fallback') != 'none' or commit.get('actual_action') != 'recompute' or
                    commit.get('external_tokens') != 0 or commit.get('joint_capacity') is not True or
                    commit.get('eligible') is not True):
                errors.append('replay_not_joint_legal_native_recompute')
            if any(record.get(k + '_after') != commit.get(k)
                   for k in ('known_tokens', 'generated_tokens', 'prefix_sha256')):
                errors.append('replay_changed_known_progress')
            before, after, ready = (record.get(k) for k in
                                    ('cursor_before', 'cursor_after', 'ready_prefix_chunk_index'))
            if (not all(type(x) is int for x in (before, after, ready)) or
                    not before > after == ready > 0):
                errors.append('replay_not_a_strict_rewind_to_ready_prefix')
            chunk, hit = record.get('tokens_per_chunk'), commit.get('host_hit_tokens')
            if (type(chunk) is not int or chunk <= 0 or type(hit) is not int or
                    hit <= 0 or hit % chunk or ready != hit // chunk):
                errors.append('replay_ready_prefix_chunk_not_verified')
    report['valid'] = not errors
    return report


def audit_cell(path, group, spec, service):
    raw = base.read_json(path)
    cell = path.parent
    status = base.read_json(cell / 'status.json', {})
    config = base.read_json(cell / 'config.json', {})
    selection = raw.get('target_selection', {})
    commits, decisions = raw.get('commits', []), raw.get('decisions', [])
    flagged = [c for c in commits if c.get('target_committed')]
    selected_commits = [c for c in commits if c.get('target_selected')]
    report = dict(cell=str(cell.relative_to(group)), raw_path=str(path), status=status,
        configured_target_action=config.get('policy'), target_selection=selection,
        observed_target_commits=len(flagged), observed_selected_commits=len(selected_commits),
        selected_lookup_attempts=sum(bool(e.get('target_selected')) for e in decisions),
        target_commits=flagged, failures=[], target_valid=False)
    errors = report['failures']
    report['store_replay'] = audit_store_replay(
        config, raw, base.read_json(cell / 'warmup.json'), commits)
    errors.extend('store_replay.' + e for e in report['store_replay']['failures'])
    if path.name != 'raw.json' or status.get('status') != 'COMPLETE':
        errors.append('episode_not_COMPLETE_raw')
    if service is None:
        errors.append('not_accepted_by_base_service_analysis')
    if base.read_json(cell / 'target_spec.json') != spec:
        errors.append('cell_frozen_spec_missing_or_mismatched')
    if len(flagged) != 1 or len(selected_commits) != 1:
        errors.append('target_commit_count_must_be_exactly_one')
    for label, summary in [('raw', selection), ('status', status)]:
        expected = dict(selection_mode='one_event', target_commit_count=1,
                        target_reached=True, target_result='TARGET_COMMITTED',
                        target_comparison_eligible=True,
                        target_effective_action=config.get('policy'))
        for key, value in expected.items():
            if summary.get(key) != value:
                errors.append(f'{label}.{key}_mismatch')
    parent_status = base.read_json(cell.parent / 'status.json', {})
    if 'target_results' in parent_status and parent_status['target_results'].get(cell.name) != selection:
        errors.append('parent_target_results_mismatch')
    non_target_violations = [dict(event=e.get('event'), request_id=e.get('request_id'))
        for e in decisions + commits if not e.get('target_selected') and
        (e.get('effective_policy') != 'host' or e.get('action') == 'recompute')]
    report['non_target_native_host_violations'] = non_target_violations
    if non_target_violations:
        errors.append('non_target_did_not_retain_native_Host_policy')
    if len(flagged) != 1:
        return report, None

    target, commit = spec['target'], flagged[0]
    rows = {r['request_id']: r for r in raw['requests']}
    if len(rows) != len(raw['requests']):
        raise ValueError('duplicate internal request IDs')
    row = rows[commit['request_id']]
    observed = dict(external_id=row['external_id'], num_preemptions=commit.get('preemptions'),
        known_tokens=commit.get('known_tokens'), generated_tokens=commit.get('generated_tokens'),
        prefix_sha256=commit.get('prefix_sha256'))
    if observed != target or commit.get('external_id') != target['external_id']:
        errors.append('committed_target_identity_or_history_mismatch')
    if commit.get('actual_action') != config.get('policy') or commit.get('action') != config.get('policy'):
        errors.append('target_action_not_preassigned_action')
    if any(commit.get(k) != config.get('policy') for k in ('policy', 'effective_policy')):
        errors.append('target_policy_record_mismatch')
    if commit.get('selection_mode') != 'one_event' or commit.get('target_reason') != 'matched_legal_target':
        errors.append('target_mode_or_reason_mismatch')
    required_flags = ('target_selected', 'target_state_match', 'joint_capacity',
                      'eligible', 'request_state_preserved')
    if not all(commit.get(k) is True for k in required_flags):
        errors.append('target_legality_or_state_flags_not_true')
    if (commit.get('host_hit_tokens', 0) or 0) <= 0 or commit.get('fallback') != 'none':
        errors.append('target_has_no_positive_ready_Host_hit_or_has_fallback')
    if commit.get('target_commit_count_before') != 0 or commit.get('target_commit_count_after') != 1:
        errors.append('target_latch_transition_not_zero_to_one')
    if commit['full_required_blocks'] + commit['reserved_blocks'] + commit['watermark_blocks'] > commit['free_blocks']:
        errors.append('joint_capacity_arithmetic_failed')
    groups = base.read_json(cell / 'resources.json', {}).get('group_config', [])
    block = groups[0].get('tokens_per_block') if len(groups) == 1 else None
    if not block or (commit['known_tokens'] + block - 1) // block != commit['full_required_blocks']:
        errors.append('full_required_blocks_not_independently_verified')
    external = commit.get('external_tokens', 0)
    if (config.get('policy') == 'host' and external <= 0) or (config.get('policy') == 'recompute' and external != 0):
        errors.append('allocated_external_tokens_do_not_match_action')
    matching = [d for d in decisions if base.event_key(d) == base.event_key(commit)]
    if len(matching) != 1 or not matching[0].get('target_selected'):
        errors.append('target_commit_has_no_unique_selected_lookup')
    elif any(matching[0].get(k) != commit.get(k) for k in
             ('decision_s', 'preemptions', 'known_tokens', 'generated_tokens', 'prefix_sha256', 'action')):
        errors.append('target_lookup_commit_changed')

    when, allocated = commit['decision_s'], commit['allocation_s']
    generated = commit['generated_tokens']
    inputs = base.read_json(cell / 'inputs.json', [])
    source_index = next(i for i, r in enumerate(raw['requests']) if r['request_id'] == commit['request_id'])
    prompt = inputs[source_index]['prompt_token_ids']
    history = prompt + row['output_token_ids'][:generated]
    reconstructed = hashlib.sha256(json.dumps(history).encode()).hexdigest()
    report['target_prefix_reconstructed_sha256'] = reconstructed
    if (reconstructed != target['prefix_sha256'] or len(history) != target['known_tokens'] or
            len(prompt) != commit.get('prompt_tokens')):
        errors.append('target_prefix_reconstruction_failed')
    if bisect_right(row['token_times_s'], when) != generated or bisect_right(row['token_times_s'], allocated) != generated:
        errors.append('observed_output_progress_at_decision_or_allocation_mismatch')
    selected_attempts = sorted((d for d in decisions if d.get('target_selected')),
                               key=lambda d: d['decision_s'])
    first = selected_attempts[0] if selected_attempts else commit
    before_when = first['decision_s']
    report['first_selected_lookup'] = first
    report['selected_retries_before_commit'] = sum(d['decision_s'] < when for d in selected_attempts)
    if any(d['decision_s'] > allocated for d in selected_attempts):
        errors.append('selected_lookup_after_target_commit')
    steps, all_schedules = raw['steps'], raw['scheduler_steps']
    starts = [s['start_s'] for s in steps]
    schedules, extra_schedules = [[] for _ in steps], []
    for schedule in all_schedules:
        index = bisect_right(starts, schedule['time_s']) - 1
        if index >= 0 and schedule['time_s'] <= steps[index]['end_s']:
            schedules[index].append(schedule)
        else:
            extra_schedules.append(schedule)
    if any(len(s) != 1 for s in schedules):
        errors.append('service_step_must_have_one_schedule_snapshot')
        return report, None
    schedules = [s[0] for s in schedules]
    report['extra_schedule_snapshots_outside_service_steps'] = len(extra_schedules)
    if extra_schedules and any(s['time_s'] <= steps[-1]['end_s'] for s in extra_schedules):
        errors.append('unmapped_schedule_snapshot_before_service_end')
    boundaries = [i for i, step in enumerate(steps) if step['start_s'] <= before_when < step['end_s']]
    commit_boundaries = [i for i, step in enumerate(steps) if step['start_s'] <= when < step['end_s']]
    if len(boundaries) != 1 or len(commit_boundaries) != 1:
        errors.append('cannot_identify_unique_target_step')
        return report, None
    boundary, commit_boundary = boundaries[0], commit_boundaries[0]
    report['first_selected_step_index_zero_based'] = boundary
    report['target_commit_step_index_zero_based'] = commit_boundary
    if report['store_replay']['enabled'] is True:
        executed_replays = [r for r in report['store_replay']['records'] or []
                            if isinstance(r, dict) and r.get('executed') is True]
        if any(type(r.get('step_index')) is not int or r['step_index'] != commit_boundary
               for r in executed_replays):
            reason = 'formal_step_does_not_match_target_commit'
            report['store_replay']['failures'].append(reason)
            report['store_replay']['valid'] = False
            errors.append('store_replay.' + reason)
    if not when <= allocated <= schedules[commit_boundary]['time_s'] <= steps[commit_boundary]['end_s']:
        errors.append('target_allocation_schedule_order_invalid')
    if any(s['time_s'] >= before_when for s in schedules[:boundary]):
        errors.append('prior_schedule_not_strictly_before_target')
    if service is not None:
        matched = [c for c in service['committed_event_diagnostics']
                   if base.event_key(c) == base.event_key(commit)]
        report['target_next_output'] = matched
        if len(matched) != 1 or matched[0]['censored'] or matched[0]['next_output_token_index'] != generated:
            errors.append('target_next_output_missing_or_progress_mismatch')
        else:
            output_step = bisect_left([s['end_s'] for s in steps], matched[0]['next_output_s'])
            native = [e for e in schedules[output_step]['scheduled'] if e['request_id'] == commit['request_id']]
            if len(native) != 1 or native[0]['end_computed'] < native[0]['known_tokens']:
                errors.append('target_next_output_without_full_known_history_compute')

    # Preserve native scheduled order, replacing random IDs via explicit rows.
    trace = [dict(free_blocks_after_schedule=s.get('free_blocks_after_schedule'),
        scheduled=[dict(external_id=rows[e['request_id']]['external_id'],
                        **{k: e[k] for k in ('count', 'start_computed', 'end_computed',
                                            'known_tokens', 'generated_tokens')})
                   for e in s['scheduled']]) for s in schedules]
    before = []
    emissions = [[] for _ in steps]
    step_ends = [s['end_s'] for s in steps]
    for request in sorted(rows.values(), key=lambda r: r['external_id']):
        count = bisect_right(request['token_times_s'], before_when)
        completed_before = request.get('completion_s', float('inf')) <= before_when
        before.append(dict(external_id=request['external_id'],
            output_token_ids=request['output_token_ids'][:count], completed=completed_before,
            finish_reason=request.get('finish_reason') if completed_before else None,
            stop_reason=request.get('stop_reason') if completed_before else None))
        for token, time in zip(request['output_token_ids'], request['token_times_s']):
            index = bisect_left(step_ends, time)
            if index == len(steps) or step_ends[index] != time:
                raise ValueError('output timestamp is not an engine step completion')
            emissions[index].append((request['external_id'], token))
    def normalize(events, fields, time_key):
        return [dict(external_id=rows[e['request_id']]['external_id'],
                     **{key: e.get(key) for key in fields})
                for e in events if e[time_key] < before_when]
    recovery_history = normalize(commits,
        ('preemptions', 'known_tokens', 'generated_tokens', 'prefix_sha256',
         'action', 'actual_action', 'fallback', 'host_hit_tokens', 'external_tokens'), 'allocation_s')
    preempt_history = normalize(raw.get('preemptions', []),
        ('known_tokens', 'generated_tokens', 'computed_tokens'), 'time_s')
    context = dict(schedule=trace, output_emissions=emissions,
        before_schedule=trace[:boundary], before_output_emissions=emissions[:boundary],
        before_output_history=before, before_recovery_history=recovery_history,
        before_preemption_history=preempt_history, target_step=boundary)
    report['pre_target_trace_fingerprints'] = {
        key: dict(count=len(value), sha256=base.fingerprint(value))
        for key, value in context.items() if key.startswith('before_')}
    report['target_valid'] = not errors
    return report, context


def analyze_one_event(group, spec):
    group = Path(group).resolve()
    service = base.analyze_group(group)
    by_cell = {c['cell']: c for c in service['cells']}
    # Read failed/partial episodes too. Never silently discard a target failure.
    raw_files = {p.parent: p for p in group.rglob('partial_raw.json')}
    raw_files.update({p.parent: p for p in group.rglob('raw.json')})
    # run_group and the same-engine run_cell container each publish status.
    # Complete episode files can survive a later controller/container failure.
    containers = {group} | {folder.parent for folder in raw_files
                            if folder.parent == group or group in folder.parent.parents}
    group_state_checks = [dict(directory=str(folder.relative_to(group)),
        status=base.read_json(folder / 'status.json', {})) for folder in sorted(containers)]
    group_failures = [dict(check, reason='group_or_same_engine_status_not_COMPLETE')
                      for check in group_state_checks if check['status'].get('status') != 'COMPLETE']
    audits, contexts = [], {}
    for folder, path in sorted(raw_files.items()):
        name = str(folder.relative_to(group))
        try:
            audit, context = audit_cell(path, group, spec, by_cell.get(name))
        except (KeyError, TypeError, ValueError, OSError, IndexError) as error:
            audit, context = dict(cell=name, raw_path=str(path), target_valid=False,
                failures=[f'{type(error).__name__}: {error}']), None
        audits.append(audit)
        if context is not None:
            contexts[name] = context
    pairs = []
    for left, right in combinations(audits, 2):
        a, b = contexts.get(left['cell']), contexts.get(right['cell'])
        pair = dict(left_cell=left['cell'], right_cell=right['cell'],
                    target_comparison_eligible=False)
        if a is not None and b is not None:
            checks = {key: difference(a[key], b[key]) for key in a if key.startswith('before_')}
            left_meta = by_cell.get(left['cell'], {}).get('fingerprints', {})
            right_meta = by_cell.get(right['cell'], {}).get('fingerprints', {})
            metadata = {key: None if left_meta.get(key) is None or right_meta.get(key) is None
                        else left_meta[key] == right_meta[key]
                for key in ('external_workload', 'prompt_token_ids', 'fixed_config', 'engine_args', 'resources')}
            pair.update(pre_target=checks, matching_metadata=metadata,
                full_schedule=difference(a['schedule'], b['schedule']),
                full_output_emissions_by_step=difference(a['output_emissions'], b['output_emissions']),
                post_target_output_emissions=difference(a['output_emissions'][a['target_step']:],
                                                      b['output_emissions'][b['target_step']:]),
                target_comparison_eligible=left['target_valid'] and right['target_valid'] and
                    all(c['equal'] for c in checks.values()) and all(metadata.values()))
        else:
            pair['reason'] = 'one_or_both_target_boundaries_unavailable'
        if group_failures:
            pair.update(target_comparison_eligible=False, reason='group_or_same_engine_not_COMPLETE')
        pairs.append(pair)
    missing = list(service['incomplete_cells'])
    for config_path in sorted(group.rglob('config.json')):
        config = base.read_json(config_path, {})
        if config_path.parent in raw_files or not config.get('policies'):
            continue
        planned = [f'{i:02d}_{p}' for i, p in enumerate(config['policies'].split(','))]
        if config_path.parent.name in planned:
            continue
        for index, policy in enumerate(config['policies'].split(',')):
            folder = config_path.parent / f'{index:02d}_{policy}'
            name = str(folder.relative_to(group))
            if folder not in raw_files and not any(m['cell'] == name for m in missing):
                missing.append(dict(cell=name, reason='planned_arm_has_no_raw_or_partial_raw'))
    passed = bool(audits) and all(a['target_valid'] for a in audits) and not (
        group_failures or missing or service['invalid_cells'] or service['analysis_errors'])
    passed = passed and len(audits) >= 2 and all(p['target_comparison_eligible'] for p in pairs)
    return dict(schema='E.one_event_analysis.v1', group=str(group), validation_passed=passed,
        frozen_target_spec=spec, target_audits=audits, target_pair_checks=pairs,
        missing_or_incomplete_cells=missing, invalid_cells=service['invalid_cells'],
        analysis_errors=service['analysis_errors'], group_status=service['group_status'],
        group_state_checks=group_state_checks, group_failures=group_failures,
        measurement_notes=service['measurement_notes'],
        full_service_cells=service['cells'], greedy_token_consistency=service['greedy_token_consistency'],
        definitions=service['definitions'], interpretation=[
            'Pre-target schedule excludes the step containing the FIRST selected lookup, even when allocation retries precede the unique commit. Earlier tokens and per-step emissions use explicit internal-to-external ID mapping; commit-to-next-output timing uses the committed decision.',
            'Only native service steps enter schedule comparisons; extra drain snapshots are counted separately. Observed trace equality does not establish equality of unrecorded asynchronous, LRU, or device state.',
            'A failed or absent target is retained; its full service, when COMPLETE, remains descriptive but cannot enter a target-action comparison.',
            'The target action was assigned before execution. These are controlled single-action diagnostics, not an online policy gain or a hindsight oracle.',
            'When STORE-cursor replay is enabled, validation additionally requires one executed rewind on the existing jointly legal Recompute target, unchanged known history/progress, and replay disabled during warmup. This is a single-entry engineering-cause probe, not a selector benefit; skipped or unsupported replay does not qualify as an executed intervention.',
            'Output consistency and recorded prefix/progress checks do not establish KV/logit/RNG correctness or task quality. Unequal natural outputs are not equal-work speedups.',
            'No warmup enters formal service metrics. Transfer event sums and overlapping request-gap sums are not additive wall-clock time.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('group', type=Path)
    parser.add_argument('--target-spec', type=Path,
                        default=Path(__file__).with_name('one_event_target_spec.json'))
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if not args.group.is_dir():
        parser.error('group must be an already available local directory')
    spec = base.read_json(args.target_spec)
    if not spec or spec.get('schema') != 'E.one_event_target.v1':
        parser.error('a frozen E.one_event_target.v1 spec is required')
    result = analyze_one_event(args.group, spec)
    output = args.out or args.group / 'one_event_summary.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps(dict(output=str(output), validation_passed=result['validation_passed'],
        target_cells=[dict(cell=a['cell'], valid=a['target_valid'], failures=a['failures'])
                      for a in result['target_audits']],
        missing_or_incomplete_cells=result['missing_or_incomplete_cells'],
        group_failures=result['group_failures'],
        invalid_cells=result['invalid_cells'], analysis_errors=result['analysis_errors']), indent=2))
    return 0 if result['validation_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
