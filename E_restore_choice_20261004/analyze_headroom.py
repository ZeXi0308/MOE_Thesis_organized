"""CPU analysis for the frozen H/L/G/G/L/H experiment (G=headroom).

Usage: python analyze_headroom.py LOCAL_GROUP [--out REPORT.json] [--profile NAME]
Writes headroom_summary.json by default. Reuses analyze.py once for all actual
service/work/transfer/output metrics; does not execute a selector or import vLLM.
Exit 0: validated COMPLETE group; 2: UNRUN/incomplete; 1: invalid observations.
Old raw without headroom snapshots is UNRUN. Rule replay is action-only on each
observed trajectory, never a counterfactual latency or a single-event trial.
"""
import argparse
from bisect import bisect_right
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import analyze as base


EXPECTED_ARMS = ['host', 'length', 'headroom', 'headroom', 'length', 'host']
DEFAULT_PROFILE = 'summary256_headroom6'
PROFILES = {
    DEFAULT_PROFILE: dict(order=EXPECTED_ARMS),
    'multinews320_headroom8': dict(
        order=['host', 'recompute', 'length', 'headroom', 'headroom', 'length', 'recompute', 'host'],
        requests=320, max_num_seqs=320,
        input_path=Path(__file__).resolve().parent / 'inputs_multinews320/pro6000_multinews_candidate_320.json',
        input_sha256='21c601c5aa8bda312ba3a4f10c88b8cc34bbecfbc2fb15f98daf2ac53f707995'),
}
# Expected source identities only. Digests come from an explicit, independently
# frozen receipt, never from the current mutable source or the run under audit.
RUNTIME_SOURCE_SUFFIXES = (
    '/vllm/v1/core/sched/scheduler.py',
    '/vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py',
    '/vllm/v1/kv_offload/cpu/manager.py', '/vllm/v1/core/kv_cache_manager.py',
    '/model/config.json', '/run_cell.py', '/selector.py', '/timing_observer.py',
)


def source_receipt_binding(path, name, profile):
    if 'input_path' not in profile:
        return dict(status='NOT_REQUESTED', errors=[])
    result = dict(status='INVALID', errors=[], runtime_source_hashes={})
    try:
        if path is None:
            raise ValueError('explicit --source-receipt required for this profile')
        data = Path(path).read_bytes()
        receipt = json.loads(data)
        if not isinstance(receipt, dict):
            raise ValueError('receipt must be a JSON object')
        expected = dict(schema='E.headroom_source_receipt.v1', profile=name,
                        input_sha256=profile['input_sha256'], expected_order=profile['order'],
                        config=dict(requests=profile['requests'], max_num_seqs=profile['max_num_seqs'], threshold=1792))
        if not all(receipt.get(k) == v for k, v in expected.items()):
            raise ValueError('receipt profile/input/order/config mismatch')
        hashes = receipt['runtime_source_hashes']
        if not isinstance(hashes, dict) or set(hashes) != set(RUNTIME_SOURCE_SUFFIXES) or not all(
                isinstance(v, str) and len(v) == 64 and all(c in '0123456789abcdef' for c in v) for v in hashes.values()):
            raise ValueError('receipt requires eight named runtime source suffixes and SHA256 digests')
        result.update(status='VALID', path=str(Path(path).resolve()), sha256=hashlib.sha256(data).hexdigest(),
                      runtime_source_hashes=hashes)
    except (KeyError, TypeError, ValueError, OSError) as exc:
        result['errors'].append(f'invalid_source_receipt:{type(exc).__name__}:{exc}')
    return result


def profile_binding(cell, raw, profile, receipt):
    """Explicit future profile only; default historical behavior stays unchanged."""
    if 'input_path' not in profile:
        return dict(status='NOT_REQUESTED', errors=[])
    errors = []
    def require(ok, why):
        if not ok:
            errors.append(why)
    try:
        data = profile['input_path'].read_bytes()
        require(hashlib.sha256(data).hexdigest() == profile['input_sha256'], 'frozen_input_hash_mismatch')
        frozen = json.loads(data)['source_requests']
        config = base.read_json(cell / 'config.json', {})
        args = base.read_json(cell / 'engine_args.json', {})
        if not isinstance(config, dict) or not isinstance(args, dict):
            raise ValueError('config and engine args must be JSON objects')
        require(len(frozen) == profile['requests'] and len(raw['requests']) == profile['requests'], 'profile_request_count_mismatch')
        require(base.read_json(cell / 'inputs.json') == frozen, 'episode_inputs_differ_from_frozen_input')
        expected = dict(requests=profile['requests'], max_num_seqs=profile['max_num_seqs'],
                        natural=True, threshold=1792, target_spec=None, timing_observer=True,
                        kv_bytes=None, host_gib=32, batch_tokens=2048,
                        policies=','.join(profile['order']))
        require(all(k in config and config[k] == v for k, v in expected.items()), 'profile_config_mismatch')
        engine_expected = dict(max_num_seqs=profile['max_num_seqs'], max_model_len=4096,
                               gpu_memory_utilization=0.9, max_num_batched_tokens=2048,
                               kv_cache_memory_bytes=None, kv_offloading_size=32)
        require(all(k in args and args[k] == v for k, v in engine_expected.items()), 'profile_engine_resource_mismatch')
        require(all(args[k] is False for k in ('enable_prefix_caching', 'async_scheduling') if k in args),
                'profile_engine_APC_or_async_enabled')
        for i, (row, source) in enumerate(zip(raw['requests'], frozen)):
            expected_row = dict(external_id=f'measured/E{i:03d}', prompt_tokens=len(source['prompt_token_ids']),
                                max_output_tokens=source['output_tokens'], arrival_s=source['arrival_s'])
            require(all(row.get(k) == v for k, v in expected_row.items()), f'profile_raw_input_binding_mismatch:{i}')
        hash_path = next((p / 'runtime_source_hashes.json' for p in (cell, cell.parent)
                          if (p / 'runtime_source_hashes.json').is_file()), None)
        require(hash_path is not None, 'missing_runtime_source_hashes')
        hashes = base.read_json(hash_path, {}) if hash_path is not None else {}
        if not isinstance(hashes, dict):
            raise ValueError('runtime source hashes must be a JSON object')
        require(hashes.get(config.get('workload')) == profile['input_sha256'], 'runtime_workload_hash_mismatch')
        require(receipt['status'] == 'VALID', 'source_receipt_not_validated')
        for suffix, digest in receipt.get('runtime_source_hashes', {}).items():
            require([value for path, value in hashes.items() if path.endswith(suffix)] == [digest],
                    f'frozen_runtime_hash_mismatch:{suffix}')
    except (KeyError, TypeError, ValueError, OSError) as exc:
        errors.append(f'malformed_profile_binding:{type(exc).__name__}:{exc}')
    return dict(status='INVALID' if errors else 'VALID', errors=errors,
                input_sha256=profile['input_sha256'],
                limits='Saved complete inputs and trusted runner source hashes bind the profile; raw does not independently attest prompt tokens or executed machine code.')


def integer(x):
    return type(x) is int and x >= 0


def finite(x):
    return type(x) in (int, float) and math.isfinite(x)


def validate_episode(raw, policy, threshold, max_model_len, block_tokens, instrumented_profile=False):
    """Validate serialized observations; unrecorded runtime guards stay unverified."""
    decisions, commits = raw.get('decisions', []), raw.get('commits', [])
    empty = dict(events=[], commits=[], errors=[])
    if not all(isinstance(xs, list) and all(isinstance(x, dict) for x in xs) for xs in (decisions, commits)):
        return dict(empty, status='INVALID', errors=['malformed_decisions_or_commits'])
    if not instrumented_profile and not any(isinstance(e, dict) and e.get('headroom') is not None
                                          for e in decisions + commits):
        return dict(empty, status='UNRUN', reason='no_headroom_snapshot')
    errors, events, committed = [], [], []
    def require(ok, why):
        if not ok:
            errors.append(why)
    if not (integer(threshold) and threshold > 0 and integer(max_model_len)
            and max_model_len > 0 and integer(block_tokens) and block_tokens > 0):
        return dict(empty, status='INVALID', errors=['invalid_config_or_resource_bounds'])
    steps, schedules = raw.get('steps', []), raw.get('scheduler_steps', [])
    if not all(isinstance(s, dict) and finite(s.get('start_s')) and finite(s.get('end_s'))
               and s['start_s'] < s['end_s'] for s in steps):
        return dict(empty, status='INVALID', errors=['malformed_service_steps'])
    starts = [s['start_s'] for s in steps]
    require(all(a['end_s'] <= b['start_s'] for a, b in zip(steps, steps[1:])), 'overlapping_service_steps')
    def local_step(t):
        i = bisect_right(starts, t) - 1 if finite(t) else -1
        return i if i >= 0 and t <= steps[i]['end_s'] else None
    by_step = {}
    for schedule in schedules:
        i = local_step(schedule.get('time_s'))
        if i is not None:
            by_step.setdefault(i, []).append(schedule)
    rows = {r['request_id']: r for r in raw.get('requests', [])}
    require(len(rows) == len(raw.get('requests', [])), 'duplicate_request_id')
    lookup = {}
    for event in decisions:
        key = base.event_key(event)
        tag = f'event {key!r}'
        require(key not in lookup, f'{tag}: duplicate_lookup')
        lookup[key] = event
        if instrumented_profile:
            require(type(event.get('eligible')) is bool and 'headroom' in event and 'headroom_supported' in event,
                    f'{tag}: missing_instrumented_event_fields')
        if not event.get('eligible'):
            require(event.get('headroom') is None and event.get('headroom_supported') is None,
                    f'{tag}: ineligible_snapshot')
            continue
        h = event.get('headroom')
        if not isinstance(h, dict):
            errors.append(f'{tag}: eligible_snapshot_missing')
            continue
        try:
            require(h['schema'] == 'E.headroom.v1', f'{tag}: unknown_schema')
            require(h['block_tokens'] == block_tokens, f'{tag}: block_size_mismatch')
            require(all(integer(h.get(k)) for k in ('block_tokens', 'current_slack_blocks',
                    'background_next_decode_growth_blocks', 'target_next_decode_growth_blocks')),
                    f'{tag}: invalid_snapshot_counts')
            for k in ('known_tokens', 'free_blocks', 'full_required_blocks', 'reserved_blocks',
                      'watermark_blocks', 'scheduler_step', 'running'):
                require(integer(event.get(k)), f'{tag}: invalid_{k}')
            n = event['known_tokens']
            target_row = rows.get(event['request_id'])
            require(target_row is not None, f'{tag}: unknown_target_request')
            if target_row is not None:
                require(n == target_row['prompt_tokens'] + bisect_right(target_row['token_times_s'], event['decision_s']),
                        f'{tag}: target_known_history_mismatch')
            require(n > 0 and event.get('joint_capacity') is True and event.get('host_hit_tokens', 0) > 0,
                    f'{tag}: not_jointly_eligible')
            require(event['full_required_blocks'] == (n + block_tokens - 1) // block_tokens,
                    f'{tag}: full_capacity_mismatch')
            slack = event['free_blocks'] - event['full_required_blocks'] - event['reserved_blocks'] - event['watermark_blocks']
            require(slack >= 0 and h['current_slack_blocks'] == slack, f'{tag}: slack_mismatch')
            i = local_step(event['decision_s'])
            require(i is not None and steps[i]['start_s'] <= event['decision_s'] < steps[i]['end_s'],
                    f'{tag}: decision_outside_service_step')
            native = by_step.get(i, [])
            require(len(native) == 1, f'{tag}: expected_one_native_schedule')
            schedule = native[0] if len(native) == 1 else None
            if schedule is not None:
                require(event['decision_s'] <= schedule['time_s'], f'{tag}: decision_after_schedule')
            entries = schedule['scheduled'] if schedule is not None else []
            native_by_id = {x['request_id']: x for x in entries}
            require(len(native_by_id) == len(entries), f'{tag}: duplicate_native_request')
            running = h['running_requests']
            require(isinstance(running, list), f'{tag}: malformed_running_snapshot')
            require(len(running) == event['running'], f'{tag}: running_count_mismatch')
            require(len({x['request_id'] for x in running}) == len(running), f'{tag}: duplicate_running_request')
            require(event['request_id'] not in {x['request_id'] for x in running}, f'{tag}: target_in_background')
            growth, unsupported, reasons = [], [], {}
            checked = 0
            for r in running:
                rid, known, computed = r['request_id'], r['known_tokens'], r['computed_tokens']
                require(integer(known) and known > 0 and integer(computed), f'{tag}: invalid_background_progress')
                require(type(r['is_prefill_chunk']) is bool and integer(r['next_decode_eligible_step']),
                        f'{tag}: invalid_background_scope_fields')
                require(rid in rows, f'{tag}: unknown_background_request')
                if rid in rows:
                    observed_n = rows[rid]['prompt_tokens'] + bisect_right(rows[rid]['token_times_s'], event['decision_s'])
                    require(known == observed_n, f'{tag}: background_known_history_mismatch')
                why = []
                if known != computed + 1:
                    why.append('not_one_token_ready')
                if event['scheduler_step'] < r['next_decode_eligible_step']:
                    why.append('decode_cadence_skip')
                if known >= max_model_len:
                    why.append('model_length_skip')
                if r['is_prefill_chunk']:
                    why.append('prefill_chunk')
                row = dict(request_id=rid, known_tokens=known, computed_tokens=computed)
                if why:
                    unsupported.append(row)
                    reasons[rid] = why
                else:
                    if known % block_tokens == 0:
                        growth.append(row)
                    entry = native_by_id.get(rid, {})
                    require((entry.get('count'), entry.get('start_computed'), entry.get('end_computed'), entry.get('known_tokens'))
                            == (1, computed, known, known), f'{tag}: supported_background_native_mismatch:{rid}')
                    checked += 1
            supported = not unsupported
            target_growth = int(n % block_tokens == 0)
            margin = slack - len(growth) - target_growth if supported else None
            require(type(h['margin_blocks']) is int if supported else h['margin_blocks'] is None,
                    f'{tag}: invalid_margin_type')
            require(h['supported'] is supported and event.get('headroom_supported') is supported,
                    f'{tag}: ALL_scope_mismatch')
            require(h['background_requests'] == growth and h['unsupported_background_requests'] == unsupported
                    and h['unsupported_background_reasons'] == reasons, f'{tag}: background_membership_or_reasons_mismatch')
            require(h['background_next_decode_growth_blocks'] == len(growth)
                    and h['target_next_decode_growth_blocks'] == target_growth
                    and h['margin_blocks'] == margin, f'{tag}: growth_or_margin_mismatch')
            length = 'recompute' if n <= threshold else 'host'
            g = 'recompute' if supported and n <= threshold and margin >= 0 else 'host'
            reason = ('headroom_background_unsupported' if not supported else
                      'headroom_length_above_threshold' if n > threshold else
                      'headroom_negative_margin' if margin < 0 else 'headroom_nonnegative_margin')
            predicted = {'host': 'host', 'recompute': 'recompute', 'length': length, 'headroom': g}.get(policy)
            require(predicted is not None and event.get('action') == predicted, f'{tag}: selected_action_mismatch')
            require(event.get('policy') == policy and event.get('fallback') == 'none', f'{tag}: eligible_policy_or_fallback_mismatch')
            if policy == 'headroom':
                require(event.get('policy_reason') == reason, f'{tag}: G_policy_reason_mismatch')
            events.append(dict(request_id=event['request_id'], event=event['event'],
                external_id=rows.get(event['request_id'], {}).get('external_id'),
                local_step=i, scheduler_step=event['scheduler_step'], decision_s=event['decision_s'],
                supported=supported, unsupported_reasons=reasons, margin_blocks=margin,
                supported_background_native_checks=checked, known_tokens=n,
                headroom_prediction=g, headroom_prediction_reason=reason, length_prediction=length,
                predictions_disagree=g != length, selected_action=event['action'], policy_reason=event.get('policy_reason')))
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            errors.append(f'{tag}: malformed_snapshot:{type(exc).__name__}:{exc}')
    replay = {base.event_key(e): e for e in events}
    seen = set()
    fields = ('eligible', 'headroom', 'headroom_supported', 'scheduler_step', 'policy_reason',
              'known_tokens', 'free_blocks', 'full_required_blocks', 'reserved_blocks', 'watermark_blocks',
              'joint_capacity', 'host_hit_tokens', 'running', 'action', 'policy', 'fallback', 'decision_s',
              'prefix_sha256', 'prompt_tokens', 'generated_tokens', 'preemptions')
    for commit in commits:
        key = base.event_key(commit)
        require(key not in seen, f'commit {key!r}: duplicate_commit')
        seen.add(key)
        source = lookup.get(key)
        require(source is not None, f'commit {key!r}: orphan_commit')
        if source is not None:
            require(all(source.get(k) == commit.get(k) for k in fields), f'commit {key!r}: lookup_commit_mismatch')
        i = local_step(commit.get('decision_s'))
        native = by_step.get(i, [])
        require(len(native) == 1 and finite(commit.get('allocation_s'))
                and finite(commit.get('decision_s'))
                and commit['decision_s'] <= commit['allocation_s'] <= native[0]['time_s'],
                f'commit {key!r}: invalid_allocation_timing')
        if commit.get('eligible'):
            require(commit.get('actual_action') == commit.get('action'), f'commit {key!r}: actual_action_mismatch')
            require((commit.get('external_tokens', 0) > 0) if commit.get('actual_action') == 'host'
                    else commit.get('external_tokens') == 0,
                    f'commit {key!r}: actual_transfer_action_mismatch')
            if key in replay:
                committed.append(dict(replay[key], actual_action=commit.get('actual_action')))
    def summarize(records):
        return dict(eligible_records=len(records), supported=sum(e['supported'] for e in records),
            unsupported=sum(not e['supported'] for e in records),
            unsupported_reason_occurrences=dict(Counter(reason for e in records for rs in e['unsupported_reasons'].values() for reason in rs)),
            policy_reasons=dict(Counter(e['policy_reason'] for e in records)),
            headroom_vs_length_action_disagreements=sum(e['predictions_disagree'] for e in records))
    return dict(status='INVALID' if errors else 'VALID' if events else 'VALID_NO_ELIGIBLE_EVENTS', errors=errors, events=events, commits=committed,
        event_summary=summarize(events), commit_summary=summarize(committed),
        limits='Serialized flags are checked for logical consistency, not independently reconstructed. Unserialized spec/placeholders/inflight/encoder/LoRA/static guards were asserted by runtime only. Matching supported rows to native schedule does not prove absence of omitted members beyond the recorded running count.')


def recovery_progress(raw, service, audited):
    """Use base next-strict-output matching; add only the immediately following gap."""
    rows = {r['request_id']: r for r in raw['requests']}
    diagnostics = {base.event_key(e): e for e in service['committed_event_diagnostics']}
    result, unique_gaps = [], {}
    for event in audited.get('commits', []):
        d = diagnostics[base.event_key(event)]
        index = d.get('next_output_token_index')
        times = rows.get(d['request_id'], {}).get('token_times_s', [])
        gap = times[index + 1] - times[index] if not d['censored'] and index is not None and index + 1 < len(times) else None
        item = dict(event, **{k: d.get(k) for k in ('decision_to_next_output_s', 'next_output_s',
            'next_output_token_index', 'censored', 'censor_reason', 'commits_sharing_next_output', 'later_commit_before_next_output')},
            subsequent_token_gap_s=gap, subsequent_gap_unavailable=gap is None)
        result.append(item)
        if gap is not None:
            unique_gaps[(d['request_id'], index)] = gap
    return dict(records=result, choice_to_next_output_s=base.stats(e['decision_to_next_output_s'] for e in result),
        subsequent_gap_s_unique_outputs=base.stats(unique_gaps.values()),
        censored_commits=sum(e['censored'] for e in result),
        subsequent_gap_unavailable_commits=sum(e['subsequent_gap_unavailable'] for e in result),
        shared_next_output_commits=sum((e['commits_sharing_next_output'] or 0) > 1 for e in result),
        interpretation='Choice latencies retain every commit, including shared outputs; following-gap summary deduplicates by internal request ID and output index. These are descriptive dependent observations, not independent trials.')


def container_checks(group, raw_files):
    containers = {group} | {p.parent.parent for p in raw_files if p.parent.parent == group or group in p.parent.parent.parents}
    return [dict(directory=str(p.relative_to(group)), status=base.read_json(p / 'status.json', {}).get('status')) for p in sorted(containers)]


def analyze_headroom_group(group, threshold=1792, profile=DEFAULT_PROFILE, source_receipt=None):
    group = Path(group).resolve()
    selected = PROFILES[profile]
    expected_order = selected['order']
    receipt = source_receipt_binding(source_receipt, profile, selected)
    service = base.analyze_group(group)
    cells = []
    for cell in service['cells']:
        path = Path(cell['raw_path'])
        raw = base.read_json(path)
        args = base.read_json(path.parent / 'engine_args.json', {})
        resources = base.read_json(path.parent / 'resources.json', {})
        groups = resources.get('group_config', [])
        block = groups[0].get('tokens_per_block') if len(groups) == 1 else None
        binding = profile_binding(path.parent, raw, selected, receipt)
        instrumented = binding['status'] == 'VALID'
        audit = validate_episode(raw, cell['policy'], threshold, args.get('max_model_len'), block,
                                 instrumented_profile=instrumented)
        if binding['status'] == 'INVALID':
            audit['errors'].extend(binding['errors'])
            audit['status'] = 'INVALID'
        if audit['status'] != 'UNRUN' and cell['policy'] in ('length', 'headroom') and cell['threshold'] != threshold:
            audit['errors'].append('executed_threshold_differs_from_analysis_threshold')
            audit['status'] = 'INVALID'
        progress = recovery_progress(raw, cell, audit) if audit['status'] != 'UNRUN' else None
        cells.append(dict(cell=cell['cell'], policy=cell['policy'], profile_binding=binding,
                          observation_validation=audit, recovery_progress=progress))
    checks = container_checks(group, sorted(group.rglob('raw.json')))
    actual = [c['policy'] for c in service['cells']]
    has_observations = any(c['observation_validation']['status'] != 'UNRUN' for c in cells)
    invalid = bool(receipt['status'] == 'INVALID' or service['analysis_errors'] or service['invalid_cells'] or
                   any(c['observation_validation']['status'] == 'INVALID' for c in cells))
    complete = bool(checks) and all(x['status'] == 'COMPLETE' for x in checks)
    if invalid or any(x['status'] == 'FAILED' for x in checks):
        status = 'INVALID'
    elif not has_observations:
        status = 'UNRUN'
    elif complete and actual == expected_order and not service['incomplete_cells'] and all(
            c['observation_validation']['status'] in ('VALID', 'VALID_NO_ELIGIBLE_EVENTS') for c in cells):
        status = ('VALIDATED_NO_ELIGIBLE_EVENTS' if all(c['observation_validation']['status'] == 'VALID_NO_ELIGIBLE_EVENTS'
                                                      for c in cells) else 'VALIDATED_COMPLETE')
    else:
        status = 'INCOMPLETE'
    return dict(schema='E.headroom_analysis.v1', group=str(group), status=status,
        validation_passed=status in ('VALIDATED_COMPLETE', 'VALIDATED_NO_ELIGIBLE_EVENTS'),
        profile=profile, source_receipt_binding=receipt, expected_order=expected_order,
        headroom_mechanism_exercised=any(c['policy'] == 'headroom' and c['observation_validation']['events'] for c in cells),
        observed_order=actual, container_status_checks=checks, length_threshold=threshold,
        cells=cells, base_analysis=service,
        interpretation='Base analysis preserves actual complete-cell service/cost/output data even when validation fails. G-versus-L replay compares actions on observed trajectories, not counterfactual outcomes or benefit. Default old data without snapshots stays UNRUN. Explicit frozen-profile/source-receipt binding can validate complete zero-eligible execution as VALIDATED_NO_ELIGIBLE_EVENTS: feature not exercised, no G mechanism or performance result.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('group', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--profile', choices=PROFILES, default=DEFAULT_PROFILE)
    parser.add_argument('--source-receipt', type=Path,
                        help='Independently frozen E.headroom_source_receipt.v1 JSON; required for multinews320_headroom8')
    args = parser.parse_args()
    result = analyze_headroom_group(args.group, profile=args.profile, source_receipt=args.source_receipt)
    out = args.out or args.group / 'headroom_summary.json'
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(dict(status=result['status'], validation_passed=result['validation_passed'],
                         cells=len(result['cells']), report=str(out)), ensure_ascii=False))
    return 0 if result['validation_passed'] else (1 if result['status'] == 'INVALID' else 2)


if __name__ == '__main__':
    raise SystemExit(main())
