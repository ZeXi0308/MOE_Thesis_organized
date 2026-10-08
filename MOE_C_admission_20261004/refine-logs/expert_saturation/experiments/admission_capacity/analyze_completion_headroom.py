#!/usr/bin/env python3
"""Recompute fixed-pool native/headroom request outcomes and actual held work."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from analyze_recovery_admission import module, read, require, repreemptions

LABELS = ('repeat0-native', 'repeat0-headroom', 'repeat1-headroom', 'repeat1-native')
FOUR_ARMS = ('native', 'safe29', 'headroom', 'rotate')
FOUR_LABELS = tuple(f'repeat{i}-{p}' for i, arms in ((0, FOUR_ARMS), (1, reversed(FOUR_ARMS))) for p in arms)
ROTATION_CONFIG = dict(min_absence_steps=30, min_steps_between_swaps=20, free_block_slack=0,
                       min_residency_steps=30, protect_progress_fraction=0.9, max_absences_per_request=8, enabled=True)
KV_BYTES, KV_BLOCKS = 16089350144, 7671


def held_accounting(raw, decisions, policy):
    require(len(decisions) == len(raw['scheduler_steps']), 'decision/step count differs')
    ids = raw['internal_to_source']
    traces = {t['attempted_step']: t for t in raw['memory_trace']}
    totals, streaks, longest, intervals = Counter(), Counter(), Counter(), []
    starts, held_ids, seconds = {}, set(), 0.0
    for index, (d, step) in enumerate(zip(decisions, raw['scheduler_steps'])):
        require(d['step'] == step['step'] == index and d['mode'] == ('native' if policy == 'safe29' else policy) and d['status'] == 'APPLIED', 'decision identity/status differs')
        held = {ids[rid] for rid in d['held']}
        actual = {ids[rid]: n for rid, n in d['actual_scheduled'].items()}
        require(actual == {r['request_id']: r['scheduled_tokens'] for r in step['scheduled']}, 'decision/action trace mismatch')
        require(not held.intersection(actual) and len(held) == len(d['held']), 'held request was scheduled or duplicated')
        require(not held or (d['active'] and policy in ('headroom', 'rotate')), 'hold outside active intervention')
        for rid in d['held']:
            before, after = traces[index]['before'], traces[index]['after']
            require(rid in before['running_ids'] and rid in after['running_ids'] and before['requests'][rid] == after['requests'][rid], 'held request state or retained block count changed')
        cost = d['decision_seconds']
        require(isinstance(cost, (int, float)) and math.isfinite(cost) and cost >= 0, 'invalid decision time')
        seconds += cost
        for rid in set(streaks) - held:
            if streaks[rid]:
                intervals.append(dict(request_id=rid, first_step=starts[rid], last_step=index-1, steps=streaks[rid]))
            streaks[rid] = 0
        for rid in held:
            if not streaks[rid]:
                starts[rid] = index
            streaks[rid] += 1
            longest[rid] = max(longest[rid], streaks[rid])
        held_ids.update(held)
        totals.update(held)
    for rid, length in streaks.items():
        if length:
            intervals.append(dict(request_id=rid, first_step=starts[rid], last_step=len(decisions)-1, steps=length))
    require(policy != 'headroom' or held_ids, 'headroom did not perform an action')
    return dict(held_request_ids=sorted(held_ids), held_request_count=len(held_ids), held_request_steps=sum(totals.values()),
                longest_continuous_held_steps=max(longest.values(), default=0), decision_seconds_total=seconds,
                held_steps_by_request=dict(totals), longest_continuous_held_steps_by_request=dict(longest), intervals=intervals,
                active_steps=sum(d['active'] for d in decisions), decision_count=len(decisions),
                scope='Decision CPU wall time excludes subsequent native scheduling and checks; already included in request wall time. Held step counts are not durations or GPU time.')


def preemption_accounting(raw, decisions, events, policy, distribution):
    """Raw native events are primary; decision labels must partition them exactly."""
    kinds, resumed = {}, []
    ids = raw['internal_to_source']
    for d, step in zip(decisions, raw['scheduler_steps']):
        require(policy == 'rotate' or not d.get('forced_preempted'), 'forced action outside rotate arm')
        forced = d['forced_preempted'] if policy == 'rotate' else []
        natural = d['natural_preempted'] if policy == 'rotate' else d['preempted']
        require(len(set(forced+natural)) == len(forced+natural), 'duplicate/overlapping preemption labels')
        actual = Counter(ids[rid] for rid in d['preempted'])
        require(actual == Counter(ids[rid] for rid in forced+natural) == Counter(step['preempted_request_ids']), 'decision/native preemption partition differs')
        for kind, victims in (('forced', forced), ('natural', natural)):
            for rid in victims:
                kinds[(d['step'], ids[rid])] = kind
        for rid in d.get('resumed', []):
            require(rid in d['actual_scheduled'], 'resumed request was not scheduled')
            resumed.append(dict(step=d['step'], request_id=ids[rid]))
    successful = [e for e in events if e['original_preemption_returned']]
    require(Counter((e['attempted_step'], e['request_id']) for e in successful) == Counter(kinds.keys()), 'raw preemptions differ from decisions and scheduler')
    previous, previous_kind, rows = {}, {}, []
    for event in successful:
        step, rid = event['attempted_step'], event['request_id']
        kind, host = kinds[(step, rid)], event['host_call_interval_s'][0]
        prior = previous.get(rid)
        rows.append(dict(request_id=rid, step=step, kind=kind,
            previous_global_gap_steps=step-rows[-1]['step'] if rows else None,
            previous_same_kind_gap_steps=step-previous_kind[kind] if kind in previous_kind else None,
            previous_request_gap_steps=step-prior[0] if prior else None,
            previous_request_gap_s=host-prior[1] if prior else None,
            new_outputs_since_previous=event['victim_state']['output_tokens']-prior[2] if prior else None))
        previous[rid] = (step, host, event['victim_state']['output_tokens'])
        previous_kind[kind] = step
    counts = Counter(r['kind'] for r in rows)
    require(policy != 'rotate' or counts['forced'] > 0, 'rotation performed no forced action')
    require(policy != 'safe29' or not rows, 'safe29 preempted despite full reservation')
    return dict(forced_count=counts['forced'], natural_count=counts['natural'], events=rows, resumed_events=resumed,
        by_kind={kind: dict(count=counts[kind], spacing_steps=distribution([r['previous_same_kind_gap_steps'] for r in rows
                  if r['kind'] == kind and r['previous_same_kind_gap_steps'] is not None])) for kind in ('forced', 'natural')},
        spacing={k: distribution([r[k] for r in rows if r[k] is not None]) for k in
                 ('previous_global_gap_steps', 'previous_request_gap_steps', 'previous_request_gap_s', 'new_outputs_since_previous')},
        scope='Intervals separate successful native preemptions, not absence-to-recovery duration. Forced and natural labels reconcile with raw events and native scheduler output; time is host time.')


def inspect(directory, label, native, frozen, four_arm=False):
    policy = label.split('-')[1]
    row = dict(label=label, policy=policy, status='UNRUN', full_episode_comparison_eligible=False)
    try:
        terminal = read(directory/'status.json') if (directory/'status.json').exists() else None
        row['terminal_status'] = terminal
        raw_path = next((directory/n for n in ('raw.json', 'raw.json.gz') if (directory/n).exists()), None)
        if raw_path is None:
            row['status'] = 'INCOMPLETE' if terminal and terminal.get('status') != 'UNRUN' else 'UNRUN'
            return row
        raw = read(raw_path)
        row.update(status='INCOMPLETE', raw_status=raw['status'], raw_path=str(raw_path),
                   requests=[dict(request_id=r['request_id'], status=r['status'], output_tokens=len(r['output_token_ids'])) for r in raw['requests']])
        names = ('config.json', 'engine_args.json', 'memory-before.json', 'safe-cap-qualification.json', 'metrics.json', 'headroom-decisions.json', 'status.json')
        missing = [n for n in names if not (directory/n).exists()]
        if missing:
            row['missing_artifacts'] = missing
            return row
        config, engine, memory, q, saved, decisions, terminal = [read(directory/n) for n in names]
        require(config['completion_policy'] == policy and config['reservation_policy'] == 'full' and engine['scheduler_reserve_full_isl'] is True, 'policy mismatch')
        cap = 29 if four_arm and policy == 'safe29' else 32
        require(config['cap'] == raw['target_cap'] == cap and config['prompt_tokens'] == 3072 and config['output_tokens'] == 1024, 'workload/cap differs')
        if four_arm:
            require(policy in FOUR_ARMS and config['engine_max_num_seqs'] == engine['max_num_seqs'] == 32, 'four-arm engine capacity differs')
            require(config['rotation_config'] == ROTATION_CONFIG, 'frozen rotation configuration differs')
            require(q['block_size'] == 16 and q['per_request_reserved_blocks'] == 256 and q['safe_cap'] == KV_BLOCKS//256 == 29, 'safe29 reservation not qualified')
        require(engine['max_num_batched_tokens'] == 1024 and engine['kv_cache_memory_bytes'] == memory['kv_storage_bytes'] == KV_BYTES, 'physical/token budget differs')
        require(q['status'] == 'QUALIFIED' and q['usable_blocks'] == KV_BLOCKS and q['observed_scheduler_reserve_full_isl'] is True, 'live pool/reservation unqualified')
        require(raw['preemption_mode'] == config['preemption_mode'] == 'native_recompute', 'preemption mode differs')
        identity = native.base.validate_identity(raw, config, frozen/'inputs_preparation/prepared', 'long', cap)
        metrics = native.base.summarize_episode_requests(raw['requests'], observation_end_s=raw['observation_end_s'], ttft_slo_s=5.0, tpot_slo_s=0.2)
        require(all(k in saved and saved[k] == v for k, v in metrics.items()), 'saved metrics differ from raw recomputation')
        work, calls = native.execution_accounting(raw)
        events = native.preemptions(raw, calls)
        effects = native.request_effects(raw, events, work)
        pools = [t[k]['pool'] for t in raw['memory_trace'] for k in ('before', 'after') if t[k] is not None]
        pools += [e[k] for e in raw['preemption_events'] for k in ('pool', 'pool_after') if k in e]
        pools += [a[k] for t in raw['memory_trace'] for a in t['allocation_failures'] for k in ('before', 'after')]
        require(pools and all(0 <= p['used_blocks'] <= p['usable_blocks'] == KV_BLOCKS and p['used_blocks']+p['free_blocks'] == KV_BLOCKS for p in pools), 'block conservation failed')
        complete = raw['status'] == terminal['status'] == 'COMPLETE' and identity['all_requests_completed'] and metrics['n_completed'] == 32 and raw['capacity_boundary'] is None
        held = held_accounting(raw, decisions, policy) if complete else dict(status='PARTIAL_NOT_QUALIFIED', retained_decisions=len(decisions))
        indexed = {r['request_id']: r for r in metrics['per_request']}
        original = {r['request_id']: r for r in raw['requests']}
        requests = [dict(r, **{k: indexed[r['request_id']][k] for k in ('ttft_s', 'tpot_s', 'queue_s')}) for r in effects['per_request']]
        for r in requests:
            r.update({k: original[r['request_id']][k] for k in ('arrival_s', 'completion_s', 'prompt_token_ids_sha256', 'internal_request_id')})
        latencies = [r['completion_latency_s'] for r in requests if r['completion_latency_s'] is not None]
        row.update(status='COMPLETE' if complete else 'INCOMPLETE', full_episode_comparison_eligible=complete,
                   metrics=metrics, identity_check=identity, requests=requests, effects=effects, work=work, held=held,
                   mean_completion_s=sum(latencies)/len(latencies) if latencies else None,
                   max_itl_s=max((r['longest_itl']['itl_s'] for r in requests if r['longest_itl']), default=None),
                   preemptions=raw['actual_preemption_count'], preemption_events=events, repeated_preemption=repreemptions(events),
                   engine_args=engine, config=config, physical_kv_storage_bytes=KV_BYTES, usable_blocks=KV_BLOCKS,
                   token_timing_scope=raw.get('host_chunk_diagnostics', {'token_level_itl_resolved': 'UNVERIFIED'}), saved_metrics_match=True,
                   output_hashes={r['request_id']: hashlib.sha256(json.dumps(r['output_token_ids']).encode()).hexdigest() for r in raw['requests']})
        if four_arm and complete:
            row['preemption_accounting'] = preemption_accounting(raw, decisions, events, policy, native.distribution)
    except (OSError, json.JSONDecodeError) as exc:
        row.update(status='INCOMPLETE', error=str(exc))
    except (KeyError, ValueError, TypeError, IndexError) as exc:
        row.update(status='INVALID_EVIDENCE', error=str(exc), full_episode_comparison_eligible=False)
    return row


def compare(a, b, eligible, four_arm=False, distribution=None, additional_config_exclusions=()):
    pair = dict(baseline=a['label'], action=b['label'], status='UNRUN_OR_INCOMPLETE_CAMPAIGN')
    if not eligible:
        return pair
    excluded = (('completion_policy', 'cap') if four_arm else ('completion_policy',)) + tuple(additional_config_exclusions)
    configs = [{k: v for k, v in r['config'].items() if k not in excluded} for r in (a, b)]
    matched = a['engine_args'] == b['engine_args'] and configs[0] == configs[1] and a['identity_check']['workload_sha256'] == b['identity_check']['workload_sha256']
    if four_arm:
        matched = matched and all(r['config']['cap'] == (29 if r['policy'] == 'safe29' else 32) for r in (a, b))
    pair['status'] = 'DESCRIPTIVE_MATCHED_PAIR' if matched else 'UNMATCHED'
    if not matched:
        return pair
    x, y = a['metrics'], b['metrics']
    pair.update(throughput_relative_change=y['throughput_rps']/x['throughput_rps']-1,
                wall_relative_change=y['observation_duration_s']/x['observation_duration_s']-1,
                mean_completion_relative_change=b['mean_completion_s']/a['mean_completion_s']-1,
                max_itl_change_s=b['max_itl_s']-a['max_itl_s'],
                request_max_itl_p99_change_s=b['effects']['request_max_itl_s']['p99']-a['effects']['request_max_itl_s']['p99'],
                latency_change_s={f'{metric}_{q}': y['latency_s'][metric][q]-x['latency_s'][metric][q] for metric in ('ttft', 'tpot', 'itl') for q in ('p50', 'p99')},
                preemptions_change=b['preemptions']-a['preemptions'],
                recomputed_positions_change=b['work']['totals'].get('recomputed_tokens', 0)-a['work']['totals'].get('recomputed_tokens', 0),
                output_sequences_equal_count=sum(v == b['output_hashes'][rid] for rid, v in a['output_hashes'].items()),
                output_sequences_total=len(a['output_hashes']))
    indexed = {r['request_id']: r for r in a['requests']}
    pair['per_request_changes'] = [dict(request_id=r['request_id'], completion_change_s=r['completion_latency_s']-indexed[r['request_id']]['completion_latency_s'],
        max_itl_change_s=r['longest_itl']['itl_s']-indexed[r['request_id']]['longest_itl']['itl_s'],
        ttft_change_s=r['ttft_s']-indexed[r['request_id']]['ttft_s'], tpot_change_s=r['tpot_s']-indexed[r['request_id']]['tpot_s']) for r in b['requests']]
    if four_arm:
        changes = pair['per_request_changes']
        pair.update(admission_caps=dict(baseline=a['config']['cap'], action=b['config']['cap']),
            completion_worse_requests=sum(r['completion_change_s'] > 0 for r in changes),
            completion_better_requests=sum(r['completion_change_s'] < 0 for r in changes),
            request_change_distributions={k: distribution([r[k] for r in changes]) for k in ('completion_change_s', 'max_itl_change_s')},
            max_itl_distributions=dict(baseline=a['effects']['request_max_itl_s'], action=b['effects']['request_max_itl_s']))
    return pair


def readable(result):
    lines = ['# Native versus retained-KV completion headroom', '', result['status'], '',
             '| Cell | Status | Completed | Wall s | Requests/s | Mean completion s | Max ITL s | Preemptions / recomputed positions | Held requests / longest held steps | Decision s |',
             '|---|---|---:|---:|---:|---:|---:|---|---|---:|']
    for r in result['cells']:
        if not r['full_episode_comparison_eligible']:
            lines.append(f"| {r['label']} | {r['status']} | — | — | — | — | — | — | — | — |")
            continue
        m, h = r['metrics'], r['held']
        lines.append(f"| {r['label']} | COMPLETE | {m['n_completed']}/{m['n_arrived']} | {m['observation_duration_s']:.6f} | {m['throughput_rps']:.6f} | {r['mean_completion_s']:.6f} | {r['max_itl_s']:.6f} | {r['preemptions']} / {r['work']['totals'].get('recomputed_tokens', 0)} | {h['held_request_count']} / {h['longest_continuous_held_steps']} | {h['decision_seconds_total']:.6f} |")
    if 'expected_rotation_config' in result:
        lines[0] = '# Native / safe29 / headroom / absence rotation'
        lines += ['', '| Cell | Natural preemptions | Forced preemptions |', '|---|---:|---:|']
        for r in result['cells']:
            p = r.get('preemption_accounting')
            lines.append(f"| {r['label']} | {p['natural_count'] if p else 'UNRUN_OR_UNQUALIFIED'} | {p['forced_count'] if p else 'UNRUN_OR_UNQUALIFIED'} |")
    return '\n'.join(lines + ['', 'All request identities, metrics, paired differences and same-policy repeats are retained in analysis.json. '+result.get('pairing_rule', 'Pairing requires all four cells COMPLETE.'), '', result['scope'], ''])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--four-arm', action='store_true', help='native / safe29 / headroom / rotate, two repeats')
    args = parser.parse_args()
    require(not args.output_dir.exists(), 'output directory must be new; refusing overwrite')
    outputs = Path(__file__).resolve().parents[2]/'outputs/admission_capacity'
    frozen = args.run_dir/'frozen'
    if not frozen.exists():
        frozen = args.run_dir.parent/'preparation/source'
    metrics_path = frozen/'metrics.py'
    module('metrics', metrics_path)
    native = module('completion_native_helpers', outputs/'20260908_native_preemption_r01/analyze_native_preemption.py')
    labels = FOUR_LABELS if args.four_arm else LABELS
    rows = [inspect(args.run_dir/'gpu_results'/label, label, native, frozen, args.four_arm) for label in labels]
    indexed = {r['label']: r for r in rows}
    eligible = all(r['full_episode_comparison_eligible'] for r in rows)
    arms = FOUR_ARMS if args.four_arm else ('native', 'headroom')
    combinations = [(arms[a], arms[b]) for a in range(len(arms)) for b in range(a+1, len(arms))]
    pairs = [compare(indexed[f'repeat{i}-{a}'], indexed[f'repeat{i}-{b}'], eligible, args.four_arm, native.distribution) for i in (0, 1) for a, b in combinations]
    repeats = [compare(indexed[f'repeat0-{p}'], indexed[f'repeat1-{p}'], eligible, args.four_arm, native.distribution) for p in arms]
    result = dict(status='MEASUREMENT_ONLY' if eligible and all(p['status'] == 'DESCRIPTIVE_MATCHED_PAIR' for p in pairs) else 'UNRUN' if all(r['status'] == 'UNRUN' for r in rows) else 'INCOMPLETE_CAMPAIGN',
                  cells=rows, comparisons=pairs, same_policy_repeats=repeats,
                  metrics_source=str(metrics_path), metrics_sha256=hashlib.sha256(metrics_path.read_bytes()).hexdigest(),
                  scope='Single-model native in-process fixed-pool pilot. Host request time includes waiting, holding, recomputation and instrumentation; work counts and decision time are not pure GPU time. Repeated 32-request cohorts do not establish production tails, quality equivalence, generalization or a method GO. Reference TTFT/mean-TPOT SLO does not constrain maximum ITL.')
    if args.four_arm:
        result.update(pairing_rule='All eight cells must be COMPLETE; compare all six within-repeat pairs and each policy repeat. Safe29 alone changes admission cap; engine maximum and physical KV pool stay fixed.', expected_rotation_config=ROTATION_CONFIG)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir/'analysis.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    (args.output_dir/'report.md').write_text(readable(result))
    print(json.dumps(dict(status=result['status'], cells=[dict(label=r['label'], status=r['status'], error=r.get('error')) for r in rows]), indent=2))


if __name__ == '__main__':
    main()
