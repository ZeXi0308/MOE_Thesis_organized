#!/usr/bin/env python3
"""Supplement four-arm or cohort results with actual recovery and host accounting."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from analyze_completion_headroom import FOUR_LABELS, held_accounting, preemption_accounting
from analyze_recovery_admission import module, read, require


def fingerprint(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def cell(run, label, native, qualified, policy=None, campaign=False):
    directory = run/'gpu_results'/label
    paths = [next(p for p in (directory/'raw.json', directory/'raw.json.gz') if p.exists()), directory/'headroom-decisions.json']
    raw, decisions = map(read, paths)
    require(raw['status'] == qualified['status'] == 'COMPLETE', 'requires complete qualified cell')
    require(raw['host_chunk_diagnostics']['token_level_itl_resolved'], 'token gaps unresolved')
    work, calls = native.execution_accounting(raw)
    events = native.preemptions(raw, calls)
    policy = policy or label.split('-')[1]
    held = held_accounting(raw, decisions, policy)
    partition = preemption_accounting(raw, decisions, events, policy, native.distribution)
    kinds = {(e['step'], e['request_id']): e['kind'] for e in partition['events']}
    ids, steps = raw['internal_to_source'], raw['scheduler_steps']
    requests = {r['request_id']: r for r in raw['requests']}
    by_return = {c['returned_s']: c for c in raw['engine_steps']}
    require(all(c['scheduler_step_end']-c['scheduler_step_start'] == 1 for c in raw['engine_steps']), 'requires one schedule per call')
    recoveries, protections = [], []
    for event in events:
        rid, k = event['request_id'], event['attempted_step']
        internal, n = event['victim_internal_request_id'], event['victim_state']['output_tokens']
        times = requests[rid]['token_times_s']
        first = next(i for i in range(k, len(steps)) if internal in decisions[i]['actual_scheduled'])
        end = by_return[times[n]]['scheduler_step_start']
        require(first <= end and calls[k]['start_s'] <= times[n], 'recovery/new token misaligned')
        recoveries.append(dict(request_id=rid, kind=kinds[(k, rid)], preempt_step=k, outputs_before=n,
            last_token_step=by_return[times[n-1]]['scheduler_step_start'], last_token_s=times[n-1],
            first_schedule_step=first, first_schedule_s=calls[first]['start_s'], first_new_token_step=end,
            first_new_token_s=times[n], gap_s=times[n]-times[n-1],
            boundary_s=calls[k]['start_s']-times[n-1], wait_before_recovery_s=calls[first]['start_s']-calls[k]['start_s'],
            recovery_span_s=times[n]-calls[first]['start_s'],
            recomputed_positions=sum(x['recompute_tokens'] for s in steps[first:end+1] for x in s['scheduled'] if x['request_id'] == rid)))
    for d in decisions:
        if not d.get('forced_preempted'):
            continue
        k, internal, n = d['step'], d['recovery_target'], d['recovery_output_at_start']
        rid = ids[internal]
        returned = requests[rid]['token_times_s'][n]
        end = by_return[returned]['scheduler_step_start']
        segment = decisions[k:end+1]
        require(all(x['recovery_target'] == internal and x['actual_scheduled'].get(internal, 0) > 0
                    and not x['natural_preempted'] and x['free_after'] >= x['recovery_remaining_blocks_after'] for x in segment), 'protection failed')
        require(decisions[end+1].get('recovery_completed') == internal, 'protection released before first new token')
        protections.append(dict(step=k, target=rid, victim=ids[d['forced_preempted'][0]], outputs_before=n,
            first_new_token_step=end, release_step=end+1, recovery_calls=end-k+1, recovery_span_s=returned-calls[k]['start_s'],
            released_blocks=d['candidate_released_blocks'], required_blocks=d['candidate_required_blocks'], free_before=d['free_before'],
            held_request_steps=sum(len(x['held']) for x in segment)))
    longest = max(recoveries, key=lambda e: e['gap_s'], default=None)
    if policy == 'rotate':
        require(longest and longest['gap_s'] <= qualified['max_itl_s']+1e-12, 'recovery exceeds qualified maximum ITL')
        if not campaign:
            require(abs(longest['gap_s']-qualified['max_itl_s']) < 1e-12, 'longest gap outside preemption recovery')
        longest = dict(longest, decision_gates=[])
        for d in decisions[longest['preempt_step']:longest['first_schedule_step']+1]:
            proposal = d.get('proposal') or {}
            reason = proposal.get('reason', 'protected recovery' if d.get('recovery_target') else d.get('not_applied_reason', 'none'))
            if reason.startswith('absence ') and 'below threshold' in reason:
                reason = 'absence below threshold'
            target = proposal.get('resume_id') or d.get('recovery_target')
            key = (reason, ids.get(target))
            groups = longest['decision_gates']
            if groups and (groups[-1]['reason'], groups[-1]['target']) == key:
                groups[-1]['last_step'] = d['step']
            else:
                groups.append(dict(first_step=d['step'], last_step=d['step'], reason=reason, target=ids.get(target)))
    buckets, widths = defaultdict(Counter), defaultdict(Counter)
    for s in steps:
        c = calls[s['step']]
        amounts = {k: sum(x[k] for x in s['scheduled']) for k in ('prefill_tokens', 'decode_tokens', 'recompute_tokens')}
        kind = 'recompute_bearing' if amounts['recompute_tokens'] else 'prefill_bearing' if amounts['prefill_tokens'] else 'pure_decode'
        buckets[kind].update(dict(calls=1, engine_s=c['returned_s']-c['start_s'], scheduler_s=s['end_s']-s['start_s'], **amounts))
        widths[len(s['scheduled'])].update(dict(calls=1, engine_s=c['returned_s']-c['start_s']))
    engine = sum(c['returned_s']-c['start_s'] for c in raw['engine_steps'])
    scheduler = sum(s['end_s']-s['start_s'] for s in steps)
    return dict(label=label, inputs=[fingerprint(p) for p in paths], check_status='PASS_NATIVE_EVENTS_AND_NEW_OUTPUT_ALIGNMENT',
        forced=partition['forced_count'], natural=partition['natural_count'], recoveries=recoveries, protected_recoveries=protections,
        longest_recovery=longest, held_request_steps=held['held_request_steps'], decision_s=held['decision_seconds_total'],
        total_engine_calls=len(raw['engine_steps']), scheduler_s=scheduler, engine_excluding_scheduler_s=engine-scheduler,
        outside_engine_s=qualified['metrics']['observation_duration_s']-engine, work_totals=work['totals'],
        work_buckets=dict(buckets), width_buckets=dict(widths))


def campaign_layout(primary, run, qualified):
    """Use frozen identities, never infer a cohort policy from a label position."""
    from analyze_rotation_holdout import validate_manifest
    manifest_path = run/'frozen/campaign.json'
    manifest = read(manifest_path)
    require(manifest == primary['campaign_manifest'] and
            fingerprint(manifest_path)['sha256'] == primary['campaign_manifest_sha256'], 'analysis/campaign mismatch')
    validate_manifest(manifest, run/'frozen')
    require(primary['all_twenty_cells_qualified'] and primary['comparisons_eligible'], 'requires all twenty qualified cells')
    specs = manifest['cells']
    index = {s['label']: s for s in specs}
    require(set(qualified) == set(index), 'qualified labels differ from campaign')
    for label, spec in index.items():
        row = qualified[label]
        require(row['status'] == 'COMPLETE' and row['full_episode_comparison_eligible'] and
                row['policy'] == spec['completion_policy'] and
                all(row[k] == spec[k] for k in ('cohort_id', 'block', 'role')), 'qualified cell identity/status mismatch')
    pairs = [p for p in primary['comparisons'] if index[p['action']]['role'] == 'rotate']
    expected = {(c['id'], b, r) for c in manifest['cohorts'] for b in (0, 1) for r in ('native', 'safe29', 'headroom')}
    observed = []
    for pair in pairs:
        a, b = index[pair['baseline']], index[pair['action']]
        require(pair['status'] == 'DESCRIPTIVE_MATCHED_PAIR' and
                (a['cohort_id'], a['block']) == (b['cohort_id'], b['block']), 'unqualified or cross-cohort/block rotation pair')
        observed.append((a['cohort_id'], a['block'], a['role']))
    require(len(observed) == len(expected) and set(observed) == expected, 'missing or duplicated rotation pairs')
    return specs, pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'refusing to overwrite existing accounting')
    primary = read(args.analysis)
    require(primary['status'] == 'MEASUREMENT_ONLY', 'requires completed qualified analysis')
    qualified = {r['label']: r for r in primary['cells']}
    require(len(qualified) == len(primary['cells']), 'duplicate qualified labels')
    campaign = 'campaign_manifest' in primary
    if campaign:
        specs, pairs = campaign_layout(primary, args.run_dir, qualified)
    else:
        specs = [dict(label=label, completion_policy=label.split('-')[1]) for label in FOUR_LABELS]
        pairs = [p for p in primary['comparisons'] if p['action'].endswith('-rotate')]
    module('metrics', args.run_dir/'frozen/metrics.py')
    helper = Path(__file__).resolve().parents[2]/'outputs/admission_capacity/20260908_native_preemption_r01/analyze_native_preemption.py'
    native = module('rotation_native_accounting_helpers', helper)
    rows = [cell(args.run_dir, s['label'], native, qualified[s['label']], s['completion_policy'], campaign) for s in specs]
    if campaign:
        for row, spec in zip(rows, specs):
            row.update({k: spec[k] for k in ('cohort_id', 'block', 'role')})
            longest = row['longest_recovery']
            row['longest_recovery_is_max_itl'] = bool(longest and abs(longest['gap_s']-qualified[row['label']]['max_itl_s']) < 1e-12)
    comparisons = [dict(baseline=p['baseline'], action=p['action'], throughput_relative_change=p['throughput_relative_change'],
        mean_completion_relative_change=p['mean_completion_relative_change'], output_sequences_equal_count=p['output_sequences_equal_count'],
        max_itl_worse_requests=sum(r['max_itl_change_s'] > 0 for r in p['per_request_changes']),
        completion_worse_requests=p['completion_worse_requests'], per_request_changes=p['per_request_changes']) for p in pairs]
    if campaign:
        for pair in comparisons:
            baseline = qualified[pair['baseline']]
            pair.update(cohort_id=baseline['cohort_id'], block=baseline['block'], baseline_role=baseline['role'], action_role='rotate')
    path_fields = ('preempt_step', 'request_id', 'kind', 'first_schedule_step', 'first_new_token_step', 'outputs_before', 'recomputed_positions')
    rotations = [r for r in rows if r['label'].endswith('-rotate')]
    paths = {r['label']: [tuple(e[k] for k in path_fields) for e in r['recoveries']] for r in rotations}
    if campaign:
        equal = {c['id']: paths[f"{c['id']}-block0-rotate"] == paths[f"{c['id']}-block1-rotate"] for c in primary['campaign_manifest']['cohorts']}
        repeats = dict(repeat_rotation_recovery_paths_equal_by_cohort=equal,
                       campaign_manifest_sha256=primary['campaign_manifest_sha256'])
    else:
        same = list(paths.values())
        equal = same[0] == same[1]
        repeats = dict(repeat_rotation_recovery_paths_equal=equal)
    result = dict(status='MEASUREMENT_ONLY', analysis_input=fingerprint(args.analysis), script=fingerprint(Path(__file__).resolve()),
        cells=rows, comparisons=comparisons, **repeats,
        scope='Host capture; all differences pair within block. Recovery spans contain useful concurrent decode, not pure recomputation or GPU time. '
              'Held counts/progress reconcile with telemetry; exact block IDs are checked by the runtime adapter but a full worker block table is not independently captured. '
              'Two repeats do not establish statistical significance or noninferiority. Zero held actions do not establish that reservation protection is necessary.')
    if campaign:
        result['scope'] += ' Policy pairs stay within cohort/block; recovery-path repeat equality is evaluated separately within each cohort. A non-preemption token gap may be the maximum ITL on new inputs.'
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(status=result['status'], output=str(args.output), rotation_paths_equal=equal)))


if __name__ == '__main__':
    main()
