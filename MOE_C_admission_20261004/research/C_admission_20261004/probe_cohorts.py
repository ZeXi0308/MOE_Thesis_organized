#!/usr/bin/env python3
"""Descriptive posthoc cohorts for the two recorded admission-probe pairs.

Usage: python3 probe_cohorts.py OPPORTUNITIES.json --output NEW.json
Uses recorded first-opportunity states and analyze.py per-request data only.
"""
import argparse
import collections
import json
from pathlib import Path

from analyze import finite


LATENCIES = ('ttft_s', 'flow_s', 'max_generation_gap_s')
EPSILON_S = 1e-9


def paired_summary(ids, a, b):
    ids = sorted(ids)
    result = dict(requests=len(ids), request_ids=sorted(ids),
        candidate_outcomes=dict(collections.Counter(a[r]['outcome'] for r in ids)),
        baseline_outcomes=dict(collections.Counter(b[r]['outcome'] for r in ids)), metrics={})
    for key in LATENCIES:
        pairs = [(a[r][key], b[r][key]) for r in ids if finite(a[r].get(key)) and finite(b[r].get(key))]
        deltas = [x-y for x, y in pairs]
        n = len(pairs)
        result['metrics'][key] = dict(observed_pairs=n, missing_pairs=len(ids)-n,
            candidate_mean=sum(x for x, _ in pairs)/n if n else None,
            baseline_mean=sum(y for _, y in pairs)/n if n else None,
            mean_delta=sum(deltas)/n if n else None,
            benefited=sum(d < -EPSILON_S for d in deltas),
            harmed=sum(d > EPSILON_S for d in deltas), tied=sum(abs(d) <= EPSILON_S for d in deltas))
    changes = [a[r]['output_tokens']-b[r]['output_tokens'] for r in ids]
    result['output'] = dict(candidate_tokens=sum(a[r]['output_tokens'] for r in ids),
        baseline_tokens=sum(b[r]['output_tokens'] for r in ids),
        mean_token_delta=sum(changes)/len(ids) if ids else None,
        fewer_tokens=sum(d < 0 for d in changes), more_tokens=sum(d > 0 for d in changes),
        equal_tokens=sum(d == 0 for d in changes),
        sequence_mismatch_requests=sum(a[r]['output_token_ids_sha256'] != b[r]['output_token_ids_sha256'] for r in ids),
        stop_reason_mismatch_requests=sum(a[r]['stop_reason'] != b[r]['stop_reason'] for r in ids))
    return result


def completion_window(event_time, rows):
    before = sorted(rid for rid, r in rows.items() if finite(r.get('completion_s'))
                    and event_time-.25 <= r['completion_s'] < event_time)
    after = sorted(rid for rid, r in rows.items() if finite(r.get('completion_s'))
                   and event_time <= r['completion_s'] <= event_time+.25)
    return dict(event_external_s=event_time, before_interval_s=[event_time-.25, event_time],
        after_interval_s=[event_time, event_time+.25], before_count=len(before), after_count=len(after),
        before_request_ids=before, after_request_ids=after,
        semantics='Host-recorded completion events: before is [event-.25,event), after is [event,event+.25]. '
                  'These are completed-request counts, not measured KV release times or freed block amounts.')


def analyze_pair(pair, cells):
    ca, cb = cells[pair['candidate']], cells[pair['baseline']]
    namespace = pair.get('comparison_event_namespace', 'pending_recovery_probe')
    if namespace not in ('pending_recovery_probe', 'reservation_probe'):
        raise ValueError('Unknown comparison event namespace: ' + str(namespace))
    scoped_a, scoped_b = (ca['reservation_probe'], cb['reservation_probe']) if namespace == 'reservation_probe' else (ca, cb)
    ea, eb = scoped_a['first_opportunity'], scoped_b['first_opportunity']
    if ea is None or eb is None:
        return dict(candidate=pair['candidate'], baseline=pair['baseline'],
            comparison_event_namespace=namespace, status='MISSING_FIRST_OPPORTUNITY')
    a, b = ({r['request_id']: r for r in json.loads(Path(c['service']['per_request_json']).read_text())}
            for c in (ca, cb))
    if a.keys() != b.keys() or ca['service']['workload_identity_sha256'] != cb['service']['workload_identity_sha256']:
        raise ValueError('Pair does not share the same planned requests and external trace.')
    sa, sb = ({r['request_id']: r for r in e['before_state']['requests']} for e in (ea, eb))
    ids = set(a)
    if not sa.keys() <= ids or not sb.keys() <= ids:
        raise ValueError('First-event state contains unmapped requests.')
    admitted = {rid for rid in sa.keys() & sb.keys() if sa[rid]['admitted'] and sb[rid]['admitted']}
    never = {rid for rid in sa.keys() & sb.keys() if not sa[rid]['admitted'] and not sb[rid]['admitted']}
    # The report's admitted set retains unfinished preempted requests. Confirm
    # the opposite category has no recorded evidence of an earlier admission.
    for rid in never:
        for st in (sa[rid], sb[rid]):
            if st['computed'] or st['output'] or st['preemptions']:
                raise ValueError('Unadmitted state has previous computation: ' + rid)
    remaining = ids-admitted-never
    completed_a = {rid for rid in ids if finite(a[rid].get('completion_s'))
                   and a[rid]['completion_s'] <= ea['external_time_s']}
    completed_b = {rid for rid in ids if finite(b[rid].get('completion_s'))
                   and b[rid]['completion_s'] <= eb['external_time_s']}
    if admitted & never or len(admitted)+len(never)+len(remaining) != len(ids):
        raise AssertionError('Cohorts are not an exhaustive partition.')
    intervention = scoped_a if namespace == 'reservation_probe' else ca['intervention']
    direct_key, fifo_key = ('direct_action', 'fifo_action') if namespace == 'reservation_probe' else ('direct_probe', 'fifo_probe')
    direct = set(intervention[direct_key]['request_ids'])
    fifo = set(intervention[fifo_key]['request_ids'])
    if not direct | fifo <= ids:
        raise ValueError('Controller-held identity missing from planned population.')
    cohorts = dict(both_admitted_unfinished=admitted, both_never_admitted=never,
                   differing_admission_or_absent_from_unfinished_state=remaining)
    overlays = dict(candidate_direct_probe=direct, candidate_fifo_probe=fifo, candidate_controller_held=direct | fifo)
    rows = []
    for rid in sorted(ids):
        rows.append(dict(request_id=rid,
            cohort=next(key for key, members in cohorts.items() if rid in members),
            candidate_event_state=sa.get(rid), baseline_event_state=sb.get(rid),
            candidate_direct_probe=rid in direct, candidate_fifo_probe=rid in fifo,
            candidate_outcome=a[rid]['outcome'], baseline_outcome=b[rid]['outcome'],
            paired_values={key: dict(candidate=a[rid].get(key), baseline=b[rid].get(key),
                delta=a[rid][key]-b[rid][key] if finite(a[rid].get(key)) and finite(b[rid].get(key)) else None)
                for key in (*LATENCIES, 'output_tokens')},
            output_sequence_mismatch=a[rid]['output_token_ids_sha256'] != b[rid]['output_token_ids_sha256']))
    return dict(candidate=pair['candidate'], baseline=pair['baseline'], status='DESCRIPTIVE_POSTHOC',
        comparison_event_namespace=namespace,
        candidate_event_request=ea['request_id'], baseline_event_request=eb['request_id'],
        candidate_event_external_s=ea['external_time_s'], baseline_event_external_s=eb['external_time_s'],
        same_first_request=ea['request_id'] == eb['request_id'],
        overall=paired_summary(ids, a, b),
        partition={key: paired_summary(members, a, b) for key, members in cohorts.items()},
        controller_held_overlays={key: dict(paired_summary(members, a, b),
            partition_membership={group: len(members & group_ids) for group, group_ids in cohorts.items()})
            for key, members in overlays.items()},
        completion_windows=dict(candidate=completion_window(ea['external_time_s'], a),
                                baseline=completion_window(eb['external_time_s'], b)),
        residual_host_completion_by_own_event=dict(both=len(remaining & completed_a & completed_b),
            candidate_only=len((remaining & completed_a)-completed_b),
            baseline_only=len((remaining & completed_b)-completed_a),
            neither=len(remaining-completed_a-completed_b)),
        remaining_state_patterns=dict(collections.Counter(
            f"candidate={'absent' if rid not in sa else 'admitted' if sa[rid]['admitted'] else 'unadmitted'},"
            f"baseline={'absent' if rid not in sb else 'admitted' if sb[rid]['admitted'] else 'unadmitted'}"
            for rid in remaining)), paired_requests=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('opportunities', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Refusing to overwrite an existing output.')
    source = json.loads(args.opportunities.read_text())
    if len(source['cells']) != 4 or len(source['abba_pairs']) != 2:
        parser.error('Expected the four-cell ABBA opportunity summary.')
    cells = {Path(c['cell']).name: c for c in source['cells']}
    pairs = [analyze_pair(pair, cells) for pair in source['abba_pairs']]
    result = dict(schema_version=1, source_opportunities=str(args.opportunities.resolve()),
        source_analysis=source['source_analysis'], independent_unit='run',
        semantics='Posthoc groups use each run\'s own first-opportunity pre-action host state. '
            'Both-admitted means admitted and unfinished in both runs, not identical native status. '
            'Both-never-admitted requires explicit unfinished unadmitted state in both; absent requests '
            'remain in the residual group and are not silently called failures or completions. '
            'Held cohorts overlap the exhaustive partition; FIFO held does not prove native-fit eligibility. '
            'Full-lifetime latency changes, including pre-event TTFT and decode gaps, are descriptive, '
            'not effects caused by the subsequent intervention. Request-level signs are not independent '
            'replicates; separate-run states and outputs differ. These groups do not replace all-arrival results.',
        sign_semantics=f'Latency delta=candidate-baseline; lower is benefited, higher harmed, tie tolerance '
            f'{EPSILON_S} s is numerical only, not an application SLO. Output changes have no benefit sign.',
        pairs=pairs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    for pair in pairs:
        print(pair['candidate'], '/', pair['baseline'], pair['status'])
        if pair['status'] == 'DESCRIPTIVE_POSTHOC':
            for name, cohort in pair['partition'].items():
                flow = cohort['metrics']['flow_s']
                print(name, 'N=', cohort['requests'], 'flow mean delta=', flow['mean_delta'],
                      'benefited/harmed/tied=', flow['benefited'], flow['harmed'], flow['tied'])
    print(args.output.resolve())


if __name__ == '__main__':
    main()
