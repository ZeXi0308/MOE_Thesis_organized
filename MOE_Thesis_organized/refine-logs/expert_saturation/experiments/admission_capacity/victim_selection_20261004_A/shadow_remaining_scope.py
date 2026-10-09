#!/usr/bin/env python3
"""Stdout-only membership/output reconstruction; never reconstruct physical KV state.

Frozen source: staged_store_rotation.py::admit_running records the real append;
rotation_native.py::patched_schedule_tree replaces running.append with that hook;
request_measurement.py::measure_episode records synchronous engine.step returns.
Ranking matches staged_store_rotation.py::_remaining_budget_choice: max remaining,
then latest running index. Admission timestamps are native admissions, not arrivals.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path


def short(request):
    return request.split('article-')[-1].split('-')[0]


def hindsight_endpoint(archive, raw, store):
    print('HINDSIGHT_ENDPOINT_DIAGNOSTIC')
    print('Future final lengths from this reference policy trajectory only; policy-dependent endpoints.')
    print('Not an online model, counterfactual rollout, or performance upper bound.')
    path = archive / 'metrics.json'
    if not path.is_file():
        print('UNKNOWN: metrics.json is absent')
        return
    metrics = json.loads(path.read_text()).get('per_request')
    if not isinstance(metrics, list):
        print('UNKNOWN: metrics.per_request is not recorded as a list')
        return
    grouped = defaultdict(list)
    for record in metrics:
        grouped[record.get('request_id')].append(record)
    identity = raw.get('internal_to_source', {})
    unique, valid, changed, unknown, winners_at_cap = {}, 0, [], [], 0
    for event in store['victim_decisions']:
        rows, endpoints, errors = event['candidates'], {}, []
        for row in rows:
            rid = row['request']
            found = grouped.get(identity.get(rid), [])
            cap, output = row.get('max_tokens'), row.get('output_tokens')
            final = found[0].get('n_output_tokens') if len(found) == 1 else None
            ok = (len(found) == 1 and found[0].get('status') == 'completed'
                  and type(cap) is int and cap > 0 and type(output) is int
                  and type(final) is int and final > 0 and 0 <= output <= final <= cap)
            unique[rid] = (cap, final, ok and unique.get(rid, (None, None, True))[2])
            if not ok:
                errors.append(short(rid))
            else:
                endpoints[rid] = final
        proposed = (event.get('remaining_budget') or {}).get('proposed_request')
        if errors or event.get('active_protection_or_phase') is not False:
            unknown.append(event['step'])
            print('hindsight step', event['step'], 'UNKNOWN', errors or ['PROTECTED_OR_UNKNOWN_PHASE'])
            continue
        declared = max(rows, key=lambda r: (r['max_tokens']-r['output_tokens'], r['index']))
        if proposed != declared['request']:
            unknown.append(event['step'])
            print('hindsight step', event['step'], 'UNKNOWN_RECORDED_REMAINING_PROPOSAL')
            continue
        picked = max(rows, key=lambda r: (endpoints[r['request']]-r['output_tokens'], r['index']))
        valid += 1
        winners_at_cap += endpoints[declared['request']] == declared['max_tokens']
        if picked['request'] != proposed:
            changed.append(event['step'])
            print('hindsight step', event['step'], 'different', short(proposed), '->', short(picked['request']))
    print('HINDSIGHT SUMMARY', dict(valid_decisions=valid, unknown_decision_steps=unknown,
        changed_decision_steps=changed, recorded_winners_ending_at_declared_cap=winners_at_cap,
        unique_suffix_candidates=len(unique),
        unique_at_declared_cap=sum(ok and cap == final for cap, final, ok in unique.values()),
        unique_below_declared_cap=sum(ok and final < cap for cap, final, ok in unique.values()),
        unique_unknown=sum(not ok for _, _, ok in unique.values())))


def run(session, cell):
    archive = session / cell / 'archive'
    raw = json.loads((archive / 'raw.json').read_text())
    store = json.loads((archive / 'selective-store.json').read_text())
    origin = raw['measurement_origin_perf_counter_s']
    requests = {r['internal_request_id']: r for r in raw['requests']}
    internal = {r['request_id']: r['internal_request_id'] for r in raw['requests']}
    life = defaultdict(list)
    for row in store['residency_admissions']:
        life[row['host_perf_counter_s']].append(('admit', row['request']))
    for row in raw['preemption_events']:
        if row.get('original_preemption_returned') is True:
            life[origin + row['method_returned_s']].append(('preempt', row['internal_request_id']))
    for row in raw['requests']:
        if row['status'] == 'completed':
            life[origin + row['completion_s']].append(('finish', row['internal_request_id']))
    timeline = sorted(life.items())
    outputs = raw['output_events']
    running, last_output, last_output_time = [], {}, {}
    life_index = output_index = 0
    uncertain, known, different, suffix_matches, output_mismatches = [], 0, [], 0, 0
    proposal_mismatches = []
    prefixes = []
    print('SOURCE:', archive)
    print('Only admissions/exits/outputs strictly BEFORE each decision are consumed.')
    print('UNKNOWN: prefix held/freeable/refcount/host/pending/scheduled-token state; no full-size policy or causal rollout.')
    for event in store['victim_decisions']:
        timestamp = event['host_perf_counter_s']
        while life_index < len(timeline) and timeline[life_index][0] < timestamp:
            clock, events = timeline[life_index]
            ids = [rid for _, rid in events]
            admits = [rid for kind, rid in events if kind == 'admit']
            # Distinct removals commute; a disjoint single append also commutes.
            # Multiple appends or transitions of the same request do not.
            if len(ids) != len(set(ids)) or len(admits) > 1:
                uncertain.append(('AMBIGUOUS_LIFECYCLE_TIMESTAMP', clock-origin, events))
            else:
                for kind, rid in events:
                    if kind == 'admit':
                        if rid in running:
                            uncertain.append(('DUPLICATE_ADMISSION', short(rid), clock-origin))
                        else:
                            running.append(rid)
                    elif rid not in running:
                        uncertain.append(('EXIT_WITHOUT_RUNNING_MEMBERSHIP', short(rid), clock-origin))
                    else:
                        running.remove(rid)
            life_index += 1
        while output_index < len(outputs) and origin + outputs[output_index]['received_s'] < timestamp:
            output = outputs[output_index]
            rid = internal[output['request_id']]
            clock = output['received_s']
            if (rid in last_output_time and clock <= last_output_time[rid]
                    and output['cumulative_tokens'] != last_output[rid]):
                uncertain.append(('AMBIGUOUS_OR_UNORDERED_OUTPUT', short(rid), clock))
            last_output[rid], last_output_time[rid] = output['cumulative_tokens'], clock
            output_index += 1
        if (life_index < len(timeline) and timeline[life_index][0] == timestamp
                or output_index < len(outputs) and origin + outputs[output_index]['received_s'] == timestamp):
            uncertain.append(('AMBIGUOUS_DECISION_BOUNDARY', event['step']))
        rows, start = event['candidates'], event['unprocessed_suffix_start']
        suffix_ok = running[start:] == [r['request'] for r in rows]
        suffix_matches += int(suffix_ok)
        bad_outputs = [short(r['request']) for r in rows
                       if last_output.get(r['request'], 0) != r['output_tokens']]
        output_mismatches += len(bad_outputs)
        if not suffix_ok or bad_outputs:
            uncertain.append(('OBSERVED_SUFFIX_OR_OUTPUT_MISMATCH', event['step'], bad_outputs))
        if uncertain or event.get('active_protection_or_phase') is not False:
            print('step', event['step'], 'UNKNOWN', uncertain[-1:] or ['PROTECTED_OR_UNKNOWN_PHASE'])
            continue
        budgets = {rid: (requests[rid].get('max_output_tokens'), last_output.get(rid, 0)) for rid in running}
        if any(type(cap) is not int or type(out) is not int or cap <= 0 or not 0 <= out <= cap
               for cap, out in budgets.values()):
            print('step', event['step'], 'UNKNOWN_OUTPUT_BUDGET')
            continue
        remaining = {rid: cap-out for rid, (cap, out) in budgets.items()}
        rank = {rid: i for i, rid in enumerate(running)}
        key = lambda rid: (remaining[rid], rank[rid])
        full = max(running, key=key)
        suffix = max((r['request'] for r in rows), key=key)
        proposed = (event.get('remaining_budget') or {}).get('proposed_request')
        if suffix != proposed:
            proposal_mismatches.append(event['step'])
            print('step', event['step'], 'UNKNOWN_RECORDED_REMAINING_PROPOSAL', proposed)
            continue
        prefix = max(running[:start], key=key) if start else None
        known += 1
        prefixes.append(start)
        if full != suffix:
            different.append(event['step'])
        print('step', event['step'], 'running/prefix/suffix', (len(running), start, len(rows)),
              'suffix_best', (short(suffix), remaining[suffix]),
              'prefix_best', None if prefix is None else (short(prefix), remaining[prefix]),
              'full_same_as_suffix', full == suffix)
    print('SUMMARY', dict(decisions=len(store['victim_decisions']), known_remaining_comparisons=known,
        suffix_order_matches=suffix_matches, observed_output_mismatches=output_mismatches,
        recorded_proposal_mismatches=proposal_mismatches,
        lifecycle_or_clock_unknowns=uncertain, different_decision_steps=different,
        observed_prefix_count_range=(min(prefixes), max(prefixes)) if prefixes else None))
    print('This is a lifecycle-contract/output-count shadow, not a full-running physical-state snapshot.')
    # Final endpoint information is loaded only after online reconstruction ends.
    hindsight_endpoint(archive, raw, store)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--cell', required=True)
    args = parser.parse_args()
    run(args.session, args.cell)
