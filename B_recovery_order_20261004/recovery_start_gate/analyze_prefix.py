#!/usr/bin/env python3
"""Reproduce only the pre-decision output diagnostics of canonical comparisons."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

ANCHORS = (65536, 100000, 131072)
ORDINAL = 128


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def mapped_id(row):
    return {key: row.get(key) for key in
            ('request_id', 'external_request_id', 'internal_request_id')}


def read_cell(cell):
    directory = Path(cell['directory'])
    path = directory/'raw.json'
    decisions = [row.get('decision_s') for row in
                 cell.get('recovery_start_gate_actions', {}).get('rows', [])]
    decision = decisions[0] if len(decisions) == 1 and number(decisions[0]) else None
    result = dict(directory=str(directory), mode=cell.get('mode'),
                  canonical_status=cell.get('status'), raw_path=str(path),
                  decision_s=decision, recorded_decisions_s=decisions,
                  decision_status='AVAILABLE' if decision is not None else 'UNAVAILABLE',
                  expected_request_ids=[r['request'] for r in cell.get('per_request', [])])
    if not path.is_file():
        result.update(status='UNAVAILABLE', reason='RAW_MISSING')
        return result, {}
    payload = path.read_bytes()
    raw = json.loads(payload)
    result.update(status='AVAILABLE', raw_status=raw.get('status'),
                  raw_sha256=hashlib.sha256(payload).hexdigest(),
                  measurement_origin_perf_counter_s=raw.get('measurement_origin_perf_counter_s'))
    requests = {}
    for row in raw['requests']:
        rid = row['request_id']
        if rid in requests:
            raise ValueError('Duplicate raw request ID in '+str(path)+': '+rid)
        times, ids = row.get('token_times_s', []), row.get('output_token_ids', [])
        completion, arrival = row.get('completion_s'), row.get('arrival_s')
        requests[rid] = dict(mapped_id(row), arrival_s=arrival, completion_s=completion,
            status=row.get('status'), first_output_s=times[0] if times else None,
            token128_s=times[ORDINAL-1] if len(times) >= ORDINAL else None,
            prefix128=ids[:ORDINAL] if len(ids) >= ORDINAL else None,
            flow_s=completion-arrival if number(completion) and number(arrival) else None)
    first = [r['first_output_s'] for r in requests.values() if number(r['first_output_s'])]
    completed = [dict(mapped_id(r), completion_s=r['completion_s'], flow_s=r['flow_s'])
                 for r in requests.values() if decision is not None
                 and r['status'] == 'completed' and number(r['completion_s'])
                 and r['completion_s'] <= decision]
    result.update(input_mapped_ids=[mapped_id(r) for r in requests.values()],
        missing_expected_request_ids=sorted(set(result['expected_request_ids'])-requests.keys()),
        max_first_output_s=max(first) if first else None,
        first_output_present_requests=len(first),
        completed_before_decision=completed if decision is not None else None,
        completed_before_decision_count=len(completed) if decision is not None else None)
    anchors = {str(k): dict(time_s=None, cumulative_tokens=None, before_decision=None)
               for k in ANCHORS}
    count = 0
    events = raw.get('output_events')
    events_known = isinstance(events, list)
    for event in events if events_known else []:
        ids, received = event.get('new_token_ids'), event.get('received_s')
        if not isinstance(ids, list) or not number(received):
            events_known = False
            break
        count += len(ids)
        for target in ANCHORS:
            anchor = anchors[str(target)]
            if anchor['time_s'] is None and count >= target:
                anchor.update(time_s=received, cumulative_tokens=count,
                              before_decision=received < decision if decision is not None else None)
    if not events_known:
        anchors = {str(k): dict(time_s=None, cumulative_tokens=None, before_decision=None)
                   for k in ANCHORS}
    result.update(total_output_anchors=anchors,
                  output_event_status='AVAILABLE' if events_known else 'UNAVAILABLE')
    return result, requests


def compare(pair, cells, requests):
    candidate, native = pair['candidate'], pair['native']
    c, n = cells[candidate], cells[native]
    result = dict(candidate=candidate, native=native,
                  reference_rule=pair.get('reference_rule'))
    if c['status'] != 'AVAILABLE' or n['status'] != 'AVAILABLE':
        return dict(result, status='UNAVAILABLE', reason='RAW_MISSING')
    cr, nr = requests[candidate], requests[native]
    result['status'] = 'AVAILABLE'
    cdone = c['completed_before_decision']
    ndone = n['completed_before_decision']
    cids = {r['request_id'] for r in cdone or []}
    nids = {r['request_id'] for r in ndone or []}
    done_rows = []
    for rid in sorted(cids | nids):
        a, b = cr.get(rid, {}), nr.get(rid, {})
        af, bf = a.get('flow_s'), b.get('flow_s')
        done_rows.append(dict(request_id=rid, candidate_ids=mapped_id(a), native_ids=mapped_id(b),
            candidate_completed_before_decision=rid in cids if cdone is not None else None,
            native_completed_before_decision=rid in nids if ndone is not None else None,
            candidate_flow_s=af, native_flow_s=bf,
            flow_delta_candidate_minus_native_s=af-bf if number(af) and number(bf) else None))
    result['completed_before_decision'] = dict(candidate_count=len(cids) if cdone is not None else None,
        native_count=len(nids) if ndone is not None else None, per_request=done_rows)
    anchors = {}
    for k in map(str, ANCHORS):
        a, b = c['total_output_anchors'][k], n['total_output_anchors'][k]
        reasons = []
        for role, cell, anchor in (('candidate', c, a), ('native', n, b)):
            if anchor['time_s'] is None:
                reasons.append(role+': anchor missing or not reached')
            if cell['decision_s'] is None:
                reasons.append(role+': decision unavailable')
            elif anchor['time_s'] is not None and not anchor['before_decision']:
                reasons.append(role+': anchor not before decision')
        anchors[k] = dict(status='UNAVAILABLE' if reasons else 'AVAILABLE', reasons=reasons,
            candidate=a, native=b,
            delta_candidate_minus_native_s=a['time_s']-b['time_s'] if not reasons else None)
    result['common_total_output_anchors'] = anchors
    rows, deltas, different, content_count = [], [], 0, 0
    all_ids = sorted(set(c['expected_request_ids']) | set(n['expected_request_ids']) | cr.keys() | nr.keys())
    for rid in all_ids:
        a, b = cr.get(rid, {}), nr.get(rid, {})
        at, bt = a.get('token128_s'), b.get('token128_s')
        reasons = []
        for role, cell, row, value in (('candidate', c, a, at), ('native', n, b, bt)):
            if not row:
                reasons.append(role+': raw request missing')
            elif not number(value):
                reasons.append(role+': token128 missing')
            if cell['decision_s'] is None:
                reasons.append(role+': decision unavailable')
            elif number(value) and value >= cell['decision_s']:
                reasons.append(role+': token128 not before decision')
        delta = at-bt if not reasons else None
        equal = None
        if not reasons and a.get('prefix128') is not None and b.get('prefix128') is not None:
            equal = a['prefix128'] == b['prefix128']; content_count += 1; different += not equal
        if delta is not None:
            deltas.append(delta)
        rows.append(dict(request_id=rid, candidate_ids=mapped_id(a), native_ids=mapped_id(b),
            status='UNAVAILABLE' if reasons else 'AVAILABLE', reasons=reasons,
            candidate_token128_s=at, native_token128_s=bt,
            delta_candidate_minus_native_s=delta, prefix128_equal=equal))
    result['common_request_token128'] = dict(denominator=len(all_ids), comparable_time_requests=len(deltas),
        mean_delta_s=statistics.mean(deltas) if deltas else None,
        minimum_delta_s=min(deltas) if deltas else None, maximum_delta_s=max(deltas) if deltas else None,
        earlier=sum(d < 0 for d in deltas), later=sum(d > 0 for d in deltas), equal=sum(d == 0 for d in deltas),
        comparable_content_requests=content_count, different_prefix_requests=different,
        equal_prefix_requests=content_count-different, per_request=rows)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('metrics', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or args.metrics.parent/'prefix-diagnostics.json'
    if output.exists():
        raise FileExistsError('Refusing to overwrite '+str(output))
    payload = args.metrics.read_bytes(); metrics = json.loads(payload)
    cells, requests = {}, {}
    for cell in metrics['cells']:
        row, raw_rows = read_cell(cell)
        cells[cell['directory']], requests[cell['directory']] = row, raw_rows
    result = dict(input_metrics=str(args.metrics.resolve()), input_metrics_sha256=hashlib.sha256(payload).hexdigest(),
        analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        cells=list(cells.values()), comparisons=[compare(p, cells, requests) for p in metrics['comparisons']],
        semantics={
            'pairing': 'Use canonical comparisons without choosing a new control. Keep original mapped request IDs.',
            'clock': 'Raw arrival_s, completion_s and token_times_s are relative to each raw measurement origin. Canonical decision_s is the recorded selection or native shadow decision in the same origin.',
            'cutoff': 'Completed requests use completion_s <= decision_s. Output anchors and token128 comparisons require strictly before each side\'s own decision; otherwise explicitly unavailable.',
            'output_anchors': 'First raw output event reaching the fixed total via cumulative len(new_token_ids), in recorded event order; no timestamp interpolation.',
            'limits': 'Equal output counts or request token ordinals are not equal model/MoE work or matched runtime state. Token content can differ before any action. These are descriptive prefix diagnostics, not counterfactual effects; do not subtract them to correct latency.'})
    encoded = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n'
    with output.open('x') as stream:
        stream.write(encoded)
    print(json.dumps(dict(output=str(output), sha256=hashlib.sha256(encoded.encode()).hexdigest(),
                          cells=len(cells), comparisons=len(result['comparisons']))))


if __name__ == '__main__':
    main()
