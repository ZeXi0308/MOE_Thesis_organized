#!/usr/bin/env python3
"""Join existing repeat actions' first post-output preemption to victim decisions."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]


def read(path, inputs):
    data = path.read_bytes()
    inputs[str(path)] = hashlib.sha256(data).hexdigest()
    return json.loads(data)


def candidate(decision, rid, mapping, block_size):
    matches = [row for row in decision.get('candidates', []) if row.get('request') == rid]
    if len(matches) != 1:
        return None
    row = matches[0]
    computed = row.get('computed_tokens', row.get('bidkv', {}).get('computed_tokens'))
    held = row.get('held_blocks')
    gap = None
    if (row.get('qualified') is True and type(computed) is int and computed >= 0
            and type(held) is int and held >= 0 and type(block_size) is int and block_size > 0):
        gap = max(0, (computed+1+block_size-1)//block_size-held)
    return dict(internal_request=rid, source_request=mapping.get(rid),
        held_blocks=held, computed_tokens=computed,
        computed_field='computed_tokens' if 'computed_tokens' in row else 'bidkv.computed_tokens',
        residency_outputs=row.get('residency_outputs'), qualified=row.get('qualified'),
        num_preemptions=row.get('bidkv', {}).get('num_preemptions'),
        computed_plus_one_block_gap=gap)


def summarize(session):
    inputs, rows = {}, []
    metrics = read(session/'recovery-repeat-metrics.json', inputs)
    for cell in metrics['cells']:
        if cell['mode'] != 'repeat8':
            continue
        # Prefer the provided local session; canonical metadata can contain a
        # path from the machine where it was originally analyzed.
        output = session/Path(cell['directory']).parent.name/'output'
        raw = read(output/'raw.json', inputs)
        stored = read(output/'selective-store.json', inputs)
        capacity = read(output/'capacity-handoff.json', inputs)
        origin, mapping = raw['measurement_origin_perf_counter_s'], raw.get('internal_to_source', {})
        block_size = capacity.get('qualification', {}).get('block_size')
        for number, action in enumerate(cell['recovery_repeat_actions']['rows'], 1):
            rid = action['candidate_head']['internal_request']
            evidence = next((e for e in action['request_evidence'] if e['internal_request'] == rid), {})
            first_output = evidence.get('next_output_after_decision_s')
            later = [p for p in evidence.get('later_successful_preemptions', [])
                if p.get('internal_request_id') == rid and p.get('original_preemption_returned') is True
                and first_output is not None and p['method_entered_s'] >= first_output]
            preempt = min(later, key=lambda p:p['method_entered_s'], default=None)
            row = dict(cell=output.parent.name, action_number=number,
                action_event_index=action['event_index'], action_decision_s=action['decision_s'],
                target_internal_request=rid, target_source_request=mapping.get(rid),
                action_preemption_episode=action.get('candidate_num_preemptions'),
                next_output_s=first_output, first_later_preemption=preempt,
                status='UNKNOWN', missing=[], victim_decision=None)
            rows.append(row)
            if preempt is None:
                row['missing'].append('first_later_successful_preemption'); continue
            matched = [(i, d) for i, d in enumerate(stored.get('victim_decisions', []))
                if d.get('selected') == rid and d.get('step') == preempt.get('engine_call_index')]
            row['same_request_and_step_match_count'] = len(matched)
            if len(matched) != 1:
                row['missing'].append('unique_same_request_and_engine_step_decision'); continue
            index, decision = matched[0]
            stamp = decision.get('host_perf_counter_s')
            if type(stamp) not in (int, float):
                row['missing'].append('victim_decision_host_timestamp'); continue
            delta = origin+preempt['method_entered_s']-stamp
            if delta < 0:
                row['missing'].append('victim_decision_precedes_preempt_entry')
            failed = decision.get('failed_request')
            target = candidate(decision, rid, mapping, block_size)
            cause = candidate(decision, failed, mapping, block_size) if failed is not None else None
            if target is None: row['missing'].append('target_candidate_row')
            if cause is None: row['missing'].append('failed_request_candidate_row')
            for name, item in [('target', target), ('failed', cause)]:
                if item is not None:
                    row['missing'].extend(name+'.'+key for key in
                        ('held_blocks', 'computed_tokens', 'residency_outputs') if item[key] is None)
            row.update(status='MATCHED' if not row['missing'] else 'UNKNOWN',
                victim_decision_index=index, victim_decision_to_preempt_us=delta*1e6,
                failed_request_relation='UNKNOWN' if failed is None else
                    'TARGET_SELF' if failed == rid else 'OTHER_RUNNING_REQUEST',
                target=target, failed_request=cause, block_size=block_size,
                victim_decision={key:decision.get(key) for key in (
                    'step', 'host_perf_counter_s', 'rule', 'failed_request', 'native_tail',
                    'selected', 'changed', 'free_blocks', 'fallback_unknown',
                    'current_guard_eligible', 'current_guard_applied')},
                native_tail_equals_selected=(decision.get('native_tail') == decision.get('selected'))
                    if decision.get('native_tail') is not None else None)
    matched = [r for r in rows if r['status'] == 'MATCHED']
    summary = dict(actions=len(rows), matched=len(matched), unknown=len(rows)-len(matched),
        failed_request_relations=dict(Counter(r.get('failed_request_relation', 'UNKNOWN') for r in rows)),
        free_blocks_zero=sum(r['victim_decision']['free_blocks'] == 0 for r in matched),
        native_tail_equals_selected=sum(r['native_tail_equals_selected'] is True for r in matched),
        changed_true=sum(r['victim_decision']['changed'] is True for r in matched),
        target_computed_plus_one_block_gap=dict(Counter(str(r['target']['computed_plus_one_block_gap']) for r in matched)),
        failed_computed_plus_one_block_gap=dict(Counter(str(r['failed_request']['computed_plus_one_block_gap']) for r in matched)),
        match_delta_us_min=min((r['victim_decision_to_preempt_us'] for r in matched), default=None),
        match_delta_us_max=max((r['victim_decision_to_preempt_us'] for r in matched), default=None))
    source_paths = ['pkg/staged_store_rotation.py', 'pkg/rotation_native.py', 'pkg/bidkv_score.py']
    return dict(summary=summary, rows=rows, inputs_sha256=inputs,
        source_sha256={p:hashlib.sha256((BASE/p).read_bytes()).hexdigest() for p in source_paths},
        analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        join_rule='Unique exact selected internal request ID and victim decision step equal to '
            'successful preemption engine_call_index. Host timestamp delta is recorded and must be nonnegative; '
            'there is no nearest-time fallback or across-run join.',
        logger_semantics='rotation_native.py FCFS allocation-failure branch calls pick_victim before pop/preempt. '
            'staged_store_rotation.py failed_request is running[req_index]; native_tail is running[-1]; '
            'changed compares the selected index with that native tail. Candidate held_blocks is current owned '
            'block count; bidkv.computed_tokens reads num_computed_tokens. residency_outputs is output progress '
            'since the most recent native admit_running epoch, not total request output count.',
        arithmetic_semantics='computed_plus_one_block_gap=max(0,ceil((computed+1)/block_size)-held), '
            'reported only for qualified candidates. It checks the observed pure-decode next-token boundary, '
            'not an independently recorded allocate_slots argument or a counterfactual victim cost.',
        scope='Existing within-run records only; no victim policy proposal or change, no GPU run.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or args.session/'repreempt-causes.json'
    if output.exists():
        raise FileExistsError(output)
    result = summarize(args.session.resolve())
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(output=str(output), summary=result['summary'])))


if __name__ == '__main__':
    main()
