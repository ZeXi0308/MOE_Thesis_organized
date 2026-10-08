#!/usr/bin/env python3
"""One tail run with a burst followed by a fixed external arrival stream."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import analyze_native_mixed_budget_probe as mixed

HERE = Path(__file__).resolve().parent
PACKAGE = HERE / 'candidate_native_mixed_overlap_r01'
DIAGNOSTIC = 'FIXED_256_BURST_THEN_64_STREAM_FROM_30S'


def overlap_diagnostic(cell, workload):
    ids = [r['request_id'] for r in workload['source_requests']]
    arrivals = dict(zip(ids, workload['arrival_traces_s']['steady']))
    cohort = {rid: ('initial' if i < 256 else 'stream') for i, rid in enumerate(ids)}
    native = cell['native_victim_observation']
    joins = {j['decision_index']: j for j in native['actual_release']['joins']
             if j.get('join_status') == 'UNIQUE'}
    # The raw decision's candidate arrival is an absolute scheduler clock. It
    # must never be compared directly with external trace times.
    decisions = native['raw_decisions']
    return dict(arrivals=arrivals, cohort=cohort, decisions=decisions, joins=joins)


def temporal_coverage(cell, workload, identity):
    if cell['capacity_profile'].get('block_size_tokens') != 16:
        raise ValueError('Expected the frozen 16-token block configuration')
    state = overlap_diagnostic(cell, workload)
    arrivals, cohort = state['arrivals'], state['cohort']
    prompts = {rid: len(tokens) for rid, tokens in zip(arrivals, workload['actual_prompt_token_ids'])}
    stream_times = [arrivals[rid] for rid in arrivals if cohort[rid] == 'stream']
    stream_start, stream_end = min(stream_times), max(stream_times)
    events = []
    for ordinal, decision in enumerate(state['decisions']):
        join = state['joins'].get(ordinal)
        preempt = (join or {}).get('actual_preemption') or {}
        clock = preempt.get('method_entered_s')
        rows = decision.get('candidates', [])
        blocked = (decision.get('fallback_unknown') is not False
                   or decision.get('active_protection_or_phase') is not False
                   or not rows or any(r.get('qualified') is not True for r in rows))
        tail = rows[-1] if rows else {}
        tail_source = identity.get(decision.get('native_tail'))
        row_cohorts = [cohort.get(identity.get(r.get('request'))) for r in rows]
        phases = Counter()
        for row in rows:
            source = identity.get(row.get('request'))
            label = cohort.get(source, 'UNKNOWN')
            computed, output, prompt = row.get('computed_tokens'), row.get('output_tokens'), prompts.get(source)
            if not all(type(v) is int for v in (computed, output, prompt)):
                phase_label = 'UNKNOWN_COUNTS'
            elif output == 0:
                phase_label = 'NO_OUTPUT_YET'
            elif output > 0 and computed == prompt + output - 1:
                phase_label = 'PURE_DECODE_BY_COUNTS'
            else:
                phase_label = 'OTHER_COMPUTE_PROGRESS'
            qualified = row.get('qualified')
            phases[f'{label}/{phase_label}/qualified={qualified}'] += 1
        cross_capacity = [identity.get(r.get('request')) for r in rows if not blocked
            and type(tail.get('held_blocks')) is int and type(r.get('held_blocks')) is int
            and r['held_blocks'] == tail['held_blocks']
            and type(tail.get('computed_tokens')) is int and type(r.get('computed_tokens')) is int
            and r['computed_tokens'] // 16 == tail['computed_tokens'] // 16
            and cohort.get(identity.get(r.get('request'))) is not None
            and cohort.get(tail_source) is not None
            and cohort[identity[r['request']]] != cohort[tail_source]]
        counts = Counter(str(x) if x is not None else 'UNKNOWN' for x in row_cohorts)
        events.append(dict(decision_index=ordinal, step=decision.get('step'),
            preemption_s=clock, actual_release=(join or {}).get('actual_released_blocks'),
            executed_changed=decision.get('changed'), whole_suffix_selection_blocked=blocked,
            tail_source=tail_source, tail_cohort=cohort.get(tail_source),
            candidate_cohorts=dict(counts), candidate_progress_qualification=dict(phases),
            observed_cross_cohort_suffix=(counts['initial'] > 0 and counts['stream'] > 0
                                         and counts['UNKNOWN'] == 0),
            known_cross_cohort_suffix=(not blocked
                and counts['initial'] > 0 and counts['stream'] > 0 and counts['UNKNOWN'] == 0),
            same_held_and_computed_pages_cross_cohort_candidates=cross_capacity,
            during_external_stream=(stream_start <= clock <= stream_end) if clock is not None else None,
            future_external_arrivals=sum(t > clock for t in arrivals.values()) if clock is not None else None))
    clocks = [e['preemption_s'] for e in events if e['preemption_s'] is not None]
    first, last = (min(clocks), max(clocks)) if clocks else (None, None)
    unknown = sum(e['candidate_cohorts'].get('UNKNOWN', 0) for e in events)
    return dict(status='ANALYZED' if len(clocks) == len(events) and not unknown else 'PARTIAL_UNKNOWN',
        unknown_candidate_identities=unknown,
        first_preemption_s=first, last_preemption_s=last,
        last_external_arrival_s=max(arrivals.values()), stream_start_s=stream_start, stream_end_s=stream_end,
        arrivals_after_first_preemption=sum(t > first for t in arrivals.values()) if first is not None else None,
        preemptions_during_external_stream=sum(e['during_external_stream'] is True for e in events),
        observed_cross_cohort_decisions=sum(e['observed_cross_cohort_suffix'] for e in events),
        known_cross_cohort_decisions=sum(e['known_cross_cohort_suffix'] for e in events),
        tail_capacity_matched_cross_cohort_decisions=sum(bool(e['same_held_and_computed_pages_cross_cohort_candidates'])
                                                       for e in events),
        per_decision=events,
        semantics='Preemption method-entered clocks are measured relative to the same origin as external '
          'arrivals. A first-to-last preemption interval does not imply continuous KV pressure. '
          'Cohorts use input position, not future outcomes. Same-held/computed candidates are current-state '
          'observations, not counterfactual releases or causal benefit. Declared caps are not EOS predictions. '
          'qualified is the frozen selector pure-decode/residency guard, not general native preemption '
          'legality: native tail may still preempt an unqualified suffix request. NO_OUTPUT_YET does not '
          'distinguish partial prefill from completed prefill awaiting its first output; other progress '
          'states are not diagnosed as recomputation or transfer without additional evidence.')


def analyze(session):
    result = mixed.analyze(session, PACKAGE / 'pkg/staged_store_rotation.py',
                           PACKAGE / 'pkg/inputs/pro_high/config.json')
    if result['plan'].get('arrival_diagnostic') != DIAGNOSTIC:
        raise ValueError('Expected the fixed burst-plus-stream plan')
    source = Path(result['budget_consistency']['input_config_path']).with_name('workload.json')
    workload = json.loads(source.read_text())
    ids = [r['request_id'] for r in workload['source_requests']]
    times = [i / 100 if i < 256 else 30 + (i - 256) * .2 for i in range(320)]
    expected = dict(zip(ids, times))
    rows = result['cell']['metrics']['requests']
    errors = [dict(request=r['request_id'], actual=r.get('arrival_s'), expected=expected.get(r['request_id']))
              for r in rows if r.get('arrival_s') != expected.get(r['request_id'])]
    checks = dict(input_trace=workload['arrival_traces_s']['steady'] == times,
        observed_ids=len(rows) == len(expected) == 320 and {r['request_id'] for r in rows} == set(expected),
        observed_arrivals=not errors, workload_hash=hashlib.sha256(source.read_bytes()).hexdigest()
            == result['plan']['input_files_sha256']['pkg/inputs/pro_high/workload.json'])
    result['arrival_consistency'] = dict(status='VERIFIED' if all(checks.values()) else 'MISMATCH',
        checks=checks, errors=errors, semantics='Fixed external times; sender is not paused by policy. '
        'This is synthetic temporal coverage on seen inputs, not an independent confirmation or policy gain.')
    archive = mixed.base.archive_dir(session / 'cell-00-mixed-budget-tail')
    raw = mixed.base.optional(archive / 'raw.json', {})
    result['temporal_coverage'] = temporal_coverage(result['cell'], workload, raw.get('internal_to_source', {}))
    result['analysis_code_sha256'][Path(__file__).name] = mixed.sha(Path(__file__))
    if not all(checks.values()) or result['temporal_coverage']['status'] != 'ANALYZED':
        result['status'] = 'INCOMPLETE_OR_INVALID'
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.session.resolve())
    with args.output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status=result['status'], output=str(args.output))))
