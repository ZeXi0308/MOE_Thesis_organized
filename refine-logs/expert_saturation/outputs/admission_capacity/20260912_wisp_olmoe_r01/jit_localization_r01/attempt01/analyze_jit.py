"""Join actual compiler/cache/load intervals to frozen pager and request records."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import finite_metrics


def read(path):
    return json.loads(path.read_text())


def union_ns(intervals):
    total, end = 0, -1
    for a, b in sorted(intervals):
        total += max(0, b - max(a, end))
        end = max(end, b)
    return total


def analyze_cell(root, plan, execution):
    result = finite_metrics.cell(root, plan, execution)
    path = root / 'results' / plan['label']
    probe, raw, warmup = [read(path / n) for n in ('jit_probe.json', 'raw.json', 'warmup.json')]
    events, issues = probe['events'], result['issues']

    def check(ok, why):
        if not ok:
            issues.append(why)

    check(probe['schema'] == 'triton_host_localization_v1' and probe['status'] == 'COMPLETE', 'observer incomplete')
    check(probe['instrumentation_sha256'] == hashlib.sha256(
        (root / 'instrumentation/run_jit_probe.py').read_bytes()).hexdigest(), 'observer source mismatch')
    check([e['event_id'] for e in events] == list(range(len(events))), 'observer event sequence mismatch')
    check(all(e.get('status') == 'complete' and e.get('end_perf_ns', -1) >= e.get('start_perf_ns', 0)
              for e in events), 'unwrapped, failed or untimed observer event')
    warm = warmup['requests'][0]
    expected_doc = read(root / 'protocol.json')['warmup']['document_id']
    check(warm['document_id'] == expected_doc and warm['prompt_tokens'] == 128
          and warm['max_output_tokens'] == len(warm['output_token_ids']) == 2, 'exact warmup mismatch')
    trace = [json.loads(line) for line in (path / 'pager/calls.jsonl').read_text().splitlines()]
    warm_final = {r['layer_name']: r['retention']['final_resident_experts'] for r in trace
                  if r['context']['phase'] == 'warmup'}
    initial = read(path / 'pager_summary.json')['measurement_initial_cache']
    check(set(warm_final) == set(initial) and all(set(warm_final[k]) ==
          {e for e in initial[k]['slot_to_expert'] if e is not None} for k in warm_final),
          'warmup-final to measurement-initial resident mismatch')
    applies = {e['context']['call_id']: e for e in events if e['kind'] == 'pager_apply'}
    check(set(applies) == {r['call_id'] for r in trace} and len(applies) == len(trace), 'apply/call coverage mismatch')
    timed = [e for e in events if e['kind'] in ('compiler_call', 'launcher_and_driver_load')]
    check(all(type(e.get('cache_hit')) is bool for e in timed if e['kind'] == 'compiler_call'),
          'compiler invocation missing cache classification')
    check(all(e.get('attribution') == 'same_thread_apply' or 'apply_event_id' not in e['context']
              for e in timed), 'invalid apply attribution')
    check(all(e['context']['apply_event_id'] in {a['event_id'] for a in applies.values()}
              for e in timed if 'apply_event_id' in e['context']), 'missing attributed parent apply')
    rows = []
    for r in trace:
        a = applies.get(r['call_id'])
        if a is None:
            continue
        c = a['context']
        check((c['layer'], c['phase'], c['step_id'], c['x_shape'][0]) ==
              (r['layer_name'], r['context']['phase'], r['context'].get('step_id'), r['rows']),
              'apply identity/phase/shape mismatch: ' + str(r['call_id']))
        nested = [e for e in timed if e['context'].get('apply_event_id') == a['event_id']]
        check(all(a['start_perf_ns'] <= e['start_perf_ns'] <= e['end_perf_ns'] <= a['end_perf_ns']
                  and a['thread_id'] == e['thread_id'] for e in nested), 'nested event outside attributed apply')
        spent = union_ns([(e['start_perf_ns'], e['end_perf_ns']) for e in nested]) / 1e6
        outer = (a['end_perf_ns'] - a['start_perf_ns']) / 1e6
        rows.append(dict(call_id=r['call_id'], phase=c['phase'], step_id=c['step_id'], layer=c['layer'],
                         rows=r['rows'], host_apply_ms=r['host_apply_ms'], observer_apply_ms=outer,
                         compiler_and_load_union_ms=spent, unitemized_observer_apply_ms=outer-spent,
                         events=[e['event_id'] for e in nested]))
    by_phase = {}
    for phase in dict.fromkeys(['initialization', 'warmup', 'measurement'] +
                              [e['context'].get('phase') for e in timed]):
        subset = [e for e in timed if e['context'].get('phase') == phase]
        by_phase[phase] = dict(counts=dict(Counter(e.get('result_kind', e['kind']) for e in subset)),
            observed_union_ms=union_ns([(e['start_perf_ns'], e['end_perf_ns']) for e in subset]) / 1e6,
            events_without_apply=sum('apply_event_id' not in e['context'] for e in subset))
    origin = raw['measurement_origin_perf_counter_s'] * 1e9
    engine_rows = []
    for call in raw['engine_calls']:
        start, end = origin + call['start_s'] * 1e9, origin + call['return_s'] * 1e9
        overlap = [(max(start, e['start_perf_ns']), min(end, e['end_perf_ns'])) for e in timed
                   if e['start_perf_ns'] < end and e['end_perf_ns'] > start]
        engine_rows.append(dict(index=call['index'], start_step=call['scheduler_step_start'],
            stop_step=call['scheduler_step_stop'], engine_ms=(end-start)/1e6,
            compiler_and_load_union_ms=union_ns(overlap)/1e6))
    result.update(observer_phase_summary=by_phase, apply_localization=rows, engine_localization=engine_rows,
                  observer_events=len(events), cache_entries_before=len(execution['cache_before']),
                  cache_entries_after=len(execution['cache_after']))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root = args.input_dir.resolve()
    execution = read(root / 'results/execution.json') if (root / 'results/execution.json').exists() else {'cells': []}
    entries = {c['label']: c for c in execution['cells']}
    rows, issues, missing = [], [], []
    for plan in read(root / 'run_cells.json'):
        if entries.get(plan['label'], {}).get('status') != 'COMPLETE':
            missing.append(plan['label'])
            continue
        try:
            row = analyze_cell(root, plan, entries[plan['label']])
            rows.append(row)
            issues.extend(plan['label'] + ': ' + issue for issue in row['issues'])
        except Exception as exc:
            issues.append(plan['label'] + ': ' + repr(exc))
    if len(rows) == 2:
        a, b = (entries[r['label']] for r in rows)
        if a['cache_before'] or a['cache_after'] != b['cache_before']:
            issues.append('cold/retained private cache inventory mismatch')
    result = dict(status='INVALID_OR_INCOMPLETE' if issues else 'UNRUN_OR_PARTIAL' if missing else
                  'DESCRIPTIVE_COMPILER_LOCALIZATION', issues=issues, missing_cells=missing, cells=rows,
                  claim_ceiling='Host source localization only. Nested intervals use unions; unitemized is not pure CPU. '
                  'Cold/retained order, shapes and outputs can differ; no stable performance effect or statistical noise floor.')
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status=result['status'], cells=len(rows), issues=issues, missing=missing)))


if __name__ == '__main__':
    main()
