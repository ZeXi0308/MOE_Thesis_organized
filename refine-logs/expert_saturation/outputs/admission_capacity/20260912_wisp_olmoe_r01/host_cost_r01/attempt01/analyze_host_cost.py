"""Bounded host observations on own X trajectories; all observer costs remain."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import analyze_covered
import finite_metrics


def read(path):
    return json.loads(path.read_text())


def union_ns(intervals):
    total, right = 0, None
    for start, end in sorted(intervals):
        if end <= start:
            continue
        total += end - max(start, right if right is not None else start) if right is None or end > right else 0
        right = end if right is None else max(right, end)
    return total


def overlaps(span, events, same_thread):
    intervals = []
    for event in events:
        a, b = event.get('start_perf_ns'), event.get('stop_perf_ns')
        if a is None or b is None:
            continue
        same = event.get('native_thread_id') == span.get('native_thread_id')
        if same == same_thread:
            intervals.append((max(a, span['start_perf_ns']), min(b, span['end_perf_ns'])))
    return union_ns(intervals) / 1e6


def span_metrics(span, gc_events):
    return dict(wall_ms=(span['end_perf_ns']-span['start_perf_ns'])/1e6,
                thread_cpu_ms=(span['end_thread_cpu_ns']-span['start_thread_cpu_ns'])/1e6,
                process_cpu_ms=(span['end_process_cpu_ns']-span['start_process_cpu_ns'])/1e6,
                same_thread_gc_overlap_ms=overlaps(span, gc_events, True),
                other_thread_gc_overlap_ms=overlaps(span, gc_events, False))


def analyze_cell(root, plan, execution):
    row = analyze_covered.analyze_cell(root, plan, execution)
    path = root/'results'/plan['label']
    host, raw = read(path/'host_cost_observation.json'), read(path/'raw.json')
    spans, events, issues = host['spans'], host['gc_events'], row['issues']
    enabled = plan['observer'] == 'on'

    def check(ok, why):
        if not ok:
            issues.append(why)

    check(host['schema']=='host_cost_observation_v1' and host['mode']==plan['observer']
          and host['status']=='COMPLETE' and host['hooks_restored'] is True, 'host observer status/restore mismatch')
    check(execution.get('host_observation_valid') is True, 'driver host-observation gate failed')
    sources = host['source_sha256']
    check({'instrumentation/run_host_cost.py','source/native_capture.py','source/runtime_variation_observer.py'}
          <= sources.keys(), 'missing observer source identities')
    check(all(hashlib.sha256((root/n).read_bytes()).hexdigest()==h for n,h in sources.items()),
          'executed host observation source mismatch')
    check(raw['cpu_diagnostics'] is enabled and raw['runtime_observer_enabled'] is enabled,
          'ordinary capture CPU observation switch mismatch')
    check([s['event_id'] for s in spans]==list(range(len(spans))), 'host span IDs not contiguous')
    check(all(s['status']=='complete' and s['end_perf_ns']>=s['start_perf_ns']
              and s['end_thread_cpu_ns']>=s['start_thread_cpu_ns']
              and s['end_process_cpu_ns']>=s['start_process_cpu_ns'] for s in spans), 'failed or invalid host span')
    row.update(observer=plan['observer'], host_observation_environment={k:host.get(k) for k in
               ('environment_before','environment_after')}, host_span_count=len(spans), gc_event_count=len(events))
    if not enabled:
        check(not spans and not events, 'off arm unexpectedly instrumented')
        check(all('cpu_delta' not in c for c in raw['engine_calls']), 'off arm has CPU deltas')
        return row
    calls = [json.loads(line) for line in (path/'pager/calls.jsonl').read_text().splitlines()]
    measured = {c['call_id']:c for c in calls if c['measurement']}
    engines = raw['engine_calls']
    by_kind = {kind:[s for s in spans if s['kind']==kind] for kind in ('engine','planner','kernel')}
    check(sum(map(len,by_kind.values()))==len(spans), 'unexpected host span kind')
    check([s['engine_call'] for s in by_kind['engine']]==list(range(len(engines))), 'engine span ordinal mismatch')
    for kind in ('planner','kernel'):
        check(Counter(s['call_id'] for s in by_kind[kind])==Counter({k:1 for k in measured}), kind+' coverage mismatch')
        for span in by_kind[kind]:
            rec = measured[span['call_id']]
            check(span['layer']==rec['layer_name'] and span['rows']==rec['rows']
                  and span['context']==rec['context'], kind+' trace identity mismatch')
            engine = engines[span['engine_call']]
            check(engine['scheduler_step_start'] <= rec['context']['step_id'] < engine['scheduler_step_stop'],
                  kind+' engine/step identity mismatch')
    engine_spans = {s['engine_call']:s for s in by_kind['engine']}
    for span in by_kind['planner']+by_kind['kernel']:
        parent = engine_spans[span['engine_call']]
        check(parent['start_perf_ns'] <= span['start_perf_ns'] <= span['end_perf_ns'] <= parent['end_perf_ns'],
              'layer host span outside parent engine span')
    summaries = {}
    for kind, selected in by_kind.items():
        values = [span_metrics(s,events) for s in selected]
        summaries[kind] = dict(count=len(selected), totals={k:sum(v[k] for v in values) for k in values[0]},
                               interval_union_ms=union_ns([(s['start_perf_ns'],s['end_perf_ns']) for s in selected])/1e6)
    focused = [dict(**s, measured=span_metrics(s,events)) for s in spans if s['engine_call']<3]
    wide_cpu = {key:sum(c['cpu_delta'][key] for c in engines)
                if all(c['cpu_delta'].get(key) is not None for c in engines) else None
                for key in engines[0]['cpu_delta']}
    sched = host.get('environment_before',{}).get('sched_schedstats',{})
    sched_available = sched.get('status') == 'observed' and str(sched.get('value')).strip() == '1'
    row.update(schedstat_interpretation='enabled sample; process leader only' if sched_available else
               'unavailable or disabled; zero schedstat values are not evidence of zero CPU queueing',
               host_span_summaries=summaries, first_three_engine_spans=focused,
               wide_cpu_deltas_including_observation=wide_cpu,
               raw_observation_envelope_ms=sum(c['observation_envelope_s'] for c in engines)*1000,
               incomplete_gc_events=sum(e.get('start_perf_ns') is None or e.get('stop_perf_ns') is None for e in events),
               host_timing_scope='Kind summaries overlap. Planner excludes metadata argument preparation; native-kernel host span '
               'excludes caller argument construction and is not GPU time. Wide native CPU deltas include observation envelopes. '
               'Thread/process/rusage are overlapping, not additive; wall-thread difference is not a diagnosed wait cause. '
               'GC overlaps are unions clipped to spans, not subtractable savings. Preserve unavailable counters as unknown.')
    return row


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();root=args.input_dir.resolve()
    execution=read(root/'results/execution.json') if (root/'results/execution.json').exists() else {'cells':[]}
    entries={c['label']:c for c in execution['cells']}
    rows,missing,issues=[],[],[]
    for plan in read(root/'run_cells.json'):
        if entries.get(plan['label'],{}).get('status')!='COMPLETE':
            missing.append(plan['label']);continue
        try:
            row=analyze_cell(root,plan,entries[plan['label']]);rows.append(row)
            issues.extend(plan['label']+': '+i for i in row['issues'])
        except Exception as exc:
            issues.append(plan['label']+': '+repr(exc))
    report=dict(status='INVALID_OBSERVATION_OR_ANALYSIS' if issues else 'UNRUN_OR_PARTIAL' if missing else
                'DESCRIPTIVE_HOST_SOURCE_LOCALIZATION',issues=issues,missing_cells=missing,cells=rows,
                observer_on_vs_off=[finite_metrics.comparison(rows[a],rows[b]) for a,b in ((0,1),(3,2))] if len(rows)==4 else [],
                same_switch_repeats=[finite_metrics.comparison(rows[a],rows[b]) for a,b in ((0,3),(1,2))] if len(rows)==4 else [],
                claim_ceiling='Same16 input documents, four fresh X engines, observer off/on/on/off, own actual future '
                'trajectories. Source localization and observed instrumentation effects only, no performance GO, noise bound, '
                'significance, old-run root-cause attribution, GC savings subtraction or pure GPU timing.')
    with args.out.open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:report[k] for k in ('status','issues','missing_cells')}))


if __name__=='__main__':main()
