#!/usr/bin/env python3
"""Read-only cross-run first-recovery diagnostic; all calls and full wall retained."""
import argparse
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'refine-logs').is_dir())
BASE = ROOT / 'refine-logs/expert_saturation/outputs/admission_capacity'
OLD = BASE / '20260913_rotation_strong_baseline_r01'
NEW = BASE / '20260914_rotation_runtime_repeat_r01'
HELPER = OLD / 'diagnostics/most_output_block_difference.py'
spec = importlib.util.spec_from_file_location('old_diagnostic', HELPER)
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)
SKIP = ('start_s', 'end_s', 'waiting_before', 'waiting_requests')

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def cell(base, name, summary):
    paths = [next(p for p in (base/'raw.json', base/'raw.json.gz') if p.exists())]
    paths += [base/n for n in ('headroom-decisions.json', 'status.json', 'config.json', 'engine_args.json')]
    raw, decisions = h.read_raw(paths[0]), h.read_raw(paths[1])
    assert raw['status'] == 'COMPLETE' and raw['host_chunk_diagnostics']['token_level_itl_resolved']
    steps, calls = raw['scheduler_steps'], raw['engine_steps']
    assert len(steps) == len(calls) == len(decisions)
    assert all(c['completed'] and c['scheduler_step_start'] == i and c['scheduler_step_end'] == i+1 for i,c in enumerate(calls))
    aliases = dict(raw['internal_to_source'])
    aliases.update({r['external_request_id']: r['request_id'] for r in raw['requests']})
    norm = lambda value, skip=(): h.normalize(value, aliases, skip)
    fingerprints = dict(executed_schedule=digest(norm(steps, SKIP)),
        decisions=digest(norm(decisions, ('decision_seconds', 'wrapped_schedule_seconds'))),
        engine_logical_outputs=digest(norm(calls, ('start_s', 'returned_s'))),
        output_sequences=digest({r['request_id']: r['output_token_ids'] for r in raw['requests']}))
    event = h.first_recovery(raw, decisions)
    if event:
        first, last = event['first_step'], event['last_step']
        event['window_work_sha256'] = digest(norm(steps[first:last+1], SKIP+('step',)))
        event['final_call_composition'] = norm(steps[last]['scheduled'])
        event['final_call_composition_sha256'] = digest(event['final_call_composition'])
        event['recovery_calls'] = last-first+1
        event['recovery_span_s'] = calls[last]['returned_s']-calls[first]['start_s']
        event['recovery_engine_s'] = sum(c['returned_s']-c['start_s'] for c in calls[first:last+1])
        event['recovery_scheduler_s'] = sum(s['end_s']-s['start_s'] for s in steps[first:last+1])
        event['recovery_engine_nonscheduler_s'] = event['recovery_engine_s']-event['recovery_scheduler_s']
        event['final_call']['engine_nonscheduler_s'] = event['final_call']['engine_s']-event['final_call']['scheduler_s']
    engine = sum(c['returned_s']-c['start_s'] for c in calls)
    scheduler = sum(s['end_s']-s['start_s'] for s in steps)
    result = dict(name=name, inputs=[h.fingerprint(p) for p in paths], fingerprints=fingerprints,
        config_sha256=digest(h.read_raw(paths[3])), engine_args_sha256=digest(h.read_raw(paths[4])),
        calls=len(calls), requests=len(raw['requests']), wall_s=raw['observation_end_s'], scheduler_s=scheduler,
        engine_nonscheduler_s=engine-scheduler, outside_engine_s=raw['observation_end_s']-engine,
        first_successful_forced_recovery=event)
    if summary:
        result['existing_recovery_summary'] = {k:summary[k] for k in ('forced','natural','held_request_steps','work_totals')}
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--new-analysis', type=Path)
    args = parser.parse_args()
    groups = [('reference', OLD/'execution02_westc_53036'), ('repeat', NEW/'execution02_after_finite')]
    sources = [(g, b, p/'gpu_results'/f'cohort2-block{b}-most_output') for g,p in groups for b in (0,1)]
    missing = [str(p) for _,_,p in sources if not (p/'status.json').exists() or
        h.read_raw(p/'status.json').get('status') != 'COMPLETE' or not any((p/n).exists() for n in ('raw.json','raw.json.gz'))]
    if missing:
        print(json.dumps(dict(status='NOT_READY', missing_or_incomplete=missing))); return
    summaries, inputs = {}, []
    for group, path in [('reference', OLD/'analysis02_westc_53036/analysis.json'), ('repeat', args.new_analysis)]:
        if path:
            analysis = h.read_raw(path)
            assert analysis['status'] == 'MEASUREMENT_ONLY'
            inputs.append(h.fingerprint(path))
            summaries[group] = {r['label']:r for r in analysis['recovery_accounting']}
    cells = [cell(p, f'{g}/block{b}', summaries.get(g, {}).get(p.name)) for g,b,p in sources]
    event_keys = ('target','victim','outputs_before','recomputed_positions','window_work_sha256','final_call_composition_sha256')
    pairs = []
    for a,b in itertools.combinations(cells, 2):
        ea, eb = [c['first_successful_forced_recovery'] for c in (a,b)]
        match = bool(ea and eb and all(ea[k] == eb[k] for k in event_keys))
        pairs.append(dict(a=a['name'], b=b['name'], fingerprints_equal={k:a['fingerprints'][k]==b['fingerprints'][k] for k in a['fingerprints']},
            same_first_recovery_event=match, final_call_b_minus_a_s=eb['final_call']['engine_s']-ea['final_call']['engine_s'] if match else None))
    result = dict(status='MEASUREMENT_ONLY', evidence_type='NATIVE_SERVING_INPROCESS_HOST_CAPTURE',
        question='Do the two new most_output executions repeat the old first forced-swap recovery event and final-call cost?',
        accounting='wall = scheduler + engine_nonscheduler + outside_engine; recovery and final-call costs overlap these totals.',
        scope='All calls retained. No trimming, significance, JIT/root-cause, new holdout, or method claim.',
        normalization=dict(schedule_skip=SKIP, decision_skip=['decision_seconds','wrapped_schedule_seconds'], engine_skip=['start_s','returned_s'],
            request_ids='Internal and external IDs mapped to source IDs; dictionary keys sorted; list order retained.'),
        code_inputs=[h.fingerprint(Path(__file__)), h.fingerprint(HELPER)], summary_inputs=inputs, cells=cells, comparisons=pairs,
        invocation=dict(argv=sys.argv, cwd=str(Path.cwd())))
    out = Path(__file__).resolve().parent
    assert not any((out/n).exists() for n in ('comparison.json','comparison.md')), 'refusing to overwrite results'
    lines = ['MEASUREMENT_ONLY — first forced-swap recovery comparison.', '',
        '| Cell | Calls | Full wall s | Scheduler s | Engine nonscheduler s | Recovery span s | Final call s |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for c in cells:
        e = c['first_successful_forced_recovery']
        lines.append(f"| {c['name']} | {c['calls']} | {c['wall_s']:.6f} | {c['scheduler_s']:.6f} | {c['engine_nonscheduler_s']:.6f} | {e['recovery_span_s'] if e else 'absent'} | {e['final_call']['engine_s'] if e else 'absent'} |")
    lines += ['', *[f"{p['a']} → {p['b']}: full execution/decision/output fingerprints equal={all(p['fingerprints_equal'].values())}; first recovery event equal={p['same_first_recovery_event']}." for p in pairs], '', result['accounting'], result['scope'], '', 'Actual target/victim, outputs before recovery, scheduled work and source hashes: comparison.json. Cost values are descriptive; no fixed step index or recurrence threshold was assumed.']
    with (out/'comparison.json').open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
    with (out/'comparison.md').open('x') as stream:
        stream.write('\n'.join(lines)+'\n')
    print(json.dumps(dict(status=result['status'], output=str(out))))

if __name__ == '__main__':
    main()
