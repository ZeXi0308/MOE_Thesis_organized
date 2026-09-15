"""Two fixed absence thresholds from the first complete pure-decode cohort; CPU only."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean
import sys

BASE = Path(__file__).resolve().parents[4]
REPO = BASE.parents[1]
CODE = Path(__file__).resolve().parent/'source'
RAW = BASE/'outputs/admission_capacity/20260915_repeated_kv_service_r01/execution_weste_26862/readback/results/diag-on'
sys.path.insert(0, str(CODE))
from absence_rotation import AbsenceRotation, RequestView
from recovery_progress_model import simulate


class Immediate(AbsenceRotation):
    def __post_init__(self):
        super().__post_init__()
        self.config.min_absence_steps = 0


def summarize(pred, initial):
    high = {r: q['computed'] for r, q in initial['requests'].items()}
    recompute = scheduled = 0
    open_segments, segments = {}, []
    for row in pred['trace']:
        for rid in row['preempted']:
            if rid in open_segments:
                segments.append(dict(open_segments.pop(rid), end='repreempted', end_step=row['step']))
            open_segments[rid] = dict(request=rid, preempt_step=row['step'], first_output_step=None, outputs=0, recompute=0)
        for rid, q in row['scheduled'].items():
            end = q['computed']+q['tokens']
            repeated = max(0, min(end, high[rid])-q['computed'])
            recompute += repeated
            scheduled += q['tokens']
            high[rid] = max(high[rid], end)
            if rid in open_segments:
                open_segments[rid]['recompute'] += repeated
        for rid in row['outputs']:
            if rid in open_segments:
                s = open_segments[rid]
                if s['first_output_step'] is None:
                    s['first_output_step'] = row['step']
                s['outputs'] += 1
        for rid in row['completed']:
            if rid in open_segments:
                segments.append(dict(open_segments.pop(rid), end='completed', end_step=row['step']))
    loads = [dict(step=r['step'], **q) for r in pred['trace'] for q in r.get('loads', [])]
    return dict(status=pred['status'], last_step=pred['last_step'], total_calls_including_shared_prefix=pred['last_step']+1,
        simulated_calls=len(pred['trace']), completed_count=len(pred['completed']), completion_steps=pred['completed'],
        mean_completion_step=mean(pred['completed'].values()), preemptions=sum(len(r['preempted']) for r in pred['trace']),
        resumed=sum(len(r['resumed']) for r in pred['trace']), forced_commits=sum(r['forced'] is not None for r in pred['trace']),
        recovery_recompute_tokens=recompute, executed_tokens=scheduled,
        load_jobs=len(loads), load_tokens=sum(q['tokens'] for q in loads), stored_incremental_tokens=pred['stored_tokens'],
        peak_live_saved_tokens=pred['peak_host_tokens'], staging_events=pred['staging_events'], loads=loads,
        segments=segments, open_segments=list(open_segments.values()),
        zero_output_reinvalidated=sum(s['end']=='repreempted' and s['outputs']==0 for s in segments),
        one_two_output_reinvalidated=sum(s['end']=='repreempted' and 1<=s['outputs']<=2 for s in segments), scope=pred['scope'])


def run(output):
    raw = json.loads((RAW/'raw.json').read_text())
    config = json.loads((RAW/'config.json').read_text())
    selective = json.loads((RAW/'selective-store.json').read_text())
    assert vars(AbsenceRotation().config) == config['rotation_config']
    assert config['requests'] == 32 and config['output_tokens'] == 1024
    qualify = lambda b: len(b['requests'])==len(b['running_ids'])==32 and b['waiting_count']==0 and all(
        q['output_tokens']>0 and q['computed_tokens']==q['prompt_tokens']+q['output_tokens']-1 for q in b['requests'].values())
    memory = next(m for m in raw['memory_trace'] if qualify(m['before']))
    step, before = memory['attempted_step'], memory['before']
    snap = next(s for s in selective['eligibility_snapshots'] if s['step']==step)
    assert snap['cohort_active'] and not snap['waiting_ids'] and not snap['skipped_ids']
    assert not any(snap['tracker'][k] for k in ('absent_since', 'absence_count', 'resident_since'))
    assert snap['tracker']['last_swap_step'] == -10**9
    assert not any(p['attempted_step']<step for p in raw['preemption_events'])
    assert not any(e['event']=='prepare' and e['step']<step for e in selective['events'])
    canonical = lambda rid: 'measured/'+raw['internal_to_source'][rid]
    initial = dict(running=list(map(canonical, before['running_ids'])), waiting=[], free=before['pool']['free_blocks'], requests={})
    for rid, q in before['requests'].items():
        assert len(q['block_counts'])==1
        initial['requests'][canonical(rid)] = dict(computed=q['computed_tokens'], prompt=q['prompt_tokens'], output=q['output_tokens'],
            max_tokens=config['output_tokens'], preemptions=q['num_preemptions'], status='RUNNING', blocks=[None]*q['block_counts'][0])
    assert initial['free']+sum(len(q['blocks']) for q in initial['requests'].values()) == before['pool']['usable_blocks'] == 6656
    history = [dict(step=n, preempted=[], resumed=[]) for n in range(step)]
    predictions = {str(t): simulate(initial, history, 'staged_most_on', selector, RequestView, start_step=step, restore_delay_steps=2)
        for t, selector in ((30, AbsenceRotation), (0, Immediate))}
    rows = {k: summarize(v, initial) for k, v in predictions.items()}
    keys = ('scheduled', 'free_before', 'free_after_schedule', 'outputs', 'preempted', 'resumed', 'forced', 'loads')
    first = next((dict(step=a['step'], threshold30=a, threshold0=b) for a, b in zip(predictions['30']['trace'], predictions['0']['trace'])
        if {k:a.get(k, []) for k in keys}!={k:b.get(k, []) for k in keys}), None)
    first_prepare = {k: next(e for e in p['staging_events'] if e['event']=='prepare') for k, p in predictions.items()}
    # Future native data is confined to this baseline validation, never passed into either simulate call.
    mismatch = None
    for r in predictions['30']['trace']:
        n = r['step']
        actual = {canonical(q['internal_request_id']): dict(computed=q['scheduled_start_computed'], tokens=q['scheduled_tokens'], output=q['output_tokens_before']) for q in raw['scheduler_steps'][n]['scheduled']}
        if list(r['scheduled'].items())!=list(actual.items()) or r['free_after_schedule']!=raw['memory_trace'][n]['after']['pool']['free_blocks'] or len(r['outputs'])!=raw['engine_steps'][n]['new_output_tokens']:
            mismatch = n
            break
    compact_initial = {rid: dict({k:v for k,v in q.items() if k!='blocks'}, held_blocks=len(q['blocks'])) for rid,q in initial['requests'].items()}
    result = dict(evidence='STRUCTURAL', start_step=step, initial_free=initial['free'], initial_requests=compact_initial,
        source_sha256={str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (RAW/'raw.json', CODE/'recovery_progress_model.py', CODE/'absence_rotation.py')},
        frozen=dict(restore_delay_steps=2, thresholds=[30,0], unchanged_config={k:v for k,v in config['rotation_config'].items() if k!='min_absence_steps'}),
        first_prepare=first_prepare, first_trace_difference=first, predictions=rows,
        validation_only=dict(threshold=30, first_mismatch_step=mismatch, actual_calls=len(raw['engine_steps']), predicted_calls=rows['30']['total_calls_including_shared_prefix']),
        boundary='Only before-step98 state and pre-cutoff empty tracker enter prediction; held-block placeholders encode counts, not physical addresses. Known declared output cap1024, closed cohort; no future EOS/load notifications, timings or output identities. Store fences/host allocation assumed successful, no host eviction, fixed delay2; no wall-clock gain or global optimum claim.')
    with output.open('x') as stream:
        stream.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('initial_requests','predictions','first_trace_difference')}, indent=2))
    for k,r in rows.items():
        print(k, {n:v for n,v in r.items() if n not in ('completion_steps','staging_events','loads','segments','open_segments','scope')})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New JSON file; existing outputs are never overwritten')
    run(parser.parse_args().output)
