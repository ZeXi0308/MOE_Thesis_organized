"""Observe post-completion ownership separately from native store fences."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def analyze(raw, scope, selective):
    origin = raw['measurement_origin_perf_counter_s']
    trace = raw['memory_trace']
    calls = {c['returned_s']: c for c in raw['engine_steps'] if c['completed']}
    finals = {e['request_id']: e for e in raw['output_events'] if e['finished']}
    jobs = scope['native_full_scope']['observed_store_jobs']
    flushes = {j['job_id']: e for e in scope['native_full_scope']['flush_events'] for j in e['jobs']}
    counts = Counter()
    rows = []
    for r in raw['requests']:
        assert r['status'] == 'completed'
        event = finals[r['request_id']]
        call = calls[event['received_s']]
        assert r['completion_s'] == event['received_s']
        assert r['external_request_id'] in call['output_request_ids']
        rid = r['internal_request_id']
        next_step = call['scheduler_step_end']
        last = trace[next_step-1]
        held_at_final_schedule = sum(last['after']['requests'][rid]['block_counts'])
        points = []
        for m in trace[next_step:]:
            for phase, time_key in [('before','host_start_perf_counter_s'), ('after','host_end_perf_counter_s')]:
                state = m[phase]
                blocks = sum(state['requests'].get(rid, {}).get('block_counts', []))
                points.append(dict(step=m['attempted_step'], phase=phase, time_s=m[time_key]-origin,
                    request_present=rid in state['requests'], blocks=blocks, free=state['pool']['free_blocks']))
                if blocks == 0:
                    break
            if points[-1]['blocks'] == 0:
                break
        category = ('no_later_snapshot' if not points else
                    'held_after_output_return' if points[0]['blocks'] else 'zero_by_first_next_snapshot')
        counts[category] += 1
        stores = []
        for job in jobs:
            if job['request'] != rid:
                continue
            done = job['completed_perf_s']-origin
            meta = job['metadata_perf_s']-origin
            if done <= r['completion_s']:
                continue
            stores.append(dict(job_id=job['job_id'], finished_at_metadata=job['finished_at_metadata'],
                metadata_step=job['metadata_step'], metadata_s=meta, worker_completion_report_s=done,
                source_blocks=job['source_gpu_blocks'], logical_indices=job['logical_chunk_indices'],
                flush_step=flushes[job['job_id']]['step'] if job['job_id'] in flushes else None,
                flush_metadata_s=flushes[job['job_id']]['metadata_perf_s']-origin
                    if job['job_id'] in flushes else None))
        # Sparse eligibility captures actual request status and held count;
        # unlike trace request absence, these may retain connector-only state.
        after_snapshot = next((s for s in selective['eligibility_snapshots']
            if s['host_perf_counter_s'] >= origin+r['completion_s']), None)
        rows.append(dict(request_id=r['request_id'], internal_request_id=rid,
            final_output_call=call['call_index'], completion_s=r['completion_s'],
            final_schedule_step=next_step-1, held_blocks_at_final_schedule=held_at_final_schedule,
            category=category, later_ownership_snapshots=points,
            store_jobs_reported_complete_after_output=stores,
            next_eligibility_step=after_snapshot['step'] if after_snapshot else None))
    late = [j for r in rows for j in r['store_jobs_reported_complete_after_output']]
    return dict(status='MEASUREMENT_ONLY', request_categories=dict(counts),
        late_store_counts=dict(jobs=len(late), finished_at_metadata=sum(j['finished_at_metadata'] for j in late),
            flush_jobs=sum(j['flush_step'] is not None for j in late)), requests=rows,
        semantics=[
            'Completion is an engine-output-return boundary; schedule snapshots do not observe the precise KV free call.',
            'Zero at the first subsequent snapshot bounds scheduler ownership release; the time difference is not measured free latency.',
            'Worker completion report can lag hardware completion and does not timestamp a GPU release callback.',
            'Registered source block plus native flush metadata is a dependency observation, not measured exposed wait duration.',
            'Logical allocator availability and fence-free physical reuse are distinct; native block flush provides the latter.',
            'The last completion without a later snapshot remains unobserved, not guessed from request completion.',
            'This reuses one diagnostic, does not compare policy performance, and does not change allocation or store semantics.'])


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--cell', type=Path, required=True)
    p.add_argument('--scope-analysis', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    raw_bytes = (args.cell/'raw.json').read_bytes()
    result = analyze(json.loads(raw_bytes), json.loads(args.scope_analysis.read_text()),
                     json.loads((args.cell/'selective-store.json').read_text()))
    result['source'] = dict(raw=str(args.cell/'raw.json'), raw_sha256=hashlib.sha256(raw_bytes).hexdigest(),
                           scope_analysis=str(args.scope_analysis))
    with args.output.open('x') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        f.write('\n')
    print(json.dumps(dict(categories=result['request_categories'], late_stores=result['late_store_counts'])))
