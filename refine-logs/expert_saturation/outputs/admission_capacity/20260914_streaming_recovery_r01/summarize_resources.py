"""Read-only supplement to the retained streaming analysis; never rewrites raw."""
import collections
import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(path):
    with (gzip.open(path, 'rt') if path.suffix == '.gz' else path.open()) as f:
        return json.load(f)


def counters(value):
    return {k: int(v) for k, v in (line.split() for line in value.splitlines())}


def main():
    source = read(ROOT / 'analysis/analysis.json')
    result = {'status': 'MEASUREMENT_ONLY', 'cells': [], 'boundaries': [
        'Host samples cover initialization, warmup, measurement and teardown, not only the request window.',
        'Shared parent cgroup limit is observed; no independent process-tree hard cap is enforced.',
        'Sampled parent charge maximum is neither kernel peak nor this command peak; memory.peak is unavailable.',
        'Descendant RSS can double count shared pages and miss short-lived/reparented processes.',
        'EOS ID counts refer to IDs returned by the engine, not visible text or client receipt.',
        'Decision-time free blocks are actual observations; no finite horizon or millisecond benefit is inferred.',
    ]}
    for cell in source['cells']:
        label = cell['label']
        folder = ROOT / 'execution/readback/results' / label
        host = folder.with_name(label + '-host')
        terminal = read(host / 'terminal.json')
        samples = [json.loads(line) for line in (host / 'samples.jsonl').read_text().splitlines()]
        assert len(samples) == terminal['samples_written']
        parents = [terminal['prelaunch_parent']] + [s['parent_cgroup'] for s in samples]
        assert all(p['values']['memory.max'] == '96636764160' for p in parents)
        event_start = counters(parents[0]['values']['memory.events'])
        event_end = counters(parents[-1]['values']['memory.events'])
        parent_errors = collections.Counter(e['path'] for p in parents for e in p['errors'])
        process_errors = collections.Counter(str(e) for s in samples for e in s['process_errors'])
        decisions = read(folder / 'headroom-decisions.json')
        raw_path = folder / 'raw.json'
        raw = read(raw_path if raw_path.exists() else folder / 'raw.json.gz')
        requests = {q['request_id']: q for q in raw['requests']}
        eos = read(folder / 'resolved-eos.json')['hf_eos_token_id']
        eos = eos if isinstance(eos, list) else [eos]
        early = []
        for q in requests.values():
            if q['stop_reason'] == 'stop':
                ids = q['output_token_ids']
                early.append(dict(request_id=q['request_id'], returned_ids=len(ids),
                    last_returned_id=ids[-1] if ids else None,
                    last_returned_id_matches_model_eos=bool(ids and ids[-1] in eos),
                    native_stop_reason=q.get('native_stop_reason')))
        eos_count = sum(sum(token in eos for token in q['output_token_ids']) for q in requests.values())
        steps = raw['scheduler_steps']
        active = [s['actual_active'] for s in steps]
        free = [d['free_before'] for d in decisions]
        proposals = collections.Counter((d.get('proposal') or {}).get('reason', 'no_proposal') for d in decisions)
        res = cell['lifecycle']['residencies']
        result['cells'].append(dict(label=label,
            host=dict(status=terminal['status'], child_returncode=terminal['returncode'],
                child_running_at_monitor_exit=terminal['child_running_at_monitor_exit'],
                samples=len(samples), command_wall_s=terminal['finished_unix_s']-terminal['started_unix_s'],
                parent_limit_bytes=96636764160,
                parent_sampled_charge_max_bytes=max(int(p['values']['memory.current']) for p in parents),
                parent_memory_event_delta={k: event_end[k]-v for k, v in event_start.items()},
                observed_swap_current_values=sorted({int(p['values']['memory.swap.current']) for p in parents}),
                observed_swap_max_values=sorted({p['values']['memory.swap.max'] for p in parents}),
                sampled_tree_rss_sum_max_bytes=max(s['sampled_process_tree_rss_sum_bytes'] for s in samples),
                parent_errors=dict(parent_errors), process_errors=dict(process_errors),
                sampling_cpu_s=terminal['sampling_cpu_s'], sampling_wall_s=terminal['sampling_wall_s']),
            gpu=dict(before=read(folder / 'memory-before.json'), after=read(folder / 'memory-after.json'),
                qualification=read(folder / 'safe-cap-qualification.json')),
            pressure=dict(steps=len(steps), actual_active_max=max(active),
                actual_active_step_histogram=dict(sorted(collections.Counter(active).items())),
                free_before_min=min(free), free_before_zero_steps=free.count(0),
                free_after_min=min(d['free_after'] for d in decisions),
                proposal_reasons=dict(proposals),
                max_held_requests=max(len(d['held']) for d in decisions),
                decision_wall_s=sum(d['decision_seconds'] for d in decisions),
                waiting_before_max=max(s['waiting_before'] for s in steps),
                preemption_steps=[dict(step=d['step'], free_before=d['free_before'], free_after=d['free_after'],
                    victims=d['preempted']) for d in decisions if d['preempted']]),
            lifecycle=dict(episodes=len(res), restored_before_first_output=sum(r['outputs_before']==0 for r in res),
                generation_recoveries=sum(r['outputs_before']>0 for r in res),
                new_outputs_after_recovery=[r['new_outputs'] for r in res],
                recovery_start_to_first_output_s=[r['recovery_start_to_first_output_s'] for r in res],
                end_reason_counts=dict(collections.Counter(r['end_reason'] for r in res))),
            eos=dict(early_stops=early, model_eos_ids=eos, returned_eos_id_count=eos_count,
                returned_id_count=cell['metrics']['output_tokens'],
                returned_non_eos_id_count=cell['metrics']['output_tokens']-eos_count,
                terminal_without_new_ids=sum(e['finished'] and e['chunk_size']==0 for e in raw['output_events']))))
        print(label, json.dumps({k: result['cells'][-1][k] for k in ('host','lifecycle','eos')}))
    out = ROOT / 'analysis/resource_summary.json'
    with out.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')


if __name__ == '__main__':
    main()
