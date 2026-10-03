#!/usr/bin/env python3
"""Fixed-pool full/chunk reservation comparison using retained native accounting."""
import argparse
from collections import Counter
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

LABELS = ('repeat0-full', 'repeat0-chunk', 'repeat1-chunk', 'repeat1-full')
KV_BYTES, KV_BLOCKS = 16089350144, 7671


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    with (gzip.open if path.suffix == '.gz' else open)(path, 'rt') as stream:
        return json.load(stream)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def repreemptions(events):
    """Use native generated-prefix counters, not delayed host receipt timestamps."""
    previous, rows = {}, []
    for event in events:
        if not event['original_preemption_returned']:
            continue
        rid, step = event['request_id'], event['attempted_step']
        outputs = event['victim_state']['output_tokens']
        prior = previous.get(rid)
        require(prior is None or (step >= prior[0] and outputs >= prior[1]), 'preemption history reversed')
        delta = outputs-prior[1] if prior else None
        rows.append(dict(request_id=rid, step=step, output_tokens=outputs,
                         previous_preemption_step=prior[0] if prior else None,
                         new_outputs_since_previous_preemption=delta,
                         repreempted_before_new_output=delta == 0))
        previous[rid] = (step, outputs)
    return dict(count=sum(r['repreempted_before_new_output'] for r in rows), events=rows,
                scope='Consecutive successful native preemptions of the same request; no new generated token between events. Not all recomputation is wasted work.')


def inspect(directory, label, native, prepared):
    policy = label.split('-')[1]
    row = dict(label=label, policy=policy, status='UNRUN', full_episode_comparison_eligible=False)
    try:
        terminal = read(directory/'status.json') if (directory/'status.json').exists() else None
        row['terminal_status'] = terminal
        raw_path = next((directory/n for n in ('raw.json', 'raw.json.gz') if (directory/n).exists()), None)
        if raw_path is None:
            row['status'] = 'INCOMPLETE' if terminal and terminal.get('status') != 'UNRUN' else 'UNRUN'
            return row
        raw = read(raw_path)
        row.update(raw_status=raw['status'], raw_path=str(raw_path), status='INCOMPLETE',
                   requests=[dict(request_id=r['request_id'], status=r['status'], output_tokens=len(r['output_token_ids'])) for r in raw['requests']])
        needed = ('config.json', 'engine_args.json', 'status.json', 'metrics.json', 'memory-before.json', 'safe-cap-qualification.json')
        missing = [n for n in needed if not (directory/n).exists()]
        if missing:
            row['missing_artifacts'] = missing
            return row
        config, engine, memory, qualification = [read(directory/n) for n in
            ('config.json', 'engine_args.json', 'memory-before.json', 'safe-cap-qualification.json')]
        require(config['reservation_policy'] == policy and engine['scheduler_reserve_full_isl'] is (policy == 'full'), 'reservation policy mismatch')
        require(config['cap'] == raw['target_cap'] == 32 and config['prompt_tokens'] == 3072 and config['output_tokens'] == 1024, 'workload/cap changed')
        require(engine['max_num_batched_tokens'] == 1024 and engine['kv_cache_memory_bytes'] == memory['kv_storage_bytes'] == KV_BYTES, 'token/physical KV budget changed')
        require(qualification['status'] == 'QUALIFIED' and qualification['usable_blocks'] == KV_BLOCKS, 'actual pool unqualified or changed')
        require(qualification['observed_scheduler_reserve_full_isl'] is (policy == 'full'), 'live scheduler reservation flag differs from arm')
        require(raw['preemption_mode'] == config['preemption_mode'] == 'native_recompute', 'preemption mode changed')
        identity = native.base.validate_identity(raw, config, prepared, 'long', 32)
        metrics = native.base.summarize_episode_requests(raw['requests'], observation_end_s=raw['observation_end_s'], ttft_slo_s=5.0, tpot_slo_s=0.2)
        saved = read(directory/'metrics.json')
        require(all(k in saved and saved[k] == v for k, v in metrics.items()), 'saved metrics mismatch')
        work, calls = native.execution_accounting(raw)
        events = native.preemptions(raw, calls)
        effects = native.request_effects(raw, events, work)
        pools = [t[k]['pool'] for t in raw['memory_trace'] for k in ('before', 'after') if t[k] is not None]
        pools += [e[k] for e in raw['preemption_events'] for k in ('pool', 'pool_after') if k in e]
        pools += [a[k] for t in raw['memory_trace'] for a in t['allocation_failures'] for k in ('before', 'after')]
        require(bool(pools) and all(0 <= p['used_blocks'] <= p['usable_blocks'] == KV_BLOCKS and p['used_blocks']+p['free_blocks'] == KV_BLOCKS for p in pools), 'observed block accounting mismatch')
        complete = raw['status'] == terminal['status'] == 'COMPLETE' and identity['all_requests_completed'] and metrics['n_completed'] == 32 and raw['capacity_boundary'] is None
        latencies = [r['completion_latency_s'] for r in effects['per_request'] if r['completion_latency_s'] is not None]
        row.update(status='COMPLETE' if complete else 'INCOMPLETE', full_episode_comparison_eligible=complete,
                   identity_check=identity, metrics=metrics, requests=effects['per_request'], effects=effects,
                   mean_completion_s=sum(latencies)/len(latencies) if latencies else None,
                   work=work, preemption_events=events, preemptions=raw['actual_preemption_count'],
                   repeated_preemption=repreemptions(events), physical_kv_storage_bytes=memory['kv_storage_bytes'],
                   token_timing_scope=raw.get('host_chunk_diagnostics', {'token_level_itl_resolved': 'UNVERIFIED'}),
                   max_itl_s=max((r['longest_itl']['itl_s'] for r in effects['per_request'] if r['longest_itl']), default=None),
                   usable_blocks=qualification['usable_blocks'], engine_args=engine, saved_metrics_match=True,
                   output_hashes={r['request_id']: hashlib.sha256(json.dumps(r['output_token_ids']).encode()).hexdigest() for r in raw['requests']})
    except (OSError, json.JSONDecodeError) as exc:
        row.update(status='INCOMPLETE', error=str(exc))
    except (KeyError, ValueError, TypeError, IndexError) as exc:
        row.update(status='INVALID_EVIDENCE', error=str(exc), full_episode_comparison_eligible=False)
    return row


def comparisons(rows):
    pairs = []
    for repeat in (0, 1):
        full, chunk = [next(r for r in rows if r['label'] == f'repeat{repeat}-{arm}') for arm in ('full', 'chunk')]
        pair = dict(repeat=repeat, baseline=full['label'], action=chunk['label'], status='UNRUN_OR_INCOMPLETE')
        if full['full_episode_comparison_eligible'] and chunk['full_episode_comparison_eligible']:
            engines = [{k:v for k,v in r['engine_args'].items() if k != 'scheduler_reserve_full_isl'} for r in (full, chunk)]
            matched = engines[0] == engines[1] and full['identity_check']['workload_sha256'] == chunk['identity_check']['workload_sha256']
            pair.update(status='DESCRIPTIVE_MATCHED_PAIR' if matched else 'UNMATCHED', engine_args_equal_except_reservation=engines[0] == engines[1])
            if matched:
                a, b = full['metrics'], chunk['metrics']
                pair.update(throughput_relative_change=b['throughput_rps']/a['throughput_rps']-1,
                            wall_relative_change=b['observation_duration_s']/a['observation_duration_s']-1,
                            mean_completion_relative_change=chunk['mean_completion_s']/full['mean_completion_s']-1,
                            max_itl_change_s=chunk['max_itl_s']-full['max_itl_s'],
                            request_max_itl_p99_change_s=chunk['effects']['request_max_itl_s']['p99']-full['effects']['request_max_itl_s']['p99'],
                            preemptions_change=chunk['preemptions']-full['preemptions'],
                            recomputed_positions_change=chunk['work']['totals'].get('recomputed_tokens', 0)-full['work']['totals'].get('recomputed_tokens', 0),
                            output_token_sequences_equal=chunk['output_hashes'] == full['output_hashes'])
        pairs.append(pair)
    return pairs


def readable(result):
    lines = ['# Fixed KV pool: full versus chunk reservation', '', result['status'], '',
             '| Cell | Status | Completed | Wall s | Requests/s | Mean completion s | Maximum ITL s | Request max-ITL p99 s | Pooled ITL p99 s | Preemptions / repeated before output / recomputed positions |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for r in result['cells']:
        if not r['full_episode_comparison_eligible']:
            counts = Counter(x['status'] for x in r.get('requests', []))
            lines.append(f"| {r['label']} | {r['status']} | {counts.get('completed', 0)}/{len(r.get('requests', []))} | — | — | — | — | — | — | — |")
            continue
        m = r['metrics']
        lines.append(f"| {r['label']} | COMPLETE | {m['n_completed']}/{m['n_arrived']} | {m['observation_duration_s']:.6f} | {m['throughput_rps']:.6f} | {r['mean_completion_s']:.6f} | {r['max_itl_s']:.6f} | {r['effects']['request_max_itl_s']['p99']:.6f} | {m['latency_s']['itl']['p99']:.6f} | {r['preemptions']} / {r['repeated_preemption']['count']} / {r['work']['totals'].get('recomputed_tokens', 0)} |")
    return '\n'.join(lines + ['', 'Each request and all partial cells remain in analysis.json. Only complete matched pairs are compared.', '', *('- '+json.dumps(p) for p in result['comparisons']), '', result['scope'], ''])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output_dir.exists(), 'output directory must be new; refusing overwrite')
    outputs = Path(__file__).resolve().parents[2]/'outputs/admission_capacity'
    metrics_path = args.run_dir/'frozen/metrics.py'
    module('metrics', metrics_path)
    native = module('recovery_native_helpers', outputs/'20260908_native_preemption_r01/analyze_native_preemption.py')
    rows = [inspect(args.run_dir/'gpu_results'/label, label, native, args.run_dir/'frozen/inputs_preparation/prepared') for label in LABELS]
    pairs = comparisons(rows)
    result = dict(status='MEASUREMENT_ONLY' if all(p['status'] == 'DESCRIPTIVE_MATCHED_PAIR' for p in pairs) else 'UNRUN' if all(r['status'] == 'UNRUN' for r in rows) else 'INCOMPLETE_CAMPAIGN',
                  cells=rows, comparisons=pairs, metrics_source=str(metrics_path), metrics_sha256=hashlib.sha256(metrics_path.read_bytes()).hexdigest(),
                  scope='Single-model native in-process fixed-pool observation. Request times include waiting, recomputation and instrumentation. Work counts are executed interval overlap, not pure GPU time. Repeated preemption identifies no new generated output between consecutive preemptions, not causal wasted-time savings. Small repeated document cohorts do not establish production tails, quality or policy generalization.')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir/'analysis.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    (args.output_dir/'report.md').write_text(readable(result))
    print(json.dumps(dict(status=result['status'], comparisons=pairs), indent=2))


if __name__ == '__main__':
    main()
