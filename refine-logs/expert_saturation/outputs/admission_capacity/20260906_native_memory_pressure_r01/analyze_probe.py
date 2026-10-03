#!/usr/bin/env python3
"""Summarize all expected capacity cells, including missing and boundary stops."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import tarfile
from metrics import summarize_episode_requests


def read(path):
    return json.loads(path.read_text())


def validate_identity(raw, config, prepared, domain, cap):
    """Bind each retained cell to the frozen domain, cap, arrivals and tokens."""
    frozen = read(prepared/domain/'config.json')
    workload = read(prepared/domain/'workload.json')
    digest = hashlib.sha256(json.dumps(workload, sort_keys=True).encode()).hexdigest()
    if digest != frozen['workload_sha256'] or digest != config['workload_sha256']:
        raise ValueError('frozen workload hash mismatch')
    if config['domain'] != domain or config['cap'] != cap or raw['target_cap'] != cap:
        raise ValueError('directory/config/raw domain or cap mismatch')
    if any(config[k] != frozen[k] for k in ['model', 'requests', 'prompt_tokens', 'output_tokens', 'seed']):
        raise ValueError('cell differs from frozen input configuration')
    if raw['regime'] != 'steady' or raw['arrival_scale'] != 1.0:
        raise ValueError('unexpected arrival regime or scale')
    sources, tokens = workload['source_requests'], workload['actual_prompt_token_ids']
    arrivals = workload['arrival_traces_s']['steady']
    indexed = {r['request_id']: r for r in raw['requests']}
    if not (len(indexed) == len(raw['requests']) == len(sources) == len(tokens) == len(arrivals) == config['requests']):
        raise ValueError('request identity or count mismatch')
    if set(indexed) != {s['request_id'] for s in sources}:
        raise ValueError('request identity set mismatch')
    for source, ids, arrival in zip(sources, tokens, arrivals):
        row = indexed[source['request_id']]
        token_hash = hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()
        if not (row['document_id'] == source['document_id'] and
                row['prompt_token_ids_sha256'] == source['prompt_token_ids_sha256'] == token_hash and
                row['prompt_tokens'] == len(ids) == config['prompt_tokens'] and
                math.isclose(row['arrival_s'], arrival, rel_tol=0, abs_tol=1e-12)):
            raise ValueError('request document/prompt/arrival mismatch')
        if row['status'] == 'completed' and not (
                len(row['output_token_ids']) == len(row['token_times_s']) == config['output_tokens']):
            raise ValueError('completed request output length mismatch')
    if any(s['target_cap'] != cap for s in raw['scheduler_steps']):
        raise ValueError('scheduler target cap changed')
    return dict(checked_requests=len(indexed), workload_sha256=digest,
                frozen_directory=str((prepared/domain).resolve()),
                all_requests_completed=all(r['status']=='completed' for r in raw['requests']))


def locate_boundary(raw):
    if not raw['capacity_boundary']:
        return None
    first = next(t for t in raw['memory_trace'] if t['allocation_failures'])
    before = first['before']
    running = [before['requests'][rid] for rid in before['running_ids']]
    computed = [r['computed_tokens'] for r in running]
    counts = [len(r['output_token_ids']) for r in raw['requests']]
    return dict(attempted_step=first['attempted_step'], last_successful_step=len(raw['scheduler_steps'])-1,
        schedule_completed=first['schedule_completed'], model_execution_confirmed=first['model_execution_confirmed'],
        running_before=len(running), waiting_before=before['waiting_count'],
        decode_eligible_before=sum(r['computed_tokens'] >= r['prompt_tokens'] for r in running),
        running_after=len(first['after']['running_ids']), before_pool=before['pool'], after_pool=first['after']['pool'],
        computed_token_range=[min(computed), max(computed)], allocation_failures=first['allocation_failures'],
        generated_token_range=[min(counts), max(counts)], request_status_counts={s:sum(r['status']==s for r in raw['requests'])
            for s in sorted({r['status'] for r in raw['requests']})},
        interpretation='First allocation failure; caller may remove victim before guard raises. No GPU forward for this attempt. '
            'Failed labels on unfinished trajectories reflect episode abort; truncated rates and partial TPOT are not completed-request performance.')


def verify_execution(rows, root):
    names = ['run_probe.py', 'memory_telemetry.py', 'native_capture.py', 'metrics.py']
    archive = Path(__file__).resolve().parent/'execution-20260908.tar.gz'
    with tarfile.open(archive, 'r:gz') as tar:
        archived = {name:hashlib.sha256(tar.extractfile(name).read()).hexdigest() for name in names}
    environments = {r['label']:read(root/r['label']/'environment.json') for r in rows
                    if (root/r['label']/'environment.json').exists()}
    warmups=[]
    for row in rows:
        for path in sorted((root/row['label']).glob('warmup-*.json')):
            raw=read(path)
            warmups.append(dict(cell=row['label'], path=str(path), status=raw['status'],
                request_count=len(raw['requests']), completed_requests=sum(r['status']=='completed' for r in raw['requests']),
                all_output_lengths_16=all(len(r['output_token_ids'])==16 for r in raw['requests'])))
    fields = ['python', 'torch', 'cuda', 'vllm', 'transformers', 'cpu_threads', 'vllm_source_sha256']
    first = next(iter(environments.values()), {})
    return dict(archive=str(archive), archive_source_sha256=archived,
        per_cell_source_match={label:e['source_sha256']==archived for label,e in environments.items()},
        software_fields_equal={k:all(e.get(k)==first.get(k) for e in environments.values()) for k in fields},
        environments=environments,
        warmups=warmups, warmup_count=len(warmups),
        analysis_metrics_matches_execution=hashlib.sha256((Path(__file__).resolve().parent/'metrics.py').read_bytes()).hexdigest()==archived['metrics.py'],
        limit='Same engine arguments do not imply equal profiled KV pool sizes. Per-engine actual usable blocks are reported.')


def readable(report):
    lines = ['# Natural context capacity: retained execution analysis', '', report['status'], '',
        'All eight planned cells remain listed. Full-episode rates are reported only for complete cells. Reference SLO 5 s / 200 ms is secondary.', '',
        '| Cell | Status | Complete | Active/decode/wait max | Used/usable blocks peak | KV peak fraction | Min free | Full episode s | Full throughput req/s | TTFT p50 s | TPOT p50 ms |',
        '|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|']
    for r in report['cells']:
        if 'metrics' not in r:
            lines.append(f"| {r['label']} | {r['status']} | — | — | — | — | — | — | — | — | — |")
            continue
        m=r['metrics']; eligible=r['full_episode_comparison_eligible']; pool=r['pool_ranges']
        full=(f"{m['observation_duration_s']:.5f} | {m['throughput_rps']:.5f} | {m['latency_s']['ttft']['p50']:.5f} | {m['latency_s']['tpot']['p50']*1000:.5f}"
              if eligible else '— | — | — | —')
        lines.append(f"| {r['label']} | {r['status']} | {m['n_completed']}/32 | {r['max_scheduled_active']}/{r['max_decode_active']}/{r['max_waiting']} | "
            f"{pool['used_blocks'][1]}/{pool['usable_blocks'][0]} | {r['max_used_fraction']:.5f} | {r['minimum_free_blocks']} | {full} |")
    lines += ['', '## Capacity stops', '']
    for r in report['cells']:
        b=r.get('boundary_location')
        if b:
            lines.append(f"- {r['label']}: attempt {b['attempted_step']} (zero based), {b['running_before']} running / {b['waiting_before']} waiting, "
                f"all-decode eligibility {b['decode_eligible_before']}; computed tokens {b['computed_token_range']}. Before {b['before_pool']}; "
                f"allocation failure {b['allocation_failures']}; generated {b['generated_token_range']} of 1024 tokens. "
                f"No destructive preemption or GPU execution for failed attempt. Stop time {r['metrics']['observation_duration_s']:.5f} s is a truncated horizon.")
    verification=report['execution_verification']
    lines += ['', '## Accounting and qualification', '',
        f"Identity-checked cells: {sum('identity_check' in r for r in report['cells'])}; engine arguments equal: {report['engine_args_equal']}.",
        f"Four execution-source hashes match archive for every observed cell: {all(verification['per_cell_source_match'].values())}; "
        f"software fields equal: {all(verification['software_fields_equal'].values())}; analysis metrics match executed metrics: {verification['analysis_metrics_matches_execution']}.",
        f"Retained full warmup raw files: {verification['warmup_count']}; all completed with 16 outputs per request: "
        f"{all(w['status']=='COMPLETE' and w['completed_requests']==w['request_count'] and w['all_output_lengths_16'] for w in verification['warmups'])}.",
        'KV occupied blocks are inside the preallocated KV pool. Expert bytes are a subset of parameter bytes; Torch reserved includes allocated.',
        'Per-engine KV pool sizes may differ despite common arguments. Boundary checks do not certify continuous GPU isolation.',
        'Reference SLO request counts and all observed partial metrics remain in JSON; partial metrics must not be compared as completed performance.', '',
        *[f'- {s}' for s in report['limits']], '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    rows, engine_args = [], []
    for repeat in [0, 1]:
        for domain in ['short', 'long']:
            for cap in [16, 32]:
                label = f'repeat{repeat}-{domain}-cap{cap}'
                directory = args.results_dir/label
                row = dict(label=label, domain=domain, cap=cap, repeat=repeat, status='MISSING')
                rows.append(row)
                terminal = read(directory/'status.json') if (directory/'status.json').exists() else None
                if not (directory/'raw.json').exists():
                    if terminal is not None:
                        row.update(status='INCOMPLETE', terminal_status=terminal,
                            missing_artifacts=['raw.json'], full_episode_comparison_eligible=False)
                    continue
                raw = read(directory/'raw.json')
                required = ['config.json', 'environment.json', 'status.json', 'metrics.json', 'engine_args.json', 'memory-before.json',
                            'memory-after.json', 'gpu-after.json']
                missing = [name for name in required if not (directory/name).exists()]
                if missing:
                    row.update(status='INCOMPLETE', terminal_status=terminal, raw_status=raw['status'],
                        missing_artifacts=missing, full_episode_comparison_eligible=False,
                        capacity_boundary=raw.get('capacity_boundary'),
                        requests_completed=sum(r['status']=='completed' for r in raw['requests']))
                    continue
                config = read(directory/'config.json')
                try:
                    identity = validate_identity(raw, config,
                        Path(__file__).resolve().parent/'inputs_preparation/prepared', domain, cap)
                except (KeyError, ValueError, TypeError) as exc:
                    row.update(status='INVALID', identity_error=str(exc), raw_status=raw['status'],
                        full_episode_comparison_eligible=False)
                    continue
                metrics = summarize_episode_requests(raw['requests'], observation_end_s=raw['observation_end_s'],
                    ttft_slo_s=5.0, tpot_slo_s=0.2)
                saved = read(directory/'metrics.json')
                if any(saved[k] != v for k, v in metrics.items()):
                    raise ValueError(f'{label}: saved metrics mismatch')
                engine_args.append(read(directory/'engine_args.json'))
                traces = raw['memory_trace']
                pools = [t[k]['pool'] for t in traces for k in ['before', 'after'] if t[k] is not None]
                for pool in pools:
                    if not (0 <= pool['used_blocks'] <= pool['usable_blocks']
                            and pool['used_blocks']+pool['free_blocks'] == pool['usable_blocks']):
                        raise ValueError(f'{label}: invalid block accounting')
                failed = [t for t in traces if not t['schedule_completed']]
                if any(t['model_execution_confirmed'] is not False for t in failed):
                    raise ValueError('failed scheduling attempt claimed model execution')
                boundary = raw['capacity_boundary']
                if boundary and boundary['original_preemption_called']:
                    raise ValueError('destructive preemption was executed')
                expected_status = 'CAPACITY_BOUNDARY_STOP' if boundary else raw['status']
                status = expected_status if terminal['status'] == expected_status else 'INCOMPLETE'
                if status == 'COMPLETE' and not (identity['all_requests_completed'] and
                        metrics['n_completed'] == config['requests']):
                    status = 'INCOMPLETE'
                row.update(status=status, terminal_status=terminal, raw_status=raw['status'],
                    identity_check=identity,
                    full_episode_comparison_eligible=status=='COMPLETE' and not boundary and identity['all_requests_completed'],
                    strict_nonpreemptive_status=raw['strict_nonpreemptive_status'],
                    max_scheduled_active=max((s['actual_active'] for s in raw['scheduler_steps']), default=0),
                    max_decode_active=max((s['decode_requests'] for s in raw['scheduler_steps']), default=0),
                    decode_width_step_counts=dict(sorted(Counter(s['decode_requests'] for s in raw['scheduler_steps']).items())),
                    active_step_counts=dict(sorted(Counter(s['actual_active'] for s in raw['scheduler_steps']).items())),
                    scheduler_step_count=len(raw['scheduler_steps']),
                    waiting_positive_steps=sum(s['waiting_requests']>0 for s in raw['scheduler_steps']),
                    max_waiting=max((s['waiting_requests'] for s in raw['scheduler_steps']), default=0),
                    pool_ranges={k:[min(p[k] for p in pools), max(p[k] for p in pools)] for k in pools[0]} if pools else {},
                    max_used_fraction=max((p['used_blocks']/p['usable_blocks'] for p in pools), default=None),
                    minimum_free_blocks=min((p['free_blocks'] for p in pools), default=None),
                    allocation_failures=sum(len(t['allocation_failures']) for t in traces),
                    decode_not_scheduled=sum(len(t['existing_decode_not_scheduled_ids']) for t in traces),
                    capacity_boundary=boundary, metrics=metrics,
                    boundary_location=locate_boundary(raw),
                    request_metric_scope='COMPLETE_EPISODE' if status=='COMPLETE' else 'TRUNCATED_OBSERVATIONS_NOT_COMPARABLE',
                    memory_before=read(directory/'memory-before.json'), memory_after=read(directory/'memory-after.json'))
    if engine_args and any(a != engine_args[0] for a in engine_args):
        raise ValueError('engine arguments differ across cells')
    complete = all(r['status'] in ['COMPLETE', 'CAPACITY_BOUNDARY_STOP'] for r in rows)
    report = dict(status='MEASUREMENT_ONLY' if complete else 'INCOMPLETE_OR_UNRUN', cells=rows,
        engine_args_equal=bool(engine_args) and all(a==engine_args[0] for a in engine_args),
        engine_args=engine_args[0] if engine_args else None,
        execution_verification=verify_execution(rows, args.results_dir),
        limits=['Truncated boundary-cell rates are not full-horizon policy throughput.',
            'KV occupancy is within the physical KV pool, not additive to it.',
            'Resident expert parameters do not establish expert transfer or reclaim headroom.',
            'Reference SLO is secondary; no controller, task quality or paper claim.'])
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir/'analysis.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    (args.output_dir/'report.md').write_text(readable(report))


if __name__ == '__main__':
    main()
