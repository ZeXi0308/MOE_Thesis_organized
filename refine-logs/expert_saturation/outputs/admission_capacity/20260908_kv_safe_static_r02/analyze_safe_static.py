#!/usr/bin/env python3
"""Four retained safe-static cells; reuse prior identity, metrics and boundary checks."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tarfile

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / '20260906_native_memory_pressure_r01/analyze_probe.py'
spec = importlib.util.spec_from_file_location('previous_capacity_analysis', PREVIOUS)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
read = base.read
LABELS = ['repeat0-baseline16', 'repeat0-safe', 'repeat1-safe', 'repeat1-baseline16']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def inspect(directory, label):
    repeat, arm = label.split('-')
    row = dict(label=label, repeat=int(repeat[-1]), arm=arm, domain='long', status='MISSING',
               full_episode_comparison_eligible=False, raw_path=str(directory/'raw.json'))
    terminal = read(directory/'status.json') if (directory/'status.json').exists() else None
    qualification = read(directory/'safe-cap-qualification.json') if (directory/'safe-cap-qualification.json').exists() else None
    row.update(terminal_status=terminal, qualification=qualification)
    if not (directory/'raw.json').exists():
        row['status'] = 'UNRUN' if terminal and terminal['status']=='UNRUN' else 'INCOMPLETE' if terminal else 'MISSING'
        return row
    required = ['config.json', 'status.json', 'safe-cap-qualification.json', 'metrics.json',
                'engine_args.json', 'environment.json', 'memory-before.json', 'memory-after.json', 'gpu-after.json']
    missing = [name for name in required if not (directory/name).exists()]
    if missing:
        row.update(status='INCOMPLETE', missing_artifacts=missing)
        return row
    try:
        config, raw = read(directory/'config.json'), read(directory/'raw.json')
        row.update(raw_status=raw['status'], cap=config['cap'])
        q = qualification
        require(q['status']=='QUALIFIED', 'measurement ran without qualified layout')
        reserved = (config['prompt_tokens']+config['output_tokens']+q['block_size']-1)//q['block_size']
        derived = min(q['engine_max_num_seqs'], q['usable_blocks']//reserved)
        require(q['safe_cap']==derived and q['per_request_reserved_blocks']==reserved and derived>16,
                'live safe-cap formula differs from recorded qualification')
        require(config['requested_arm']==arm and config['cap']==(derived if arm=='safe' else 16),
                'requested arm or applied cap differs from qualification')
        require(q['total_blocks']-1==q['usable_blocks']==q['free_blocks'], 'initial pool accounting differs')
        identity = base.validate_identity(raw, config, HERE/'inputs_preparation/prepared', 'long', config['cap'])
        metrics = base.summarize_episode_requests(raw['requests'], observation_end_s=raw['observation_end_s'],
                                               ttft_slo_s=5.0, tpot_slo_s=0.2)
        saved=read(directory/'metrics.json')
        require(all(saved.get(k)==v for k,v in metrics.items()), 'saved metrics differ from raw recomputation')
        traces=raw['memory_trace']
        pools=[t[k]['pool'] for t in traces for k in ['before','after'] if t[k] is not None]
        for p in pools:
            require(0<=p['used_blocks']<=p['usable_blocks'] and p['used_blocks']+p['free_blocks']==p['usable_blocks'],
                    'invalid measured pool accounting')
            require(p['usable_blocks']==q['usable_blocks'], 'measured pool differs from qualification')
        failed=[t for t in traces if not t['schedule_completed']]
        require(all(t['model_execution_confirmed'] is False for t in failed), 'failed schedule claimed model execution')
        boundary=raw['capacity_boundary']
        require(not boundary or not boundary['original_preemption_called'], 'destructive preemption executed')
        expected='CAPACITY_BOUNDARY_STOP' if boundary else raw['status']
        status=expected if terminal['status']==expected else 'INCOMPLETE'
        if status=='COMPLETE' and not (identity['all_requests_completed'] and metrics['n_completed']==config['requests']):
            status='INCOMPLETE'
        eligible=status=='COMPLETE' and not boundary and identity['all_requests_completed']
        row.update(status=status, identity_check=identity, full_episode_comparison_eligible=eligible,
            qualification_formula_recomputed=True, strict_nonpreemptive_status=raw['strict_nonpreemptive_status'],
            derived_safe_cap=derived, per_request_reserved_blocks=reserved,
            max_scheduled_active=max((s['actual_active'] for s in raw['scheduler_steps']), default=0),
            max_decode_active=max((s['decode_requests'] for s in raw['scheduler_steps']), default=0),
            max_waiting=max((s['waiting_requests'] for s in raw['scheduler_steps']), default=0),
            decode_width_step_counts=dict(sorted(Counter(s['decode_requests'] for s in raw['scheduler_steps']).items())),
            scheduler_step_count=len(raw['scheduler_steps']), waiting_positive_steps=sum(s['waiting_requests']>0 for s in raw['scheduler_steps']),
            pool_ranges={k:[min(p[k] for p in pools),max(p[k] for p in pools)] for k in pools[0]} if pools else {},
            max_used_fraction=max((p['used_blocks']/p['usable_blocks'] for p in pools), default=None),
            minimum_free_blocks=min((p['free_blocks'] for p in pools),default=None),
            allocation_failures=sum(len(t['allocation_failures']) for t in traces),
            decode_not_scheduled=sum(len(t['existing_decode_not_scheduled_ids']) for t in traces),
            capacity_boundary=boundary, boundary_location=base.locate_boundary(raw), metrics=metrics,
            request_metric_scope='COMPLETE_EPISODE' if eligible else 'TRUNCATED_OBSERVATIONS_NOT_COMPARABLE',
            memory_before=read(directory/'memory-before.json'),memory_after=read(directory/'memory-after.json'),
            engine_args=read(directory/'engine_args.json'))
    except (KeyError, ValueError, TypeError) as exc:
        row.update(status='INVALID', error=str(exc), full_episode_comparison_eligible=False)
    return row


def verification(rows, root, archive):
    names=['run_probe.py','memory_telemetry.py','native_capture.py','metrics.py','safe_static.py']
    with tarfile.open(archive,'r:gz') as tar:
        archived={name:hashlib.sha256(tar.extractfile(name).read()).hexdigest() for name in names}
    env={r['label']:read(root/r['label']/'environment.json') for r in rows if (root/r['label']/'environment.json').exists()}
    first=next(iter(env.values()),{})
    fields=['python','torch','cuda','vllm','transformers','cpu_threads','vllm_source_sha256']
    warmups=[]
    for r in rows:
        for path in sorted((root/r['label']).glob('warmup-*.json')):
            raw=read(path)
            warmups.append(dict(cell=r['label'],path=str(path),status=raw['status'],requests=len(raw['requests']),
                all_completed=all(q['status']=='completed' and len(q['output_token_ids'])==16 for q in raw['requests'])))
    return dict(archive=str(archive),archive_source_sha256=archived,environments=env,
        per_cell_source_match={label:e['source_sha256']==archived for label,e in env.items()},
        software_fields_equal={k:all(e.get(k)==first.get(k) for e in env.values()) for k in fields},
        analysis_metrics_matches_execution=hashlib.sha256((HERE/'metrics.py').read_bytes()).hexdigest()==archived['metrics.py'],
        warmups=warmups,warmup_count=len(warmups))


def compare(rows):
    pairs=[]
    for repeat in [0,1]:
        a=next(r for r in rows if r['repeat']==repeat and r['arm']=='baseline16')
        b=next(r for r in rows if r['repeat']==repeat and r['arm']=='safe')
        pair=dict(repeat=repeat,baseline=a['label'],safe=b['label'],safe_cap=b.get('cap'),status='UNRUN_OR_UNQUALIFIED')
        if a['full_episode_comparison_eligible'] and b['full_episode_comparison_eligible']:
            x,y=a['metrics'],b['metrics']
            pair.update(status='DESCRIPTIVE_COMPLETE_EPISODE_COMPARISON',
                throughput_relative_change=y['throughput_rps']/x['throughput_rps']-1,
                duration_relative_change=y['observation_duration_s']/x['observation_duration_s']-1,
                ttft_p50_change_s=y['latency_s']['ttft']['p50']-x['latency_s']['ttft']['p50'],
                ttft_p99_change_s=y['latency_s']['ttft']['p99']-x['latency_s']['ttft']['p99'],
                tpot_p50_change_ms=1000*(y['latency_s']['tpot']['p50']-x['latency_s']['tpot']['p50']),
                tpot_p99_change_ms=1000*(y['latency_s']['tpot']['p99']-x['latency_s']['tpot']['p99']),
                reference_slo_pass_change=y['n_slo_pass']-x['n_slo_pass'])
        pairs.append(pair)
    return pairs


def readable(result):
    lines=['# Safe static versus baseline16: retained analysis','',result['status'],'',
        'Four fixed cells, no post-hoc cap choice. Complete and truncated episodes are separated; reference SLO is secondary.','',
        '| Cell | Cap/derived safe | Status | Complete | Active/decode/wait max | Peak used/usable | Full duration s | Throughput req/s | TTFT p50 s | TPOT p50 ms |',
        '|---|---|---|---:|---|---|---:|---:|---:|---:|']
    for r in result['cells']:
        if 'metrics' not in r:
            lines.append(f"| {r['label']} | — | {r['status']} | — | — | — | — | — | — | — |")
            continue
        m=r['metrics'];p=r['pool_ranges']
        perf=(f"{m['observation_duration_s']:.5f} | {m['throughput_rps']:.5f} | {m['latency_s']['ttft']['p50']:.5f} | {m['latency_s']['tpot']['p50']*1000:.5f}"
              if r['full_episode_comparison_eligible'] else '— | — | — | —')
        lines.append(f"| {r['label']} | {r['cap']}/{r['derived_safe_cap']} | {r['status']} | {m['n_completed']}/32 | "
            f"{r['max_scheduled_active']}/{r['max_decode_active']}/{r['max_waiting']} | {p['used_blocks'][1]}/{p['usable_blocks'][0]} | {perf} |")
    lines+=['','## Complete-request tails (32 requests per cell; descriptive p99)','',
        '| Cell | TTFT p99 s | Per-request mean TPOT p99 ms | Peak KV occupied | Steps with waiting / all scheduled steps |',
        '|---|---:|---:|---:|---|']
    for r in result['cells']:
        if r['full_episode_comparison_eligible']:
            m=r['metrics']
            lines.append(f"| {r['label']} | {m['latency_s']['ttft']['p99']:.5f} | {m['latency_s']['tpot']['p99']*1000:.5f} | "
                f"{r['max_used_fraction']*100:.4f}% | {r['waiting_positive_steps']}/{r['scheduler_step_count']} |")
        else:
            lines.append(f"| {r['label']} | — | — | — | — |")
    lines+=['','## Fixed comparisons','']
    for p in result['comparisons']:
        lines.append('- '+json.dumps(p,ensure_ascii=False))
    v=result['execution_verification']
    lines+=['','## Qualification and limits','',
        f"Engine arguments equal: {result['engine_args_equal']}; execution sources including safe_static.py match: {all(v['per_cell_source_match'].values())}; "
        f"software equal: {all(v['software_fields_equal'].values())}; warmup raw count: {v['warmup_count']}.",
        'Each derived cap comes from its own engine pool. Different derived caps are repetitions of the formula, not the same numeric action.',
        'No unrun or boundary cell enters full-episode throughput comparison. No expert-reclaim, Oracle, task-quality or MoE-method claim.',
        'KV occupied blocks lie inside the physical KV region; parameter/expert, allocated/reserved quantities are not additive.',
        'The 32 repeated articles and four independent engine processes do not establish population-level statistics.','']
    return '\n'.join(lines)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--results-dir',type=Path,default=HERE/'gpu_results')
    parser.add_argument('--output-dir',type=Path,default=HERE/'analysis')
    parser.add_argument('--execution-archive',type=Path,default=HERE/'execution.tar.gz')
    args=parser.parse_args()
    require(not args.output_dir.exists(),'output directory must be new')
    rows=[inspect(args.results_dir/label,label) for label in LABELS]
    engines=[r['engine_args'] for r in rows if 'engine_args' in r]
    equal=bool(engines) and all(e==engines[0] for e in engines)
    require(equal or not engines,'engine arguments differ across retained cells')
    verified=verification(rows,args.results_dir,args.execution_archive)
    complete=all(r['status'] in ['COMPLETE','CAPACITY_BOUNDARY_STOP','UNRUN'] for r in rows)
    result=dict(status='MEASUREMENT_ONLY' if complete and any('metrics' in r for r in rows) else 'INCOMPLETE_OR_UNRUN',
        cells=rows,comparisons=compare(rows),engine_args_equal=equal,execution_verification=verified,
        dependencies=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                      for p in [Path(__file__).resolve(),PREVIOUS,Path(sys.modules['metrics'].__file__)]])
    args.output_dir.mkdir(parents=True,exist_ok=False)
    (args.output_dir/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (args.output_dir/'report.md').write_text(readable(result))
    print(json.dumps(dict(status=result['status'],cells=[{k:r.get(k) for k in ['label','status','cap']} for r in rows],comparisons=result['comparisons']),indent=2))


if __name__=='__main__':
    main()
