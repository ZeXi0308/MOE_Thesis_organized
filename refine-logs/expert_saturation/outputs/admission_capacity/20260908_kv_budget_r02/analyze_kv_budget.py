#!/usr/bin/env python3
"""Fixed cap32 budget intervention; reuse native request/recomputation analysis."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
PREVIOUS=HERE.parent/'20260908_native_preemption_r01/analyze_native_preemption.py'
spec=importlib.util.spec_from_file_location('native_preemption_helpers',PREVIOUS)
native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
base,read,require=native.base,native.read,native.require
LABELS=['repeat0-budget95','repeat0-budget90','repeat1-budget90','repeat1-budget95']


def inspect(directory,label):
    repeat,arm=label.split('-');budget=0.95 if arm=='budget95' else 0.90
    out=dict(label=label,repeat=int(repeat[-1]),arm=arm,budget=budget,status='MISSING',
             measurement_executed=False,scientific_status='UNRUN',
             full_episode_comparison_eligible=False,raw_path=str(directory/'raw.json'))
    terminal=read(directory/'status.json') if (directory/'status.json').exists() else None
    q=read(directory/'safe-cap-qualification.json') if (directory/'safe-cap-qualification.json').exists() else None
    out.update(terminal_status=terminal,qualification=q)
    if (directory/'memory-after-init.json').exists():
        out['memory_after_init']=read(directory/'memory-after-init.json')
    if not (directory/'raw.json').exists():
        out['status']='UNRUN' if terminal and terminal['status']=='UNRUN' else 'INCOMPLETE' if terminal else 'MISSING'
        return out
    try:
        raw,config,engine=[read(directory/name) for name in ['raw.json','config.json','engine_args.json']]
        out.update(raw_status=raw['status'],measurement_executed=True,scientific_status='UNVERIFIED')
        require(config['requested_arm']==arm and config['cap']==raw['target_cap']==32,'arm/cap mismatch')
        require(config['gpu_memory_utilization']==engine['gpu_memory_utilization']==q['gpu_memory_utilization']==budget,'budget mismatch')
        require(raw['preemption_mode']==config['preemption_mode']=='native_recompute','native preemption mode changed')
        require(q['status']=='QUALIFIED','measurement ran without live qualification')
        reserve=(config['prompt_tokens']+config['output_tokens']+q['block_size']-1)//q['block_size']
        required=32*reserve;sufficient=q['usable_blocks']>=required
        require(q['per_request_reserved_blocks']==reserve and q['full_cap_reserved_blocks']==required and
                q['full_reservation_sufficient']==sufficient and q['full_reservation_required']==(budget==0.95),'reservation accounting mismatch')
        require(budget==0.90 or sufficient,'budget95 insufficient for full reservation')
        identity=base.validate_identity(raw,config,HERE/'inputs_preparation/prepared','long',32)
        metrics=base.summarize_episode_requests(raw['requests'],observation_end_s=raw['observation_end_s'],ttft_slo_s=5.0,tpot_slo_s=0.2)
        saved=read(directory/'metrics.json');require(all(saved.get(k)==v for k,v in metrics.items()),'saved host metrics mismatch')
        work,calls=native.execution_accounting(raw);events=native.preemptions(raw,calls)
        effects=native.request_effects(raw,events,work)
        boundary_pools=[t[k]['pool'] for t in raw['memory_trace'] for k in ['before','after'] if t[k] is not None]
        pools=boundary_pools+[e[k] for e in raw['preemption_events'] for k in ['pool','pool_after'] if k in e]
        pools += [a[k] for t in raw['memory_trace'] for a in t['allocation_failures'] for k in ['before','after']]
        require(all(0<=p['used_blocks']<=p['usable_blocks']==q['usable_blocks'] and p['used_blocks']+p['free_blocks']==p['usable_blocks'] for p in pools),'pool accounting mismatch')
        status=raw['status'] if terminal['status']==raw['status'] else 'INCOMPLETE'
        if status=='COMPLETE' and not (identity['all_requests_completed'] and metrics['n_completed']==32):status='INCOMPLETE'
        before,after=read(directory/'memory-before.json'),read(directory/'memory-after.json')
        out.update(status=status,scientific_status='MEASUREMENT_ONLY' if status=='COMPLETE' else 'INCOMPLETE',
            identity_check=identity,full_episode_comparison_eligible=status=='COMPLETE',
            metrics=metrics,work=work,preemption_events=events,actual_preemption_count=raw['actual_preemption_count'],effects=effects,
            max_scheduled_active=max(s['actual_active'] for s in raw['scheduler_steps']),max_decode_active=max(s['decode_requests'] for s in raw['scheduler_steps']),
            max_waiting=max(s['waiting_requests'] for s in raw['scheduler_steps']),
            decode_width_step_counts=dict(sorted(Counter(s['decode_requests'] for s in raw['scheduler_steps']).items())),
            allocation_failures=sum(len(t['allocation_failures']) for t in raw['memory_trace']),
            scheduler_boundary_pool_ranges={k:[min(p[k] for p in boundary_pools),max(p[k] for p in boundary_pools)] for k in boundary_pools[0]},
            all_observed_pool_ranges={k:[min(p[k] for p in pools),max(p[k] for p in pools)] for k in pools[0]},
            pool_sampling_scope='All observations include scheduler boundaries, allocation failures, and preemption before/after.',
            max_used_fraction=max(p['used_blocks']/p['usable_blocks'] for p in pools),minimum_free_blocks=min(p['free_blocks'] for p in pools),
            physical_kv_storage_bytes=before['kv_storage_bytes'],usable_blocks=q['usable_blocks'],full_cap_reserved_blocks=required,
            full_reservation_sufficient=sufficient,memory_before=before,memory_after=after,engine_args=engine)
    except (KeyError,ValueError,TypeError,OSError) as exc:
        out.update(status='INVALID',scientific_status='INVALID',error=str(exc),full_episode_comparison_eligible=False)
    return out


def comparisons(rows):
    pairs=[]
    for repeat in [0,1]:
        low=next(r for r in rows if r['repeat']==repeat and r['arm']=='budget90')
        high=next(r for r in rows if r['repeat']==repeat and r['arm']=='budget95')
        p=dict(repeat=repeat,base=low['label'],action=high['label'],status='UNRUN_OR_UNQUALIFIED')
        if low['full_episode_comparison_eligible'] and high['full_episode_comparison_eligible']:
            a,b=low['metrics'],high['metrics']
            p.update(status='DESCRIPTIVE_FULL_EPISODE_BUDGET_INTERVENTION',
                physical_kv_added_bytes=high['physical_kv_storage_bytes']-low['physical_kv_storage_bytes'],
                usable_blocks_added=high['usable_blocks']-low['usable_blocks'],
                throughput_relative_change=b['throughput_rps']/a['throughput_rps']-1,
                duration_relative_change=b['observation_duration_s']/a['observation_duration_s']-1,
                budget95_minus_budget90_s={f'{metric}_{q}':b['latency_s'][metric][q]-a['latency_s'][metric][q]
                                          for metric in ['ttft','tpot','itl'] for q in ['p50','p99']},
                request_max_itl_p99_change_s=high['effects']['request_max_itl_s']['p99']-low['effects']['request_max_itl_s']['p99'],
                completion_p99_change_s=high['effects']['completion_latency_s']['p99']-low['effects']['completion_latency_s']['p99'])
        pairs.append(p)
    return pairs


def readable(result):
    lines=['# Native cap32: fixed KV budget intervention','',result['status'],'',
        'Budget95 receives more GPU memory than budget90. This is a configuration intervention, not a same-budget scheduling gain.','',
        '| Cell | Runtime / scientific status | Complete | KV GiB / usable blocks | Full reservation enough | Active/decode/wait max | Duration s | Throughput req/s | Preemptions / recomputed tokens |',
        '|---|---|---:|---|---|---|---:|---:|---|']
    for r in result['cells']:
        if not r['full_episode_comparison_eligible']:
            lines.append(f"| {r['label']} | {r['status']} / {r['scientific_status']} | — | — | — | — | — | — | — |");continue
        m=r['metrics']
        lines.append(f"| {r['label']} | {r['status']} | {m['n_completed']}/32 | {r['physical_kv_storage_bytes']/2**30:.5f} / {r['usable_blocks']} | "
            f"{r['full_reservation_sufficient']} | {r['max_scheduled_active']}/{r['max_decode_active']}/{r['max_waiting']} | "
            f"{m['observation_duration_s']:.5f} | {m['throughput_rps']:.5f} | {r['actual_preemption_count']} / {r['work']['totals']['recomputed_tokens']} |")
    lines+=['','## Complete-request latency (32 requests per cell; descriptive tails)','',
        '| Cell | TTFT p50 / p99 s | TPOT p50 / p99 ms | Completion p99 s | Pooled ITL p99 ms | Request-max-ITL p99 s | Maximum ITL s |',
        '|---|---|---|---:|---:|---:|---:|']
    for r in result['cells']:
        if not r['full_episode_comparison_eligible']:continue
        t=r['metrics']['latency_s'];e=r['effects']
        lines.append(f"| {r['label']} | {t['ttft']['p50']:.5f} / {t['ttft']['p99']:.5f} | {t['tpot']['p50']*1000:.5f} / {t['tpot']['p99']*1000:.5f} | "
            f"{e['completion_latency_s']['p99']:.5f} | {t['itl']['p99']*1000:.5f} | {e['request_max_itl_s']['p99']:.5f} | {e['longest_itl_requests'][0]['longest_itl']['itl_s']:.5f} |")
    lines+=['','## Frozen within-repeat comparisons','']
    lines += ['- '+json.dumps(p) for p in result['comparisons']]
    v=result['execution_verification']
    sources=all(v['per_cell_source_match'].values()) if v['per_cell_source_match'] else 'UNRUN: no execution environment captured'
    software=all(v['software_fields_equal'].values()) if v['environments'] else 'UNRUN'
    lines+=['','## Qualification and scope','',
        f"Measured cells: {result['measured_cells']}. Runtime initialization failure is separate from scientific UNRUN; absent measurement has no throughput value.",
        f"Only gpu_memory_utilization differs in engine arguments: {result['engine_args_equal_except_budget']}. "
        f"Five executed source hashes match: {sources}; software equal: {software}; warmups: {v['warmup_count']}.",
        'The live full-reservation threshold is a sufficient condition for capacity, not a necessary condition for native completion or zero preemption.',
        'All request metrics include waiting, recomputation and capture overhead. Recomputed tokens come from executed interval overlap, not output receipts.',
        'KV occupancy is within physical KV storage; expert/parameter and allocated/reserved memory are not additive. More KV is not evidence of expert reclaim.',
        'Reference 5 s / 200 ms SLO is secondary. No quality, second-model, EP, production-SLO or same-budget method claim.','']
    return '\n'.join(lines)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--results-dir',type=Path,default=HERE/'gpu_results')
    parser.add_argument('--output-dir',type=Path,default=HERE/'analysis');parser.add_argument('--execution-archive',type=Path,default=HERE/'execution.tar.gz')
    args=parser.parse_args();require(not args.output_dir.exists(),'output directory must be new')
    rows=[inspect(args.results_dir/label,label) for label in LABELS]
    engines=[{k:v for k,v in r['engine_args'].items() if k!='gpu_memory_utilization'} for r in rows if 'engine_args' in r]
    equal=all(e==engines[0] for e in engines) if engines else None;require(equal or not engines,'engine arguments differ beyond memory budget')
    input_checks={domain:read(HERE/'inputs_preparation/prepared'/domain/'config.json')['workload_sha256']==
        read(PREVIOUS.parent/'inputs_preparation/prepared'/domain/'config.json')['workload_sha256'] for domain in ['short','long']}
    require(all(input_checks.values()),'workload differs from inherited native-preemption experiment')
    measured=sum(r['measurement_executed'] for r in rows)
    result=dict(status='UNRUN' if measured==0 else 'MEASUREMENT_ONLY' if all(r['full_episode_comparison_eligible'] for r in rows) else 'INCOMPLETE_OR_UNRUN',
        measured_cells=measured,
        cells=rows,comparisons=comparisons(rows),engine_args_equal_except_budget=equal,inherited_workload_sha_equal=input_checks,
        execution_verification=native.safe.verification(rows,args.results_dir,args.execution_archive),
        dependencies=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in
                      [Path(__file__).resolve(),PREVIOUS,native.SAFE_PATH,native.safe.PREVIOUS,Path(sys.modules['metrics'].__file__)]])
    args.output_dir.mkdir(parents=True,exist_ok=False)
    (args.output_dir/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (args.output_dir/'report.md').write_text(readable(result))
    print(json.dumps(dict(status=result['status'],comparisons=result['comparisons']),indent=2))


if __name__=='__main__':main()
