#!/usr/bin/env python3
"""Actual context-calibration cells, request costs and causal victim replay."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import re
import statistics
import sys


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def work_accounting(library, raw, requests):
    source = inspect.getsource(library.base.work_accounting)
    require(source.count('3072') == 4 and source.count('4096') == 1, 'work-accounting source layout changed')
    source = source.replace('3072', "requests[rid]['prompt_tokens']")
    source = source.replace('4096', "(requests[rid]['prompt_tokens'] + 1024)")
    ns = dict(library.base.work_accounting.__globals__)
    exec(compile(source, __file__+':context-work-accounting', 'exec'), ns)
    result = ns['work_accounting'](raw, requests)
    require(result['totals']['fresh_prefill']==sum(r['prompt_tokens'] for r in requests.values())
        and result['totals']['fresh_decode']==len(requests)*1023, 'complete no-APC coverage differs')
    return result


def retain_no_action_calibration(library):
    source = inspect.getsource(library.headroom.preemption_accounting)
    gate = "    require(policy != 'rotate' or counts['forced'] > 0, 'rotation performed no forced action')"
    require(source.count(gate)==1, 'exposure gate source changed')
    source = source.replace(gate, '    # Exposure is measured, not an eligibility gate for this calibration.')
    ns = dict(library.headroom.preemption_accounting.__globals__)
    exec(compile(source, __file__+':context-preemption-accounting', 'exec'), ns)
    library.headroom.preemption_accounting = ns['preemption_accounting']


def rotation_replay(library, order):
    """Keep qualified d6 checks; change only order, per-request cap and exposure."""
    source = inspect.getsource(library.rotation.replay)
    lines = [line for line in source.splitlines() if not any(s in line for s in
        ("q=d['pre_exchange_ownership']", "q['live_owned_blocks']", "qualification_receipts+=1"))]
    source = '\n'.join(lines)
    source = source.replace("victim_order='most_output'", "victim_order=order")
    source = source.replace("d['effective_victim_order']=='most_output'", "d['effective_victim_order']==order")
    source = source.replace("states[r]['prompt_tokens'],4096,states[r]['output_tokens']",
        "states[r]['prompt_tokens'],states[r]['prompt_tokens']+1024,states[r]['output_tokens']")
    source = source.replace("    require(forced>0, 'INVALID_NO_ACTION')", "    # No-action calibration is retained.")
    ns = dict(library.rotation.replay.__globals__, order=order, check_release=library.old_release)
    exec(compile(source, __file__+':context-replay', 'exec'), ns)
    return ns['replay']


def inspect_cell(bundle, spec, library, metadata, source, inputs, input_cfg, metrics, selector):
    label, variant = spec['label'], spec['variant']
    folder = bundle/'execution/readback/results'/label
    result = dict(**spec, status='UNRUN', eligible=False)
    if not (folder/'raw.json').exists():
        if folder.exists():
            result.update(status='INCOMPLETE', terminal=read(folder/'status.json') if (folder/'status.json').exists() else {})
        return result
    try:
        raw, cfg, env, engine, q, saved, decisions, status, memory = [read(folder/n) for n in
            ('raw.json', 'config.json', 'environment.json', 'engine_args.json', 'safe-cap-qualification.json',
             'metrics.json', 'headroom-decisions.json', 'status.json', 'memory-before.json')]
        result.update(raw_sha256=sha(folder/'raw.json'), raw_path=str(folder/'raw.json'),
            retained_requests=[dict(id=r['request_id'], status=r['status'], outputs=len(r['output_token_ids'])) for r in raw['requests']])
        require(all(sha(source/n)==h==metadata['files_sha256'][n] for n,h in env['source_sha256'].items()), 'executed source differs')
        require(all(env['vllm_source_sha256'][n]==h for n,h in metadata['expected_runtime_sources'].items()), 'pinned native runtime differs')
        require(all(cfg[k]==v for k,v in input_cfg.items()), 'input configuration differs')
        require(cfg['variant']==variant and cfg['cap']==raw['target_cap']==32 and cfg['rotation_victim_order']==('least_progress' if variant=='native' else variant), 'arm/cap differs')
        require(cfg['rotation_config']==library.rotation.ROTATION and cfg['completion_policy']==('native' if variant=='native' else 'rotate'), 'rotation policy differs')
        require(q['status']=='QUALIFIED' and q['usable_blocks']==6656 and q['block_size']==16 and q['summed_full_length_blocks']==7680
            and not q['prefix_caching'] and q['observed_scheduler_reserve_full_isl'] is True and memory['kv_storage_bytes']==13960740864, 'actual KV qualification differs')
        require(engine['kv_cache_memory_bytes']==13960740864 and engine['max_model_len']==4096 and engine['max_num_seqs']==32
            and engine['max_num_batched_tokens']==1024 and not engine['enable_prefix_caching'] and not engine['async_scheduling'], 'engine budget/backend differs')
        require(raw['status']==status['status']=='COMPLETE' and raw['error'] is None and raw['capacity_boundary'] is None, 'episode incomplete')
        requests = {r['request_id']:r for r in raw['requests']}
        require(len(requests)==len(raw['requests'])==32, 'missing/duplicate requests')
        for row, ids, arrival in zip(inputs['source_requests'], inputs['actual_prompt_token_ids'], inputs['arrival_traces_s']['steady']):
            r = requests[row['request_id']]
            require(r['prompt_tokens']==len(ids)==row['prompt_token_count'] and r['prompt_token_ids_sha256']==row['prompt_token_ids_sha256']
                and r['document_id']==row['document_id'] and r['arrival_s']==arrival and r['status']=='completed'
                and len(r['output_token_ids'])==len(r['token_times_s'])==1024 and r['stop_reason']=='length'
                and raw['internal_to_source'][r['internal_request_id']]==r['request_id'], 'request input/output identity differs')
            require(r['arrival_s']<=r['admission_s']<=r['engine_add_return_s']<=r['token_times_s'][0]
                and r['completion_s']==r['token_times_s'][-1], 'request timing order differs')
        for i,n in enumerate((32,32,2)):
            warm=read(folder/f'warmup-{i}.json')
            require(warm['status']=='COMPLETE' and len(warm['requests'])==n and all(r['status']=='completed' and len(r['output_token_ids'])==16 for r in warm['requests']), 'common warmup missing')
        computed=metrics.summarize_episode_requests(raw['requests'], observation_end_s=raw['observation_end_s'], ttft_slo_s=5., tpot_slo_s=.2)
        require(all(saved.get(k)==v for k,v in computed.items()), 'saved request metrics differ')
        work=work_accounting(library, raw, requests)
        library.rotation_replay=rotation_replay(library, variant)
        action=library.native_accounting(raw, decisions, dict(adapter='native' if variant=='native' else 'rotation',
            completion_policy=cfg['completion_policy']), selector, metrics)
        pool=[m[k]['pool'] for m in raw['memory_trace'] for k in ('before','after')]
        per=[]
        for r in raw['requests']:
            times=r['token_times_s']
            per.append(dict(request_id=r['request_id'],prompt_tokens=r['prompt_tokens'],
                ttft_s=times[0]-r['arrival_s'],tpot_s=(times[-1]-times[0])/(len(times)-1),
                max_itl_s=max(b-a for a,b in zip(times,times[1:])),completion_s=r['completion_s']-r['arrival_s']))
        engine_s=sum(c['returned_s']-c['start_s'] for c in raw['engine_steps'])
        scheduler_s=sum(s['end_s']-s['start_s'] for s in raw['scheduler_steps'])
        wall=computed['observation_duration_s']
        require(0<=scheduler_s<=engine_s<=wall+1e-7, 'overlapping cost buckets')
        result.update(status='COMPLETE', eligible=True, metrics=computed, wall_s=wall,
            throughput_rps=computed['throughput_rps'], mean_completion_s=statistics.mean(r['completion_s'] for r in per),
            max_itl_s=max(r['max_itl_s'] for r in per), per_request=per, work=work, action_checks=action,
            pressure=dict(peak_used_blocks=max(p['used_blocks'] for p in pool),
                min_free_blocks=min(p['free_blocks'] for p in pool),
                full_pool_snapshots=sum(p['free_blocks']==0 for p in pool),snapshot_count=len(pool),
                actual_preemptions=len(raw['preemption_events']),
                forced_exchanges=sum(len(d.get('forced_preempted', [])) for d in decisions),
                policy_active_steps=sum(bool(d['active']) for d in decisions),
                rejected_unfunded_proposals=sum(d.get('not_applied_reason')=='victim cannot fund complete recovery history' for d in decisions)),
            timing=dict(scheduler_inclusive_s=scheduler_s, engine_non_scheduler_s=engine_s-scheduler_s,
                outside_engine_s=wall-engine_s),
            engine=engine, software={k:env[k] for k in ('python','torch','cuda','vllm','transformers')},
            runtime_sources=env['vllm_source_sha256'], gpu_uuid=re.search(r'GPU-[\w-]+',env['gpu_before']['device']).group(),
            outputs={r['request_id']:r['output_token_ids'] for r in raw['requests']})
    except (OSError, KeyError, ValueError, TypeError, IndexError, AttributeError) as exc:
        result.update(status='INVALID_OR_INCOMPLETE', eligible=False, error=str(exc))
    return result


def compare(a,b):
    for k in ('engine', 'software', 'runtime_sources', 'gpu_uuid'):
        require(a[k]==b[k], 'pair environment differs: '+k)
    old={r['request_id']:r for r in a['per_request']}
    per=[dict(request_id=r['request_id'], **{k:r[k]-old[r['request_id']][k] for k in
        ('ttft_s','tpot_s','max_itl_s','completion_s')}) for r in b['per_request']]
    return dict(baseline=a['label'], action=b['label'],
        throughput_delta_pct=100*(b['throughput_rps']/a['throughput_rps']-1),
        max_itl_delta_s=b['max_itl_s']-a['max_itl_s'],
        mean_completion_delta_pct=100*(b['mean_completion_s']/a['mean_completion_s']-1),
        per_request_deltas=per, same_output_sequences=sum(a['outputs'][k]==v for k,v in b['outputs'].items()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--analysis-library', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args=p.parse_args()
    require(not args.output.exists(), 'preserve earlier analysis')
    sys.dont_write_bytecode=True
    sys.path.insert(0,str(args.analysis_library))
    import analyze_recovery_holdout_comparison as library
    retain_no_action_calibration(library)
    bundle=args.bundle; source=bundle/'preparation/pkg'
    metadata=read(bundle/'preparation/preparation.json')
    require(all(sha(source/n)==h for n,h in metadata['files_sha256'].items()), 'prepared package changed')
    if (bundle/'execution/readback/pkg').exists():
        require(all(sha(bundle/'execution/readback/pkg'/n)==h for n,h in metadata['files_sha256'].items()), 'readback package changed')
    inputs=read(source/'inputs_preparation/prepared/heterogeneous/workload.json')
    cfg=read(source/'inputs_preparation/prepared/heterogeneous/config.json')
    require(hashlib.sha256(json.dumps(inputs,sort_keys=True).encode()).hexdigest()==cfg['workload_sha256'], 'input hash differs')
    metrics=library.module('context_metrics',source/'metrics.py')
    selector=library.module('context_selector',source/'absence_rotation.py')
    cells=[inspect_cell(bundle,s,library,metadata,source,inputs,cfg,metrics,selector) for s in read(source/'campaign.json')['cells']]
    pairs=[]
    if all(c['eligible'] for c in cells):
        for block in (0,1):
            rows={c['variant']:c for c in cells if c['block']==block}
            pairs += [compare(rows[a],rows[b]) for a,b in [('native','most_output'),('native','least_progress'),('most_output','least_progress')]]
        pairs += [compare(cells[i],cells[5-i]) for i in range(3)]
    result=dict(status='MEASUREMENT_ONLY' if pairs else ('UNRUN' if all(c['status']=='UNRUN' for c in cells) else 'INCOMPLETE'),
        scope='Exploratory reused documents, one actual resource pool, two reversed blocks. No matched pressure to prior homogeneous run, quality, independence, noise bound, optimality or method GO.',
        producer_sha256=sha(Path(__file__)), reused_analysis={str(Path(m.__file__)):sha(Path(m.__file__)) for m in
            (library,library.rotation,library.token,library.base,library.headroom)}, cells=cells, comparisons=pairs)
    for c in cells:
        c.pop('outputs',None)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:
        f.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'], cells=[dict(label=c['label'],status=c['status'],error=c.get('error')) for c in cells],comparisons=len(pairs))))


if __name__=='__main__':
    main()
