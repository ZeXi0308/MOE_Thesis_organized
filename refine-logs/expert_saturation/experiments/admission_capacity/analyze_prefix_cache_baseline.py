#!/usr/bin/env python3
"""Four native APC cells: retained host timing and executed-position accounting."""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys

KV, BLOCKS = 16089350144, 7671
WORKLOAD = 'feff45f7b6dd3cfe209f7af9ed47b82f31371aedbb5109f8054315fa73b2a75d'
REVISION = '6d84c48581ece794365f2b8e9cfb043c68ade9c5'
LABELS = ['cohort2-block0-apc_off', 'cohort2-block0-apc_on', 'cohort2-block1-apc_on', 'cohort2-block1-apc_off']
SCOPE = ('CPU preparation or descriptive native in-process measurements only; no SLO, Oracle, quality, '
         'significance or method GO. Cache computed jumps are observed skipped positions, not '
         'counterfactual savings. Free cached blocks can retain content. Per-event recovery spans overlap '
         'and must not be summed as wall time. Same-input repeats are correlated, not a noise bound.')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    with (gzip.open(path, 'rt') if path.suffix == '.gz' else path.open()) as f:
        return json.load(f)


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(',', ':')).encode()).hexdigest()


def work_accounting(raw, requests):
    steps, calls, memory = raw['scheduler_steps'], raw['engine_steps'], raw['memory_trace']
    require(len(memory) == len(steps), 'memory/scheduler alignment differs')
    receipts, outputs, times, finished = defaultdict(list), defaultdict(list), defaultdict(list), set()
    for event in raw['output_events']:
        rid, t, ids = event['request_id'], event['received_s'], event['cumulative_token_ids']
        require(rid in requests and rid not in finished and math.isfinite(t) and ids[:len(outputs[rid])] == outputs[rid], 'receipt identity/prefix differs')
        added = ids[len(outputs[rid]):]
        require(event['prefix_valid'] and added == event['new_token_ids'] and len(added) == event['chunk_size']
                and len(ids) == event['cumulative_tokens'] and len(added) <= 1, 'receipt count or unresolved token ITL')
        outputs[rid] = ids
        times[rid].extend([t] * len(added))
        receipts[t].append(event)
        require(event['external_request_id'] == requests[rid]['external_request_id'], 'receipt external identity differs')
        if event['finished']:
            require(len(ids) == 1024 and event['finish_reason'] == 'length', 'completion receipt differs')
            finished.add(rid)
    require(finished == set(requests), 'finished receipts missing')
    require(all(outputs[r] == q['output_token_ids'] and times[r] == q['token_times_s'] for r, q in requests.items()), 'receipt/request timeline differs')
    covered, high, totals, jump_rows, call_for_step = defaultdict(list), Counter(), Counter(), [], {}
    previous, last_return = 0, 0.
    for i, call in enumerate(calls):
        a, b = call['start_s'], call['returned_s']
        require(call['completed'] is True and call['call_index'] == i and math.isfinite(a) and math.isfinite(b)
                and last_return <= a <= b <= raw['observation_end_s'], 'failed/reversed/uncovered engine call')
        require(call['scheduler_step_start'] == previous and previous < call['scheduler_step_end'] <= len(steps), 'call/step identity differs')
        events = receipts.pop(b, [])
        require(sum(e['chunk_size'] for e in events) == call['new_output_tokens']
                and [e['external_request_id'] for e in events] == call['output_request_ids'], 'call receipts differ')
        require(len({e['request_id'] for e in events}) == len(events), 'duplicate request receipt in call')
        served = {x['request_id'] for s in steps[previous:call['scheduler_step_end']] for x in s['scheduled']}
        require(all(e['request_id'] in served for e in events), 'receipt has no scheduled work in call')
        for k in range(previous, call['scheduler_step_end']):
            s, m = steps[k], memory[k]
            require(s['step'] == m['attempted_step'] == k and a <= s['start_s'] <= s['end_s'] <= b, 'scheduler time alignment differs')
            require(s['total_scheduled_tokens'] == sum(x['scheduled_tokens'] for x in s['scheduled']), 'scheduled token sum differs')
            require(len({x['request_id'] for x in s['scheduled']}) == len(s['scheduled']), 'duplicate scheduled request')
            for x in s['scheduled']:
                rid, iid, lo, hi = x['request_id'], x['internal_request_id'], x['scheduled_start_computed'], x['computed_after']
                require(rid in requests and raw['internal_to_source'][iid] == rid and x['prompt_tokens'] == 3072, 'scheduled identity differs')
                require(0 <= lo < hi <= 4096 and hi-lo == x['scheduled_tokens'] and x['executed_high_water_before'] == high[rid]
                        and lo-x['computed_before'] == x['computed_adjustment'], 'scheduled positions differ')
                recorded = max(0, min(hi, high[rid])-lo)
                prefill = max(0, min(hi, 3072)-max(lo, high[rid]))
                require((recorded, prefill, hi-lo-recorded-prefill) == (x['recompute_tokens'], x['prefill_tokens'], x['decode_tokens']), 'capture high-water classification differs')
                repeated = sum(max(0, min(hi, v)-max(lo, u)) for u, v in covered[rid])
                fresh_prompt = max(0, min(hi, 3072)-lo)-sum(max(0, min(hi, 3072, v)-max(lo, u)) for u, v in covered[rid])
                totals.update(scheduled=hi-lo, repeated_executed=repeated, fresh_prefill=fresh_prompt,
                              fresh_decode=hi-lo-repeated-fresh_prompt, capture_high_water_recompute=recorded)
                union = []
                for u, v in sorted(covered[rid]+[[lo, hi]]):
                    if union and u <= union[-1][1]:
                        union[-1][1] = max(v, union[-1][1])
                    else:
                        union.append([u, v])
                covered[rid], high[rid] = union, max(high[rid], hi)
                if x['computed_adjustment'] > 0:
                    before = m['before']['requests'][iid]
                    jump_rows.append(dict(step=k, request_id=rid, tokens=x['computed_adjustment'],
                                          role='post_preemption' if before['num_preemptions'] else 'first_admission'))
            require(s['recompute_tokens'] == sum(x['recompute_tokens'] for x in s['scheduled']), 'step recompute total differs')
            call_for_step[k] = call
        previous, last_return = call['scheduler_step_end'], b
    require(previous == len(steps) and not receipts and totals['scheduled'] == totals['repeated_executed']+totals['fresh_prefill']+totals['fresh_decode'], 'execution/receipt conservation failed')
    pauses = []
    for event in raw['preemption_events']:
        rid = raw['internal_to_source'][event['victim_internal_request_id']]
        q, k, before, after = requests[rid], event['attempted_step'], event['victim_state'], event['victim_state_after']
        n = before['output_tokens']
        require(memory[k]['host_start_perf_counter_s'] <= event['host_perf_counter_s']
                <= event['returned_host_perf_counter_s'] <= memory[k]['host_end_perf_counter_s'], 'preemption absolute clock outside telemetry')
        require(event['original_preemption_called'] and event['original_preemption_returned'] and after['computed_tokens'] == 0
                and after['num_preemptions'] == before['num_preemptions']+1, 'native preemption not completed')
        require(event['output_token_ids_before'] == event['output_token_ids_after'] == q['output_token_ids'][:n], 'preemption changed output prefix')
        later = next((call_for_step[j] for j in range(k, len(steps)) if any(x['request_id'] == rid for x in steps[j]['scheduled'])), None)
        require(later is not None and n < len(q['token_times_s']), 'recovery or new output missing')
        anchor, new = q['token_times_s'][n-1] if n else q['arrival_s'], q['token_times_s'][n]
        boundary = min(new, call_for_step[k]['returned_s'])
        service = max(boundary, later['start_s'])
        require(anchor <= boundary <= service <= new, 'recovery gap order differs')
        pauses.append(dict(request_id=rid, preempt_step=k, outputs_before=n, first_service_step=later['scheduler_step_start'],
                           anchor_s=anchor, next_token_s=new, gap_s=new-anchor,
                           through_preempting_call_s=boundary-anchor, wait_after_preempting_call_s=service-boundary,
                           recovery_service_span_s=new-service))
    require(len(pauses) == raw['actual_preemption_count'], 'preemption count differs')
    return dict(totals=dict(totals), completed_calls=len(calls), pauses=pauses,
                cache=dict(successful_positive_adjustments=jump_rows,
                           first_admission_tokens=sum(x['tokens'] for x in jump_rows if x['role'] == 'first_admission'),
                           post_preemption_tokens=sum(x['tokens'] for x in jump_rows if x['role'] == 'post_preemption')))


def inspect(run, spec, source, workload, input_config, metrics):
    base = run/'gpu_results'/spec['label']
    row = dict(label=spec['label'], role=spec['role'], block=spec['block'], status='UNRUN', eligible=False)
    try:
        terminal = read(base/'status.json') if (base/'status.json').exists() else {}
        raw_path = next((p for p in (base/'raw.json', base/'raw.json.gz') if p.exists()), None)
        if raw_path is None:
            row.update(status='INCOMPLETE' if base.exists() and any(base.iterdir()) else 'UNRUN', terminal=terminal)
            return row
        raw = read(raw_path)
        row['retained_requests'] = [dict(request_id=r['request_id'], status=r['status'], outputs=len(r['output_token_ids'])) for r in raw['requests']]
        require((run/'frozen').is_dir(), 'measured run lacks frozen source')
        cfg, engine, q, memory, env, reset, saved = [read(base/n) for n in ('config.json','engine_args.json','safe-cap-qualification.json','memory-before.json','environment.json','prefix-cache-reset.json','metrics.json')]
        enabled = spec['role'] == 'apc_on'
        require(cfg['enable_prefix_caching'] is engine['enable_prefix_caching'] is q['prefix_caching'] is enabled, 'APC flag differs')
        require(cfg['completion_policy'] == 'native' and cfg['headroom_observer'] == 'not_installed' and cfg['rotation_config'] is None
                and not (base/'headroom-decisions.json').exists(), 'non-native policy installed')
        require(cfg['model'] == input_config['model'] and cfg['workload_sha256'] == WORKLOAD and cfg['prompt_tokens'] == 3072
                and cfg['output_tokens'] == 1024 and cfg['cap'] == raw['target_cap'] == 32, 'workload/model changed')
        require(engine['model'] == 'allenai/OLMoE-1B-7B-0924' and engine['revision'] == engine['tokenizer_revision'] == REVISION
                and engine['dtype'] == 'bfloat16' and engine['max_num_batched_tokens'] == 1024 and engine['max_num_seqs'] == 32
                and engine['scheduler_reserve_full_isl'] is True and engine['async_scheduling'] is False, 'engine scope changed')
        require(engine['max_model_len'] == 4096 and engine['seed'] == input_config['seed'] and engine['stream_interval'] == 1
                and engine['scheduling_policy'] == 'fcfs' and engine['enable_chunked_prefill'] is True
                and engine['enforce_eager'] is False and engine['enable_return_routed_experts'] is False, 'execution options changed')
        require(engine['kv_cache_memory_bytes'] == memory['kv_storage_bytes'] == KV and q['usable_blocks'] == BLOCKS
                and q['block_size'] == 16 and q['status'] == 'QUALIFIED' and q['observed_scheduler_reserve_full_isl'] is True, 'actual KV qualification differs')
        require(reset['returned'] is True and reset['after']['free_blocks'] == reset['after']['usable_blocks'] == BLOCKS
                and all(reset['after'][k] == 0 for k in ('nonzero_refcount_blocks','negative_refcount_blocks','hashed_blocks','pending_requests'))
                and reset['after']['counts'] == [0,0], 'measurement cache reset unqualified')
        require(set(env['source_sha256']) == {'run_probe.py','native_capture.py','memory_telemetry.py','metrics.py','safe_static.py'}, 'executed source inventory incomplete')
        require(env['vllm'] == '0.26.0' and all(sha(source/n) == h for n,h in env['source_sha256'].items()), 'executed source/runtime differs')
        for index, count in enumerate((32,32,2)):
            warm = read(base/f'warmup-{index}.json')
            require(warm['status'] == 'COMPLETE' and len(warm['requests']) == count
                    and all(r['status'] == 'completed' and len(r['output_token_ids']) == 16 for r in warm['requests']), 'common warmup incomplete')
        require(raw['status'] == terminal['status'] == 'COMPLETE' and not raw['error'] and raw['capacity_boundary'] is None
                and raw['preemption_mode'] == 'native_recompute', 'incomplete/failed measurement')
        requests = {r['request_id']:r for r in raw['requests']}
        require(len(requests) == len(raw['requests']) == len(workload['source_requests']) == 32, 'request identities/count differ')
        for expected, ids, arrival in zip(workload['source_requests'], workload['actual_prompt_token_ids'], workload['arrival_traces_s']['steady']):
            r = requests[expected['request_id']]
            require(r['document_id'] == expected['document_id'] and r['prompt_tokens'] == len(ids) == 3072
                    and r['prompt_token_ids_sha256'] == expected['prompt_token_ids_sha256'] == digest(ids)
                    and r['arrival_s'] == arrival and r['status'] == 'completed' and len(r['output_token_ids']) == 1024, 'request workload/completion changed')
            require(raw['internal_to_source'][r['internal_request_id']] == r['request_id'], 'internal mapping differs')
            require(r['arrival_s'] <= r['admission_s'] <= r['engine_add_return_s'] <= r['token_times_s'][0]
                    and r['completion_s'] == r['token_times_s'][-1] and r['stop_reason'] == 'length', 'submission/completion clock differs')
        recalculated = metrics.summarize_episode_requests(raw['requests'], observation_end_s=raw['observation_end_s'], ttft_slo_s=5., tpot_slo_s=.2)
        require(all(saved.get(k) == v for k,v in recalculated.items()), 'saved metrics differ from raw')
        work = work_accounting(raw, requests)
        cache_saved = read(base/'cache-accounting.json')
        require(all(cache_saved.get(k) == v for k,v in work['cache'].items()), 'saved cache jumps differ')
        pools = [m[k]['pool'] for m in raw['memory_trace'] for k in ('before','after')]
        require(pools and all(p['usable_blocks'] == BLOCKS and 0 <= p['used_blocks'] <= BLOCKS
                             and p['used_blocks']+p['free_blocks'] == BLOCKS for p in pools), 'global pool count invalid')
        per_request = [dict(request_id=r['request_id'], ttft_s=r['token_times_s'][0]-r['arrival_s'],
                            completion_s=r['completion_s']-r['arrival_s'], max_itl_s=max(b-a for a,b in zip(r['token_times_s'], r['token_times_s'][1:])),
                            output_sha256=digest(r['output_token_ids'])) for r in raw['requests']]
        row.update(status='COMPLETE', eligible=True, engine=engine, config=cfg, work=work, per_request=per_request,
                   wall_s=recalculated['observation_duration_s'], throughput_rps=recalculated['throughput_rps'],
                   mean_completion_s=sum(r['completion_s'] for r in per_request)/32,
                   ttft_s=metrics._distribution([r['ttft_s'] for r in per_request]),
                   request_max_itl_s=metrics._distribution([r['max_itl_s'] for r in per_request]), max_itl_s=max(r['max_itl_s'] for r in per_request),
                   software={k:env[k] for k in ('python','torch','cuda','vllm','transformers')},
                   gpu_uuid=re.search(r'GPU-[\w-]+', env['gpu_before']['device']).group(),
                   raw_path=str(raw_path), raw_sha256=sha(raw_path), _outputs={r:q['output_token_ids'] for r,q in requests.items()})
    except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError, StopIteration) as exc:
        row.update(status='INVALID_OR_INCOMPLETE', error=str(exc), eligible=False)
    return row


def compare(a, b, kind):
    result = dict(baseline=a['label'], action=b['label'], kind=kind, status='UNRUN_OR_INCOMPLETE')
    if not (a['eligible'] and b['eligible']):
        return result
    same = all({k:v for k,v in a[key].items() if k != 'enable_prefix_caching'} == {k:v for k,v in b[key].items() if k != 'enable_prefix_caching'} for key in ('engine','config'))
    require(same and a['software'] == b['software'] and a['gpu_uuid'] == b['gpu_uuid'], 'pair differs beyond APC')
    changes = []
    for x,y in zip(a['per_request'],b['per_request']):
        rid = x['request_id']; require(rid == y['request_id'], 'pair request order differs')
        left,right = a['_outputs'][rid],b['_outputs'][rid]
        changes.append(dict(request_id=rid, **{k:y[k]-x[k] for k in ('ttft_s','completion_s','max_itl_s')},
                            output_equal=left == right, first_output_difference=next((i for i,(u,v) in enumerate(zip(left,right)) if u != v), None)))
    result.update(status='DESCRIPTIVE_MATCHED_PAIR', per_request_changes=changes,
                  deltas={k:b[k]-a[k] for k in ('wall_s','throughput_rps','mean_completion_s','max_itl_s')},
                  relative_changes={k:b[k]/a[k]-1 for k in ('wall_s','throughput_rps','mean_completion_s')})
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--run-dir',type=Path,required=True); p.add_argument('--output-dir',type=Path,required=True); args=p.parse_args()
    require(not args.output_dir.exists(), 'output directory exists; refuse overwrite')
    source=args.run_dir/'frozen'
    if not source.is_dir(): source=Path(__file__).resolve().parents[2]/'outputs/admission_capacity/20260914_prefix_cache_baseline_r01/preparation/source'
    campaign=read(source/'campaign.json'); require([x['label'] for x in campaign['cells']] == LABELS, 'campaign order differs')
    require(all(x['role'] == x['label'].split('-')[-1] and x['block'] == int(x['label'].split('-')[1][-1])
                and x['cohort_id'] == 'cohort2' and x['completion_policy'] == 'native' and x['cap'] == 32
                and x['prefix_caching'] == x['role'].split('_')[-1] for x in campaign['cells']), 'campaign roles changed')
    inp=source/'cohorts/cohort2/inputs_preparation/prepared/long'; workload=read(inp/'workload.json'); config=read(inp/'config.json')
    require(hashlib.sha256(json.dumps(workload,sort_keys=True).encode()).hexdigest() == config['workload_sha256'] == WORKLOAD, 'frozen workload changed')
    spec=importlib.util.spec_from_file_location('apc_metrics',source/'metrics.py'); metrics=importlib.util.module_from_spec(spec); spec.loader.exec_module(metrics)
    rows=[inspect(args.run_dir,c,source,workload,config,metrics) for c in campaign['cells']]
    result=dict(status='UNRUN' if all(r['status']=='UNRUN' for r in rows) else 'INCOMPLETE_CAMPAIGN',cells=rows,comparisons=[],scope=SCOPE, source=str(source), metrics_sha256=sha(source/'metrics.py'))
    if all(r['eligible'] for r in rows):
        try:
            result['comparisons']=[compare(rows[i],rows[j],kind) for i,j,kind in [(0,1,'block0_off_on'),(3,2,'block1_off_on'),(0,3,'off_repeat'),(1,2,'on_repeat')]]
            result['status']='MEASUREMENT_ONLY'
        except ValueError as exc: result.update(status='INVALID_PAIRING',error=str(exc),comparisons=[])
    for row in rows: row.pop('_outputs',None)
    args.output_dir.mkdir(parents=True)
    (args.output_dir/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (args.output_dir/'REPORT.md').write_text('# Native APC baseline\n\n'+result['status']+'\n\n'+SCOPE+'\n\n'+ '\n'.join(f"- {r['label']}: {r['status']}" for r in rows)+'\n\nFull metrics, per-request changes, recovery spans and cache jumps are retained in analysis.json.\n')
    print(json.dumps(dict(status=result['status'],cells=[dict(label=r['label'],status=r['status']) for r in rows])))


if __name__ == '__main__':
    main()
