"""Describe actual static layer-budget executions; fixed-trace calibration is not a GPU prediction."""
import argparse
from collections import Counter, defaultdict
import hashlib
from itertools import combinations
import json
from pathlib import Path
import statistics
from analyze_retention_lifecycle import engine_result, describe, FIELDS, check, read, digest

LABELS = ['0_uniform', '1_selected', '2_min_groups', '3_min_groups', '4_selected', '5_uniform']
MODES = ('uniform', 'selected', 'min_groups')
RESOURCE = dict(actual_kv_bytes=1073741824, scratch_bytes=4831838208, expert_cap=24)
ERRORS = (OSError, KeyError, ValueError, TypeError, IndexError, StopIteration, AssertionError)


def difference(a, b, fields=FIELDS):
    return dict(delta_b_minus_a={k:b[k]-a[k] for k in fields},
                delta_pct={k:100*(b[k]/a[k]-1) if a[k] else None for k in fields})


def augment(root, engine, protocol, issues):
    label=engine['label']; path=root/'results'/label; mode=protocol['engine_allocations'][label]
    expected=protocol['allocations'][mode]; pager=read(path/'pager_summary.json')
    actual={r['layer_name']:r['cap'] for r in pager['layers']}
    check(actual==expected and len(pager['layers'])==16,label+': actual layer caps mismatch',issues)
    check(pager['scratch_bytes']==sum(r['scratch_bytes'] for r in pager['layers'])==RESOURCE['scratch_bytes'],label+': scratch mismatch',issues)
    check(all(r['scratch_bytes']==r['cap']*r['pinned_bytes']//r['num_experts'] for r in pager['layers']),label+': per-layer scratch mismatch',issues)
    config=read(path/'config.json'); resource=read(path/'resources.json'); workload=read(path/'workload.json')
    check(config['expert_cap']==24 and config['execution']=='expert' and config['same_engine_runtime_diagnostic']
          and config['trace_retention']=='episode' and not config['verify_kernel'] and config['requests']==3
          and config['injection_chunk']==8 and config['token_budget']==160,label+': runtime config mismatch',issues)
    check(resource['kv_cache_memory_bytes']==RESOURCE['actual_kv_bytes'],label+': requested KV mismatch',issues)
    docs=[r['document_id'] for r in workload['source_requests']]
    check(docs==protocol['workload']['selection']['evaluation_document_ids'],label+': performance document identity mismatch',issues)
    prepared=read(root/'prepared/workload.json')
    check(workload['source_requests']==prepared['source_requests'] and workload['actual_prompt_token_ids']==
          [ids[:n] for ids,n in zip(prepared['actual_prompt_token_ids'],protocol['workload']['lengths'])],label+': prepared request/prefix mismatch',issues)
    aliases={}; hashes={}; layer_totals=defaultdict(Counter)
    for row in engine['rows']:
        d=path/row['name']; res=read(d/'measurement_resources.json'); raw=read(d/'raw.json')
        declared=res.get('layer_caps',{n:res['expert_cap'] for n in actual})
        check(declared==actual and all(row['fixed_resources'][k]==v for k,v in RESOURCE.items()),row['name']+': actual resources mismatch',issues)
        check(res['actual_unique_kv_storage_bytes']==RESOURCE['actual_kv_bytes'] and res['scheduler_requests']==0,row['name']+': KV/drained mismatch',issues)
        check(res['cpu_affinity']==protocol['runtime']['cpu_affinity'],row['name']+': CPU affinity mismatch',issues)
        check(row['arm']=='none_early' and res.get('group_retention_guard','none')=='none',row['name']+': unexpected retention action',issues)
        if 'layer_caps' in res:
            check(res['expert_slots_total']==384 and res['expert_scratch_bytes']==RESOURCE['scratch_bytes'],row['name']+': declared budget totals mismatch',issues)
        row.update(allocation_mode=mode,layer_caps=actual,source_requests=workload['source_requests'])
        aliases[row['phase']]=raw['internal_to_source']; hashes[row['phase']]=hashlib.sha256()
    with (path/'pager/calls.jsonl').open() as stream:
        for line in stream:
            r=json.loads(line)
            if not r['measurement']: continue
            phase=r['context']['phase']; ids=aliases[phase]
            signature=dict(layer=r['layer_name'],step=r['context']['step_id'],topk=r['row_topk_experts'],
                rows=[(ids[x['internal_request_id']],x['computed_position'],x['prompt_tokens']) for x in r['context']['rows']])
            hashes[phase].update((digest(signature)+'\n').encode())
            layer_totals[r['layer_name']].update(miss=r['miss'],groups=len(r['groups']),bytes=r['weight_copy_bytes'],load_section_ms=r['load_cuda_span_ms'])
    for row in engine['rows']: row['route_sha256']=hashes[row['phase']].hexdigest()
    means={k:statistics.mean(r[k] for r in engine['rows']) for k in FIELDS}
    engine.update(allocation_mode=mode,layer_caps=actual,source_requests=workload['source_requests'],
        prompt_sha256=digest(workload['actual_prompt_token_ids']),description=describe(engine['rows']),means=means,
        per_layer_measurement={n:dict(cap=actual[n],**v) for n,v in layer_totals.items()},
        measurement_capture_sum_s=sum(r['capture_wall_s'] for r in engine['rows']),
        repeat_cycle_sum_s=sum(r['cycle_wall_s'] for r in engine['rows']),
        max_measurement_peak_allocated_bytes=max(r['cuda_memory']['after_measurement']['peak_allocated_bytes'] for r in engine['rows']),
        warmup_counts={k:sum(r['warmup_counts'][k] for r in engine['rows']) for k in ('episodes','requests','tokens')})


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--input-dir',type=Path,required=True); p.add_argument('--out',type=Path,required=True); args=p.parse_args()
    root=args.input_dir; issues=[]; engines=[]; pairs=[]; comparisons=[]
    try:
        protocol=read(root/'protocol.json'); execution=read(root/'results/execution.json')
        check(execution['status']=='COMPLETE' and [c['label'] for c in execution['cells']]==LABELS,'six-engine completion/order mismatch',issues)
        check(protocol['engine_allocations']==dict(zip(LABELS,('uniform','selected','min_groups','min_groups','selected','uniform'))),'declared allocation order mismatch',issues)
        check(protocol['engine_sequences']=={n:['none_early']*8 for n in LABELS},'declared repeat sequences mismatch',issues)
        check(set(protocol['allocations'])==set(MODES),'allocation modes mismatch',issues)
        for mode,mapping in protocol['allocations'].items():
            check(set(mapping)=={f'model.layers.{i}.mlp.experts' for i in range(16)} and
                  all(type(c) is int and 16<=c<=32 for c in mapping.values()) and sum(mapping.values())==384,mode+': invalid static allocation',issues)
        check(set(protocol['allocations']['uniform'].values())=={24},'uniform budget mismatch',issues)
        selection=protocol['workload']['selection']; evaluation=set(selection['evaluation_document_ids']); calibration=set(selection['calibration_document_ids'])
        check(len(evaluation)==len(calibration)==3 and not evaluation&calibration,'calibration/performance documents overlap or missing',issues)
        for cell in execution['cells']:
            try:
                engine=engine_result(root,cell,protocol); engines.append(engine)
                issues.extend(cell['label']+': '+s for s in engine['issues'])
                check(len(engine['rows'])==8,cell['label']+': measurement count mismatch',issues)
                augment(root,engine,protocol,issues)
            except ERRORS as exc:
                issues.append(cell['label']+': '+repr(exc))
                if not any(e['label']==cell['label'] for e in engines): engines.append(dict(label=cell['label'],status=cell.get('status'),rows=[],issues=[repr(exc)]))
        by_label={e['label']:e for e in engines}
        for block,labels in enumerate((LABELS[:3],list(reversed(LABELS[3:])))):
            for al,bl in combinations(labels,2):
                a,b=by_label[al],by_label[bl]
                if 'means' not in a or 'means' not in b: continue
                comparisons.append(dict(block=block,a=al,b=bl,**difference(a['means'],b['means']),
                    engine_totals=difference(a,b,('process_wall_s','post_init_cycle_wall_s','repeat_cycle_sum_s','measurement_capture_sum_s'))))
                for ordinal,(ar,br) in enumerate(zip(a['rows'],b['rows'])):
                    pairs.append(dict(block=block,ordinal=ordinal,a=al,b=bl,**difference(ar,br),
                        route_equal=ar['route_sha256']==br['route_sha256'],trace_equal=ar['trace_sha256']==br['trace_sha256'],output_equal=ar['output_sha256']==br['output_sha256'],
                        request_deltas={rid:{k:br['requests'][rid][k]-ar['requests'][rid][k] for k in ('ttft_s','tpot_s','max_itl_s','completion_latency_s')} for rid in ar['requests']}))
    except ERRORS as exc: issues.append('campaign: '+repr(exc))
    rows=[r for e in engines for r in e['rows']]; usable=[r for r in rows if 'route_sha256' in r]
    check(len(rows)==len(usable)==48 and len(pairs)==48 and len(comparisons)==6,'incomplete usable measurements/pairs/engine comparisons',issues)
    warmups=sum(e.get('warmup_counts',{}).get('episodes',0) for e in engines)
    check(warmups==240,'warmup episode count mismatch',issues)
    check(len({e.get('prompt_sha256') for e in engines})==1 and all(e.get('prompt_sha256') for e in engines),'performance prompt inputs differ',issues)
    equality={m:{k:len({r[k] for r in usable if r['allocation_mode']==m})==1 for k in ('route_sha256','trace_sha256','output_sha256')} for m in MODES}
    output=dict(status='ISSUES' if issues else 'DESCRIPTIVE_STATIC_LAYER_BUDGET_EXECUTION',issues=issues,engines=engines,
        repeat_pairs=pairs,engine_mean_comparisons=comparisons,within_allocation_equal=equality,
        counts=dict(measurements=len(rows),warmups=warmups),scope=[
            'Six fresh engines in U/M/G/G/M/U order; eight dependent repeats per engine and three shared performance documents. No significance, population noise floor or novelty claim.',
            'Pairs use the same repeat ordinal within each predeclared three-engine direction block; six engine comparisons describe eight-repeat means, not independent samples.',
            'Each static allocation executes its own future state. Route/trace/output equality is descriptive; fixed calibration miss counts are neither target expectations nor counterfactual GPU outcomes.',
            'Request/capture clocks include observations. Repeat cycle includes reset, five warmups, IO and flush, excluding its own cycle_progress write and shared tail; complete engine/process totals retain these costs.',
            'Host, CUDA and GC intervals overlap: never add them to wall or subtract them as benefit. GC observer closes before export/shutdown and only covers observed intervals.',
            'All completed, failed and partial engines are retained; CUDA allocated/reserved/peak snapshots remain in each row. Source/GPU checks are boundary evidence, not continuous isolation.'])
    with args.out.open('x') as f: json.dump(output,f,indent=2,allow_nan=False); f.write('\n')
    if issues: raise SystemExit('issues retained in output')


if __name__=='__main__': main()
