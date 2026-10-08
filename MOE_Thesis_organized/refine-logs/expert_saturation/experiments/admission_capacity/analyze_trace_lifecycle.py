"""Cost-complete comparison of retained-memory versus episode trace flushing."""
import argparse
import hashlib
import json
from pathlib import Path
from analyze_runtime_variance import overlap, union_length


def read(path):
    return json.loads(path.read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input-dir',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();root=args.input_dir
    execution=read(root/'results/execution.json');protocol=read(root/'protocol.json')
    result={};signatures=[];issues=[]
    for cell in execution['cells']:
        label=cell['label'];r=root/'results'/label
        phases=read(root/f'{label}_phases.json');observed=read(root/f'{label}_observations.json')
        cycle=read(r/'cycle_complete.json');pager=read(r/'pager_summary.json')
        events=[dict(start=e['start_perf_ns']/1e9,stop=e['stop_perf_ns']/1e9,
                     generation=e['generation']) for e in read(r/'runtime_events.json')['events']
                if e.get('start_perf_ns') is not None and e.get('stop_perf_ns') is not None]
        snapshots=[read(r/e['phase'].split('/')[0]/'raw.json') for e in read(r/'episodes.json')]
        windows={k:[] for k in ('measurement','warmup','flush')}
        counts={k:dict(episodes=0,requests=0,tokens=0) for k in ('measurement','warmup')}
        for d in sorted(r.glob('repeat_*')):
            for path in [d/'raw.json',*d.glob('warmup*.json')]:
                raw=read(path);kind='measurement' if path.name=='raw.json' else 'warmup'
                if raw['status']!='COMPLETE':issues.append(f'{label}/{path.name}: incomplete')
                start=raw['measurement_origin_perf_counter_s']
                windows[kind].append((start,start+raw['observation_end_s']))
                counts[kind]['episodes']+=1;counts[kind]['requests']+=len(raw['requests'])
                counts[kind]['tokens']+=sum(len(x['output_token_ids']) for x in raw['requests'])
        for f in cycle['trace_flushes']:
            windows['flush'].append((f['start_perf_ns']/1e9,f['end_perf_ns']/1e9))
        all_windows=[w for v in windows.values() for w in v]
        assert abs(union_length(all_windows)-sum(b-a for a,b in all_windows))<1e-6
        start,end=cycle['start_perf_ns']/1e9,cycle['end_perf_ns']/1e9
        gc={kind:sum(overlap(events,a,b)['union_s'] for a,b in v) for kind,v in windows.items()}
        classified_gc=sum(gc.values())
        gc['cycle']=overlap(events,start,end)['union_s']
        assert classified_gc <= gc['cycle']+1e-6
        gc['other']=max(0.0,gc['cycle']-classified_gc)
        # A gen2 event may overlap more than one boundary; event counts are not added.
        gen2=overlap(events,start,end,2)
        trace_sig={};identities={}
        for ep,raw in zip(read(r/'episodes.json'),snapshots,strict=True):
            identities[ep['phase']]=raw['internal_to_source'];trace_sig[ep['phase']]=[]
        all_calls=0
        with (r/'pager/calls.jsonl').open() as stream:
            for line in stream:
                record=json.loads(line)
                assert record['call_id']==all_calls
                all_calls+=1;phase=record['context']['phase']
                if not record['measurement']:continue
                ids=identities[phase]
                trace_sig[phase].append(dict(step=record['context']['step_id'],layer=record['layer_name'],
                    rows=[(ids[x['internal_request_id']],x['computed_position'],x['prompt_tokens']) for x in record['context']['rows']],
                    topk=record['row_topk_experts'],active=record['active_experts'],entry=record['entry_resident_experts'],
                    groups=[{k:g[k] for k in ('required_experts','ensure_experts','loaded_experts','evicted_experts','weight_copy_bytes')} for g in record['groups']],
                    final_cache=record['retention']['final_resident_experts']))
        assert all_calls==pager['all_calls']
        trace_hashes=[digest(s) for s in trace_sig.values()]
        output_hashes=[digest({x['request_id']:x['output_token_ids'] for x in raw['requests']}) for raw in snapshots]
        signatures.extend(trace_hashes)
        boundary=[read(r/'environment.json')['sources']==protocol['source_sha256']]
        for line in (r/'gpu_checks.jsonl').read_text().splitlines():
            boundary.append(json.loads(line)['decision']=='PASS')
        if not all(boundary):issues.append(label+': source/GPU boundary mismatch')
        if not phases['phase_sum_matches_global'] or any(x['issues'] for x in phases['repeats'].values()) or observed['issues']:
            issues.append(label+': underlying phase/observation issue')
        xs=list(observed['repeats'].values());ys=list(phases['repeats'].values())
        row=dict(mode=cycle['trace_retention'],status=cell['status'],process_wall_s=cell['process_wall_s'],
            post_init_cycle_wall_s=cycle['post_init_cycle_wall_s'],counts=counts,all_calls=all_calls,
            measurement_layer_calls=pager['measurement_calls'],gpu_boundary_checks=len(boundary)-1,
            capture_wall_s=[x['wall_s'] for x in xs],capture_wall_sum_s=sum(x['wall_s'] for x in xs),
            capture_gc_s=[x['gc_episode']['union_s'] for x in xs],gc_by_window_s=gc,gen2_cycle=gen2,
            flushes=cycle['trace_flushes'],flush_wall_s=sum(b-a for a,b in windows['flush']),
            held_before_measurement=[x['retained_records_before'] for x in xs],
            rss_before_measurement=[x['rss_before'] for x in xs],
            payload_bytes=[x['actual_bytes'] for x in ys],groups=[x['retention']['groups'] for x in ys],
            requests=[x['requests'] for x in xs],trace_sha256=trace_hashes,output_sha256=output_hashes,
            source_gpu_checks_pass=all(boundary),allocated_peaks=[x['memory']['after_measurement']['peak_allocated_bytes'] for x in ys])
        result[label]=row
    pairs=[]
    labels=list(result)
    for a,b in [(labels[0],labels[1]),(labels[3],labels[2])]:
        x,y=result[a],result[b];assert x['mode']=='memory' and y['mode']=='episode'
        pairs.append(dict(memory=a,episode=b,deltas_pct={k:(y[k]/x[k]-1)*100 for k in
            ('process_wall_s','post_init_cycle_wall_s','capture_wall_sum_s')},
            gc_cycle_delta_s=y['gc_by_window_s']['cycle']-x['gc_by_window_s']['cycle']))
    output=dict(status='MEASUREMENT_INFRASTRUCTURE_INTERVENTION',issues=issues,engines=result,pairs=pairs,
        all_measurement_traces_equal=len(set(signatures))==1,
        all_outputs_equal=len({h for x in result.values() for h in x['output_sha256']})==1,
        semantics=['Process wall includes child startup, all retained artifacts, teardown and exit.',
            'Post-init cycle excludes setup and writing its final marker; process wall covers those costs.',
            'GC intervals overlap CPU/CUDA; intersections are descriptive, never subtracted as speedup.',
            'Two engines per mode; dependent episodes from three reused documents, no formal significance or scheduler benefit.'])
    with args.out.open('x') as f:json.dump(output,f,indent=2);f.write('\n')
    if issues:raise SystemExit('issues retained')


if __name__=='__main__':
    main()
