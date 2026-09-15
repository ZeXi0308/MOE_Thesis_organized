"""Format frozen primary metrics and existing raw EOS/host facts; no new observation."""
import argparse
import json
import math
from pathlib import Path
from statistics import median


def distribution(values):
    rows=sorted(v for v in values if v is not None)
    def q(f):
        if not rows:return None
        x=(len(rows)-1)*f;a=math.floor(x);b=math.ceil(x)
        return rows[a]+(rows[b]-rows[a])*(x-a)
    return dict(defined=len(rows),undefined=len(values)-len(rows),median=q(.5),p90=q(.9),p95=q(.95),maximum=q(1))


def summarize(analysis,results):
    cells={}
    for name,cell in analysis['cells'].items():
        if not cell['comparable']:
            cells[name]=dict(status=cell['status']);continue
        rows=cell['requests']['requests'];folder=results/name
        raw=json.loads((folder/'raw.json').read_text())
        events=[e for e in raw['preemption_events'] if e['original_preemption_returned']
            and e['engine_call_index']<raw['engine_return_count']]
        first=min((e['method_entered_s'] for e in events),default=None)
        eos=[]
        for row in rows:
            if row['stop_reason']!='stop':continue
            own=[e for e in events if e['request_id']==row['request_id']]
            eos.append(dict(request_id=row['request_id'],arrival_s=row['arrival_s'],completion_s=row['completion_s'],
                outputs=row['outputs'],own_preemptions=len(own),
                completed_before_any_preemption=first is None or row['completion_s']<first,
                own_segments=[s for s in cell['sparse_recovery']['segments'] if s['request_id']==row['request_id']]))
        worst=None
        for row in raw['requests']:
            times=sorted(set(row['token_times_s']))
            for a,b in zip(times,times[1:]):
                if worst is None or b-a>worst['gap_s']:
                    inside=[e for e in events if e['request_id']==row['request_id'] and a<=e['method_entered_s']<=b]
                    worst=dict(request_id=row['request_id'],last_output_s=a,next_output_s=b,gap_s=b-a,
                        own_preemptions_inside=len(inside))
        timing=cell['timing'];host=cell['host_snapshots'];after=host['host-after.json']
        memory=json.loads((folder/'memory-after.json').read_text())
        terminal=[dict(e,source_request_id=raw['internal_to_source'].get(e['request']))
            for e in cell['policy_events'] or [] if e['event']=='target_terminal']
        cells[name]=dict(status=cell['status'],gap_distribution=distribution([r['max_engine_return_gap_s'] for r in rows]),
            largest_gap_interval=worst,eos=eos,eos_with_own_preemption=sum(e['own_preemptions']>0 for e in eos),
            eos_before_any_preemption=sum(e['completed_before_any_preemption'] for e in eos),target_terminal_events=terminal,
            sparse_counts=cell['sparse_recovery']['counts'],applied_rotations=cell['applied_rotations'],
            gpu_kv_bytes=memory['kv_storage_bytes'],host_allocated_bytes=after['cpu_kv']['unique_storage_bytes'],
            host_before_valid_blocks=host['host-before.json']['manager']['valid_host_kv_entries'],
            host_end_valid_blocks=host['host-request-end.json']['manager']['valid_host_kv_entries'],
            host_after_valid_blocks=after['manager']['valid_host_kv_entries'],host_after_pending=after['pending'],
            host_after_pending_entries=after['manager']['pending_store_entries'],
            host_after_valid_gib=after['cpu_kv']['derived_valid_host_kv_bytes']/1024**3,
            vmhwm_gib=max(s['process_peak_rss_bytes'] for s in host.values())/1024**3,
            parent_cgroup=after['parent_cgroup'],
            engine_init_s=timing['engine_init_end_perf_s']-timing['engine_init_start_perf_s'],
            warmup_s=timing['warmup_end_perf_s']-timing['warmup_start_perf_s'],
            process_wall_s=timing['process_end_perf_s']-timing['process_start_perf_s'])
    contrasts=[]
    for pair in analysis['performance_comparisons']+analysis['system_reference_comparisons']:
        result={k:pair[k] for k in ('block','baseline','arm','status')}
        if pair['status']=='COMPLETE':
            rows=pair['request_deltas'];deltas={}
            for metric in ('gap','completion','ttft'):
                values=[r[f'{metric}_arm_minus_baseline_s'] for r in rows]
                known=[v for v in values if v is not None]
                deltas[metric]=dict(improved=sum(v<0 for v in known),worsened=sum(v>0 for v in known),
                    unchanged=sum(v==0 for v in known),undefined=len(values)-len(known),
                    median=median(known) if known else None,largest_worsening=max(known,default=None))
            result.update(output_sequences_identical=sum(r['output_identical'] for r in rows),
                output_lengths_identical=sum(r['outputs_arm_minus_baseline']==0 for r in rows),
                stop_reasons_identical=sum(r['stop_baseline']==r['stop_arm'] for r in rows),request_delta_summaries=deltas)
        contrasts.append(result)
    return dict(cells=cells,contrasts=contrasts,semantics='Primary values are inherited from frozen analysis.json; quantiles use(n-1)q interpolation. Sparse preemptions/EOS/host intervals are observed facts, not load/recompute or removable-time inference. No late64 prediction reanalysis here.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--analysis',type=Path,required=True)
    p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();value=summarize(json.loads(args.analysis.read_text()),args.results)
    with args.output.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
