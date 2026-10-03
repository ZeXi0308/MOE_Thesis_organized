"""Bounded Chrome trace accounting; temporal overlap is not causal attribution."""
import argparse,json
from collections import Counter
from pathlib import Path


def union_us(intervals):
    end=None;total=0.
    for lo,hi in sorted(intervals):
        if hi<=lo:continue
        if end is None or lo>end:total+=hi-lo;end=hi
        elif hi>end:total+=hi-end;end=hi
    return total


def clipped(events,lo,hi):
    return [(max(lo,float(e['ts'])),min(hi,float(e['ts'])+float(e['dur'])))
            for e in events if float(e['ts'])<hi and float(e['ts'])+float(e['dur'])>lo]


def analyze(folder,expected):
    read=lambda p:json.loads(p.read_text())
    status=folder/'status.json'
    if not status.exists():return dict(status='UNRUN')
    terminal=read(status)
    if terminal['status']!='COMPLETE':return dict(status='INCOMPLETE',terminal=terminal)
    raw=read(folder/'raw.json');window=raw['scheduler_steps'][500:532]
    actual=[dict(step=s['step'],scheduled=[{k:r[k] for k in ['request_id','scheduled_start_computed','scheduled_tokens','output_tokens_before']} for r in s['scheduled']]) for s in window]
    qualification=read(folder/'decode-trace-status.json')
    if actual!=expected or not qualification['complete'] or not qualification['exported']:
        return dict(status='UNALIGNED_OR_INCOMPLETE_TRACE',schedule_matches=actual==expected,trace_status=qualification)
    events=read(folder/'decode-trace.json')['traceEvents']
    spans=[e for e in events if e.get('ph')=='X' and 'ts' in e and 'dur' in e]
    # Kineto also exports GPU annotations with identical names. Only the CPU
    # record_function scopes define this analysis window.
    markers=[e for e in spans if e.get('cat')=='user_annotation' and e.get('name','').startswith('decode_call_')]
    counts=Counter(e['name'] for e in markers)
    if counts!=Counter({'decode_call_'+str(i):1 for i in range(500,532)}):
        return dict(status='INVALID_MARKERS',markers=dict(counts))
    markers.sort(key=lambda e:e['ts']);lo=markers[0]['ts'];hi=max(e['ts']+e['dur'] for e in markers)
    gpu=[e for e in spans if str(e.get('cat','')).lower() in ['kernel','gpu_memcpy','gpu_memset']]
    if not gpu:return dict(status='NO_CUDA_ACTIVITY',categories=dict(Counter(str(e.get('cat','')) for e in spans)))
    apis=[e for e in spans if str(e.get('cat','')).lower() in ['cuda_runtime','cuda_driver']]
    rows=[]
    for m in markers:
        start,end=m['ts'],m['ts']+m['dur']
        rows.append(dict(call=int(m['name'].rsplit('_',1)[1]),cpu_scope_us=m['dur'],
                         gpu_activity_union_us=union_us(clipped(gpu,start,end)),
                         cuda_api_union_us=union_us(clipped(apis,start,end))))
    # This is a temporal intersection over all GPU activities, not per-request
    # kernel ownership or GPU busy time across all processes.
    by_name=Counter()
    for e in gpu:
        duration=max(0.,min(hi,e['ts']+e['dur'])-max(lo,e['ts']))
        if duration:by_name[e['name']]+=duration
    summaries={}
    for tag,subset in [('kernel',[e for e in gpu if e['cat']=='kernel']),('memcpy',[e for e in gpu if e['cat']=='gpu_memcpy']),('memset',[e for e in gpu if e['cat']=='gpu_memset'])]:
        summaries[tag]=union_us(clipped(subset,lo,hi))
    return dict(status='BOUNDED_TRACE_DIAGNOSTIC',window_us=hi-lo,cpu_scope_union_us=union_us(clipped(markers,lo,hi)),
                gpu_activity_union_us=union_us(clipped(gpu,lo,hi)),cuda_api_union_us=union_us(clipped(apis,lo,hi)),
                activity_classes_union_us=summaries,calls=rows,top_gpu_names_summed_us=by_name.most_common(15),
                gpu_event_count=sum(e['ts']<hi and e['ts']+e['dur']>lo for e in gpu),
                scope='Activity union within CPU marker window; class unions and CPU/API spans overlap and must not be added. Summed per-kernel names are ranking only. No CUDA activity gaps interpreted as hardware idle without coverage verification; no kernel ownership inferred.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--expected',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    expected=json.loads(a.expected.read_text());rows={label:analyze(a.results/label,expected) for label in ['trace-offload-off','trace-offload-on']}
    a.output.write_text(json.dumps(dict(status='DIAGNOSTIC_ONLY',arms=rows),indent=2)+'\n')
    print({k:v['status'] for k,v in rows.items()})


if __name__=='__main__':main()
