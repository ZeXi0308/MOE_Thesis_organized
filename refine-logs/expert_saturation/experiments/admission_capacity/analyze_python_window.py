"""Qualify and summarize bounded cProfile/GC data without claiming speedup."""
import argparse, json
from pathlib import Path


def analyze(folder, expected, baseline):
    read=lambda p:json.loads(p.read_text())
    if not (folder/'status.json').exists():return dict(status='UNRUN')
    if read(folder/'status.json')['status']!='COMPLETE':return dict(status='INCOMPLETE')
    raw=read(folder/'raw.json');q=read(folder/'decode-trace-status.json')
    actual=[dict(step=s['step'],scheduled=[{k:r[k] for k in
        ['request_id','scheduled_start_computed','scheduled_tokens','output_tokens_before']}
        for r in s['scheduled']]) for s in raw['scheduler_steps'][500:532]]
    if actual!=expected or not q['complete']:
        return dict(status='UNALIGNED',schedule_matches=actual==expected,window=q)
    d=read(folder/'decode-trace.json')
    if d['kind']!='CPROFILE_GC_DIAGNOSTIC' or d['calls']!=list(range(500,532)):
        raise ValueError('Wrong profile type or window')
    active={};pairs=[];unmatched=[]
    for e in d['gc_events']:
        gen=e['info']['generation']
        if e['phase']=='start':
            if gen in active:unmatched.append(active[gen])
            active[gen]=e
        elif gen not in active:unmatched.append(e)
        else:
            start=active.pop(gen)
            pairs.append(dict(generation=gen,start_call=start['call'],end_call=e['call'],
                              duration_s=(e['time_ns']-start['time_ns'])/1e9,
                              collected=e['info'].get('collected'),uncollectable=e['info'].get('uncollectable')))
    unmatched.extend(active.values())
    outputs=lambda r:{x['request_id']:x['output_token_ids'] for x in r['requests']}
    return dict(status='HOST_PROFILE_DIAGNOSTIC',outputs_equal=outputs(raw)==outputs(baseline),
                profile_scope_s=sum((x['end_ns']-x['start_ns'])/1e9 for x in d['spans']),
                gc_pairs=pairs,gc_unmatched=unmatched,
                top_self=sorted(d['functions'],key=lambda x:x['self_s'],reverse=True)[:25],
                top_cumulative=sorted(d['functions'],key=lambda x:x['cumulative_s'],reverse=True)[:25],
                scope='Main-thread cProfile perturbs timing. Inclusive functions, GC spans and measured call spans overlap; do not sum or interpret as removable cost.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folder',type=Path,required=True)
    p.add_argument('--expected',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    d=analyze(a.folder,json.loads(a.expected.read_text()),json.loads(a.baseline.read_text()))
    a.output.write_text(json.dumps(d,indent=2)+'\n');print(d['status'])
