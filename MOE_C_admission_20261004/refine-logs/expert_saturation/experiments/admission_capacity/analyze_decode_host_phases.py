"""CPU marker phases around CUDA APIs; boundaries are descriptive, not causal."""
import argparse, json
from pathlib import Path


def analyze(path):
    events = json.loads(path.read_text())['traceEvents']
    spans = [e for e in events if e.get('ph') == 'X' and 'dur' in e]
    scopes = sorted([e for e in spans if e.get('cat') == 'user_annotation'
                     and e.get('name', '').startswith('decode_call_')], key=lambda e:e['ts'])
    apis = [e for e in spans if e.get('cat') in ['cuda_runtime', 'cuda_driver']]
    rows = []
    for s in scopes:
        lo, hi = s['ts'], s['ts']+s['dur']
        own = [e for e in apis if e['pid']==s['pid'] and e['tid']==s['tid']
               and lo <= e['ts'] < hi]
        sync = [e for e in own if e['name']=='cudaEventSynchronize']
        if not own or len(sync)!=1:
            raise ValueError('Expected main-thread APIs and one completion sync: '+s['name'])
        first = min(e['ts'] for e in own)
        end_sync = sync[0]['ts']+sync[0]['dur']
        if not first <= end_sync <= hi:
            raise ValueError('Invalid temporal boundaries')
        row = dict(call=int(s['name'].rsplit('_',1)[1]),
                   before_first_api_us=first-lo,
                   first_api_to_sync_return_us=end_sync-first,
                   after_sync_return_us=hi-end_sync,
                   scope_us=hi-lo)
        assert abs(sum(row[k] for k in ['before_first_api_us',
                   'first_api_to_sync_return_us','after_sync_return_us'])-row['scope_us'])<.01
        rows.append(row)
    if len(rows)!=32:raise ValueError('Expected 32 CPU scopes')
    keys=['before_first_api_us','first_api_to_sync_return_us','after_sync_return_us','scope_us']
    return dict(calls=rows, totals_us={k:sum(r[k] for r in rows) for k in keys},
                scope='Disjoint CPU temporal phases; middle includes asynchronous GPU work, waits and host work. Before/after are not assigned to any specific Python function.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    d={f.parent.name:analyze(f) for f in a.results.glob('*/decode-trace.json')}
    a.output.write_text(json.dumps(d,indent=2)+'\n')
    print({k:v['totals_us'] for k,v in d.items()})
