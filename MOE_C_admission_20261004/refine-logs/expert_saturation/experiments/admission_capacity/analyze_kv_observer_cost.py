"""Full retained-request outcomes and exact non-clock observer equivalence."""
import argparse, json, statistics
from collections import Counter
from pathlib import Path


def signature(raw):
    keys=['request_id','scheduled_start_computed','scheduled_tokens','output_tokens_before']
    return [dict(step=s['step'],scheduled=[{k:r[k] for k in keys} for r in s['scheduled']],
                 recompute_tokens=s['recompute_tokens'],preempted=s['preempted_request_ids'])
            for s in raw['scheduler_steps']]


def clean_memory(value, identities=None):
    identities = identities or {}
    if isinstance(value, dict):
        return {identities.get(k,k):clean_memory(v, identities) for k,v in value.items()
                if k not in ['host_start_perf_counter_s','host_end_perf_counter_s']}
    if isinstance(value, list):return [clean_memory(v, identities) for v in value]
    if isinstance(value, str):return identities.get(value,value)
    return value


def metrics(raw):
    rs=raw['requests']
    if len(rs)!=32 or any(r['status']!='completed' for r in rs):
        raise ValueError('Expected all 32 requests completed; retain failed raw separately')
    wall=max(r['completion_s'] for r in rs)-min(r['arrival_s'] for r in rs)
    return dict(mean_completion_s=statistics.mean(r['completion_s']-r['arrival_s'] for r in rs),
                wall_s=wall,output_tokens_per_s=sum(len(r['output_token_ids']) for r in rs)/wall,
                max_itl_s=max(b-a for r in rs for a,b in zip(r['token_times_s'],r['token_times_s'][1:])),
                recompute_tokens=sum(s['recompute_tokens'] for s in raw['scheduler_steps']),
                preemptions=raw['actual_preemption_count'])


def compare(a,b):
    ma,mb=metrics(a),metrics(b)
    for raw in [a,b]:
        ids=raw['internal_to_source']
        assert len(set(ids.values()))==len(ids), 'Ambiguous internal identity mapping'
    outputs=lambda d:{r['request_id']:r['output_token_ids'] for r in d['requests']}
    checks=dict(schedule_equal=signature(a)==signature(b),
                memory_equal=clean_memory(a['memory_trace'],a['internal_to_source'])==clean_memory(b['memory_trace'],b['internal_to_source']),
                outputs_equal=outputs(a)==outputs(b))
    return dict(status='EQUIVALENT_OBSERVER_COMPARISON' if all(checks.values()) else 'DIVERGED_REQUIRES_LOCALIZATION',
                checks=checks,original=ma,direct=mb,
                delta_pct={k:100*(mb[k]/ma[k]-1) if ma[k] else None for k in ma})


def transfer_counts(folder):
    data=json.loads((folder/'offload-events.json').read_text())
    # Poll frequency and completed-transfer grouping may differ. Compare actual
    # worker byte totals and the multiset of transfer sizes, never lookup sums.
    return {kind:dict(bytes=sum(r[kind]['bytes'] for r in data['transfers']),
                     sizes=sorted(Counter(v for r in data['transfers']
                                          for v in r[kind]['sizes']).items()))
            for kind in ['load','store']}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();pairs={}
    for block in [0,1]:
        for offload in ['off','on']:
            name=f'b{block}-{offload}'
            folders=[a.results/(name+'-'+m) for m in ['original','direct']]
            if not all((f/'status.json').exists() for f in folders):
                pairs[name]=dict(status='UNRUN');continue
            statuses=[json.loads((f/'status.json').read_text()) for f in folders]
            if any(s['status']!='COMPLETE' for s in statuses):
                pairs[name]=dict(status='INCOMPLETE',terminals=statuses);continue
            result=compare(*(json.loads((f/'raw.json').read_text()) for f in folders))
            transfers=[transfer_counts(f) for f in folders]
            result['transfers']=dict(original=transfers[0],direct=transfers[1])
            result['checks']['transfer_counts_equal']=transfers[0]==transfers[1]
            if not all(result['checks'].values()):
                result['status']='DIVERGED_REQUIRES_LOCALIZATION'
            pairs[name]=result
    a.output.write_text(json.dumps(dict(pairs=pairs,
        scope='Observer implementation comparison. Both blocks retained; no automatic GO threshold or policy benefit claim.'),indent=2)+'\n')
    print({k:v['status'] for k,v in pairs.items()})
