"""Post-hoc accounting of independently executed trajectories, not replay."""
import argparse
from collections import defaultdict
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); summary=json.loads((a.input_dir/'analysis.json').read_text())
    assert not summary['issues']
    totals=[]; comparisons=[]
    for engine in summary['engines']:
        root=a.input_dir/'results'/engine['label']; phases=defaultdict(dict)
        with (root/'pager/calls.jsonl').open() as f:
            for line in f:
                r=json.loads(line)
                if r['measurement']:
                    key=(r['context']['step_id'],r['layer_name'])
                    assert key not in phases[r['context']['phase']]
                    phases[r['context']['phase']][key]=r
        for row in engine['rows']:
            records=phases[row['phase']].values()
            active=sum(len(r['active_experts']) for r in records)
            hits=sum(len(set(r['active_experts']) & set(r['entry_resident_experts'])) for r in records)
            loads=sum(r['miss'] for r in records)
            assert active-hits==loads and row['payload_bytes']==loads*12582912
            totals.append(dict(engine=engine['label'],repeat=row['repeat'],arm=row['arm'],
                active_occurrences=active,entry_resident_hits=hits,loads=loads,payload_bytes=row['payload_bytes']))
        for start in (0,3):
            rows={r['arm']:r for r in engine['rows'][start:start+3]}
            x,y=[rows[k] for k in ('none_early','frequency_late_no_extra_groups')]
            aliases=[json.loads((root/r['name']/'raw.json').read_text())['internal_to_source'] for r in (x,y)]
            old,new=phases[x['phase']],phases[y['phase']]
            assert old.keys()==new.keys()
            first=None; different_groups=[]
            for key,u in old.items():
                v=new[key]
                identities=[[(m[r['internal_request_id']],r['computed_position'],r['prompt_tokens'])
                    for r in record['context']['rows']] for record,m in zip((u,v),aliases)]
                assert identities[0]==identities[1]
                detail=dict(step=key[0],layer=key[1],none_call_id=u['call_id'],guard_call_id=v['call_id'],
                    none_active=len(u['active_experts']),guard_active=len(v['active_experts']),
                    none_groups=len(u['groups']),guard_groups=len(v['groups']),guard_applied=v['retention']['applied'])
                if first is None and u['row_topk_experts']!=v['row_topk_experts']:first=detail
                if len(u['groups'])!=len(v['groups']):different_groups.append(detail)
            comparisons.append(dict(engine=engine['label'],block=start//3,
                first_observed_topk_difference=first,different_group_counts=different_groups))
    with a.out.open('x') as f:
        json.dump(dict(status='POST_HOC_EXECUTED_TRAJECTORY_ACCOUNTING',totals=totals,comparisons=comparisons,
            scope='Actual entry misses = active occurrences - actual resident hits. Cross-policy differences include route/state evolution; no shared-future replay, pure-cache causal attribution, or latency counterfactual.'),f,indent=2)
        f.write('\n')


if __name__=='__main__':main()
