"""Compare two actual pre-action fingerprints; no performance inference."""
import argparse
import json
from pathlib import Path


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();load=lambda p:json.loads(p.read_text())
    snapshots=[load(a.results/f'repeat{i}/action-state.json') for i in (0,1)]
    statuses=[load(a.results/f'repeat{i}/status.json') for i in (0,1)]
    for s,status in zip(snapshots,statuses):
        assert status['status']=='COMPLETE' and status['requests_completed']==32
        assert s['status']=='CAPTURED' and s['step']==329
        assert len(s['state']['requests'])==32 and len(s['layers'])==16
    x,y=snapshots
    assert x['layers'].keys()==y['layers'].keys()
    differences=[];segments=nonempty=byte_count=0
    for layer,xx in x['layers'].items():
        yy=y['layers'][layer]
        assert xx['shape']==yy['shape'] and xx['stride']==yy['stride']
        assert xx['requests'].keys()==yy['requests'].keys()
        for rid,kv in xx['requests'].items():
            other=yy['requests'][rid]
            assert kv['tokens']==x['state']['requests'][rid]['computed']
            assert other['tokens']==y['state']['requests'][rid]['computed']
            if kv!=other:differences.append(dict(layer=layer,request=rid,baseline=kv,repeat=other))
            segments+=1;nonempty+=kv['bytes']>0;byte_count+=kv['bytes']
    logical=[]
    for s in snapshots:
        state=json.loads(json.dumps(s['state']))
        for r in state['requests'].values():r.pop('blocks')
        logical.append(state)
    result=dict(status='MEASUREMENT_ONLY',step=329,
        full_recorded_state_equal=x['state']==y['state'],logical_request_state_equal=logical[0]==logical[1],
        kv_segments=segments,nonempty_segments=nonempty,bytes_checked_per_snapshot=byte_count,
        differing_segments=len(differences),differences=differences,
        diagnostic_seconds=[s['diagnostic_seconds'] for s in snapshots],completed_requests=64,
        scope='Two same-configuration native rebuilds; recorded request state and actual valid KV bytes only. '
              'Not an engine checkpoint, candidate branch evaluation, performance comparison or method GO.')
    with a.output.open('x') as f:json.dump(result,f,indent=2)
    print({k:v for k,v in result.items() if k!='differences'})


if __name__=='__main__':main()
