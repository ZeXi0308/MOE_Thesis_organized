#!/usr/bin/env python3
"""Select next native arrival rate from completed cells; never launch or retry.

All cells must share prompt/model/KV identity. Bracket by doubling/halving,
then bisect. The development budget is six valid calibration cells per point;
missing/failed runs are reported separately, not treated as low preemption.
"""
import argparse
import json
from pathlib import Path


def decide(observations,initial=10.,max_cells=6):
    valid=[o for o in observations if o['complete']]
    accepted=[o for o in valid if .05<=o['preempted_request_fraction']<=.15]
    if accepted:
        selected=accepted[-1]
        return dict(status='CALIBRATED',arrival_rate=selected['arrival_rate'],
                    high_rate=2*selected['arrival_rate'],source=selected['session'])
    if len(valid)>=max_cells:
        return dict(status='UNRESOLVED_BUDGET',next_arrival_rate=None)
    low=[o['arrival_rate'] for o in valid if o['preempted_request_fraction']<.05]
    high=[o['arrival_rate'] for o in valid if o['preempted_request_fraction']>.15]
    if low and high:
        lower,upper=max(low),min(high)
        if lower>=upper:
            return dict(status='NONMONOTONE_REVIEW',next_arrival_rate=None)
        rate=(lower+upper)/2
    elif low:rate=max(low)*2
    elif high:rate=min(high)/2
    else:rate=initial
    return dict(status='NEXT_NATIVE',next_arrival_rate=rate,valid_cells=len(valid))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('sessions',nargs='*',type=Path)
    p.add_argument('--kv-gib',type=int,choices=(24,48),
                   help='Select one KV budget from a mixed-budget group')
    args=p.parse_args();observations=[];identities=set();seen=set();unmeasured=[]
    for session in args.sessions:
        if session.resolve() in seen:raise ValueError('Duplicate calibration session')
        seen.add(session.resolve())
        receipt=json.loads((session/'receipt.json').read_text())
        metrics=json.loads((session/'metrics.json').read_text())
        plan=receipt['plan']
        by_name={Path(r['cell']).name:r for r in metrics}
        if len(by_name)!=len(metrics):raise ValueError('Duplicate metric cell')
        expected={f'cell-{i:02d}-{c["policy"]}' for i,c in enumerate(plan['cells'])}
        if set(by_name)-expected:raise ValueError('Metrics contain cells absent from plan')
        for i,cell in enumerate(plan['cells']):
            if cell['policy']!='native' or cell['cap']!=256:continue
            if args.kv_gib is not None and cell['kv_gib']!=args.kv_gib:continue
            name=f'cell-{i:02d}-{cell["policy"]}'
            if name not in by_name:
                unmeasured.append(str(session/name));continue
            r=by_name[name]
            folder=Path(r['cell'])
            # Analyzer may have run after moving a downloaded session.
            if not folder.exists():folder=session/folder.name
            inputs=json.loads((folder/'inputs.json').read_text())
            ids=inputs['workload']['actual_prompt_token_ids']
            import hashlib
            identity=(inputs['config']['model_revision'],cell['kv_gib'],
                      hashlib.sha256(json.dumps(ids).encode()).hexdigest())
            identities.add(identity)
            observations.append(dict(session=str(session),cell=name,arrival_rate=cell['arrival_rate'],
                complete=r.get('complete',False),preempted_request_fraction=r.get('preempted_request_fraction')))
    if len(identities)>1:raise ValueError('Calibration model/workload/KV identities differ')
    print(json.dumps(dict(kv_gib=args.kv_gib,observations=observations,
                         unmeasured_cells=unmeasured,decision=decide(observations)),indent=2))
