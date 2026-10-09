"""Complete-service analysis of one bounded head-completion intervention.

Reuses existing native accounting and preserves all requests/output divergence.
No automatic gain threshold, repeat, or conversion into an online-method claim.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))
from prefill_demand import analyze_demand as accounting
from prefill_demand.decompose_service import request_times
component=accounting.component


def load(path,design,signature,calibration,design_hash):
    row=component.load_run(path,design,signature,calibration,design_hash)
    if '_raw' not in row:
        return row
    raw=row['_raw']
    steps=raw['steps']
    probes=[(i,s['component_decision']) for i,s in enumerate(steps)
            if s['component_decision'].get('probe_snapshot') is not None]
    action=[i for i,s in enumerate(steps) if s['budget']!=1024]
    formal=not path.parent.name.startswith('warm_')
    if formal:
        if len(probes)>1: row['errors'].append('More than one opportunity snapshot')
        if raw['policy']=='fixed1024' and action: row['errors'].append('Baseline action changed')
        if raw['policy']=='head2048':
            if len(action)>4 or any(steps[i]['budget']!=2048 for i in action):
                row['errors'].append('Intervention exceeds frozen four-decision/2048 action range')
            if action and action!=list(range(action[0],action[-1]+1)):
                row['errors'].append('Intervention is not a single consecutive episode')
    head=probes[0][1].get('probe_head_id') if probes else None
    requests={q['request_id']:q for q in raw['requests']}
    row['probe']=dict(trigger_count=len(probes),trigger_step=probes[0][0] if probes else None,
        trigger_time_s=probes[0][1]['decision_time_s'] if probes else None,
        head_id=head,snapshot=probes[0][1]['probe_snapshot'] if probes else None,
        requested_nonbaseline_indices=action if formal else None,
        actual_above1024_indices=[i for i in action if steps[i]['prefill_tokens']>1024] if formal else None,
        head_times=request_times(requests[head]) if head in requests else None,
        head_first_s=requests[head]['token_times_s'][0] if head in requests and requests[head]['token_times_s'] else None,
        scope='One observed decision-time snapshot. Runtime performs a frozen probe; no online rollout or future arrivals.')
    row['valid']=row['valid'] and not row['errors']
    return row


def compare(f,x,design):
    result=accounting.compare(f,x,design)
    if not f or not x or '_raw' not in f or '_raw' not in x:
        return result
    fr,xr=({q['request_id']:q for q in r['_raw']['requests']} for r in (f,x))
    if fr.keys()!=xr.keys():return result
    deltas={rid:{k:request_times(xr[rid])[k]-request_times(fr[rid])[k]
                 if request_times(xr[rid])[k] is not None and request_times(fr[rid])[k] is not None else None
                 for k in ('ttft_s','generation_span_s','flow_s')} for rid in fr}
    result['all_request_decomposition_s']={k:component.native.describe([r[k] for r in deltas.values()]) for k in ('ttft_s','generation_span_s','flow_s')}
    result['per_request_decomposition_s']=deltas
    fp,xp=f['probe'],x['probe']
    def logical(snapshot):
        if snapshot is None:return None
        keys=('id','prompt','output_limit','arrival','prefill_done','emitted')
        return {queue:[tuple(q[k] for k in keys) for q in snapshot[queue]] for queue in ('running','waiting')}
    result['prefix_comparability']=dict(fixed_trigger_step=fp['trigger_step'],candidate_trigger_step=xp['trigger_step'],
        head_matches=fp['head_id'] is not None and fp['head_id']==xp['head_id'],
        logical_running_waiting_equal=fp['snapshot'] is not None and logical(fp['snapshot'])==logical(xp['snapshot']),
        trigger_clock_candidate_minus_fixed_s=xp['trigger_time_s']-fp['trigger_time_s'] if xp['trigger_time_s'] is not None and fp['trigger_time_s'] is not None else None,
        scope='Same fixed policy prefix does not guarantee identical wall-clock ages or arrival injection. Full strategy results remain separate from matched-state claims.')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run_directory',type=Path)
    p.add_argument('--design',type=Path,default=ROOT/'probe_design.json')
    p.add_argument('--output',type=Path,default=None)
    a=p.parse_args()
    db=a.design.read_bytes();design=json.loads(db)
    wb=(a.design.parent/design['workload']).read_bytes()
    cb=(a.design.parent/design['calibration_file']).read_bytes()
    assert component.sha(wb)==design['workload_sha256']
    assert component.sha(cb)==design['calibration_sha256']
    expected=[f'{i:02d}_{name}' for i,name in enumerate(design['policies'])]
    warms=['warm_'+name for name in design['warm_policies']]
    rows=[load(path,design,component.signature(json.loads(wb)),json.loads(cb),component.sha(db))
          for path in sorted(a.run_directory.glob('*/raw.json'))]
    lookup={r['cell']:r for r in rows}
    missing=[name for name in expected+warms if name not in lookup]
    status_path=a.run_directory/'status.json'
    driver=json.loads(status_path.read_text()) if status_path.exists() else dict(status='NOT_SUBMITTED')
    ready=not missing and driver.get('status')=='COMPLETE' and all(r['valid'] for r in rows)
    coverage={str(cap):sum(s['prefill_tokens']==cap for s in lookup.get('warm_fixed'+str(cap),{}).get('_raw',{}).get('steps',[])) for cap in (1024,2048)}
    ready=ready and all(coverage.values())
    out=dict(status='COMPLETE_EXPLORATORY_BOUNDED_PROBE' if ready else 'UNRUN' if not rows else 'INCOMPLETE_OR_INVALID',
        driver=driver,missing_cells=missing,warm_saturation=coverage,
        observed_formal_runs=sum(n in lookup for n in expected),
        valid_formal_runs=sum(lookup.get(n,{}).get('valid',False) for n in expected),
        pairs=[compare(lookup.get(expected[0]),lookup.get(expected[1]),design),compare(lookup.get(expected[3]),lookup.get(expected[2]),design)],
        runs=[{k:v for k,v in r.items() if not k.startswith('_')} for r in rows],
        frozen_design_sha256=component.sha(db),
        interpretation='All arrivals/full elapsed and original research SLO; no automatic winner or added runs. Bounded action probe on seen input, not complete online method, strict same-state causality, independent confirmation or novelty.')
    output=a.output or a.run_directory/'probe_results.json'
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(dict(status=out['status'],observed_formal_runs=out['observed_formal_runs'],output=str(output))))


if __name__=='__main__':main()
