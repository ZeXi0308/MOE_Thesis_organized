#!/usr/bin/env python3
"""Native cap224 versus cap256: complete-request gaps and service costs."""
import argparse
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parents[1]
PARENT_SHA='d491d4ec0ec50d3c994e7127f2809c3a8d8b72d918f1455d364a93efeae3dfc0'
CAPS=[256,224,224,256]


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def comparisons(core,cells):
    text=inspect.getsource(core.comparisons)
    old="Path(c['directory']).parent.name.endswith('-native')"
    if text.count(old)!=1:raise RuntimeError('Frozen comparison-role boundary changed')
    ns=dict(vars(core));exec(compile(text.replace(old,"c['declared_cap'] == 256"),'<static-cap-comparisons>','exec'),ns)
    return ns['comparisons'](cells)


def distribution(core,values):
    result=core.summary(values);known=sorted(v for v in values if v is not None)
    pos=(len(known)-1)*.99;lo=int(pos);hi=min(lo+1,len(known)-1)
    result['p99']=known[lo]+(known[hi]-known[lo])*(pos-lo) if known else None
    return result


def analyze_session(session):
    path=ROOT.parent/'analyze.py'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=PARENT_SHA:
        raise RuntimeError('Frozen canonical analysis changed')
    parent=load('simple_cap_canonical',path);result=parent.analyze_session(session)
    core=load('simple_cap_request_metrics',BASE/'analyze.py')
    by_path={c['directory']:c for c in result['cells']};indices=[]
    for cell in result['cells']:
        match=re.search(r'cell-(\d+)-cap(\d+)-native/',cell['directory']+'/')
        if match is None:raise ValueError('Not a native static-cap cell')
        index,cap=map(int,match.groups());indices.append(index)
        if index not in range(4) or cap!=CAPS[index]:raise ValueError('Unexpected fixed cap order')
        cell['baseline_role']='cap'+str(cap)
        action=cell['recovery_start_gate_actions']
        cell['no_B_intervention_verified']=action.get('actual_gate_count')==action.get('executed_break_count')==0 and action.get('status')=='ANALYZED'
        rows=cell.get('per_request',[]);planned=cell.get('planned')
        cell['primary_maxgap']=dict(planned_denominator=planned,recorded_requests=len(rows),
            exact_complete_distribution=len(rows)==planned and all(r['status']=='completed' and r['maxgap_s'] is not None for r in rows),
            observed_closed_gaps=distribution(core,[r['maxgap_s'] for r in rows]),
            all_request_censored_lower_bounds=distribution(core,[r['maxgap_lower_bound_s'] for r in rows]),
            semantics='All original requests/statuses retained. Incomplete requests supply lower bounds, not final maximum gaps; no-output gaps remain unknown, never zero.')
    result['comparisons']=comparisons(core,result['cells'])
    for pair in result['comparisons']:
        low,high=by_path[pair['candidate']],by_path.get(pair.get('native'))
        pair['cap224']=pair.pop('candidate');pair['cap256']=pair.pop('native')
        pair.update(compared_variable='fixed runtime cap224 minus cap256; both original native policies',
            reference_rule='Nearest cap256 cell in execution order, earlier on ties; prescribed pairs1→0 and2→3')
        if 'aggregate_delta_candidate_minus_native' in pair:
            pair['aggregate_delta_cap224_minus_cap256']=pair.pop('aggregate_delta_candidate_minus_native')
        for row in pair['per_request']:
            row['cap224_status']=row.pop('candidate_status');row['cap256_status']=row.pop('native_status')
        if high is None or low.get('actual_cap')!=224 or high.get('actual_cap')!=256:
            pair.update(status='UNAVAILABLE',reason='Actual cap224/cap256 pair not verified')
        exact=pair['status']=='AVAILABLE' and high is not None and low['primary_maxgap']['exact_complete_distribution'] and high['primary_maxgap']['exact_complete_distribution']
        pair['primary_maxgap_delta_cap224_minus_cap256_s']={k:low['primary_maxgap']['observed_closed_gaps'][k]-high['primary_maxgap']['observed_closed_gaps'][k] if exact else None for k in ('mean','p99','maximum')}
        pair['drain_delta_cap224_minus_cap256_s']=(low['external_arrival_observations']['arrival_end_to_observation_end_s']-high['external_arrival_observations']['arrival_end_to_observation_end_s']) if high and all(c['external_arrival_observations'].get('arrival_end_to_observation_end_s') is not None for c in (low,high)) else None
    layout=result['execution_layout'];layout.pop('expected_abba',None);layout.pop('complete_abba',None)
    layout.update(design='STATIC_NATIVE_CAP_BASELINE',expected_caps=CAPS,expected_modes=['native']*4,
        observed_cell_indices=indices,complete_cap_abba=indices==list(range(4)),comparison_count=len(result['comparisons']),
        planned_but_not_started=[dict(cell_index=i,mode='native',cap=c) for i,c in enumerate(CAPS) if i not in indices])
    result['semantics']+=' Cross-cap contrasts here are explicitly the intended fixed-cap baseline experiment, overriding the inherited same-cap-only comparison scope.'
    result.pop('start_gate_semantics')
    result['native_observer_semantics']='Inherited start-gate records are native shadows only in all four cells; no B action is enabled. Recovery, copy work, fixed-output validation, compact-storage materialization, GC callback wall observations, process phases and external-arrival lag retain the frozen canonical calculations.'
    result['simple_cap_semantics']='One prespecified224 midpoint between previously seen192/256, not an optimum or B method. Same256 fixed1024 requests, arrivals, engine maxseq256, budgets, warmups and observation. Primary all-request maxgap mean/P99/max; TTFT/flow/throughput/drain are costs. Historical joint SLO retained only diagnostically. Zero recovery/B actions is a possible successful baseline outcome, not mechanism failure. Original failures/unfinished and missing denominators remain; no cross-run matched-state claim.'
    result['analyzer_sources_sha256']['recovery_start_gate/simple_cap/analyze.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return result


def self_check():
    core=load('simple_cap_parser_check',BASE/'analyze.py')
    cells=[dict(directory=f'/UNRUN/cell-{i:02d}-cap{cap}-native/output',declared_cap=cap,status='RAW_UNAVAILABLE') for i,cap in enumerate(CAPS)]
    pairs=comparisons(core,cells)
    assert [(p['candidate'],p['native']) for p in pairs]==[(cells[1]['directory'],cells[0]['directory']),(cells[2]['directory'],cells[3]['directory'])]
    result=distribution(core,[0.,1.,2.,None]);assert result['p99']==1.98 and result['missing']==1
    assert distribution(core,[])['p99'] is None
    print('PASS: canonical request comparisons assign cap224 to adjacent cap256; same interpolation for P99; missing values retained. CPU only; no results fabricated.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session',type=Path);parser.add_argument('--output',type=Path);parser.add_argument('--self-check',action='store_true')
    args=parser.parse_args()
    if args.self_check:self_check()
    if args.session is None:
        if args.self_check:return 0
        parser.error('--session and --output required')
    if args.output is None:parser.error('--output required')
    if args.output.exists():raise FileExistsError(args.output)
    result=analyze_session(args.session)
    with args.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print(args.output);return 0


if __name__=='__main__':raise SystemExit(main())
