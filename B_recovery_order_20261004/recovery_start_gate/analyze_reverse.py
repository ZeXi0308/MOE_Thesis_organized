#!/usr/bin/env python3
"""Frozen start-gate analysis with explicit BAAB layout and fixed adjacent pairs."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
PARENT_SHA = 'd491d4ec0ec50d3c994e7127f2809c3a8d8b72d918f1455d364a93efeae3dfc0'
EXPECTED = ['wait_release', 'native', 'native', 'wait_release']
PAIRS = {0:1, 3:2}


def load_parent():
    path=ROOT/'analyze.py'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=PARENT_SHA:
        raise RuntimeError('Frozen complete-service analysis changed')
    spec=importlib.util.spec_from_file_location('reverse_start_gate_analysis',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def cell_index(path):
    match=re.search(r'cell-(\d+)-cap\d+-(?:native|wait_release)(?:/|$)',str(path))
    if match is None:raise ValueError('Unrecognized BAAB cell path: '+str(path))
    return int(match[1])


def label_reverse(result):
    cells=result['cells'];indices=[cell_index(c['directory']) for c in cells]
    if len(set(indices))!=len(indices) or any(i not in range(4) for i in indices):
        raise ValueError('Unexpected BAAB cell indices')
    if any(c['mode']!=EXPECTED[i] for c,i in zip(cells,indices)):
        raise ValueError('Observed cell mode differs from BAAB plan')
    by_index={i:c for i,c in zip(indices,cells)}
    for pair in result['comparisons']:
        candidate_index=cell_index(pair['candidate'])
        expected_native=PAIRS[candidate_index]
        native=by_index.get(expected_native)
        actual_index=cell_index(pair['native']) if pair.get('native') else None
        if actual_index!=expected_native:
            # A missing planned native is never replaced with a more distant
            # cell or fabricated data. No candidate-only outcome is a contrast.
            pair.clear();pair.update(candidate=by_index[candidate_index]['directory'],
                native=native['directory'] if native else None,status='UNAVAILABLE',
                reason='Planned adjacent native comparison is unavailable')
        pair.update(planned_candidate_cell_index=candidate_index,
                    planned_native_cell_index=expected_native,
                    reference_rule='Prespecified BAAB adjacent pairs: candidate0/native1, candidate3/native2; run-level descriptive contrast, not matched runtime state')
    layout=result['execution_layout']
    layout.pop('expected_abba',None);layout.pop('complete_abba',None)
    layout.update(design='ONE_TARGET_ASYNC_RECOVERY_START_GATE_BAAB',expected_baab=EXPECTED,
        observed_cell_indices=indices,complete_baab=indices==list(range(4)) and layout['modes']==EXPECTED,
        planned_but_not_started=[dict(cell_index=i,mode=mode) for i,mode in enumerate(EXPECTED) if i not in by_index],
        planned_adjacent_pairs=[dict(candidate_cell_index=i,native_cell_index=j) for i,j in PAIRS.items()])
    result['reverse_order_semantics']='One BAAB order reversal only. Original policy, observations and service calculations retained. First-candidate zero action can leave no native control; unstarted arms and missing contrasts remain explicit. No ABBA-completion claim or same-state causality.'
    result['analyzer_sources_sha256']['recovery_start_gate/analyze_reverse.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return result


def analyze_session(session):
    return label_reverse(load_parent().analyze_session(session))


def self_check():
    # Exercise the existing actual nearest-native implementation on parser-only
    # unavailable cells; no raw/results are fabricated or written to disk.
    path=ROOT.parent/'analyze.py'
    spec=importlib.util.spec_from_file_location('reverse_pairing_check',path)
    base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
    cells=[dict(directory=f'/UNRUN/cell-{i:02d}-cap256-{mode}/output',mode=mode,status='RAW_UNAVAILABLE') for i,mode in enumerate(EXPECTED)]
    def parsed(rows):
        return label_reverse(dict(cells=rows,comparisons=base.comparisons(rows),
            execution_layout=dict(modes=[r['mode'] for r in rows],expected_abba=[],complete_abba=False),analyzer_sources_sha256={}))
    full=parsed(cells)
    assert [(p['planned_candidate_cell_index'],cell_index(p['native'])) for p in full['comparisons']]==[(0,1),(3,2)]
    assert full['execution_layout']['complete_baab'] and 'complete_abba' not in full['execution_layout']
    early=parsed(cells[:1]);layout=early['execution_layout']
    assert not layout['complete_baab'] and [r['cell_index'] for r in layout['planned_but_not_started']]==[1,2,3]
    assert early['comparisons'][0]['status']=='UNAVAILABLE' and early['comparisons'][0]['native'] is None
    assert [r['mode'] for r in layout['planned_but_not_started']]==['native','native','wait_release']
    load_parent()
    print('PASS: frozen analysis pin, actual nearest-native BAAB pairs 0/1 and 3/2, explicit BAAB layout and unstarted arms, candidate-only no-control outcome. CPU only; no results written.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--self-check',action='store_true')
    args=parser.parse_args()
    if args.self_check:self_check()
    if args.session is None:
        if args.self_check:return 0
        parser.error('--session and --output are required')
    if args.output is None:parser.error('--output is required')
    if args.output.exists():raise FileExistsError(args.output)
    result=analyze_session(args.session)
    with args.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps(dict(output=str(args.output),layout=result['execution_layout'])))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
