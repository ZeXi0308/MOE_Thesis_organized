"""Integrate one default-off execution change; preserve the upstream selector."""
import argparse
import json
from pathlib import Path
from recovery_execution_model_adapter import load_simulator
import validate_context_progress_model as helper


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--resource-model',type=Path,required=True)
    p.add_argument('--funding-bundle',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args()
    if a.output_dir.exists():raise FileExistsError(a.output_dir)
    helper.RESULTS=a.funding_bundle/'execution/readback/results'
    selector=helper.load_module('execution_upstream_selector', a.funding_bundle/'execution/readback/pkg/absence_rotation.py')
    original=helper.load_module('execution_upstream_model',a.resource_model)
    rows=[]
    for cell,action,filtered in (('funding-block0-most_output','continuous_most',False),
                                 ('funding-block0-least_feasible','least',True)):
        root=helper.RESULTS/cell
        raw=helper.load(root/'raw.json');decisions=helper.load(root/'headroom-decisions.json')
        initial,history,qualified,config,memory=helper.build_input(cell,raw,decisions)
        simulate,source=load_simulator(a.resource_model,funding_filter=filtered)
        kw=dict(start_step=qualified['step'],max_steps=6000)
        off=simulate(initial,history,action,selector.AbsenceRotation,selector.RequestView,**kw)
        mismatch,compared,completed=helper.first_mismatch(off,raw,decisions,memory,initial)
        if mismatch or off['completed']!=completed:
            raise AssertionError(('default path changed',cell,mismatch))
        if not filtered:
            previous=original.simulate(initial,history,action,selector.AbsenceRotation,selector.RequestView,**kw)
            assert off==previous, 'default-off whole result changed'
        events=[]
        on=simulate(initial,history,action,selector.AbsenceRotation,selector.RequestView,
                    protect_native_recovery=True,natural_protection_events=events,**kw)
        differences=[(x,y) for x,y in zip(off['trace'],on['trace']) if x!=y]
        # Exact full request structural deltas, no time surrogate.
        completion_delta={rid:on['completed'][rid]-step for rid,step in off['completed'].items()}
        rows.append(dict(cell=cell,source=source,raw_sha256=helper.sha(root/'raw.json'),
                         default_compared_steps=compared,default_exact=True,
                         off=helper.summarize(off,initial),on=helper.summarize(on,initial),
                         native_protection_starts=events,
                         first_differing_call=({'off':differences[0][0],'on':differences[0][1]} if differences else None),
                         completion_step_delta=completion_delta,
                         full_traces_equal=off['trace']==on['trace'],
                         focused_trace={name:[r for r in pred['trace'] if 1026<=r['step']<=1030]
                                        for name,pred in (('off',off),('on',on))}))
    result=dict(status='CPU_RESOURCE_EXECUTION_COMPOSITION',rows=rows,
                scope='Two reused calibration cells; fixed upstream selector per pair, no victim search. '
                'Independent scalar futures, declared fixed output cap. No wall time, quality, GPU or method win.')
    a.output_dir.mkdir(parents=True)
    (a.output_dir/'model.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps([dict(cell=r['cell'],default_steps=r['default_compared_steps'],
        protected_starts=len(r['native_protection_starts']),traces_equal=r['full_traces_equal'],
        first_difference=None if r['first_differing_call'] is None else r['first_differing_call']['on']['step'],
        last_steps=[r['off']['last_step'],r['on']['last_step']],
        recompute=[r['off']['recomputed_positions'],r['on']['recomputed_positions']],
        completion_better=sum(v<0 for v in r['completion_step_delta'].values()),
        completion_worse=sum(v>0 for v in r['completion_step_delta'].values())) for r in rows]))


if __name__=='__main__':main()
