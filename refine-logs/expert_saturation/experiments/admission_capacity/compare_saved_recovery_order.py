"""Conditional order scenarios, not runtime results or validated latency bounds."""
import importlib.util
import json
from pathlib import Path
import statistics
import sys
from recovery_progress_model import simulate

O=Path(__file__).resolve().parents[2]/'outputs/admission_capacity'


def main():
    out=O/'20260914_saved_recovery_order_r01';out.mkdir(exist_ok=False)
    spec=importlib.util.spec_from_file_location('order_rotation',O/'20260914_d6_action_branches_r01/pkg/absence_rotation.py')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    x=json.loads((O/'20260914_recovery_progress_model_r01/input.json').read_text())
    state=x['state'];victim=min(state['running'],key=lambda rid:state['requests'][rid]['output']);target=state['waiting'][0]
    predictions={};rows=[]
    for order in ('victim','original_waiting'):
        for delay in (None,1,2,4):
            result=simulate(state,x['history'],'native',module.AbsenceRotation,module.RequestView,
                            native_preemptions={329:victim},native_resume_first={329:target} if order=='original_waiting' else None,
                            saved_prefixes={victim:3296} if delay else None,restore_delay_steps=delay or 2)
            key=f'{order}/'+('no_save' if delay is None else f'load_delay_{delay}')
            predictions[key]=result
            rows.append(dict(scenario=key,status=result['status'],total_calls=result['last_step']+1,
                             mean_completion_step=statistics.mean(result['completed'].values()),
                             first_new_token_step={rid:next(t['step'] for t in result['trace'] if rid in t['outputs']) for rid in (victim,target)},
                             post_cutoff_preemptions=sum(len(t['preempted']) for t in result['trace']),
                             loads=sum(len(t.get('loads',[])) for t in result['trace'])))
    a=predictions['victim/no_save']['completed'];b=predictions['original_waiting/no_save']['completed']
    changed={rid:dict(victim_first=a[rid],original_waiting_first=b[rid],delta=b[rid]-a[rid]) for rid in a if a[rid]!=b[rid]}
    measured=[]
    for block in (0,1):
        for arm in ('off','on'):
            r=json.loads((O/f'20260914_selective_store_repeat_r01/readback/results/block{block}-{arm}/raw.json').read_text())
            first={rid:next(s['step'] for s in r['scheduler_steps'][329:] if any('measured/'+q['request_id']==rid and q['decode_tokens'] for q in s['scheduled'])) for rid in (victim,target)}
            measured.append(dict(block=block,arm=arm,first_new_token_step=first))
    original=json.loads((O/'20260914_recovery_progress_model_r01/predictions.json').read_text())
    regression={a:simulate(state,x['history'],a,module.AbsenceRotation,module.RequestView)==original[a] for a in ('least','most','defer')}
    assert all(regression.values())
    data=dict(status='CONDITIONAL_CPU_ORDER_SCENARIOS',measured_current_order=measured,scenarios=rows,
              no_save_changed_completion_steps=changed,legacy_regression=regression,
              scope='Order alternatives are UNRUN. Load delays 1/2/4 are sensitivity assumptions, not validated timing bounds; '
                    'no observed future completion sequence supplied. Steps are not seconds, and mean completion step is not mean latency. '
                    'Only one forced preemption and one saved prefix; no general scheduler or universal upper bound.')
    (out/'analysis.json').write_text(json.dumps(data,indent=2)+'\n')
    (out/'model_source.py').write_text(Path(__file__).with_name('recovery_progress_model.py').read_text())
    print(json.dumps(data,indent=2))


if __name__=='__main__':main()
