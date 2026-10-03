"""Post-hoc structural model validation; no wall-time or online-policy claim."""
import importlib.util
import json
from pathlib import Path
import sys
from recovery_progress_model import simulate

ROOT=Path(__file__).resolve().parents[2]
O=ROOT/'outputs/admission_capacity'


def main():
    out=O/'20260914_saved_prefix_model_r01'
    out.mkdir(exist_ok=False)
    spec=importlib.util.spec_from_file_location('frozen_rotation',O/'20260914_d6_action_branches_r01/pkg/absence_rotation.py')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    inp=json.loads((O/'20260914_recovery_progress_model_r01/input.json').read_text())
    state,history=inp['state'],inp['history']
    victim=min(state['running'],key=lambda rid:state['requests'][rid]['output'])
    prefix=(state['requests'][victim]['computed']//16)*16
    assert prefix==3296
    # The delay is an explicit first-event calibration assumption, not future labels.
    predicted={arm:simulate(state,history,'native',module.AbsenceRotation,module.RequestView,
                            native_preemptions={329:victim},saved_prefixes={victim:prefix} if arm=='on' else None,
                            restore_delay_steps=2) for arm in ('off','on')}
    original=json.loads((O/'20260914_recovery_progress_model_r01/predictions.json').read_text())
    regression={a:simulate(state,history,a,module.AbsenceRotation,module.RequestView)==original[a] for a in ('least','most','defer')}
    assert all(regression.values()), 'Legacy model behavior changed'
    checks=[]
    for block in (0,1):
        for arm,pred in predicted.items():
            path=O/f'20260914_selective_store_repeat_r01/readback/results/block{block}-{arm}/raw.json'
            raw=json.loads(path.read_text());mem=raw['memory_trace'][329]['before']
            assert mem['pool']['free_blocks']==state['free']
            assert ['measured/'+raw['internal_to_source'][rid] for rid in mem['running_ids']]==state['running']
            for rid,actual in mem['requests'].items():
                expected=state['requests']['measured/'+raw['internal_to_source'][rid]]
                assert (actual['computed_tokens'],actual['output_tokens'],actual['num_preemptions'],actual['block_counts'][0])==(expected['computed'],expected['output'],expected['preemptions'],len(expected['blocks']))
            first=None
            for row in pred['trace']:
                n=row['step']
                if n>=len(raw['scheduler_steps']):
                    first=dict(step=n,kind='actual_ended');break
                actual=raw['scheduler_steps'][n]
                wanted={f"measured/{q['request_id']}":dict(computed=q['scheduled_start_computed'],tokens=q['scheduled_tokens'],output=q['output_tokens_before']) for q in actual['scheduled']}
                matched=dict(schedule=list(row['scheduled'].items())==list(wanted.items()),
                             free_blocks=row['free_after_schedule']==raw['memory_trace'][n]['after']['pool']['free_blocks'],
                             returned_tokens=len(row['outputs'])==raw['engine_steps'][n]['new_output_tokens'])
                if not all(matched.values()):first=dict(step=n,checks=matched);break
            checks.append(dict(block=block,arm=arm,initial_state_matches=True,
                               predicted_last_step=pred['last_step'],actual_last_step=len(raw['engine_steps'])-1,
                               first_mismatch=first,matched_until=(first['step']-1 if first else pred['last_step'])))
    result=dict(status='PARTIAL_STRUCTURAL_CALIBRATION',legacy_regression=regression,checks=checks,
                input_source='Existing frozen pre-action state, independently matched to all four actual states.',
                assumptions=dict(native_preempt_step=329,victim=victim,saved_tokens=prefix,load_delay_steps=2),
                scope='No future trace rows passed to simulator. Delay two is calibrated from first observed load, '
                      'not a generally validated latency law. Terminal count agreement does not override trajectory mismatch. '
                      'No wall-clock, quality, new workload holdout or deployable policy result.')
    (out/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    (out/'model_source.py').write_text(Path(__file__).with_name('recovery_progress_model.py').read_text())
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
