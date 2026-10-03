"""Actual staged-event structural replay, not a predictive performance estimate."""
from bisect import bisect_right
import json
from pathlib import Path
from absence_rotation import AbsenceRotation,RequestView
from recovery_progress_model import simulate

O=Path(__file__).resolve().parents[2]/'outputs/admission_capacity'

def main():
    inp=json.loads((O/'20260914_recovery_progress_model_r01/input.json').read_text())
    initial,history=inp['state'],inp['history']
    base=O/'20260914_staged_store_probe_r01';results={};models={}
    for arm in ('off','on'):
        root=base/f'readback/results/save-{arm}'
        raw=json.loads((root/'raw.json').read_text())
        action=json.loads((root/'selective-store.json').read_text())
        plan=next(e for e in action['events'] if e['event']=='prepare')
        canonical=lambda rid:'measured/'+raw['internal_to_source'][rid]
        victim,target=canonical(plan['victim']),canonical(plan['target'])
        # Check the full observed initial state, not just chosen victim/target.
        before=raw['memory_trace'][329]['before']
        assert before['pool']['free_blocks']==initial['free']
        for rid,r in before['requests'].items():
            q=initial['requests'][canonical(rid)]
            assert (r['computed_tokens'],r['prompt_tokens'],r['output_tokens'],r['block_counts'])==(q['computed'],q['prompt'],q['output'],[len(q['blocks'])])
        assert [canonical(r) for r in before['running_ids']]==initial['running']
        events={};observations=[]
        if arm=='on':
            trans=json.loads((root/'offload-events.json').read_text());starts=[r['host_start_perf_counter_s'] for r in raw['memory_trace']]
            for event in trans['completed_jobs']:
                for job in event['jobs']:
                    if job['is_store']:continue
                    rid=canonical(job['request']);assert rid==victim
                    n=bisect_right(starts,event['time_s'])-1
                    assert raw['memory_trace'][n]['host_end_perf_counter_s']<=event['time_s']<starts[n+1]
                    events.setdefault(n,[]).append(rid)
                    observations.append(dict(job=job['job_id'],completed_call=n,eligible_call=n+1))
        pred=simulate(initial,history,'native',AbsenceRotation,RequestView,
            native_preemptions={330:victim},staged_recovery_targets={330:target},
            saved_prefixes={victim:plan['saved_tokens']} if arm=='on' else None,
            observed_load_completions=events if arm=='on' else None)
        first=None
        for row in pred['trace']:
            n=row['step']
            if n>=len(raw['scheduler_steps']):first=dict(step=n,reason='extra_model_step');break
            actual={f"measured/{q['request_id']}":dict(computed=q['scheduled_start_computed'],tokens=q['scheduled_tokens'],output=q['output_tokens_before']) for q in raw['scheduler_steps'][n]['scheduled']}
            matched=dict(schedule=list(row['scheduled'].items())==list(actual.items()),free=row['free_after_schedule']==raw['memory_trace'][n]['after']['pool']['free_blocks'],outputs=len(row['outputs'])==raw['engine_steps'][n]['new_output_tokens'])
            if not all(matched.values()):first=dict(step=n,checks=matched);break
        results[arm]=dict(first_mismatch=first,matched_steps=len(pred['trace']) if first is None else first['step']-329,last_step=pred['last_step'],actual_last_step=len(raw['engine_steps'])-1,observations=observations)
        models[arm]=pred
    original=json.loads((O/'20260914_recovery_progress_model_r01/predictions.json').read_text())
    regression={a:simulate(initial,history,a,AbsenceRotation,RequestView)==original[a] for a in ('least','most','defer')}
    assert all(regression.values())
    diff={rid:models['on']['completed'][rid]-end for rid,end in models['off']['completed'].items()}
    result=dict(status='OBSERVED_TRANSITION_CHECK',arms=results,completion_step_on_minus_off=diff,legacy_regression=regression,scope='Complete initial state matched. Saved arm consumes observed future completion notification only for state-machine validation. No transfer prediction, wall timing or unexecuted-policy benefit.')
    with (base/'structural_replay.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
