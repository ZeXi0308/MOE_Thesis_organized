"""Replay actual worker completions; validate transitions, never claim prediction."""
from bisect import bisect_right
import importlib.util
import json
from pathlib import Path
import sys
from recovery_progress_model import simulate

O=Path(__file__).resolve().parents[2]/'outputs/admission_capacity'


def main():
    root=O/'20260914_load_ready_contract_r01'
    output=root/'event_replay.json'
    if output.exists():raise FileExistsError(output)
    spec=importlib.util.spec_from_file_location('rotation_replay',O/'20260914_d6_action_branches_r01/pkg/absence_rotation.py')
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    inp=json.loads((O/'20260914_recovery_progress_model_r01/input.json').read_text())
    initial,history=inp['state'],inp['history']
    victim=min(initial['running'],key=lambda r:initial['requests'][r]['output'])
    checks=[]
    for block in (0,1):
        path=O/f'20260914_selective_store_repeat_r01/readback/results/block{block}-on'
        raw=json.loads((path/'raw.json').read_text())
        transfer=json.loads((path/'offload-events.json').read_text())
        starts=[r['host_start_perf_counter_s'] for r in raw['memory_trace']]
        assert starts==sorted(starts)
        events={};observations=[]
        for event in transfer['completed_jobs']:
            for job in event['jobs']:
                if job['is_store']:continue
                rid='measured/'+raw['internal_to_source'][job['request']]
                assert rid==victim
                n=bisect_right(starts,event['time_s'])-1
                assert 0<=n<len(starts)-1
                assert raw['memory_trace'][n]['host_end_perf_counter_s']<=event['time_s']<starts[n+1]
                events.setdefault(n,[]).append(rid)
                observations.append(dict(job=job['job_id'],completed_in_call=n,first_eligible_call=n+1))
        pred=simulate(initial,history,'native',m.AbsenceRotation,m.RequestView,
                      native_preemptions={329:victim},saved_prefixes={victim:3296},
                      observed_load_completions=events)
        first=None
        for row in pred['trace']:
            n=row['step'];s=raw['scheduler_steps'][n]
            actual={f"measured/{q['request_id']}":dict(computed=q['scheduled_start_computed'],tokens=q['scheduled_tokens'],output=q['output_tokens_before']) for q in s['scheduled']}
            matched=dict(schedule=list(row['scheduled'].items())==list(actual.items()),
                         free=row['free_after_schedule']==raw['memory_trace'][n]['after']['pool']['free_blocks'],
                         outputs=len(row['outputs'])==raw['engine_steps'][n]['new_output_tokens'])
            if not all(matched.values()):first=dict(step=n,checks=matched);break
        checks.append(dict(block=block,worker_completion_observations=observations,
                           last_step=pred['last_step'],actual_last_step=len(raw['engine_steps'])-1,
                           first_mismatch=first,matched_steps=len(pred['trace']) if first is None else first['step']-329))
    original=json.loads((O/'20260914_recovery_progress_model_r01/predictions.json').read_text())
    regression={a:simulate(initial,history,a,m.AbsenceRotation,m.RequestView)==original[a] for a in ('least','most','defer')}
    assert all(regression.values())
    result=dict(status='OBSERVED_EVENT_CONDITIONED_VALIDATION',checks=checks,legacy_regression=regression,
                scope='Actual future worker completions supplied as replay events, consumed only at the end of their observed call. '
                      'Tests state-transition accounting and notification cutoff. Does NOT predict transfer latency, '
                      'evaluate unexecuted policies, establish a policy benefit, or validate quality.')
    output.write_text(json.dumps(result,indent=2)+'\n')
    (root/'model_source.py').write_text(Path(__file__).with_name('recovery_progress_model.py').read_text())
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
