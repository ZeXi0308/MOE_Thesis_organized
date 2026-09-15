"""Given the actual staged recovery target, locate remaining compute contention.

Read-only reuse of diagnostic off/on; no new target, victim, trajectory, timing
prediction or performance table. Resolved scheduled_start_computed is observed
inside native execution, not a pre-load predictor or free external KV.
"""
import argparse
from bisect import bisect_right
from collections import Counter
import hashlib
import json
from pathlib import Path

import recovery_execution_share as share


def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def compute_envelope(remaining, peers, budget=1024):
    if remaining <= 0 or not 0 <= peers < budget:
        raise ValueError('requires a ready recovery and pending-one decode peers')
    calls = (remaining + budget - 1)//budget
    minimum = remaining - (calls-1)*budget
    return dict(remaining_positions=remaining,peer_positions=peers,budget=budget,
        compute_only_min_calls=calls,preserve_calls_min_positions=minimum,
        all_work_fits_this_call=remaining+peers<=budget,
        minimum_target_plus_all_peers_fits=minimum+peers<=budget,
        minimum_peer_opportunities_withheld_over_min_calls=max(0, remaining+calls*peers-calls*budget))


def inspect_cell(root, label):
    folder=root/'execution_weste_26862/readback/results'/label
    raw=read(folder/'raw.json'); data=read(folder/'selective-store.json')
    assert raw['status']=='COMPLETE' and data['diagnostic']
    if any(x['event']=='target_terminal' for x in data['events']):
        raise ValueError('Terminal recovery episode unsupported by this first-output probe; '
                         'retain the cell and use actual output/finish boundaries, never a later episode release')
    assert len(raw['memory_trace'])==len(raw['scheduler_steps'])==data['schedule_calls']
    aliases=raw['internal_to_source']; requests={r['request_id']:r for r in raw['requests']}
    calls={k:c for c in raw['engine_steps'] for k in range(c['scheduler_step_start'],c['scheduler_step_end'])}
    snapshots={x['step']:x for x in data['eligibility_snapshots']}
    commits=[x for x in data['events'] if x['event']=='commit_check' and x['reason']=='READY']
    rows=[]; all_calls=[]
    for event in commits:
        start,target,victim=event['step'],event['target'],event['victim']
        release=next(x['step'] for x in data['events'] if x['event']=='target_new_output'
                     and x['step']>start and x['request']==target)
        outputs=raw['memory_trace'][start]['before']['requests'][target]['output_tokens']
        request=requests[aliases[target]]; timestamp=request['token_times_s'][outputs]
        final=min(k for k,c in calls.items() if k>=start and c['returned_s']==timestamp)
        assert final<release and bisect_right(request['token_times_s'],calls[start]['start_s'])==outputs
        observed=[]
        for k in range(start,release):
            step=raw['scheduler_steps'][k]; before=raw['memory_trace'][k]['before']; snap=snapshots[k]
            actual={s['internal_request_id']:s for s in step['scheduled']}
            target_row=actual.get(target); peers=[rid for rid in before['running_ids'] if rid not in (target,victim if k==start else None)]
            assert all(before['requests'][r]['computed_tokens']==before['requests'][r]['prompt_tokens']+
                       before['requests'][r]['output_tokens']-1 for r in peers)
            peer_outputs={aliases[r]:sum(calls[k]['start_s']<t<=calls[k]['returned_s'] for t in requests[aliases[r]]['token_times_s'])
                          for r in peers}
            row=dict(step=k,target_scheduled=bool(target_row),peer_count=len(peers),
                target_status_before=snap['requests'][target]['status'],
                peer_new_outputs=sum(peer_outputs.values()),held_peers=[aliases[r] for r in peers if r not in actual],
                actual_total_positions=sum(s['scheduled_tokens'] for s in actual.values()))
            assert row['actual_total_positions']<=1024
            if target_row:
                state=before['requests'][target]
                remaining=state['prompt_tokens']+state['output_tokens']-target_row['scheduled_start_computed']
                row.update(compute_envelope(remaining,len(peers)),
                    target_positions=target_row['scheduled_tokens'],
                    resolved_prefix_adjustment=target_row['computed_adjustment'])
                assert 0<row['target_positions']<=remaining
                row['target_completed_pending_work']=row['target_positions']==remaining
                # Only replay from a genuinely resident before-state. Native load
                # transitions and commit-time victim release are not pre-state plans.
                if k>start and target in before['running_ids'] and not target_row['computed_adjustment']:
                    def view(rid):
                        r=snap['requests'][rid]
                        return share.Request(rid,r['prompt']+r['output'],r['computed'],r['held_blocks'],r['output'],r['max_output']-r['output'])
                    st=share.State(view(target),tuple(view(r) for r in peers),before['pool']['free_blocks'])
                    simple=share.allocate(st); bounded=share.allocate(st,'preserve_calls')
                    assert simple['scheduled']=={r:s['scheduled_tokens'] for r,s in actual.items()}
                    row.update(resident_model_exact=True,preserve_calls_changes_map=bounded['scheduled']!=simple['scheduled'])
            observed.append(row); all_calls.append(row)
        executed=[x for x in observed if x['target_scheduled']]
        assert executed and executed[-1]['step']==final and executed[-1]['target_completed_pending_work']
        rows.append(dict(target=aliases[target],victim=aliases[victim],commit_step=start,
            first_new_output_step=final,observed_release_step=release,initial_outputs=outputs,
            commit_to_new_output_s=timestamp-calls[start]['start_s'],
            ready_compute_calls=len(executed),no_target_compute_calls=len(observed)-len(executed),
            first_ready_remaining=executed[0]['remaining_positions'],calls=observed))
    ready=[r for r in all_calls if r['target_scheduled']]
    resident=[r for r in ready if r.get('resident_model_exact')]
    summary=dict(commits=len(rows),fulfilled=len(rows),protected_calls=len(all_calls),ready_compute_calls=len(ready),
        no_target_compute_calls=sum(not r['target_scheduled'] for r in all_calls),
        no_compute_statuses=dict(Counter(r['target_status_before'] for r in all_calls if not r['target_scheduled'])),
        one_compute_call_recoveries=sum(r['ready_compute_calls']==1 for r in rows),
        first_ready_pending_positions=dict(sorted(Counter(r['first_ready_remaining'] for r in rows).items())),
        ready_calls_all_work_fits=sum(r['all_work_fits_this_call'] for r in ready),
        ready_calls_minimum_plus_peers_exceeds_budget=sum(not r['minimum_target_plus_all_peers_fits'] for r in ready),
        ready_calls_completing_pending=sum(r['target_completed_pending_work'] for r in ready),
        pure_resident_model_calls=len(resident),resident_allocation_exact=len(resident),
        preserve_calls_changes_map=sum(r['preserve_calls_changes_map'] for r in resident),
        held_peer_calls=sum(len(r['held_peers']) for r in all_calls),
        peer_new_outputs_during_protection=sum(r['peer_new_outputs'] for r in all_calls))
    return dict(label=label,summary=summary,recoveries=rows,
                dependencies={str(folder/n):sha(folder/n) for n in ('raw.json','selective-store.json')})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--service-bundle',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--labels',nargs='+',default=['diag-off','diag-on'])
    p.add_argument('--full-service-reference',type=Path)
    a=p.parse_args();assert not a.output.exists()
    cells=[inspect_cell(a.service_bundle,label) for label in a.labels]
    reference=a.full_service_reference or a.service_bundle/'analysis/analysis.json'
    result=dict(status='MEASUREMENT_ONLY',question=__doc__,cells=cells,
        source_sha256={str(Path(__file__)):sha(Path(__file__)),str(Path(share.__file__)):sha(Path(share.__file__))},
        full_service_reference=dict(path=str(reference),
            sha256=sha(reference) if reference.exists() else None,recomputed_here=False),
        requested_cells=a.labels,
        scope='All committed targets in the requested complete diagnostic cells; no unprotected recovery scope is implied. '
              'No excluded ready call is silently treated as model-qualified. Diagnostic times are not performance repeats. '
              'Compute demand counts do not predict batch latency or prove zero wall-time headroom. '
              'No alternative trajectory, controller, new GPU experiment, EOS inference or global waiting bound.')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps([dict(label=c['label'],**c['summary']) for c in cells]))


if __name__=='__main__':main()
