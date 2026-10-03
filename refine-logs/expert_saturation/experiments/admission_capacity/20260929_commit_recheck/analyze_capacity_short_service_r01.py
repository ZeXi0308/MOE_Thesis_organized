"""Locate the observed 1–2-token repeat-preemption cycles, without changing policy."""
from bisect import bisect_right
from collections import Counter
import argparse
import json
from pathlib import Path
from analyze_capacity_victim_costs_r01 import dist

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--session',type=Path,default=HERE/'moe-a-capacity-victim-session-r01-20261001')
parser.add_argument('--output',type=Path,default=HERE/'A_CAPACITY_SHORT_SERVICE_LOCATION_R01_20261001.json')
args = parser.parse_args()
archive = next(args.session.glob('*_on/archive'))
raw = json.loads((archive/'raw.json').read_text())
store = json.loads((archive/'selective-store.json').read_text())
requests = {r['internal_request_id']:r for r in raw['requests']}
events = store['events']
origin = raw['measurement_origin_perf_counter_s']
forced = {(e['step'],e['victim']):e for e in events if e['event']=='commit_check' and e['reason']=='READY'}
capacity = {(e['step'],e['new_victim']):e for e in events if e['event']=='capacity_victim_commit'}
preemptions = {(e['engine_call_index'],e['internal_request_id']):e for e in raw['preemption_events']}
gap_order = sorted(((b-a,rid,a,b) for rid,r in requests.items()
                   for a,b in zip(r['token_times_s'],r['token_times_s'][1:])),reverse=True)
gap_rank = {(rid,a,b):i+1 for i,(_,rid,a,b) in enumerate(gap_order)}
previous, rows = {}, []
for current in raw['preemption_events']:
    rid = current['internal_request_id']
    earlier = previous.get(rid)
    previous[rid] = current
    if earlier is None:
        continue
    delta = current['last_returned_output_count']-earlier['last_returned_output_count']
    if delta not in (1,2):
        continue
    times = requests[rid]['token_times_s']
    first = times[bisect_right(times,earlier['method_entered_s'])]
    recoveries = [(e,preemptions[key]) for key,e in forced.items() if e['target']==rid
        and earlier['method_entered_s']<preemptions[key]['method_entered_s']<first]
    if len(recoveries)>1:
        raise ValueError('Recovery attribution is ambiguous')
    recovery = recoveries[0][0] if recoveries else None
    kind = ('capacity_replacement' if (recovery['step'],recovery['victim']) in capacity
            else 'ordinary_rotation') if recovery else 'no_adapter_recovery_commit_observed'
    key = (current['engine_call_index'],rid)
    releases = [e for e in events if e['event']=='target_new_output' and e['request']==rid
        and first<=e['host_perf_counter_s']-origin<current['method_entered_s']]
    after = bisect_right(times,current['method_entered_s'])
    gap_start, gap_end = times[after-1],times[after]
    rows.append(dict(request_id=current['request_id'], new_tokens_between_preemptions=delta,
        earlier_preemption_step=earlier['engine_call_index'], later_preemption_step=current['engine_call_index'],
        later_preemption_cause='adapter_forced' if key in forced else 'native_no_adapter_commit',
        recovery_kind=kind, recovery_commit_step=recovery['step'] if recovery else None,
        first_output_after_earlier_preemption_s=first,
        protection_release_events=[dict(step=e['step'],time_s=e['host_perf_counter_s']-origin) for e in releases],
        later_preemption_s=current['method_entered_s'],
        first_output_to_later_preemption_ms=1000*(current['method_entered_s']-first),
        later_preemption_generation_gap=dict(start_s=gap_start,end_s=gap_end,gap_s=gap_end-gap_start,
            rank_among_all_on_generation_gaps=gap_rank[(rid,gap_start,gap_end)])))
affected = {r['request_id'] for r in rows}
tail = [dict(request_id=requests[rid]['request_id'],gap_s=gap,
             request_in_short_service_set=requests[rid]['request_id'] in affected,
             short_service_later_preemption_in_gap=any(r['request_id']==requests[rid]['request_id']
                and start<r['later_preemption_s']<end for r in rows)) for gap,rid,start,end in gap_order[:3]]
summary = dict(short_service_pairs=len(rows),distinct_requests=len(affected),
    output_count_histogram=dict(Counter(r['new_tokens_between_preemptions'] for r in rows)),
    later_preemption_causes=dict(Counter(r['later_preemption_cause'] for r in rows)),
    preceding_recovery_kinds=dict(Counter(r['recovery_kind'] for r in rows)),
    pairs_with_one_observed_protection_release=sum(len(r['protection_release_events'])==1 for r in rows),
    release_to_later_preemption_step_difference=dict(Counter(r['later_preemption_step']-r['protection_release_events'][0]['step']
        for r in rows if len(r['protection_release_events'])==1)),
    first_output_to_later_preemption_ms=dist([r['first_output_to_later_preemption_ms'] for r in rows]),
    ensuing_generation_gap_s=dist([r['later_preemption_generation_gap']['gap_s'] for r in rows]),
    ensuing_gaps_at_least_two_seconds=sum(r['later_preemption_generation_gap']['gap_s']>=2 for r in rows),
    top_three_gap_overlap=tail)
result = dict(source=str(archive),summary=summary,observations=rows,
    interpretation='Rows with adapter recovery and protection-release events locate a concrete first-output release boundary; rows without those events are reported separately. This does not establish that extending protection improves full-request performance or explains the largest observed gaps.',
    limits=['Native cause means no matching adapter forced commit at that request and step; full scheduling/allocation snapshots are absent.',
            'No claim that these 23 cycles explain the entire tail; exact overlap with the three largest observed gaps is reported.',
            'Any protection extension can delay other requests or consume capacity; benefit remains unmeasured.'])
output = args.output
with output.open('x') as handle:
    json.dump(result,handle,indent=2)
    handle.write('\n')
print(json.dumps(summary,indent=2))
