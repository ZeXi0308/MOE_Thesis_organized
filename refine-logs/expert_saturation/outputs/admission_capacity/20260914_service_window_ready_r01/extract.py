"""Observed notification/state counterexample; no future event is used by a policy."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / '20260914_selective_store_once_r01/readback/results/selective-on'
raw = json.loads((SOURCE / 'raw.json').read_text())
events = json.loads((SOURCE / 'offload-events.json').read_text())
memory = raw['memory_trace']
rows = []
for completion in events['completed_jobs']:
    for job in completion['jobs']:
        if job['is_store']:
            continue
        timestamp = completion['time_s']
        candidates = [k for k in range(len(memory)-1)
                      if memory[k]['host_end_perf_counter_s'] < timestamp
                      < memory[k+1]['host_start_perf_counter_s']]
        assert len(candidates) == 1
        k = candidates[0]
        rid = job['request']
        state0 = memory[k]['before']['requests'][rid]
        state1 = memory[k+1]['before']['requests'][rid]
        dispatch = [[r for r in raw['scheduler_steps'][i]['scheduled']
                     if r['internal_request_id'] == rid] for i in (k, k+1)]
        assert not dispatch[0] and len(dispatch[1]) == 1
        rows.append(dict(job=job['job_id'], request=rid, completed_after_schedule_call=k,
            earliest_native_promotion_call=k+1, completion_perf_counter_s=timestamp,
            previous_schedule_end_perf_counter_s=memory[k]['host_end_perf_counter_s'],
            next_schedule_start_perf_counter_s=memory[k+1]['host_start_perf_counter_s'],
            before_previous_call=state0, before_next_call=state1,
            same_target_numeric_state=state0==state1,
            actual_next_dispatch=dispatch[1][0]))
result = dict(status='OBSERVED_SOURCE_CONTRACT_COUNTEREXAMPLE', source=str(SOURCE), cases=rows,
    scope='Completion timestamps only locate already observed native call boundaries. They do not predict another action, define a load-delay bound or supply C_remaining. Memory snapshots lack request.status; readiness follows the installed source contract plus observed completion receipt.')
with (ROOT / 'counterexample.json').open('x') as f:
    json.dump(result, f, indent=2)
    f.write('\n')
print(json.dumps(result))
