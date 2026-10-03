#!/usr/bin/env python3
"""One actual step only: full recovery reservation plus one token per ready resident."""
import ast
from bisect import bisect_right
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parents[1]
SOURCE = BUNDLE / 'preparation/pkg/ltr_recompute_native.py'
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
nodes = [n for n in ast.parse(SOURCE.read_text()).body
         if getattr(n, 'name', None) in ('Candidate', 'Plan', 'plan')]
ns = dict(dataclass=dataclass, __name__=__name__)
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), ns)
out = dict(status='STRUCTURAL_ONE_STEP_DIAGNOSTIC', step=406,
           source_sha256=sha(SOURCE), cells=[])
for block in (0, 1):
    label = f'block{block}-d6-restore-on'
    folder = BUNDLE / 'execution/readback/results' / label
    raw = read(folder / 'raw.json')
    decisions = read(folder / 'component-decisions.json')
    d, m = decisions[406], raw['memory_trace'][406]['before']
    aliases = raw['internal_to_source']
    requests = {q['request_id']: q for q in raw['requests']}
    rows = [ns['Candidate'](rid, d['priorities'][rid], requests[aliases[rid]]['arrival_s'],
             rid in m['running_ids'], v['block_counts'][0],
             (v['prompt_tokens'] + v['output_tokens'] + 15)//16,
             v['prompt_tokens'] + v['output_tokens'] - v['computed_tokens'])
            for rid, v in m['requests'].items()]
    active = [r for r in rows if r.priority == -2]
    assert len(active) == 1
    target = active[0]
    ready = [r for r in rows if r.resident and r.pending_tokens == 1 and r != target]
    assert len(ready) == 30 and target.owned_blocks == 63 and target.history_blocks == 205
    free = m['pool']['free_blocks']
    original = ns['plan'](rows, free, 1024, 32, 0, packing='rank_prefix')
    assert original.tokens == d['actual_scheduled'] == {target.request_id: 1024}
    assert original.victims == d['victims'] == []
    # Existing pure planner's chunk argument represents a ONE-STEP 994 cap here.
    # No execution config is modified. Every other selected candidate has pending=1,
    # so this call checks the proposed split without implementing a new scheduler.
    capped = ns['plan'](rows, free, 1024, 32, 1024-len(ready), packing='rank_prefix')
    expected = {target.request_id: 994, **{r.request_id: 1 for r in ready}}
    assert capped.tokens == expected and capped.victims == []
    growth = {aliases[r.request_id]: max(0, r.history_blocks-r.owned_blocks) for r in ready}
    assert free == 148 and sum(growth.values()) == 3 and capped.free_after_reservation == 3
    events = []
    for k in range(405, 412):
        dec, mem = decisions[k], raw['memory_trace'][k]['before']
        events.append(dict(step=k, free_before=mem['pool']['free_blocks'],
            target_tokens=dec['actual_scheduled'].get(target.request_id, 0),
            ready_selected=sum(r.request_id in dec['actual_scheduled'] for r in ready),
            target_output_before=mem['requests'][target.request_id]['output_tokens'],
            target_priority=dec['priorities'][target.request_id],
            victims=[aliases[i] for i in dec['victims']],
            free_after=dec['free_after'], released_obligations=dec['obligation_begin']['released']))
    outputs = []
    off_folder = folder.parent / f'block{block}-d6-restore-off'
    off = read(off_folder / 'raw.json')
    off_requests = {q['request_id']: q for q in off['requests']}
    calls = {e['scheduler_step_start']: e for e in raw['engine_steps']}
    off_calls = {e['scheduler_step_start']: e for e in off['engine_steps']}
    for r in ready:
        rid = aliases[r.request_id]
        n = m['requests'][r.request_id]['output_tokens']
        q, oq = requests[rid], off_requests[rid]
        assert bisect_right(q['token_times_s'], calls[405]['returned_s']) == n
        assert bisect_right(oq['token_times_s'], off_calls[405]['returned_s']) == n
        assert q['token_times_s'][n] == calls[408]['returned_s']
        assert oq['token_times_s'][n] == off_calls[406]['returned_s']
        outputs.append(dict(request_id=rid, output_before=n,
            on_next_output_step=408, off_next_output_step=406,
            on_gap_s=q['token_times_s'][n]-q['token_times_s'][n-1],
            off_gap_s=oq['token_times_s'][n]-oq['token_times_s'][n-1]))
    out['cells'].append(dict(label=label, raw_sha256=sha(folder/'raw.json'),
        target=aliases[target.request_id], free_blocks=free, target_remaining_history_blocks=142,
        ready_count=len(ready), ready_growth_blocks=sum(growth.values()),
        ready_growing={r: n for r, n in growth.items() if n},
        diagnostic_tokens={aliases[i]: n for i, n in capped.tokens.items()},
        diagnostic_victims=capped.victims, diagnostic_reservation_slack=3,
        actual_events=events, ready_output_gaps=outputs))
out['boundary'] = ('Only step 406 is replayed from each actual on prestate; the 994 cap is a '
    'diagnostic input to the unchanged frozen pure planner. No alternate state is advanced, '
    'no future trace supplies a decision, and no latency saving is predicted. 406/407 exclude '
    '30 ready residents; 408 actually serves them. The later 411 eviction occurs after the '
    'first output and obligation release, and is distinct from this execution-token exclusion. '
    'Global recompute delta 21973 cannot be attributed in full to this single dispatch.')
(HERE/'first_dispatch.json').write_text(json.dumps(out, indent=2)+'\n')
print(json.dumps(dict(status=out['status'], cells=len(out['cells']),
      token_split='994+30', victims=0, reservation_slack=3)))
