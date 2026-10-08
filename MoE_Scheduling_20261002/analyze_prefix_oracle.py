"""Fixed-call, single-layer expert-prefix opportunity; no latency/trace replay claim."""
import collections
import json
import sys
from pathlib import Path

arm, output = map(Path, sys.argv[1:3])
root = Path(__file__).resolve().parent
readiness = json.loads((arm / 'request_readiness.json').read_text())
raw = json.loads((arm / 'raw.json').read_text())
workload = json.loads((root / 'inputs/olmoe_gsm8k16/workload.json').read_text())
source_order = {r['request_id']: i for i, r in enumerate(workload['source_requests'])}
ages = {r['internal_request_id']: (r['arrival_s'], source_order[r['request_id']]) for r in raw['requests']}
pager_rows = [json.loads(line) for line in (arm / 'pager/calls.jsonl').open()]
pager = {r['call_id']: r for r in pager_rows}
assert len(pager) == len(pager_rows)
bits = lambda experts: sum(1 << e for e in set(experts))
ids = lambda mask: [e for e in range(mask.bit_length()) if mask >> e & 1]
records = []
for call in readiness['calls']:
    if not (call.get('completed') and call.get('mapping_verified') and call.get('decode_rows')):
        continue
    p, rows = pager[call['call_id']], call['decode_rows']
    assert p['status'] == 'complete' and p['measurement'] and not p['validation_run']
    assert p['layer_name'] == call['layer_name'] and p['context']['step_id'] == call['step_id']
    assert set(p['entry_resident_experts']) == set(call['entry_resident_experts'])
    assert len(rows) <= 16 and len({r['internal_request_id'] for r in rows}) == len(rows)
    assert all(p['row_topk_experts'][r['row_index']] == r['topk'] for r in rows)
    resident = bits(call['entry_resident_experts'])
    remaining = [bits(r['topk']) & ~resident for r in rows]
    freq = collections.Counter(e for row in p['row_topk_experts'] for e in row)
    cold_order = sorted((e for e in freq if not (resident >> e & 1)), key=lambda e: (-freq[e], e))
    age_order = sorted(range(len(rows)), key=lambda i: ages[rows[i]['internal_request_id']])
    age_rank = {i: rank for rank, i in enumerate(age_order)}
    loaded_list = [e for g in call['groups'][:2] for e in g['loaded_experts']]
    assert len(loaded_list) == len(set(loaded_list)) and not (bits(loaded_list) & resident)
    assert all(g['weight_copy_bytes'] == len(g['loaded_experts']) * 12582912 for g in call['groups'])
    actual_load = bits(loaded_list)
    executed = bits(e for g in call['groups'][:2] for e in g['required_experts'])
    complete = lambda selected: sum(not (need & ~selected) for need in remaining)
    actual = sum(not (bits(r['topk']) & ~executed) for r in rows)
    assert actual == complete(actual_load)
    budgets = {'fixed24': min(24, len(cold_order)), 'matched_actual2': len(loaded_list)}
    unions, best = [0] * (1 << len(rows)), {name: (0, 0) for name in budgets}
    for subset in range(1 << len(rows)):
        if subset:
            bit = subset & -subset
            unions[subset] = unions[subset ^ bit] | remaining[bit.bit_length() - 1]
        for name, budget in budgets.items():
            if unions[subset].bit_count() <= budget and subset.bit_count() > best[name][0]:
                best[name] = (subset.bit_count(), unions[subset])
    result = {}
    for name, budget in budgets.items():
        high_load = bits(cold_order[:budget])
        oldest_order = list(dict.fromkeys(e for i in age_order for e in ids(remaining[i])))
        oldest_order += [e for e in cold_order if e not in oldest_order]
        oldest, greedy = bits(oldest_order[:budget]), 0
        while True:
            candidates = [i for i, need in enumerate(remaining) if need & ~greedy]
            if not candidates:
                break
            i = min(candidates, key=lambda i: ((remaining[i] & ~greedy).bit_count(), age_rank[i]))
            if (greedy | remaining[i]).bit_count() > budget:
                break
            greedy |= remaining[i]
        greedy |= bits([e for e in cold_order if not (greedy >> e & 1)][:budget - greedy.bit_count()])
        selected = dict(high_load=high_load, oldest=oldest, greedy=greedy, oracle=best[name][1])
        counts = {policy: complete(mask) for policy, mask in selected.items()}
        assert counts['oracle'] == best[name][0] and max(counts.values()) == counts['oracle']
        assert all(mask.bit_count() <= budget for mask in selected.values())
        result[name] = dict(budget=budget, complete=counts, loaded={k: ids(v) for k, v in selected.items()})
    records.append(dict(call_id=call['call_id'], step_id=call['step_id'], layer=call['layer_name'],
                        kind='pure_decode' if p['rows'] == len(rows) else 'mixed', width=len(rows),
                        total_rows=p['rows'], entry_complete=complete(0), actual2_complete=actual,
                        actual2_budget=len(loaded_list), policies=result))

def aggregate(rows):
    out = dict(calls=len(rows), decode_rows=sum(r['width'] for r in rows),
               entry_complete=sum(r['entry_complete'] for r in rows),
               actual2_complete=sum(r['actual2_complete'] for r in rows),
               actual2_budget_counts=dict(sorted(collections.Counter(r['actual2_budget'] for r in rows).items())))
    for name in ('fixed24', 'matched_actual2'):
        cs = [r['policies'][name]['complete'] for r in rows]
        out[name] = {policy: dict(complete_rows=sum(c[policy] for c in cs),
                     below_oracle_calls=sum(c[policy] < c['oracle'] for c in cs),
                     oracle_gap_rows=sum(c['oracle'] - c[policy] for c in cs))
                     for policy in ('high_load', 'oldest', 'greedy', 'oracle')}
        out[name]['greedy_vs_high_load_calls'] = dict(win=sum(c['greedy'] > c['high_load'] for c in cs),
            tie=sum(c['greedy'] == c['high_load'] for c in cs), lose=sum(c['greedy'] < c['high_load'] for c in cs))
    return out

groups = {'all': records, **{k: [r for r in records if r['kind'] == k] for k in ('pure_decode', 'mixed')}}
for width in sorted({r['width'] for r in records}):
    for kind in ('all', 'pure_decode', 'mixed'):
        groups[f'{kind}/width{width}'] = [r for r in records if r['width'] == width and (kind == 'all' or r['kind'] == kind)]
report = dict(scope='FIXED_SINGLE_LAYER_PREFIX_OPPORTUNITY_ONLY',
    assumptions=['All entry-resident contributions computed first; current-call known routes only.',
                 'Equal expert bytes, 12 MiB each; kernel batching and later contention unmodeled.',
                 'High-load counts ALL call rows including prefill; age ties use workload source order.',
                 'Greedy chooses least additional experts, then oldest; fills spare budget by high-load.',
                 'Exact subset oracle; fixed24 actual2 is reference only when budgets differ.',
                 'Rows are request-token-layer observations, not independent requests or service gains.'],
    summaries={k: aggregate(v) for k, v in groups.items() if v}, calls=records)
output.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report['summaries'], indent=2))
