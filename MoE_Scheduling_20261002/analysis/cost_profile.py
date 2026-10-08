"""Read one completed episode; descriptive costs only, no causal replay or fit."""
import collections
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'results_opt_r01/00_default24'
OUT = Path(__file__).with_suffix('.json')

def distribution(values):
    a = sorted(values)
    if not a:
        return {'n': 0}
    return dict(n=len(a), min=a[0], p10=a[max(0, math.ceil(.1*len(a))-1)],
                p50=a[math.ceil(.5*len(a))-1], p90=a[math.ceil(.9*len(a))-1],
                p95=a[math.ceil(.95*len(a))-1], max=a[-1], mean=sum(a)/len(a))

raw = json.loads((SRC/'raw.json').read_text())
summary = json.loads((SRC/'pager/summary.json').read_text())
assert raw['status'] == 'COMPLETE' and raw['error'] is None
layers = collections.defaultdict(list)
for line in (SRC/'pager/calls.jsonl').open():
    c = json.loads(line)
    if c['measurement'] and not c['validation_run']:
        assert c['status'] == 'complete' and c['context']['phase'] == 'measurement'
        layers[c['context']['step_id']].append(c)
engines = {}
for e in raw['engine_calls']:
    assert e['returned'] and e['scheduler_step_stop'] == e['scheduler_step_start'] + 1
    assert e['scheduler_step_start'] not in engines
    engines[e['scheduler_step_start']] = e
rows, group_rates, active = [], collections.defaultdict(list), collections.defaultdict(list)
for s in raw['scheduler_steps']:
    i, cs, e = s['step'], layers[s['step']], engines[s['step']]
    assert len(cs) == 16 and len({c['layer_name'] for c in cs}) == 16
    p = sum(r['prefill_tokens'] for r in s['scheduled'])
    d = sum(r['decode_tokens'] > 0 for r in s['scheduled'])
    assert d == s['decode_requests'] and p + sum(r['decode_tokens'] for r in s['scheduled']) == s['total_scheduled_tokens']
    kind = 'mixed' if p and d else 'pure_prefill' if p else 'pure_decode'
    pf_counts = []
    for c in cs:
        ctx, routes = c['context'], c['row_topk_experts']
        assert ctx['row_request_order_verified'] and ctx['valid_row_start'] == 0
        meta = ctx['rows']; assert len(meta) == ctx['valid_row_stop'] == s['total_scheduled_tokens']
        ids = {r['internal_request_id'] for r in s['scheduled']}
        assert all(r['internal_request_id'] in ids for r in meta)
        assert sum(r['computed_position'] < r['prompt_tokens'] for r in meta) == p
        if p:
            pf_counts.append(len({x for j,r in enumerate(meta) if r['computed_position'] < r['prompt_tokens'] for x in routes[j]}))
        active[kind].append(len(c['active_experts']))
        assert c['weight_copy_bytes'] == sum(g['weight_copy_bytes'] for g in c['groups'])
        for g in c['groups']:
            if g['weight_copy_bytes']:
                assert g['load_cuda_span_ms'] > 0
                group_rates[kind].append(g['weight_copy_bytes']/g['load_cuda_span_ms']/1e6)
    span = sum(c['load_cuda_span_ms'] for c in cs)
    rows.append(dict(step=i, kind=kind, decode_requests=d, prefill_tokens=p,
        engine_wall_ms=(e['return_s']-e['start_s'])*1000,
        load_bytes=sum(c['weight_copy_bytes'] for c in cs), groups=sum(c['group_count'] for c in cs),
        ensure_span_ms=span, prefill_only_route_full64_layers=sum(n == 64 for n in pf_counts),
        all_rows_full64_layers=sum(len(c['active_experts']) == 64 for c in cs)))
assert set(layers) == set(engines) == {r['step'] for r in rows}
assert sum(r['load_bytes'] for r in rows) == summary['measurement']['weight_copy_bytes']
by_kind = {}
for kind in ('pure_prefill', 'mixed', 'pure_decode'):
    rs = [r for r in rows if r['kind'] == kind]
    by_kind[kind] = dict(steps=len(rs), engine_wall_ms=distribution([r['engine_wall_ms'] for r in rs]),
        load_bytes=distribution([r['load_bytes'] for r in rs]), groups=distribution([r['groups'] for r in rs]),
        decode_requests=sorted({r['decode_requests'] for r in rs}), prefill_tokens=distribution([r['prefill_tokens'] for r in rs]),
        effective_ensure_GBps=distribution(group_rates[kind]), layer_active_experts=distribution(active[kind]),
        full64_prefill_route_layers=sum(r['prefill_only_route_full64_layers'] for r in rs),
        prefill_bearing_layers=16*sum(r['prefill_tokens'] > 0 for r in rs))
comparisons = []
for width in sorted({r['decode_requests'] for r in rows if r['kind'] == 'mixed'}):
    arms = {k:[r for r in rows if r['kind'] == k and r['decode_requests'] == width] for k in ('mixed','pure_decode')}
    comparisons.append(dict(decode_width=width, same_width_available=all(arms.values()),
        **{k:dict(steps=[r['step'] for r in v], engine_wall_ms=distribution([r['engine_wall_ms'] for r in v]), load_bytes=distribution([r['load_bytes'] for r in v])) for k,v in arms.items()}))
paths = [SRC/'raw.json', SRC/'pager/calls.jsonl', SRC/'pager/summary.json']
result = dict(schema='observed-expert-step-cost-v1', source={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    scope='One completed forced-32-output-token GSM8K episode; no counterfactual or causal effect; correlated steps.',
    join='Measurement step_id to scheduler step; one returned engine call and 16 unique layers per step; identities and row counts checked.',
    load_span_scope=summary['load_span_scope'], bytes_scope=summary['bytes_scope'], quantiles='nearest rank; decimal GB/s; positive-byte groups only',
    calibration_rule='Offline summaries include later steps: not an online predictor result. Runtime calibration must use only previously completed steps; cache, context and queue state are not matched by decode width.',
    totals=dict(steps=len(rows), layer_calls=sum(map(len,layers.values())), load_bytes=sum(r['load_bytes'] for r in rows), groups=sum(r['groups'] for r in rows), engine_wall_ms=sum(r['engine_wall_ms'] for r in rows), episode_ms=1000*raw['observation_end_s']),
    by_kind=by_kind, same_decode_width_observations=comparisons, steps=rows)
with OUT.open('x') as f:
    json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')
print(json.dumps({'output':str(OUT), 'totals':result['totals'], 'by_kind':by_kind, 'shared_widths':[c['decode_width'] for c in comparisons if c['same_width_available']]}, ensure_ascii=False))
