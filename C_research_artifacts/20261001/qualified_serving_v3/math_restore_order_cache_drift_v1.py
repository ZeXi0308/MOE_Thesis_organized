#!/usr/bin/env python3
"""Narrow CPU audit of adjacent APC observations within one unresumed restore wait."""
from collections import Counter, defaultdict
import hashlib, json, lzma
from pathlib import Path
import sys
sys.dont_write_bytecode = True
from health_analyze_v2 import native_id_mapping
BASE = Path(__file__).resolve().parent
RUN = BASE/'math_restore_order_raw_v1_20261002/math1024-order-minrecompute-b4096/native'
OUT = BASE/'math_restore_order_cache_drift_v1.json'
hashes = {}
def read(name):
    p = RUN/name
    data = p.read_bytes() if p.exists() else lzma.decompress(Path(str(p)+'.xz').read_bytes())
    hashes[name] = hashlib.sha256(data).hexdigest()
    return json.loads(data)
source, sampling = read('source-input.json'), read('measured-native-sampling.json')
mapping, errors = native_id_mapping(sampling, {'measured/'+r['request_id'] for r in source['requests']})
assert not errors, errors
policy, gate, pressure = read('restore-cost-order-policy.json'), read('restore-cost-order-fixed-gate.json'), read('measured-pressure.json')
assert all(x['status']=='COMPLETE' for x in (policy,gate,pressure)) and policy['mode']=='min-recompute'
offset = policy['initial_scheduler_step']+1
calls = {c['call_id']:c for c in pressure['scheduler_calls']}
by_key, successful, gate_key = defaultdict(list), defaultdict(list), {}
for a in pressure['allocation_attempts']:
    if a['status_before']=='PREEMPTED': by_key[(a['call_id'],a['request_id'])].append(a)
for g in gate['decisions']:
    cid, rid = g['scheduler_step']-offset, g['request_id']
    assert (cid,rid) not in gate_key
    gate_key[(cid,rid)] = g
    rows = by_key[(cid,rid)]
    assert len(rows)==(1 if g['native_called'] else 0)
    if rows:
        a = rows[0]
        assert a['attempt_id'] in calls[cid]['attempt_ids'] and a['returned_none']==g.get('native_returned_none')
        assert a['free_blocks_before']==g['free_blocks'] and a['num_tokens']==g['candidate_known_tokens']
        assert a['arguments']['num_new_computed_tokens']==g['candidate_cached_tokens']
        if a['succeeded']: successful[rid].append((g['decision'],cid,a['attempt_id']))
assert sum(len(v) for v in by_key.values())==sum(g['native_called'] for g in gate['decisions'])
receipts = {r['decision']:r for r in policy['allocation_receipts']}
previous, observed, compared, loss_requests = {}, set(), set(), set()
counts = Counter(candidate_observations=0, adjacent_observations=0, changed_known_excluded=0,
    intervening_success_excluded=0, eligible_adjacent_pairs=0, decreased_pairs=0,
    increased_pairs=0, unchanged_pairs=0, gross_cached_tokens_lost=0, gross_cached_tokens_gained=0)
losses, last_boundary = [], -1
for d in policy['decisions']:
    cid = d['scheduler_step']-offset
    anchor = gate_key[(cid,d['selected_id'] or d['restore_head_id'])]
    boundary = anchor['decision']
    assert boundary>last_boundary; last_boundary=boundary
    if d['selected_id']: assert receipts[d['decision']]['fixed_gate_decision']==boundary
    else: assert anchor['policy_deferred']
    for c in d['candidates']:
        rid = c['request_id']; observed.add(rid); counts['candidate_observations']+=1
        now = dict(call_id=cid, policy_decision=d['decision'], before_fixed_gate_decision=boundary,
            known_tokens=c['candidate_known_tokens'], cached_tokens=c['candidate_cached_tokens'],
            known_recompute_tokens=c['known_recompute_tokens'], free_blocks=c['free_blocks'],
            full_plus_one_blocks=c['candidate_full_plus_one_blocks'], required_with_margin=c['required_blocks'],
            fits_fixed48=not c['policy_deferred'], original_queue_position=c['original_queue_position'])
        old = previous.get(rid); previous[rid]=now
        if old is None: continue
        counts['adjacent_observations']+=1
        resumed = [x for x in successful[rid] if old['before_fixed_gate_decision']<=x[0]<boundary]
        if old['known_tokens']!=now['known_tokens']: counts['changed_known_excluded']+=1; continue
        if resumed: counts['intervening_success_excluded']+=1; continue
        counts['eligible_adjacent_pairs']+=1; compared.add(rid)
        delta = old['cached_tokens']-now['cached_tokens']
        assert now['known_recompute_tokens']-old['known_recompute_tokens']==delta
        counts['decreased_pairs' if delta>0 else 'increased_pairs' if delta<0 else 'unchanged_pairs']+=1
        if delta<0: counts['gross_cached_tokens_gained']-=delta
        if delta>0:
            loss_requests.add(rid); counts['gross_cached_tokens_lost']+=delta
            losses.append(dict(request_id=mapping[rid][len('measured/'):],native_request_id=rid,
                cached_tokens_lost=delta, before=old, after=now, intervening_successful_preempted_allocations=[]))
counts.update(observed_requests=len(observed), compared_requests=len(compared), requests_with_cache_loss=len(loss_requests))
result = dict(schema='c-math-restore-order-cache-drift-v1', integrity='VALID', run=str(RUN), counts=dict(counts),
    input_sha256=hashes, analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    largest_observed_loss=max(losses,key=lambda x:x['cached_tokens_lost']) if losses else None, observed_losses=losses,
    pressure_validation=dict(preempted_native_allocation_attempts=sum(len(v) for v in by_key.values()),
        fixed_gate_decisions=len(gate['decisions']), successful_preempted_allocations=sum(len(v) for v in successful.values()),
        scheduler_step_minus_call_id=offset, ambiguous_joins=0),
    interval_rule='Each candidate snapshot precedes its selector\'s selected-or-head fixed-gate event. Reject any pressure-confirmed successful PREEMPTED allocation at gate indices [old_anchor,new_anchor); this also orders snapshots within one call. Compare only adjacent observations of the same native ID with unchanged known_tokens.',
    limitations=['Only observed bounded-prefix candidates; no claim about unobserved waiting requests.',
        'Gross token losses sum adjacent observed decreases, not unique evicted tokens, avoidable GPU time, or policy benefit.',
        'Different restoration episodes are excluded through native allocation events; no future EOS or output length is used.'])
with OUT.open('x',encoding='utf-8') as stream: json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False); stream.write('\n')
print(json.dumps(dict(counts=result['counts'],largest_observed_loss=result['largest_observed_loss'],pressure_validation=result['pressure_validation'])))
