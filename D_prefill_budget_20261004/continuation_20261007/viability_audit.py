#!/usr/bin/env python3
"""CPU diagnostic on existing normal-arrival R traces; no counterfactual performance.

Use only events observed by engine.step start, a lower bound on decision time.
Failure is irreversible under the existing conjunctive 4s/100ms/20s SLO.
Reconstruct logical state from past executed tokens, not future output budgets.
Usage: python3 -B viability_audit.py RAW.json [...] > viability_audit.json
"""
import bisect
from collections import Counter
import json
from pathlib import Path
import sys


def audit(path):
    raw = json.loads(Path(path).read_text())
    assert raw['policy'] == 'elapsed_replay'
    reqs = {r['request_id']: r for r in raw['requests']}
    computed = Counter()
    rows = []
    for i, s in enumerate(raw['steps']):
        assert not s['preempted']
        t = s['start_s']
        active, decoding, pending, reasons = [], [], [], {}
        for rid, r in reqs.items():
            if r['add_s'] is None or r['add_s'] > t:
                continue
            # The terminal event is used only after it has actually been observed.
            if r['completion_s'] is not None and r['completion_s'] <= t:
                continue
            active.append(rid)
            prefix = r['token_times_s'][:bisect.bisect_right(r['token_times_s'], t)]
            age = t-r['arrival_s']
            failed = []
            if (prefix and prefix[0]-r['arrival_s'] > 4) or (not prefix and age > 4):
                failed.append('ttft')
            if max((b-a for a, b in zip(prefix, prefix[1:])), default=0) > .1:
                failed.append('past_gap')
            if prefix and t-prefix[-1] > .1:
                failed.append('current_gap')
            if age > 20:
                failed.append('flow')
            reasons[rid] = failed
            if computed[rid] >= r['prompt_tokens']:
                assert prefix, (i, rid, 'decoder without observed output')
                decoding.append(rid)
            else:
                pending.append(rid)
        executed_d = {r['request_id'] for r in s['requests'] if r['decode_tokens']}
        assert set(decoding) == executed_d
        assert len(decoding) == s['decode_count_before'] == s['decode_tokens']
        backlog = sum(reqs[rid]['prompt_tokens']-computed[rid] for rid in pending)
        assert backlog == s['prefill_backlog_tokens_before']
        assert len(pending) == s['prefill_backlog_requests_before']
        assert sum(computed[rid] for rid in decoding) == s['decode_context_sum']
        viable_d = [rid for rid in decoding if not reasons[rid]]
        viable_p = [rid for rid in pending if not reasons[rid]]
        opportunity = bool(decoding and not viable_d and backlog > s['budget'] and s['budget'] < 2048)
        row = dict(step=i, start_s=t, baseline_cap=s['budget'], actual_prefill=s['prefill_tokens'],
                   D=len(decoding), K=len(viable_d), pending=len(pending), still_qualifiable_pending=len(viable_p),
                   backlog_tokens=backlog, opportunity=opportunity,
                   free_kv_blocks=s['kv_total_blocks']-s['kv_used_blocks'],
                   running_after_schedule=s['running_count'],
                   failure_reasons=dict(Counter(x for rid in decoding for x in reasons[rid])))
        if opportunity:
            row['decoder_ids'] = decoding
            row['still_qualifiable_pending_ids'] = viable_p
        rows.append(row)
        for r in s['requests']:
            rid = r['request_id']
            assert computed[rid] == r['computed_start']
            computed[rid] += r['prefill_tokens']+r['decode_tokens']
    opportunities = [r for r in rows if r['opportunity']]
    spans = []
    for r in opportunities:
        if not spans or spans[-1]['last_step'] != r['step']-1:
            spans.append(dict(first_step=r['step'], last_step=r['step'], start_s=r['start_s']))
        else:
            spans[-1]['last_step'] = r['step']
    for span in spans:
        span['end_s'] = raw['steps'][span['last_step']]['end_s']
    return dict(raw_path=str(path), interpretation='observed opportunity only; no executed candidate or service effect',
                timing='engine.step start is before policy decision; monotone failures make K=0 conservative',
                steps=len(rows), opportunities=len(opportunities), spans=spans,
                opportunity_prefill_tokens=sum(r['actual_prefill'] for r in opportunities),
                opportunity_with_viable_pending=sum(r['still_qualifiable_pending'] > 0 for r in opportunities),
                opportunity_min_free_kv_blocks=min((r['free_kv_blocks'] for r in opportunities), default=None),
                opportunity_max_running=max((r['running_after_schedule'] for r in opportunities), default=None),
                first_opportunity=opportunities[0] if opportunities else None, states=rows)


if __name__ == '__main__':
    print(json.dumps([audit(p) for p in sys.argv[1:]], indent=2, allow_nan=False))
