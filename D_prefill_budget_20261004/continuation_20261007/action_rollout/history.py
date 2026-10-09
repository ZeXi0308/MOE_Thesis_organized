"""Extract decision-time logical states from the fixed-length serving history.

Requires component_decision timestamps and the frozen no-preemption, no-prefix-
cache, no-skipped-request, synchronous one-token decode domain. Declared output
limits are known inputs: reaching that limit in the observed timestamp prefix
removes a request. Neither final finished/completion_s nor output_token_ids is
read. This is not allocator/kernel replay or a forecast of future arrivals.
"""
from collections.abc import Iterator
import argparse
import json
import math
from pathlib import Path


class StateExtractionError(ValueError):
    """A recorded state cannot be reconstructed under the stated assumptions."""


def iter_states(raw: dict, *, backlog_only: bool = True) -> Iterator[dict]:
    """Yield fresh states before each original decision (backlog states by default).

    Each request has id, prompt, output_limit, arrival, add, computed,
    prefill_done, emitted, first, last, max_gap. max_gap is the largest closed
    emitted-output interval; it is 0 if fewer than two outputs are observed.
    running/waiting preserve inferred native FIFO. free_blocks is sampled before
    scheduling; total_blocks is the fixed usable pool size, never post-step usage.
    Every step is checked/advanced even when its state is not yielded. Mismatch
    raises StateExtractionError with the step index; nothing is silently repaired.
    """
    requests = raw['requests']
    by_id = {r['request_id']: r for r in requests}
    if len(by_id) != len(requests): raise StateExtractionError('Duplicate request IDs')
    order = {r['request_id']: i for i, r in enumerate(requests)}
    progress = {rid: dict(prefill=0, computed=0, emitted=0, first=None, last=None, gap=0.) for rid in by_id}
    admitted, admitted_set = [], set()
    previous_time, previous_end, total_blocks = -math.inf, -math.inf, None
    for index, step in enumerate(raw['steps']):
        def require(condition, message):
            if not condition: raise StateExtractionError(f'step {index}: {message}')
        decision = step.get('component_decision')
        require(isinstance(decision, dict) and 'decision_time_s' in decision,
                'Missing exact component decision timestamp; start_s is not a substitute')
        now = decision['decision_time_s']
        require(math.isfinite(now) and now >= previous_time and now >= previous_end, 'Nonmonotonic decision clock')
        require(decision['step_index'] == index, 'Recorded step index differs')
        require(not step['preempted'], 'Preemption is outside this reconstruction domain')
        require(decision.get('forward_completed') is True and step['end_s'] is not None,
                'Original forward is not confirmed; cannot advance its scheduled work')
        require(step['start_s'] <= now <= step['end_s'], 'Decision clock outside engine-step interval')
        known = [r for r in requests if r['add_s'] is not None and r['add_s'] <= now]
        known.sort(key=lambda r: (r['arrival_s'], order[r['request_id']]))
        require(all(a['add_s'] <= b['add_s'] for a, b in zip(known, known[1:])), 'Stable FCFS arrival/submission order differs')
        active = {}
        for request in known:
            rid = request['request_id']; p = progress[rid]
            require(request['arrival_s'] <= request['add_s'], f'{rid}: submitted before external arrival')
            require(request['prompt_tokens'] > 0 and request['max_tokens'] > 0, f'{rid}: unsupported empty request')
            times = request['token_times_s']
            while p['emitted'] < len(times) and times[p['emitted']] <= now:
                emitted_at = times[p['emitted']]
                require(p['last'] is None or emitted_at >= p['last'], f'{rid}: output timestamps out of order')
                if p['last'] is not None: p['gap'] = max(p['gap'], emitted_at-p['last'])
                if p['first'] is None: p['first'] = emitted_at
                p['last'] = emitted_at; p['emitted'] += 1
            prompt, limit = request['prompt_tokens'], request['max_tokens']
            require(0 <= p['prefill'] <= prompt and p['emitted'] <= limit, f'{rid}: progress exceeds declared contract')
            require((p['prefill'] == prompt) == (p['emitted'] > 0), f'{rid}: prompt/output-prefix alignment differs')
            require(p['computed'] == p['prefill']+max(0, p['emitted']-1), f'{rid}: computed/output-prefix alignment differs')
            if p['emitted'] == limit: continue  # Observed declared-length exit, no future completion lookup.
            active[rid] = dict(id=rid, prompt=prompt, output_limit=limit, arrival=request['arrival_s'],
                add=request['add_s'], computed=p['computed'], prefill_done=p['prefill'], emitted=p['emitted'],
                first=p['first'], last=p['last'], max_gap=p['gap'])
        running = [active[rid] for rid in admitted if rid in active]
        waiting = [active[r['request_id']] for r in known if r['request_id'] in active and r['request_id'] not in admitted_set]
        require(all(q['computed'] == q['emitted'] == 0 for q in waiting), 'Waiting request already has executed work')
        decoders = [q for q in running if q['prefill_done'] == q['prompt']]
        pending = [q for q in running+waiting if q['prefill_done'] < q['prompt']]
        backlog = sum(q['prompt']-q['prefill_done'] for q in pending)
        context = sum(q['computed']+1 for q in decoders)
        head = pending[0] if pending else None
        require(len(decoders) == decision['decode_count'] == step['decode_count_before'], 'Decode count differs')
        require(context == decision['decode_context_source'] and context-len(decoders) == step['decode_context_sum'], 'Decode context differs')
        require(backlog == decision['prefill_backlog_tokens'] == step['prefill_backlog_tokens_before'], 'Prefill backlog tokens differ')
        require(len(pending) == step['prefill_backlog_requests_before'], 'Prefill backlog count differs')
        if 'pending_prefill_count' in decision:
            require(len(pending) == decision['pending_prefill_count'], 'Decision pending-prefill count differs')
        require((head['id'] if head else None) == decision['head_prefill_id'], 'Head request differs')
        require((head['computed'] if head else 0) == decision['head_prefill_context_estimate'], 'Head context differs')
        if decision.get('pending_prefill_order') is not None:
            require([q['id'] for q in pending] == decision['pending_prefill_order'], 'Recorded pending FIFO differs')
        actual_decoders = [q['request_id'] for q in step['requests']
            if q['decode_tokens'] > 0 and q['computed_start'] >= by_id[q['request_id']]['prompt_tokens']]
        require(actual_decoders == [q['id'] for q in decoders], 'Scheduled decoder FIFO differs')
        actual_prefills = [q for q in step['requests'] if q['prefill_tokens'] > 0]
        require([q['request_id'] for q in actual_prefills] == [q['id'] for q in pending[:len(actual_prefills)]],
                'Scheduled prefill work bypasses inferred FIFO')
        require(all(q['computed_start']+q['prefill_tokens'] == by_id[q['request_id']]['prompt_tokens']
                    for q in actual_prefills[:-1]), 'Native prefill bypasses an unfinished predecessor')
        if total_blocks is None: total_blocks = step['kv_total_blocks']
        require(step['kv_total_blocks'] == total_blocks, 'Usable KV pool size changed')
        require(0 <= decision['free_kv_blocks'] <= total_blocks, 'Invalid pre-decision free KV count')
        state = dict(now=now, step_index=index, running=running, waiting=waiting,
            free_blocks=decision['free_kv_blocks'], total_blocks=total_blocks,
            head_id=head['id'] if head else None, head_context=head['computed'] if head else 0,
            backlog_tokens=backlog, backlog_requests=len(pending), decode_count=len(decoders), decode_context_source=context)
        if backlog or not backlog_only: yield state
        # Only after exposing the pre-decision state, advance this original step.
        seen = set()
        for scheduled in step['requests']:
            rid = scheduled['request_id']; p = progress[rid]
            require(rid in active and rid not in seen, f'{rid}: unknown, completed or duplicate scheduled request')
            seen.add(rid)
            require(p['computed'] == scheduled['computed_start'], f'{rid}: scheduled computed prefix differs')
            if rid not in admitted_set:
                require(scheduled['prefill_tokens'] > 0, f'{rid}: admission without positive prefill')
                admitted.append(rid); admitted_set.add(rid)
            p['prefill'] += scheduled['prefill_tokens']
            p['computed'] += scheduled['prefill_tokens']+scheduled['decode_tokens']
        previous_time, previous_end = now, step['end_s']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('raw_paths', nargs='+', type=Path)
    args = parser.parse_args()
    for path in args.raw_paths:
        raw = json.loads(path.read_text())
        count = sum(1 for _ in iter_states(raw))
        print(f'{path}: {len(raw["steps"])} steps checked; {count} backlog states; PASS')
