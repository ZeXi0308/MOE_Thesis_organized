"""Approximate FCFS state transition model, not native replay or a bound.

Only declared fixed output budgets are supported. Full future arrivals may be
provided for OFFLINE model validation; online callers pass current known state
and an empty arrival list. No measured future step/finish times are consumed.
KV fragmentation/cache/preemption are not modeled: reject a state when all
currently known requests' full prompt+output allocation cannot fit. This is a
conservative domain guard, not a replacement for native allocation.
"""
import math
from collections import Counter

CAPS = (512, 1024, 2048)


def predict(models, p, d, cd, cp):
    m = models['le512' if p+d <= 512 else 'gt512']
    cost = m['intercept_s']+sum(a*b for a,b in zip(m['coefficients'],(p,d,cd,cp)))
    if not math.isfinite(cost) or cost <= 0:
        raise ValueError('Nonpositive/nonfinite projected step cost')
    return cost


def new_request(row):
    return dict(id=row.get('id',row.get('request_id')),prompt=row.get('prompt',row.get('prompt_tokens',len(row.get('prompt_token_ids',[])))),
                output_limit=row.get('output_limit',row.get('max_tokens')),arrival=row.get('arrival',row.get('arrival_s')),
                prefill_done=0,emitted=0,first=None,last=None,max_gap=0.)


def pending(running, waiting):
    return [q for q in running+waiting if q['prefill_done'] < q['prompt']]


def demand_cap(now, running, waiting, models):
    ps = pending(running,waiting)
    if not ps:
        return 1024
    ds = [q for q in running if q['emitted']]
    total = sum(q['prompt']-q['prefill_done'] for q in ps)
    cumulative, rate = 0, 0.
    for q in ps:
        cumulative += q['prompt']-q['prefill_done']
        slack = q['arrival']+4-now
        if slack <= 0:
            return 2048
        rate = max(rate,cumulative/slack)
    for cap in CAPS:
        p = min(cap,total)
        cost = predict(models,p,len(ds),sum(q['prompt']+q['emitted'] for q in ds),ps[0]['prefill_done'])
        if p/(1.2*cost) >= rate:
            return cap
    return 2048


def simulate(state, models, policy='fixed1024', *, future_arrivals=(),
             candidate_cap=None, hold_until_head=False, freeze_decode_population=False,
             max_steps=10000, trace=False):
    """Drain all known requests. Optional candidate applies once or through head.

    Then continuation is fixed1024. freeze_decode_population is a DIAGNOSTIC
    cost-only ablation: logical entries/exits remain, but each mixed step's
    decoder features are frozen at fork (no online use).
    """
    running = [dict(q) for q in state['running']]
    waiting = [dict(q) for q in state['waiting']]
    future = sorted([new_request(q) for q in future_arrivals],key=lambda q:q['arrival'])
    now = state['now']
    n = 0
    completed, costs, cap_counts = {}, [], Counter()
    initial_pending = pending(running,waiting)
    head = initial_pending[0]['id'] if initial_pending else None
    head_first = next((q['first'] for q in running if q['id']==head),None)
    initial_ds = [q for q in running if q['emitted']]
    initial_d = len(initial_ds)
    initial_cd = sum(q['prompt']+q['emitted'] for q in initial_ds)
    transitions = dict(entries=0,exits=0,changed_steps=0,max_decode=initial_d,max_running=len(running),max_reserved_blocks=0)
    while running or waiting or future:
        if n >= max_steps:
            raise ValueError('Rollout step limit exceeded')
        if not running and not waiting:
            now = max(now,future[0]['arrival'])
        while future and future[0]['arrival'] <= now:
            waiting.append(future.pop(0))
        # Sufficient (not necessary) fit check: pessimistically reserve every
        # currently known request to its declared final length, including queued.
        reserved = sum((q['prompt']+q['output_limit']+15)//16 for q in running+waiting)
        transitions['max_reserved_blocks'] = max(transitions['max_reserved_blocks'],reserved)
        if reserved > state.get('total_blocks',36752):
            raise ValueError('Known full allocation outside modeled KV domain')
        ps = pending(running,waiting)
        ds = [q for q in running if q['emitted']]
        d = len(ds)
        cd = sum(q['prompt']+q['emitted'] for q in ds)
        cp = ps[0]['prefill_done'] if ps else 0
        if candidate_cap is not None and (n==0 or (hold_until_head and head_first is None)):
            cap = candidate_cap
        elif policy == 'prefill_demand':
            cap = demand_cap(now,running,waiting,models)
        else:
            cap = int(policy[5:])
        if cap not in CAPS:
            raise ValueError('Unsupported cap')
        cap_counts[cap] += 1
        eligible_p = sum(q['prompt']-q['prefill_done'] for q in running)
        eligible_p += sum(q['prompt']-q['prefill_done'] for q in waiting[:max(0,192-len(running))])
        baseline_p = min(1024,eligible_p)
        p_left = min(cap,4096-d)
        p, scheduled = 0, []
        for q in running:
            if q['emitted']:
                scheduled.append((q,0))
            elif p_left:
                amount = min(p_left,q['prompt']-q['prefill_done'])
                scheduled.append((q,amount))
                p_left -= amount
                p += amount
        while waiting and p_left and len(running)<192:
            q = waiting.pop(0)
            amount = min(p_left,q['prompt']-q['prefill_done'])
            running.append(q)
            scheduled.append((q,amount))
            p_left -= amount
            p += amount
        if not scheduled:
            raise ValueError('No progress in modeled domain')
        transitions['max_running'] = max(transitions['max_running'],len(running))
        model_d,model_cd = (initial_d,initial_cd) if freeze_decode_population and p else (d,cd)
        duration = predict(models,p,model_d,model_cd,cp)
        end = now+duration
        entries, exits = [], []
        for q,amount in scheduled:
            was_decode = q['emitted']>0
            q['prefill_done'] += amount
            if q['prefill_done'] == q['prompt']:
                if not was_decode:
                    q['first'] = end
                    entries.append(q['id'])
                    if q['id']==head:
                        head_first = end
                else:
                    q['max_gap'] = max(q['max_gap'],end-q['last'])
                q['last'] = end
                q['emitted'] += 1
                if q['emitted']==q['output_limit']:
                    q['completion'] = end
                    completed[q['id']] = q
                    exits.append(q['id'])
        running = [q for q in running if q['id'] not in completed]
        transitions['entries'] += len(entries)
        transitions['exits'] += len(exits)
        transitions['changed_steps'] += int(p!=baseline_p)
        transitions['max_decode'] = max(transitions['max_decode'],d)
        if trace:
            costs.append(dict(step=n,start=now,end=end,P=p,D=d,Cd=cd,Cp=cp,cap=cap,entries=entries,exits=exits))
        now = end
        n += 1
    rows = list(completed.values())
    if not rows:
        raise ValueError('Empty rollout')
    good = [q['id'] for q in rows if q['first']-q['arrival']<=4 and q['max_gap']<=.1 and q['completion']-q['arrival']<=20]
    mean = lambda values: sum(values)/len(values)
    metrics = dict(qualified_count=len(good),qualified_ids=sorted(good),elapsed_s=now,
        conditional_goodput=len(good)/now,mean_ttft_s=mean([q['first']-q['arrival'] for q in rows]),
        mean_generation_s=mean([q['completion']-q['first'] for q in rows]),
        mean_flow_s=mean([q['completion']-q['arrival'] for q in rows]),head_first_s=head_first,
        steps=n,cap_counts=dict(cap_counts),**transitions)
    return dict(metrics=metrics,requests=completed,steps=costs)
