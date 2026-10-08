"""Existing FCFS deadline-demand principle applied only to the prefill cap.

Predicted capacity is an approximation, not a TTFT guarantee. All native decode
requests still execute; observed output length or future arrivals are unused.
"""
import math
from numbers import Real

from niyama_component.run_component import ComponentController

CAPS = (512, 1024, 2048)
TTFT_S = 4.
COST_MARGIN = 1.2


def prefix_demand(pending, now, head_only=False):
    """pending is native-order (external_id, remaining_prompt, deadline) tuples.

    head_only is a CPU diagnostic only. Like the original opportunity screen,
    any expired pending deadline invalidates either finite demand calculation.
    """
    cumulative, expired, entries = 0, 0, []
    for request_id, remaining, deadline in pending:
        assert remaining > 0 and math.isfinite(deadline) and math.isfinite(now)
        cumulative += remaining
        slack = deadline-now
        expired += int(slack<=0)
        entries.append(dict(request_id=request_id,prefix_tokens=cumulative,
            deadline_s=deadline,remaining_deadline_s=slack,
            required_rate_tokens_per_s=cumulative/slack if slack>0 else None))
    if not entries:
        return dict(demand_tokens_per_s=None,demand_limiting_request=None,expired_pending_count=0)
    if expired:
        limiting = next(r for r in entries if r['remaining_deadline_s']<=0)
        return dict(demand_tokens_per_s=None,demand_limiting_request=limiting,expired_pending_count=expired)
    limiting = entries[0] if head_only else max(entries,key=lambda r:r['required_rate_tokens_per_s'])
    return dict(demand_tokens_per_s=limiting['required_rate_tokens_per_s'],
                demand_limiting_request=limiting,expired_pending_count=0)


def choose_demand_cap(pending, now, decode_count, decode_context, head_context, predict):
    """Pure CPU rule; candidate work is min(cap, actual current backlog)."""
    backlog = sum(r[1] for r in pending)
    state = prefix_demand(pending,now)
    state.update(candidate_cost_s={str(c):None for c in CAPS},
        candidate_capacity_tokens_per_s={str(c):None for c in CAPS},
        candidate_effective_prefill_tokens={str(c):min(c,backlog) for c in CAPS},
        demand_model_error=None)
    if not backlog:
        return 1024,dict(state,reason='no_prefill_backlog',predicted_host_step_s=None)
    for cap in CAPS:
        effective = min(cap,backlog)
        prediction = predict(effective,decode_count,decode_context,head_context)
        if isinstance(prediction,bool) or not isinstance(prediction,Real) or not math.isfinite(prediction) or prediction<=0:
            state['demand_model_error']=f'Invalid/nonpositive host-cost prediction for cap {cap}: {prediction!r}'
            return 2048,dict(state,reason='invalid_model_progress_fallback',predicted_host_step_s=None)
        prediction = float(prediction)
        capacity = effective/(COST_MARGIN*prediction)
        if not math.isfinite(capacity) or capacity<=0:
            state['demand_model_error']=f'Invalid/nonpositive capacity for cap {cap}: {capacity!r}'
            return 2048,dict(state,reason='invalid_model_progress_fallback',predicted_host_step_s=None)
        state['candidate_cost_s'][str(cap)]=prediction
        state['candidate_capacity_tokens_per_s'][str(cap)]=capacity
    if state['expired_pending_count']:
        cap,reason = 2048,'expired_pending_deadline_progress_fallback'
    else:
        feasible = [c for c in CAPS if state['candidate_capacity_tokens_per_s'][str(c)]>=state['demand_tokens_per_s']]
        cap,reason = (min(feasible),'minimum_feasible_cap') if feasible else (2048,'capacity_insufficient_progress_fallback')
    return cap,dict(state,reason=reason,predicted_host_step_s=state['candidate_cost_s'][str(cap)])


class DemandController(ComponentController):
    def __init__(self,name,engine):
        assert name in ('fixed512','fixed1024','fixed2048','prefill_demand'),name
        super().__init__('fixed1024',engine)
        self.name=name

    def choose(self,decode_count,scheduler,rows,step_index,now):
        decision_time=now()
        decoders=[r for r in scheduler.running if r.num_computed_tokens>=r.num_prompt_tokens]
        assert len(decoders)==decode_count
        context=sum(r.num_computed_tokens+1 for r in decoders)
        prefills=[r for r in scheduler.running if r.num_computed_tokens<r.num_prompt_tokens]
        prefills += [r for r in scheduler.waiting if r.num_computed_tokens<r.num_prompt_tokens]
        # In the frozen no-preemption/APC/connector domain each pending prompt
        # must occur exactly once in the current native running+waiting order.
        assert len({r.request_id for r in prefills})==len(prefills), 'Duplicate native prefill entry'
        backlog=sum(r.num_prompt_tokens-r.num_computed_tokens for r in prefills)
        assert backlog==sum(max(0,r.num_prompt_tokens-r.num_computed_tokens) for r in scheduler.requests.values()), 'Native prefill order omitted current prompt work'
        head=prefills[0] if prefills else None
        head_context=0 if head is None else head.num_computed_tokens
        head_id=None if head is None else self.external(head)
        free=scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
        memory_budget=(free*self.engine.vllm_config.cache_config.block_size)//32
        fixed=self.name.startswith('fixed')
        if fixed:
            cap=int(self.name[5:])
            details=dict(reason='fixed_policy',demand_tokens_per_s=None,demand_limiting_request=None,
                expired_pending_count=None,candidate_cost_s={},candidate_capacity_tokens_per_s={},
                candidate_effective_prefill_tokens={},demand_model_error=None,predicted_host_step_s=None)
            pending_order=None
        else:
            pending=[(self.external(r),r.num_prompt_tokens-r.num_computed_tokens,
                      rows[self.external(r)]['arrival_s']+TTFT_S) for r in prefills]
            pending_order=[r[0] for r in pending]
            cap,details=choose_demand_cap(pending,decision_time,decode_count,context,head_context,self.predict)
        limiter=details['demand_limiting_request']
        self.pending=dict(step_index=step_index,decision_time_s=decision_time,policy=self.name,
            decode_count=decode_count,decode_context_source=context,
            head_prefill_context_estimate=head_context,head_prefill_id=head_id,
            prefill_backlog_tokens=backlog,pending_prefill_count=len(prefills),pending_prefill_order=pending_order,
            free_kv_blocks=free,memory_budget_tokens=memory_budget,
            legal_prefill_caps=[cap] if fixed else list(CAPS),
            legal_total_tiers=[c+decode_count for c in ([cap] if fixed else CAPS)],
            min_slack_s=None if limiter is None else limiter['remaining_deadline_s'],
            limiting_request=limiter,evaluated_total_predictions=[],
            baseline_cap=1024,candidate_cap=None if fixed else cap,
            requested_cap=cap,selected_total_tier=None,requested_total_budget=cap+decode_count,
            execution_source='fixed_policy' if fixed else 'prefill_demand',
            execution_status='REQUESTED',forward_completed=False,
            ttft_deadline_s=TTFT_S,cost_margin=COST_MARGIN,**details)
        self.decisions+=1
        self.changed_cap+=int(backlog>0 and cap!=1024)
        return cap
