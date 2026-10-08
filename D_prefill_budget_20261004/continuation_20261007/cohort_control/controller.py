"""Prefill-cap-only cohort experiment; current-cohort forecasts are not bounds."""
import math
from numbers import Integral

from niyama_component.run_component import ComponentController, choose_by_slack
from cohort_control.cohort_cost import remaining_costs


class InvalidCohort(ValueError):
    """Only explicitly checked cohort inputs or predictions are recoverable."""


class CohortController(ComponentController):
    def __init__(self, name, engine):
        assert name in ('fixed1024', 'progress_floor', 'cohort_protection'), name
        super().__init__('fixed1024' if name == 'fixed1024' else 'niyama_component', engine)
        self.name = name
        self.cohort_triggered = self.cohort_invalid = self.cohort_empty = 0

    def _cohort_action(self, scheduler, rows, record):
        current = [r for r in scheduler.running if r.num_computed_tokens >= r.num_prompt_tokens]
        if len(current) != record['decode_count']:
            raise InvalidCohort('Current decoder count changed after original decision')
        decision_time = record['decision_time_s']
        tau = self.design['batch_token_deadline_s']
        completion_s = self.design['primary_slo']['completion_s']
        cohort, details = [], []
        for request in current:
            external = self.external(request)
            row = rows[external]
            context = request.num_computed_tokens+1
            observed = context-row['prompt_tokens']
            declared = row['max_tokens']
            times = row['token_times_s']
            if (isinstance(context, bool) or not isinstance(context, Integral) or context < 0
                or isinstance(declared, bool) or not isinstance(declared, Integral)
                or observed != len(row['output_token_ids']) or observed != len(times)
                or not times or times[-1] > decision_time):
                raise InvalidCohort('Declared/observed/native alignment failed for '+external)
            remaining = declared-observed
            if remaining <= 0:
                raise InvalidCohort('Nonpositive declared remaining tokens for '+external)
            arrival, first = row['arrival_s'], times[0]
            if not all(math.isfinite(t) for t in (arrival,first,decision_time)):
                raise InvalidCohort('Nonfinite observed time for '+external)
            actual_prefill = first-arrival
            internal_target = completion_s-declared*tau
            deadline = arrival+max(actual_prefill,internal_target)+(observed+1)*tau
            cohort.append((remaining,context))
            details.append(dict(request_id=external,C=context,m=observed,L=declared,R=remaining,
                arrival_s=arrival,actual_prefill_s=actual_prefill,internal_prefill_target_s=internal_target,
                token_deadline_s=deadline,token_slack_s=deadline-decision_time))
        if sum(c for _,c in cohort) != record['decode_context_source']:
            raise InvalidCohort('Physical decoder context changed after original decision')
        try:
            costs = remaining_costs(cohort, self.calibration['runtime_models']['le512'])
        except ValueError as exc:
            raise InvalidCohort('Current-cohort cost invalid: '+str(exc)) from exc
        protected, excluded = [], []
        for item in details:
            # Missing keys are implementation errors, not silently recovered input errors.
            cost = costs[item['R']]
            if not math.isfinite(cost) or cost <= 0:
                raise InvalidCohort('Nonpositive/nonfinite remaining cost')
            headroom = item['arrival_s']+completion_s-decision_time-cost
            if not math.isfinite(headroom):
                raise InvalidCohort('Nonfinite conditional completion headroom')
            item.update(cost_s=cost,completion_headroom_s=headroom,protected=headroom>0)
            (protected if headroom>0 else excluded).append(item['request_id'])
        result = dict(cohort_requests=details,cohort_protected_ids=protected,cohort_excluded_ids=excluded)
        if not protected:
            result['cohort_status'] = 'EMPTY_PROTECTION_PROGRESS_FALLBACK'
            return result
        limiting = min((r for r in details if r['protected']),key=lambda r:r['token_slack_s'])
        slack = limiting['token_slack_s']
        if not math.isfinite(slack):
            raise InvalidCohort('Nonfinite protected token slack')
        def checked_prediction(p,d,cd,cp):
            prediction = self.predict(p,d,cd,cp)
            if not math.isfinite(prediction) or prediction <= 0:
                raise InvalidCohort('Protected-set step prediction is nonpositive/nonfinite')
            return prediction
        cap,tier,prediction,evaluated,reason = choose_by_slack(slack,
            record['decode_count'], record['decode_context_source'],
            record['head_prefill_context_estimate'], record['legal_total_tiers'], checked_prediction)
        result.update(cohort_status='PROTECTED_SET',requested_cap=cap,selected_total_tier=tier,
            predicted_host_step_s=prediction,evaluated_total_predictions=evaluated,
            min_slack_s=slack,limiting_request=dict(request_id=limiting['request_id'],
                observed_outputs=limiting['m'],declared_output_tokens=limiting['L'],
                internal_prefill_target_s=limiting['internal_prefill_target_s'],
                actual_prefill_s=limiting['actual_prefill_s'],deadline_s=limiting['token_deadline_s']),
            reason=reason,execution_source='cohort_protection')
        return result

    def choose(self, decode_count, scheduler, rows, step_index, now):
        changed_before = self.changed_cap
        original_cap = super().choose(decode_count,scheduler,rows,step_index,now)
        record = self.pending
        fixed = self.name == 'fixed1024'
        normal_backlog = record['prefill_backlog_tokens']>0 and record['memory_budget_tokens']>=1000
        progress_cap = max(original_cap,1024) if normal_backlog else original_cap
        trigger = (not fixed and normal_backlog and record['min_slack_s'] is not None
                   and record['min_slack_s']<=0)
        original_tiers = list(record['legal_total_tiers'])
        original_caps = list(record['legal_prefill_caps'])
        record.update(original_N_legal_total_tiers=original_tiers,original_N_legal_prefill_caps=original_caps,
            original_N_cap=None if fixed else original_cap,
            original_N_min_slack_s=None if fixed else record['min_slack_s'],
            original_N_limiting_request=None if fixed else record['limiting_request'],
            original_N_selected_total_tier=None if fixed else record['selected_total_tier'],
            original_N_predicted_host_step_s=None if fixed else record['predicted_host_step_s'],
            original_N_evaluated_total_predictions=[] if fixed else record['evaluated_total_predictions'],
            original_N_reason='NOT_EVALUATED_FIXED' if fixed else record['reason'],
            progress_floor_cap=None if fixed else progress_cap,
            progress_floor_reason=('NOT_EVALUATED_FIXED' if fixed else
                'normal_memory_backlog_floor' if normal_backlog else 'original_N_low_memory_or_no_backlog'),
            reference_caps=dict(fixed1024=1024,original_N=None if fixed else original_cap,
                                progress_floor=None if fixed else progress_cap),
            cohort_trigger=trigger,cohort_status=('NOT_EVALUATED_FIXED' if fixed else
                'NOT_EVALUATED_PROGRESS_FLOOR' if self.name=='progress_floor' else 'NOT_TRIGGERED'),
            cohort_requests=[],cohort_protected_ids=[],cohort_excluded_ids=[],cohort_error=None,
            execution_source='fixed1024' if fixed else 'original_N')
        cap = original_cap
        floor_source = None
        if self.name == 'progress_floor':
            cap = progress_cap
            if normal_backlog:
                floor_source = 'progress_floor'
        elif self.name == 'cohort_protection' and trigger:
            self.cohort_triggered += 1
            try:
                proposal = self._cohort_action(scheduler,rows,record)
            except InvalidCohort as exc:
                self.cohort_invalid += 1
                record.update(cohort_status='INVALID_FALLBACK_ORIGINAL_N',cohort_error=str(exc),
                    execution_source='invalid_cohort_original_N')
            else:
                record.update(proposal)
                if proposal['cohort_status']=='EMPTY_PROTECTION_PROGRESS_FALLBACK':
                    self.cohort_empty += 1
                    cap = progress_cap
                    floor_source = 'empty_protection_progress_fallback'
                else:
                    cap = proposal['requested_cap']
        if floor_source is not None:
            record['execution_source'] = floor_source
            if cap != original_cap:
                record.update(selected_total_tier=None,evaluated_total_predictions=[],
                    predicted_host_step_s=self.predict(cap,decode_count,record['decode_context_source'],
                                                      record['head_prefill_context_estimate']),
                    reason=floor_source+'_1024_cap')
            else:
                record['reason'] = floor_source+'_original_cap_already_at_least_1024'
        record['legal_prefill_caps'] = ([1024] if fixed else
            sorted(set(original_caps+([1024] if normal_backlog else []))))
        record.update(requested_cap=cap,final_cap=cap,candidate_cap=None if fixed else cap,
            requested_total_budget=cap+decode_count,
            predicted_host_step_s=self.predict(cap,decode_count,record['decode_context_source'],
                                              record['head_prefill_context_estimate']))
        self.changed_cap = changed_before+int(record['prefill_backlog_tokens']>0 and cap!=1024)
        return cap

    def summary(self):
        result = super().summary()
        result.update(cohort_triggered_decisions=self.cohort_triggered,
            cohort_scored_decisions=self.cohort_triggered-self.cohort_invalid,
            cohort_invalid_fallback_decisions=self.cohort_invalid,
            cohort_empty_protection_decisions=self.cohort_empty)
        return result
