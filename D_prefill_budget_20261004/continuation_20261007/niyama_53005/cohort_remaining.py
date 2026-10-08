#!/usr/bin/env python3
"""Two-run, CPU-only recorded-state opportunity diagnostic, never a controller.

Use declared fixed lengths and only outputs observed before each decision.
Forecast only the current native decode cohort with P=0, Cp=0 and the frozen
linear models. Future arrivals, pending-prefill transitions and actual finishes
are not inputs. Predictions are neither bounds nor completion guarantees.
"""
import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent/'niyama_component'))
from calibrate import predict, regime, quantile


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def describe(v):
    return dict(n=len(v), min=min(v, default=None), mean=statistics.fmean(v) if v else None,
        p05=quantile(v, .05), p25=quantile(v, .25), median=quantile(v, .5),
        p75=quantile(v, .75), p95=quantile(v, .95), max=max(v, default=None))
def sign(v): return 'positive' if v > 0 else 'negative' if v < 0 else 'zero'


def remaining(cohort, cost):
    """At pre-step k, retain exactly R>k; cumulative[R] includes R steps."""
    expires = defaultdict(list)
    for q in cohort: expires[q['R']].append(q['C'])
    d, cd0 = len(cohort), sum(q['C'] for q in cohort)
    cumulative, feature_rows = [0.0], []
    for k in range(max(q['R'] for q in cohort)):
        for c in expires[k]: d -= 1; cd0 -= c
        x = [0, d, cd0+k*d, 0]; t = cost(x)
        if not math.isfinite(t) or t < 0: raise ValueError(f'Unscored negative/nonfinite cohort cost at k={k}: {t}')
        cumulative.append(cumulative[-1]+t); feature_rows.append(x)
    return {q['request_id']: cumulative[q['R']] for q in cohort}, feature_rows


def algebra_check():
    qs = [dict(request_id='A', C=103, R=2), dict(request_id='B', C=201, R=3)]
    result, xs = remaining(qs, lambda x: 1+.1*x[1]+.001*x[2])
    assert xs == [[0, 2, 304, 0], [0, 2, 306, 0], [0, 1, 203, 0]]
    assert math.isclose(result['A'], 3.01) and math.isclose(result['B'], 4.313)
    try: remaining(qs, lambda x: -1)
    except ValueError: return 'PASS: two lengths/off-by-one and negative-cost exclusion'
    raise AssertionError('Negative predicted cost was scored')


def extract(step, requests, tau, slo):
    d = step['component_decision']; now = d['decision_time_s']; cohort = []
    if not d['forward_completed'] or step['end_s'] is None: raise ValueError('Unconfirmed forward')
    if not step['start_s'] <= now <= step['end_s']: raise ValueError('Decision outside recorded step')
    for row in step['requests']:
        if row['decode_tokens'] <= 0: continue
        rid = row['request_id']
        if rid not in requests: raise ValueError(f'Unknown decoder {rid}')
        q = requests[rid]; c = row['computed_start']+1; m = c-q['prompt_tokens']; length = q['max_tokens']
        observed = bisect_left(q['token_times_s'], now)
        if row['decode_tokens'] != 1 or row['prefill_tokens'] or q['add_s'] > now or not 0 < m < length or observed != m:
            raise ValueError(f'Unscored decoder alignment: {rid}, m={m}, observed={observed}, L={length}')
        times = q['token_times_s'][:m]
        if observed < len(q['token_times_s']) and q['token_times_s'][observed] == now:
            raise ValueError(f'Ambiguous emission/decision timestamp: {rid}')
        f = times[0]-q['arrival_s']; age = now-q['arrival_s']
        slack = q['arrival_s']+max(f, slo['completion_s']-length*tau)+(m+1)*tau-now
        past_viable = f <= slo['ttft_s'] and age <= slo['completion_s'] and now-times[-1] <= slo['gap_s'] and max((b-a for a, b in zip(times, times[1:])), default=0) <= slo['gap_s']
        cohort.append(dict(request_id=rid, C=c, m=m, R=length-m, age_s=age,
            original_token_slack_s=slack, completion_wall_budget_s=slo['completion_s']-age,
            past_observed_joint_viable=past_viable))
    if len(cohort) != d['decode_count'] or len(cohort) != step['decode_tokens'] or sum(q['C'] for q in cohort) != d['decode_context_source']:
        raise ValueError('Native decoder count/context mismatch')
    if not cohort: raise ValueError('No native decoders')
    minimum = min(q['original_token_slack_s'] for q in cohort)
    limiter = d['limiting_request']['request_id']
    if not math.isclose(minimum, d['min_slack_s'], abs_tol=1e-8) or not any(q['request_id'] == limiter and math.isclose(q['original_token_slack_s'], minimum, abs_tol=1e-8) for q in cohort):
        raise ValueError('Recorded minimum slack/limiter mismatch')
    return cohort


def shadow(slack, d, models):
    # Only the original logged head proxy is reused for this optional shadow;
    # it never enters the decode-cohort remaining-service estimate above.
    if slack is None: return dict(status='EMPTY_ELIGIBLE', cap=None)
    best = d['legal_total_tiers'][0]; feasible = []
    for total in d['legal_total_tiers']:
        x = [max(0, total-d['decode_count']), d['decode_count'], d['decode_context_source'], d['head_prefill_context_estimate']]
        value = predict(models[regime(x)], x)
        if not math.isfinite(value) or value < 0: return dict(status='UNSCORED_NEGATIVE_OR_NONFINITE_MODEL', cap=None)
        if slack > 0 and 1.2*value <= slack: feasible.append(total)
    if feasible: best = max(feasible)
    return dict(status='SHADOW_ONLY', cap=max(0, best-d['decode_count']), selected_total=best)


def analyze(path, calibration, slo, tau):
    raw = json.loads(path.read_text()); requests = {q['request_id']: q for q in raw['requests']}
    models = calibration['models']; rows = []; errors = []; distributions = {str(m): defaultdict(list) for m in (1.0, 1.2)}
    by_d = {str(m): defaultdict(Counter) for m in (1.0, 1.2)}; opportunities = 0
    for i, step in enumerate(raw['steps']):
        d = step['component_decision']
        if not (d['prefill_backlog_tokens'] > 0 and d['min_slack_s'] is not None and d['min_slack_s'] < 0): continue
        opportunities += 1
        try:
            if d['step_index'] != i: raise ValueError('Decision index mismatch')
            cohort = extract(step, requests, tau, slo)
            costs, xs = remaining(cohort, lambda x: predict(models[regime(x)], x))
        except (ValueError, KeyError, TypeError) as exc:
            errors.append(dict(step=i, status='UNSCORED', error=str(exc))); continue
        limiter = next(q for q in cohort if q['request_id'] == d['limiting_request']['request_id'])
        row = dict(step=i, decision_time_s=d['decision_time_s'], D=len(cohort),
            current_recorded_cap=d['requested_cap'], original_min_token_slack_s=d['min_slack_s'],
            remaining_outputs=describe([q['R'] for q in cohort]),
            remaining_service_s=describe(list(costs.values())),
            forecast_steps=len(xs), forecast_regimes=dict(Counter(regime(x) for x in xs)),
            limiter=dict(request_id=limiter['request_id'], C=limiter['C'], m=limiter['m'], R=limiter['R'],
                age_s=limiter['age_s'], remaining_service_s=costs[limiter['request_id']],
                original_token_slack_s=limiter['original_token_slack_s'],
                past_observed_joint_viable=limiter['past_observed_joint_viable']), scenarios={})
        for margin in (1.0, 1.2):
            key = str(margin); heads = {q['request_id']: q['completion_wall_budget_s']-margin*costs[q['request_id']] for q in cohort}
            positive = {rid for rid, v in heads.items() if v > 0}; past = {q['request_id'] for q in cohort if q['past_observed_joint_viable']}
            eligible_slack = min((q['original_token_slack_s'] for q in cohort if q['request_id'] in positive), default=None)
            selected = shadow(eligible_slack, d, models); limhead = heads[limiter['request_id']]
            classification = 'all_positive' if len(positive) == len(cohort) else 'none_positive' if not positive else 'mixed'
            distributions[key]['all_decoder_headroom_s'].extend(heads.values()); distributions[key]['limiter_headroom_s'].append(limhead)
            distributions[key]['eligible_min_original_token_slack_s'].extend([] if eligible_slack is None else [eligible_slack])
            by_d[key][str(len(cohort))][classification] += 1
            row['scenarios'][key] = dict(completion_headroom_s=describe(list(heads.values())),
                limiter_headroom_s=limhead, limiter_headroom_sign=sign(limhead), classification=classification,
                positive_decoders=len(positive), other_positive_decoders=len(positive-{limiter['request_id']}),
                eligible_min_original_token_slack_s=eligible_slack,
                eligible_min_changed=eligible_slack is not None and not math.isclose(eligible_slack, d['min_slack_s'], abs_tol=1e-8),
                shadow=selected, shadow_cap_changed=selected['cap'] is not None and selected['cap'] != d['requested_cap'],
                past_observed_joint_viable_count=len(past), positive_but_past_failed=len(positive-past),
                past_viable_but_nonpositive=len(past-positive), eligible_equals_past_viability=positive == past)
        rows.append(row)
    summaries = {}
    for key in distributions:
        scenarios = [r['scenarios'][key] for r in rows]
        summaries[key] = dict(distributions={k: describe(v) for k, v in distributions[key].items()},
            limiter_signs=dict(Counter(s['limiter_headroom_sign'] for s in scenarios)),
            cohort_signs=dict(Counter(s['classification'] for s in scenarios)),
            steps_with_other_positive=sum(s['other_positive_decoders'] > 0 for s in scenarios),
            eligible_min_changed_steps=sum(s['eligible_min_changed'] for s in scenarios),
            shadow_cap_changed_steps=sum(s['shadow_cap_changed'] for s in scenarios),
            shadow_caps=dict(Counter(str(s['shadow']['cap']) for s in scenarios)),
            empty_eligible_steps=sum(s['positive_decoders'] == 0 for s in scenarios),
            eligible_equals_past_viability_steps=sum(s['eligible_equals_past_viability'] for s in scenarios),
            any_positive_but_past_failed_steps=sum(s['positive_but_past_failed'] > 0 for s in scenarios),
            any_past_viable_but_nonpositive_steps=sum(s['past_viable_but_nonpositive'] > 0 for s in scenarios),
            cohort_classification_by_exact_D={d: dict(c) for d, c in by_d[key].items()})
    return dict(raw_path=str(path), raw_sha256=sha(path), negative_backlog_decisions=opportunities,
        scored_decisions=len(rows), unscored=errors, summaries=summaries, decisions=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory', nargs='?', type=Path, default=ROOT.parent/'niyama53005_01')
    args = parser.parse_args(); base = args.run_directory
    protocol = json.loads((base/'protocol.json').read_text()); design = protocol['frozen_component_design']
    cp = base/'calibration.json'; calibration = json.loads(cp.read_text())
    if sha(cp) != design['calibration_sha256']: raise ValueError('Frozen calibration hash mismatch')
    check = algebra_check()
    runs = [analyze(base/name/'raw.json', calibration, protocol['slo'], design['batch_token_deadline_s']) for name in ('01_niyama_component', '02_niyama_component')]
    output = dict(schema='d-current-decode-cohort-remaining-v1', algebra_check=check,
        calibration_sha256=sha(cp), slo=protocol['slo'], margins=[1.0, 1.2], runs=runs,
        scope=[
            'Opportunity diagnostic for exactly two formal recorded trajectories, not an online method, counterfactual performance, contribution or novelty result.',
            'C=computed_start+1; m=C-prompt matches strictly prior emitted timestamps. Declared fixed output length supplies R=L-m. No actual completion, future emitted output or future arrival is used.',
            'For k=0..maxR-1, retain all current decoders with R>k and predict frozen cost(P=0,Dk,sum(C+k),Cp=0); accumulate exactly R steps for each request. All current decoders remain in the forecast even if excluded from the shadow eligibility set.',
            'No pending-prefill state enters remaining-service estimates. No future prefill/decode admissions, elapsed host gaps or new arrivals are forecast. This conditional estimate is not a lower bound, irrecoverability proof, or SLO guarantee.',
            'Both unscaled and original1.2-scaled costs are reported without refit, margin selection or outcome-based threshold search. Distributions pool request-decision observations, not independent requests.',
            'Positive completion headroom alone does not ensure joint SLO; observed TTFT/closed-gap/open-gap/age viability is separately compared using only past emissions.',
            'Shadow eligible set has positive modeled completion headroom; its minimum ORIGINAL token slack feeds the original tiers/1.2step margin/min-tier rule using logged current D/Cd/head proxy. Empty eligibility is EMPTY_ELIGIBLE, never infinity. This is not a proposed second controller or executed action.',
            'Unknown/misaligned decoder states or negative/nonfinite cohort predictions are explicitly unscored; negative shadow predictions suppress only the shadow score. Full E2E SLO metrics and elapsed denominators remain untouched.',
            'Time/D tables are descriptive, with no fitted classifier or threshold search; differing predicted eligibility is not proof of information value beyond age/count/remaining length.'])
    dest = base/'cohort_remaining.json'; dest.write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps(dict(output=str(dest), runs=[dict(path=r['raw_path'], scored=r['scored_decisions'],
        unscored=len(r['unscored']), scenarios={k: {field: v[field] for field in
        ('limiter_signs', 'cohort_signs', 'empty_eligible_steps', 'shadow_cap_changed_steps')}
        for k, v in r['summaries'].items()}) for r in runs])))


if __name__ == '__main__': main()
