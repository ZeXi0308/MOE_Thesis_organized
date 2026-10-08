#!/usr/bin/env python3
"""CPU description of the recorded component trajectory; no counterfactual or new gate.

Only the two completed N formal runs are grouped. Existing full-service scores,
all requests, all observations and original denominators remain unchanged.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics as st

ROOT = Path(__file__).resolve().parent


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def describe(v):
    return dict(count=len(v), min=min(v, default=None), mean=st.mean(v) if v else None,
                median=st.median(v) if v else None, max=max(v, default=None))
def decision(s): return s['component_decision']
def nonpositive(s):
    d = decision(s)
    return d['min_slack_s'] is not None and d['min_slack_s'] <= 0
def binding(s):
    return s['budget'] < 1024 and s['prefill_tokens'] == s['budget'] < decision(s)['prefill_backlog_tokens']


def summary(rows):
    ds = [decision(s) for _, s in rows]
    return dict(steps=len(rows), host_s=sum(s['end_s']-s['start_s'] for _, s in rows),
        prefill_tokens=sum(s['prefill_tokens'] for _, s in rows),
        actual_P=describe([s['prefill_tokens'] for _, s in rows]),
        actual_D=describe([s['decode_tokens'] for _, s in rows]),
        min_slack_s=describe([d['min_slack_s'] for d in ds if d['min_slack_s'] is not None]),
        zero_P_steps=sum(s['prefill_tokens'] == 0 for _, s in rows),
        cap0_steps=sum(d['requested_cap'] == 0 for d in ds),
        cap_below1024_steps=sum(d['requested_cap'] < 1024 for d in ds),
        selected_total_tiers=dict(Counter(str(d['selected_total_tier']) for d in ds)))


def streaks(rows):
    groups = []
    for i, s in rows:
        if not groups or i != groups[-1][-1][0]+1: groups.append([])
        groups[-1].append((i, s))
    result = []
    for g in groups:
        a, z = g[0], g[-1]; ds = [decision(s) for _, s in g]
        result.append(dict(first_step=a[0], last_step=z[0],
            first_decision_s=ds[0]['decision_time_s'], last_end_s=z[1]['end_s'],
            wall_s=z[1]['end_s']-ds[0]['decision_time_s'],
            backlog_before_first=ds[0]['prefill_backlog_tokens'],
            backlog_before_last=ds[-1]['prefill_backlog_tokens'],
            peak_backlog=max(d['prefill_backlog_tokens'] for d in ds), **summary(g)))
    return result


def analyze(path, slo, tau):
    raw = json.loads(path.read_text()); steps = raw['steps']
    completed = [(i, s) for i, s in enumerate(steps) if decision(s)['forward_completed'] and s['end_s'] is not None]
    backlog = [(i, s) for i, s in completed if decision(s)['prefill_backlog_tokens'] > 0]
    grouped = defaultdict(list)
    for i, s in backlog:
        d = decision(s); slack = d['min_slack_s']
        grouped[(d['reason'], 'none' if slack is None else 'negative' if slack < 0 else 'zero' if slack == 0 else 'positive')].append((i, s))
    neg = [(i, s) for i, s in backlog if nonpositive(s)]
    failed = {q['request_id']: q for q in raw['requests'] if q['token_times_s'][0]-q['arrival_s'] > slo['ttft_s']}
    cumulative, trace = defaultdict(int), defaultdict(list)
    for i, s in completed:
        d = decision(s); t = d['decision_time_s']
        ps = {q['request_id']: q['prefill_tokens'] for q in s['requests']}
        for rid, q in failed.items():
            if q['add_s'] <= t < q['token_times_s'][0]:
                trace[rid].append(dict(step=i, decision_time_s=t, end_s=s['end_s'],
                    actual_request_P=ps.get(rid, 0), prior_scheduled_P=cumulative[rid],
                    reason=d['reason'], min_slack_s=d['min_slack_s'], cap=d['requested_cap'],
                    backlog=d['prefill_backlog_tokens'], binding_below1024=binding(s)))
        for q in s['requests']: cumulative[q['request_id']] += q['prefill_tokens']
    failed_rows = []
    for rid, q in failed.items():
        ts = trace[rid]; first = q['token_times_s'][0]
        cross = next((x for x in ts if x['decision_time_s'] > q['arrival_s']+slo['ttft_s']), None)
        failed_rows.append(dict(request_id=rid, cohort='long' if q['max_tokens'] == 512 else 'background',
            arrival_s=q['arrival_s'], add_s=q['add_s'], add_lag_s=q['add_s']-q['arrival_s'],
            first_scheduled_s=q['first_scheduled_s'], first_scheduled_age_s=q['first_scheduled_s']-q['arrival_s'],
            first_output_s=first, ttft_s=first-q['arrival_s'], first_pending_decision=ts[0],
            first_pending_decision_without_request_P=next((x for x in ts if x['actual_request_P'] == 0), None),
            first_decision_after_ttft_deadline=cross,
            progress_at_deadline_crossing=None if cross is None else 'not_started' if cross['prior_scheduled_P'] == 0 else 'partial_prefill' if cross['prior_scheduled_P'] < q['prompt_tokens'] else 'prompt_complete',
            nonpositive_completed_span_overlap_s=sum(min(x['end_s'], first)-x['decision_time_s'] for x in ts if x['min_slack_s'] is not None and x['min_slack_s'] <= 0),
            binding_completed_span_overlap_s=sum(min(x['end_s'], first)-x['decision_time_s'] for x in ts if x['binding_below1024'])))
    negative_cost = {}
    for label, subset in [('D_ge128', [s for _, s in neg if s['decode_tokens'] >= 128]), ('D_lt128', [s for _, s in neg if s['decode_tokens'] < 128])]:
        costs = [s['end_s']-s['start_s'] for s in subset]
        negative_cost[label] = dict(host_s=describe(costs), host_above_tau=sum(v > tau for v in costs), actual_P=sum(s['prefill_tokens'] for s in subset))
    return dict(raw_path=str(path), raw_sha256=sha(path), completed_steps=len(completed), backlog_steps=len(backlog),
        grouped=[dict(reason=k[0], slack_group=k[1], **summary(v)) for k, v in grouped.items()],
        streaks={label: streaks(rows) for label, rows in [('nonpositive_slack', neg),
            ('minimum_total128', [(i, s) for i, s in backlog if decision(s)['selected_total_tier'] == 128]),
            ('binding_cap_below1024', [(i, s) for i, s in backlog if binding(s)]),
            ('cap0', [(i, s) for i, s in backlog if s['budget'] == 0])]},
        negative_slack_cost=negative_cost,
        negative_limiting_output_contracts=dict(Counter(str(decision(s)['limiting_request']['declared_output_tokens']) for _, s in neg)),
        ttft_failed_count=len(failed_rows), ttft_failed_cohorts=dict(Counter(x['cohort'] for x in failed_rows)),
        failed_first_pending_reasons=dict(Counter(x['first_pending_decision']['reason'] for x in failed_rows)),
        failed_deadline_crossing_reasons=dict(Counter(x['first_decision_after_ttft_deadline']['reason'] if x['first_decision_after_ttft_deadline'] else 'none' for x in failed_rows)),
        failed_deadline_crossing_progress=dict(Counter(x['progress_at_deadline_crossing'] for x in failed_rows)),
        failed_scheduled_after_deadline=sum(x['first_scheduled_age_s'] > slo['ttft_s'] for x in failed_rows),
        failed_ttft_s=describe([x['ttft_s'] for x in failed_rows]),
        failed_first_scheduled_age_s=describe([x['first_scheduled_age_s'] for x in failed_rows]),
        failed_add_lag_s=describe([x['add_lag_s'] for x in failed_rows]),
        failed_nonpositive_overlap_s=describe([x['nonpositive_completed_span_overlap_s'] for x in failed_rows]),
        failed_requests=failed_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory', nargs='?', type=Path, default=ROOT.parent/'niyama53005_01')
    args = parser.parse_args(); base = args.run_directory
    protocol = json.loads((base/'protocol.json').read_text()); slo = protocol['slo']
    design = protocol['frozen_component_design']; tau = design['batch_token_deadline_s']
    cp = base/'component_results.json'; result = json.loads(cp.read_text())
    runs = [analyze(base/name/'raw.json', slo, tau) for name in ('01_niyama_component', '02_niyama_component')]
    output = dict(schema='d-recorded-component-wait-diagnosis-v1', slo=slo, tau_s=tau,
        component_results_sha256=sha(cp), original_analysis_status=result['status'], runs=runs,
        actual_feature_model_errors={r['cell']: r['execution']['actual_feature_host_model_errors'] for r in result['runs'] if not r['cell'].startswith('warm')},
        same_ttft_failed_IDs=set(x['request_id'] for x in runs[0]['failed_requests']) == set(x['request_id'] for x in runs[1]['failed_requests']),
        scope=[
            'Only recorded trajectories and completed decisions. Final TTFT labels are retrospective diagnostics, never an online signal or counterfactual.',
            'First pending decision without P does not prove native admission eligibility or that this cap alone caused the wait.',
            'Overlap sums decision_time_s through end_s while the request has arrived but not emitted its first output; it excludes gaps between these spans, and is not a causal allocation of waiting time.',
            'Actual-feature model errors are copied from the existing analysis, not refitted; unexecuted tiers and online head-context estimates remain unvalidated.',
            'The negative-slack branch returns the minimum total tier independently of model feasibility. This does not establish why the trajectory first entered debt.',
            'Declared max_tokens=384 gives internal prefill target 20-384*0.03125=8s versus evaluation TTFT4s; max_tokens=512 gives 4s. This is a chunk-only adaptation, not full Niyama or a quality/novelty claim.',
            'Full-service metrics, all request counts, failures, thresholds and elapsed denominators are unchanged. No GPU or new policy is run.'])
    dest = base/'wait_diagnosis.json'; dest.write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps(dict(output=str(dest), runs=[dict(path=r['raw_path'], backlog_steps=r['backlog_steps'], ttft_failed=r['ttft_failed_count']) for r in runs])))


if __name__ == '__main__': main()
