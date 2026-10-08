#!/usr/bin/env python3
"""CPU-only latest-output deadline screen on recorded cohort03 trajectories.

This is one controlled chunk mapping, not full SLOWeave, an online policy run,
or a service/performance counterfactual. No fitting or fallback is introduced.
"""
import argparse
from bisect import bisect_right
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics

BASE = Path(__file__).resolve().parent.parent
CELLS = ('00_fixed1024', '05_fixed1024', '02_cohort_protection', '03_cohort_protection')
GAP, MARGIN = .1, 1.2


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def stats(xs):
    return dict(n=len(xs), min=min(xs, default=None), max=max(xs, default=None),
                mean=statistics.fmean(xs) if xs else None)


def predict(models, p, d, cd, cp):
    model = models['le512' if p+d <= 512 else 'gt512']
    return model['intercept_s'] + sum(a*b for a, b in zip(model['coefficients'], (p, d, cd, cp)))


def screen(path, calibration):
    models = calibration['runtime_models']
    raw = json.loads(path.read_text())
    requests = {r['request_id']: r for r in raw['requests']}
    counts, errors, rows = Counter(), [], []
    for index, step in enumerate(raw['steps']):
        rec = step.get('component_decision', {})
        counts['steps'] += 1
        if rec.get('forward_completed') is not True or step.get('end_s') is None:
            errors.append(dict(step=index, error='forward not confirmed')); continue
        counts['completed_decisions'] += 1
        if rec.get('prefill_backlog_tokens', 0) <= 0:
            counts['no_backlog'] += 1; continue
        counts['backlog_decisions'] += 1
        counts['backlog_zero_actual_prefill'] += int(step['prefill_tokens'] == 0)
        try:
            now, d, cd = rec['decision_time_s'], rec['decode_count'], rec['decode_context_source']
            cp = rec['head_prefill_context_estimate']
            decoders = [x for x in step['requests'] if x['decode_tokens'] > 0
                        and x['computed_start'] >= requests[x['request_id']]['prompt_tokens']]
            assert len({x['request_id'] for x in decoders}) == len(decoders) == d, 'D / decoder IDs mismatch'
            assert step['decode_count_before'] == d, 'native / recorded decision D mismatch'
            assert all(x['decode_tokens'] == 1 for x in decoders), 'not one decode per active request'
            assert sum(x['computed_start']+1 for x in decoders) == cd, 'source context mismatch'
            assert rec['step_index'] == index, 'step index mismatch'
            assert now <= step['end_s'], 'decision after completed forward'
            latest = []
            for x in decoders:
                request = requests[x['request_id']]
                times = request['token_times_s']
                observed = bisect_right(times, now)  # Only emitted outputs at/before this decision.
                assert observed > 0, 'decoder without previously observed output'
                assert observed == x['computed_start']+1-request['prompt_tokens'], 'past-output alignment mismatch'
                assert request['add_s'] <= now, 'decoder not yet submitted'
                latest.append(times[observed-1])
            counts['aligned_backlog_decisions'] += 1
            if not d:
                counts['decoder_free_backlog_decisions_unscored'] += 1; continue
            tiers = rec.get('original_N_legal_total_tiers', rec.get('legal_total_tiers'))
            assert tiers and tiers == sorted(set(tiers)), 'missing or unordered original tiers'
            slack = min(t+GAP-now for t in latest)
            evaluated = [(t, max(0, t-d), predict(models, max(0, t-d), d, cd, cp)) for t in tiers]
            decode_cost = predict(models, 0, d, cd, 0)
            assert all(math.isfinite(y) and y >= 0 for _, _, y in evaluated), 'invalid tier model prediction'
            assert math.isfinite(decode_cost) and decode_cost >= 0, 'invalid decode-only model prediction'
            feasible = [(t, p, y) for t, p, y in evaluated if MARGIN*y <= slack]
            chosen = max(feasible, default=None)
            top, top_p, top_cost = evaluated[-1]
            training = calibration['training_ranges']['total_le_512' if top_p+d <= 512 else 'total_gt_512']['feature_ranges']
            coordinates = dict(zip(('P', 'D', 'Cdecode', 'Cprefill'), (top_p, d, cd, cp)))
            outside = {k: ('below' if v < training[k]['min'] else 'above')
                       for k, v in coordinates.items() if not training[k]['min'] <= v <= training[k]['max']}
            rows.append(dict(step=index, decision_time_s=now, D=d, Cd=cd, Cp=cp,
                backlog=rec['prefill_backlog_tokens'], slack_s=slack, max_total_tier=top,
                max_cap=top_p, max_cost_s=top_cost, max_margined_cost_s=MARGIN*top_cost,
                max_reserve_s=slack-MARGIN*top_cost, max_feasible=MARGIN*top_cost <= slack,
                selected_total_tier=None if chosen is None else chosen[0],
                selected_cap=None if chosen is None else chosen[1],
                selected_cost_s=None if chosen is None else chosen[2],
                selected_reserve_s=None if chosen is None else slack-MARGIN*chosen[2],
                decode_only_infeasible=MARGIN*decode_cost > slack,
                maximum_tier_training_coordinate_violations=outside,
                last_output_spread_s=max(latest)-min(latest), distinct_last_output_times=len(set(latest)),
                oldest_open_gap_s=now-min(latest), observed_host_step_s=rec['observed_host_step_s']))
        except (AssertionError, KeyError, TypeError, ValueError) as exc:
            errors.append(dict(step=index, error=str(exc)))
    def values(key): return [r[key] for r in rows if r[key] is not None]
    count = len(rows)
    return dict(cell=path.parent.name, raw_path=str(path), raw_sha256=sha(path),
        status='ALIGNMENT_OR_MODEL_FAILURE' if errors else 'CPU_DIAGNOSTIC', request_count=len(requests),
        counts=dict(counts), errors=errors, scored_backlog_decoder_decisions=count,
        maximum_legal_tier_feasible_count=sum(r['max_feasible'] for r in rows),
        maximum_legal_tier_feasible_fraction=sum(r['max_feasible'] for r in rows)/count if count else None,
        no_feasible_original_tier_count=sum(r['selected_total_tier'] is None for r in rows),
        selected_zero_prefill_count=sum(r['selected_cap'] == 0 for r in rows),
        decode_only_infeasible_count=sum(r['decode_only_infeasible'] for r in rows),
        nonpositive_latest_deadline_budget_count=sum(r['slack_s'] <= 0 for r in rows),
        synchronized_last_output_count=sum(r['distinct_last_output_times'] == 1 for r in rows),
        original_maximum_tier_distribution=dict(Counter(str(r['max_total_tier']) for r in rows)),
        maximum_tier_outside_training_coordinate_range_count=sum(bool(r['maximum_tier_training_coordinate_violations']) for r in rows),
        maximum_tier_training_coordinate_violations={k: dict(
            below=sum(r['maximum_tier_training_coordinate_violations'].get(k) == 'below' for r in rows),
            above=sum(r['maximum_tier_training_coordinate_violations'].get(k) == 'above' for r in rows))
            for k in ('P', 'D', 'Cdecode', 'Cprefill')},
        shadow_selected_total_tier_distribution=dict(Counter(str(r['selected_total_tier']) for r in rows)),
        shadow_selected_prefill_cap_distribution=dict(Counter(str(r['selected_cap']) for r in rows)),
        ranges={key: stats(values(key)) for key in ('D', 'Cd', 'Cp', 'backlog', 'slack_s', 'max_cap',
            'max_cost_s', 'max_margined_cost_s', 'max_reserve_s', 'selected_cap', 'selected_cost_s',
            'selected_reserve_s', 'last_output_spread_s', 'distinct_last_output_times',
            'oldest_open_gap_s', 'observed_host_step_s')},
        minimum_max_tier_reserve_example=min(rows, key=lambda r: r['max_reserve_s']) if rows else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory', nargs='?', type=Path, default=BASE/'cohort53005_03')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    calibration = BASE/'niyama_component/calibration.json'
    result = dict(status='CPU_DIAGNOSTIC', input_run_directory=str(args.run_directory),
        calibration_path=str(calibration), calibration_sha256=sha(calibration), script_sha256=sha(Path(__file__)),
        gap_deadline_s=GAP, model_margin=MARGIN,
        scope='Recorded states only; no future outputs, fitting, service simulation or full SLOWeave claim. '
              'Original N total tiers are enumerated. A saturated total tier 2552 means pure-prefill '
              'cap max(0,2552-D), not fixed pure-prefill 2048. Empty feasible sets receive no invented fallback. '
              'D=0 has no output deadline and is counted separately. All scored states have native backlog. '
              'Candidate costs are unexecuted predictions, not covered by actual-feature transfer validation; '
              'coordinate-wise training range inclusion does not establish joint-state support. Cp is the '
              'logged head estimate, while training uses executed aggregate prefill context; caps may underfill. '
              'Identical observed output timestamps do not establish GPU or network synchrony.',
        formula='min(previously emitted last_output_s + 0.1 - component_decision.decision_time_s); '
                'tier feasible iff 1.2*runtime_model(max(0,total-D),D,Cd,Cp)<=budget; '
                'decode-only feasibility uses runtime_model(0,D,Cd,0).',
        runs=[screen(args.run_directory/cell/'raw.json', json.loads(calibration.read_text())) for cell in CELLS])
    if any(r['errors'] for r in result['runs']): result['status'] = 'ALIGNMENT_OR_MODEL_FAILURE'
    destination = args.output or args.run_directory/'latest_deadline_screen.json'
    destination.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'], output=str(destination),
        runs=[{k: r[k] for k in ('cell', 'scored_backlog_decoder_decisions',
            'maximum_legal_tier_feasible_fraction', 'no_feasible_original_tier_count',
            'decode_only_infeasible_count', 'synchronized_last_output_count', 'errors')} for r in result['runs']]), indent=2))


if __name__ == '__main__': main()
