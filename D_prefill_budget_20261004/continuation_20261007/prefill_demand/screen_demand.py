#!/usr/bin/env python3
"""CPU shadow screen: cumulative FCFS prefill demand versus head-only demand.

Uses two original fixed1024 trajectories, never future arrivals/outputs/EOS.
This is an existing deadline-demand principle, not a new method or service run.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE/'fixed_half_53005'))
from screen_latest_deadline import predict, sha, stats

CELLS, CAPS, TTFT, MARGIN = ('00_fixed1024', '05_fixed1024'), (512, 1024, 2048), 4., 1.2


def screen(path, models):
    raw = json.loads(path.read_text())
    requests = {r['request_id']: r for r in raw['requests']}
    # Stable external-arrival sort is the actual episode submission order.
    order = sorted(requests, key=lambda rid: requests[rid]['arrival_s'])
    done = dict.fromkeys(requests, 0)
    rows, errors, counts = [], [], Counter()
    for index, step in enumerate(raw['steps']):
        rec = step.get('component_decision', {})
        prefills = [x for x in step['requests'] if x['prefill_tokens'] > 0]
        counts['steps'] += 1
        try:
            now = rec['decision_time_s']
            pending = [rid for rid in order if requests[rid]['add_s'] is not None
                       and requests[rid]['add_s'] <= now and done[rid] < requests[rid]['prompt_tokens']]
            remaining = [requests[rid]['prompt_tokens']-done[rid] for rid in pending]
            backlog = sum(remaining)
            assert rec['forward_completed'] and step['end_s'] is not None, 'unconfirmed forward'
            assert not step['preempted'], 'preemption invalidates cumulative reconstruction'
            assert rec['step_index'] == index and now <= step['end_s'], 'decision alignment mismatch'
            assert backlog == rec['prefill_backlog_tokens'] == step['prefill_backlog_tokens_before'], 'backlog sum mismatch'
            assert len(pending) == step['prefill_backlog_requests_before'], 'backlog count mismatch'
            assert [x['request_id'] for x in prefills] == pending[:len(prefills)], 'executed prefill is not FCFS prefix'
            assert all(x['computed_start'] == done[x['request_id']] for x in prefills), 'computed prompt prefix mismatch'
            assert all(done[x['request_id']]+x['prefill_tokens'] == requests[x['request_id']]['prompt_tokens']
                       for x in prefills[:-1]), 'FCFS request bypassed before prompt completion'
            expected_head = pending[0] if pending else None
            assert rec['head_prefill_id'] == expected_head, 'head ID mismatch'
            assert rec['head_prefill_context_estimate'] == (done[expected_head] if pending else 0), 'head context mismatch'
            d, cd, cp = rec['decode_count'], rec['decode_context_source'], rec['head_prefill_context_estimate']
            decoders = [x for x in step['requests'] if x['decode_tokens'] > 0
                        and x['computed_start'] >= requests[x['request_id']]['prompt_tokens']]
            assert len({x['request_id'] for x in decoders}) == len(decoders) == d == step['decode_count_before'], 'D mismatch'
            assert sum(x['computed_start']+1 for x in decoders) == cd, 'Cd mismatch'
            counts['aligned_steps'] += 1
            if not backlog:
                counts['no_backlog_steps'] += 1
            else:
                counts['backlog_steps'] += 1
                counts['backlog_steps_zero_actual_P'] += int(step['prefill_tokens'] == 0)
                slack = [requests[rid]['arrival_s']+TTFT-now for rid in pending]
                expired = sum(s <= 0 for s in slack)
                cumulative = 0
                rates = []
                for amount, s in zip(remaining, slack):
                    cumulative += amount
                    rates.append(cumulative/s if s > 0 else None)
                demand = None if expired else max(rates)
                head_demand = None if expired else remaining[0]/slack[0]
                costs = {c: predict(models, min(c, backlog), d, cd, cp) for c in CAPS}
                assert all(v > 0 for v in costs.values()), 'nonpositive model cost'
                capacity = {c: min(c, backlog)/(MARGIN*costs[c]) for c in CAPS}
                def choose(rate):
                    if expired: return 2048, 'expired_pending_deadline_progress_fallback'
                    feasible = [c for c in CAPS if capacity[c] >= rate]
                    return (min(feasible), 'minimum_feasible_cap') if feasible else (2048, 'capacity_insufficient_progress_fallback')
                cap, reason = choose(demand)
                head_cap, head_reason = choose(head_demand)
                rows.append(dict(step=index, decision_time_s=now, backlog=backlog, pending_count=len(pending),
                    D=d, Cd=cd, Cp=cp, expired_pending_count=expired, demand_tokens_per_s=demand,
                    head_demand_tokens_per_s=head_demand, head_request_id=expected_head,
                    limiting_request_id=None if expired else pending[rates.index(demand)],
                    cap=cap, reason=reason, head_cap=head_cap, head_reason=head_reason,
                    actual_original_P=step['prefill_tokens'], oldest_remaining_deadline_s=min(slack),
                    capacity_tokens_per_s=capacity, candidate_cost_s=costs))
        except (AssertionError, KeyError, TypeError, ValueError) as exc:
            errors.append(dict(step=index, error=str(exc)))
        # Advance solely with this original step's completed scheduled prompt tokens.
        for x in prefills: done[x['request_id']] += x['prefill_tokens']
    if any(done[rid] != r['prompt_tokens'] for rid, r in requests.items()):
        errors.append(dict(error='final cumulative prompt quantities mismatch'))
    def values(key): return [r[key] for r in rows if r[key] is not None]
    def distribution(key): return dict(Counter(str(r[key]) for r in rows))
    differences = [r for r in rows if r['cap'] != r['head_cap']]
    return dict(cell=path.parent.name, raw_path=str(path), raw_sha256=sha(path),
        status='ALIGNMENT_OR_MODEL_FAILURE' if errors else 'CPU_DIAGNOSTIC', request_count=len(requests),
        counts=dict(counts), errors=errors, scored_backlog_decisions=len(rows),
        cumulative_cap_distribution=distribution('cap'), head_only_cap_distribution=distribution('head_cap'),
        cumulative_reason_distribution=distribution('reason'), head_only_reason_distribution=distribution('head_reason'),
        cumulative_vs_fixed1024=dict(lower=sum(r['cap'] < 1024 for r in rows), same=sum(r['cap'] == 1024 for r in rows),
                                     higher=sum(r['cap'] > 1024 for r in rows)),
        original_P_above_shadow_cap_count=sum(r['actual_original_P'] > r['cap'] for r in rows),
        original_P_by_shadow_cap={str(c): stats([r['actual_original_P'] for r in rows if r['cap'] == c]) for c in CAPS},
        cumulative_vs_head_only=dict(different=len(differences), higher=sum(r['cap'] > r['head_cap'] for r in rows),
            lower=sum(r['cap'] < r['head_cap'] for r in rows),
            joint_cap_counts=dict(Counter(f"{r['cap']}/{r['head_cap']}" for r in rows))),
        nonhead_demand_limiter_count=sum(r['limiting_request_id'] not in (None, r['head_request_id']) for r in rows),
        consecutive_backlog_decision_cap_changes=sum(a['cap'] != b['cap'] for a, b in zip(rows, rows[1:])),
        expired_pending_decision_count=sum(r['expired_pending_count'] > 0 for r in rows),
        ranges={key: stats(values(key)) for key in ('backlog', 'pending_count', 'D', 'Cd', 'Cp',
            'demand_tokens_per_s', 'head_demand_tokens_per_s', 'oldest_remaining_deadline_s', 'actual_original_P')},
        candidate_capacity_tokens_per_s={str(c): stats([r['capacity_tokens_per_s'][c] for r in rows]) for c in CAPS},
        candidate_cost_s={str(c): stats([r['candidate_cost_s'][c] for r in rows]) for c in CAPS},
        different_head_only_examples=differences[:3],
        first_capacity_insufficient_example=next((r for r in rows if r['reason'].startswith('capacity_insufficient')), None))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory', nargs='?', type=Path, default=BASE/'cohort53005_03')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    calibration = BASE/'niyama_component/calibration.json'
    result = dict(status='CPU_DIAGNOSTIC', input_directory=str(args.run_directory), script_sha256=sha(Path(__file__)),
        calibration_path=str(calibration), calibration_sha256=sha(calibration),
        native_schedule_sha256=sha(args.run_directory/'patched_schedule.py'),
        shared_prediction_helper_sha256=sha(BASE/'fixed_half_53005/screen_latest_deadline.py'),
        caps=list(CAPS), ttft_s=TTFT, cost_margin=MARGIN,
        scope='Original fixed1024 states only, no alternate service outcomes or deployed actions. '
              'Known FCFS submission order is reconstructed from stable external-arrival order and actual add_s; '
              'executed prefill prefixes, head and total backlog are checked on every step. This establishes '
              'recorded-path consistency in the no-preemption/no-skip domain, not arbitrary scheduler behavior. '
              'Only past actual scheduled P reduces known prompt work. No future arrivals/outputs/EOS are used. '
              'An expired deadline or insufficient predicted capacity picks2048 for progress, not a guarantee. '
              'Frozen candidate-cost predictions/head-context approximation are not counterfactual measurements. '
              'Existing OA/EDF demand principles are not a novelty claim.',
        formula='r=max_j(sum_{i<=j}remaining_prompt_i)/(arrival_j+4-decision_time). '
                'Head-only changes only r. p_eff=min(cap,backlog); capacity=p_eff/(1.2*T(p_eff,D,Cd,headCp)); '
                'enumerate512/1024/2048 and choose minimum feasible cap; any nonpositive pending slack or '
                'no feasible cap falls back2048. No backlog is counted separately.',
        runs=[screen(args.run_directory/c/'raw.json', json.loads(calibration.read_text())['runtime_models']) for c in CELLS])
    if any(r['errors'] for r in result['runs']): result['status'] = 'ALIGNMENT_OR_MODEL_FAILURE'
    output = args.output or args.run_directory/'demand_screen.json'
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'], output=str(output), runs=[{k: r[k] for k in ('cell',
        'scored_backlog_decisions', 'cumulative_cap_distribution', 'head_only_cap_distribution',
        'cumulative_vs_head_only', 'cumulative_reason_distribution', 'errors')} for r in result['runs']]), indent=2))


if __name__ == '__main__': main()
