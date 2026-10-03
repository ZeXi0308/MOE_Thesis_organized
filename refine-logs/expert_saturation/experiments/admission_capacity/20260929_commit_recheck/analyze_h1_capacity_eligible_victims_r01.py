"""Count funding alternatives in recorded H1 prepare rejections; no counterfactual rollout."""
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import statistics
import sys


HERE = Path(__file__).resolve().parent
PKG = HERE / 'candidate_h1_guard_qual_r02' / 'pkg'
sys.path.insert(0, str(PKG))
from absence_rotation import AbsenceRotation, RequestView, RotationConfig


def distribution(values):
    return dict(n=len(values), min=min(values), median=statistics.median(values), max=max(values)) if values else dict(n=0)


def analyze(source):
    data = json.loads(source.read_text())
    snapshots = defaultdict(list)
    decisions = defaultdict(list)
    for row in data['eligibility_snapshots']:
        snapshots[row['step']].append(row)
    for row in data['selector_decisions']:
        decisions[row['step']].append(row)
    events = [e for e in data['events'] if e['event'] == 'prepare_rejected'
              and e['reason'] == 'victim cannot fund target full history']
    rows, unknown = [], []
    for event in events:
        step = event['step']
        try:
            if len(snapshots[step]) != 1 or len(decisions[step]) != 1:
                raise ValueError('snapshot/decision not unique')
            s, recorded = snapshots[step][0], decisions[step][0]
            requests, state = s['requests'], s['tracker']
            cfg = RotationConfig(**state['config'])
            tracker = AbsenceRotation(config=cfg, victim_order='most_output',
                absent_since=dict(state['absent_since']), absence_count=dict(state['absence_count']),
                resident_since=dict(state['resident_since']), last_swap_step=state['last_swap_step'])
            views = {rid: RequestView(rid, requests[rid]['computed'], requests[rid]['prompt'],
                         requests[rid]['prompt'] + requests[rid]['max_output'], requests[rid]['output'])
                     for rid in s['running_ids']}
            waiting = [rid for rid in s['waiting_ids'] if requests[rid]['status'] == 'PREEMPTED']
            need = {rid: max(0, (requests[rid]['prompt'] + requests[rid]['output'] + 15) // 16
                                  - requests[rid]['held_blocks']) for rid in waiting}
            replay = tracker.decide(step, list(views.values()), waiting, s['free_blocks'], need)
            if asdict(replay) != recorded or replay.action != 'rotate':
                raise ValueError('frozen selector replay differs from recorded decision')
            if not (s['cohort_active'] and s['plan_target'] is None and s['plan_victim'] is None
                    and s['protected_id'] is None and not s['skipped_ids']
                    and all(requests[r]['output'] > 0
                            and requests[r]['computed'] == requests[r]['prompt'] + requests[r]['output'] - 1
                            for r in s['running_ids'])):
                raise ValueError('adapter selector preconditions do not match snapshot')
            target, victim, free = replay.resume_id, replay.victim_id, s['free_blocks']
            if free + requests[victim]['held_blocks'] >= need[target]:
                raise ValueError('recorded funding rejection not reproduced')
            eligible, alternatives, exclusions = [], [], Counter()
            for rid, view in views.items():
                r = requests[rid]
                # These defaults and comparisons are exactly those in frozen AbsenceRotation.decide.
                age = step - state['resident_since'].get(rid, -10**9)
                absences = state['absence_count'].get(rid, 0)
                reasons = []
                if view.progress >= cfg.protect_progress_fraction:
                    reasons.append('progress_protection')
                if absences >= cfg.max_absences_per_request:
                    reasons.append('max_absences')
                if age < cfg.min_residency_steps:
                    reasons.append('min_residency')
                exclusions.update(reasons)
                if reasons:
                    continue
                eligible.append(rid)
                if rid != victim and free + r['held_blocks'] >= need[target]:
                    alternatives.append(dict(request_id=rid, held_blocks=r['held_blocks'],
                        output_tokens=r['output'], computed_progress=view.progress,
                        recorded_resident_since=state['resident_since'].get(rid),
                        absence_count=absences, funding_surplus_blocks=free+r['held_blocks']-need[target],
                        visible_prepare_scalar_conditions_pass=(r['status'] == 'RUNNING'
                            and r['output'] + 1 < r['max_output']
                            and r['computed'] // 16 > 0 and r['held_blocks'] >= r['computed'] // 16)))
            if min(eligible, key=lambda rid: (-requests[rid]['output'], rid)) != victim:
                raise ValueError('replicated eligible set does not select recorded most-output victim')
            alternatives.sort(key=lambda row: (-row['output_tokens'], row['request_id']))
            rows.append(dict(step=step, host_perf_counter_s=s['host_perf_counter_s'], target=target,
                target_absence_steps=replay.absence_steps, free_blocks=free, target_required_blocks=need[target],
                selected_victim=victim, selected_held_blocks=requests[victim]['held_blocks'],
                selected_output_tokens=requests[victim]['output'],
                selected_funding_shortfall_blocks=need[target]-free-requests[victim]['held_blocks'],
                eligible_victim_count=len(eligible), excluded_constraint_counts=dict(exclusions),
                funding_alternative_count=len(alternatives), funding_alternatives=alternatives))
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
            unknown.append(dict(step=step, reason=str(error)))
    covered = [r for r in rows if r['funding_alternative_count']]
    by_target = defaultdict(lambda: dict(rejections=0, covered_rejections=0))
    for row in rows:
        by_target[row['target']]['rejections'] += 1
        by_target[row['target']]['covered_rejections'] += bool(row['funding_alternative_count'])
    runs = []
    for row in covered:
        if runs and row['step'] == runs[-1]['last_step'] + 1 and row['target'] == runs[-1]['target']:
            runs[-1]['last_step'] = row['step']
            runs[-1]['observations'] += 1
            runs[-1]['observed_endpoint_span_s'] = row['host_perf_counter_s'] - runs[-1]['first_host_perf_counter_s']
        else:
            runs.append(dict(target=row['target'], first_step=row['step'], last_step=row['step'],
                observations=1, first_host_perf_counter_s=row['host_perf_counter_s'], observed_endpoint_span_s=0.0))
    return dict(source=str(source), selector_source=str(PKG / 'absence_rotation.py'),
        evidence_level='STRUCTURAL_ON_RECORDED_NATIVE_STATES',
        summary=dict(matching_prepare_rejected_events=len(events), exact_reproduced_events=len(rows),
            unknown_or_mismatched_events=len(unknown), covered_by_other_eligible_victim=len(covered),
            uncovered_reproduced_events=len(rows)-len(covered),
            covered_fraction_of_reproduced=len(covered)/len(rows) if rows else None,
            distinct_targets=len(by_target), distinct_covered_targets=sum(v['covered_rejections'] > 0 for v in by_target.values()),
            covered_observations_with_visible_prepare_scalar_pass=sum(any(a['visible_prepare_scalar_conditions_pass']
                for a in r['funding_alternatives']) for r in covered),
            funding_alternative_count=distribution([r['funding_alternative_count'] for r in covered]),
            selected_shortfall_blocks=distribution([r['selected_funding_shortfall_blocks'] for r in covered]),
            covered_consecutive_same_target_runs=len(runs),
            consecutive_run_observation_count=distribution([r['observations'] for r in runs]),
            consecutive_run_endpoint_span_s=distribution([r['observed_endpoint_span_s'] for r in runs])),
        scope=['All matching rejected prepares in this diagnostic, without threshold changes.',
               'Alternative victims obey the same progress, residency and absence-count constraints.',
               'Counts are repeated observed scheduler states, not independent opportunities or saved time.',
               'Block identities, transfer state and subsequent native scheduling are not reconstructed.',
               'Funding sufficiency is not proof of native action, full-request benefit, or a new algorithm.'],
        by_target=dict(sorted(by_target.items())), covered_runs=runs, unknown_or_mismatched=unknown, observations=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=HERE / 'moe-a-h1-guard-qual-session-r02-20260930' /
        'cell-00-eager_diagnostic_on/archive/selective-store.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.source)
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps(result['summary'], indent=2))
