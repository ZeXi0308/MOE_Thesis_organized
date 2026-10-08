#!/usr/bin/env python3
"""Summarize completed development episodes; never infer causal GPU speedup."""
import hashlib
import json
from pathlib import Path
import time

P = Path(__file__).resolve().parent


def read(name):
    return json.loads((P / name).read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write(name, value):
    with (P / name).open('x') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


names = {
    'native896_b2048': 'math_controls_native896_analysis_v1.json',
    'fixed896_b2048': 'math_new_bypass_fixed2048_analysis_v1.json',
    'bypass896_b2048': 'math_new_bypass_bypass2048_analysis_v1.json',
    'fixed896_b4096': 'math_new_bypass_fixed4096_analysis_v1.json',
    'bypass896_b4096': 'math_new_bypass_bypass4096_analysis_v1.json',
    'native1024_b4096': 'math_controls_native1024b4096_analysis_v1.json',
    'fixed1024_b4096': 'math_fixedmargin48_analysis_v1.json',
    'native768_b4096': 'math_batch4096_native768_analysis_v1.json',
}
arms, requests, grid = {}, {}, {}
for key, name in names.items():
    d = read(name)
    assert d['integrity'] == 'VALID' and d['counts']['planned'] == 1024
    requests[key] = {r['request_id']: r for r in d['per_request']}
    arms[key] = dict(analysis=name, analysis_sha256=sha(P / name),
        **{k: d[k] for k in ('integrity', 'counts', 'timing', 'episode', 'pressure_counts')},
        work_totals=d['attribution']['totals'])
    grid[key] = []
    for ttft in (5, 10, 20, 40):
        for gap in (.25, .5, 1, 2, 4, 8, 16):
            qualified = [r for r in requests[key].values() if r['completed'] and
                not r['issues'] and r['ttft_s'] is not None and r['max_host_gap_s'] is not None and
                r['ttft_s'] <= ttft and r['max_host_gap_s'] <= gap]
            correct = sum(r['correct'] for r in qualified)
            grid[key].append(dict(ttft_s=ttft, max_host_gap_s=gap, planned=1024,
                slo_pass=len(qualified), correct_and_slo=correct,
                slo_goodput=len(qualified) / d['episode']['duration_s'],
                correct_and_slo_goodput=correct / d['episode']['duration_s']))

mechanism = {}
for batch in (2048, 4096):
    name = f'math_new_bypass_bypass{batch}_evidence_v1.json'
    e = read(name)
    assert e['integrity'] == 'VALID'
    involved = []
    for r in e['joined_allocation_receipts']:
        if not r['confirmed_executed_bypass']:
            continue
        row = dict(call_id=r['observer_call_id'], scheduled_tokens=r['final_scheduled_tokens'])
        for role, rid in [('fresh', r['source_request_id']), ('restore_head', r['restore_head_source_id'])]:
            left = requests[f'fixed896_b{batch}'][rid]
            right = requests[f'bypass896_b{batch}'][rid]
            fields = ('completion_s', 'ttft_s', 'max_host_gap_s', 'correct', 'output_tokens', 'finish_reason')
            row[role] = dict(request_id=rid, fixed={k: left[k] for k in fields},
                            bypass={k: right[k] for k in fields})
        involved.append(row)
    mechanism[str(batch)] = dict(evidence=name, evidence_sha256=sha(P / name),
        counts=e['counts'], executed_actions_and_whole_request_outcomes=involved)

pairs = {}
for suffix in ('fixed_vs_bypass2048', 'fixed_vs_bypass4096',
               'strongfixed_vsfixed2048', 'strongfixed_vsfixed4096',
               'strongfixed_vsbypass2048', 'strongfixed_vsbypass4096',
               'strongnative_vsbypass4096', 'native896_vsfixed2048'):
    name = f'math_new_bypass_{suffix}_pair_v1.json'
    d = read(name)
    assert d['comparison_integrity'] in ('MATCHED', 'MATCHED_WITH_POLICY_CHANGE') and not d['issues']
    pairs[suffix] = dict(file=name, sha256=sha(P / name),
        **{k: d[k] for k in ('comparison_integrity', 'episode_changes', 'timing_changes',
            'count_changes', 'per_request_directions', 'identical_output_subset_directions',
            'output_consistency', 'transitions')})

result = dict(schema='c-math-new-bypass-result-v1', created_unix_s=time.time(),
    goal_status='ACTIVE_NOT_ACHIEVED', stage='EXPLORATORY_DEVELOPMENT',
    previous_turn_classification='PROGRESS', paper_status='NO_PAPER_GO',
    question='Does selecting a physically eligible fresh request behind a fixed48-deferred restore head improve complete requests?',
    scope='Four new finite-burst episodes of the same1024 development requests; same model,8GiBKV,APC,naturalEOS and1024outputcap.',
    protocol='math_new_bypass_protocol_v1.json', protocol_sha256=sha(P / 'math_new_bypass_protocol_v1.json'),
    new_valid_episodes=4, total_valid_math1024_episodes=20, arms=arms, mechanism=mechanism, pairs=pairs,
    descriptive_threshold_grid=grid,
    threshold_grid_scope='Post-run descriptive sensitivity, all cells retained. No selected new SLO, hypothesis test, or independent repetitions. Original20s/4s remains the historical diagnostic point.',
    decision=dict(status='SCOPED_NO_GO_FOR_CURRENT_NEW_BYPASS_RULE',
        rationale='Actual actions occurred at both batch budgets; evaluate complete outcomes rather than treating admission, fewer preemptions, or fewer recomputed tokens as gains.',
        retained_signal='Restoration timing remains useful in development, but its dynamic-state increment was already covered by a fixed-margin rule.',
        frontier_caution='The faster1024 configuration can have longer host gaps. Mean flow does not establish dominance over896 configurations.',
        next='Return to the strongest tested configuration for any next intervention; target the remaining restoration/preemption cost, with a simple recovery-cost ordering as a direct challenge. Do not expand this bypass rule into a large campaign.'),
    limitations=['No held-out confirmation or independent episode confidence interval.',
        'Natural outputs change across policies; same seed is not equal work.',
        'Within-episode request differences and identical-output subsets are descriptive, not independent randomized effects.',
        'All1024 original arrivals, including wrong and capped outputs, remain in denominators.',
        'Host-return gaps are not device ITL; token-work savings are not measured GPU-time savings.',
        'Queue backfill and recovery-cost ordering are existing ideas; no independent novelty claim.'])
write('math_new_bypass_result_v1.json', result)
print(json.dumps(dict(status='COMPLETE', episodes=4, total_math1024=20,
    result='math_new_bypass_result_v1.json', decision=result['decision']['status'])))
