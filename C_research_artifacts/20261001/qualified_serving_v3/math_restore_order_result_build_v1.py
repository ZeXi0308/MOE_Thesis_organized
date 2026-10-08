#!/usr/bin/env python3
"""Aggregate the frozen three-arm recovery-order action experiment."""
import hashlib
import json
from pathlib import Path
import time

P = Path(__file__).resolve().parent


def read(name):
    return json.loads((P / name).read_text())


def sha(name):
    h = hashlib.sha256()
    with (P / name).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


arms, comparisons, mechanisms = {}, {}, {}
for mode in ('fixed', 'firstfit', 'minrecompute'):
    name = f'math_restore_order_{mode}_analysis_v1.json'
    a = read(name)
    assert a['integrity'] == 'VALID' and a['counts']['planned'] == 1024
    arms[mode] = dict(analysis=name, sha256=sha(name),
        **{k: a[k] for k in ('integrity', 'issues', 'counts', 'timing', 'episode', 'pressure_counts')},
        scheduled_work_totals=a['attribution']['totals'])
    if mode != 'fixed':
        name = f'math_restore_order_{mode}_evidence_v1.json'
        e = read(name)
        assert e['integrity'] == 'VALID'
        policy = json.loads((Path(a['run']) / 'restore-cost-order-policy.json').read_text())
        mechanisms[mode] = dict(evidence=name, sha256=sha(name), counts=e['counts'],
            queried_candidates_per_decision=e['queried_candidates_per_decision'],
            policy_step_minus_observer_call_id=e['policy_step_minus_observer_call_id'],
            successful_selected_restores_by_source_request=e['successful_selected_restores_by_source_request'],
            selector_wall_s=policy['selector_wall_s'],
            overhead_scope=policy['cost_scope'])
for suffix in ('fixed_within', 'fixed_vsfirstfit', 'fixed_vsminrecompute', 'firstfit_vsminrecompute'):
    name = f'math_restore_order_{suffix}_pair_v1.json'
    c = read(name)
    assert c['comparison_integrity'] in ('MATCHED', 'MATCHED_WITH_POLICY_CHANGE') and not c['issues']
    comparisons[suffix] = dict(file=name, sha256=sha(name),
        **{k: c[k] for k in ('comparison_integrity', 'episode_changes', 'count_changes',
            'timing_changes', 'per_request_directions', 'identical_output_subset_directions',
            'output_consistency', 'transitions')})

qualification = {}
candidate = arms['minrecompute']
for control in ('fixed', 'firstfit'):
    base = arms[control]
    ratio = dict(mean_flow=candidate['timing']['completion_s']['mean'] / base['timing']['completion_s']['mean'],
        slo_goodput=candidate['episode']['slo_goodput_requests_per_s'] / base['episode']['slo_goodput_requests_per_s'],
        output_token_rate=candidate['episode']['output_tokens_per_s'] / base['episode']['output_tokens_per_s'],
        maximum_host_gap=candidate['timing']['max_host_gap_s']['max'] / base['timing']['max_host_gap_s']['max'])
    guards = ratio['output_token_rate'] >= .97 and ratio['mean_flow'] <= 1.05
    qualification[control] = dict(ratios=ratio, efficiency_guardrails_pass=guards,
        flow_or_goodput_signal=guards and (ratio['mean_flow'] <= .97 or ratio['slo_goodput'] >= 1.03),
        worst_gap_tradeoff_signal=guards and ratio['maximum_host_gap'] <= .80,
        correctness_count_delta=candidate['counts']['correct'] - base['counts']['correct'],
        natural_eos_count_delta=candidate['counts']['natural_eos'] - base['counts']['natural_eos'])
signal = (all(x['flow_or_goodput_signal'] for x in qualification.values()) or
          all(x['worst_gap_tradeoff_signal'] for x in qualification.values()))
assert not signal, 'Unexpected positive signal; revise the decision rather than publishing a stale negative.'
snap = read('math_restore_order_terminal_snapshot_v1.json')
assert snap['parent'] is None and len(snap['arms']) == 3
assert all(a['launcher-receipt.json']['status'] == a['native/status.json']['status'] == 'COMPLETE'
           for a in snap['arms'].values())
result = dict(schema='c-math-restoration-order-result-v1', created_unix_s=time.time(),
    goal_status='ACTIVE_NOT_ACHIEVED', previous_turn_classification='PROGRESS', paper_status='NO_PAPER_GO',
    protocol='math_restore_order_protocol_v1.json', protocol_sha256=sha('math_restore_order_protocol_v1.json'),
    scope='Three exploratory natural-generation episodes of the same1024 development requests at1024maxseq/4096batch/8GiBKV/APC/fixed48. Full original arrivals retained.',
    new_valid_episodes=3, total_valid_math1024_episodes=23,
    arms=arms, comparisons=comparisons, mechanisms=mechanisms,
    predeclared_pilot_qualification=qualification, pilot_signal_qualifies=signal,
    decision=dict(status='SCOPED_NO_GO_FOR_MINIMUM_CURRENT_RECOMPUTE_ORDERING',
        actual_action=True, extra_information_changed_actions=True,
        outcome='70 actual cheaper-over-eligible-head decisions beyond3margin-only first-fit-style bypasses; no predeclared full-request qualification signal.',
        work_accounting='73 reordered restorations are not73 saved computations. All restored requests eventually execute; total actual recomputation changes only12122 to12063tokens, while output work changes200578 to200747tokens.',
        next='Do not tune scan limit/margin or expand this rule. Check observed APC cost drift within a single waiting episode; this determines whether to target cache-value loss or move intervention earlier to the preemption decision.'),
    runtime=dict(terminal_snapshot='math_restore_order_terminal_snapshot_v1.json',
        launcher_pid=74469, start_ticks='1207105093', status='ALL_THREE_COMPLETE_PARENT_EXITED', active_job=None,
        raw_archive='math_restore_order_raw_v1_20261002.tar.xz', archive_sha256=sha('math_restore_order_raw_v1_20261002.tar.xz'),
        archive_bytes=(P / 'math_restore_order_raw_v1_20261002.tar.xz').stat().st_size,
        large_logs_verified_after_decompression=9),
    limitations=['One development episode per new policy; no held-out generalization or independent-episode significance.',
        'Natural outputs change across policies; identical-output subsets are descriptive, not causal estimates.',
        'TTFT20s/max-successive-host-gap4s is a historical diagnostic point, not a business or final paper SLO.',
        'Policy margin deferrals are not native allocation failures. No actual bypass of a physically unfit full-known+1 head occurred.',
        'Known recompute tokens are current prefix work, not a remaining-output prediction or measured GPU time.',
        'First-fit and cost ordering are known simple actions; this experiment does not establish independent novelty.'])
with (P / 'math_restore_order_result_v1.json').open('x') as f:
    json.dump(result, f, indent=2, ensure_ascii=False, allow_nan=False)
    f.write('\n')
print(json.dumps(dict(status='COMPLETE', decision=result['decision']['status'], total_valid_episodes=23)))
