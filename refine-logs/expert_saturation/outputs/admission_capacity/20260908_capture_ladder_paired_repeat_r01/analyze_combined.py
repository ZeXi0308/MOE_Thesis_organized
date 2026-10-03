#!/usr/bin/env python3
"""Run from the repository root; retain all comparisons and never overwrite outputs."""
from pathlib import Path
import hashlib
import json
import tarfile

base = Path('refine-logs/expert_saturation/outputs/admission_capacity')
original = base / '20260908_capture_ladder_paired_r01'
repeat = base / '20260908_capture_ladder_paired_repeat_r01'
read = lambda path: json.loads(path.read_text())
with tarfile.open(original / 'execution.tar.gz') as archive:
    expected = {name: hashlib.sha256(archive.extractfile(name).read()).hexdigest() for name in
                ['run_native_capacity.py', 'native_capture.py', 'admission_feedback.py', 'metrics.py']}
rows, source_matches, windows, targets = [], [], [], []
totals = dict(measurement_episodes=0, request_executions=0, warmup_raw_files=0, existing_decode_advances_checked=0)
for campaign, folder in [('original', original), ('repeat', repeat)]:
    analysis = read(folder / 'analysis/analysis.json')
    diagnostics = read(folder / 'analysis/diagnostics.json')
    assert analysis['status'] == 'MEASUREMENT_ONLY' and diagnostics['status'] == 'TARGETED_RAW_CHECKS_PASS'
    totals['measurement_episodes'] += len(analysis['cells'])
    totals['request_executions'] += diagnostics['request_executions']
    totals['warmup_raw_files'] += len(diagnostics['warmups'])
    totals['existing_decode_advances_checked'] += diagnostics['total_existing_decode_advances_checked']
    for block in ['forward', 'reverse']:
        assert read(folder / 'gpu_results' / block / 'environment.json')['source_sha256'] == expected
        source_matches.append(f'{campaign}/{block}')
    for pair in diagnostics['comparisons']:
        old, new = pair['legacy'], pair['aligned']
        rows.append(dict(campaign=campaign, engine=pair['engine'], regime=pair['regime'],
            legacy_goodput_rps=old['goodput_rps'], aligned_goodput_rps=new['goodput_rps'],
            best_static_goodput_rps=pair['best_static_goodput_rps'], best_static_caps=pair['best_static_caps'],
            legacy_slo_pass=old['n_slo_pass'], aligned_slo_pass=new['n_slo_pass'], n_requests=32,
            aligned_relative_legacy_goodput=pair['aligned_relative_legacy_goodput'],
            aligned_relative_best_static_goodput=pair['aligned_relative_best_static_goodput']))
    for cell in analysis['cells']:
        if (cell['plan']['policy'], cell['plan']['ladder_name'], cell['plan']['regime']) != ('feedback', 'aligned', 'steady'):
            continue
        raw = read(Path(cell['path']))
        first = next(d for d in raw['feedback_decisions'] if d['applied_change'])
        obs = {o['completed_step']: o for o in raw['feedback_observations']}
        steps = raw['scheduler_steps']
        window = dict(campaign=campaign, engine=cell['engine'], raw_path=cell['path'],
            first_applied_s=first['applied_s'], active_before=first['active_before'], waiting_before=first['waiting_before'],
            decision_index=first['decision_index'], signal_completed_steps=first['window_completed_steps'],
            signal_itl_ms=[obs[n]['step_median_itl_s'] * 1000 for n in first['window_completed_steps']],
            signal_decode_widths=[steps[n - 1]['decode_requests'] for n in first['window_completed_steps']],
            signal_prefill_tokens=[sum(r['prefill_tokens'] for r in steps[n - 1]['scheduled']) for n in first['window_completed_steps']])
        windows.append(window)
        diag = next(c for c in diagnostics['cells'] if c['index'] == cell['index'] and c['engine'] == cell['engine'])
        targets.append(dict(window=window, raw=raw, first=first, obs=obs, diag=diag))
summary = []
for regime in ['steady', 'bursty']:
    selected = [r for r in rows if r['regime'] == regime]
    summary.append(dict(regime=regime, n_engine_comparisons=len(selected),
        aligned_beats_legacy=sum(r['aligned_relative_legacy_goodput'] > 0 for r in selected),
        aligned_beats_best_static=sum(r['aligned_relative_best_static_goodput'] > 0 for r in selected),
        clears_frozen_three_percent_against_both=sum(r['aligned_relative_legacy_goodput'] >= .03 and r['aligned_relative_best_static_goodput'] >= .03 for r in selected),
        relative_legacy_range=[min(r['aligned_relative_legacy_goodput'] for r in selected), max(r['aligned_relative_legacy_goodput'] for r in selected)],
        relative_best_static_range=[min(r['aligned_relative_best_static_goodput'] for r in selected), max(r['aligned_relative_best_static_goodput'] for r in selected)]))
combined = dict(status='MEASUREMENT_ONLY_CURRENT_LADDER_REPLACEMENT_NOT_SUPPORTED',
    formulation='Ordinary four-step host-ITL feedback ladder [8,12,16,32] replaced by [8,16,24,32], otherwise frozen native OLMoE RTX5090 setup.',
    totals=totals, canonical='Original campaign forward remains canonical; all other engines are retained comparisons, never replacements.',
    source_of_truth=dict(archive=str(original / 'execution.tar.gz'), sha256=hashlib.sha256((original / 'execution.tar.gz').read_bytes()).hexdigest(), execution_source_sha256=expected, engines_matching_archive=source_matches),
    comparisons=rows, regime_summary=summary,
    verdict='No reproducible gain over the within-engine best measured static cap. Steady gain appeared in only one of four engines; bursty improves over legacy feedback in all four but loses to the best static point in all four.',
    stop='The single frozen full controlled repeat is complete. Do not run a third campaign or retune this ladder, threshold, seed or cohort.',
    evidence_ceiling='Single-model, single-GPU native vLLM in-process host request measurements; finite repeated cohort. Static envelope is hindsight, not a dynamic action Oracle.',
    failure_category='Current ladder replacement lacks stable full-request residual over a strong simple static baseline; this does not falsify admission or expert-aware scheduling families.',
    source_localization=dict(question='At the first feedback divergence, does the history window differ because of an arrival/prefill boundary, or because host/step latency differs at an otherwise matching request/token execution footprint?',
        next_smallest_experiment='Use only the four retained steady aligned raw traces: inspect the first threshold-crossing window and follow it to the first binding admission opportunity; align scheduled request IDs, computed-token ranges and decode/prefill footprint, then separate host gap from scheduler-start-to-receipt time. Report unmatched footprints explicitly; do not treat offline selector changes as policy outcomes.',
        first_trigger_windows=windows,
        current_boundary='All four first-trigger windows include one 128-token prefill, but active widths and pure-decode ITLs also differ. This localizes a candidate boundary and does not establish a prefill, padding, expert or hardware cause.'),
    resurrection_condition='A new natural operating regime or a newly localized action supplies reproducible full-request headroom beyond strong ordinary/static controls; renaming or retuning this ladder is insufficient.')


def footprint(step):
    return [(r['request_id'], r['computed_before'], r['prefill_tokens'], r['decode_tokens']) for r in step['scheduled']]


def step_info(step):
    return dict(step=step['step'], start_s=step['start_s'], end_s=step['end_s'],
        actual_active=step['actual_active'], running_before=step['running_before'], waiting_before=step['waiting_before'],
        waiting_after=step['waiting_requests'], target_cap=step['target_cap'], scheduled_signature=footprint(step))


reference = targets[0]
pairs, trigger_details = [], []
for target in targets:
    steps, first, obs = target['raw']['scheduler_steps'], target['first'], target['obs']
    binding_s = target['diag']['first_binding_opportunity_s']
    trigger_details.append(dict(campaign=target['window']['campaign'], engine=target['window']['engine'],
        raw_path=target['window']['raw_path'], first_action=first,
        history4=[dict(step_info(steps[n - 1]), observation_completed_step=n,
            received_s=obs[n]['received_s'], available_s=obs[n]['available_s'],
            step_median_itl_ms=obs[n]['step_median_itl_s'] * 1000) for n in first['window_completed_steps']],
        first_binding_opportunity=None if binding_s is None else step_info(next(s for s in steps if s['start_s'] == binding_s))))
    if target is reference:
        continue
    ref_steps = reference['raw']['scheduler_steps']
    common = min(len(ref_steps), len(steps))
    difference = next((i for i in range(common) if footprint(ref_steps[i]) != footprint(steps[i])), None)
    reason = 'ordered_scheduled_signature' if difference is not None else 'trace_length' if len(ref_steps) != len(steps) else 'none'
    if difference is None and len(ref_steps) != len(steps):
        difference = common
    a = ref_steps[difference] if difference is not None and difference < len(ref_steps) else None
    b = steps[difference] if difference is not None and difference < len(steps) else None
    pairs.append(dict(reference='original/forward', comparison=f"{target['window']['campaign']}/{target['window']['engine']}",
        first_observed_schedule_difference_step=difference, difference_kind=reason,
        content_differs_beyond_row_order=None if a is None or b is None else sorted(footprint(a)) != sorted(footprint(b)),
        reference_step=None if a is None else step_info(a), comparison_step=None if b is None else step_info(b),
        reference_first_action_step=reference['first']['decision_index'], comparison_first_action_step=first['decision_index'],
        difference_before_reference_first_action=difference is not None and difference < reference['first']['decision_index'],
        difference_before_comparison_first_action=difference is not None and difference < first['decision_index'],
        same_preaction_state_established=False))
branch = dict(status='OBSERVATIONAL_RECORDED_SCHEDULE_LOCALIZATION', reference='original/forward',
    signature_fields=['request_id', 'computed_before', 'prefill_tokens', 'decode_tokens'],
    comparison_rule='Compare ordered scheduled signatures at the same recorded scheduler step ordinal; preserve all three comparisons.',
    pairs=pairs, first_action_windows=trigger_details,
    limitations=['Independent engines and full policy reruns do not establish the same KV, hidden state or pre-action state.',
        'An earlier recorded schedule difference does not identify its upstream cause or prove later feedback actions have no effect.',
        'A binding opportunity is observed admission exposure, not a counterfactual performance effect.',
        'These captures cannot attribute differences to a kernel, padding or expert mechanism. No GPU execution or policy masking was performed.'])
for name, result in [('combined.json', combined), ('first_branch.json', branch)]:
    output = repeat / 'analysis' / name
    if output.exists():
        assert read(output) == json.loads(json.dumps(result)), f'existing {name} differs; write an addendum instead'
        print(f'Existing {name} retained unchanged')
    else:
        with output.open('x') as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write('\n')
        print(f'Wrote {output}')
print(json.dumps([{k: v for k, v in p.items() if k in ('comparison', 'first_observed_schedule_difference_step', 'difference_before_reference_first_action', 'difference_before_comparison_first_action')} for p in pairs]))
