#!/usr/bin/env python3
"""Compare full declaration budget with an existing MC component adapted to async.

Accept an explicitly stopped two-arm prefix or the complete four-arm ABBA.
All 20 exploratory SLO points and natural-output differences are retained.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

from compare import arm_summary, compare, workload_difference
from compare_async_simple import admission_evidence, read_hashed


ORDER = ('probe-00-declaredbudget', 'probe-01-asyncmc',
         'probe-02-asyncmc', 'probe-03-declaredbudget')


def mc_evidence(report, raw):
    """Validate each MC stage, then join requested actions to real lifecycle events."""
    mc, decisions = report['mc_budget'], report['decisions']
    identity = {r[key]: r['request_id'] for r in raw['requests']
                for key in ('request_id', 'internal_request_id', 'external_request_id') if r.get(key)}

    def counts(rows):
        ids = {identity[r['request_id']] for r in rows}
        return dict(evaluations=len(rows), unique_requests=len(ids), request_ids=sorted(ids))

    attempted = [r for r in decisions if r.get('mc_attempted') is True]
    eligible = [r for r in attempted if r['mc_eligible'] is True]
    guarded = [r for r in attempted if r['mc_eligible'] is False]
    requested = [r for r in decisions if r.get('changed_by_mc_budget') is True]
    allocated = [r for r in requested if r['native_allocation_result'] is True]
    confirmed = [r for r in decisions if r.get('mc_admission_confirmed') is True]
    if mc['release_delay_nonempty_steps'] != 2:
        raise ValueError('The frozen async MC adaptation uses a two-nonempty-step release delay')
    if len(eligible)+len(guarded) != len(attempted) or any(
            r['native_fit'] is not True or r['baseline_allowed'] is not True or
            r['ordinary_budget_allowed'] is not False or r['release_delay_nonempty_steps'] != 2 or
            r['budget_after_if_admitted_blocks'] <= r['budget_limit_blocks'] for r in attempted):
        raise ValueError('MC attempts must be native-fit, cap-allowed ordinary full-budget denials')
    if any(not isinstance(r['mc_guard_reason'], str) or not r['mc_guard_reason'] or
           r['mc_peak_blocks'] is not None for r in guarded):
        raise ValueError('Guard failures cannot claim a computed peak')
    if any(r['mc_guard_reason'] is not None or type(r['mc_peak_blocks']) is not int or
           r['mc_peak_blocks'] != r['mc_envelope_peak_blocks']+r['mc_nonlive_physical_blocks']
           for r in eligible):
        raise ValueError('Eligible peak must include the recorded nonlive physical pages exactly once')
    accepted_peak = [r for r in eligible if r['mc_peak_blocks'] <= r['budget_limit_blocks']]
    rejected_peak = [r for r in eligible if r['mc_peak_blocks'] > r['budget_limit_blocks']]
    if {id(r) for r in requested} != {id(r) for r in accepted_peak} or any(
            r['denied'] or r['reason'] != 'async_mc_budget_allow' or r['final_allowed'] is not True
            for r in requested):
        raise ValueError('A requested relaxation must be the actually selected, eligible peak-permitted action')
    if any(r not in requested or r['native_allocation_result'] is not True or
           r['native_allocation_succeeded'] is not True for r in confirmed):
        raise ValueError('Confirmed MC admission requires an actual successful native allocation')
    stages = dict(attempted_evaluations=attempted, eligible_evaluations=eligible,
        peak_evaluations=eligible, peak_rejections=rejected_peak, requested_relaxations=requested,
        actual_allocations=allocated, successful_relaxations=confirmed)
    if any(type(mc[key]) is not int or mc[key] != len(rows) for key, rows in stages.items()):
        raise ValueError('MC aggregate counter disagrees with its decision records')
    if (mc['guard_fallback_counts'] != dict(Counter(r['mc_guard_reason'] for r in guarded)) or
            report['budget_overrides'] != len(confirmed)):
        raise ValueError('MC guard/confirmed-override counters disagree with decision records')
    first = mc['first_eligible']
    if bool(eligible) != (first is not None):
        raise ValueError('First eligible event availability disagrees with eligible decisions')
    if first is not None:
        index = first['decision_index']
        if (type(index) is not int or not 0 <= index < len(decisions) or
                decisions[index] is not eligible[0] or first['request_id'] != eligible[0]['request_id'] or
                first['t'] != eligible[0]['t'] or first['peak_blocks'] != eligible[0]['mc_peak_blocks']):
            raise ValueError('First eligible event does not identify the first actual eligible decision')
    starts = {r['request_id']: r['first_prefill_perf_s'] for r in report['starts']}
    origin, external_origin = report['origin_perf_s'], raw['measurement_origin_perf_counter_s']
    execution, allocation_matches, prefill_matches = [], [], []
    for index, row in enumerate(decisions):
        if row.get('changed_by_mc_budget') is not True:
            continue
        next_t = next((later['t'] for later in decisions[index+1:]
                       if later['request_id'] == row['request_id']), float('inf'))
        matches = [a for a in report['native_allocations'] if a['request_id'] == row['request_id']
                   and row['t'] <= a['t'] < next_t]
        actual = row['native_allocation_result']
        if len(matches) != (1 if type(actual) is bool else 0) or any(a['actual'] is not actual for a in matches):
            raise ValueError('Requested MC action does not match its corresponding native allocation event')
        stamp = starts.get(row['request_id'])
        prefill = actual is True and stamp is not None and stamp >= origin+matches[0]['t']
        if actual is True:
            allocation_matches.append(row)
        if prefill and row['mc_admission_confirmed']:
            prefill_matches.append(row)
        execution.append(dict(decision_index=index, request_id=identity[row['request_id']],
            internal_request_id=row['request_id'], decision_external_s=origin+row['t']-external_origin,
            observable_state={k: row.get(k) for k in ('active', 'running', 'free_blocks',
                'budget_before_blocks', 'budget_after_if_admitted_blocks', 'budget_limit_blocks',
                'mc_peak_blocks', 'mc_envelope_peak_blocks', 'mc_nonlive_physical_blocks', 'mc_peak_h')},
            ordinary_budget_allowed=row['ordinary_budget_allowed'], candidate_action=row['candidate_action'],
            final_action=row['final_action'], native_allocation_result=actual,
            matching_native_allocation=dict(matches[0]) if matches else None,
            mc_admission_confirmed=row['mc_admission_confirmed'],
            first_prefill_schedule_return_external_s=stamp-external_origin if prefill else None,
            execution_status='confirmed_with_prefill' if prefill and row['mc_admission_confirmed'] else
                'confirmed_without_observed_prefill' if row['mc_admission_confirmed'] else
                'allocated_without_confirmation' if actual is True else
                'native_allocation_failed' if actual is False else 'requested_without_observed_allocation'))
    return dict(reported_counters={key: mc[key] for key in stages},
        **{key: counts(rows) for key, rows in stages.items()},
        guard_fallbacks={reason: counts([r for r in guarded if r['mc_guard_reason'] == reason])
                        for reason in sorted(mc['guard_fallback_counts'])},
        matching_successful_native_allocations=counts(allocation_matches),
        confirmed_first_prefill_schedule_return_matches=counts(prefill_matches),
        requested_without_successful_allocation=counts([r for r in requested if r['native_allocation_result'] is not True]),
        allocated_without_confirmation=counts([r for r in allocated if not r['mc_admission_confirmed']]),
        confirmed_without_observed_prefill=counts([r for r in confirmed if r not in prefill_matches]),
        requested_execution_records=execution, first_eligible=first,
        first_eligible_final_decision=({k: decisions[first['decision_index']].get(k) for k in
            ('request_id', 'ordinary_budget_allowed', 'mc_peak_blocks', 'denied', 'reason',
             'changed_by_mc_budget', 'native_allocation_result', 'mc_admission_confirmed', 'final_action')}
            if first is not None else None),
        release_delay_nonempty_steps=mc['release_delay_nonempty_steps'], prior_work=mc['prior_work'],
        reported_semantics=mc['semantics'],
        semantics='Attempts and peak-permitted requests are not execution. successful_relaxations uses '
            'the controller definition: real allocation followed by admission confirmation. Corresponding '
            'native allocation is joined by request ID and its decision interval; first prefill is a later '
            'host schedule-return timestamp, not GPU completion. Missing allocation/confirmation/prefill '
            'remains explicit. Same-state ordinary-budget refusal is diagnostic, not a separately run '
            'baseline outcome. Guard counts are repeated evaluations, not independent experimental units.')


def build(analysis, source, source_sha, cell_root=None, status_path=None):
    selected = analysis['cells']
    order = tuple(Path(c['cell']).name for c in selected)
    if len(order) not in (2, 4) or order != ORDER[:len(order)]:
        raise ValueError('Expected the explicit two-arm prefix or four-arm declaredbudget/asyncmc ABBA')
    cells = dict(zip(order, selected))
    if len({c['workload_identity_sha256'] for c in selected}) != 1:
        raise ValueError('External workload identity differs across arms')
    group_root = cell_root if cell_root is not None else Path(selected[0]['cell']).parent
    status_path = status_path or group_root/'status.json'
    status, status_sha = read_hashed(status_path)
    stopped = status.get('stopped_for_zero_mc_actions')
    if len(order) == 2 and (status.get('status') != 'COMPLETE' or stopped is not True):
        raise ValueError('Two arms require COMPLETE plus the explicit zero-MC-action stop marker')
    if len(order) == 4 and (status.get('status') not in ('COMPLETE', 'FAILED') or stopped is True
            or (status.get('status') == 'COMPLETE' and stopped is not False)):
        raise ValueError('Four-arm results conflict with the recorded group terminal status')
    raws, arms, reference, resource_reference = {}, [], None, None
    for name in order:
        cell = cells[name]
        path = cell_root/name if cell_root is not None else Path(cell['cell'])
        config, config_sha = read_hashed(path/'config.json')
        report, report_sha = read_hashed(path/'admission.json')
        raw, raw_sha = read_hashed(path/'raw.json')
        if raw_sha != cell['raw_sha256']:
            raise ValueError('Raw changed after full-population analysis: '+name)
        is_mc = name.endswith('-asyncmc')
        mode = 'async_mc' if is_mc else 'declared_budget'
        expected_config = dict(admission_mode=mode, admission_cap=256, cap=256,
            async_scheduling=True, policy_intervention=True, native_running_limit=256,
            engine_max_num_seqs=256, target_usable_kv_blocks=32768, kv_floor=0,
            budget_blocks=32768, ignore_eos=False)
        expected_report = dict(mode=mode, cap=256, async_scheduling=True, policy_intervention=True,
            native_running_cap=256, budget_blocks=32768, block_size=16, kv_floor=0,
            max_signal_wait_s=None, probe_enabled=False, admission_count='complete_unique_unfinished')
        for actual, expected in ((config, expected_config), (report, expected_report)):
            if any(actual.get(k) != v or (type(v) is bool and actual.get(k) is not v)
                   for k, v in expected.items()):
                raise ValueError('Configuration differs from the explicit async MC design: '+name)
        backend = raw.get('async_observation')
        if not isinstance(backend, dict) or backend.get('async_scheduling') is not True:
            raise ValueError('Missing native async/backend observation: '+name)
        if len(order) == 2 and (raw.get('status') != 'COMPLETE' or backend.get('backend_drained') is not True
                or cell['arrived_requests'] != cell['planned_requests'] or cell['outcomes']['unfinished']):
            raise ValueError('Zero-action early stop requires complete observation and actual backend drain: '+name)
        resources = {k: config.get(k) for k in ('model', 'fixed_kv_cache_memory_bytes', 'intended_usable_kv_bytes')}
        if resource_reference is not None and resources != resource_reference:
            raise ValueError('Recorded model/resource budget differs across arms')
        resource_reference = resources
        if reference is not None:
            difference = workload_difference(raw, reference)
            if difference:
                raise ValueError('Workload mismatch: '+difference)
        reference = dict(requests=raw['requests'])
        raws[cell['cell']] = reference
        service = arm_summary(cell)
        controller = service.pop('admission')
        service.update(controller_overhead=controller.get('overhead'),
            latency_distributions={k: v['observed_only'] for k, v in cell['distributions'].items()},
            actual_preemption_count=cell['actual_preemption_count'], stop_reasons=cell['stop_reasons'],
            natural_stop_count=cell['natural_stop_count'], length_stop_count=cell['length_stop_count'],
            host_chunk_diagnostics=cell['host_chunk_diagnostics'],
            observation_end_s=cell['observation_end_s'], arrival_window_s=cell['arrival_window_s'])
        native = admission_evidence(report, raw)
        if is_mc:
            native['semantics'] += ' In the MC arm these are final decisions after the guarded MC relaxation; '
            native['semantics'] += 'requested relaxations are separately matched to executed allocation and first prefill below.'
        arms.append(dict(cell=name, source_cell=cell['cell'], read_cell=str(path.resolve()),
            raw_sha256=raw_sha, config_sha256=config_sha, admission_sha256=report_sha,
            declared_mode=mode, complete_admission_cap=256, native_running_limit=256,
            engine_compiled_max_num_seqs=256, physical_budget_blocks=32768, resource_config=resources,
            service_summary=service, async_observation=backend, admission_evidence=native,
            mc_evidence=mc_evidence(report, raw) if is_mc else None))
    first_mc = arms[1]['mc_evidence']
    if len(order) == 2 and first_mc['successful_relaxations']['evaluations'] != 0:
        raise ValueError('Early-stop marker conflicts with executed MC relaxations')
    pair_indices = ((1, 0),) if len(order) == 2 else ((1, 0), (2, 3))
    pairs = [compare(cells[ORDER[a]], cells[ORDER[b]], raws) for a, b in pair_indices]
    return dict(schema_version=1, source_analysis=str(source.resolve()), source_analysis_sha256=source_sha,
        design='ASYNC_DECLAREDBUDGET_MC_MC_DECLAREDBUDGET', independent_unit='run',
        observed_design='ZERO_EXECUTION_STOP_AFTER_TWO_ARMS' if len(order) == 2 else 'FOUR_ARM_ABBA',
        protocol_deviations=['Four arms ran although the first MC arm had zero confirmed relaxations.']
            if len(order) == 4 and first_mc['successful_relaxations']['evaluations'] == 0 else [],
        group_status=dict(path=str(status_path.resolve()), sha256=status_sha, contents=status),
        planned_arms=list(ORDER), observed_arms=list(order), unrun_arms=list(ORDER[len(order):]),
        fixed_pairs=[[ORDER[a], ORDER[b]] for a, b in pair_indices], arms=arms, pairs=pairs,
        interpretation='Existing MC future-peak component adapted to native async, not a new method '
            'or independent confirmation. Full declaration budget is the matched comparator; no '
            'historical native or fixed177 run is added. Complete/native/compiled maximum256 is common. '
            'Separate full strategies may have different internal states; no same-state branch claim. '
            'MC attempts, legal eligibility, guard failure, predicted peak, requested relaxation, actual '
            'allocation and first-prefill schedule-return are distinct. Native success does not prove GPU '
            'completion or a service benefit. All20 exploratory SLO thresholds and all arrivals, failures, '
            'timeouts and unfinished requests are retained. Unrun group arms contain no measured arrivals '
            'and are not counted as failed requests. Zero executed MC relaxations cannot explain latency '
            'differences. Backend tail and controller costs remain in observation time. Natural EOS/output '
            'differences preclude equal-work or quality claims; host-return gaps do not reveal intra-chunk '
            'GPU timing. One or two run-level pairs do not establish statistical significance.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('analysis', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cell-root', type=Path)
    parser.add_argument('--status', type=Path, help='Optional relocated group status.json.')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Refusing to overwrite an existing output')
    analysis, sha = read_hashed(args.analysis)
    result = build(analysis, args.analysis, sha, args.cell_root, args.status)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    for pair in result['pairs']:
        print(pair['candidate'], '/', pair['baseline'], 'mean flow delta',
              pair['relative_metrics']['flow_mean_s']['delta'], 'all20', pair['joint_goodput_all20'])
    print(args.output.resolve())


if __name__ == '__main__':
    main()
