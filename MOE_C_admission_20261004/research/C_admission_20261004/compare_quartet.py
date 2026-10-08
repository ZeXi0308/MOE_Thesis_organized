#!/usr/bin/env python3
"""Pure service/output comparison for explicit four-arm diagnostic designs.

python compare_quartet.py ANALYSIS.json --design gc-liveness --output NEW.json
python compare_quartet.py ANALYSIS.json --design simple-recheck --output NEW.json
python compare_quartet.py ANALYSIS.json --design declared-budget --output NEW.json
python compare_quartet.py ANALYSIS.json --design declared-budget-matched --output NEW.json
python compare_quartet.py ANALYSIS.json --design mc-budget --output NEW.json
python compare_quartet.py ANALYSIS.json --design mc-budget-ablation --output NEW.json
python compare_quartet.py ANALYSIS.json --design mc-budget-phase --output NEW.json
Uses fixed pairs 01/00 and 02/03; never selects a threshold or overwrites results.
"""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path

from compare import arm_summary, compare, workload_difference


ORDERS = {
    'gc-liveness': ('probe-00-retained', 'probe-01-released',
                    'probe-02-released', 'probe-03-retained'),
    'simple-recheck': ('probe-00-fixed192', 'probe-01-kv256',
                       'probe-02-kv256', 'probe-03-fixed192'),
    'declared-budget': ('probe-00-fixed128', 'probe-01-declaredbudget',
                        'probe-02-declaredbudget', 'probe-03-fixed128'),
    'declared-budget-matched': ('probe-00-fixed177', 'probe-01-declaredbudget',
                                'probe-02-declaredbudget', 'probe-03-fixed177'),
    'mc-budget': ('probe-00-declaredbudget', 'probe-01-mcbudget',
                  'probe-02-mcbudget', 'probe-03-declaredbudget'),
    'mc-budget-ablation': ('probe-00-mcstatic', 'probe-01-mccapped',
                           'probe-02-mccapped', 'probe-03-mcstatic'),
    'mc-budget-phase': ('probe-00-mccapped', 'probe-01-mcphase',
                        'probe-02-mcphase', 'probe-03-mccapped'),
}


def read_hashed(path):
    payload = path.read_bytes()
    return json.loads(payload), hashlib.sha256(payload).hexdigest()


def validate_matched_calibration(config):
    """Validate the frozen development-only, declared-charge cap calibration."""
    calibration = config.get('declared_budget_matched_calibration')
    expected = dict(known_page_sum=70702, requests=384, uniform_charge_pages=185,
                    derived_fixed_cap=177, source='known_prompt_plus_declared_max')
    if not isinstance(calibration, dict) or any(calibration.get(key) != value
            for key, value in expected.items()):
        raise ValueError('Missing or inconsistent declared-budget matched calibration')
    mean = calibration.get('mean_pages')
    if (not isinstance(mean, (int, float)) or not math.isclose(mean, 70702/384,
            rel_tol=0., abs_tol=1e-10) or math.ceil(mean) != 185 or 32768//185 != 177):
        raise ValueError('Matched cap must follow floor(32768/ceil(70702/384))=177')
    return calibration


def budget_evidence(admission, counts, fixed_cap=128):
    """Only evaluated direct/FIFO rows; native scan breaks have no suffix count."""
    if fixed_cap not in (None, 128, 177):
        raise ValueError('Unsupported fixed-cap reference for the declared-budget design')
    decisions = admission['decisions']
    direct = [r for r in decisions if r['reason'] == 'declared_budget' and r['denied']]
    fifo = [r for r in decisions if r['reason'] == 'fifo_declared_budget' and r['denied']]
    changed = [r for r in decisions if r['changed_by_declared_budget']]
    if direct != changed or any(r['native_fit'] is not True or r['baseline_allowed'] is not True or
            r['base_allowed'] is not True or r['final_allowed'] is not False or
            r['budget_after_if_admitted_blocks'] <= r['budget_limit_blocks'] for r in direct):
        raise ValueError('Budget intervention rows disagree with native-fit/base allowance')
    if any(r['native_fit'] is not None or r['baseline_allowed'] is not None or
            r['base_allowed'] is not None or r['changed_by_declared_budget'] or not r['fifo_held'] for r in fifo):
        raise ValueError('FIFO budget rows cannot claim a native-fit counterfactual')
    breaks = [r for r in decisions if r.get('native_waiting_scan_break') is True]
    if admission['native_waiting_scan_breaks'] != len(breaks) or any(
            not r['denied'] or r.get('suffix_new_requests_not_evaluated') is not True for r in breaks):
        raise ValueError('Native waiting-scan break counter disagrees with evaluated rows')
    limit, peak, used = (admission[k] for k in ('budget_blocks', 'budget_peak_blocks', 'budget_used_blocks'))
    if (limit != 32768 or admission['block_size'] != 16 or not 0 <= used <= peak <= limit or
            admission['budget_overrides'] != 0 or admission['max_signal_wait_s'] is not None):
        raise ValueError('Budget report violates the declared no-override whole-pool configuration')
    expanded = [] if fixed_cap is None else [r for r in decisions if r['reason'] == 'allow' and r['denied'] is False
        and r.get('native_fit') is True and r.get('native_allocation_result') is True
        and r.get('final_allowed') is True and r.get('num_new_tokens', 0) > 0
        and r.get('load_kv_async') is False and fixed_cap <= r['active'] < 256]
    if any(r['active'] != r['admitted_inflight'] or
            r['budget_before_blocks'] + r['budget_required_blocks'] != r['budget_after_if_admitted_blocks'] or
            r['budget_after_if_admitted_blocks'] > r['budget_limit_blocks'] for r in expanded):
        raise ValueError('Successful expanded local admission has inconsistent active/budget accounting')
    starts = admission.get('starts')
    matched_starts = None
    if starts is not None:
        starts_by_id = {r['request_id']: r for r in starts}
        matched_starts = [r for r in expanded
            if isinstance(starts_by_id.get(r['request_id'], {}).get('first_prefill_perf_s'), (int, float))
            and starts_by_id[r['request_id']]['first_prefill_perf_s'] >= admission['origin_perf_s'] + r['t']]
    events = admission['budget_events']
    return dict(available=True, budget_blocks=limit, block_size=admission['block_size'],
        budget_overrides=admission['budget_overrides'], direct_budget_denials=counts(direct),
        direct_budget_denial_reference='Recorded native fit plus complete cap256 / KV floor0; '
            +(f'baseline_allowed and base_allowed do not mean fixed{fixed_cap} would allow.'
              if fixed_cap is not None else 'baseline_allowed and base_allowed precede the ordinary sum-budget test.'),
        **({f'direct_budget_denials_active_below{fixed_cap}':
            counts([r for r in direct if r['active'] < fixed_cap]),
        f'same_state_fixed{fixed_cap}':dict(
            **{f'successful_local_allocations_rejected_by_cap{fixed_cap}':counts(expanded)},
            first_prefill_schedule_return_matches=counts(matched_starts) if matched_starts is not None else None,
            allocations_without_matched_schedule_return=(len(expanded)-len(matched_starts)
                if matched_starts is not None else None),
            schedule_return_status='RECORDED_HOST_SCHEDULE_RETURN_NOT_GPU_EXECUTION'
                if starts is not None else 'STARTS_NOT_RECORDED_ALLOCATION_IS_NOT_EXECUTION',
            semantics=f'Apply complete cap{fixed_cap} to the actual budget-arm pre-allocation state: active>={fixed_cap} '
                'would reject, while this recorded new local allocation succeeded under cap256 and the '
                f'declared budget. This is a same-state rule diagnostic, not a replay of the fixed{fixed_cap} '
                'run or a service-benefit counterfactual. Async admissions and FIFO rows are excluded. '
                'Starts match the exact request ID and a host schedule-return time at/after the '
                'allocation decision; neither allocation nor that return proves GPU execution.')}
            if fixed_cap is not None else {}),
        fifo_budget_holds=counts(fifo), evaluated_budget_interventions=counts(direct+fifo),
        direct_budget_native_fit_true=sum(r['native_fit'] is True for r in direct),
        direct_budget_base_allowed_true=sum(r['base_allowed'] is True for r in direct),
        peak_charged_blocks=peak, peak_fraction_of_declared_limit=peak/limit,
        final_charged_blocks=used, final_live_charged_requests=len(admission['budget_live_charges']),
        charge_events=counts([e for e in events if e['action'] == 'charge']),
        release_events=counts([e for e in events if e['action'] == 'release']),
        native_waiting_scan_breaks=counts(breaks),
        native_waiting_scan_breaks_by_reason={reason:counts([r for r in breaks if r['reason'] == reason])
            for reason in ('cap', 'declared_budget')},
        skipped_suffix_requests=None,
        semantics='Ordinary budget denials are actual new-admission interventions, independent of '
            'changed_by_recovery=0. Repeated evaluations are not requests. Direct native-fit is the '
            'recorded read-only prediction; rejected allocations are not executed counterfactuals. '
            'FIFO holds have unknown native fit. A safe native scan break acts on its evaluated head '
            'and leaves the suffix unvisited; suffix requests/evaluations are unavailable, not zero '
            'or fabricated FIFO holds. The charged peak comes from actual charge events, not a sum '
            'of physical KV usage. No age override or per-request waiting bound is claimed.')


def mc_evidence(admission, counts, measurement_origin, phase_design=False):
    """A requested MC relaxation is neither allocation nor a first-prefill return."""
    report = admission['mc_budget']
    ablation = admission['mode'] in ('mc_budget_static', 'mc_budget_capped', 'mc_budget_phase')
    phase_design = phase_design or admission['mode'] == 'mc_budget_phase'
    use_future_peak = admission['mode'] != 'mc_budget_static'
    if ablation and (report['use_future_peak'] is not use_future_peak or
            admission['use_future_peak'] is not use_future_peak or
            report['maximum_declared_blocks'] != 39138 or
            admission['maximum_declared_blocks'] != 39138):
        raise ValueError('MC ablation must share a 39138-page ceiling with explicit peak mode')
    direct = [r for r in admission['decisions'] if not r['reason'].startswith('fifo_')]
    attempted = [r for r in direct if r['mc_attempted'] is True]
    eligible = [r for r in attempted if r['mc_eligible'] is True]
    fallback = [r for r in attempted if r['mc_eligible'] is False]
    requested = [r for r in direct if r['changed_by_mc_budget'] is True]
    peak_ready = [r for r in eligible if type(r['mc_peak_blocks']) is int]
    peak_rejected = [r for r in peak_ready if r['mc_peak_blocks'] > admission['budget_blocks']]
    expected_requested = [r for r in peak_ready if r['mc_peak_blocks'] <= admission['budget_blocks']]
    ceiling_rejected = []
    if ablation:
        if any(r['use_future_peak'] is not use_future_peak or
                r['maximum_declared_blocks'] != 39138 for r in direct):
            raise ValueError('MC direct decision parameters differ from their report')
        ceiling_rejected = [r for r in eligible if r['declared_ceiling_denied'] is True]
        ceiling_allowed = [r for r in eligible if r['declared_ceiling_allowed'] is True]
        eligible_indices = {id(r) for r in eligible}
        if (any(r['declared_ceiling_allowed'] is not
                (r['budget_after_if_admitted_blocks'] <= 39138) or
                r['declared_ceiling_denied'] is not (not r['declared_ceiling_allowed'])
                for r in eligible) or any(r['declared_ceiling_allowed'] is not None or
                r['declared_ceiling_denied'] is not False for r in direct if id(r) not in eligible_indices)):
            raise ValueError('MC ceiling decision disagrees with the guarded declared sum')
        expected_peak = ceiling_allowed if use_future_peak else []
        expected_peak_indices = {id(r) for r in expected_peak}
        if (peak_ready != expected_peak or
                [r for r in direct if r['mc_peak_computed'] is True] != expected_peak or
                any(r['mc_peak_blocks'] is not None or r['mc_peak_h'] is not None
                    for r in direct if id(r) not in expected_peak_indices) or
                any(r['mc_limit_reason'] != 'declared_ceiling' for r in ceiling_rejected) or
                any(r['mc_limit_reason'] != 'future_peak' for r in peak_rejected)):
            raise ValueError('MC peak evidence must match only actually computed eligible states')
        expected_requested = ([r for r in peak_ready if r['mc_peak_blocks'] <= admission['budget_blocks']]
                              if use_future_peak else ceiling_allowed)
        if any(r['mc_limit_reason'] is not None for r in expected_requested+fallback):
            raise ValueError('Allowed or guard-fallback rows cannot claim a ceiling/peak rejection')
    if (requested != expected_requested or any(r['ordinary_budget_allowed'] is not False or
            r['native_fit'] is not True or r['base_allowed'] is not True or
            r['budget_after_if_admitted_blocks'] <= r['budget_limit_blocks'] for r in attempted) or
            any(r['reason'] != ('mc_budget_allow' if use_future_peak else 'mc_budget_static_allow') or r['denied'] is not False or
                r['final_allowed'] is not True or r['changed_by_declared_budget'] for r in requested)):
        raise ValueError('MC relaxation rows disagree with ordinary-budget/native/peak conditions')
    actual = [r for r in requested if r.get('native_allocation_result') is True]
    failed = [r for r in requested if r.get('native_allocation_result') is False]
    unresolved = [r for r in requested if type(r.get('native_allocation_result')) is not bool]
    charges = [e for e in admission['budget_events'] if e['action'] == 'charge']
    extra_charges = [e for e in charges if e['mc_relaxation'] is True]
    actual_ids = {r['request_id'] for r in actual}
    if any(e['request_id'] not in actual_ids for e in extra_charges):
        raise ValueError('MC charge has no actual successful relaxation allocation')
    counters = dict(attempted_evaluations=len(attempted), eligible_evaluations=len(eligible),
        requested_relaxations=len(requested), successful_relaxations=len(extra_charges),
        peak_rejections=len(peak_rejected),
        guard_fallback_counts=dict(collections.Counter(r['mc_guard_reason'] for r in fallback)))
    if ablation:
        counters.update(peak_evaluations=len(peak_ready), declared_ceiling_rejections=len(ceiling_rejected))
    if phase_design:
        allow_scheduled_prefill = admission['mode'] == 'mc_budget_phase'
        if (not use_future_peak or admission['allow_scheduled_prefill'] is not allow_scheduled_prefill or
                report['allow_scheduled_prefill'] is not allow_scheduled_prefill or
                any(r['allow_scheduled_prefill'] is not allow_scheduled_prefill for r in direct)):
            raise ValueError('Phase allowance differs from the explicit strict/phase arm mode')
        eligible_indices = {id(r) for r in eligible}
        if any((type(r['mc_scheduled_prefill_count']) is not int or r['mc_scheduled_prefill_count'] < 0 or
                (not allow_scheduled_prefill and r['mc_scheduled_prefill_count'] != 0))
                if id(r) in eligible_indices else r['mc_scheduled_prefill_count'] is not None for r in direct):
            raise ValueError('Scheduled-prefill counts require successful full guard evaluation')
        phase_requested = [r for r in direct if r['changed_by_scheduled_prefill'] is True]
        expected_phase = [r for r in requested if allow_scheduled_prefill and r['mc_scheduled_prefill_count'] > 0]
        if phase_requested != expected_phase:
            raise ValueError('Phase actions must be final allowances beyond the strict phase guard')
        phase_actual = [r for r in phase_requested if r.get('native_allocation_result') is True]
        phase_failed = [r for r in phase_requested if r.get('native_allocation_result') is False]
        phase_unresolved = [r for r in phase_requested if type(r.get('native_allocation_result')) is not bool]
        phase_charges = [e for e in charges if e['scheduled_prefill_relaxation'] is True]
        phase_actual_ids = {r['request_id'] for r in phase_actual}
        if any(e['request_id'] not in phase_actual_ids or e['mc_relaxation'] is not True for e in phase_charges):
            raise ValueError('Phase charge lacks an actual successful phase allocation')
        counters.update(scheduled_prefill_requested_relaxations=len(phase_requested),
                        scheduled_prefill_successful_relaxations=len(phase_charges))
    if any(report[k] != value for k, value in counters.items()):
        raise ValueError('MC counters disagree with recorded decisions/admission charges')
    limit, peak, used = (admission[k] for k in ('budget_blocks', 'budget_peak_blocks', 'budget_used_blocks'))
    if (limit != 32768 or admission['block_size'] != 16 or not 0 <= used <= peak or
            admission['budget_overrides'] != len(extra_charges) or admission['max_signal_wait_s'] is not None):
        raise ValueError('MC declared-charge accounting/configuration is inconsistent')
    if ablation and peak > 39138:
        raise ValueError('MC ablation exceeded its shared declared-charge ceiling')
    starts = admission.get('starts')
    starts_by_id = {s['request_id']:s for s in starts} if starts is not None else {}
    matched = [r for r in actual if isinstance(starts_by_id.get(r['request_id'], {}).get(
        'first_prefill_perf_s'), (int,float)) and starts_by_id[r['request_id']]['first_prefill_perf_s'] >=
        admission['origin_perf_s']+r['t']]

    def first(rows):
        if not rows:
            return None
        row = rows[0]
        start = starts_by_id.get(row['request_id'], {}).get('first_prefill_perf_s')
        result = dict(request_id=counts([row])['request_ids'][0], gate_relative_s=row['t'],
            external_s=admission['origin_perf_s']-measurement_origin+row['t'],
            active=row['active'], free_blocks=row['free_blocks'],
            ordinary_sum_after_blocks=row['budget_after_if_admitted_blocks'],
            mc_peak_blocks=row['mc_peak_blocks'], mc_peak_h=row['mc_peak_h'],
            native_allocation_result=row.get('native_allocation_result'),
            first_prefill_schedule_return_external_s=start-measurement_origin if start is not None else None)
        if phase_design:
            result.update(allow_scheduled_prefill=row['allow_scheduled_prefill'],
                          mc_scheduled_prefill_count=row['mc_scheduled_prefill_count'])
        return result

    fifo = [r for r in admission['decisions'] if r['reason'] == 'fifo_declared_budget' and r['denied']]
    breaks = [r for r in direct if r.get('native_waiting_scan_break') is True]
    if (admission['native_waiting_scan_breaks'] != len(breaks) or any(not r['denied'] or
            r.get('suffix_new_requests_not_evaluated') is not True for r in breaks)):
        raise ValueError('MC native scan-break counter disagrees with evaluated rows')
    first_eligible = report['first_eligible']
    if ablation and not use_future_peak and first_eligible is not None and first_eligible['envelope'] is not None:
        raise ValueError('Static ablation must not report an uncomputed first envelope')
    result = dict(available=True, prior_work=report['prior_work'], source_sha256=report['source_sha256'],
        counters=counters, attempted=counts(attempted), eligible=counts(eligible),
        eligibility_unavailable=counts([r for r in attempted if type(r['mc_eligible']) is not bool]),
        eligible_peak_unavailable=counts([r for r in eligible if type(r['mc_peak_blocks']) is not int]),
        guard_fallbacks_by_reason={reason:counts([r for r in fallback if r['mc_guard_reason'] == reason])
            for reason in counters['guard_fallback_counts']}, peak_rejections=counts(peak_rejected),
        requested_relaxations=counts(requested), actual_successful_allocations=counts(actual),
        actual_failed_allocations=counts(failed), allocation_result_unavailable=counts(unresolved),
        actual_relaxation_admission_charges=counts(extra_charges),
        first_prefill_schedule_return_matches=counts(matched) if starts is not None else None,
        allocations_without_matched_schedule_return=len(actual)-len(matched) if starts is not None else None,
        first_requested_relaxation=first(requested), first_actual_successful_relaxation=first(actual),
        first_eligible=(dict(request_id=counts([first_eligible])['request_ids'][0],
            external_s=admission['origin_perf_s']-measurement_origin+first_eligible['t'],
            old_requests=len(first_eligible['old_rows']), free_blocks=first_eligible['free_blocks'],
            envelope=first_eligible['envelope']) if first_eligible is not None else None),
        ordinary_budget_denials_retained=counts([r for r in direct if r['reason'] == 'declared_budget' and r['denied']]),
        fifo_budget_holds=counts(fifo), native_waiting_scan_breaks=counts(breaks), skipped_suffix_requests=None,
        budget_limit_blocks=limit, peak_declared_sum_blocks=peak, final_declared_sum_blocks=used,
        final_live_charged_requests=len(admission['budget_live_charges']),
        budget_overrides_actual_admissions=admission['budget_overrides'], charge_events=counts(charges),
        release_events=counts([e for e in admission['budget_events'] if e['action'] == 'release']),
        semantics='Ordinary sum would reject the requested relaxations at their observed state. '
            'Requested decisions, native allocation successes, admission charges and host first-prefill '
            'schedule returns are separate evidence; none establishes identical baseline GPU state or '
            'net service benefit. Host schedule return is not GPU execution. The recorded declared sum '
            'may exceed32768 after MC admission; it is not physical occupancy or a peak-envelope violation. '
            'budget_overrides counts actual MC admissions, not age bypass. FIFO has no native-fit '
            'counterfactual; unvisited suffix requests remain unavailable. No fixed-cap shadow is used.')
    if ablation:
        if result['first_eligible'] is not None:
            result['first_eligible'].update(
                declared_ceiling_allowed=first_eligible['declared_ceiling_allowed'],
                mc_peak_computed=first_eligible['mc_peak_computed'],
                semantics='First common production-guard success; not necessarily ceiling or peak allowance.')
        result.update(use_future_peak=use_future_peak, maximum_declared_blocks=39138,
            declared_ceiling_rejections=counts(ceiling_rejected),
            peak_evaluations=counts(peak_ready),
            peak_evaluation_status='COMPUTED_AFTER_COMMON_GUARD_AND_CEILING' if use_future_peak else 'NOT_COMPUTED_STATIC_ARM',
            same_state_other_rule=(dict(available=True, reference='same common guard and 39138-page static ceiling',
                both_allow=counts(requested), static_allow_future_peak_deny=counts(peak_rejected),
                both_deny_at_ceiling=counts(ceiling_rejected),
                semantics='Only evaluated legal same-state decisions; rejected allocations were not executed. '
                    'This does not replay the static trajectory or estimate service benefit.') if use_future_peak else
                dict(available=False, status='FUTURE_PEAK_NOT_COMPUTED', future_peak_blocks=None,
                     future_peak_decisions=None)),
            semantics='Structure-matched oversubscription ablation: both rules use the same native fit, '
                'complete cap256, production guard and maximum declared charge39138; physical budget32768. '
                'The static arm does not compute a future peak, so its peak values and opposite-rule '
                'decisions are unknown, not zero or inferred safe. The capped arm additionally requires '
                'its computed future peak<=32768. Guard fallback and ceiling rejection are normal control '
                'outcomes. Requested relaxations, actual allocations, admission charges and host '
                'first-prefill returns are separate evidence; host return is not GPU execution. '
                'No same-state diagnostic establishes a service counterfactual. Repeated evaluations '
                'are not unique requests; unvisited queue suffixes remain unavailable. No fixed-cap '
                'shadow is used. The single39138 ceiling was calibrated on prior observed development '
                'charge peaks, not selected using this test run.')
    if phase_design:
        if result['first_eligible'] is not None:
            result['first_eligible'].update(allow_scheduled_prefill=first_eligible['allow_scheduled_prefill'],
                mc_scheduled_prefill_count=first_eligible['mc_scheduled_prefill_count'])
        matched_indices = {id(r) for r in matched}
        phase_matched = [r for r in phase_actual if id(r) in matched_indices]
        recorded_first = report['first_scheduled_prefill_relaxation']
        if (recorded_first is None) != (not phase_requested):
            raise ValueError('First scheduled-prefill relaxation disagrees with actual requested rows')
        if recorded_first is not None:
            index = recorded_first['decision_index']
            if (type(index) is not int or not 0 <= index < len(admission['decisions']) or
                    admission['decisions'][index] is not phase_requested[0] or
                    recorded_first['request_id'] != phase_requested[0]['request_id']):
                raise ValueError('First phase request must reference its exact original decision')
        phase_first = first(phase_requested)
        if phase_first is not None:
            phase_first.update(mc_scheduled_prefill_count=phase_requested[0]['mc_scheduled_prefill_count'],
                               decision_index=recorded_first['decision_index'])
        result.update(allow_scheduled_prefill=allow_scheduled_prefill,
            scheduled_prefill=dict(available=True, allowed=allow_scheduled_prefill,
                requested_relaxations=counts(phase_requested), actual_successful_allocations=counts(phase_actual),
                actual_failed_allocations=counts(phase_failed), allocation_result_unavailable=counts(phase_unresolved),
                actual_admission_charges=counts(phase_charges),
                first_prefill_schedule_return_matches=counts(phase_matched) if starts is not None else None,
                allocations_without_matched_schedule_return=len(phase_actual)-len(phase_matched) if starts is not None else None,
                first_requested_relaxation=phase_first, first_actual_successful_relaxation=first(phase_actual),
                eligible_old_scheduled_prefill_count_max=max((r['mc_scheduled_prefill_count'] for r in eligible), default=None),
                semantics='Changed phase rows passed native fit/cap, complete production guards, the shared '
                    'declared ceiling and future peak. Strict phase would reject their observed state; '
                    'this is not a replay of the strict run. Counts distinguish requests from repeated '
                    'evaluations, actual allocation, charge, and host first-prefill return. A scheduled '
                    'complete prompt is not a claim that its GPU execution has already finished.'),
            same_state_other_rule=(dict(available=True, reference='strict decode-phase guard on the same observed state',
                strict_would_reject_phase_allowed=counts(phase_requested),
                semantics='Only phase-added final allowances are identified; no strict-trajectory or '
                    'service-benefit counterfactual is reconstructed.') if allow_scheduled_prefill else
                dict(available=False, status='RELAXED_PHASE_GUARD_NOT_EVALUATED_ON_STRICT_FALLBACKS',
                     phase_rule_decisions=None)),
            semantics='Near-neighbor production phase adaptation, not a new admission principle. Both arms '
                'use cap/native/compiled256, physical budget32768, declared ceiling39138, future peaks, '
                'KV floor0 and no age override. The phase arm additionally accepts complete prompts '
                'scheduled to finish in the current step; genuinely incomplete prefill remains rejected. '
                'Running scheduling, native allocation and quantum are unchanged. Actual phase-added '
                'decisions, successful allocation, charges and first-prefill host returns are separately '
                'reported. Same-state strict rejection is not a full-run counterfactual; no execution '
                'completion or service gain follows from scheduled tokens alone.')
    return result


def gate_evidence(cell, raw, fixed_cap=128, phase_design=False):
    admission, sha = read_hashed(Path(cell['cell'])/'admission.json')
    identity = {request[key]: request['request_id'] for request in raw['requests']
                for key in ('request_id', 'internal_request_id', 'external_request_id')
                if request.get(key)}
    decisions = admission['decisions']

    def counts(rows):
        ids = set()
        for row in rows:
            if row['request_id'] not in identity:
                raise ValueError('Unmapped request identity in admission report')
            ids.add(identity[row['request_id']])
        return dict(evaluations=len(rows), unique_requests=len(ids), request_ids=sorted(ids))

    denied = [row for row in decisions if row.get('denied') is True]
    direct = [row for row in denied if not row['reason'].startswith('fifo_')]
    fifo = [row for row in denied if row['reason'].startswith('fifo_')]
    by_reason = {reason: counts([row for row in denied if row['reason'] == reason])
                 for reason in ('cap', 'kv', 'fifo_cap', 'fifo_kv')}
    extra_reasons = {'probe', 'probe_progress', 'fifo_probe', 'reservation',
                     'fifo_reservation', 'headroom', 'fifo_headroom'}
    extra_flags = ('changed_by_recovery', 'changed_by_progress', 'probe_fifo_held',
                   'changed_by_reservation', 'reservation_probe_fifo_held',
                   'changed_by_headroom', 'headroom_probe_fifo_held')
    extra = [row for row in decisions if (row.get('denied') is True and
             row.get('reason') in extra_reasons) or any(row.get(flag) is True for flag in extra_flags)]
    allocations = admission['native_allocations']
    mismatches = sum(row['predicted'] != row['actual'] for row in allocations)
    unknown = sum(type(row.get('predicted')) is not bool or type(row.get('actual')) is not bool
                  for row in allocations)
    agrees = (admission['fit_checks'] == len(allocations) and
              admission['fit_mismatches'] == mismatches)
    states = admission['snapshots'] + decisions
    peak = lambda key: max((row[key] for row in states if key in row), default=None)
    direct_base = [row for row in direct if row['reason'] in ('cap', 'kv')]
    result = dict(admission_sha256=sha, direct_denials=counts(direct), fifo_holds=counts(fifo),
        denials_by_reason=by_reason, direct_cap_kv_native_fit_true=sum(
            row.get('native_fit') is True for row in direct_base),
        direct_cap_kv_without_proven_fit=sum(row.get('native_fit') is not True for row in direct_base),
        native_capacity_evaluations=sum(row.get('reason') == 'native_capacity' for row in decisions),
        extra_probe_actions=counts(extra), probe_enabled=admission['probe_enabled'],
        fit_validation=dict(reported_checks=admission['fit_checks'],
            recorded_allocations=len(allocations), reported_mismatches=admission['fit_mismatches'],
            observed_mismatches=mismatches, unknown_boolean_results=unknown, counters_agree=agrees,
            actual_successes=sum(row['actual'] is True for row in allocations),
            actual_failures=sum(row['actual'] is False for row in allocations),
            status='CHECKED_EXECUTED_ALLOCATIONS_ONLY' if agrees and not mismatches and not unknown
                   else 'MISMATCH_OR_INCONSISTENT'),
        peak_recorded_complete_active=peak('active'),
        peak_recorded_admitted_inflight=peak('admitted_inflight'),
        active_definition=admission['active_definition'],
        semantics='Counts are actual recorded denied/held gate evaluations, with canonical request IDs. '
            'Repeated evaluations are not distinct requests. FIFO reasons identify the existing '
            'barrier; FIFO has no native-fit counterfactual and does not prove that each held request '
            'would otherwise allocate. Native fit is checked only where allocation actually executed, '
            'including recovery; a deferred fit has no executed counterfactual. Active peaks use '
            'recorded snapshots/decision states, not an exact continuous-time peak.')
    if admission['mode'] == 'declared_budget':
        result['declared_budget'] = budget_evidence(admission, counts, fixed_cap)
        result['denials_by_reason'].update({reason:counts([r for r in denied if r['reason'] == reason])
            for reason in ('declared_budget', 'fifo_declared_budget')})
    elif admission['mode'] in ('mc_budget', 'mc_budget_static', 'mc_budget_capped', 'mc_budget_phase'):
        result['mc_budget'] = mc_evidence(admission, counts, raw['measurement_origin_perf_counter_s'], phase_design)
        result['denials_by_reason'].update({reason:counts([r for r in denied if r['reason'] == reason])
            for reason in ('declared_budget', 'fifo_declared_budget')})
    return result


def metadata(cell, design):
    path = Path(cell['cell'])
    config, config_sha = read_hashed(path/'config.json')
    setup, setup_sha = read_hashed(path/'liveness-setup.json')
    gate = cell['admission']['configuration']
    kv_arm = design == 'simple-recheck' and path.name.endswith('kv256')
    budget_design = design in ('declared-budget', 'declared-budget-matched')
    mc_ablation = design == 'mc-budget-ablation'
    mc_phase = design == 'mc-budget-phase'
    mc_design = design in ('mc-budget', 'mc-budget-ablation', 'mc-budget-phase')
    mc_arm = mc_ablation or mc_phase or (mc_design and path.name.endswith('mcbudget'))
    mc_mode = (('mc_budget_phase' if path.name.endswith('mcphase') else 'mc_budget_capped') if mc_phase else
               ('mc_budget_capped' if path.name.endswith('mccapped') else 'mc_budget_static') if mc_ablation else 'mc_budget')
    fixed_cap = 177 if design == 'declared-budget-matched' else 128
    budget_arm = (budget_design or mc_design) and path.name.endswith('declaredbudget')
    if design == 'declared-budget-matched':
        calibration = validate_matched_calibration(config)
        protocol, protocol_sha = read_hashed(path.parent/'protocol.json')
        if validate_matched_calibration(protocol) != calibration:
            raise ValueError(f'{path.name}: config and protocol calibration differ')
    expected = dict(admission_mode='kv' if kv_arm else 'fixed',
                    admission_cap=256 if kv_arm else 192, kv_floor=3277 if kv_arm else 0,
                    probe_enabled=False, reservation_probe_enabled=False, record_gc=True,
                    history_retained=design == 'gc-liveness' and path.name.endswith('retained'))
    if budget_design:
        expected.update(admission_mode='declared_budget' if budget_arm else 'fixed',
            admission_cap=256 if budget_arm else fixed_cap, kv_floor=0,
            native_running_limit=256 if budget_arm else fixed_cap, engine_max_num_seqs=256,
            declared_budget_blocks=32768 if budget_arm else None, declared_budget_block_size=16,
            native_admission_guard_matches_complete_cap=True, reservation_observation=False)
    if mc_design:
        expected.update(admission_mode=mc_mode if mc_arm else 'declared_budget',
            cap=256, admission_cap=256, native_running_limit=256, engine_max_num_seqs=256,
            kv_floor=0, declared_budget_blocks=32768, declared_budget_block_size=16,
            native_admission_guard_matches_complete_cap=True, reservation_observation=False,
            admission_count='complete_unique_unfinished')
    if mc_ablation:
        expected.update(use_future_peak=path.name.endswith('mccapped'), maximum_declared_blocks=39138)
    if mc_phase:
        expected.update(use_future_peak=True, maximum_declared_blocks=39138,
                        allow_scheduled_prefill=path.name.endswith('mcphase'))
    if design == 'declared-budget-matched':
        expected.update(cap=256 if budget_arm else fixed_cap,
                        admission_count='complete_unique_unfinished')
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f'{path.name}: config {key} differs from the declared design')
    # These profiles use the same complete-inflight Gate, whose internal mode
    # label is kv even when a zero water-level floor implements the fixed cap.
    if (gate.get('mode') != (mc_mode if mc_arm else 'declared_budget' if budget_arm else 'kv') or gate.get('cap') != expected['admission_cap'] or
            gate.get('kv_floor') != expected['kv_floor'] or gate.get('probe_enabled') is not False):
        raise ValueError(f'{path.name}: Gate report differs from the declared design')
    if kv_arm and gate.get('max_signal_wait_s') != 10:
        raise ValueError(f'{path.name}: KV age bypass differs from the frozen 10 seconds')
    if (budget_arm or mc_arm) and (gate.get('budget_blocks') != 32768 or gate.get('block_size') != 16 or
            (not mc_arm and gate.get('budget_overrides') != 0) or
            (mc_arm and (type(gate.get('budget_overrides')) is not int or gate['budget_overrides'] < 0)) or
            gate.get('max_signal_wait_s') is not None):
        raise ValueError(f'{path.name}: analyzed Gate budget differs from declared config')
    if (setup.get('history_retained') != expected['history_retained'] or
            setup.get('setup_before_current_external_arrivals') is not True):
        raise ValueError(f'{path.name}: history setup differs from the declared design')
    if design in ('simple-recheck', 'declared-budget', 'declared-budget-matched', 'mc-budget', 'mc-budget-ablation', 'mc-budget-phase') and (setup.get('history_sha256') is not None or
                                      setup.get('history_events') != 0):
        raise ValueError(f'{path.name}: simple recheck unexpectedly loaded historical raw')
    service = arm_summary(cell)
    # Keep the reusable service summary, without duplicating admission-state CDFs.
    admission = service.pop('admission')
    service['controller_overhead'] = admission.get('overhead')
    result = dict(cell=path.name, raw_sha256=cell['raw_sha256'],
        workload_identity_sha256=cell['workload_identity_sha256'],
        declared_mode=config['admission_mode'], gate_reported_mode=gate['mode'],
        admission_cap=expected['admission_cap'], kv_floor=expected['kv_floor'],
        native_running_limit=config['native_running_limit'],
        engine_compiled_max_num_seqs=config['engine_max_num_seqs'],
        native_admission_guard_matches_complete_cap=config.get(
            'native_admission_guard_matches_complete_cap', False),
        probe_enabled=False, reservation_probe_enabled=False, record_gc=True,
        liveness_setup=setup, config_sha256=config_sha, liveness_setup_sha256=setup_sha,
        service_summary=service)
    if budget_design or mc_design:
        result.update(declared_budget_blocks=config['declared_budget_blocks'],
            declared_budget_block_size=config['declared_budget_block_size'],
            admission_wait_rule=config['admission_wait_rule'])
    if mc_ablation or mc_phase:
        result.update(use_future_peak=config['use_future_peak'],
                      maximum_declared_blocks=config['maximum_declared_blocks'],
                      ceiling_selection='PRIOR_DEVELOPMENT_OBSERVED_MAX_CHARGE_NOT_INDEPENDENT_CALIBRATION')
    if mc_phase:
        result['allow_scheduled_prefill'] = config['allow_scheduled_prefill']
    if design == 'declared-budget-matched':
        result.update(declared_budget_matched_calibration=calibration,
                      protocol_sha256=protocol_sha,
                      fixed_cap_selection='DEVELOPMENT_MEAN_DECLARED_CHARGE_NOT_GLOBAL_OPTIMUM')
    return result


def build(analysis, source_path, source_sha, design):
    order = ORDERS[design]
    budget_design = design in ('declared-budget', 'declared-budget-matched')
    mc_ablation = design == 'mc-budget-ablation'
    mc_phase = design == 'mc-budget-phase'
    mc_design = design in ('mc-budget', 'mc-budget-ablation', 'mc-budget-phase')
    fixed_cap = None if mc_design else 177 if design == 'declared-budget-matched' else 128
    cells = {Path(cell['cell']).name: cell for cell in analysis['cells']}
    if len(cells) != len(analysis['cells']) or set(cells) != set(order):
        raise ValueError('Expected exactly the four explicit cell names for '+design)
    if len({cell['workload_identity_sha256'] for cell in cells.values()}) != 1:
        raise ValueError('Full-population workload identity differs across the four cells')
    arms = [metadata(cells[name], design) for name in order]
    arm_by_name = {arm['cell']: arm for arm in arms}
    if design == 'gc-liveness':
        histories = {arm['liveness_setup'].get('history_sha256') for arm in arms}
        if len(histories) != 1 or None in histories:
            raise ValueError('GC-liveness requires the identical historical raw in every arm')
    reference, pairs = None, []
    for candidate_name, baseline_name in ((order[1], order[0]), (order[2], order[3])):
        candidate, baseline = cells[candidate_name], cells[baseline_name]
        raws = {}
        for cell in (candidate, baseline):
            raw, sha = read_hashed(Path(cell['cell'])/'raw.json')
            if sha != cell['raw_sha256']:
                raise ValueError('Raw SHA differs from source analysis: '+cell['cell'])
            if raw.get('gc_observation', {}).get('schema_version') != 1:
                raise ValueError('These designs require common passive GC observation')
            if reference is None:
                reference = dict(requests=[{key: request[key] for key in
                    ('request_id', 'arrival_s', 'prompt_token_ids_sha256', 'prompt_tokens',
                     'max_output_tokens')} for request in raw['requests']])
            difference = workload_difference(raw, reference)
            if difference:
                raise ValueError('Raw workload mismatch: '+difference)
            evidence = gate_evidence(cell, raw, fixed_cap, mc_phase)
            if evidence['probe_enabled'] is not False or evidence['extra_probe_actions']['evaluations']:
                raise ValueError('Unexpected extra probe intervention in a simple quartet')
            if budget_design and 'declared_budget' not in evidence:
                evidence['declared_budget'] = dict(available=False, status=f'NOT_INSTALLED_FIXED{fixed_cap}_ARM')
            if mc_design and 'mc_budget' not in evidence:
                evidence['mc_budget'] = dict(available=False, status='NOT_INSTALLED_ORDINARY_BUDGET_ARM')
            arm_by_name[Path(cell['cell']).name]['admission_evidence'] = evidence
            raws[cell['cell']] = raw
        pair = compare(candidate, baseline, raws)
        pair.update(comparison_role='FULL_SERVICE_RUN_LEVEL_DIAGNOSTIC',
                    factor='historical_raw_strong_reference_liveness' if design == 'gc-liveness'
                           else 'scheduled_complete_prefill_phase_allowance_under_same_future_peak' if mc_phase
                           else 'future_peak_check_under_same_guard_and_declared_ceiling' if mc_ablation
                           else 'guarded_MC_temporal_capacity_vs_ordinary_declared_sum' if mc_design
                           else 'ordinary_declared_token_budget_vs_fixed_concurrency' if budget_design
                           else 'complete_admission_cap_and_KV_water_level')
        if design == 'gc-liveness':
            pair.update(candidate_history='released', baseline_history='retained')
        pairs.append(pair)
    explanation = (
        'All arms use complete cap192 / KV floor0; the sole factor is whether identical historical '
        'raw remains strongly referenced. This is a host-memory liveness diagnostic, not an '
        'admission mechanism.' if design == 'gc-liveness' else
        'All arms release previous raw before measurement. Candidate KV256 uses complete cap256 / '
        'KV floor3277 / 10-second age bypass; baseline fixed192 uses complete cap192 / KV floor0. '
        'These are different simple admission configurations, explicitly compared without retuning. '
        'This same-trace recheck does not establish a globally optimal cap or independent confirmation.')
    if budget_design:
        explanation = ('All arms release previous raw before measurement and use KV floor0, compiled '
            'maximum256 and common passive GC. Candidate declared_budget uses complete cap/native guard256 '
            'and charges ceil((known prompt+declared max output)/16) per unfinished admitted request once, '
            f'with total<=32768 pages and no age override. Baseline fixed{fixed_cap} uses complete cap/native guard{fixed_cap}, '
            + ('the development mean-charge calibration floor(32768/ceil(70702/384))=177. '
               if design == 'declared-budget-matched' else 'the prior mean-flow development choice. ')+
            'No global optimality, recovery-specific novelty, '
            'independent confirmation or cross-trace speedup is established. Budget denials are actual '
            'admission interventions even when changed_by_recovery and extra_probe_actions are zero.')
    gate_explanation = (f' Fixed{fixed_cap} Gate reports mode=kv with floor0; budget Gate reports declared_budget. '
        'Unvisited queue suffixes after a safe native scan break are not fabricated FIFO evaluations.'
        if budget_design else
        ' The common Gate reports mode=kv; floor0 disables water-level blocking and implements the declared fixed cap.')
    if mc_design:
        explanation = ('All arms use complete cap/native guard256, compiled maximum256, KV floor0, '
            '32768 budget pages and common passive GC. The baseline is ordinary sum-of-declared-bounds '
            'admission; the candidate is a production-guarded MC-Benchmark-style temporal-capacity '
            'comparison, not a newly claimed algorithm. Running scheduling, victim and recovery are '
            'unchanged. MC requested relaxations are separated from actual successful allocations and '
            'recorded first-prefill schedule returns. These same-state ordinary-sum comparisons are '
            'not replays of the baseline trajectory or causal service-benefit estimates.')
        gate_explanation = (' Gate modes are declared_budget and mc_budget. No fixed128/fixed177 '
            'shadow is used. Unvisited queue suffixes after native scan breaks are not FIFO evaluations.')
    if mc_ablation:
        explanation = ('Both arms share complete cap/native guard256, compiled maximum256, physical '
            'budget32768, KV floor0, a maximum declared charge39138 and the same production guards. '
            'The baseline uses that static oversubscription ceiling without computing future peaks; '
            'the candidate additionally requires a computed future peak<=32768. The39138 ceiling '
            'is a single prior-development calibration, not an independently established optimal '
            'quota. Running scheduling, victim and recovery are unchanged. Actual requested relaxations, '
            'allocations, charges and first-prefill host returns are separated. Opposite-rule '
            'future-peak suggestions on static states are unavailable, not reconstructed.')
        gate_explanation = (' Gate modes are mc_budget_static and mc_budget_capped. Normal guard '
            'fallback is not a failed model assumption. This is a structural ablation of existing '
            'MC-Benchmark-style accounting, not a novel mechanism or a fixed211 comparison. '
            'Unvisited queue suffixes after scan breaks are not FIFO evaluations.')
    if mc_phase:
        explanation = ('Both arms use complete cap/native guard256, compiled maximum256, physical '
            'budget32768, KV floor0, no age override, the shared39138 declared ceiling and future-peak '
            'checks. Baseline strict phase accepts resident decode; candidate also accepts an old '
            'prompt whose scheduled tokens complete its known tokens in this step. Unfinished '
            'prefill remains rejected. This is a near-neighbor production phase adaptation, not '
            'a new method. Both arms use the same envelope arithmetic implementation. Running '
            'scheduling, victim, recovery, native allocation and quantum are unchanged.')
        gate_explanation = (' Gate modes are mc_budget_capped and mc_budget_phase, with '
            'allow_scheduled_prefill false and true respectively. Phase-added allowances, actual '
            'allocation, charge and host first-prefill returns are distinguished. Scheduled '
            'completion is not observed GPU completion; a same-state strict rejection is not '
            'a full-trajectory counterfactual. Unvisited queue suffixes remain unavailable.')
    return dict(schema_version=1, source_analysis=str(source_path.resolve()),
        source_analysis_sha256=source_sha, design=design, independent_unit='run',
        experiment_design='RETAINED_RELEASED_RELEASED_RETAINED' if design == 'gc-liveness'
                          else 'MCCAPPED_MCPHASE_MCPHASE_MCCAPPED' if mc_phase
                          else 'MCSTATIC_MCCAPPED_MCCAPPED_MCSTATIC' if mc_ablation
                          else 'DECLAREDBUDGET_MCBUDGET_MCBUDGET_DECLAREDBUDGET' if mc_design
                          else f'FIXED{fixed_cap}_DECLAREDBUDGET_DECLAREDBUDGET_FIXED{fixed_cap}' if budget_design
                          else 'FIXED192_KV256_KV256_FIXED192',
        fixed_pairs=[[order[1], order[0]], [order[2], order[3]]], arms=arms, pairs=pairs,
        interpretation=explanation+gate_explanation+' Both extra probes are disabled. '
            'Raw SHA and all request IDs, arrivals, prompt hashes/lengths and output caps are checked. '
            'All 20 predeclared exploratory SLO points remain in the result; no threshold is selected. '
            'All-arrival TTFT/flow and observed service denominators are unchanged. drain_s is '
            'completed_drain_s (last_completion minus arrival_window), while observed_drain_s ends '
            'at observation_end. Natural EOS and output-sequence differences prevent an equal-output '
            'work claim. No GC or other latency is subtracted. Runs, not individual requests, are '
            'the repeat units; one ABBA group is not a statistical significance claim.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('analysis', type=Path)
    parser.add_argument('--design', choices=ORDERS, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Refusing to overwrite existing output; choose a new path.')
    analysis, sha = read_hashed(args.analysis)
    result = build(analysis, args.analysis, sha, args.design)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    for pair in result['pairs']:
        print(pair['candidate']+' / '+pair['baseline'],
              'mean flow delta:', pair['relative_metrics']['flow_mean_s']['delta'],
              'all20 goodput:', pair['joint_goodput_all20'],
              'all20 attainment:', pair['joint_attainment_all20'])
    print(args.output.resolve())


if __name__ == '__main__':
    main()
