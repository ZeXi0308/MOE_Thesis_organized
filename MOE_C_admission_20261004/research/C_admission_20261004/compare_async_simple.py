#!/usr/bin/env python3
"""Compare one native-async fixed177 / declared-budget ABBA, without overwriting.

python3 compare_async_simple.py ANALYSIS.json --output NEW.comparison.json
Optional --cell-root relocates raw cells without changing the saved analysis.
"""
import argparse
import hashlib
import json
from pathlib import Path

from compare import arm_summary, compare, workload_difference


ORDER = ('probe-00-fixed177', 'probe-01-declaredbudget',
         'probe-02-declaredbudget', 'probe-03-fixed177')


def read_hashed(path):
    data = path.read_bytes()
    return json.loads(data), hashlib.sha256(data).hexdigest()


def admission_evidence(report, raw):
    identity = {r[key]: r['request_id'] for r in raw['requests']
                for key in ('request_id', 'internal_request_id', 'external_request_id') if r.get(key)}

    def counts(rows):
        ids = {identity[r['request_id']] for r in rows}
        return dict(evaluations=len(rows), unique_requests=len(ids), request_ids=sorted(ids))

    rows = report['decisions']
    direct = [r for r in rows if not r['reason'].startswith('fifo_')]
    denied = [r for r in direct if r['denied']]
    fifo = [r for r in rows if r['reason'].startswith('fifo_') and r['denied']]
    if any(r['native_fit'] is not True or r['native_allocation_result'] is not None
           or r['reason'] not in ('cap', 'declared_budget') for r in denied):
        raise ValueError('Head gate denials must have native fit but no executed allocation')
    if any(r['native_fit'] is not None or r['baseline_allowed'] is not None or
           r['native_allocation_result'] is not None for r in fifo):
        raise ValueError('FIFO rows cannot claim native permission or an allocation result')
    if any(r['native_allocation_result'] != r['native_allocation_succeeded'] for r in direct):
        raise ValueError('Allocation result aliases disagree')
    budget_denials = [r for r in denied if r['reason'] == 'declared_budget']
    if any(r['baseline_allowed'] is not True or
           r['budget_after_if_admitted_blocks'] <= r['budget_limit_blocks'] for r in budget_denials):
        raise ValueError('Declared denial must be beyond the native-fit/complete-cap baseline')
    successful = [r for r in direct if r['native_allocation_result'] is True]
    starts = report['starts']
    by_id = {r['request_id']: r['first_prefill_perf_s'] for r in starts}
    matched = [r for r in successful if r['request_id'] in by_id and
               by_id[r['request_id']] >= report['origin_perf_s'] + r['t']]
    breaks = [r for r in direct if r.get('native_waiting_scan_break') is True]
    if report['native_waiting_scan_breaks'] != len(breaks) or any(
            not r['denied'] or r.get('suffix_new_requests_not_evaluated') is not True for r in breaks):
        raise ValueError('Safe scan-break counter disagrees with the recorded head decisions')
    allocations = report['native_allocations']
    mismatches = sum(r['actual'] != r['predicted'] for r in allocations)
    unknown = sum(type(r['actual']) is not bool or type(r['predicted']) is not bool for r in allocations)
    states = report['snapshots'] + rows
    peaks = {key: max((r[key] for r in states if key in r), default=None) for key in
             ('active', 'running', 'physical_used_blocks', 'nonlive_physical_blocks', 'budget_before_blocks')}
    matches = [dict(request_id=identity[r['request_id']],
        allocation_decision_external_s=report['origin_perf_s']+r['t']-raw['measurement_origin_perf_counter_s'],
        first_prefill_schedule_return_external_s=by_id[r['request_id']]-raw['measurement_origin_perf_counter_s'])
        for r in matched]
    return dict(direct_head_gate_denials=counts(denied), fifo_holds=counts(fifo),
        denials_by_reason={reason: counts([r for r in rows if r['reason'] == reason and r['denied']])
                          for reason in ('cap', 'declared_budget', 'fifo_cap', 'fifo_declared_budget')},
        successful_new_allocations=counts(successful),
        native_capacity_refusals=counts([r for r in direct if r['native_allocation_result'] is False]),
        native_result_unavailable_without_gate_denial=counts([r for r in direct if not r['denied']
            and type(r['native_allocation_result']) is not bool]),
        first_prefill_starts=counts(starts), first_prefill_schedule_return_matches=counts(matched),
        allocations_without_matched_schedule_return=len(successful)-len(matched),
        matched_first_prefill_times=matches,
        native_waiting_scan_breaks=counts(breaks), skipped_suffix_requests=None,
        fit_validation=dict(scope='EXECUTED_NEW_ALLOCATIONS_ONLY_NOT_RECOVERY',
            reported_checks=report['native_fit_checks'], recorded_allocations=len(allocations),
            reported_mismatches=report['native_fit_mismatches'], observed_mismatches=mismatches,
            unknown_boolean_results=unknown, counters_agree=report['native_fit_checks']==len(allocations)
                and report['native_fit_mismatches']==mismatches),
        observed_peaks=peaks, observed_budget_peak_blocks=report['observed_budget_peak_blocks'],
        semantics='Actual ordinary gate delays, not MC suggestions or policy counterfactuals. '
            'baseline_allowed means native fit plus this arm\'s complete cap. FIFO native permission '
            'is unknown; a safe scan break leaves its suffix unevaluated. Repeated evaluations are '
            'not unique requests. Allocation success and host schedule-return are distinct; neither '
            'proves GPU completion. Peaks are observed states, not continuous extrema. Budget state '
            'is physical_used + live unallocated declared remainder; finished/deferred physical '
            'pages remain charged, without double-counting live resident pages or future release credit.')


def build(analysis, source, source_sha, cell_root=None):
    cells = {Path(c['cell']).name: c for c in analysis['cells']}
    if len(cells) != len(analysis['cells']) or set(cells) != set(ORDER):
        raise ValueError('Expected exactly fixed177 / declaredbudget / declaredbudget / fixed177')
    if len({c['workload_identity_sha256'] for c in cells.values()}) != 1:
        raise ValueError('External workload identity differs across arms')
    raws, arms, reference, resource_reference = {}, [], None, None
    for name in ORDER:
        cell = cells[name]
        path = cell_root/name if cell_root is not None else Path(cell['cell'])
        config, config_sha = read_hashed(path/'config.json')
        report, report_sha = read_hashed(path/'admission.json')
        raw, raw_sha = read_hashed(path/'raw.json')
        if raw_sha != cell['raw_sha256']:
            raise ValueError('Raw changed after full-population analysis: '+name)
        fixed = name.endswith('fixed177')
        mode, cap = ('fixed', 177) if fixed else ('declared_budget', 256)
        expected_config = dict(admission_mode=mode, admission_cap=cap, cap=cap,
            async_scheduling=True, policy_intervention=True, native_running_limit=256,
            engine_max_num_seqs=256, target_usable_kv_blocks=32768, kv_floor=0)
        expected_report = dict(mode=mode, cap=cap, async_scheduling=True, policy_intervention=True,
            native_running_cap=256, budget_blocks=32768, block_size=16, kv_floor=0,
            max_signal_wait_s=None, probe_enabled=False, admission_count='complete_unique_unfinished')
        for actual, expected in ((config, expected_config), (report, expected_report)):
            if any(actual.get(k) != v or (type(v) is bool and actual.get(k) is not v)
                   for k, v in expected.items()):
                raise ValueError('Configuration differs from the explicit async simple design: '+name)
        if config.get('budget_blocks') != (None if fixed else 32768):
            raise ValueError('Config declaration limit differs from the explicit arm: '+name)
        backend = raw.get('async_observation')
        if not isinstance(backend, dict) or backend.get('async_scheduling') is not True:
            raise ValueError('Missing native async/backend observation: '+name)
        resources = {k:config.get(k) for k in
                     ('fixed_kv_cache_memory_bytes', 'intended_usable_kv_bytes')}
        if resource_reference is not None and resources != resource_reference:
            raise ValueError('Recorded resource budget differs across arms')
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
            latency_distributions={k:v['observed_only'] for k,v in cell['distributions'].items()},
            actual_preemption_count=cell['actual_preemption_count'],
            stop_reasons=cell['stop_reasons'], natural_stop_count=cell['natural_stop_count'],
            length_stop_count=cell['length_stop_count'], host_chunk_diagnostics=cell['host_chunk_diagnostics'],
            observation_end_s=cell['observation_end_s'], arrival_window_s=cell['arrival_window_s'])
        arms.append(dict(cell=name, source_cell=cell['cell'], read_cell=str(path.resolve()),
            raw_sha256=raw_sha, config_sha256=config_sha, admission_sha256=report_sha,
            declared_mode=mode, gate_reported_mode=report['mode'], complete_admission_cap=cap,
            native_running_limit=256, engine_compiled_max_num_seqs=256,
            physical_budget_blocks=32768, resource_config=resources,
            budget_enforced=not fixed, service_summary=service,
            async_observation=backend, admission_evidence=admission_evidence(report,raw)))
    pairs = [compare(cells[ORDER[a]],cells[ORDER[b]],raws) for a,b in ((1,0),(2,3))]
    return dict(schema_version=1, source_analysis=str(source.resolve()), source_analysis_sha256=source_sha,
        design='ASYNC_FIXED177_DECLAREDBUDGET_DECLAREDBUDGET_FIXED177', independent_unit='run',
        fixed_pairs=[[ORDER[1],ORDER[0]],[ORDER[2],ORDER[3]]], arms=arms, pairs=pairs,
        interpretation='Exploratory ordinary async baseline engineering comparison, not a new method, '
            'MC comparison or independent confirmation. Native/compiled maximum256 is common; complete '
            'cap177 versus physical-plus-declared-remainder budget32768/cap256 is the explicit factor. '
            'No cap retuning, future release credit, output prediction or age exemption is implied. '
            'All external arrivals, failures and unfinished requests retain the original full-population '
            'denominators and all20 exploratory SLO points. Backend tail and controller/observer costs '
            'remain in observation time; completed_drain and backend drain are distinct. Missing or '
            'undrained backend states are retained, not treated as success. Host chunks cannot reveal '
            'intra-chunk GPU token timing. Natural EOS/output differences preclude an equal-work or '
            'quality claim. Two run-level pairs do not establish statistical significance; normal '
            'placeholders and terminal-inflight state are not execution failure.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('analysis',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cell-root',type=Path)
    args=parser.parse_args()
    if args.output.exists():
        parser.error('Refusing to overwrite an existing output')
    analysis,sha=read_hashed(args.analysis)
    result=build(analysis,args.analysis,sha,args.cell_root)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
        stream.write('\n')
    for pair in result['pairs']:
        print(pair['candidate'],'/',pair['baseline'],'mean flow delta',
              pair['relative_metrics']['flow_mean_s']['delta'],'all20',pair['joint_goodput_all20'])
    print(args.output.resolve())


if __name__=='__main__':
    main()
