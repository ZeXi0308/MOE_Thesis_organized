#!/usr/bin/env python3
"""Supplement frozen F/D/D/F results with exact service-time decompositions.

Run only after full local readback and analyze_demand.py. No replay, fitting,
threshold selection, causal attribution, or replacement of full elapsed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent/'niyama_component'))
import analyze_component as component

CELLS = ['00_fixed1024', '01_prefill_demand', '02_prefill_demand', '03_fixed1024']
METRICS = ('ttft_s', 'generation_span_s', 'finish_remainder_s', 'flow_s')


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def describe(values):
    out = component.native.describe(values)
    known = [v for v in values if v is not None]
    out.update(min=min(known) if known and len(known) == len(values) else None,
               median=statistics.median(known) if known and len(known) == len(values) else None)
    return out


def request_times(request):
    times, arrival, completion = request['token_times_s'], request['arrival_s'], request['completion_s']
    return dict(ttft_s=times[0]-arrival if times else None,
        generation_span_s=times[-1]-times[0] if times else None,
        finish_remainder_s=completion-times[-1] if times and completion is not None else None,
        flow_s=completion-arrival if completion is not None else None)


def load_run(directory, cell, source):
    out = dict(cell=cell, status='UNAVAILABLE', errors=[], source_valid=bool(source and source.get('valid')))
    path = directory/cell/'raw.json'
    if not path.exists(): return out
    try:
        raw = json.loads(path.read_text()); requests = {r['request_id']: r for r in raw['requests']}
        assert len(requests) == len(raw['requests']), 'Duplicate request IDs'
        if not source: raise ValueError('No corresponding frozen analysis row')
        report = {r['request_id']: r for r in source['requests']}
        assert requests.keys() == report.keys(), 'Raw / frozen summary request IDs differ'
        times = {rid: request_times(r) for rid, r in requests.items()}
        residuals = [abs(v['flow_s']-v['ttft_s']-v['generation_span_s']-v['finish_remainder_s'])
                     for v in times.values() if all(v[k] is not None for k in METRICS)]
        out.update(status='DESCRIBED', raw_path=str(path), raw_sha256=sha(path),
            summary=source['summary'], timing=source['timing'], request_count=len(requests),
            finished_count=sum(r['finished'] for r in requests.values()),
            unfinished_ids=sorted(rid for rid, r in requests.items() if not r['finished']),
            declared_output_tokens=sum(r['max_tokens'] for r in requests.values()),
            actual_output_tokens=sum(len(r['output_token_ids']) for r in requests.values()),
            output_count_mismatch_ids=sorted(rid for rid, r in requests.items() if len(r['output_token_ids']) != r['max_tokens']),
            timestamp_count_mismatch_ids=sorted(rid for rid, r in requests.items() if len(r['token_times_s']) != len(r['output_token_ids'])),
            flow_identity_max_abs_residual_s=max(residuals, default=None),
            service_times_s={k: describe([v[k] for v in times.values()]) for k in METRICS},
            _raw=requests, _report=report, _times=times)
    except Exception as exc: out['errors'].append(repr(exc))
    return out


def group_description(ids, fixed, demand):
    ids = sorted(ids)
    out = dict(request_count=len(ids), request_ids=ids, metrics={})
    for metric in METRICS:
        fs, ds = ([r['_times'][rid][metric] for rid in ids] for r in (fixed, demand))
        delta = {rid: d-f if f is not None and d is not None else None for rid, f, d in zip(ids, fs, ds)}
        out['metrics'][metric] = dict(fixed=describe(fs), demand=describe(ds),
            demand_minus_fixed=describe(list(delta.values())),
            negative_ids=[rid for rid, v in delta.items() if v is not None and v < 0],
            positive_ids=[rid for rid, v in delta.items() if v is not None and v > 0],
            unchanged_ids=[rid for rid, v in delta.items() if v == 0],
            unobserved_ids=[rid for rid, v in delta.items() if v is None])
        item = out['metrics'][metric]
        for name in ('negative', 'positive', 'unchanged', 'unobserved'): item[name+'_count'] = len(item[name+'_ids'])
    return out


def completed(row, rid):
    return row['_raw'][rid]['finished'] and row['_times'][rid]['flow_s'] is not None


def qualified(row):
    return {rid for rid, q in row['_report'].items() if completed(row, rid) and q['joint_slo_pass']}


def failure_migration(ids, fixed, demand):
    groups = dict(repaired_joint_pass=[], still_completion_failed=[],
                  completion_pass_but_other_slo_failed=[], unfinished_or_unobserved=[])
    for rid in sorted(ids):
        q = demand['_report'][rid]
        if not completed(demand, rid) or any(demand['_times'][rid][k] is None for k in METRICS):
            groups['unfinished_or_unobserved'].append(rid)
        elif q['joint_slo_pass']: groups['repaired_joint_pass'].append(rid)
        elif 'completion_s' in q['failed_slo_fields']: groups['still_completion_failed'].append(rid)
        else: groups['completion_pass_but_other_slo_failed'].append(rid)
    return dict(partition=groups, counts={k: len(v) for k, v in groups.items()},
        newly_ttft_failed_ids=sorted(rid for rid in ids
            if 'ttft_s' not in fixed['_report'][rid]['failed_slo_fields']
            and 'ttft_s' in demand['_report'][rid]['failed_slo_fields']))


def compare(fixed, demand, historical):
    out = dict(fixed_cell=fixed['cell'], demand_cell=demand['cell'], status='UNAVAILABLE')
    if any('_times' not in r for r in (fixed, demand)): return out
    if fixed['_times'].keys() != demand['_times'].keys(): return {**out, 'status': 'REQUEST_ID_MISMATCH'}
    ids = set(fixed['_times']); fq, dq = qualified(fixed), qualified(demand)
    failed = {rid for rid, q in fixed['_report'].items() if 'completion_s' in q['failed_slo_fields']}
    only = {rid for rid, q in fixed['_report'].items() if q['failed_slo_fields'] == ['completion_s']}
    fs, ds = fixed['summary'], demand['summary']
    nf, nd = fs['joint_slo_qualified_requests'], ds['joint_slo_qualified_requests']
    tf, td = fs['elapsed_s'], ds['elapsed_s']
    quantity, duration, delta = (nd-nf)/tf, nd*(1/td-1/tf), nd/td-nf/tf
    historical_present = historical & ids
    rawf, rawd = fixed['_raw'], demand['_raw']
    changed = sorted(rid for rid in ids if rawf[rid]['output_token_ids'] != rawd[rid]['output_token_ids'])
    deltas = {rid: {k: demand['_times'][rid][k]-fixed['_times'][rid][k]
        if demand['_times'][rid][k] is not None and fixed['_times'][rid][k] is not None else None for k in METRICS} for rid in sorted(ids)}
    residuals = [abs(v['flow_s']-v['ttft_s']-v['generation_span_s']-v['finish_remainder_s'])
                 for v in deltas.values() if all(v[k] is not None for k in METRICS)]
    out['status'] = 'DESCRIBED' if fixed['source_valid'] and demand['source_valid'] else 'INCOMPLETE_SOURCE'
    return dict(**out,
        all_requests=group_description(ids, fixed, demand), per_request_delta_s=deltas,
        delta_flow_identity_max_abs_residual_s=max(residuals, default=None),
        U_decomposition=dict(fixed_N=nf, fixed_full_T_s=tf, demand_N=nd, demand_full_T_s=td,
            fixed_U=nf/tf, demand_U=nd/td, delta_U_req_per_s=delta,
            relative_delta_percent=100*delta/(nf/tf) if nf else None,
            qualification_term_at_fixed_time_req_per_s=quantity,
            time_term_at_demand_count_req_per_s=duration,
            identity_abs_residual=abs(delta-quantity-duration),
            anchor='Change Nf to Nd at Tf first, then Tf to Td holding Nd. Exact ordered algebra, not unique or causal attribution; never use drain as denominator.'),
        qualified_added_ids=sorted(dq-fq), qualified_lost_ids=sorted(fq-dq),
        qualification_source_counts_match=(len(fq) == nf and len(dq) == nd),
        newly_ttft_failed_ids=sorted(rid for rid in ids if 'ttft_s' not in fixed['_report'][rid]['failed_slo_fields']
                                     and 'ttft_s' in demand['_report'][rid]['failed_slo_fields']),
        contemporaneous_fixed_completion_failed=dict(ids=sorted(failed), flow_only_ids=sorted(only),
            migration=failure_migration(failed, fixed, demand), service=group_description(failed, fixed, demand)),
        historical_fixedhalf_73=dict(present_count=len(historical_present), missing_ids=sorted(historical-ids),
            fixed_current_joint_pass_count=len(historical_present & fq), demand_current_joint_pass_count=len(historical_present & dq),
            service=group_description(historical_present, fixed, demand),
            scope='Historical fixed ID cohort only; all values use this pair current F/D clocks. Historical membership is not a contemporaneous failure or matched causal state.'),
        output_lengths_match=all(len(rawf[rid]['output_token_ids']) == len(rawd[rid]['output_token_ids']) for rid in ids),
        output_token_ids_different_count=len(changed), output_token_ids_different_ids=changed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory', type=Path); parser.add_argument('--output', type=Path)
    args = parser.parse_args(); source_path = args.run_directory/'demand_results.json'
    if not args.run_directory.is_dir(): parser.error('Run directory absent; no output created')
    source = json.loads(source_path.read_text()) if source_path.exists() else {}
    bycell = {r['cell']: r for r in source.get('runs', [])}
    historical_path = ROOT.parent/'fixedhalf53005_01/fixed_results.json'
    old = json.loads(historical_path.read_text()); oldrows = {r['cell']: r for r in old['runs']}
    oldsets = [{q['request_id'] for q in oldrows[c]['requests'] if q['failed_slo_fields'] == ['completion_s']}
               for c in ('01_fixed1024', '02_fixed1024')]
    historical_valid = len(oldsets[0]) == len(oldsets[1]) == 73 and oldsets[0] == oldsets[1]
    runs = [load_run(args.run_directory, c, bycell.get(c)) for c in CELLS]
    pairs = [compare(runs[0], runs[1], oldsets[0] if historical_valid else set()),
             compare(runs[3], runs[2], oldsets[0] if historical_valid else set())]
    observed = sum((args.run_directory/c/'raw.json').exists() for c in CELLS)
    complete = (source.get('status') == 'COMPLETE_EXPLORATORY' and historical_valid
        and all(r['status'] == 'DESCRIBED' and r['source_valid'] and not r['errors']
                and not r['unfinished_ids'] and not r['output_count_mismatch_ids']
                and not r['timestamp_count_mismatch_ids'] for r in runs)
        and all(p['status'] == 'DESCRIBED' and p['qualification_source_counts_match'] for p in pairs))
    result = dict(status='COMPLETE_SUPPLEMENTAL_ANALYSIS' if complete else 'UNRUN' if not observed else 'INCOMPLETE_OR_UNAVAILABLE',
        source_results_path=str(source_path), source_results_sha256=sha(source_path) if source_path.exists() else None,
        source_status=source.get('status'), observed_formal_raw_files=observed, script_sha256=sha(Path(__file__)),
        historical_source_path=str(historical_path), historical_source_sha256=sha(historical_path),
        historical_sets_identical_73=historical_valid, pairs=pairs,
        runs=[{k: v for k, v in r.items() if not k.startswith('_')} for r in runs],
        scope='Supplement only: all requests retained, missing/unfinished values never zero-filled or passed. '
              'Flow=TTFT+first-to-last generation span+last-output-to-completion remainder. '
              'Differences are descriptive diverged trajectories, not same-state causal effects. '
              'Fixed output counts do not establish equal content/routes/actual computation/quality. '
              'No new threshold, SLO, experiment or automatic continuation decision.')
    output = args.output or args.run_directory/'service_decomposition.json'
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(f'{result["status"]}; {observed} formal raw files; {output}')


if __name__ == '__main__': main()
