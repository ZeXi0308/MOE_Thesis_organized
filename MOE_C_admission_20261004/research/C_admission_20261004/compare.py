#!/usr/bin/env python3
"""Compare both three-arm test blocks without pooling requests or selecting an SLO.

Usage: python3 compare.py ANALYSIS.json --output NEW_COMPARISON.json
Input may also contain dev cells; test-00 through test-05 are selected by name.
If dev-cap128/192/256 are present, comparable workloads are additionally reported
as development references, never as a replacement for the reversed-order tests.
Absolute cell paths in the analysis must still locate the immutable raw.json files.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re


ORDER = ('fixed', 'kv', 'recovery', 'recovery', 'kv', 'fixed')
PAIRS = (('kv', 'fixed'), ('recovery', 'fixed'), ('recovery', 'kv'))
METRICS = ('ttft_mean_s', 'ttft_p95_s', 'flow_mean_s', 'flow_p95_s',
           'gap_p95_s', 'output_tokens_per_s', 'drain_s')


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def metrics(cell):
    dist = cell['distributions']
    return dict(ttft_mean_s=dist['ttft_s']['observed_only']['mean'],
        ttft_p95_s=dist['ttft_s']['observed_only']['p95'],
        flow_mean_s=dist['flow_s']['observed_only']['mean'],
        flow_p95_s=dist['flow_s']['observed_only']['p95'],
        gap_p95_s=dist['max_generation_gap_s']['observed_only']['p95'],
        output_tokens_per_s=cell['output_tokens_per_s'],
        drain_s=cell['completed_drain_s'], observed_drain_s=cell['observed_drain_s'])


def admission_cap(cell):
    admission = cell.get('admission', {})
    return admission.get('configuration', {}).get('cap', admission.get('cap'))


def arm_summary(cell):
    return dict(run=Path(cell['cell']).name, planned_requests=cell['planned_requests'],
        admission_cap=admission_cap(cell), outcomes=cell['outcomes'], metrics=metrics(cell),
        drain_censored=cell['drain_censored'], total_output_tokens=cell['total_output_tokens'],
        admission=cell['admission'])


def request_map(raw):
    result = {r['request_id']: r for r in raw['requests']}
    if len(result) != len(raw['requests']):
        raise ValueError('Duplicate request identity in raw file')
    return result


def compare_outputs(candidate, baseline):
    a, b = request_map(candidate), request_map(baseline)
    if a.keys() != b.keys():
        raise ValueError('Cannot compare arms with different request IDs')
    rows = []
    for rid in a:
        ar, br = a[rid], b[rid]
        for field in ('arrival_s', 'prompt_token_ids_sha256', 'max_output_tokens'):
            if ar.get(field) != br.get(field):
                raise ValueError(f'Workload mismatch for {rid}: {field}')
        at, bt = ar.get('output_token_ids', []), br.get('output_token_ids', [])
        mismatch = next((i for i, (x, y) in enumerate(zip(at, bt)) if x != y), None)
        if mismatch is None and len(at) != len(bt):
            mismatch = min(len(at), len(bt))
        rows.append(dict(request_id=rid, candidate_tokens=len(at), baseline_tokens=len(bt),
            token_count_delta=len(at)-len(bt), sequence_mismatch=at != bt,
            first_mismatch_index=mismatch, candidate_status=ar.get('status'), baseline_status=br.get('status'),
            candidate_stop=ar.get('stop_reason', ar.get('finish_reason')),
            baseline_stop=br.get('stop_reason', br.get('finish_reason'))))
    total_a, total_b = sum(r['candidate_tokens'] for r in rows), sum(r['baseline_tokens'] for r in rows)
    return dict(requests=len(rows), sequence_mismatch_requests=sum(r['sequence_mismatch'] for r in rows),
        token_count_mismatch_requests=sum(r['token_count_delta'] != 0 for r in rows),
        stop_reason_mismatch_requests=sum(r['candidate_stop'] != r['baseline_stop'] for r in rows),
        candidate_total_tokens=total_a, baseline_total_tokens=total_b,
        token_total_delta=total_a-total_b, token_total_delta_pct=100*(total_a/total_b-1) if total_b else None,
        per_request=rows)


def slo_map(cell):
    return {(r['ttft_limit_s'], r['maxgap_limit_s'], r['flow_limit_s']): r for r in cell['joint_slo_grid']}


def sign_counts(rows, field):
    values = [r[field] for r in rows]
    return dict(wins=sum(x > 1e-12 for x in values), losses=sum(x < -1e-12 for x in values),
                ties=sum(abs(x) <= 1e-12 for x in values), grid_points=len(values))


def compare(candidate, baseline, raws):
    am, bm = metrics(candidate), metrics(baseline)
    relative = {}
    for key in METRICS:
        a, b = am[key], bm[key]
        delta = a-b if number(a) and number(b) else None
        relative[key] = dict(candidate=a, baseline=b, delta=delta,
            delta_pct=100*delta/b if delta is not None and b != 0 else None,
            preferred_direction='higher' if key == 'output_tokens_per_s' else 'lower')
    agrid, bgrid = slo_map(candidate), slo_map(baseline)
    expected = {(t, g, 120) for t in (2, 5, 10, 20, 40) for g in (.25, .5, 1, 2)}
    if set(agrid) != expected or set(bgrid) != expected:
        raise ValueError('Comparison requires the entire predeclared 20-point SLO grid')
    grid = []
    for threshold in sorted(expected):
        a, b = agrid[threshold], bgrid[threshold]
        grid.append(dict(ttft_limit_s=threshold[0], maxgap_limit_s=threshold[1], flow_limit_s=threshold[2],
            candidate_good_requests=a['good_requests'], baseline_good_requests=b['good_requests'],
            candidate_goodput=a['goodput_requests_per_s'], baseline_goodput=b['goodput_requests_per_s'],
            goodput_delta=a['goodput_requests_per_s']-b['goodput_requests_per_s'],
            joint_attainment_delta=a['joint_attainment']-b['joint_attainment']))
    return dict(candidate=Path(candidate['cell']).name, baseline=Path(baseline['cell']).name,
        candidate_admission_cap=admission_cap(candidate), baseline_admission_cap=admission_cap(baseline),
        relative_metrics=relative, joint_goodput_all20=sign_counts(grid, 'goodput_delta'),
        joint_attainment_all20=sign_counts(grid, 'joint_attainment_delta'), joint_slo_grid=grid,
        outputs=compare_outputs(raws[candidate['cell']], raws[baseline['cell']]))


def workload_difference(a_raw, b_raw):
    a, b = request_map(a_raw), request_map(b_raw)
    if a.keys() != b.keys():
        return f'request population mismatch ({len(a)} vs {len(b)} requests)'
    for rid in a:
        for field in ('arrival_s', 'prompt_token_ids_sha256', 'prompt_tokens', 'max_output_tokens'):
            if a[rid].get(field) != b[rid].get(field):
                return f'{field} differs for request {rid}'
    return None


def development_references(dev_cells, test_cells, raws):
    result = dict(role='DEVELOPMENT_REFERENCES_NOT_REVERSED_ORDER_PAIRS',
        interpretation='These are development references, not independent confirmation. '
            'They do not replace the six test runs or the within-block recovery/KV comparison.',
        arms=[arm_summary(dev_cells[cap]) for cap in sorted(dev_cells)], comparisons=[], skipped=[])
    if set(dev_cells) != {128, 192, 256}:
        result.update(status='SKIPPED_INCOMPLETE_REFERENCE_SET',
                      missing_caps=sorted({128, 192, 256}-set(dev_cells)))
        return result
    for index in (1, 2, 3, 4):
        candidate = test_cells[index]
        for cap in (128, 192, 256):
            baseline = dev_cells[cap]
            reason = workload_difference(raws[candidate['cell']], raws[baseline['cell']])
            if reason:
                result['skipped'].append(dict(candidate=Path(candidate['cell']).name,
                    baseline=Path(baseline['cell']).name, reason=reason))
                continue
            pair = compare(candidate, baseline, raws)
            pair.update(test_block=index//3+1, reference_role=result['role'])
            result['comparisons'].append(pair)
    result['status'] = ('COMPLETE' if not result['skipped'] else
        'PARTIALLY_SKIPPED_WORKLOAD_MISMATCH' if result['comparisons'] else 'SKIPPED_WORKLOAD_MISMATCH')
    return result


def fmt(value, pct=False):
    return f'{value:+.1f}%' if pct and number(value) else f'{value:.3f}' if number(value) else 'NA'


def report(result):
    print('All latency summaries condition on observed values; outcomes retain every planned request.')
    print('| Run | Admission cap | Complete/N | TTFT mean/p95 s | Flow mean/p95 s | Gap p95 s | Tokens | Token/s | Drain s |')
    print('|---|---:|---:|---:|---:|---:|---:|---:|---:|')
    development = result.get('development_references')
    groups = result['blocks'] + ([dict(arms=development['arms'])] if development else [])
    for block in groups:
        for arm in block['arms']:
            m = arm['metrics']
            print(f"| {arm['run']} | {arm.get('admission_cap', 'NA')} | {arm['outcomes']['completed']}/{arm['planned_requests']} | "
                f"{fmt(m['ttft_mean_s'])}/{fmt(m['ttft_p95_s'])} | {fmt(m['flow_mean_s'])}/{fmt(m['flow_p95_s'])} | "
                f"{fmt(m['gap_p95_s'])} | {arm['total_output_tokens']} | {fmt(m['output_tokens_per_s'])} | {fmt(m['drain_s'])} |")
    print('\nDelta % is (candidate/baseline - 1); lower latency and higher token/s are preferred.')
    print('| Block pair | Admission cap A/B | TTFT mean/p95 | Flow mean/p95 | Gap p95 | Token/s | Drain | SLO goodput W/L/T | Output mismatch/N |')
    print('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    groups = result['blocks'] + ([dict(block='DEV REF', comparisons=development['comparisons'])] if development else [])
    for block in groups:
        for pair in block['comparisons']:
            m, slo, out = pair['relative_metrics'], pair['joint_goodput_all20'], pair['outputs']
            v = lambda k: fmt(m[k]['delta_pct'], True)
            label = (f"{pair['candidate']}/{pair['baseline']}" if block['block'] == 'DEV REF' else
                     f"{pair['candidate'].split('-')[-1]}/{pair['baseline'].split('-')[-1]}")
            print(f"| {block['block']} {label} | {pair.get('candidate_admission_cap', 'NA')}/{pair.get('baseline_admission_cap', 'NA')} | "
                f"{v('ttft_mean_s')}/{v('ttft_p95_s')} | {v('flow_mean_s')}/{v('flow_p95_s')} | "
                f"{v('gap_p95_s')} | {v('output_tokens_per_s')} | {v('drain_s')} | "
                f"{slo['wins']}/{slo['losses']}/{slo['ties']} | {out['sequence_mismatch_requests']}/{out['requests']} |")
    print('\nTwo run-level blocks only: descriptive repeat consistency, no statistical significance claim.')
    if development:
        print('Development references: '+development['status']+'. '+development['interpretation'])
        if development.get('missing_caps'):
            print('Missing development caps: '+str(development['missing_caps']))
        for skipped in development['skipped']:
            print(f"SKIP {skipped['candidate']}/{skipped['baseline']}: {skipped['reason']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('analysis', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Refusing to overwrite an existing output; choose a new --output path.')
    summary = json.loads(args.analysis.read_text())
    cells, dev_cells = {}, {}
    for cell in summary['cells']:
        dev_match = re.fullmatch(r'dev-cap(128|192|256)', Path(cell['cell']).name)
        if dev_match:
            cap = int(dev_match[1])
            if cap in dev_cells:
                raise ValueError(f'Duplicate development cap {cap}')
            actual_cap = admission_cap(cell)
            if actual_cap is not None and actual_cap != cap:
                raise ValueError(f'Development label/cap mismatch: {cell["cell"]}')
            mode = cell.get('admission', {}).get('configuration', {}).get('mode')
            if mode is not None and mode != 'fixed':
                raise ValueError(f'Development reference is not fixed admission: {cell["cell"]}')
            dev_cells[cap] = cell
        match = re.fullmatch(r'test-(\d+)-(fixed|kv|recovery)', Path(cell['cell']).name)
        if match:
            index = int(match[1])
            if index in cells:
                raise ValueError(f'Duplicate test index {index}')
            if index >= len(ORDER) or match[2] != ORDER[index]:
                raise ValueError('Test labels do not match fixed/kv/recovery and reverse execution order')
            cells[index] = cell
    if set(cells) != set(range(6)):
        raise ValueError('Expected exactly test-00 through test-05; partial blocks are not pooled')
    raws = {}
    for cell in [*cells.values(), *dev_cells.values()]:
        raw_bytes = (Path(cell['cell'])/'raw.json').read_bytes()
        if hashlib.sha256(raw_bytes).hexdigest() != cell['raw_sha256']:
            raise ValueError('Raw file changed after analysis: '+cell['cell'])
        # Comparisons need request records only; large output-event traces stay on disk.
        raws[cell['cell']] = dict(requests=json.loads(raw_bytes)['requests'])
    result = dict(schema_version=1, source_analysis=str(args.analysis.resolve()),
        independent_unit='run', statistical_inference='NONE; two descriptive reversed-order blocks',
        slo_semantics='All 20 predeclared exploratory thresholds are retained. No selected best threshold.',
        output_semantics='Token counts and exact sequence differences describe observed work, not task quality.',
        blocks=[])
    for block in range(2):
        selected = [cells[i] for i in range(block*3, block*3+3)]
        by_mode = {Path(c['cell']).name.split('-')[-1]: c for c in selected}
        arms = [arm_summary(c) for c in selected]
        pairs = [compare(by_mode[a], by_mode[b], raws) for a, b in PAIRS]
        result['blocks'].append(dict(block=block+1, arms=arms, comparisons=pairs))
    result['same_arm_repeat_outputs'] = {}
    for mode in ORDER[:3]:
        matched = [cells[i] for i in range(6) if ORDER[i] == mode]
        result['same_arm_repeat_outputs'][mode] = compare_outputs(
            raws[matched[1]['cell']], raws[matched[0]['cell']])
    if dev_cells:
        result['development_references'] = development_references(dev_cells, cells, raws)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    report(result)
    print(args.output.resolve())


if __name__ == '__main__':
    main()
