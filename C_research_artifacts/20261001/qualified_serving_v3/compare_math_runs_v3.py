#!/usr/bin/env python3
"""CPU-only paired comparison of existing math analyses and native artifacts.

No generation or rescoring. Left is the reference; all deltas are right-left.
The output is exclusively created, including when comparison checks fail.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

LEGACY_NATIVE_CELL_SHA = 'cce7d5fcd4b5d88dce46d4dd483bde7c1c11514e15747ec1ac9965445df827bb'


def policy_evidence(analysis, summary):
    """Only the reviewed original scale cell may omit its native policy field."""
    declared = summary['config'].get('policy')
    run = Path(summary['run'])
    provenance = read(run/'provenance.json')
    cell_sha = digest(run/'cell-source.py')
    issues = []
    provenance_valid = digest(run/'provenance.json') == analysis['input_sha256'].get('provenance.json')
    cell_valid = provenance.get('cell_sha256') == cell_sha
    if not provenance_valid or not cell_valid:
        issues.append('policy_cell_or_provenance_hash_mismatch')
    normalized = declared
    legacy = declared is None and 'policy' not in summary['config']
    if legacy:
        if (analysis['schema'] == 'c-math-scale-analysis-v1'
                and analysis['counts']['planned'] == 1024
                and cell_sha == LEGACY_NATIVE_CELL_SHA and provenance_valid and cell_valid):
            normalized = 'native'
        else:
            issues.append('unrecognized_missing_policy_declaration')
    if normalized not in ('native', 'restore-fifo', 'restore-runway'):
        issues.append('unsupported_policy_declaration')
    return dict(declared=declared, normalized=normalized, defaulted_legacy_native=legacy and normalized == 'native',
                cell_sha256=cell_sha, provenance_hash_verified=provenance_valid,
                cell_provenance_hash_verified=cell_valid,
                normalization_rule='Only c-math-scale-analysis-v1/1024 requests with the reviewed '
                'legacy cell SHA may omit policy. The cell directly calls common.generate under '
                'observe_native_pressure, with no policy hook.'), issues


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    with Path(path).open(encoding='utf-8') as stream:
        return json.load(stream)


def numeric(x):
    return type(x) in (int, float) and math.isfinite(x)


def change(left, right):
    valid = numeric(left) and numeric(right)
    return dict(left=left, right=right, right_minus_left=right-left if valid else None,
                percent_change=100 * (right/left-1) if valid and left else None)


def differences(left, right):
    return {k: dict(left=left.get(k), right=right.get(k))
            for k in sorted(set(left) | set(right)) if left.get(k) != right.get(k)}


def analysis_summary(path):
    raw = read(path)
    # Do not keep the large scheduled_work list in memory for a paired summary.
    keep = ('schema', 'integrity', 'run', 'input_sha256', 'analyzer_sha256', 'issues',
            'counts', 'timing', 'episode', 'diagnostic_thresholds', 'per_request', 'pressure_counts')
    result = {k: raw.get(k) for k in keep}
    result['work_totals'] = raw['attribution']['totals']
    result['gate_time_coverage'] = raw['attribution']['gate_time_coverage']
    return result


def pressure_peak(path, runtime, expected_hash):
    # Upstream VALID analysis already parsed and validated this pretty-printed
    # observer receipt. Hash-verified line scanning avoids a second 170MB parse.
    pattern = re.compile(rb'^\s*"(free_blocks(?:_before|_after)?|running)":\s*(\d+)\s*,?\s*$')
    h, minimum, peak_running, samples = hashlib.sha256(), None, 0, 0
    with Path(path).open('rb') as stream:
        for line in stream:
            h.update(line)
            match = pattern.match(line)
            if match:
                name, raw = match.groups()
                value = int(raw)
                if name == b'running':
                    peak_running = max(peak_running, value)
                else:
                    minimum = value if minimum is None else min(minimum, value)
                    samples += 1
    usable = runtime['usable_blocks']
    peak = usable-minimum if minimum is not None else None
    return dict(sha256=h.hexdigest(), upstream_hash_matches=h.hexdigest() == expected_hash,
                usable_blocks=usable, minimum_observed_free_blocks=minimum,
                peak_observed_used_blocks=peak,
                peak_usable_fraction=peak/usable if peak is not None and usable else None,
                peak_running=peak_running, free_block_samples=samples,
                scope='Schedule/allocation boundaries only; not unseen internal peaks.')


def output_signatures(path, expected_hash):
    rows = read(path)
    signatures = {}
    errors = []
    for row in rows:
        rid = row['request_id']
        if rid in signatures:
            errors.append('duplicate_raw_output_id:' + rid)
        signatures[rid] = dict(
            text_sha256=hashlib.sha256(row['output_text'].encode()).hexdigest(),
            token_ids_sha256=hashlib.sha256(json.dumps(row['output_token_ids'],
                separators=(',', ':')).encode()).hexdigest(),
            output_tokens=len(row['output_token_ids']))
    if digest(path) != expected_hash:
        errors.append('raw_output_upstream_hash_mismatch')
    return signatures, errors


def load_arm(analysis_path, run, label):
    a = analysis_summary(analysis_path)
    run = Path(run) if run else Path(a['run'])
    config, runtime = read(run/'config.json'), read(run/'resolved-runtime.json')
    peak = pressure_peak(run/'measured-pressure.json', runtime,
                         a['input_sha256'].get('measured-pressure.json'))
    outputs, issues = output_signatures(run/'measured-outputs.json',
                                        a['input_sha256'].get('measured-outputs.json'))
    if a['integrity'] != 'VALID':
        issues.append('upstream_analysis_not_valid')
    if not peak['upstream_hash_matches']:
        issues.append('pressure_upstream_hash_mismatch')
    for name in ('config.json', 'resolved-runtime.json', 'source-input.json', 'rendered-inputs.json'):
        if digest(run/name) != a['input_sha256'].get(name):
            issues.append('artifact_upstream_hash_mismatch:' + name)
    indexed = {r['request_id']: r for r in a['per_request']}
    if len(indexed) != len(a['per_request']) or len(indexed) != a['counts']['planned']:
        issues.append('analysis_request_inventory_invalid')
    if set(outputs) != set(indexed):
        issues.append('raw_analysis_request_inventory_mismatch')
    for rid in set(outputs) & set(indexed):
        if outputs[rid]['output_tokens'] != indexed[rid]['output_tokens']:
            issues.append('raw_analysis_token_count_mismatch:' + rid)
    wall = a['work_totals']['schedule_host_wall_s']
    duration = a['episode']['duration_s']
    summary = dict(label=label, analysis_path=str(Path(analysis_path).resolve()),
        analysis_sha256=digest(analysis_path), run=str(run.resolve()),
        analyzer_sha256=a['analyzer_sha256'], integrity=a['integrity'],
        counts=a['counts'], timing=a['timing'], episode=a['episode'],
        config=config, resolved_runtime=runtime, pressure_counts=a['pressure_counts'],
        pressure_peak=peak, work_totals=a['work_totals'], gate_time_coverage=a['gate_time_coverage'],
        schedule_wrapper_wall=dict(seconds=wall, fraction_of_episode=wall/duration if duration else None,
            definition='Instrumented scheduler wrapper wall time; includes native scheduler and observer, not isolated observer overhead or process CPU.'))
    return a, indexed, outputs, summary, issues


def compare(left_analysis, right_analysis, left_run=None, right_run=None,
            left_label='left', right_label='right', allowed=('max_num_seqs',), allow_policy_change=False):
    la, left, lo, ls, li = load_arm(left_analysis, left_run, left_label)
    ra, right, ro, rs, ri = load_arm(right_analysis, right_run, right_label)
    issues = ['left:' + e for e in li] + ['right:' + e for e in ri]
    lp, lpi = policy_evidence(la, ls)
    rp, rpi = policy_evidence(ra, rs)
    issues.extend(['left:' + e for e in lpi] + ['right:' + e for e in rpi])
    policy_changed = lp['normalized'] != rp['normalized']
    if policy_changed and not allow_policy_change:
        issues.append('policy_change_not_explicitly_allowed')
    keys = sorted(set(left) | set(right))
    if set(left) != set(right):
        issues.append('paired_request_inventory_mismatch')
    if la['counts']['planned'] != ra['counts']['planned']:
        issues.append('planned_denominator_mismatch')
    same_hashes = {key: la['input_sha256'].get(key) == ra['input_sha256'].get(key)
                   and la['input_sha256'].get(key) is not None
                   for key in ('answers', 'frozen_inputs', 'source-input.json', 'rendered-inputs.json')}
    if not all(same_hashes.values()):
        issues.append('source_answers_or_rendered_input_hash_mismatch')
    engine_diff = differences(ls['config']['engine_args'], rs['config']['engine_args'])
    if set(engine_diff) - set(allowed):
        issues.append('unexpected_engine_configuration_difference')
    config_diff = differences({k:v for k,v in ls['config'].items() if k not in ('engine_args','scope','policy')},
                              {k:v for k,v in rs['config'].items() if k not in ('engine_args','scope','policy')})
    if config_diff:
        issues.append('sampling_arrival_or_other_configuration_difference')
    if la['diagnostic_thresholds'] != ra['diagnostic_thresholds']:
        issues.append('diagnostic_thresholds_mismatch')
    transitions = {k: Counter() for k in ('correct', 'slo_pass', 'correct_and_slo')}
    metrics = ('ttft_s', 'completion_s', 'max_host_gap_s')
    directions = {k: Counter() for k in metrics}
    stable_directions = {k: Counter() for k in metrics}
    consistency, pairs = Counter(), []
    for rid in keys:
        l, r = left.get(rid, {}), right.get(rid, {})
        if l.get('gold') != r.get('gold'):
            issues.append('gold_mismatch:' + rid)
        sig_l, sig_r = lo.get(rid, {}), ro.get(rid, {})
        text_same = bool(sig_l and sig_r) and sig_l['text_sha256'] == sig_r['text_sha256']
        tokens_same = bool(sig_l and sig_r) and sig_l['token_ids_sha256'] == sig_r['token_ids_sha256']
        stable = text_same and tokens_same
        consistency['identical_text'] += text_same
        consistency['identical_token_ids'] += tokens_same
        consistency['identical_text_and_tokens'] += stable
        consistency['changed_text_or_tokens'] += not stable
        pair = dict(request_id=rid, identical_text=text_same, identical_token_ids=tokens_same,
                    output_tokens=change(l.get('output_tokens'), r.get('output_tokens')))
        for key in transitions:
            state = ('true' if l.get(key) is True else 'false') + '_to_' + ('true' if r.get(key) is True else 'false')
            transitions[key][state] += 1
            pair[key] = dict(left=l.get(key), right=r.get(key))
        for key in metrics:
            delta = change(l.get(key), r.get(key))
            value = delta['right_minus_left']
            direction = 'missing' if value is None else 'improved' if value < -1e-9 else 'worsened' if value > 1e-9 else 'tied'
            directions[key][direction] += 1
            if stable:
                stable_directions[key][direction] += 1
            pair[key] = dict(delta, direction=direction)
        pairs.append(pair)
    timing = {key: {stat: change(la['timing'][key].get(stat), ra['timing'][key].get(stat))
                    for stat in ('mean','max','p50','p90','p95','observed','missing')}
              for key in la['timing'] if key in ra['timing']}
    integrity = 'MISMATCH' if issues else 'MATCHED_WITH_POLICY_CHANGE' if policy_changed else 'MATCHED'
    tail_request_ids = set()
    for arm in (left, right):
        completed = [r for r in arm.values() if numeric(r.get('completion_s'))]
        if completed:
            tail_request_ids.add(max(completed, key=lambda r:r['completion_s'])['request_id'])
    tail_request_ids.add('gsm8k-test-0236')
    output_tail_sensitivity = []
    for rid in sorted(tail_request_ids & set(left) & set(right)):
        l, r = left[rid], right[rid]
        output_tail_sensitivity.append(dict(request_id=rid,
            output_tokens=change(l['output_tokens'],r['output_tokens']),
            completion_s=change(l['completion_s'],r['completion_s']),
            left_truncated=l.get('truncated'), right_truncated=r.get('truncated'),
            left_correct=l.get('correct'), right_correct=r.get('correct'),
            left_finish_reason=l.get('finish_reason'), right_finish_reason=r.get('finish_reason')))
    return dict(schema='c-math-run-pair-v3', comparison_integrity=integrity,
        issues=issues, direction='All changes are right minus left; lower timing is better.',
        left=ls, right=rs,
        comparability=dict(planned_requests=la['counts']['planned'], source_hash_matches=same_hashes,
            allowed_engine_difference_keys=list(allowed), engine_arg_differences=engine_diff,
            policy=dict(left=lp, right=rp, changed=policy_changed,
                        change_explicitly_allowed=allow_policy_change),
            other_config_differences=config_diff,
            same_analyzer=la['analyzer_sha256'] == ra['analyzer_sha256'],
            scope='Configuration receipts establish the declared changes; they do not prove arbitrary custom policy code equivalence.'),
        count_changes={k:change(la['counts'].get(k),ra['counts'].get(k)) for k in set(la['counts']) | set(ra['counts'])},
        episode_changes={k:change(la['episode'].get(k),ra['episode'].get(k)) for k in set(la['episode']) | set(ra['episode'])},
        timing_changes=timing, per_request_directions={k:dict(v) for k,v in directions.items()},
        transitions={k:dict(v) for k,v in transitions.items()}, output_consistency=dict(consistency),
        identical_output_subset_directions={k:dict(v) for k,v in stable_directions.items()},
        output_tail_sensitivity=output_tail_sensitivity,
        per_request=pairs, diagnostic_thresholds=la['diagnostic_thresholds'],
        limitations=['All planned requests remain in the comparison, including wrong answers, truncations and failures.',
            'Historical TTFT20s/max-host-gap4s thresholds are diagnostic, not a final paper or business SLO.',
            'Different generated outputs confound a pure scheduling speed comparison; identical-output sensitivity is a selected subset.',
            'Makespan improvements with shorter tail outputs are not pure scheduling gains; inspect mean completion, individual regressions and output_tail_sensitivity.',
            'Single finite bursts do not establish repeatability or a causal policy benefit.',
            'Gate coverage can overlap. Scheduler wrapper time cannot isolate observer overhead.'],
        comparator_sha256=digest(__file__))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--left-analysis', required=True, type=Path)
    p.add_argument('--right-analysis', required=True, type=Path)
    p.add_argument('--left-run', type=Path)
    p.add_argument('--right-run', type=Path)
    p.add_argument('--left-label', default='left')
    p.add_argument('--right-label', default='right')
    p.add_argument('--allow-engine-difference', action='append', default=None)
    p.add_argument('--allow-policy-change', action='store_true',
                   help='Explicitly permit a declared native/FIFO/runway policy comparison.')
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = compare(args.left_analysis, args.right_analysis, args.left_run, args.right_run,
                     args.left_label, args.right_label,
                     args.allow_engine_difference or ('max_num_seqs',), args.allow_policy_change)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k:result[k] for k in ('comparison_integrity','issues','episode_changes',
                    'per_request_directions','transitions','output_consistency')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
