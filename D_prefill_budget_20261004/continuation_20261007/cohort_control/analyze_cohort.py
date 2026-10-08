#!/usr/bin/env python3
"""Three-arm adapter over the existing complete-service component analysis."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent/'niyama_component'))
import analyze_component as component

POLICIES = ['fixed1024', 'progress_floor', 'cohort_protection', 'cohort_protection', 'progress_floor', 'fixed1024']
WARMS = ['fixed1024', 'progress_floor', 'cohort_protection']
REFERENCES = ['original_N', 'progress_floor', 'fixed1024']


def relative_execution(raw):
    """Actual binding/progress relative to logged caps, never a service counterfactual."""
    result = {name: dict(scored_completed_steps=0, unavailable_completed_steps=0,
        backlog_nominal_cap_differs_steps=0, actual_P_above_reference_indices=[],
        binding_cap_below_reference_indices=[]) for name in REFERENCES}
    errors, sources, statuses = [], Counter(), Counter()
    for i, s in enumerate(raw['steps']):
        d = s['component_decision']
        if not d.get('forward_completed') or s.get('end_s') is None: continue
        sources[d.get('execution_source', 'MISSING')] += 1
        statuses[d.get('cohort_status', 'MISSING')] += 1
        caps = d.get('reference_caps', {})
        for name in REFERENCES:
            out = result[name]; ref = caps.get(name)
            if ref is None:
                out['unavailable_completed_steps'] += 1
                if not (raw['policy'] == 'fixed1024' and name != 'fixed1024'):
                    errors.append(f'Missing completed reference cap {name} at step {i}')
                continue
            if isinstance(ref, bool) or not isinstance(ref, int) or ref < 0:
                errors.append(f'Invalid reference cap {name} at step {i}'); continue
            out['scored_completed_steps'] += 1
            p, cap, backlog = s['prefill_tokens'], d['requested_cap'], d['prefill_backlog_tokens']
            out['backlog_nominal_cap_differs_steps'] += int(backlog > 0 and cap != ref)
            if p > ref: out['actual_P_above_reference_indices'].append(i)
            if cap < ref and p == cap and backlog > cap: out['binding_cap_below_reference_indices'].append(i)
    for out in result.values():
        out['actual_P_above_reference_steps'] = len(out['actual_P_above_reference_indices'])
        out['binding_cap_below_reference_steps'] = len(out['binding_cap_below_reference_indices'])
        out['actual_intervention_steps'] = out['actual_P_above_reference_steps']+out['binding_cap_below_reference_steps']
        out['zero_actual_intervention_observed'] = out['actual_intervention_steps'] == 0 if out['scored_completed_steps'] else None
    return dict(errors=errors, references=result, completed_execution_sources=dict(sources),
        completed_cohort_statuses=dict(statuses), scope='Completed forwards only. P exceeds the reference cap, or final cap is below it and binds actual P with backlog remaining. Fixed runs do not evaluate N/progress reference caps; null is unavailable, not zero intervention. Same-state cap comparisons are not performance counterfactuals.')


def load(path, design, signature, calibration, design_hash):
    row = component.load_run(path, design, signature, calibration, design_hash)
    if '_raw' in row:
        try:
            row['relative_execution'] = relative_execution(row['_raw'])
            row['errors'].extend(row['relative_execution']['errors'])
        except Exception as exc: row['errors'].append('Relative execution: '+repr(exc))
    row['valid'] = row['valid'] and not row['errors']
    row['status'] = 'VALID_COMPLETE' if row['valid'] else 'INVALID_OR_INCOMPLETE'
    return row


def compare(reference, candidate, design):
    # Reuse all native request migration/cost gates; replace only the action
    # evidence so C/P is checked against P, not the helper's fixed1024 default.
    adapted = candidate
    action = None
    if candidate and reference and 'execution' in candidate:
        evidence = candidate.get('relative_execution', {}).get('references', {}).get(reference['policy'], {})
        action = evidence.get('actual_intervention_steps') if evidence.get('scored_completed_steps', 0) else None
        adapted = dict(candidate, execution=dict(candidate['execution'], actual_intervention_steps=action or 0))
    out = component.compare(reference, adapted, design)
    out['reference_cell'] = out.pop('fixed_cell')
    out['candidate_cell'] = out.pop('component_cell')
    out['candidate_actual_intervention_against_reference_steps'] = action
    out['directions'] = {('candidate_over_reference' if k == 'component_over_fixed' else 'reference_over_candidate'): v for k, v in out['directions'].items()}
    for direction in out['directions'].values():
        checks = direction['gate_checks']
        checks['actual_candidate_intervention_against_reference_observed'] = checks.pop('actual_component_intervention_observed')
    for old, new in [('flow_component_minus_fixed_s_by_request', 'flow_candidate_minus_reference_s_by_request'),
                     ('component_flow_harmed_ids', 'candidate_flow_harmed_ids'), ('component_flow_improved_ids', 'candidate_flow_improved_ids')]:
        if old in out: out[new] = out.pop(old)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory', type=Path)
    parser.add_argument('--design', type=Path, default=ROOT/'design.json')
    parser.add_argument('--output', type=Path); args = parser.parse_args()
    db = args.design.read_bytes(); design = json.loads(db)
    if design['policies'] != POLICIES or design['warm_policies'] != WARMS: parser.error('Expected frozen F/P/C/C/P/F and F/P/C warm order')
    work = (args.design.parent/design['workload']).read_bytes()
    if component.sha(work) != design['workload_sha256']: parser.error('Frozen workload hash mismatch')
    cp = args.run_directory/'calibration.json'
    if not cp.exists(): cp = args.design.parent/design['calibration_file']
    cb = cp.read_bytes(); calibration = json.loads(cb)
    if component.sha(cb) != design['calibration_sha256']: parser.error('Frozen calibration hash mismatch')
    expected = [f'{i:02d}_{p}' for i, p in enumerate(POLICIES)]; warms = ['warm_'+p for p in WARMS]
    signature = component.signature(json.loads(work))
    rows = [load(p, design, signature, calibration, component.sha(db)) for p in sorted(args.run_directory.rglob('raw.json'))]
    def unique(name):
        matches = [r for r in rows if r['cell'] == name]
        return matches[0] if len(matches) == 1 else None
    missing = [n for n in expected+warms if not any(r['cell'] == n for r in rows)]
    duplicates = [n for n in expected+warms if sum(r['cell'] == n for r in rows) > 1]
    unexpected = [r['raw_path'] for r in rows if r['cell'] not in expected+warms]
    pairs = {label: [compare(unique(expected[r]), unique(expected[c]), design) for r, c in indices]
        for label, indices in {'C_vs_F': [(0, 2), (5, 3)], 'C_vs_P': [(1, 2), (4, 3)], 'P_vs_F': [(0, 1), (5, 4)]}.items()}
    sp = args.run_directory/'status.json'
    try: controller = json.loads(sp.read_text()) if sp.exists() else dict(status='MISSING')
    except Exception as exc: controller = dict(status='UNREADABLE', error=repr(exc))
    coverage = {}
    for policy in WARMS:
        warm = unique('warm_'+policy); formal = [r for r in rows if r['cell'] in expected and r.get('policy') == policy]
        we = (warm or {}).get('execution', {})
        coverage[policy] = dict(warm_valid_complete=bool(warm and warm['valid']))
        for name in ('completed_cap_distribution', 'completed_selected_total_tier_distribution', 'actual_P_distribution', 'actual_total_P_plus_D_distribution'):
            values = set().union(*(set(r.get('execution', {}).get(name, {})) for r in formal))
            coverage[policy][name] = dict(formal_values=sorted(values, key=int), warm_values=sorted(we.get(name, {}), key=int))
    observed = sum(r['cell'] in expected for r in rows)
    ready = not missing and not duplicates and not unexpected and controller.get('status') == 'COMPLETE' and all((unique(n) or {}).get('valid', False) for n in expected+warms)
    zero_against_p = [n for n in (expected[2], expected[3]) if unique(n) and unique(n).get('relative_execution', {}).get('references', {}).get('progress_floor', {}).get('zero_actual_intervention_observed') is True]
    comparison_passes = {label: ready and all(p['directions'].get('candidate_over_reference', {}).get('passes_frozen_gate', False) for p in ps) for label, ps in pairs.items()}
    candidate_passes = comparison_passes['C_vs_F'] and comparison_passes['C_vs_P']
    status = 'UNRUN' if not rows and controller.get('status') not in ('COMPLETE', 'FAILED', 'UNREADABLE') else 'INCOMPLETE_OR_INVALID'
    if ready: status = 'ZERO_INTERVENTION_AGAINST_PROGRESS_FLOOR' if zero_against_p else 'PROVISIONAL_CANDIDATE_SERVICE_WIN' if candidate_passes else 'NO_FROZEN_CANDIDATE_SERVICE_WIN'
    result = dict(schema='d-cohort-three-arm-service-v1', status=status, observed_formal_runs=observed,
        valid_complete_formal_runs=sum(r['cell'] in expected and r['valid'] for r in rows),
        controller_status=controller, missing_cells=missing, duplicate_cells=duplicates, unexpected_raw_paths=unexpected,
        candidate_zero_intervention_against_progress_floor_cells=zero_against_p,
        both_pairs_pass_by_comparison=comparison_passes, candidate_passes_both_comparators=candidate_passes,
        pairs=pairs, warm_execution_coverage=coverage,
        runs=[{k: v for k, v in r.items() if not k.startswith('_')} for r in rows],
        frozen_design=design, frozen_design_sha256=component.sha(db), calibration_sha256=component.sha(cb),
        scope='Reuse native complete-service summaries, model errors, request statistics and unchanged 3%/raw/p95/100ms gates. Both mirrored C/F and C/P pairs must pass; P/F is also reported. Preserve all requests, output divergence, arrival/add lag and full elapsed/drain; full elapsed remains the denominator. Warm coverage is descriptive, not an invented exact-cap gate. Same-state N/P/F cap differences are execution evidence, not causal service counterfactuals. Fixed output counts do not establish equal content/routes/computation/quality. No refit, new SLO or automatic repeat.')
    output = args.output or args.run_directory/'cohort_results.json'; output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    print(f'{observed} formal, {result["valid_complete_formal_runs"]} valid complete; {status}; {output}')


if __name__ == '__main__': main()
