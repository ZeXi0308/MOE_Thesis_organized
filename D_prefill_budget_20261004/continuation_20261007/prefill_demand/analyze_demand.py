#!/usr/bin/env python3
"""Small complete-service adapter for an ordinary prefill-demand baseline."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent/'niyama_component'))
import analyze_component as component

POLICIES = ['fixed1024', 'prefill_demand', 'prefill_demand', 'fixed1024']
WARMS = ['fixed512', 'fixed1024', 'fixed2048', 'prefill_demand']
FALLBACKS = ['expired_pending_deadline_progress_fallback',
             'capacity_insufficient_progress_fallback', 'invalid_model_progress_fallback']


def load(path, design, signature, calibration, design_hash):
    row = component.load_run(path, design, signature, calibration, design_hash)
    if '_raw' not in row: return row
    try:
        raw = row['_raw']
        completed = [(s, s['component_decision']) for s in raw['steps']
                     if s['component_decision'].get('forward_completed') and s['end_s'] is not None]
        if row['policy'].startswith('fixed'):
            cap = int(row['policy'][5:])
            if any(d['requested_cap'] != cap for _, d in completed):
                row['errors'].append(f'Fixed{cap} completed cap changed')
        backlog = [(s, d) for s, d in completed if d['prefill_backlog_tokens'] > 0]
        reasons = Counter(d['reason'] for _, d in backlog)
        is_demand = row['policy'] == 'prefill_demand'
        row['demand_decisions'] = dict(completed_backlog_steps=len(backlog),
            backlog_zero_actual_P_steps=sum(s['prefill_tokens'] == 0 for s, _ in backlog),
            backlog_actual_P_sum=sum(s['prefill_tokens'] for s, _ in backlog),
            backlog_actual_P=component.native.describe([s['prefill_tokens'] for s, _ in backlog]),
            backlog_actual_P_distribution=component.distribution(s['prefill_tokens'] for s, _ in backlog),
            completed_backlog_reason_distribution=dict(reasons),
            completed_backlog_cap_distribution=component.distribution(d['requested_cap'] for _, d in backlog),
            demand_evaluated=is_demand,
            fallback_counts={name: reasons[name] for name in FALLBACKS} if is_demand else None,
            expired_pending_decisions=sum(d['expired_pending_count'] > 0 for _, d in backlog) if is_demand else None,
            invalid_model_decisions=sum(d['demand_model_error'] is not None for _, d in backlog) if is_demand else None,
            finite_demand_tokens_per_s=component.native.describe([
                d['demand_tokens_per_s'] for _, d in backlog if d.get('demand_tokens_per_s') is not None]) if is_demand else None,
            scope='Completed forwards; every backlog step counts, including actual P=0. Fixed-policy demand/fallback fields are unavailable, not zero. Actual interventions and total control cost are retained in execution.')
    except Exception as exc: row['errors'].append('Demand execution summary: '+repr(exc))
    row['valid'] = row['valid'] and not row['errors']
    row['status'] = 'VALID_COMPLETE' if row['valid'] else 'INVALID_OR_INCOMPLETE'
    return row


def compare(fixed, demand, design):
    out = component.compare(fixed, demand, design)
    for direction in out['directions'].values():
        direction['legacy_gate_checks'] = direction.pop('gate_checks')
        direction['passes_legacy_gate'] = direction.pop('passes_frozen_gate')
        cn, rn = direction['candidate_qualified_numerator'], direction['reference_qualified_numerator']
        ct, rt = direction['candidate_full_elapsed_s'], direction['reference_full_elapsed_s']
        cu, ru = cn/ct if ct > 0 else None, rn/rt if rt > 0 else None
        direction.update(candidate_U_req_per_s=cu, reference_U_req_per_s=ru,
            delta_U_req_per_s=cu-ru if cu is not None and ru is not None else None)
    out['interpretation'] = 'Both directions retain every request and full elapsed. Old3%/5% gates are legacy descriptions, not this study status or a scientific acceptance rule.'
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory', type=Path)
    parser.add_argument('--design', type=Path, default=ROOT/'design.json')
    parser.add_argument('--output', type=Path); args = parser.parse_args()
    db = args.design.read_bytes(); design = json.loads(db)
    if design['policies'] != POLICIES or design['warm_policies'] != WARMS:
        parser.error('Expected F/D/D/F and fixed512/fixed1024/fixed2048/demand warm order')
    work = (args.design.parent/design['workload']).read_bytes()
    if component.sha(work) != design['workload_sha256']: parser.error('Frozen workload hash mismatch')
    cp = args.run_directory/'calibration.json'
    if not cp.exists(): cp = args.design.parent/design['calibration_file']
    cb = cp.read_bytes(); calibration = json.loads(cb)
    if component.sha(cb) != design['calibration_sha256']: parser.error('Frozen calibration hash mismatch')
    expected = [f'{i:02d}_{p}' for i, p in enumerate(POLICIES)]; warms = ['warm_'+p for p in WARMS]
    signature = component.signature(json.loads(work))
    rows = [load(p, design, signature, calibration, component.sha(db))
            for p in sorted(args.run_directory.rglob('raw.json'))]
    def unique(name):
        matches = [r for r in rows if r['cell'] == name]
        return matches[0] if len(matches) == 1 else None
    missing = [n for n in expected+warms if not any(r['cell'] == n for r in rows)]
    duplicates = [n for n in expected+warms if sum(r['cell'] == n for r in rows) > 1]
    unexpected = [r['raw_path'] for r in rows if r['cell'] not in expected+warms]
    pairs = [compare(unique(expected[0]), unique(expected[1]), design),
             compare(unique(expected[3]), unique(expected[2]), design)]
    sp = args.run_directory/'status.json'
    try: controller = json.loads(sp.read_text()) if sp.exists() else dict(status='MISSING')
    except Exception as exc: controller = dict(status='UNREADABLE', error=repr(exc))
    coverage = {}
    for policy in WARMS:
        warm = unique('warm_'+policy); we = (warm or {}).get('execution', {})
        cap = int(policy[5:]) if policy.startswith('fixed') else None
        count = we.get('actual_P_distribution', {}).get(str(cap), 0) if cap is not None else None
        coverage[policy] = dict(warm_valid_complete=bool(warm and warm['valid']),
            fixed_actual_cap_required=cap, completed_actual_P_equals_fixed_cap_steps=count,
            completed_cap_distribution=we.get('completed_cap_distribution', {}),
            actual_P_distribution=we.get('actual_P_distribution', {}),
            actual_total_P_plus_D_distribution=we.get('actual_total_P_plus_D_distribution', {}))
    missing_warm_caps = [int(p[5:]) for p in WARMS if p.startswith('fixed')
                         and not coverage[p]['completed_actual_P_equals_fixed_cap_steps']]
    observed = sum(r['cell'] in expected for r in rows)
    ready = (not missing and not duplicates and not unexpected and not missing_warm_caps
             and controller.get('status') == 'COMPLETE'
             and all((unique(n) or {}).get('valid', False) for n in expected+warms))
    status = 'UNRUN' if not rows and controller.get('status') not in ('COMPLETE', 'FAILED', 'UNREADABLE') else 'INCOMPLETE_OR_INVALID'
    if ready: status = 'COMPLETE_EXPLORATORY'
    zero_action = [n for n in (expected[1], expected[2]) if unique(n)
                   and unique(n).get('valid') and unique(n)['execution']['actual_intervention_steps'] == 0]
    repetitions = {}
    for policy in ('fixed1024', 'prefill_demand'):
        episodes = [dict(cell=r['cell'], qualified_numerator=r['summary']['joint_slo_qualified_requests'],
            full_elapsed_s=r['summary']['elapsed_s'], U=r['summary']['joint_slo_requests_per_s'])
            for r in rows if r['cell'] in expected and r.get('policy') == policy and r.get('valid')]
        repetitions[policy] = dict(episodes=episodes,
            second_minus_first_U_req_per_s=episodes[1]['U']-episodes[0]['U'] if len(episodes) == 2 else None,
            sum_qualified_over_sum_full_elapsed=sum(r['qualified_numerator'] for r in episodes)/sum(r['full_elapsed_s'] for r in episodes)
                if ready and len(episodes) == 2 else None)
    result = dict(schema='d-prefill-demand-service-v1', status=status, observed_formal_runs=observed,
        valid_complete_formal_runs=sum(r['cell'] in expected and r['valid'] for r in rows),
        observed_warm_runs=sum(r['cell'] in warms for r in rows),
        valid_complete_warm_runs=sum(r['cell'] in warms and r['valid'] for r in rows), controller_status=controller,
        demand_cells_without_actual_intervention=zero_action,
        zero_action_note='If the frozen runner stops after the first zero-action demand arm, retain both completed cells and missing reverse cells; this remains incomplete, not a service negative or permission to retry.',
        missing_cells=missing, duplicate_cells=duplicates, unexpected_raw_paths=unexpected,
        missing_fixed_warm_actual_caps=missing_warm_caps, warm_execution_coverage=coverage,
        pairs=pairs, descriptive_repetitions=repetitions,
        runs=[{k: v for k, v in r.items() if not k.startswith('_')} for r in rows],
        frozen_design=design, frozen_design_sha256=component.sha(db), calibration_sha256=component.sha(cb),
        analysis_source_sha256=component.sha(Path(__file__).read_bytes()),
        reused_component_analyzer_sha256=component.sha(Path(component.__file__).read_bytes()),
        scope='Ordinary deadline-demand baseline development comparison, not a novelty or significance claim. '
              'Primary4s/100ms/20s is a research contract without established production justification. '
              'COMPLETE_EXPLORATORY means complete valid4formal+4warm, driverCOMPLETE and fixedwarm actual512/1024/2048 coverage; '
              'it does not require old3%/5% performance gates or actual intervention. All requests, failed attempts, '
              'output divergence, arrival/add lag, full elapsed/drain and control/model diagnostics are retained. '
              'Full elapsed is the denominator; pooled descriptions do not override pair signs. Fixed output counts '
              'do not establish equal content/routes/computation/quality. No automatic winner, continuation or retry.')
    output = args.output or args.run_directory/'demand_results.json'; output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    print(f'{observed} formal, {result["valid_complete_formal_runs"]} valid complete; {status}; {output}')


if __name__ == '__main__': main()
