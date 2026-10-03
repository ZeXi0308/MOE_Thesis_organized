"""Retain all F/X costs and reject unfulfilled compiler-domain qualification."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import finite_metrics

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'instrumentation'))
from compile_domain import EXPECTED
from host_probe import EXPECTED as OBSERVED_SOURCES


def read(path):
    return json.loads(path.read_text())


def analyze_cell(root, plan, execution, workload_path=None, warmup_document_id=None):
    row = finite_metrics.cell(root, plan, execution, workload_path=workload_path)
    path, issues = root / 'results' / plan['label'], row['issues']

    def check(ok, why):
        if not ok:
            issues.append(why)

    setup, probe = [read(path / name) for name in ('compile_domain.json', 'jit_probe.json')]
    check(setup['schema'] == 'moe_compile_domain_v1' and setup['status'] == 'COMPLETE_COMPILE_AND_LOAD_ONLY',
          'compile-domain setup incomplete')
    check(setup['mode'] == plan['mode'] and setup['actual_invocations'] == setup['expected_invocations'] == 320
          and len(setup['calls']) == 320, 'compile invocation count/mode mismatch')
    check({(c['m'], c['gemm']) for c in setup['calls']} == {(m, g) for m in range(1,161) for g in (1,2)},
          'incomplete fixed integer M/GEMM domain')
    check(all(c['status'] == 'COMPLETE' and c['interceptions'] == 1 and c['handles_initialized_after']
              and c['end_perf_ns'] >= c['start_perf_ns'] for c in setup['calls']), 'compile/handle invocation failed')
    check(setup['state_equal'] and setup['before'] == setup['after'], 'pager state changed during setup')
    check({k:v['sha256'] for k,v in setup['sources'].items()} == EXPECTED, 'native helper source mismatch')
    check(probe['schema'] == 'covered_moe_host_probe_v1' and probe['status'] == 'COMPLETE', 'observer incomplete')
    check(probe['installed_sources'] == OBSERVED_SOURCES, 'observer installed source mismatch')
    check(probe['kv_before_compile_setup'] == probe['kv_after_compile_setup']
          and not probe['kv_before_compile_setup']['requests'], 'KV setup state mismatch or nonempty request set')
    check(all(hashlib.sha256((root/'instrumentation'/name).read_bytes()).hexdigest() == h
              for name,h in probe['observer_sources'].items()), 'executed instrumentation mismatch')
    events = probe['events']
    check([e['event_id'] for e in events] == list(range(len(events))) and all(e.get('status') == 'complete'
          and e.get('end_perf_ns', -1) >= e.get('start_perf_ns', 0) for e in events), 'failed/unwrapped observer event')
    compiler = [e for e in events if e['kind'] == 'compiler_call']
    check(all(type(e.get('cache_hit')) is bool for e in compiler), 'unclassified compiler event')
    measured_compiler = [e for e in compiler if e['context'].get('phase') == 'measurement']
    check(not any(e['cache_hit'] is False for e in measured_compiler), 'new measurement compiler pipeline: coverage unfulfilled')
    check(execution.get('coverage_valid') is True, 'driver coverage check failed')
    check(not execution['cache_before'] and execution['cache_after'], 'per-engine private cache did not start empty')
    warm = read(path/'warmup.json')['requests'][0]
    check(warm['document_id'] == (warmup_document_id or read(root/'protocol.json')['warmup']['document_id'])
          and warm['prompt_tokens'] == 128 and len(warm['output_token_ids']) == 2, 'exact request warmup mismatch')
    row.update(compile_setup_wall_s=setup['setup_wall_s'], compile_domain_invocations=len(setup['calls']),
               unique_precompiled_keys=len(setup['unique_cache_keys']),
               setup_memory={k:setup[k] for k in ('memory_before','memory_with_buffers','memory_after_release')},
               temporary_setup_bytes=setup['temporary_buffer_bytes'],
               compiler_events_by_phase={str(phase):dict(Counter(e['result_kind'] for e in compiler
                   if e['context'].get('phase') == phase)) for phase in
                   dict.fromkeys(e['context'].get('phase') for e in compiler)},
               measurement_new_compiler_events=[e['event_id'] for e in measured_compiler if not e['cache_hit']],
               observer_event_count=len(events), cache_entries_after=len(execution['cache_after']))
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=ROOT)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root = args.input_dir.resolve()
    execution = read(root/'results/execution.json') if (root/'results/execution.json').exists() else {'cells':[]}
    entries = {c['label']:c for c in execution['cells']}
    rows, missing, issues = [], [], []
    for plan in read(root/'run_cells.json'):
        if entries.get(plan['label'],{}).get('status') != 'COMPLETE':
            missing.append(plan['label'])
            continue
        try:
            row = analyze_cell(root,plan,entries[plan['label']]); rows.append(row)
            issues.extend(plan['label']+': '+s for s in row['issues'])
        except Exception as exc:
            issues.append(plan['label']+': '+repr(exc))
    comparisons = [finite_metrics.comparison(rows[a],rows[b]) for a,b in ((0,1),(3,2))] if len(rows)==4 else []
    repeats = [finite_metrics.comparison(rows[a],rows[b]) for a,b in ((0,3),(1,2))] if len(rows)==4 else []
    result = dict(status='INVALID_COVERAGE_OR_ANALYSIS' if issues else 'UNRUN_OR_PARTIAL' if missing
                  else 'DESCRIPTIVE_FX_WITH_COMPILER_DOMAIN', issues=issues, missing_cells=missing,
                  cells=rows, comparisons=comparisons, same_arm_repeats=repeats,
                  claim_ceiling='Single new16-document finite cohort, two engines/mode, ordered F/X/X/F. '
                  'MoE compile-domain setup is fully costed, not a guarantee of all-runtime steady state. '
                  'Own trajectories; no time subtraction, significance, quality or method GO.')
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('status','issues','missing_cells')}))


if __name__ == '__main__':
    main()
