"""Read retained finite-arrival cells; all repeats count, no inferential noise floor."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'analysis_source'))
from analyze_layer_budget import UPSTREAM_SHA
from analyze_native_pager_transfer import requests, call_accounting
from analyze_shared_pool_execution import amounts, trace_check


def read(p):
    return json.loads(p.read_text())


def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def cell(root, declared, execution, workload_path=None):
    path = root / 'results' / declared['label']
    issues = []

    def check(ok, why):
        if not ok:
            issues.append(why)

    raw, work, pager = [read(path / n) for n in ('raw.json', 'workload.json', 'pager_summary.json')]
    config, resource = [read(path / n) for n in ('config.json', 'measurement_resources.json')]
    prepared = read(workload_path or root / 'prepared' / 'workload.json')
    check(execution['returncode'] == 0 and execution['status'] == 'COMPLETE' and
          read(path / 'status.json')['status'] == raw['status'] == 'COMPLETE', 'incomplete execution')
    check(work['source_requests'] == prepared['source_requests'] and work['actual_prompt_token_ids'] ==
          prepared['actual_prompt_token_ids'], 'prepared input mismatch')
    check(work['arrival_traces_s']['steady'] == [i * 0.25 for i in range(16)], 'arrival schedule mismatch')
    fixed = dict(requests=16, prompt_tokens=128, output_tokens=32, arrival_interval=0.25,
                 token_budget=160, prefill_limit=32, expert_cap=24, execution='expert',
                 group_retention='none', verify_kernel=False, injection_chunk=None,
                 same_engine_runtime_diagnostic=False, warmup_full=False, trace_retention='memory')
    check(all(config.get(k) == v for k, v in fixed.items()), 'runtime configuration mismatch')
    check(not pager['validation_run'] and not pager['kernel_validation'], 'performance reference enabled')
    check(pager['upstream_sha256'] == UPSTREAM_SHA, 'external WiSP source mismatch')
    hashes = read(root / 'sources.json')
    actual = read(path / 'environment.json')['sources']
    check(bool(actual) and all(hashes.get(n) == h for n, h in actual.items()), 'runtime source mismatch')
    mode = declared['mode']
    if mode in ('oneshot', 'fullstage'):
        shared = read(path / 'shared_pool.json')
        check(shared['status'] == 'COMPLETE' and shared['mode'] == mode and not shared['qualification']
              and not shared['validation'] and shared['sources'] == hashes, 'shared execution mismatch')
    caps = {r['layer_name']: r['cap'] for r in pager['layers']}
    sizes = {r['layer_name']: r['pinned_bytes'] // r['num_experts'] for r in pager['layers']}
    expected = (read(root / 'allocations' / (mode + '.json')) if mode in ('uniform', 'selected') else
                {name: 20 if mode == 'fullstage' else 21 for name in caps})
    check(caps == expected and len(caps) == 16 and set(sizes.values()) == {12582912}, 'layer allocation mismatch')
    check(pager['expert_scratch_bytes'] == resource['expert_scratch_bytes'] == 4831838208 and
          resource['expert_slots_total'] == 384 and resource['actual_unique_kv_storage_bytes'] == 1073741824,
          'actual expert/KV budget mismatch')
    check(resource['cpu_affinity'] == list(range(8)) and resource['scheduler_requests'] == 0,
          'affinity or drained start mismatch')
    req, req_issues = requests(raw, work)
    issues.extend(req_issues)
    check(len(req) == 16 and all(r['status'] == 'completed' and len(r['output_token_ids']) == 32
                               for r in req.values()), 'request/output coverage mismatch')
    warmup = read(path / 'warmup.json')
    check(warmup['status'] == 'COMPLETE' and len(warmup['requests']) == 1 and
          len(warmup['requests'][0]['output_token_ids']) == 2 and not (path / 'warmup_full.json').exists(),
          'single initial warmup mismatch')
    trace = [json.loads(line) for line in (path / 'pager/calls.jsonl').read_text().splitlines()]
    measured = [r for r in trace if r['measurement']]
    check([r['call_id'] for r in trace] == list(range(len(trace))) and
          all(r['status'] == 'complete' and not r['validation_run'] for r in trace), 'trace status/sequence mismatch')
    check(all(r['byte_count_valid'] for r in call_accounting(trace, sizes)), 'all-phase H2D accounting mismatch')
    rebuilt = trace_check(measured, caps, sizes, pager['measurement_initial_cache'], raw)
    issues.extend(rebuilt['issues'])
    phases = defaultdict(list)
    for r in trace:
        phases[r['context']['phase']].append(r)
    check(set(phases) <= {'initialization', 'warmup', 'measurement'}, 'unexpected reset/warmup phase')
    all_cost, measured_cost = amounts(trace), amounts(measured)
    check(all_cost['payload_bytes'] == pager['all']['weight_copy_bytes'] and
          measured_cost['payload_bytes'] == pager['measurement']['weight_copy_bytes'], 'pager summary bytes mismatch')
    checks = [json.loads(line) for line in (path / 'gpu_checks.jsonl').read_text().splitlines()]
    check(len(checks) >= 3 and all(g['decision'] == 'PASS' and not g['query_errors'] and not g['foreign_pids']
          and 'GPU-70fa1c0a-77d4-c14a-9daf-7e685874eef9' in g['gpu'] for g in checks), 'GPU boundary mismatch')
    first_schedule = {}
    absences = []
    for step in raw['scheduler_steps']:
        output_bearing = {r['request_id'] for r in raw['requests']
                          if r['token_times_s'][0] < step['start_s'] < r['completion_s']}
        advancing = {r['request_id'] for r in step['scheduled'] if r['decode_tokens'] > 0}
        if output_bearing - advancing:
            absences.append(dict(step=step['step'], request_ids=sorted(output_bearing - advancing)))
        for scheduled in step['scheduled']:
            first_schedule.setdefault(scheduled['request_id'], step['start_s'])
    for row in raw['requests']:
        q = req[row['request_id']]
        first = first_schedule[row['request_id']]
        q.update(engine_add_duration_s=row['engine_add_return_s'] - row['admission_s'],
                 after_add_to_first_schedule_s=first - row['engine_add_return_s'],
                 first_schedule_to_first_token_s=row['token_times_s'][0] - first)
        check(min(q[k] for k in ('client_submission_lag_s', 'engine_add_duration_s',
                  'after_add_to_first_schedule_s', 'first_schedule_to_first_token_s')) >= -1e-6,
              'negative request time decomposition')
    metrics = {k + '_mean': statistics.mean(r[k] for r in req.values()) for k in
               ('ttft_s', 'tpot_s', 'max_itl_s', 'completion_latency_s', 'client_submission_lag_s',
                'engine_add_duration_s', 'after_add_to_first_schedule_s')}
    metrics.update(capture_wall_s=raw['observation_end_s'], cohort_last_completion_s=max(r['completion_s'] for r in raw['requests']),
                   max_request_itl_s=max(r['max_itl_s'] for r in req.values()),
                   process_wall_s=execution['process_wall_s'], **measured_cost)
    return dict(label=declared['label'], mode=mode, issues=issues, metrics=metrics, requests=req,
                request_count=len(req), generated_tokens=sum(len(r['output_token_ids']) for r in req.values()),
                warmup_wall_s=warmup['observation_end_s'], all_cost=all_cost,
                by_phase={p: amounts(rs) for p, rs in phases.items()}, gpu_boundary_checks=len(checks),
                memory=read(path / 'cuda_memory.json'), actual_resources=resource,
                scheduled_positions=sum(s['total_scheduled_tokens'] for s in raw['scheduler_steps']),
                preemption_summary=raw['preemption_summary'],
                output_bearing_request_absences=absences,
                absence_scope='Previously returned at least one token, not completed at scheduler start, but no decode token scheduled. Does not assert native readiness or subtract preemption.',
                output_sha256=digest({rid: r['output_token_ids'] for rid, r in req.items()}), **{
                    k: rebuilt[k] for k in ('route_sha256', 'trace_sha256', 'final_cache_sha256')})


def comparison(a, b):
    fields = a['metrics'].keys() & b['metrics'].keys()
    return dict(a=a['label'], b=b['label'], delta_b_minus_a={k: b['metrics'][k] - a['metrics'][k] for k in fields},
                delta_pct={k: 100 * (b['metrics'][k] / a['metrics'][k] - 1) if a['metrics'][k] else None for k in fields},
                equal_outputs=sum(a['requests'][rid]['output_token_ids'] == b['requests'][rid]['output_token_ids'] for rid in a['requests']),
                equal_route_hash=a['route_sha256'] == b['route_sha256'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir', type=Path, default=HERE)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    root = args.input_dir.resolve()
    plans = read(root / 'run_cells.json')
    execution_path = root / 'results/execution.json'
    execution = read(execution_path) if execution_path.exists() else {'status': 'UNRUN', 'cells': []}
    entries = {x['label']: x for x in execution['cells']}
    rows, issues, missing = [], [], []
    for plan in plans:
        if plan['label'] not in entries:
            missing.append(plan['label'])
            continue
        try:
            row = cell(root, plan, entries[plan['label']])
            rows.append(row)
            issues.extend(plan['label'] + ': ' + s for s in row['issues'])
        except (OSError, ValueError, TypeError, KeyError, IndexError, AssertionError) as exc:
            issues.append(plan['label'] + ': ' + repr(exc))
    by_mode = defaultdict(list)
    for row in rows:
        by_mode[row['mode']].append(row)
    pairs = [comparison(baseline, target) for mode in ('uniform', 'selected', 'fullstage')
             for baseline, target in zip(by_mode[mode], by_mode['oneshot'])]
    drift = {mode: comparison(rs[0], rs[1]) for mode, rs in by_mode.items() if len(rs) == 2}
    done = execution['status'] == 'COMPLETE' and len(rows) == len(plans) == 8 and not missing and not issues
    report = dict(status='DESCRIPTIVE_FINITE_COHORT' if done else 'UNRUN' if not entries else 'INCOMPLETE_OR_INVALID',
                  issues=issues, missing_cells=missing, retained_execution=execution, cells=rows,
                  comparisons=pairs, same_arm_observed_drift=drift,
                  scope='Two engines per arm, one shared 16-document finite arrival sequence; no population noise floor, CI, significance, SLO, quality, continuous isolation or steady-state claim. Capture excludes startup/warmup/finalization; process time includes them. Nested timing spans must not be summed.')
    with args.out.open('x') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps(dict(status=report['status'], cells=len(rows), issues=len(issues), missing=len(missing))))
    return 0 if done or not entries else 1


if __name__ == '__main__':
    raise SystemExit(main())
