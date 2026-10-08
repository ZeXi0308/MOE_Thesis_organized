"""Summarize observed whole-process startup costs without adding graph peaks."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNS = ROOT/'evidence/review_20261008'
rows = []
for name in ('compact', 'dense'):
    path = RUNS/f'{name}.json'
    r = json.loads(path.read_text())
    row = dict(plan=name, status=r['status'], source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    if r['status'] == 'STARTUP_COMPLETE_NO_SERVICE_RUN':
        events = {x['event']:x for x in r['stage_events']}
        steady = events['startup_explicit_empty_cache_after']['memory']
        ready = events['startup_explicit_empty_cache_before']['memory']
        initial = events['init_device_after']['memory']
        kv = events['kv_allocate_after']['allocated_kv']['unique_storage_bytes']
        capture = events['capture_model_after']
        estimate = events['profile_cudagraph_memory_after']
        capture_groups = events['live_capture_order']['groups']
        used = steady['total_bytes']-steady['free_bytes']
        initial_used = initial['total_bytes']-initial['free_bytes']
        pools = steady['torch_allocator_pools']
        row.update(
            actual_graphs=sum(len(g['token_buckets']) for g in capture_groups),
            graph_groups={g['mode']:len(g['token_buckets']) for g in capture_groups},
            kv_total_blocks=r['scheduler_kv_blocks_including_reserved_block'],
            kv_usable_blocks=r['scheduler_free_kv_blocks'], kv_storage_bytes=kv,
            graph_estimate_bytes=estimate['native_return_bytes'],
            native_capture_delta_bytes=capture['native_return_bytes'],
            shared_workspace_bytes=steady['shared_workspace']['unique_storage_bytes'],
            torch_graph_pool_reserved_bytes=sum(v['total_bytes'] for k,v in pools.items() if k not in ('[0, 0]', '"unknown"')),
            torch_allocated_bytes=steady['allocated_bytes'], torch_reserved_bytes=steady['reserved_bytes'],
            graph_pool_attribution='Non-default segment pool IDs only; not all driver graph metadata',
            device_total_bytes=steady['total_bytes'], steady_device_used_bytes=used,
            steady_device_nonkv_bytes=used-kv,
            init_device_used_bytes=initial_used,
            steady_used_increase_from_init_bytes=used-initial_used,
            steady_nonkv_increase_from_init_bytes=used-initial_used-kv,
            explicit_empty_cache_released_bytes=steady['free_bytes']-ready['free_bytes'],
            blocks_before_cleanup=events['startup_ready']['num_blocks'],
            blocks_after_cleanup=events['startup_explicit_empty_cache_after']['num_blocks'],
            graph_estimation_wall_s=estimate['elapsed_s'], capture_wall_s=capture['elapsed_s'],
            engine_startup_wall_s=r['engine_startup_wall_s'],
            independent_startup_runs=1, performance_repeats=0,
            timing_scope='Includes startup instrumentation; compact first compile-cache encounter; not steady-state execution latency',
            budget_scope='gpu_memory_utilization=0.9 is native startup policy, not a measured exact 90% all-device hard cap; initial driver/context and runner allocations reported separately',
            steady_nonkv_scope='Observed entire device used minus live unique KV storages; includes model, workspace, allocator, context and driver; NOT graph-exclusive residency',
        )
    rows.append(row)
result = dict(evidence='STARTUP_ONLY_NO_SERVICE_COMPARISON',rows=rows,
              gpu_order='compact completed; dense lock-busy and not run',
              limitations=['No between-plan cost effect or noise can be estimated from one completed endpoint',
                           'Counter values and native capture delta are not mutually additive',
                           'Cache cleanup never resizes already allocated KV',
                           'Current generated-token or service throughput evidence does not exist'])
(ROOT/'evidence/startup_summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
cols=['plan','status','actual_graphs','kv_total_blocks','kv_usable_blocks','graph_estimate_bytes',
      'native_capture_delta_bytes','shared_workspace_bytes','torch_graph_pool_reserved_bytes',
      'steady_device_nonkv_bytes','explicit_empty_cache_released_bytes','engine_startup_wall_s','capture_wall_s']
with (ROOT/'startup_results.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
    w.writerows({k:r.get(k,'NA') for k in cols} for r in rows)
print(json.dumps(result,indent=2,ensure_ascii=False))
