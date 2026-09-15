"""Domain interpretation of the canonical diagnostic; no new performance comparison."""
from collections import Counter
import json
from pathlib import Path
from statistics import median


root = Path(__file__).resolve().parent
execution = root / 'execution_weste_26862'
folder = execution / 'readback/results/diagnostic-current'
read = lambda p: json.loads(p.read_text())
analysis = read(execution / 'analysis.json')
raw = read(folder / 'raw.json')
selective = read(folder / 'selective-store.json')
origin = raw['measurement_origin_perf_counter_s']
snapshots = {s['step']: s for s in selective['eligibility_snapshots']}
prepares = [dict(e, time_s=snapshots[e['step']]['host_perf_counter_s'] - origin)
    for e in selective['events'] if e['event'] == 'prepare']
segments = analysis['diagnostic']['segments']
eos_id = read(folder / 'resolved-eos.json')['hf_eos_token_id']
early_stops = [dict(request_id=r['request_id'], outputs=len(r['output_token_ids']),
    completion_s=r['completion_s'], native_stop_reason=r['native_stop_reason'],
    final_token_is_eos=r['output_token_ids'][-1] == eos_id)
    for r in raw['requests'] if r['stop_reason'] != 'length']
last_arrival = max(r['arrival_s'] for r in raw['requests'])
cooldown = [s for s in analysis['funding_observations']
    if s['gate'] == 'selector:swap cooldown' and s['absence_steps'] >= 30]
host = read(folder / 'host-after.json')
result = dict(
    status=analysis['qualification'], primary_analysis='execution_weste_26862/analysis.json',
    preparations=prepares, last_arrival_s=last_arrival,
    preparations_before_last_arrival=sum(p['time_s'] < last_arrival for p in prepares),
    early_stops=early_stops,
    all_early_stops_precede_first_prepare=bool(prepares) and all(
        r['completion_s'] < prepares[0]['time_s'] for r in early_stops),
    recovered_request_count=len({s['request_id'] for s in segments}),
    recovered_requests_ending_at_cap=sorted({s['request_id'] for s in segments
        if next(r for r in raw['requests'] if r['request_id'] == s['request_id'])['stop_reason'] == 'length'}),
    first_new_output_observed_segments=sum(s['F_s'] is not None for s in segments),
    recompute_tokens=sum(s['confirmed_recompute_tokens'] for s in segments),
    cumulative_recovery_gap_request_seconds=sum(s['F_s'] - s['L_s'] for s in segments
        if s['F_s'] is not None and s['L_s'] is not None),
    median_L_to_S_call_s=median(s['S_host_engine_call']['time_s'] - s['L_s']
        for s in segments if s['L_s'] is not None and s['S_host_engine_call']),
    median_S_call_to_F_s=median(s['F_s'] - s['S_host_engine_call']['time_s']
        for s in segments if s['F_s'] is not None and s['S_host_engine_call']),
    cooldown_observations_absence_at_least30=dict(Counter(s['resource_case'] for s in cooldown)),
    resource_observation_warning='Necessary full-history funding only; direct funding does not prove the cooldown blocked native dispatch, and victim funding does not prove sustained service.',
    actual_gpu_kv_bytes=read(folder / 'memory-before.json')['kv_storage_bytes'],
    host_allocation_bytes=host['cpu_kv']['unique_storage_bytes'],
    valid_host_blocks=host['manager']['valid_host_kv_entries'],
    valid_host_bytes=host['cpu_kv']['derived_valid_host_kv_bytes'],
    host_pending=host['pending'], post_request_drain=read(folder / 'post-request-drain.json'),
    host_rss_bytes=host['process_rss_bytes'], host_hwm_bytes=host['process_peak_rss_bytes'],
    parent_memory_max_bytes=host['parent_cgroup']['values']['memory.max'],
    independent_process_tree_limit=None,
    claim_ceiling='One native diagnostic, real arrivals and heterogeneous inputs; six EOS occur before recovery interventions. No comparative performance, quality, or MoE-specific claim.')
with (execution / 'domain_summary.json').open('x') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
    f.write('\n')
