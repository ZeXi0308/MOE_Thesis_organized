"""One native-full diagnostic; reuse natural request/recovery analysis unchanged."""
import argparse
import json
from pathlib import Path

from analyze_natural_gate import analyze as analyze_natural


def analyze(folder):
    result = analyze_natural(folder)
    path = folder/'selective-store.json'
    if not path.exists():
        result['native_full_qualification'] = 'UNRUN'
        return result
    selected = json.loads(path.read_text())
    events = [e for e in selected['events'] if e['event'] == 'native_full_metadata']
    jobs = [dict(j, metadata_step=e['step'], metadata_perf_s=e['host_perf_counter_s'])
        for e in events for j in e['store_jobs']]
    offload_path = folder/'offload-events.json'
    offload = json.loads(offload_path.read_text()) if offload_path.exists() else {}
    completions = {j['job_id']: e['time_s'] for e in offload.get('completed_jobs', [])
        for j in e['jobs'] if j['is_store']}
    dispatches = {d['job_id'] for d in offload.get('dispatch', []) if d['accepted'] and d['is_store']}
    for job in jobs:
        job['accepted_dispatch'] = job['job_id'] in dispatches
        job['completed_perf_s'] = completions.get(job['job_id'])
    outside = [j for j in jobs if j['outside_selected_preparation']]
    decode = [j for j in jobs if j['chunks_containing_decode_positions']]
    beyond = [j for j in jobs if j['beyond_selected_prefix_blocks']]
    result['native_full_scope'] = dict(configured=selected.get('store_scope'),
        native_calc_overridden=selected.get('native_calc_overridden'),
        observed_store_jobs=jobs,
        outside_selected_preparation_jobs=len(outside),
        outside_selected_preparation_completed=sum(j['completed_perf_s'] is not None for j in outside),
        decode_range_jobs=len(decode), decode_range_completed=sum(j['completed_perf_s'] is not None for j in decode),
        beyond_preparation_prefix_jobs=len(beyond),
        finished_request_store_jobs=sum(j['finished_at_metadata'] for j in jobs),
        flush_events=[dict(step=e['step'],metadata_perf_s=e['host_perf_counter_s'],jobs=e['flush_jobs'])
            for e in events if e['flush_jobs']],
        scope='Outside preparation is a contemporaneous label, never a future victim label; completion joins actual native job IDs.')
    for name in ('post-request-drain', 'timing', 'host-request-end', 'host-after'):
        file = folder/(name+'.json')
        result[name] = json.loads(file.read_text()) if file.exists() else None
    valid_scope = selected.get('store_scope') == 'native_full' and selected.get('native_calc_overridden') is False
    if result['status'] != 'COMPLETE' or result['errors']:
        result['native_full_qualification'] = 'INCOMPLETE_OR_INVALID'
    elif not valid_scope:
        result['native_full_qualification'] = 'INVALID_STORE_SCOPE'
    elif any(j['completed_perf_s'] is not None for j in outside) and any(j['completed_perf_s'] is not None for j in decode):
        result['native_full_qualification'] = 'NATIVE_FULL_INCREMENTAL_SCOPE_QUALIFIED'
    else:
        result['native_full_qualification'] = 'COMPLETE_WITHOUT_FULL_SCOPE_WITNESS'
    result['native_full_interpretation'] = [
        'This is one diagnostic, not a timing comparison with the prior selected-victim diagnostic.',
        'Full means unrestricted native incremental eligibility; it does not promise permanent/full host residency.',
        'Current prepare/commit, cancellation, protection, target/victim ranking and token allocation are unchanged.',
        'Recovery qualification remains separate from full-store scope; no action or incompatible pending state is retained.',
        'EOS terminal with F=None has no observed first-new-output event even if completed_recoveries increments.',
        'Metadata time precedes transfer completion; actual accepted/complete jobs and flushes are retained.',
        'Finished-request stores, final drain and actual host state are reported separately from request completion.']
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with args.output.open('x') as f:
        json.dump(analyze(args.results/'diagnostic-native-full'), f, indent=2, ensure_ascii=False)
        f.write('\n')
