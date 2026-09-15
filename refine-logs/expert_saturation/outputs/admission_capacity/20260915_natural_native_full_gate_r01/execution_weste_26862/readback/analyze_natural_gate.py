"""One diagnostic cell: distinguish resource and execution-qualification boundaries."""
import argparse
from collections import Counter
import json
from pathlib import Path

from saved_kv_analysis_base import cell, eligibility


def analyze(folder):
    result = cell(folder, True)
    result['performance_comparison'] = False
    result['claim_ceiling'] = 'NATIVE_INPROCESS_DIAGNOSTIC / MEASUREMENT_ONLY'
    if result['status'] != 'COMPLETE' or result['errors']:
        result['qualification'] = 'UNRUN' if result['status'] == 'UNRUN' else 'INCOMPLETE_OR_INVALID'
        return result
    read = lambda name: json.loads((folder/(name+'.json')).read_text())
    raw, selective, offload = (read(n) for n in ('raw', 'selective-store', 'offload-events'))
    gates = {g['step']: g for g in selective['gate_observations']}
    decisions = {d['step']: d for d in selective['selector_decisions']}
    completed = {str(j['job_id']): e['time_s'] for e in offload['completed_jobs'] for j in e['jobs']}
    funding = []
    for snapshot in selective['eligibility_snapshots']:
        tracked = snapshot['tracker']['absent_since']
        waiting = [rid for rid in snapshot['waiting_ids'] if rid in tracked
            and snapshot['requests'][rid]['status'] == 'PREEMPTED']
        if not waiting:
            continue
        # Same longest-observed-absence target ordering; no alternative-target oracle.
        target = max(waiting, key=lambda rid: (snapshot['step']-tracked[rid], rid))
        row = eligibility(snapshot, target, raw['measurement_origin_perf_counter_s'],
            offload['dispatch'], completed)
        if row is None:
            continue
        gate = gates.get(snapshot['step'], {}).get('gate', 'UNKNOWN')
        decision = decisions.get(snapshot['step'])
        if gate == 'selector_eligible' and decision is not None:
            gate = 'selector:' + decision['reason']
        funding.append(dict(row, target=target, gate=gate,
            resource_case='direct_history_funding' if row['direct'] else
                ('fixed_most_victim_funding' if row['fixed_most_funded'] else 'no_full_history_funding')))
    dispatch = Counter('store' if d['is_store'] else 'load' for d in offload['dispatch'] if d['accepted'])
    completed_kinds = Counter('store' if j['is_store'] else 'load'
        for e in offload['completed_jobs'] for j in e['jobs'])
    transfer_bytes = {kind: sum(t[kind]['bytes'] for t in offload['transfers']) for kind in ('store', 'load')}
    diag = result['diagnostic']
    if selective['applied_rotations'] and completed_kinds['store'] and completed_kinds['load']:
        qualification = 'NATIVE_SAVED_RECOVERY_QUALIFIED'
    elif selective['applied_rotations']:
        qualification = 'ROTATION_WITHOUT_COMPLETED_SAVE_LOAD_PAIR'
    elif any(r['resource_case'] != 'no_full_history_funding' for r in funding):
        qualification = 'FUNDED_OBSERVATIONS_WITHOUT_COMMITTED_ROTATION'
    elif diag['counts']['preemptions']:
        qualification = 'NATIVE_RECOVERY_OBSERVED_NO_FUNDED_ROTATION'
    else:
        qualification = 'NO_RECOVERY_ACTION_SPACE_OBSERVED'
    result.update(qualification=qualification, applied_rotations=selective['applied_rotations'],
        accepted_jobs=dict(dispatch), completed_jobs=dict(completed_kinds), completed_transfer_bytes=transfer_bytes,
        gate_counts=dict(Counter(g['gate'] for g in gates.values())),
        legacy_closed_activation_boundaries=sum(g['legacy_closed_activation_condition_met'] for g in gates.values()),
        funding_observations=funding,
        funding_by_gate=dict(Counter(f"{r['gate']} | {r['resource_case']}" for r in funding)),
        prepare_commit_events=[e for e in selective['events'] if e['event'] in
            ('prepare', 'prepare_rejected', 'commit_check', 'target_terminal', 'target_new_output')],
        timing=read('timing'), low_pressure_reference=dict(
            path='refine-logs/expert_saturation/outputs/admission_capacity/20260914_streaming_recovery_r01/execution/readback',
            contract='Original0.5s/no-host-offload/4096usable-block natural-EOS allowed diagnostic; reused boundary, not a matched save-on repeat.'),
        interpretation=[
            'One predeclared controlled arrival point, not production traffic or a pressure sweep.',
            'No action is retained and ends this version; no hidden pressure increase follows.',
            'Full-history funding is an observed necessary resource condition, not immediate dispatch or sustainable service.',
            'Mixed prefill, native recovery, skipped queues and current selector guards are reported separately.',
            'The legacy closed activation predicate is only descriptive; the open adapter does not wait for32pure-decode requests.',
            'EOS is enabled, but cap1024 remains a configured limit; actual stop reasons/lengths are retained.',
            'L/S/F, internal recovery work and complete service descriptors remain distinct; no timing gain is inferred from this cell.',
            'All source, failure/raw retention, initialization, warmup, host/GPU allocations and overlapping memory views remain available.'])
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with args.output.open('x') as f:
        json.dump(analyze(args.results/'diagnostic-current'), f, indent=2, ensure_ascii=False)
        f.write('\n')
