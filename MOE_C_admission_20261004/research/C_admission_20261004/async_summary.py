#!/usr/bin/env python3
"""Describe real native async opportunities, without inventing MC decisions."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def summarize(cell):
    path = cell/'admission.json'
    report = json.loads(path.read_text())
    assert report['mode'] == 'passive_native_async' and not report['policy_intervention']
    rows, snapshots = report['decisions'], report['snapshots']
    successful = [r for r in rows if r['native_allocation_succeeded']]

    def events(selected):
        return dict(events=len(selected), unique_new_requests=len({r['request_id'] for r in selected}),
                    actual_native_allocations=sum(r['native_allocation_succeeded'] for r in selected))

    categories = sorted({k for r in rows for k in r['old_progress_counts']})
    selected = {
        'old_nonrunning': [r for r in rows if r['nonrunning']],
        'mixed_running_nonrunning': [r for r in rows if r['mixed_running_nonrunning']],
        'normal_placeholders_present': [r for r in rows if r['placeholders']],
        'deferred_free_present': [r for r in rows if r['deferred_free_entries']],
        'async_discard_present': [r for r in rows if r['async_discard']],
        **{k: [r for r in rows if r['old_progress_counts'].get(k, 0)] for k in categories}}
    sampled = snapshots+rows
    return dict(evidence='SINGLE_NATIVE_ASYNC_DEPLOYMENT_OBSERVATION_NOT_MC_OR_POLICY_COMPARISON',
        cell=str(cell.resolve()), admission_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        new_policy_requested_actions=0, new_policy_executed_actions=0,
        decision_events=len(rows), unique_new_requests=len({r['request_id'] for r in rows}),
        actual_native_allocations=len(successful), native_capacity_refusals=len(rows)-len(successful),
        first_prefill_starts=len(report['starts']), conditions={k: events(v) for k,v in selected.items()},
        observed_state_extrema={k: dict(min=min(r[k] for r in sampled), max=max(r[k] for r in sampled))
            for k in ('active','running','free_blocks','nonrunning','placeholders','deferred_free_blocks')}
            if sampled else {},
        snapshot_count=len(snapshots), nonrunning_snapshot_count=sum(bool(r['nonrunning']) for r in snapshots),
        old_status_event_occurrences=dict(Counter(k for r in rows for k,v in r['old_status_counts'].items() if v)),
        observer_wall_s=report['controller_wall_s'], observer_cpu_s=report['controller_cpu_s'],
        interpretation='Native allocation success is an actual action, not an MC counterfactual. '
            'Category counts overlap and repeated events are not independent runs. Placeholder or '
            'terminal-inflight presence alone does not mean failed progress. Non-running coexistence '
            'can justify later investigation but does not establish loss under strong simple/MC baselines. '
            'Extrema cover decision-time states and100ms snapshots, not continuous occupancy.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cell', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Refusing to overwrite existing analysis')
    args.output.write_text(json.dumps(summarize(args.cell), indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
