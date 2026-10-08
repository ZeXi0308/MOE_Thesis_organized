#!/usr/bin/env python3
"""Normal-capacity observations; policy contrasts are confined to one actual cap."""
import argparse, collections, json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze import analyze, comparisons, read, summary


def optional(path):
    return read(path) if path.exists() else dict(status='UNAVAILABLE', path=str(path))


def analyze_group(session):
    cells = []; groups = collections.defaultdict(list)
    for directory in sorted(session.glob('cell-*')):
        if not directory.is_dir():
            continue
        match = re.fullmatch(r'cell-\d+-cap(\d+)-(native|age|flush_first)', directory.name)
        if match is None:
            raise ValueError(f'Unrecognized cell name: {directory.name}')
        cap, mode = int(match[1]), match[2]; output = directory/'output'; cell = analyze(output)
        artifacts = {name: optional(output/(name+'.json')) for name in (
            'config', 'resolved-scheduler-config', 'safe-cap-qualification', 'memory-after-init',
            'memory-before', 'memory-after', 'host-before', 'host-request-end', 'host-after')}
        actual_cap = artifacts['resolved-scheduler-config'].get('max_num_running_reqs')
        cell.update(declared_cap=cap, actual_cap=actual_cap, mode=mode, resource_observations=artifacts)
        rawpath = next((output/n for n in ('raw.json', 'raw.json.gz') if (output/n).exists()), None)
        raw = read(rawpath) if rawpath else {}
        recoveries = cell.get('recovery_events', []); jobs = cell.get('jobs', [])
        cell['mechanism'] = dict(actual_preemption_count=raw.get('actual_preemption_count'),
            preemption_attempt_count=raw.get('preemption_attempt_count'),
            observed_recovery_episodes=len(recoveries) if rawpath else None,
            recovery_to_next_output_s=summary([r['demand_to_next_output_s'] for r in recoveries]),
            recovery_without_next_output=sum(r['next_output_s'] is None for r in recoveries),
            recovery_wait_lower_bound_s=summary([r['censored_wait_lower_bound_s'] for r in recoveries]),
            host_wait_s=summary([w['elapsed_host_s'] for w in cell.get('waits', [])]),
            lifecycle_stage_counts={stage: sum(j[stage+'_s'] is not None for j in jobs)
                for stage in ('ready', 'submit_begin', 'submit_end', 'job_completed', 'ack_retired')},
            legal_flush_reorder_opportunities=sum(e.get('legal_flush_order', e['before']) != e['before']
                and not e.get('flush_fallback') for e in cell.get('reorders', [])),
            observation_available=True) if rawpath and (output/'recovery-order.json').exists() else dict(
                status='UNAVAILABLE', actual_preemption_count=raw.get('actual_preemption_count'),
                reason='Raw or recovery-order observations unavailable; missing activity is not zero')
        cells.append(cell); groups[cap].append(cell)
    contrasts = []
    for cap, members in groups.items():
        by_path = {c['directory']: c for c in members}
        for contrast in comparisons(members):
            contrast['declared_cap'] = cap
            pair = [by_path.get(contrast[key]) for key in ('candidate', 'native')]
            if any(c is None or c['actual_cap'] != cap for c in pair):
                contrast = dict(candidate=contrast['candidate'], native=contrast['native'], declared_cap=cap,
                    status='UNAVAILABLE', reason='Same actual scheduler cap is unverified or mismatched')
            contrasts.append(contrast)
    return dict(cells=cells, comparisons=contrasts, semantics=(
        'All baseline cells, zero-action cells, failures and unfinished requests retained. Different-cap '
        'native cells describe capacity regimes and are not policy controls. Only verified equal-cap '
        'cells are compared. Recovery/lifecycle/host waits use parent host timing semantics; no stage '
        'sums imply request benefit. GPU peak counters cover their recorded reset interval; host RSS '
        'and shared cgroup peaks retain prior history. Missing observation is not zero activity.'))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--output', type=Path); args = parser.parse_args()
    destination = args.output or args.session/'normal-metrics.json'
    if destination.exists():
        raise FileExistsError(destination)
    result = analyze_group(args.session)
    with destination.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(destination)


if __name__ == '__main__':
    main()
