"""Reconstruct old recovery waits from retained native preempt/readmission logs."""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path


RELATIVE_ROOT = Path('refine-logs/expert_saturation/experiments/admission_capacity/'
                     '20260929_commit_recheck/recovery_quantum_20261002')
SESSIONS = ('strong-session-r01', 'lease-session-r02')


def quantile(values, fraction):
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * fraction
    index = int(position)
    return values[index] + (values[min(index + 1, len(values) - 1)] - values[index]) * (position - index)


def describe_cell(repository, archive):
    paths = [archive / 'raw.json.gz', archive / 'selective-store.json']
    with gzip.open(paths[0], 'rt') as stream:
        raw = json.load(stream)
    store = json.loads(paths[1].read_text())
    if raw['status'] != 'COMPLETE':
        raise ValueError(f'Incomplete history requires separate interpretation: {archive}')
    origin = raw['measurement_origin_perf_counter_s']
    admissions = {}
    for row in store['residency_admissions']:
        key = (row['request'], row['num_preemptions'])
        if key in admissions:
            raise ValueError(f'Duplicate admission epoch: {key}')
        admissions[key] = row['host_perf_counter_s'] - origin
    epochs, intervals = Counter(), []
    preempt_events = raw['preemption_events']
    if any(not row['original_preemption_returned'] for row in preempt_events):
        raise ValueError('Cannot infer epochs across failed preempt calls')
    if any(a['method_returned_s'] > b['method_entered_s']
           for a, b in zip(preempt_events, preempt_events[1:])):
        raise ValueError('Preempt calls are not serial and chronologically ordered')
    for row in preempt_events:
        rid = row['internal_request_id']
        epochs[rid] += 1
        key = (rid, epochs[rid])
        if (rid, 0) not in admissions or key not in admissions:
            raise ValueError(f'Missing initial admission or recovery epoch: {key}')
        start, end = row['method_returned_s'], admissions[key]
        if not 0 <= start <= end <= raw['observation_end_s']:
            raise ValueError(f'Invalid recovery interval: {key}, {start}, {end}')
        if admissions[(rid, epochs[rid] - 1)] > row['method_entered_s']:
            raise ValueError(f'Recovery epochs overlap: {key}')
        intervals.append((start, end))
    recovered_keys = {key for key in admissions if key[1] > 0}
    expected_keys = {(rid, epoch) for rid, count in epochs.items()
                     for epoch in range(1, count + 1)}
    if recovered_keys != expected_keys or len(intervals) != raw['actual_preemption_count']:
        raise ValueError('Admission/preemption epoch coverage differs')
    changes = defaultdict(int)
    for start, end in intervals:
        changes[start] += 1
        changes[end] -= 1
    count = peak = 0
    nonzero = multiple = previous = 0.
    for timestamp, delta in sorted(changes.items()):
        if count:
            nonzero += timestamp - previous
        if count >= 2:
            multiple += timestamp - previous
        count += delta
        if count < 0:
            raise ValueError('Negative reconstructed backlog')
        peak = max(peak, count)
        previous = timestamp
    if count:
        raise ValueError('Unmatched recovery remains')
    durations = [end - start for start, end in intervals]
    free = [row['free_blocks']
            for name in ('events', 'victim_decisions', 'lease_execution', 'lease_peer_admissions')
            for row in store.get(name, []) if 'free_blocks' in row]
    return dict(
        cell=str(archive.parent.relative_to(repository / RELATIVE_ROOT)),
        sources=[dict(path=str(path.relative_to(repository)),
                      sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in paths],
        run_status=raw['status'], requests=len(raw['requests']),
        completed=sum(row['status'] == 'completed' for row in raw['requests']),
        preemption_count=raw['actual_preemption_count'], matched_epochs=len(intervals),
        unmatched_epochs=0, max_backlog=peak, backlog_nonzero_s=nonzero,
        backlog_at_least_two_s=multiple,
        wait_to_running_readmission_s=dict(p50=quantile(durations, .5),
                                           p95=quantile(durations, .95),
                                           maximum=max(durations, default=None)),
        last_running_readmission_s=max((end for _, end in intervals), default=None),
        observation_end_s=raw['observation_end_s'], backlog_at_end=count,
        sparse_free_block_samples=dict(count=len(free), zero_count=free.count(0),
                                       minimum=min(free, default=None), maximum=max(free, default=None)),
        scheduler_diagnostics=raw.get('scheduler_diagnostics'),
        memory_diagnostics=raw.get('memory_diagnostics'),
        raw_diagnostics_scope=raw.get('diagnostics_scope'),
        eligibility_snapshot_count=len(store.get('eligibility_snapshots', [])),
        selector_decision_count=len(store.get('selector_decisions', [])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', required=True, type=Path,
                        help='Absolute path to the original MOE_Thesis_organized checkout')
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).with_name('history_recovery_summary.json'))
    args = parser.parse_args()
    if not args.repository.is_absolute():
        parser.error('--repository must be absolute')
    repository = args.repository.resolve()
    cells = []
    for session in SESSIONS:
        archives = sorted((repository / RELATIVE_ROOT / session).glob('cell-*/archive'))
        if len(archives) != 3:
            raise ValueError(f'Expected precisely three cells: {session}')
        cells.extend(describe_cell(repository, archive) for archive in archives)
    result = dict(
        status='MEASUREMENT_ONLY', repository=str(repository),
        scope=list(SESSIONS),
        method=dict(
            join='Internal request ID plus preemption epoch. Starting from initial admission epoch 0, '
                 'increment epoch on each successful serial preempt call; match exactly one '
                 'residency_admissions record with the same request and num_preemptions.',
            interval_start='raw.preemption_events.method_returned_s',
            interval_end='selective-store.residency_admissions.host_perf_counter_s minus '
                         'raw.measurement_origin_perf_counter_s',
            interpretation='Native preempt-return to actual reinsertion into running; includes '
                           'waiting for remote KV, excludes later recompute/first-output latency.',
            backlog='Count overlapping half-open recovery intervals [start, end); '
                    'report union time and time with at least two requests.',
            quantiles='Linear interpolation between adjacent sorted observations.',
            validation='All six complete runs; initial epoch, unique admission epochs, serial '
                       'preempt events, interval ordering and complete one-to-one coverage checked.'),
        limitations=[
            'Raw traces retain sparse preemptions and request outputs, not full scheduler/KV snapshots.',
            'Actual free KV is recorded only at selected victim/anchor/lease events; sampling '
            'is conditioned on those events and differs between arms.',
            'There is no simultaneous complete running-count/free-KV/recovery snapshot series. '
            'These logs cannot directly establish incremental recovery information conditional '
            'on KV availability and concurrency.',
            'The original arms modify recovery/victim execution. Their differences are not '
            'causal evidence for the new-prefill-only C admission mechanism.',
            'Recovery intervals within a run are dependent observations, not independent repeats.'
        ],
        conclusion='Persistent recovery waits motivate the selected pressure workload; '
                   'admission decision value and service benefit remain untested by this reconstruction.',
        cells=cells)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
    print(f'Wrote {args.output}: {len(cells)} cells; '
          f'{sum(cell["matched_epochs"] for cell in cells)} matched recovery epochs')


if __name__ == '__main__':
    main()
