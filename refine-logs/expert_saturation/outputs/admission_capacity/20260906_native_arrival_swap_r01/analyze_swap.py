#!/usr/bin/env python3
"""Summarize the retained identity-swap control; no kernel-causal attribution."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[5]
METRICS_DIR = ROOT / 'refine-logs/expert_saturation/experiments/admission_capacity'
sys.path.insert(0, str(METRICS_DIR))
from metrics import summarize_episode_requests


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    bundle = Path(__file__).resolve().parent
    with tarfile.open(bundle / 'execution.tar.gz') as t:
        if t.extractfile('metrics.py').read() != (METRICS_DIR / 'metrics.py').read_bytes():
            raise ValueError('analysis metric source differs from executed source')
        source_hashes = {n: hashlib.sha256(t.extractfile(n).read()).hexdigest()
                         for n in ('run_native_capacity.py', 'native_capture.py', 'metrics.py')}
    cells = []
    engine_args = []
    for group in ('a0_original', 'b0_swapped', 'b1_swapped', 'a1_original'):
        directory = args.results_dir / group
        config = read(directory / 'config.json')
        workload = read(directory / 'workload.json')
        environment = read(directory / 'environment.json')
        if environment['source_sha256'] != source_hashes:
            raise ValueError('execution source mismatch')
        engine_args.append(read(directory / 'engine_args.json'))
        if config['engine_max_num_seqs'] != 8 or config['cap'] != 8:
            raise ValueError('wrong engine/admission cap')
        positions = {r['request_id']: i for i, r in enumerate(workload['source_requests'])}
        for index, plan in enumerate(config['plans']):
            path = directory / f'cell-{index:03d}.json'
            cell = dict(group=group, cell=index, plan=plan, status='MISSING')
            cells.append(cell)
            if not path.exists():
                continue
            raw = read(path)
            metrics = summarize_episode_requests(raw['requests'], observation_end_s=raw['observation_end_s'],
                ttft_slo_s=config['ttft_slo_s'], tpot_slo_s=config['tpot_slo_s'])
            saved = read(directory / f'metrics-{index:03d}.json')
            if raw['plan'] != plan or any(saved[k] != v for k, v in metrics.items()):
                raise ValueError('plan or saved metrics mismatch')
            details = []
            for request in raw['requests']:
                times = request['token_times_s']
                if len(times) < 2:
                    details.append(dict(request_id=request['request_id'], status=request['status']))
                    continue
                intervals = [b-a for a, b in zip(times, times[1:])]
                peak = max(range(len(intervals)), key=lambda i: intervals[i])
                mean_tpot = sum(intervals) / len(intervals)
                steps = [s for s in raw['scheduler_steps'] if times[peak] <= s['start_s'] <= times[peak+1]]
                details.append(dict(request_id=request['request_id'], slot=positions[request['request_id']],
                    arrival_s=request['arrival_s'], status=request['status'], ttft_s=times[0]-request['arrival_s'],
                    mean_tpot_s=mean_tpot, tpot_failed=mean_tpot > config['tpot_slo_s'],
                    first_itl_s=intervals[0], max_itl_s=intervals[peak], max_itl_from_token=peak+1,
                    remaining_itl_mean_s=statistics.mean(intervals[1:]),
                    peak_interval_scheduler_steps=[dict(step=s['step'], actual_active=s['actual_active'],
                        waiting=s['waiting_requests'], prefill_tokens=sum(r['prefill_tokens'] for r in s['scheduled']),
                        decode_tokens=sum(r['decode_tokens'] for r in s['scheduled'])) for s in steps]))
            cell.update(status=raw['status'], metrics=metrics, requests=details,
                boundary_check=read(directory / f'checks-{index:03d}.json')['status'],
                host_chunk_diagnostics=raw['host_chunk_diagnostics'],
                reference_slo_metrics=saved['reference_slo_metrics'])
    if any(a != engine_args[0] for a in engine_args):
        raise ValueError('engine arguments differ across arms')
    complete = len(cells) == 8 and all(c['status'] == 'COMPLETE' and c['boundary_check'] == 'PASS'
        and c['host_chunk_diagnostics']['token_level_itl_resolved'] for c in cells)
    result = dict(status='MEASUREMENT_ONLY' if complete else 'INCOMPLETE_OR_UNQUALIFIED',
        complete_expected_episodes=complete, cells=cells,
        limit='host interval and scheduling alignment; no isolated GPU kernel cause or removable-time Oracle')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / 'analysis.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
