#!/usr/bin/env python3
"""Describe pre-decision timing imbalance without discarding or adjusting runs."""
import argparse
import bisect
import hashlib
import json
from pathlib import Path
import statistics


def summary(values):
    return dict(n=len(values), mean=statistics.mean(values) if values else None,
                maximum=max(values) if values else None)


def analyze(path):
    data = json.loads(path.read_text())
    cells = []
    for cell in data['cells']:
        directory = Path(cell['directory'])
        raw = json.loads((directory/'raw.json').read_text())
        decisions = cell['recovery_fit_actions'].get('rows', [])
        if len(decisions) != 1:
            cells.append(dict(cell=directory.parent.name, status='NO_UNIQUE_DECISION'))
            continue
        stamp = decisions[0]['decision_s']
        requests = []
        for request in raw['requests']:
            times = request['token_times_s']
            index = bisect.bisect_right(times, stamp)
            gaps = [(times[i]-times[i-1], times[i-1], times[i]) for i in range(1, index)]
            largest = max(gaps, default=None)
            requests.append(dict(request=request['request_id'],
                first_output_s=times[0] if times else None,
                outputs_before_decision=index,
                largest_closed_gap_before_decision=largest,
                admission_lag_s=request['admission_s']-request['arrival_s']
                    if request.get('admission_s') is not None else None,
                engine_add_return_lag_s=request['engine_add_return_s']-request['arrival_s']
                    if request.get('engine_add_return_s') is not None else None))
        closed = [r['largest_closed_gap_before_decision'][0] if r['largest_closed_gap_before_decision'] else 0
                  for r in requests]
        first = [r['first_output_s'] for r in requests if r['first_output_s'] is not None]
        cells.append(dict(cell=directory.parent.name, mode=cell['mode'], decision_s=stamp,
            status='ANALYZED', planned=cell['planned'], raw_requests=len(requests),
            outputs_before_decision=sum(r['outputs_before_decision'] for r in requests),
            latest_first_output_s=max(first) if first else None,
            first_outputs_before_decision=sum(t<=stamp for t in first),
            pre_decision_maxgap=summary(closed),
            requests_already_failing_fixed_gap_slo=sum(t>cell['joint_slo']['maxgap_limit_s'] for t in closed),
            admission_lag=summary([r['admission_lag_s'] for r in requests if r['admission_lag_s'] is not None]),
            engine_add_return_lag=summary([r['engine_add_return_lag_s'] for r in requests if r['engine_add_return_lag_s'] is not None]),
            drain_from_last_external_arrival_s=raw['observation_end_s']-max(r['arrival_s'] for r in raw['requests']),
            per_request=requests))
    return dict(input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), cells=cells,
        semantics='Host/client observations. Closed pre-decision gaps only; a gap crossing the decision is excluded. '
        'No run is dropped or adjusted. Different decision times are not matched causal prefixes. '
        'Pre-decision TTFT/gap differences cannot be caused by the later queue mutation. '
        'Admission and engine-add-return lags include client scheduling and engine-call latency, respectively; '
        'they are not hidden or subtracted from external-arrival request metrics.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = analyze(args.metrics)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps([{k:v for k,v in row.items() if k!='per_request'} for row in result['cells']]))


if __name__ == '__main__':
    main()
