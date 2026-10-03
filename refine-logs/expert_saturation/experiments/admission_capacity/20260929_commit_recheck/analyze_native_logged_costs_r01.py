#!/usr/bin/env python3
"""Describe observed printed transfer counters without filling missing fields."""
import argparse
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re


def analyze(session):
    receipt = json.loads((session / 'receipt.json').read_text())
    assert receipt['status'] == 'CELLS_COMPLETE'
    cells = []
    for i, cell in enumerate(receipt['cells']):
        path = session / f"cell-{i:02d}-{cell['arm']}" / 'launch.log'
        lines = path.read_text(errors='replace').splitlines()
        starts = [n for n, line in enumerate(lines) if line == 'PHASE MEASUREMENT_BEGIN']
        ends = [n for n, line in enumerate(lines) if line == 'PHASE MEASUREMENT_END']
        assert len(starts) == len(ends) == 1 and starts[0] < ends[0]
        prints = [(n + 1, line) for n, line in enumerate(lines)
                  if starts[0] < n < ends[0] and 'KV Transfer metrics:' in line]
        assert len(prints) >= 2
        kinds = {}
        for kind in ('store', 'load'):
            observed, omitted = [], []
            totals = {'bytes': 0, 'time': Decimal(0), 'size_count': 0}
            for number, line in prints[1:]:
                values = {key: re.search(r'\bvllm:kv_offload_' + kind + '_' + key +
                          r'=([^,\s]+)', line) for key in totals}
                if not all(values.values()):
                    omitted.append(dict(line=number, printed_fields=[k for k,v in values.items() if v]))
                    continue
                row = {k: Decimal(v.group(1)) if k == 'time' else int(v.group(1))
                       for k,v in values.items()}
                assert all(v >= 0 for v in row.values())
                observed.append(number)
                for key in totals: totals[key] += row[key]
            kinds[kind] = dict(observed_interval_lines=observed,
                omitted_intervals=omitted,
                observed_sum={k: (float(v) if k == 'time' else v) if observed else None
                              for k,v in totals.items()})
        cells.append(dict(arm=cell['arm'], launch_log=str(path),
            launch_log_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            formal_prints=len(prints), excluded_first_line=prints[0][0],
            included_complete_interval_lines=[n for n,_ in prints[1:]], kinds=kinds))
    return dict(status='OBSERVED_PRINTED_COUNTERS_DESCRIBED', cells=cells,
        limitations=[
            'Exclude the first possibly warmup-mixed print; the unprinted tail is unknown.',
            'Sum only intervals printing all three fields for a transfer kind. Missing groups are listed, not imputed as zero.',
            'These are observed counter sums, not complete episode traffic or comparable full-episode transfer totals.',
            'CUDA event time is not request-visible waiting time; do not add it to overlapping service time.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.session)
    with args.output.open('x') as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write('\n')
