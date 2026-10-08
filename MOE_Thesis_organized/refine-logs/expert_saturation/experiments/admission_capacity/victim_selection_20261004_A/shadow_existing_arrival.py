"""Print arrival opportunities in both real tail runs; no replay or benefit claim."""
import argparse
from collections import Counter
import json
from pathlib import Path

from analyze import read
from analyze_pro import shadow_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, required=True)
    args = parser.parse_args()
    for cell in ('cell-00-tail', 'cell-03-tail'):
        archive = args.session / cell / 'archive'
        def load(name):
            path = archive / name
            return read(path if path.exists() else path.with_suffix(path.suffix + '.gz'))
        raw, store = load('raw.json'), load('selective-store.json')
        requests = {r['request_id']: r for r in raw['requests']}
        events = store['funding_victim_decisions']
        for event in events:
            for row in event['candidates']:
                source = raw['internal_to_source'][row['request']]
                # The actual runner passes exactly this value to engine.add_request.
                row['original_arrival_time'] = (raw['measurement_origin_unix_s'] +
                                                requests[source]['arrival_s'])
        summary = shadow_summary(events)
        changed = [r['max_original_arrival'] for r in summary['raw_shadows']
                   if r['max_original_arrival'].get('changed')]
        print(json.dumps(dict(cell=cell,
            semantics='Observed legal candidates only; original arrival reconstructed from exact runner clock. No executed arrival policy, replay, or causal benefit.',
            arrival=summary['rules']['max_original_arrival'],
            changed_capacity_signs=dict(Counter('fewer' if r['capacity_delta_blocks'] < 0
                else 'more' if r['capacity_delta_blocks'] > 0 else 'equal' for r in changed))),
            ensure_ascii=False))


if __name__ == '__main__':
    main()
