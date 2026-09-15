"""Reuse qualified funding cells; no alternative state or selected SLO threshold."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

p = argparse.ArgumentParser()
p.add_argument('--qualified', type=Path, required=True)
p.add_argument('--library', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
sys.path.insert(0, str(a.library))
from analyze_gap_service_tradeoff import baseline_envelope, compare

data = json.loads(a.qualified.read_text())
assert data['status'] == 'MEASUREMENT_ONLY' and len(data['cells']) == 6
assert len({c['gpu_uuid'] for c in data['cells']}) == 1
cells = []
for c in data['cells']:
    assert c['eligible'] and c['status'] == 'COMPLETE'
    assert len(c['per_request']) == 32
    rows = [dict(r, max_gap_s=r['max_itl_s']) for r in c['per_request']]
    cells.append(dict(label=c['label'], wall_s=c['wall_s'], requests=rows,
                      raw_path=c['raw_path'], raw_sha256=c['raw_sha256']))
by = {c['label']: c for c in cells}
envelopes, pairs = [], []
for block in (0, 1):
    most, least, filtered = [by[f'funding-block{block}-{role}'] for role in
                            ('most_output', 'least_progress', 'least_feasible')]
    envelopes.append(baseline_envelope([most, least], filtered))
    pairs.extend([compare(most, filtered), compare(least, filtered)])
result = dict(status='DESCRIPTIVE_FULL_THRESHOLD_CURVES', cells=cells,
    envelopes=envelopes, comparisons=pairs,
    qualified_sha256=hashlib.sha256(a.qualified.read_bytes()).hexdigest(),
    definition='Q(g)=completed requests with own maximum engine-return gap <= g / entire measured episode wall',
    boundary='All thresholds retained. No TTFT or TPOT constraint, client timing, future Oracle, within-episode switching or independent-workload claim; new GPU cells compared only internally.')
with a.output.open('x') as f:
    json.dump(result, f, indent=2, allow_nan=False)
    f.write('\n')
for e in envelopes:
    nonzero = [r for r in e['intervals'] if any(v > 0 for v in r['rates'].values())]
    positive = [r for r in nonzero if r['relation'] == 'higher']
    print(json.dumps(dict(action=e['action'], positive_intervals=len(positive),
                          nonzero_intervals=len(nonzero), positive=positive)))
