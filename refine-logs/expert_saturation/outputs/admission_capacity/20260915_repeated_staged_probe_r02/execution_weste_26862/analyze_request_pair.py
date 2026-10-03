"""Retain all matched request changes for the actual two-arm qualification."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

p = argparse.ArgumentParser()
p.add_argument('results', type=Path)
p.add_argument('output', type=Path)
args = p.parse_args()
arms, configs, source_hashes = {}, {}, {}
for arm in ('off', 'on'):
    folder = args.results / ('save-' + arm)
    assert json.loads((folder / 'status.json').read_text())['status'] == 'COMPLETE'
    source = folder / 'raw.json'
    raw = json.loads(source.read_text())
    assert raw['status'] == 'COMPLETE'
    rows = {r['request_id']: r for r in raw['requests']}
    assert len(rows) == len(raw['requests']) == 32
    for r in rows.values():
        assert r['status'] == 'completed' and len(r['output_token_ids']) == len(r['token_times_s']) == 1024
        assert all(b >= a for a, b in zip(r['token_times_s'], r['token_times_s'][1:]))
    arms[arm] = rows
    configs[arm] = json.loads((folder / 'config.json').read_text())
    source_hashes[arm] = hashlib.sha256(source.read_bytes()).hexdigest()
assert set(arms['off']) == set(arms['on'])
assert {k: v for k, v in configs['off'].items() if k != 'selective_save'} == {
    k: v for k, v in configs['on'].items() if k != 'selective_save'}


def measures(r):
    t = r['token_times_s']
    return dict(completion_s=r['completion_s'] - r['arrival_s'],
                ttft_s=t[0] - r['arrival_s'], tpot_s=(t[-1] - t[0]) / (len(t) - 1),
                max_itl_s=max(b - a for a, b in zip(t, t[1:])))


changes = []
for rid in sorted(arms['off']):
    a, b = arms['off'][rid], arms['on'][rid]
    assert all(a[k] == b[k] for k in ('document_id', 'arrival_s', 'prompt_tokens', 'prompt_token_ids_sha256'))
    left, right = measures(a), measures(b)
    changes.append(dict(request_id=rid, off=left, on=right,
                        delta={k: right[k] - left[k] for k in left},
                        output_equal=a['output_token_ids'] == b['output_token_ids'],
                        first_output_difference=next((i for i, (u, v) in enumerate(
                            zip(a['output_token_ids'], b['output_token_ids'])) if u != v), None)))
summary = {}
for key in changes[0]['delta']:
    values = [x['delta'][key] for x in changes]
    summary[key] = dict(off_mean=statistics.mean(x['off'][key] for x in changes),
                        on_mean=statistics.mean(x['on'][key] for x in changes),
                        improved=sum(v < -1e-12 for v in values),
                        harmed=sum(v > 1e-12 for v in values),
                        tied=sum(abs(v) <= 1e-12 for v in values))
result = dict(status='DESCRIPTIVE_COMPLETE_REQUEST_PAIR', raw_sha256=source_hashes,
              request_count=32, summary=summary,
              equal_output_requests=sum(x['output_equal'] for x in changes),
              per_request_changes=changes,
              scope='One fixed-order pair only. No quality, SLO, significance, stable effect, or strongest-immediate-baseline claim.')
with args.output.open('x') as f:
    f.write(json.dumps(result, indent=2) + '\n')
print(json.dumps({k: v for k, v in result.items() if k != 'per_request_changes'}, indent=2))
