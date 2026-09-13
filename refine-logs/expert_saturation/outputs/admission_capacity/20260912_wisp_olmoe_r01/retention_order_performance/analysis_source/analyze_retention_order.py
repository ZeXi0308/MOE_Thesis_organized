"""Describe all retained order-comparison episodes; no significance/noise bounds."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics


def read(p):
    return json.loads(p.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    rows, structural, checks = [], [], []
    protocol = read(args.input_dir / 'protocol.json')
    execution = read(args.input_dir / 'results/execution.json')
    assert execution['status'] == 'COMPLETE' and len(execution['cells']) == 2
    for engine, cell in enumerate(execution['cells']):
        root = args.input_dir / 'results' / cell['label']
        derived = read(args.input_dir / (cell['label'] + '_analysis.json'))
        assert cell['returncode'] == 0 and derived['phase_sum_matches_global']
        assert read(root / 'environment.json')['sources'] == protocol['source_sha256']
        episodes = read(root / 'episodes.json')
        assert len(episodes) == 8
        for i, episode in enumerate(episodes):
            name = episode['phase'].split('/')[0]
            result = derived['repeats'][name]
            assert result['status'] == 'COMPLETE' and not result['issues']
            assert result['phase_bytes_valid'] and result['extra_bytes'] == 0
            assert result['retention']['invariants_valid'] and result['retention']['raw_row_routes_present']
            requests = list(result['requests'].values())
            assert len(requests) == 3 and all(r['status'] == 'completed' for r in requests)
            old = [r for r in requests if r['arrival_s'] == 0]
            new = [r for r in requests if r['arrival_s'] != 0]
            assert len(old) == 2 and len(new) == 1
            rows.append(dict(engine=engine, repeat=i, block=engine * 2 + i // 4,
                arm=result['variant'].removeprefix('retention_'),
                wall_s=result['whole_wall_s'], old_max_itl_s=max(r['max_itl_s'] for r in old),
                old_completion_s=max(r['completion_latency_s'] for r in old),
                new_ttft_s=new[0]['ttft_s'], new_max_itl_s=new[0]['max_itl_s'],
                new_completion_s=new[0]['completion_latency_s'],
                payload_bytes=result['actual_bytes'], groups=result['retention']['groups'],
                equality_to_first=result['equality_to_first'], resources=result['fixed_resources'],
                stages=result['retention']['stages']))
        for line in (root / 'gpu_checks.jsonl').read_text().splitlines():
            check = json.loads(line)
            assert check['decision'] == 'PASS' and not check['foreign_pids']
            checks.append({k:check[k] for k in ('stage','decision','caller_pid','allowed_pids')})
        totals = defaultdict(Counter)
        for line in (root / 'pager/calls.jsonl').read_text().splitlines():
            r = json.loads(line)
            if not r['measurement']:
                continue
            active, entry = set(r['active_experts']), set(r['entry_resident_experts'])
            protected = set(r['retention']['chosen_protected_experts'])
            a = len(entry & protected)
            bound = max(bool(active), (len(active) - a + 24 - a - 1) // (24 - a))
            assert len(r['groups']) >= bound
            if r['retention']['order'] == 'late' or not protected:
                assert len(r['groups']) == bound
            totals[r['context']['phase']].update(calls=1, groups=len(r['groups']), minimum_groups=bound)
        structural.append(dict(engine=engine, per_phase=totals))
    by_arm = defaultdict(list)
    for row in rows:
        by_arm[row['arm']].append(row)
    assert set(by_arm) == {'none_early','frequency_early','frequency_late','decode_late'}
    assert all(len(v) == 4 for v in by_arm.values())
    comparisons = []
    for block in range(4):
        group = {r['arm']:r for r in rows if r['block'] == block}
        for baseline in ('none_early','frequency_early'):
            for treatment in ('frequency_late','decode_late'):
                a, b = group[baseline], group[treatment]
                comparisons.append(dict(block=block, baseline=baseline, treatment=treatment,
                    delta_pct={k:100 * (b[k] / a[k] - 1) for k in
                        ('wall_s','old_max_itl_s','old_completion_s','new_ttft_s','new_max_itl_s','payload_bytes','groups')}))
    fields = ('wall_s','old_max_itl_s','old_completion_s','new_ttft_s','new_max_itl_s','payload_bytes','groups')
    summary = {arm:{k:dict(values=[r[k] for r in rr], min=min(r[k] for r in rr),
        max=max(r[k] for r in rr), median=statistics.median(r[k] for r in rr)) for k in fields}
        for arm,rr in by_arm.items()}
    out = dict(scope='Same3 documents, four episodes per arm across only2 engines; descriptive implementation comparison. No CI, noise bound, noninferiority, quality or production-tail conclusion.',
        rows=rows, per_arm=summary, comparisons=comparisons, current_call_bounds=structural,
        gpu_boundaries=checks, sources_sha256={n:hashlib.sha256((args.input_dir/'analysis_source'/n).read_bytes()).hexdigest()
            for n in ('analyze_native_phases.py','analyze_native_pager_transfer.py')})
    with args.out.open('x') as f:
        json.dump(out, f, indent=2, allow_nan=False)
        f.write('\n')


if __name__ == '__main__':
    main()
