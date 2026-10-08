#!/usr/bin/env python3
"""Descriptive mean accounting of completed full-population latency; no replay."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def load(path):
    return json.loads(path.read_text())


def summarize(rows):
    n = len(rows)
    if not n or any(r['outcome'] != 'completed' for r in rows):
        raise ValueError('This accounting requires the entire population completed; do not drop rows.')
    ttft = [r['ttft_s'] for r in rows]
    flow = [r['flow_s'] for r in rows]
    after = [r['completion_s'] - r['first_token_s'] for r in rows]
    residuals = [f - t - g for f, t, g in zip(flow, ttft, after)]
    return dict(requests=n, mean_ttft_s=statistics.mean(ttft),
        mean_flow_s=statistics.mean(flow), mean_after_first_token_s=statistics.mean(after),
        maximum_absolute_identity_residual_s=max(map(abs, residuals)),
        total_output_tokens=sum(r['output_tokens'] for r in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('analysis', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Refusing to overwrite output.')
    analysis = load(args.analysis)
    expected = ['probe-00-mccapped', 'probe-01-mcphase', 'probe-02-mcphase', 'probe-03-mccapped']
    cells = analysis['cells']
    if [Path(c['cell']).name for c in cells] != expected:
        raise ValueError('Expected the frozen v20 strict/phase/phase/strict order.')
    rows = [load(Path(c['per_request_json'])) for c in cells]
    indexed = [{r['request_id']: r for r in arm} for arm in rows]
    arms = [dict(cell=name, **summarize(arm)) for name, arm in zip(expected, rows)]
    pairs = []
    for candidate, baseline in [(1, 0), (2, 3)]:
        c, b = indexed[candidate], indexed[baseline]
        if c.keys() != b.keys() or len(c) != 384:
            raise ValueError('All 384 matching requests must be retained.')
        delta = {k: arms[candidate][k] - arms[baseline][k] for k in
                 ['mean_ttft_s', 'mean_flow_s', 'mean_after_first_token_s', 'total_output_tokens']}
        # Known arrival-order prefix177 is the already-developed fixed-cap reference.
        # This is a descriptive cohort, never a counterfactual or an online signal.
        ids = sorted(b, key=lambda rid: (b[rid]['arrival_s'], rid))
        groups = []
        for label, subset in [('arrival_prefix177', ids[:177]), ('arrival_suffix207', ids[177:])]:
            sc, sb = summarize([c[rid] for rid in subset]), summarize([b[rid] for rid in subset])
            groups.append(dict(label=label, candidate=sc, baseline=sb,
                deltas={k: sc[k]-sb[k] for k in delta}))
        pairs.append(dict(candidate=expected[candidate], baseline=expected[baseline],
            deltas=delta, descriptive_arrival_cohorts=groups))
    result = dict(source_analysis=str(args.analysis.resolve()),
        source_sha256=hashlib.sha256(args.analysis.read_bytes()).hexdigest(),
        independent_unit='run', new_gpu_runs=0, new_online_actions=0,
        arms=arms, pairs=pairs,
        semantics='Exact non-overlapping host-clock accounting: mean flow equals mean TTFT '
            'plus mean completion-minus-first-token interval. The latter includes all stalls, '
            'scheduling and control after first token, not pure decode GPU time. Quantiles are '
            'not added. Prefix177/suffix207 are descriptive arrival-order cohorts using the '
            'previously developed fixed-cap reference, not isolated causal treatment groups. '
            'Changed output lengths/content and run variation remain confounders. No time '
            'is subtracted from service, no counterfactual trajectory or policy is constructed, '
            'and the identities neither prove progress-promise failure nor a new method.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    for pair in pairs:
        print(pair['candidate'], pair['deltas'])
    print(args.output.resolve())


if __name__ == '__main__':
    main()
