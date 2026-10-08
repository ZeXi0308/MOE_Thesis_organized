"""Read-only call-level map-cost diagnosis; no policy replay or GPU-time subtraction.

Fit first two Qwen r04 cells and test the last two without random row splitting.
Online features are available after the current router and before ensure/map.
Actual CUDA envelopes enter explanatory diagnostics only, never the predictors.
"""
import argparse
from collections import defaultdict
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np


def stats(values):
    a = np.asarray(values, dtype=float)
    return {"n": len(a), "sum": float(a.sum()),
            **({"median": float(np.median(a)), "p95": float(np.quantile(a, .95)),
                "mean": float(a.mean())} if len(a) else {})}


def nonnegative_fit(x, y):
    """Exact small-dimensional NNLS by enumerating active coefficient sets."""
    best = (float(y @ y), np.zeros(x.shape[1]))
    for n in range(1, x.shape[1] + 1):
        for active in itertools.combinations(range(x.shape[1]), n):
            c = np.linalg.lstsq(x[:, active], y, rcond=None)[0]
            if np.any(c < -1e-10):
                continue
            beta = np.zeros(x.shape[1])
            beta[list(active)] = np.maximum(c, 0)
            loss = float(np.sum((x @ beta - y) ** 2))
            if loss < best[0]:
                best = loss, beta
    return best[1]


def error(y, p):
    return {"n": len(y), "mae_ms": float(np.mean(np.abs(p-y))),
            "wape_pct": float(100*np.sum(np.abs(p-y))/np.sum(y)),
            "observed_sum_ms": float(np.sum(y)), "predicted_sum_ms": float(np.sum(p))}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input-dir', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    raw = {p.name: json.loads((p/'raw.json').read_text())
           for p in sorted(args.input_dir.glob('repeat_*'))}
    group_rows, layer_rows = [], []
    source = args.input_dir/'pager/calls.jsonl'
    for line in source.open():
        r = json.loads(line)
        phase = r['context']['phase']
        if not phase.endswith('/measurement'):
            continue
        assert r['measurement'] and not r['validation_run'] and r['status'] == 'complete'
        cell = phase.split('/')[0]
        step = r['context']['step_id']
        sched = raw[cell]['scheduler_steps'][step]
        assert sched['step'] == step
        pf = sum(x['prefill_tokens'] for x in sched['scheduled'])
        dc = sum(x['decode_tokens'] for x in sched['scheduled'])
        assert pf+dc == r['rows']
        stage = 'mixed' if pf and dc else ('prefill' if pf else 'decode')
        groups = r['groups']
        route = r['route_to_host_ms']
        ensure = sum(x['host_ensure_ms'] for x in groups)
        mapping = sum(x['host_map_ms'] for x in groups)
        summing = sum(x['host_sum_ms'] for x in groups)
        residual = r['host_apply_ms']-route-ensure-mapping-summing
        assert residual >= -1e-6, (cell, step, residual)
        layer_rows.append(dict(cell=cell, step=step, layer=r['layer_name'], stage=stage,
            rows=r['rows'], groups=len(groups), route_ms=route, ensure_ms=ensure,
            map_ms=mapping, sum_ms=summing, residual_ms=residual,
            host_apply_ms=r['host_apply_ms']))
        for i, g in enumerate(groups):
            assert g['miss'] == len(g['loaded_experts'])
            assert g['stop']-g['start'] == r['rows']
            group_rows.append(dict(cell=cell, repeat=int(cell.split('_')[1]), step=step,
                layer=r['layer_name'], stage=stage, rows=r['rows'], group_index=i,
                group_count=len(groups), required=len(g['required_experts']), misses=g['miss'],
                payload_bytes=g['weight_copy_bytes'], map_payload_bytes=128*4,
                map_ms=g['host_map_ms'], ensure_ms=g['host_ensure_ms'],
                load_envelope_ms=g['load_cuda_span_ms']))
    assert len(layer_rows) == 7584
    by_cell = {}
    for cell in raw:
        layers = [x for x in layer_rows if x['cell'] == cell]
        totals = {k: sum(x[k] for x in layers)
                  for k in ['route_ms', 'ensure_ms', 'map_ms', 'sum_ms', 'residual_ms', 'host_apply_ms']}
        totals['map_share_of_host_apply_pct'] = 100*totals['map_ms']/totals['host_apply_ms']
        totals['groups'] = sum(x['groups'] for x in layers)
        by_cell[cell] = totals
    strata = defaultdict(list)
    for r in group_rows:
        key = (r['cell'], r['stage'], r['group_index'] == 0, r['misses'] == 0)
        strata[key].append(r)
    strata_out = [dict(cell=k[0], stage=k[1], first_group=k[2], no_misses=k[3],
                       map_ms=stats([r['map_ms'] for r in rows]),
                       required=stats([r['required'] for r in rows]),
                       misses=stats([r['misses'] for r in rows]))
                  for k, rows in sorted(strata.items())]
    # Same-size full global map each time; width counts only the Python fills.
    first = [r for r in group_rows if r['group_index'] == 0]
    train = [r for r in first if r['repeat'] < 2]
    test = [r for r in first if r['repeat'] >= 2]
    models = {}
    for name, features in [('map_size_control', ['one', 'required']),
                           ('map_plus_pending_loads', ['one', 'required', 'misses'])]:
        def mat(rows):
            return np.array([[1. if f == 'one' else r[f] for f in features] for r in rows])
        y = np.array([r['map_ms'] for r in train])
        beta = nonnegative_fit(mat(train), y)
        yt = np.array([r['map_ms'] for r in test])
        pt = mat(test) @ beta
        models[name] = dict(features=features, coefficients_ms=beta.tolist(),
            fit_cells=sorted({r['cell'] for r in train}),
            test_cells=sorted({r['cell'] for r in test}), train_n=len(train),
            test=error(yt, pt), by_stage={})
        for stage in sorted({r['stage'] for r in test}):
            mask = np.array([r['stage'] == stage for r in test])
            models[name]['by_stage'][stage] = error(yt[mask], pt[mask])
    correlations = {}
    for cell in raw:
        data = [r for r in first if r['cell'] == cell and r['misses']]
        x = np.array([r['load_envelope_ms'] for r in data])
        y = np.array([r['map_ms'] for r in data])
        correlations[cell] = dict(n=len(data),
            pearson_map_vs_load_envelope=float(np.corrcoef(x, y)[0, 1]),
            scope='Explanatory overlap association, not independent samples or pure DMA time')
    # Deterministic strata in held cells. No identity/step-based winner selection.
    matched = defaultdict(list)
    for r in first:
        if r['repeat'] >= 2:
            matched[(r['stage'], r['rows'], r['required'])].append(r)
    contrasts = []
    for key, rs in sorted(matched.items()):
        hit, miss = [r for r in rs if r['misses'] == 0], [r for r in rs if r['misses'] > 0]
        if not hit or not miss:
            continue
        contrasts.append(dict(stage=key[0], rows=key[1], required=key[2],
            hit_map_ms=stats([r['map_ms'] for r in hit]),
            miss_map_ms=stats([r['map_ms'] for r in miss]),
            median_difference_ms=float(np.median([r['map_ms'] for r in miss])-
                                       np.median([r['map_ms'] for r in hit]))))
    result = dict(status='CONDITIONAL_COST_DIAGNOSIS', source=str(source),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        layer_calls=len(layer_rows), group_calls=len(group_rows),
        host_partition_scope='Exclusive nested host intervals within apply; residual includes kernel invocation and Python/bookkeeping. No CUDA-span subtraction.',
        cells=by_cell, strata=strata_out, models=models, correlations=correlations,
        matched_shape_contrasts=contrasts,
        scope='Reanalysis of one existing engine ABBA; no new GPU run, independent cohort, causal map intervention or complete-service headroom.',
        prediction_cutoff='Current layer route and entry residency known; first-group misses legal before ensure. Later layer routes are not pre-step inputs.',
        features_excluded=['actual load CUDA time', 'future route', 'future EOS', 'request identity', 'cell identity', 'step number'],
        unmeasured=['pure mapping CPU cost', 'pure exposed transfer wait', 'recoverable wall time', 'asynchronous map performance'])
    for name, obj in [('analysis.json', result), ('groups.json', group_rows), ('layers.json', layer_rows)]:
        with (args.out_dir/name).open('x') as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
            f.write('\n')
    print(json.dumps({k:result[k] for k in ['layer_calls','group_calls','cells','models','correlations']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
