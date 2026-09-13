#!/usr/bin/env python3
"""Three-term decode step cost model, calibrated on one arm and held out on others.

The question this answers
------------------------
Earlier rounds treated the decode step cost as a function of batch width alone,
c(w), and explained policy wall-time differences by how much work ran at low
width. That worked inside a single homogeneous workload but broke immediately
once output lengths differed: the measured per-token cost at the *same* width
moved by 15-34% within one episode, and the cross-arm g(42) differed by 11.5%.
Width alone is therefore not the state variable.

The model
---------
    t_step  =  gamma  +  alpha * w  +  beta * C

    w      decode requests in the step
    C      total context tokens in the step, i.e. sum of computed_before
    gamma  per-step cost that does not scale with the batch
    alpha  marginal cost of carrying one more request
    beta   marginal cost of one more context token to scan

The three terms are physically distinct rather than curve-fitting slack:
`gamma` is kernel launch, CUDA-graph replay and host bookkeeping, paid once per
step; `alpha * w` is the per-token dense work (for OLMoE each token routes to 8
of 64 resident experts, so this is a small GEMM); `beta * C` is the attention
scan over resident KV, which is why a step gets slower as sequences grow even at
constant width.

Why this matters for scheduling
-------------------------------
If `gamma` is large relative to a step's total, then wall time is governed by the
*number of steps*, and the number of steps is set by how concentrated the decode
work is across width. Halving the width does not halve step cost; it roughly
doubles the step count for the same tokens. That single fact is what the earlier
"tail waste" bookkeeping was approximating, and stating it as a cost term makes
it transferable across workloads instead of re-fit per campaign.

Deliberate limits
-----------------
* Fits executed traces. Not a counterfactual: it predicts the cost of a step
  sequence, not what a different policy's step sequence would have been.
* Step duration uses consecutive `start_s`, because `end_s` is when `schedule()`
  returned. So every term absorbs host-side work in that interval and none of
  them is a kernel time.
* Steps carrying prefill or recompute are excluded; their duration depends on
  token budget, not on decode width and context.
* `C` uses `computed_before`, the context each scheduled request had entering the
  step, which is the quantity attention actually scans.
* Coefficients are specific to this model, dtype, backend and GPU.
"""

import argparse, json, statistics as st
from pathlib import Path

CAPTURE_BUCKETS = [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64]


def capture_bucket(w):
    for b in CAPTURE_BUCKETS:
        if w <= b:
            return b
    return w


def pure_decode_rows(run_dir):
    """(width, total_context_tokens, duration_s) for pure decode steps."""
    raw = json.load(open(Path(run_dir) / "raw.json"))
    steps = raw["scheduler_steps"]
    dur = [steps[k + 1]["start_s"] - steps[k]["start_s"] for k in range(len(steps) - 1)]
    rows = []
    for k in range(len(steps) - 1):
        s = steps[k]
        if s.get("decode_requests", 0) <= 0:
            continue
        if s.get("recompute_tokens") or any(r.get("prefill_tokens") for r in s.get("scheduled", [])):
            continue
        ctx = sum(r["computed_before"] for r in s["scheduled"])
        rows.append((s["decode_requests"], ctx, dur[k]))
    return rows


def solve(a, b):
    """Least squares for a small dense system via Gaussian elimination."""
    n = len(b)
    m = [list(a[i]) + [b[i]] for i in range(n)]
    for i in range(n):
        p = max(range(i, n), key=lambda r: abs(m[r][i]))
        m[i], m[p] = m[p], m[i]
        if abs(m[i][i]) < 1e-300:
            raise ValueError("singular design matrix")
        for r in range(n):
            if r != i:
                f = m[r][i] / m[i][i]
                for c in range(i, n + 1):
                    m[r][c] -= f * m[i][c]
    return [m[i][n] / m[i][i] for i in range(n)]


def fit(rows, width_fn=lambda w: w):
    """Fit gamma, alpha, beta by ordinary least squares."""
    design = [(1.0, float(width_fn(w)), float(c)) for w, c, _ in rows]
    y = [t for _, _, t in rows]
    n = 3
    ata = [[sum(design[k][i] * design[k][j] for k in range(len(design))) for j in range(n)]
           for i in range(n)]
    atb = [sum(design[k][i] * y[k] for k in range(len(design))) for i in range(n)]
    gamma, alpha, beta = solve(ata, atb)
    return dict(gamma_s=gamma, alpha_s_per_request=alpha, beta_s_per_context_token=beta)


def evaluate(rows, coef, width_fn=lambda w: w):
    g, a, b = coef["gamma_s"], coef["alpha_s_per_request"], coef["beta_s_per_context_token"]
    pred = [g + a * width_fn(w) + b * c for w, c, _ in rows]
    act = [t for _, _, t in rows]
    ape = sorted(abs(p - t) / t for p, t in zip(pred, act))
    n = len(ape)
    total_p, total_a = sum(pred), sum(act)
    # Share of a representative step attributable to each term, at the median
    # width and context of this trace.
    mw = st.median([w for w, _, _ in rows])
    mc = st.median([c for _, c, _ in rows])
    denom = g + a * width_fn(mw) + b * mc
    return dict(
        n_steps=n,
        decode_time_predicted_s=total_p,
        decode_time_actual_s=total_a,
        total_rel_err=total_p / total_a - 1,
        median_rel_err=ape[n // 2],
        p90_rel_err=ape[int(0.9 * n)],
        median_width=mw,
        median_context_tokens=mc,
        share_fixed=g / denom,
        share_width=a * width_fn(mw) / denom,
        share_context=b * mc / denom,
    )


def step_count_attribution(rows_a, rows_b, coef):
    """How much of a decode-time gap is explained by step count alone."""
    g = coef["gamma_s"]
    d_steps = len(rows_b) - len(rows_a)
    d_time = sum(t for _, _, t in rows_b) - sum(t for _, _, t in rows_a)
    return dict(d_steps=d_steps, d_decode_time_s=d_time,
                fixed_cost_gap_s=d_steps * g,
                explained_by_step_count=(d_steps * g / d_time) if d_time else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--calibrate", required=True, help="NAME=PATH, the single calibration cell")
    ap.add_argument("--holdout", action="append", default=[], help="NAME=PATH, repeatable")
    ap.add_argument("--attribute", action="append", default=[],
                    help="BASE:OTHER, explain their decode-time gap by step count")
    ap.add_argument("--use-bucket", action="store_true",
                    help="use the CUDA-graph capture bucket instead of the raw width")
    ap.add_argument("--output-dir", required=True, type=Path)
    args = ap.parse_args()

    width_fn = capture_bucket if args.use_bucket else (lambda w: w)
    cells = {}
    cal_name, cal_path = args.calibrate.split("=", 1)
    cells[cal_name] = pure_decode_rows(cal_path)
    for spec in args.holdout:
        name, path = spec.split("=", 1)
        cells[name] = pure_decode_rows(path)

    coef = fit(cells[cal_name], width_fn)
    results = {name: evaluate(rows, coef, width_fn) for name, rows in cells.items()}

    attributions = {}
    for spec in args.attribute:
        base, other = spec.split(":", 1)
        attributions[spec] = step_count_attribution(cells[base], cells[other], coef)

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    payload = dict(
        width_variable="capture_bucket" if args.use_bucket else "decode_requests",
        calibration_cell=cal_name,
        coefficients=coef,
        coefficients_readable=dict(
            gamma_ms_per_step=coef["gamma_s"] * 1e3,
            alpha_us_per_request=coef["alpha_s_per_request"] * 1e6,
            beta_ns_per_context_token=coef["beta_s_per_context_token"] * 1e9),
        per_cell=results,
        step_count_attribution=attributions,
        evidence_ceiling="FIT_TO_EXECUTED_TRACES_NOT_A_COUNTERFACTUAL",
        not_claimed=[
            "no term is a kernel time; all absorb host work inside the step interval",
            "does not predict what step sequence another policy would produce",
            "coefficients are specific to this model, dtype, backend and GPU",
            "prefill and recompute steps are excluded, not modelled",
        ],
    )
    json.dump(payload, open(out / "cost_model.json", "w"), indent=1)

    L = ["# Three-term decode step cost model\n"]
    L.append(f"Width variable: `{payload['width_variable']}`. "
             f"Calibrated on `{cal_name}` only.\n")
    r = payload["coefficients_readable"]
    L.append("| term | value | reading |")
    L.append("|---|---:|---|")
    L.append(f"| gamma (per step) | {r['gamma_ms_per_step']:.4f} ms | launch, graph replay, host bookkeeping |")
    L.append(f"| alpha (per request) | {r['alpha_us_per_request']:.3f} us | per-token dense work incl. routed experts |")
    L.append(f"| beta (per context token) | {r['beta_ns_per_context_token']:.4f} ns | attention scan over resident KV |")
    L.append("\n| cell | steps | decode s actual | predicted | total err | median err | p90 err |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    for name, v in results.items():
        tag = " (calibration)" if name == cal_name else ""
        L.append(f"| {name}{tag} | {v['n_steps']} | {v['decode_time_actual_s']:.3f} | "
                 f"{v['decode_time_predicted_s']:.3f} | {100*v['total_rel_err']:+.2f}% | "
                 f"{100*v['median_rel_err']:.2f}% | {100*v['p90_rel_err']:.2f}% |")
    L.append("\n## Cost share at each cell's median step\n")
    L.append("| cell | median w | median context tok | fixed | width | context |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for name, v in results.items():
        L.append(f"| {name} | {v['median_width']:.0f} | {v['median_context_tokens']:.0f} | "
                 f"{100*v['share_fixed']:.1f}% | {100*v['share_width']:.1f}% | "
                 f"{100*v['share_context']:.1f}% |")
    if attributions:
        L.append("\n## Decode-time gap explained by step count alone\n")
        L.append("| pair | d steps | d decode s | gamma x d steps | explained |")
        L.append("|---|---:|---:|---:|---:|")
        for spec, v in attributions.items():
            e = f"{100*v['explained_by_step_count']:.1f}%" if v["explained_by_step_count"] else "n/a"
            L.append(f"| {spec} | {v['d_steps']:+d} | {v['d_decode_time_s']:+.3f} | "
                     f"{v['fixed_cost_gap_s']:+.3f} | {e} |")
    open(out / "report.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
