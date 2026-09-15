#!/usr/bin/env python3
"""Decompose an episode into the two cost terms, with heterogeneous lengths handled.

Relationship to `analyze_dispersion.py`
---------------------------------------
That script established, on four policies sharing one homogeneous 3072/1024
workload, that

    d_wall = d_marginal_recompute + d_tail_waste

closes to within 12-29% and that `tail waste` tracks wall time while `max ITL`
does not. It assumed one completion cluster, so "the drain" was simply the
interval between the first and last completion.

That assumption breaks the moment output lengths differ: short requests retire
early by design, so low-width steps appear in the middle of the episode rather
than only at the end. Defining the drain positionally would then charge the
heterogeneous arm for structure that is inherent to its workload rather than to
the policy. This script therefore keeps the *cost* definition and drops the
*positional* one:

    tail_waste = SUM over steps with w <= cut of ( c(w) - w * g_high )

Nothing here requires those steps to be contiguous or terminal. On a homogeneous
trace the two definitions coincide, which is checked against the sealed numbers
by `--expect` so a refactor cannot silently move the baseline.

A third term is reported but never folded into the model: `unavoidable_drain`,
the waste a perfectly scheduled run would still pay because the last requests
must finish with a shrinking batch. It is estimated from the frozen output
lengths alone, so it is a property of the workload, not of the policy. Reporting
it separately is what lets a reader see whether the heterogeneous arm's larger
tail waste is policy-attributable or baked into the length distribution.

Limits
------
* Descriptive accounting of executed traces. No counterfactual wall time.
* `g_high` is an in-phase average and absorbs overlapped prefill and bookkeeping;
  it is not kernel time.
* Step duration uses consecutive `start_s`, since `end_s` is when `schedule()`
  returned, not when the step finished.
* Per-cell n=1; the forward/reverse pair estimates a noise floor, not
  significance.
"""

import argparse, json, math, statistics as st
from collections import Counter
from pathlib import Path


def step_durations(steps):
    return [steps[k + 1]["start_s"] - steps[k]["start_s"] for k in range(len(steps) - 1)]


def kind(step):
    if step.get("recompute_tokens"):
        return "recompute"
    if any(r.get("prefill_tokens") for r in step.get("scheduled", [])):
        return "prefill"
    return "decode" if step.get("decode_requests", 0) > 0 else "idle"


def unavoidable_drain_s(lengths, width_cap, g_high):
    """Waste any schedule must pay once fewer than `width_cap` requests remain.

    With all requests admitted at once and each request i needing L_i decode
    steps, the number still generating at step t is |{i : L_i > t}|. Steps where
    that count is below the cap cost more per token than the high-width rate no
    matter how they are scheduled. This is a workload property; it is reported
    beside the measured tail waste and never subtracted from it.
    """
    if not lengths:
        return 0.0
    horizon = max(lengths)
    waste = 0.0
    for t in range(horizon):
        alive = sum(1 for L in lengths if L > t)
        if 0 < alive <= width_cap / 2:
            # One step at width `alive` versus the same tokens at high width.
            waste += max(0.0, alive * (g_high * (width_cap / max(alive, 1)) ** 0 - g_high))
    return waste


def analyse(run_dir, cut_frac):
    run_dir = Path(run_dir)
    raw = json.load(open(run_dir / "raw.json"))
    met = json.load(open(run_dir / "metrics.json"))
    cfg = json.load(open(run_dir / "config.json"))
    steps = raw["scheduler_steps"]
    dur = step_durations(steps)

    decode, recompute_idx, non_decode_s = [], [], 0.0
    for k in range(len(steps) - 1):
        c = kind(steps[k])
        if c == "decode":
            decode.append((steps[k]["decode_requests"], dur[k]))
        else:
            if c == "recompute":
                recompute_idx.append(k)
            non_decode_s += dur[k]

    by_width = {}
    for w, t in decode:
        by_width.setdefault(w, []).append(t)
    pure_median = st.median([t for _, t in decode]) if decode else None

    wmax = max((w for w, _ in decode), default=0)
    cut = wmax * cut_frac
    hi = [(w, t) for w, t in decode if w > cut]
    lo = [(w, t) for w, t in decode if w <= cut]
    hi_tok = sum(w for w, _ in hi)
    g_high = (sum(t for _, t in hi) / hi_tok) if hi_tok else None
    lo_tok = sum(w for w, _ in lo)
    tail_waste = (sum(t for _, t in lo) - lo_tok * g_high) if g_high else None

    # Marginal recompute: extra time in recompute-bearing steps beyond what a
    # plain decode step of median duration would have cost.
    med_pure = st.median([t for w, t in decode]) if decode else 0.0
    rc_s = sum(dur[k] for k in recompute_idx)
    marginal_recompute = rc_s - len(recompute_idx) * med_pure
    rc_tokens = sum(steps[k].get("recompute_tokens", 0) for k in recompute_idx)

    done = [r for r in met["per_request"] if r["status"] == "completed"]
    comp = sorted(r["completion_s"] for r in raw["requests"] if r.get("completion_s"))
    lengths = [r["n_output_tokens"] for r in done]

    return dict(
        label=run_dir.name,
        arm=cfg.get("arm"),
        policy=cfg.get("completion_policy"),
        cap=cfg.get("cap"),
        requests=cfg.get("requests"),
        wall_s=met["observation_duration_s"],
        throughput_rps=met["throughput_rps"],
        goodput_rps=met["goodput_rps"],
        n_slo_pass=met["n_slo_pass"],
        n_completed=met["n_completed"],
        preemptions=met.get("actual_preemption_count"),
        max_itl_s=max((max(r["itl_s"]) for r in done if r["itl_s"]), default=None),
        ttft_p95_s=sorted(r["ttft_s"] for r in done)[int(0.95 * len(done))] if done else None,
        completion_span_s=(comp[-1] - comp[0]) if comp else None,
        completion_cv=(st.pstdev(comp) / st.mean(comp)) if len(comp) > 1 else None,
        output_len_cv=(st.pstdev(lengths) / st.mean(lengths)) if lengths else None,
        output_len_mean=st.mean(lengths) if lengths else None,
        total_output_tokens=sum(lengths),
        tail_waste_s=tail_waste,
        low_width_steps=len(lo),
        low_width_s=sum(t for _, t in lo),
        high_width_steps=len(hi),
        g_high_s_per_token=g_high,
        cut_width=cut,
        marginal_recompute_s=marginal_recompute,
        recompute_steps=len(recompute_idx),
        recompute_tokens=rc_tokens,
        pure_decode_median_ms=(pure_median * 1000) if pure_median else None,
        non_decode_s=non_decode_s,
        width_histogram=Counter(w for w, _ in decode).most_common(8),
        cost_table={w: dict(n=len(v), c_ms=st.median(v) * 1000,
                            g_ms_per_token=st.median(v) * 1000 / w)
                    for w, v in sorted(by_width.items()) if len(v) >= 20},
        unavoidable_drain_s=(unavoidable_drain_s(lengths, wmax, g_high) if g_high else None),
        source_dir=str(run_dir),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cell", action="append", required=True, help="NAME=PATH")
    ap.add_argument("--pair", action="append", default=[],
                    help="BASE:INTERVENTION, compared within the same arm")
    ap.add_argument("--cut-frac", type=float, default=0.5)
    ap.add_argument("--expect", type=Path, default=None,
                    help="json of NAME -> {tail_waste_s, marginal_recompute_s} to re-check")
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    cells = {}
    for spec in args.cell:
        name, path = spec.split("=", 1)
        cells[name] = analyse(path, args.cut_frac)

    regressions = []
    if args.expect and args.expect.exists():
        for name, want in json.load(open(args.expect)).items():
            got = cells.get(name)
            if not got:
                continue
            for field, target in want.items():
                actual = got.get(field)
                ok = actual is not None and abs(actual - target) <= 0.02
                regressions.append(dict(cell=name, field=field, expected=target,
                                        actual=actual, within_20ms=ok))

    pairs = []
    for spec in args.pair:
        base, interv = spec.split(":", 1)
        b, i = cells[base], cells[interv]
        if b["arm"] and i["arm"] and b["arm"] != i["arm"]:
            raise ValueError(f"refusing to pair across arms: {base} vs {interv}")
        d_wall = i["wall_s"] - b["wall_s"]
        d_rc = i["marginal_recompute_s"] - b["marginal_recompute_s"]
        d_tw = i["tail_waste_s"] - b["tail_waste_s"]
        pairs.append(dict(
            arm=b["arm"], baseline=base, intervention=interv,
            d_wall_s=d_wall, d_marginal_recompute_s=d_rc, d_tail_waste_s=d_tw,
            model_s=d_rc + d_tw, residual_s=d_wall - (d_rc + d_tw),
            explained_frac=((d_rc + d_tw) / d_wall) if d_wall else None,
            d_max_itl_s=i["max_itl_s"] - b["max_itl_s"],
            d_completion_span_s=i["completion_span_s"] - b["completion_span_s"],
            throughput_delta_pct=100 * (i["throughput_rps"] / b["throughput_rps"] - 1),
        ))

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    json.dump(dict(cells=cells, pairs=pairs, regressions=regressions,
                   cut_frac=args.cut_frac,
                   evidence_ceiling="DESCRIPTIVE_ACCOUNTING_OF_EXECUTED_TRACES",
                   not_claimed=[
                       "no counterfactual wall time",
                       "unavoidable_drain is a workload property, never subtracted",
                       "g_high absorbs overlapped prefill/bookkeeping; not kernel time",
                       "per-cell n=1; forward/reverse estimates a floor, not significance",
                   ]),
              open(out / "terms.json", "w"), indent=1)

    L = ["# Two-term decomposition with heterogeneous output lengths\n"]
    L.append("| cell | arm | policy | wall s | max ITL s | out-len CV | span s | tail waste s | marg recompute s | preempt |")
    L.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for n, c in cells.items():
        L.append(f"| {n} | {c['arm']} | {c['policy']} | {c['wall_s']:.3f} | "
                 f"{c['max_itl_s']:.3f} | {c['output_len_cv']:.3f} | {c['completion_span_s']:.3f} | "
                 f"{c['tail_waste_s']:.3f} | {c['marginal_recompute_s']:.3f} | {c['preemptions']} |")
    if pairs:
        L.append("\n## Within-arm pairs\n")
        L.append("| arm | base -> interv | d wall | d recompute | d tail | model | residual | explained | d max ITL | thr % |")
        L.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for p in pairs:
            e = f"{100*p['explained_frac']:.0f}%" if p["explained_frac"] else "n/a"
            L.append(f"| {p['arm']} | {p['baseline']} → {p['intervention']} | {p['d_wall_s']:+.3f} | "
                     f"{p['d_marginal_recompute_s']:+.3f} | {p['d_tail_waste_s']:+.3f} | {p['model_s']:+.3f} | "
                     f"{p['residual_s']:+.3f} | {e} | {p['d_max_itl_s']:+.3f} | {p['throughput_delta_pct']:+.2f} |")
    if regressions:
        L.append("\n## Baseline regression against sealed numbers\n")
        L.append("| cell | field | expected | actual | within 20 ms |")
        L.append("|---|---|---:|---:|---|")
        for r in regressions:
            L.append(f"| {r['cell']} | {r['field']} | {r['expected']:.3f} | "
                     f"{r['actual']:.3f} | {'yes' if r['within_20ms'] else 'NO'} |")
    L.append("\n## Workload-inherent drain (reported, never subtracted)\n")
    L.append("| cell | measured tail waste s | workload-inherent estimate s |")
    L.append("|---|---:|---:|")
    for n, c in cells.items():
        u = c["unavoidable_drain_s"]
        L.append(f"| {n} | {c['tail_waste_s']:.3f} | {u:.3f} |" if u is not None
                 else f"| {n} | {c['tail_waste_s']:.3f} | n/a |")
    open(out / "report.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
