#!/usr/bin/env python3
"""A closed-form admission-feasibility envelope for MoE decode serving.

What this is
------------
Three explanations for this repository's four sign-unstable scheduling failures
were tested in `20260911_aimability_r01`. Two were refuted (information was
sufficient; the capture-alignment channel is 18x too small). The third --
that the joint SLO is a knife-edge statistic in the tested regime -- held, but
only as an observation. This analyser turns that observation into a model.

Two measured primitives:

  1. the pure-decode step cost staircase  c(w)  [ms], quantised by CUDA-graph
     capture bucket, cross-run reproducible to 0.125 ms;
  2. the prefill interference tax  a + b*P  [ms], the extra cost a step pays
     when it also carries P prefill tokens. Every resident request pays it,
     not just the arriving one.

In steady state with arrival rate lambda, a resident request is overlapped by
lambda * (L * tau) admissions during its own residency, so

    tau = c(w) + lambda * tau * (a + b*P)
    => tau = c(w) / (1 - lambda * (a + b*P))            [per-token time]

which is a closed form with a pole at lambda = 1/(a+b*P). Joint-SLO feasibility
then needs BOTH

    (T) tau <= TPOT_SLO                       -- decode fast enough per token
    (Q) lambda <= mu(w) = w / (tau * L)       -- service rate covers arrivals

(Q) is necessary for bounded queueing, hence for bounded TTFT. If no width
satisfies both, the operating point admits no scheduling headroom at all and
every controller must look flat -- which is what was measured.

Validation protocol
-------------------
c(w) and (a, b) are FITTED, so validating on the same episodes would be
circular. Primitives are therefore fitted on one campaign and used to predict
the other, and vice versa. Only out-of-sample errors are reported.

The tau formula's *structure* is not fitted -- it is derived - so predicting
measured per-request TPOT from realised width and realised admission rate is a
genuine test.

What this is NOT
----------------
- Not a policy result. Nothing is executed; no controller is proposed here.
- (Q) is a necessary condition for bounded TTFT, not a sufficient one: it says
  nothing about which individual request misses its deadline.
- Steady-state mean-value model. It does not predict tails, and it treats
  arrivals as a rate rather than a sequence, so it cannot describe bursts.
- The envelope is specific to the measured staircase: one model, one GPU, one
  engine build, one output length, one prompt length.
"""

import argparse
import glob
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

CAPTURE_SIZES = (1, 2, 4, 8, 16, 24, 32)
MIN_SAMPLES_PER_WIDTH = 30


def capture_bucket(width):
    for size in CAPTURE_SIZES:
        if width <= size:
            return size
    return CAPTURE_SIZES[-1]


def step_exec_ms(data):
    """Per-step execution time via output-receipt alignment.

    `end_s` is when schedule() returned, not when the step finished; it covers
    ~1.6% of wall clock. With async_scheduling=False and stream_interval=1 one
    schedule() maps to exactly one delivery, so receipts recover true step time
    (99.2% of wall clock). Returns None if the 1:1 alignment does not hold.
    """
    steps = data.get("scheduler_steps") or []
    receipts = sorted({e["received_s"] for e in data.get("output_events", [])})
    if not steps or len(receipts) != len(steps):
        return None
    out = []
    for k, s in enumerate(steps):
        dur = receipts[k] - s["start_s"]
        if dur <= 0:
            return None
        out.append(dur * 1000.0)
    return out


def load_episodes(campaign_dirs):
    episodes = []
    for base in campaign_dirs:
        base = Path(base)
        analysis = json.load(open(base / "analysis" / "analysis.json"))
        for cell in analysis["cells"]:
            if not cell.get("eligible"):
                continue
            name, engine = Path(cell["path"]).name, Path(cell["path"]).parent.name
            hits = glob.glob(str(base / "gpu_results" / engine / name)) or \
                glob.glob(str(base / "gpu_results" / "**" / name), recursive=True)
            if not hits:
                continue
            data = json.load(open(hits[0]))
            times = step_exec_ms(data)
            if times is None:
                continue
            episodes.append(dict(campaign=base.name, engine=engine, cell=name,
                                 plan=cell["plan"], metrics=cell["metrics"],
                                 raw=data, step_ms=times))
    return episodes


def fit_primitives(episodes):
    """Staircase c(w) and prefill tax (a, b) from a set of episodes."""
    pure, mixed = defaultdict(list), []
    for ep in episodes:
        for k, s in enumerate(ep["raw"]["scheduler_steps"]):
            width = s.get("decode_requests", 0)
            if width <= 0:
                continue
            pf = sum(r.get("prefill_tokens", 0) for r in s.get("scheduled", []))
            if pf == 0:
                pure[width].append(ep["step_ms"][k])
            else:
                mixed.append((width, pf, ep["step_ms"][k]))

    cost = {w: st.median(v) for w, v in pure.items() if len(v) >= MIN_SAMPLES_PER_WIDTH}
    if not cost:
        return None
    # Bucket-level cost: the staircase is flat inside a bucket, so collapse to
    # the bucket median. This is what makes the model depend on geometry rather
    # than on per-width sampling noise.
    by_bucket = defaultdict(list)
    for w, v in pure.items():
        if len(v) >= MIN_SAMPLES_PER_WIDTH:
            by_bucket[capture_bucket(w)].extend(v)
    bucket_cost = {b: st.median(v) for b, v in by_bucket.items()}

    pts = [(pf, ms - cost[w]) for w, pf, ms in mixed if w in cost]
    if len(pts) < 50:
        return None
    xs = [p for p, _ in pts]
    ys = [t for _, t in pts]
    mx, my = st.mean(xs), st.mean(ys)
    denom = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in pts) / denom if denom else 0.0
    a = my - b * mx
    return dict(cost_by_width=cost, cost_by_bucket=bucket_cost,
                tax_intercept_ms=a, tax_slope_ms_per_token=b, n_tax_points=len(pts))


def predict_tau_ms(cost_ms, lam_per_s, tax_ms):
    """tau = c / (1 - lambda * tax). Returns None past the pole."""
    load = lam_per_s * (tax_ms / 1000.0)
    if load >= 1.0:
        return None
    return cost_ms / (1.0 - load)


def feasible_lambda(cost_ms, tax_ms, width, out_tokens, tpot_slo_ms):
    """Largest arrival rate at which width `width` satisfies (T) and (Q).

    (T) c/(1-lam*tax) <= slo          -> lam <= (1 - c/slo)/tax
    (Q) lam <= w/(tau*L), with tau itself a function of lam. Substituting and
        solving the linear equation gives lam_Q below.
    """
    tax_s = tax_ms / 1000.0
    if cost_ms >= tpot_slo_ms:
        return dict(width=width, lam_T=0.0, lam_Q=None, lam_max=0.0,
                    binding="TPOT_at_zero_load")
    lam_t = (1.0 - cost_ms / tpot_slo_ms) / tax_s
    # (Q): lam = w / (tau * L) with tau = c/(1-lam*tax)
    #      lam * c * L / (1 - lam*tax) = w
    #      lam * c * L = w - w*lam*tax
    #      lam (c*L + w*tax) = w
    base = width / ((cost_ms / 1000.0) * out_tokens + width * tax_s)
    lam_q = base
    lam_max = min(lam_t, lam_q)
    return dict(width=width, lam_T=lam_t, lam_Q=lam_q, lam_max=lam_max,
                binding="TPOT" if lam_t < lam_q else "throughput")


def validate(episodes, prim, tpot_slo_ms, out_tokens):
    """Out-of-sample: predict measured per-request TPOT from realised state."""
    rows = []
    for ep in episodes:
        steps = ep["raw"]["scheduler_steps"]
        pure_widths = [s["decode_requests"] for s in steps
                       if s.get("decode_requests", 0) > 0
                       and not any(r.get("prefill_tokens") for r in s.get("scheduled", []))]
        if not pure_widths:
            continue
        med_w = st.median(pure_widths)
        bucket = capture_bucket(int(med_w))
        cost = prim["cost_by_bucket"].get(bucket)
        if cost is None:
            continue
        # Realised admission rate: prefill-carrying steps per second of episode.
        duration = ep["metrics"]["observation_duration_s"]
        n_admit = sum(1 for s in steps if any(r.get("prefill_tokens")
                                              for r in s.get("scheduled", [])))
        lam_realised = n_admit / duration if duration else 0.0
        mean_prefill = st.mean([sum(r.get("prefill_tokens", 0) for r in s.get("scheduled", []))
                                for s in steps
                                if any(r.get("prefill_tokens") for r in s.get("scheduled", []))]) \
            if n_admit else 0.0
        tax = prim["tax_intercept_ms"] + prim["tax_slope_ms_per_token"] * mean_prefill
        pred = predict_tau_ms(cost, lam_realised, tax)

        done = [r for r in ep["metrics"].get("per_request", []) if r["status"] == "completed"]
        if not done or pred is None:
            continue
        measured = st.median(r["tpot_s"] for r in done) * 1000.0
        rows.append(dict(
            campaign=ep["campaign"], engine=ep["engine"], regime=ep["plan"]["regime"],
            cap=ep["plan"]["cap"], policy=ep["plan"]["policy"],
            median_pure_width=med_w, bucket=bucket, bucket_cost_ms=cost,
            lam_realised_per_s=lam_realised, mean_prefill_tokens=mean_prefill,
            tax_ms=tax, predicted_tpot_ms=pred, measured_tpot_ms=measured,
            abs_err_ms=abs(pred - measured), rel_err=abs(pred - measured) / measured,
            predicted_pass=pred <= tpot_slo_ms,
            measured_pass=measured <= tpot_slo_ms))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", action="append", required=True,
                    help="directory with analysis/analysis.json and gpu_results/")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--tpot-slo-ms", type=float, default=9.0)
    ap.add_argument("--ttft-slo-ms", type=float, default=200.0)
    ap.add_argument("--out-tokens", type=int, default=128)
    ap.add_argument("--prompt-tokens", type=int, default=128)
    args = ap.parse_args()

    episodes = load_episodes(args.campaign)
    if not episodes:
        raise SystemExit("no usable episodes")

    by_campaign = defaultdict(list)
    for ep in episodes:
        by_campaign[ep["campaign"]].append(ep)

    # Out-of-sample: fit on every campaign except the one being predicted.
    holdout, fits = [], {}
    for name in by_campaign:
        train = [ep for c, eps in by_campaign.items() if c != name for ep in eps]
        if not train:
            continue
        prim = fit_primitives(train)
        if prim is None:
            continue
        fits[name] = prim
        for row in validate(by_campaign[name], prim, args.tpot_slo_ms, args.out_tokens):
            row["fitted_on"] = "others"
            holdout.append(row)

    # Envelope from all episodes pooled (reported separately from validation).
    prim_all = fit_primitives(episodes)
    tax_all = (prim_all["tax_intercept_ms"]
               + prim_all["tax_slope_ms_per_token"] * args.prompt_tokens)
    envelope = []
    for bucket, cost in sorted(prim_all["cost_by_bucket"].items()):
        envelope.append(dict(bucket=bucket, cost_ms=cost,
                             **feasible_lambda(cost, tax_all, bucket,
                                               args.out_tokens, args.tpot_slo_ms)))
    best = max(envelope, key=lambda e: e["lam_max"])

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json.dump(dict(tpot_slo_ms=args.tpot_slo_ms, ttft_slo_ms=args.ttft_slo_ms,
                   out_tokens=args.out_tokens, prompt_tokens=args.prompt_tokens,
                   capture_sizes=list(CAPTURE_SIZES),
                   primitives_pooled=prim_all, tax_at_prompt_ms=tax_all,
                   envelope=envelope, critical_point=best,
                   holdout_fits={k: {kk: vv for kk, vv in v.items()
                                     if kk != "cost_by_width"}
                                 for k, v in fits.items()},
                   holdout_validation=holdout),
              open(out / "envelope.json", "w"), indent=1)

    errs = [r["rel_err"] for r in holdout]
    agree = sum(1 for r in holdout if r["predicted_pass"] == r["measured_pass"])
    lines = ["# Closed-form admission-feasibility envelope", "",
             f"Primitives fitted out-of-sample: predictions for each campaign use",
             f"only the other campaign's episodes. {len(holdout)} episodes validated.",
             "", "## Measured primitives (pooled, for the envelope only)", "",
             f'- prefill tax: `{prim_all["tax_intercept_ms"]:.4f} + '
             f'{prim_all["tax_slope_ms_per_token"]:.6f} x prefill_tokens` ms '
             f'({prim_all["n_tax_points"]} matched pairs)',
             f"- tax at prompt={args.prompt_tokens}: **{tax_all:.3f} ms per admission**",
             f"- model pole: lambda = 1/tax = **{1000/tax_all:.1f} req/s**", "",
             "| capture bucket | c(w) ms | lambda_max from TPOT | lambda_max from throughput | "
             "feasible lambda | binding constraint |",
             "|---:|---:|---:|---:|---:|---|"]
    for e in envelope:
        lt = "0 (c>=SLO)" if e["lam_T"] == 0 else f'{e["lam_T"]:.2f}'
        lq = "n/a" if e["lam_Q"] is None else f'{e["lam_Q"]:.2f}'
        lines.append(f'| {e["bucket"]} | {e["cost_ms"]:.3f} | {lt} | {lq} | '
                     f'**{e["lam_max"]:.2f}** | {e["binding"]} |')

    lines += ["", "## Critical operating point", "",
              f'- widest feasible bucket: **{best["bucket"]}** at c = {best["cost_ms"]:.3f} ms',
              f'- maximum joint-SLO-feasible arrival rate: **{best["lam_max"]:.2f} req/s** '
              f'(inter-arrival **{1000/best["lam_max"]:.1f} ms**)',
              f'- binding constraint there: {best["binding"]}', "",
              "## Out-of-sample validation of the tau formula", ""]
    if errs:
        lines += [f"- median relative error: **{st.median(errs)*100:.2f}%**",
                  f"- p90 relative error: {sorted(errs)[int(0.9*(len(errs)-1))]*100:.2f}%",
                  f"- max relative error: {max(errs)*100:.2f}%",
                  f"- TPOT pass/fail verdict agreement: **{agree}/{len(holdout)}**", ""]
        lines += ["| regime | cap | median width | bucket | realised lambda | predicted TPOT | "
                  "measured TPOT | rel err | verdict match |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
        for r in sorted(holdout, key=lambda x: (x["regime"], x["cap"]))[:24]:
            lines.append(
                f'| {r["regime"]} | {r["cap"]} | {r["median_pure_width"]:.0f} | {r["bucket"]} | '
                f'{r["lam_realised_per_s"]:.2f} | {r["predicted_tpot_ms"]:.3f} | '
                f'{r["measured_tpot_ms"]:.3f} | {r["rel_err"]*100:.2f}% | '
                f'{"yes" if r["predicted_pass"]==r["measured_pass"] else "NO"} |')
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
