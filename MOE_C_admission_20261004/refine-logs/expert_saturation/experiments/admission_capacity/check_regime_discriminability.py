#!/usr/bin/env python3
"""Regime discriminability pre-check (C1-C3), run before any mechanism.

Rationale
---------
Four mechanisms in this repository failed with sign-unstable goodput. The
diagnosis in `20260911_aimability_r01` was that the *regime*, not the
controller, was the problem: in short-context steady the joint SLO is a
knife-edge statistic and the admission cap merely exchanges TTFT failures for
TPOT failures, leaving the joint pass count nearly fixed.

`20260912_feasibility_envelope_r01` then explained why: the closed-form
envelope puts the joint-SLO-feasible arrival rate at ~10.2 req/s, while that
workload arrived at 20 req/s -- about 2x past feasibility. A saturated system
has no scheduling headroom to find.

This module makes the diagnosis reusable. Given a static cap sweep it answers,
*before* any controller is written, whether the regime can discriminate
scheduling effects at all:

  C1  target not on the knife edge   -- |TPOT_p50/SLO - 1| exceeds the
                                        effect size you hope to detect
  C2  threshold-robust               -- a 1% SLO move flips few verdicts
  C3  lever not conservative         -- the joint pass count actually moves
                                        across the cap sweep

All three need only static episodes. If C3 fails, no controller can be
evaluated there and the regime must be changed first.

Also reports the model's feasibility verdict for the offered arrival rate, so
a saturated operating point is flagged even before the SLO statistics are read.

What this is NOT
----------------
- Not a predictor of whether a mechanism will win, only of whether a win could
  be *observed*. Passing C1-C3 is necessary, not sufficient.
- C3 is measured on the caps actually swept; a lever that only moves outside
  that range will look conservative.
- Thresholds are the ones this repository's evidence supports, not universal
  constants. They are arguments, not hard-coded law.
"""

import argparse
import json
import statistics as st
from pathlib import Path

CAPTURE_SIZES = (1, 2, 4, 8, 16, 24, 32)


def capture_bucket(width):
    for size in CAPTURE_SIZES:
        if width <= size:
            return size
    return CAPTURE_SIZES[-1]


def feasible_arrival_rate(cost_ms, tax_ms, width, out_tokens, tpot_slo_ms):
    """See 20260912_feasibility_envelope_r01: tau = c/(1 - lambda*tax)."""
    tax_s = tax_ms / 1000.0
    if cost_ms >= tpot_slo_ms:
        return 0.0
    lam_tpot = (1.0 - cost_ms / tpot_slo_ms) / tax_s
    lam_thru = width / ((cost_ms / 1000.0) * out_tokens + width * tax_s)
    return min(lam_tpot, lam_thru)


def verdict_counts(per_request, ttft_slo_s, tpot_slo_s):
    done = [r for r in per_request if r.get("status") == "completed"]
    return dict(
        n_done=len(done),
        n_pass=sum(1 for r in done
                   if r["ttft_s"] <= ttft_slo_s and not r.get("tpot_vacuous")
                   and r["tpot_s"] <= tpot_slo_s),
        n_ttft_fail=sum(1 for r in done if r["ttft_s"] > ttft_slo_s),
        n_tpot_fail=sum(1 for r in done if r["tpot_s"] > tpot_slo_s))


def check(episodes, ttft_slo_s, tpot_slo_s, expected_effect=0.10):
    """episodes: list of dicts with cap, per_request, and optional widths."""
    statics = sorted(episodes, key=lambda e: e["cap"])
    if len(statics) < 3:
        return dict(status="INSUFFICIENT_SWEEP", n_static=len(statics))

    rows = []
    for ep in statics:
        base = verdict_counts(ep["per_request"], ttft_slo_s, tpot_slo_s)
        moved_up = verdict_counts(ep["per_request"], ttft_slo_s * 1.01, tpot_slo_s * 1.01)
        moved_dn = verdict_counts(ep["per_request"], ttft_slo_s * 0.99, tpot_slo_s * 0.99)
        done = [r for r in ep["per_request"] if r.get("status") == "completed"]
        tpot = [r["tpot_s"] for r in done]
        rows.append(dict(
            cap=ep["cap"], **base,
            tpot_p50_over_slo=st.median(tpot) / tpot_slo_s if tpot else None,
            flips_per_1pct=max(abs(moved_up["n_pass"] - base["n_pass"]),
                               abs(moved_dn["n_pass"] - base["n_pass"]))))

    cohort = max(r["n_done"] for r in rows) or 1
    passes = [r["n_pass"] for r in rows]
    margins = [abs(r["tpot_p50_over_slo"] - 1.0) for r in rows
               if r["tpot_p50_over_slo"] is not None]

    c1_margin = st.median(margins) if margins else 0.0
    c2_flips = st.median(r["flips_per_1pct"] for r in rows) / cohort
    c3_swing = (max(passes) - min(passes)) / cohort

    c1 = c1_margin >= expected_effect
    c2 = c2_flips < expected_effect
    c3 = c3_swing >= expected_effect

    ttft_swing = (max(r["n_ttft_fail"] for r in rows)
                  - min(r["n_ttft_fail"] for r in rows)) / cohort
    tpot_swing = (max(r["n_tpot_fail"] for r in rows)
                  - min(r["n_tpot_fail"] for r in rows)) / cohort
    # An exchange shows as large failure-mode swings with a small joint swing.
    exchange = (c3_swing < expected_effect
                and max(ttft_swing, tpot_swing) >= 2 * max(c3_swing, 1e-9))

    return dict(
        status=("DISCRIMINATING" if (c1 and c2 and c3)
                else "NON_DISCRIMINATING"),
        expected_effect=expected_effect, cohort=cohort, per_cap=rows,
        C1_tpot_margin_from_threshold=c1_margin, C1_pass=c1,
        C2_verdict_flips_per_1pct_frac=c2_flips, C2_pass=c2,
        C3_joint_pass_swing_frac=c3_swing, C3_pass=c3,
        ttft_fail_swing_frac=ttft_swing, tpot_fail_swing_frac=tpot_swing,
        looks_like_failure_mode_exchange=exchange,
        blocking=[n for n, ok in (("C1", c1), ("C2", c2), ("C3", c3)) if not ok])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paired-dir", action="append", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--ttft-slo-ms", type=float, default=200.0)
    ap.add_argument("--tpot-slo-ms", type=float, default=9.0)
    ap.add_argument("--expected-effect", type=float, default=0.10,
                    help="smallest joint-SLO effect worth detecting, as a "
                         "fraction of the cohort")
    args = ap.parse_args()

    groups = {}
    for base in args.paired_dir:
        base = Path(base)
        analysis = json.load(open(base / "analysis" / "analysis.json"))
        for cell in analysis["cells"]:
            if not cell.get("eligible") or cell["plan"]["policy"] != "static":
                continue
            key = (base.name, Path(cell["path"]).parent.name, cell["plan"]["regime"])
            groups.setdefault(key, []).append(dict(
                cap=cell["plan"]["cap"],
                per_request=cell["metrics"].get("per_request", []),
                goodput_rps=cell["metrics"]["goodput_rps"]))

    results = []
    for key, eps in sorted(groups.items()):
        res = check(eps, args.ttft_slo_ms / 1000.0, args.tpot_slo_ms / 1000.0,
                    args.expected_effect)
        res.update(campaign=key[0], engine=key[1], regime=key[2])
        results.append(res)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json.dump(dict(ttft_slo_ms=args.ttft_slo_ms, tpot_slo_ms=args.tpot_slo_ms,
                   expected_effect=args.expected_effect, results=results),
              open(out / "discriminability.json", "w"), indent=1)

    lines = ["# Regime discriminability pre-check", "",
             f"Smallest effect worth detecting: {args.expected_effect*100:.0f}% of the cohort.",
             "All three checks use static episodes only; no controller is required.", "",
             "| campaign | engine | regime | C1 margin | C1 | C2 flips/1% | C2 | "
             "C3 pass swing | C3 | verdict | blocking |",
             "|---|---|---|---:|---|---:|---|---:|---|---|---|"]
    for r in results:
        if r["status"] == "INSUFFICIENT_SWEEP":
            lines.append(f'| {r.get("campaign","?")[-8:]} | {r.get("engine","?")} | '
                         f'{r.get("regime","?")} | - | - | - | - | - | - | '
                         f'INSUFFICIENT_SWEEP | - |')
            continue
        y = lambda b: "yes" if b else "**NO**"
        lines.append(
            f'| {r["campaign"][-8:]} | {r["engine"]} | {r["regime"]} | '
            f'{r["C1_tpot_margin_from_threshold"]*100:.1f}% | {y(r["C1_pass"])} | '
            f'{r["C2_verdict_flips_per_1pct_frac"]*100:.1f}% | {y(r["C2_pass"])} | '
            f'{r["C3_joint_pass_swing_frac"]*100:.1f}% | {y(r["C3_pass"])} | '
            f'{r["status"]} | {",".join(r["blocking"]) or "-"} |')

    exch = [r for r in results if r.get("looks_like_failure_mode_exchange")]
    if exch:
        lines += ["", "## Regimes where the lever only exchanges failure modes", ""]
        for r in exch:
            lines.append(
                f'- {r["engine"]}/{r["regime"]}: joint pass swing '
                f'{r["C3_joint_pass_swing_frac"]*100:.1f}% while TTFT-fail swings '
                f'{r["ttft_fail_swing_frac"]*100:.0f}% and TPOT-fail swings '
                f'{r["tpot_fail_swing_frac"]*100:.0f}%')
    lines += ["", "A regime failing C3 cannot evaluate any controller: change the",
              "operating point before implementing a mechanism.", ""]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
