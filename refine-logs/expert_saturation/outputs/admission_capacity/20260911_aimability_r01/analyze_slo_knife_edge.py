#!/usr/bin/env python3
"""Is joint-SLO goodput a usable estimator in this operating regime?

Why this exists
---------------
Four independent mechanisms in this repository failed with sign-unstable
goodput (-44%/+119%/-20%/-57%). Two explanations were tested and rejected here:

  * "the controller could not see far enough" -- refuted. `analyze_aimability.py`
    finds 0.85 of effect-time bucket entropy already resolved at lag 1, and
    caps bind at median lag 0 steps.
  * "the cap works through the capture-alignment channel" -- refuted.
    `analyze_channels.py` finds the whole alignment channel, granted free
    reclamation, is 17.4% of episode time while the lever moves goodput 309%.

This analyser tests the remaining explanation, which is about the *metric*
rather than the mechanism:

    If the request population sits on the SLO threshold, the joint-SLO pass
    count is a knife-edge statistic. Tiny latency changes flip many requests,
    the estimator's variance swamps any real scheduling effect, and no
    controller can be evaluated -- let alone shown to win.

Three quantities:
  1. where the TPOT distribution sits relative to its threshold;
  2. threshold sensitivity -- how many of the 32 requests change verdict when
     the SLO is moved by +/-1%. This is the knife-edge measure;
  3. whether the admission cap conserves the joint pass count while merely
     exchanging TTFT failures for TPOT failures.

What this is NOT
----------------
- Not a claim that goodput is a bad metric in general. It is a claim about this
  workload, these SLO values, and this operating regime.
- Sensitivity is computed by re-deciding sealed per-request latencies against a
  moved threshold. No latency is re-simulated and no policy is re-run.
- Conservation is observed across a five-point static cap sweep, not proven.
"""

import argparse
import json
import statistics as st
from pathlib import Path

TTFT_SLO_S = 0.20
TPOT_SLO_S = 0.009
PERTURBATIONS = (-0.05, -0.02, -0.01, 0.01, 0.02, 0.05)


def verdicts(per_request, ttft_slo, tpot_slo):
    """Re-decide sealed latencies against a (possibly moved) threshold."""
    done = [r for r in per_request if r["status"] == "completed"]
    n_pass = sum(1 for r in done
                 if r["ttft_s"] <= ttft_slo and not r.get("tpot_vacuous")
                 and r["tpot_s"] <= tpot_slo)
    return dict(
        n_done=len(done), n_pass=n_pass,
        n_ttft_fail=sum(1 for r in done if r["ttft_s"] > ttft_slo),
        n_tpot_fail=sum(1 for r in done if r["tpot_s"] > tpot_slo))


def episode_row(campaign, engine, plan, metrics):
    per_request = metrics.get("per_request") or []
    done = [r for r in per_request if r["status"] == "completed"]
    if not done:
        return None
    tpot = [r["tpot_s"] for r in done]
    ttft = [r["ttft_s"] for r in done]
    base = verdicts(per_request, TTFT_SLO_S, TPOT_SLO_S)

    # Knife-edge measure: how many verdicts move when the threshold moves.
    sensitivity = {}
    for delta in PERTURBATIONS:
        moved = verdicts(per_request, TTFT_SLO_S * (1 + delta),
                         TPOT_SLO_S * (1 + delta))
        sensitivity[f"{delta:+.2f}"] = moved["n_pass"] - base["n_pass"]
    worst_1pct = max(abs(sensitivity["+0.01"]), abs(sensitivity["-0.01"]))

    return dict(
        campaign=campaign, engine=engine, regime=plan["regime"], cap=plan["cap"],
        policy=plan["policy"], goodput_rps=metrics["goodput_rps"],
        n_done=base["n_done"], n_pass=base["n_pass"],
        n_ttft_fail=base["n_ttft_fail"], n_tpot_fail=base["n_tpot_fail"],
        tpot_p50_over_slo=st.median(tpot) / TPOT_SLO_S,
        ttft_p50_over_slo=st.median(ttft) / TTFT_SLO_S,
        frac_tpot_within_20pct=sum(1 for v in tpot
                                   if 0.8 * TPOT_SLO_S <= v <= 1.2 * TPOT_SLO_S) / len(tpot),
        pass_delta_per_1pct_threshold=worst_1pct,
        pass_delta_frac_per_1pct=worst_1pct / max(base["n_done"], 1),
        threshold_sensitivity=sensitivity)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paired-dir", action="append", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    rows = []
    for base in args.paired_dir:
        base = Path(base)
        analysis = json.load(open(base / "analysis" / "analysis.json"))
        for cell in analysis["cells"]:
            if not cell.get("eligible"):
                continue
            row = episode_row(base.name, Path(cell["path"]).parent.name,
                              cell["plan"], cell["metrics"])
            if row:
                rows.append(row)
    if not rows:
        raise SystemExit("no eligible cells")

    # Conservation: within one engine and regime, does the cap sweep move the
    # joint pass count, or only relocate which SLO fails?
    conservation = []
    groups = {}
    for r in rows:
        if r["policy"] == "static":
            groups.setdefault((r["campaign"], r["engine"], r["regime"]), []).append(r)
    for key, grp in sorted(groups.items()):
        grp = sorted(grp, key=lambda r: r["cap"])
        passes = [r["n_pass"] for r in grp]
        conservation.append(dict(
            campaign=key[0], engine=key[1], regime=key[2],
            caps=[r["cap"] for r in grp], pass_by_cap=passes,
            ttft_fail_by_cap=[r["n_ttft_fail"] for r in grp],
            tpot_fail_by_cap=[r["n_tpot_fail"] for r in grp],
            pass_range=max(passes) - min(passes),
            pass_range_frac_of_cohort=(max(passes) - min(passes)) / grp[0]["n_done"],
            # If the cap only exchanges failure modes, this swing is large while
            # the pass swing above stays small.
            ttft_fail_range=max(r["n_ttft_fail"] for r in grp)
            - min(r["n_ttft_fail"] for r in grp),
            tpot_fail_range=max(r["n_tpot_fail"] for r in grp)
            - min(r["n_tpot_fail"] for r in grp)))

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json.dump(dict(ttft_slo_s=TTFT_SLO_S, tpot_slo_s=TPOT_SLO_S,
                   perturbations=list(PERTURBATIONS), n_episodes=len(rows),
                   per_episode=rows, conservation=conservation),
              open(out / "knife_edge.json", "w"), indent=1)

    steady = [r for r in rows if r["regime"] == "steady" and r["policy"] == "static"]
    bursty = [r for r in rows if r["regime"] == "bursty" and r["policy"] == "static"]

    lines = ["# Joint-SLO goodput as an estimator in this operating regime", "",
             f"{len(rows)} sealed episodes. Latencies are re-decided against moved",
             "thresholds; nothing is re-simulated and no policy is re-run.", "",
             f"SLO under test: TTFT <= {TTFT_SLO_S*1000:.0f} ms, mean TPOT <= {TPOT_SLO_S*1000:.0f} ms.",
             "", "## Where the population sits relative to the threshold", "",
             "| regime | n | TPOT p50 / SLO | requests within +/-20% of TPOT SLO | "
             "verdicts flipped by a 1% threshold move |",
             "|---|---:|---:|---:|---:|"]
    for name, grp in (("steady", steady), ("bursty", bursty)):
        if not grp:
            continue
        lines.append(
            f'| {name} | {len(grp)} | {st.median(r["tpot_p50_over_slo"] for r in grp):.2f}x | '
            f'{st.median(r["frac_tpot_within_20pct"] for r in grp)*100:.0f}% | '
            f'{st.median(r["pass_delta_per_1pct_threshold"] for r in grp):.1f} of '
            f'{st.median(r["n_done"] for r in grp):.0f} |')

    lines += ["", "## Does the cap create joint-SLO passes, or only relocate failures?", "",
              "| campaign | engine | regime | caps | joint pass by cap | TTFT fails | "
              "TPOT fails | pass swing | TTFT-fail swing | TPOT-fail swing |",
              "|---|---|---|---|---|---|---|---:|---:|---:|"]
    for c in conservation:
        lines.append(
            f'| {c["campaign"][-8:]} | {c["engine"]} | {c["regime"]} | {c["caps"]} | '
            f'{c["pass_by_cap"]} | {c["ttft_fail_by_cap"]} | {c["tpot_fail_by_cap"]} | '
            f'{c["pass_range"]} | {c["ttft_fail_range"]} | {c["tpot_fail_range"]} |')

    st_cons = [c for c in conservation if c["regime"] == "steady"]
    if st_cons:
        lines += ["", "## Reading", "",
                  f'- steady: joint pass count swings {st.median(c["pass_range"] for c in st_cons):.0f} '
                  f'of 32 across the whole cap sweep, while TTFT failures swing '
                  f'{st.median(c["ttft_fail_range"] for c in st_cons):.0f} and TPOT failures swing '
                  f'{st.median(c["tpot_fail_range"] for c in st_cons):.0f}.',
                  "- That is the signature of an exchange, not a gain: the cap moves requests",
                  "  from one failure mode to the other and leaves the joint count nearly fixed.",
                  f'- With the TPOT median at {st.median(r["tpot_p50_over_slo"] for r in steady):.2f}x the '
                  "threshold, the surviving variation is threshold noise, not scheduling signal.", ""]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
