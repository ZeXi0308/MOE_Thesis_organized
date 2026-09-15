"""Adjudicate the frozen predictions F1-F4 for the capture-aligned ladder.

Compares the 2026-09-08 aligned-ladder run (caps 8,16,24,32) against the sealed
2026-09-06 policy probe (caps 8,12,16,32). Both used the same texts, arrivals,
engine configuration, SLO and feedback rule; a differential equivalence test
(`test_sealed_feedback_equivalence.py`) proves the textual code change between
them does not alter behaviour on the `--include-feedback` path.

Verdicts are emitted mechanically from the frozen criteria in
`next_experiment/DECISIONS.md`. F1 is adjudicated in two forms, because the frozen
wording conflated the admission cap with the realised decode width:

  F1-as-stated  : does static24's pure-step median land on the 24-bucket plateau?
  F1-mechanism  : do steps whose *realised width* is 17-24 land on that plateau?

Reporting only the second would be moving the goalposts, so both are reported.
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

CAPTURE_SIZES = (1, 2, 4, 8, 16, 24, 32)
F1_PLATEAU_MS = (9.30, 9.51)      # frozen from the sealed widths 17-24
SEALED_FEEDBACK_STEADY_WASTE = 0.301   # policy_probe/forward/cell-010
F1_TOLERANCE_MS = 0.10            # allowed slack when re-checking the plateau


def ceil_capture(width):
    for size in CAPTURE_SIZES:
        if width <= size:
            return size
    return CAPTURE_SIZES[-1]


def load(path):
    data = json.loads(Path(path).read_text())
    steps, receipts = data["scheduler_steps"], sorted({e["received_s"] for e in data["output_events"]})
    if len(steps) != len(receipts) or data["status"] != "COMPLETE":
        return None
    pure, widths = [], []
    for k, step in enumerate(steps):
        if sum(r["prefill_tokens"] for r in step["scheduled"]) or step["decode_requests"] <= 0:
            continue
        widths.append(step["decode_requests"])
        pure.append((step["decode_requests"], (receipts[k] - step["start_s"]) * 1000.0))
    done = [r for r in data["requests"] if r["status"] == "completed" and len(r["token_times_s"]) > 1]
    if not widths or not done:
        return None
    tpots = [(r["token_times_s"][-1] - r["token_times_s"][0]) * 1000.0
             / (len(r["token_times_s"]) - 1) for r in done]
    ttfts = [r["token_times_s"][0] - r["arrival_s"] for r in done]
    return dict(path=str(path), regime=data["regime"], cap=data["target_cap"],
                policy=data.get("policy", "static"), wall_s=data["observation_end_s"],
                pure=pure, widths=widths, tpots=tpots, ttfts=ttfts,
                median_tpot_ms=st.median(tpots), median_width=st.median(widths),
                mean_padding_waste=st.mean(1 - w / ceil_capture(w) for w in widths),
                median_pure_step_ms=st.median(ms for _, ms in pure),
                n_actions=len(data.get("actions", [])))


def collect(dirs, ttft_slo, tpot_slo):
    cells = []
    for root in dirs:
        for path in sorted(glob.glob(str(Path(root) / "**" / "cell-*.json"), recursive=True)):
            cell = load(path)
            if cell is None:
                continue
            joint = sum(1 for t, f in zip(cell["tpots"], cell["ttfts"])
                        if t <= tpot_slo * 1000 and f <= ttft_slo)
            cell["joint_pass"] = joint
            cell["goodput"] = joint / cell["wall_s"]
            cells.append(cell)
    return cells


def width_plateau(cells, lo, hi):
    pooled = defaultdict(list)
    for cell in cells:
        for width, ms in cell["pure"]:
            if lo <= width <= hi:
                pooled[width].append(ms)
    per_width = {w: st.median(v) for w, v in sorted(pooled.items()) if len(v) >= 20}
    allv = [ms for v in pooled.values() for ms in v]
    return dict(per_width_median_ms=per_width, n_steps=len(allv),
                pooled_median_ms=st.median(allv) if allv else None,
                min_median=min(per_width.values()) if per_width else None,
                max_median=max(per_width.values()) if per_width else None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aligned-dir", required=True)
    parser.add_argument("--sealed-dir", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ttft-slo-s", type=float, default=0.20)
    parser.add_argument("--tpot-slo-s", type=float, default=0.009)
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)

    aligned = collect([args.aligned_dir], args.ttft_slo_s, args.tpot_slo_s)
    sealed = collect(args.sealed_dir, args.ttft_slo_s, args.tpot_slo_s)

    def pick(cells, regime, policy, cap=None):
        return [c for c in cells if c["regime"] == regime and c["policy"] == policy
                and (cap is None or c["cap"] == cap) and c["median_tpot_ms"] > 4.0]

    # ---- F1 ----
    f1 = {}
    for regime in ("steady", "bursty"):
        cap24 = pick(aligned, regime, "static", 24)
        if not cap24:
            continue
        stated = st.median([c["median_pure_step_ms"] for c in cap24])
        f1[regime] = dict(
            n_episodes=len(cap24), static24_median_pure_step_ms=stated,
            static24_median_realised_width=st.median([c["median_width"] for c in cap24]),
            frozen_plateau_ms=list(F1_PLATEAU_MS),
            f1_as_stated_pass=F1_PLATEAU_MS[0] <= stated <= F1_PLATEAU_MS[1])
    mech = width_plateau(aligned, 17, 24)
    sealed_mech = width_plateau(sealed, 17, 24)
    f1_mechanism = dict(
        aligned=mech, sealed=sealed_mech,
        aligned_plateau_within_frozen_band=(
            mech["min_median"] is not None
            and F1_PLATEAU_MS[0] - F1_TOLERANCE_MS <= mech["min_median"]
            and mech["max_median"] <= F1_PLATEAU_MS[1] + F1_TOLERANCE_MS),
        independent_reproduction_max_abs_diff_ms=(
            max(abs(mech["per_width_median_ms"][w] - sealed_mech["per_width_median_ms"][w])
                for w in mech["per_width_median_ms"]
                if w in sealed_mech["per_width_median_ms"])
            if mech["per_width_median_ms"] and sealed_mech["per_width_median_ms"] else None))

    # ---- F2/F3/F4 ----
    comparisons = {}
    for regime in ("steady", "bursty"):
        new_fb = pick(aligned, regime, "feedback")
        old_fb = pick(sealed, regime, "feedback")
        if not new_fb or not old_fb:
            continue
        new_static = pick(aligned, regime, "static")
        old_static = pick(sealed, regime, "static")
        best_new = max((st.median([c["goodput"] for c in pick(aligned, regime, "static", cap)])
                        for cap in sorted({c["cap"] for c in new_static})), default=None)
        best_old = max((st.median([c["goodput"] for c in pick(sealed, regime, "static", cap)])
                        for cap in sorted({c["cap"] for c in old_static})), default=None)
        new_waste = st.median([c["mean_padding_waste"] for c in new_fb])
        old_waste = st.median([c["mean_padding_waste"] for c in old_fb])
        new_tpot = st.median([c["median_tpot_ms"] for c in new_fb])
        old_tpot = st.median([c["median_tpot_ms"] for c in old_fb])
        new_gp = st.median([c["goodput"] for c in new_fb])
        comparisons[regime] = dict(
            aligned_feedback_episodes=len(new_fb), sealed_feedback_episodes=len(old_fb),
            aligned_feedback_goodput=[round(c["goodput"], 4) for c in new_fb],
            sealed_feedback_goodput=[round(c["goodput"], 4) for c in old_fb],
            aligned_feedback_waste=new_waste, sealed_feedback_waste=old_waste,
            sealed_reference_steady_waste=SEALED_FEEDBACK_STEADY_WASTE,
            aligned_feedback_median_tpot_ms=new_tpot, sealed_feedback_median_tpot_ms=old_tpot,
            aligned_feedback_median_width=st.median([c["median_width"] for c in new_fb]),
            sealed_feedback_median_width=st.median([c["median_width"] for c in old_fb]),
            aligned_actions=[c["n_actions"] for c in new_fb],
            sealed_actions=[c["n_actions"] for c in old_fb],
            best_measured_static_aligned=best_new, best_measured_static_sealed=best_old,
            f2_waste_reduced=new_waste < old_waste,
            f3_tpot_not_worse=new_tpot <= old_tpot * 1.01,
            f4_beats_best_static_in_own_engine=(best_new is not None and new_gp > best_new),
            feedback_vs_best_static_pct=(100 * (new_gp / best_new - 1)
                                         if best_new else None))

    # Repeat spread is the honest noise floor for any of the above.
    spread = {}
    for regime in ("steady", "bursty"):
        for policy in ("static", "shadow", "feedback"):
            for cap in sorted({c["cap"] for c in aligned}):
                group = pick(aligned, regime, policy, cap)
                if len(group) < 2:
                    continue
                values = [c["goodput"] for c in group]
                spread[f"{regime}_{policy}{cap}"] = dict(
                    goodput=[round(v, 4) for v in values],
                    relative_spread_pct=100 * (max(values) - min(values)) / min(values))

    report = dict(
        evidence_type="NATIVE_VLLM_INPROCESS_REQUEST_LEVEL_GPU_RUN",
        aligned_dir=args.aligned_dir, sealed_dirs=args.sealed_dir,
        n_aligned_cells=len(aligned), n_sealed_cells=len(sealed),
        slo=dict(ttft_s=args.ttft_slo_s, tpot_s=args.tpot_slo_s),
        f1_static24=f1, f1_mechanism_by_realised_width=f1_mechanism,
        f2_f3_f4=comparisons, repeat_spread=spread)
    (out / "verdict.json").write_text(json.dumps(report, indent=2) + "\n")

    lines = ["# Frozen predictions F1-F4, adjudicated", "",
             f"aligned cells {len(aligned)}; sealed comparison cells {len(sealed)}", "",
             "## F1 as stated: static24 pure-step median on the 24-bucket plateau", "",
             "| regime | episodes | static24 median pure step ms | realised median width | "
             "frozen band ms | F1-as-stated |", "|---|---:|---:|---:|---|---:|"]
    for regime, v in f1.items():
        lines.append(f'| {regime} | {v["n_episodes"]} | {v["static24_median_pure_step_ms"]:.3f} | '
                     f'{v["static24_median_realised_width"]:.0f} | {v["frozen_plateau_ms"]} | '
                     f'**{v["f1_as_stated_pass"]}** |')
    m, s = f1_mechanism["aligned"], f1_mechanism["sealed"]
    lines += ["", "## F1 mechanism: steps whose realised width is 17-24", "",
              f'- aligned run: {m["n_steps"]} steps, per-width medians '
              f'{ {w: round(v, 3) for w, v in m["per_width_median_ms"].items()} }',
              f'- sealed run: {s["n_steps"]} steps, per-width medians '
              f'{ {w: round(v, 3) for w, v in s["per_width_median_ms"].items()} }',
              f'- aligned plateau inside frozen band +/-{F1_TOLERANCE_MS} ms: '
              f'**{f1_mechanism["aligned_plateau_within_frozen_band"]}**',
              f'- max abs per-width difference between the two independent runs: '
              f'**{f1_mechanism["independent_reproduction_max_abs_diff_ms"]:.3f} ms**', ""]
    lines += ["## F2/F3/F4: aligned ladder versus sealed ladder", ""]
    for regime, v in comparisons.items():
        lines += [f'### {regime}', "",
                  f'- feedback goodput: aligned {v["aligned_feedback_goodput"]} vs '
                  f'sealed {v["sealed_feedback_goodput"]}',
                  f'- feedback padding waste: aligned {v["aligned_feedback_waste"]:.3f} vs '
                  f'sealed {v["sealed_feedback_waste"]:.3f}',
                  f'- feedback median decode width: aligned {v["aligned_feedback_median_width"]:.0f} vs '
                  f'sealed {v["sealed_feedback_median_width"]:.0f}',
                  f'- feedback median TPOT: aligned {v["aligned_feedback_median_tpot_ms"]:.3f} ms vs '
                  f'sealed {v["sealed_feedback_median_tpot_ms"]:.3f} ms',
                  f'- applied actions: aligned {v["aligned_actions"]} vs sealed {v["sealed_actions"]}',
                  f'- best measured static: aligned {v["best_measured_static_aligned"]:.3f} vs '
                  f'sealed {v["best_measured_static_sealed"]:.3f}',
                  f'- feedback vs best measured static in its own engine: '
                  f'{v["feedback_vs_best_static_pct"]:+.2f}%',
                  f'- **F2 waste reduced: {v["f2_waste_reduced"]}**',
                  f'- **F3 TPOT not worse: {v["f3_tpot_not_worse"]}**',
                  f'- **F4 beats best static: {v["f4_beats_best_static_in_own_engine"]}**', ""]
    lines += ["## Repeat spread (noise floor)", "",
              "| arm | goodput repeats | relative spread % |", "|---|---|---:|"]
    for key, v in spread.items():
        lines.append(f'| {key} | {v["goodput"]} | {v["relative_spread_pct"]:.2f} |')
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print(f"aligned {len(aligned)} cells, sealed {len(sealed)} cells; wrote {out}")
    for regime, v in f1.items():
        print(f'  F1-as-stated {regime}: {v["f1_as_stated_pass"]} '
              f'(step {v["static24_median_pure_step_ms"]:.3f} ms, width {v["static24_median_realised_width"]:.0f})')
    print(f'  F1-mechanism: {f1_mechanism["aligned_plateau_within_frozen_band"]}, '
          f'reproduction diff {f1_mechanism["independent_reproduction_max_abs_diff_ms"]:.3f} ms')
    for regime, v in comparisons.items():
        print(f'  {regime}: F2 {v["f2_waste_reduced"]} F3 {v["f3_tpot_not_worse"]} '
              f'F4 {v["f4_beats_best_static_in_own_engine"]} ({v["feedback_vs_best_static_pct"]:+.2f}%)')


if __name__ == "__main__":
    main()
