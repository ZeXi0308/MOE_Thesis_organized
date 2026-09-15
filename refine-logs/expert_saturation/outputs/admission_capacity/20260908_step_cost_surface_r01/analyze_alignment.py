"""Test the capture-alignment prediction against the already-measured goodput.

The reconstruction says pure-decode step cost is flat inside a capture bucket and
jumps across one. That yields a falsifiable prediction about the *static* caps
already measured, with no new execution:

  P1  Two caps inside the same bucket must have near-equal per-step cost, so the
      larger one serves more requests for the same money. Concretely cap12 and
      cap16 both live in bucket 16, so their median TPOT should be close while
      cap16's goodput should be higher.
  P2  A cap sitting exactly on a capture point should show near-zero padding
      waste; a cap strictly inside a bucket should show large waste.
  P3  Ranking caps by mean padding waste should track the ranking by goodput
      better in bursty (where widths are pinned by the arrival groups) than a
      ranking by cap value alone.

P1/P2 are the load-bearing ones. This script only recomputes from sealed data; a
failure here falsifies the staircase interpretation rather than the policy.
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

CAPTURE_SIZES = (1, 2, 4, 8, 16, 24, 32)


def ceil_capture(width):
    for size in CAPTURE_SIZES:
        if width <= size:
            return size
    return CAPTURE_SIZES[-1]


def load(path):
    data = json.loads(Path(path).read_text())
    steps = data["scheduler_steps"]
    receipts = sorted({e["received_s"] for e in data["output_events"]})
    if len(steps) != len(receipts) or data["status"] != "COMPLETE":
        return None
    widths, pure_ms = [], []
    for k, step in enumerate(steps):
        prefill = sum(r["prefill_tokens"] for r in step["scheduled"])
        width = step["decode_requests"]
        if prefill or width <= 0:
            continue
        widths.append(width)
        pure_ms.append((receipts[k] - step["start_s"]) * 1000.0)
    done = [r for r in data["requests"] if r["status"] == "completed" and len(r["token_times_s"]) > 1]
    if not widths or not done:
        return None
    tpots = [(r["token_times_s"][-1] - r["token_times_s"][0]) * 1000.0
             / (len(r["token_times_s"]) - 1) for r in done]
    ttfts = [r["token_times_s"][0] - r["arrival_s"] for r in done]
    return dict(
        path=str(path), regime=data["regime"], cap=data["target_cap"],
        policy=data.get("policy", "static"), wall_s=data["observation_end_s"],
        n_completed=len(done), median_tpot_ms=st.median(tpots), median_ttft_s=st.median(ttfts),
        median_width=st.median(widths), max_width=max(widths),
        mean_padding_waste=st.mean(1 - w / ceil_capture(w) for w in widths),
        median_pure_step_ms=st.median(pure_ms),
        cap_is_capture_point=data["target_cap"] in CAPTURE_SIZES,
        cap_bucket=ceil_capture(data["target_cap"]),
        median_width_bucket=ceil_capture(int(st.median(widths))),
        tpots=tpots, ttfts=ttfts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ttft-slo-s", type=float, default=0.20)
    parser.add_argument("--tpot-slo-s", type=float, default=0.009)
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=False)

    cells = []
    for root in args.results_dir:
        for path in sorted(glob.glob(str(Path(root) / "**" / "cell-*.json"), recursive=True)):
            cell = load(path)
            if cell is None:
                continue
            joint = sum(1 for t, f in zip(cell["tpots"], cell["ttfts"])
                        if t <= args.tpot_slo_s * 1000 and f <= args.ttft_slo_s)
            cell["joint_pass"] = joint
            cell["goodput_req_per_s"] = joint / cell["wall_s"]
            cell["throughput_req_per_s"] = cell["n_completed"] / cell["wall_s"]
            cells.append(cell)

    # Only static arms, main arrival load (bursty/steady both finish inside ~1.55s).
    static = [c for c in cells if c["policy"] == "static" and c["median_tpot_ms"] > 4.0]
    by_key = defaultdict(list)
    for c in static:
        by_key[(c["regime"], c["cap"])].append(c)

    caps = {}
    for (regime, cap), group in sorted(by_key.items()):
        caps[f"{regime}_cap{cap}"] = dict(
            regime=regime, cap=cap, n_episodes=len(group),
            cap_is_capture_point=group[0]["cap_is_capture_point"],
            cap_bucket=group[0]["cap_bucket"],
            median_tpot_ms=st.median([c["median_tpot_ms"] for c in group]),
            median_pure_step_ms=st.median([c["median_pure_step_ms"] for c in group]),
            median_width=st.median([c["median_width"] for c in group]),
            mean_padding_waste=st.mean([c["mean_padding_waste"] for c in group]),
            median_ttft_ms=st.median([c["median_ttft_s"] for c in group]) * 1000,
            goodput=[round(c["goodput_req_per_s"], 4) for c in group],
            median_goodput=st.median([c["goodput_req_per_s"] for c in group]),
            median_throughput=st.median([c["throughput_req_per_s"] for c in group]))

    # P1: cap12 vs cap16, both inside bucket 16.
    p1 = {}
    for regime in ("steady", "bursty"):
        a, b = caps.get(f"{regime}_cap12"), caps.get(f"{regime}_cap16")
        if not a or not b:
            continue
        p1[regime] = dict(
            same_bucket=a["cap_bucket"] == b["cap_bucket"], bucket=a["cap_bucket"],
            cap12_median_pure_step_ms=a["median_pure_step_ms"],
            cap16_median_pure_step_ms=b["median_pure_step_ms"],
            step_cost_ratio=b["median_pure_step_ms"] / a["median_pure_step_ms"],
            cap12_median_tpot_ms=a["median_tpot_ms"], cap16_median_tpot_ms=b["median_tpot_ms"],
            tpot_ratio=b["median_tpot_ms"] / a["median_tpot_ms"],
            cap12_padding_waste=a["mean_padding_waste"], cap16_padding_waste=b["mean_padding_waste"],
            cap12_goodput=a["median_goodput"], cap16_goodput=b["median_goodput"],
            goodput_gain_pct=100 * (b["median_goodput"] / a["median_goodput"] - 1),
            prediction_step_cost_within_5pct=abs(
                b["median_pure_step_ms"] / a["median_pure_step_ms"] - 1) <= 0.05,
            prediction_cap16_goodput_higher=b["median_goodput"] > a["median_goodput"])

    # P2/P3: waste versus goodput ranking inside each regime.
    ranking = {}
    for regime in ("steady", "bursty"):
        rows = [v for k, v in caps.items() if v["regime"] == regime]
        if len(rows) < 3:
            continue
        by_waste = sorted(rows, key=lambda r: r["mean_padding_waste"])
        by_goodput = sorted(rows, key=lambda r: -r["median_goodput"])
        ranking[regime] = dict(
            caps_by_ascending_padding_waste=[(r["cap"], round(r["mean_padding_waste"], 3))
                                             for r in by_waste],
            caps_by_descending_goodput=[(r["cap"], round(r["median_goodput"], 3))
                                        for r in by_goodput],
            worst_waste_cap=by_waste[-1]["cap"], best_goodput_cap=by_goodput[0]["cap"],
            worst_waste_cap_is_capture_point=by_waste[-1]["cap_is_capture_point"])

    report = dict(
        evidence_type="OBSERVATIONAL_POSTHOC_RECONSTRUCTION_OF_SEALED_NATIVE_CAPTURES",
        claim_boundary=("recomputed from sealed episodes; confirms or falsifies the staircase "
                        "reading of already-measured static caps; not a policy result"),
        capture_sizes=list(CAPTURE_SIZES), slo=dict(ttft_s=args.ttft_slo_s, tpot_s=args.tpot_slo_s),
        n_cells=len(cells), n_static_main_load=len(static),
        per_cap=caps, p1_same_bucket_caps=p1, p2_p3_ranking=ranking)
    (out_dir / "alignment.json").write_text(json.dumps(report, indent=2) + "\n")

    lines = ["# Capture-alignment prediction test (sealed data, no execution)", "",
             "## Static caps under the main arrival load", "",
             "| regime | cap | on capture point | bucket | median width | mean padding waste | "
             "median pure step ms | median TPOT ms | median TTFT ms | goodput per episode |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for key, v in caps.items():
        lines.append(f'| {v["regime"]} | {v["cap"]} | {v["cap_is_capture_point"]} | {v["cap_bucket"]} '
                     f'| {v["median_width"]:.0f} | {v["mean_padding_waste"]:.3f} | '
                     f'{v["median_pure_step_ms"]:.3f} | {v["median_tpot_ms"]:.3f} | '
                     f'{v["median_ttft_ms"]:.1f} | {v["goodput"]} |')
    lines += ["", "## P1: two caps inside the same bucket", ""]
    for regime, v in p1.items():
        lines += [f'### {regime}', "",
                  f'- same bucket: {v["same_bucket"]} (bucket {v["bucket"]})',
                  f'- pure step cost: cap12 {v["cap12_median_pure_step_ms"]:.3f} ms vs '
                  f'cap16 {v["cap16_median_pure_step_ms"]:.3f} ms, ratio {v["step_cost_ratio"]:.4f}',
                  f'- request TPOT: cap12 {v["cap12_median_tpot_ms"]:.3f} ms vs '
                  f'cap16 {v["cap16_median_tpot_ms"]:.3f} ms, ratio {v["tpot_ratio"]:.4f}',
                  f'- padding waste: cap12 {v["cap12_padding_waste"]:.3f} vs '
                  f'cap16 {v["cap16_padding_waste"]:.3f}',
                  f'- goodput: cap12 {v["cap12_goodput"]:.3f} vs cap16 {v["cap16_goodput"]:.3f} '
                  f'({v["goodput_gain_pct"]:+.2f}%)',
                  f'- **step cost equal within 5%: {v["prediction_step_cost_within_5pct"]}**',
                  f'- **cap16 goodput higher: {v["prediction_cap16_goodput_higher"]}**', ""]
    lines += ["## P2/P3: padding waste versus goodput ranking", ""]
    for regime, v in ranking.items():
        lines += [f'### {regime}', "",
                  f'- caps by ascending padding waste: {v["caps_by_ascending_padding_waste"]}',
                  f'- caps by descending goodput: {v["caps_by_descending_goodput"]}',
                  f'- worst-waste cap {v["worst_waste_cap"]} '
                  f'(on capture point: {v["worst_waste_cap_is_capture_point"]}); '
                  f'best-goodput cap {v["best_goodput_cap"]}', ""]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")
    print(f"cells {len(cells)}, static main-load {len(static)}; wrote {out_dir}")
    for regime, v in p1.items():
        print(f'  P1 {regime}: step ratio {v["step_cost_ratio"]:.4f} '
              f'(within 5%: {v["prediction_step_cost_within_5pct"]}), '
              f'goodput {v["goodput_gain_pct"]:+.2f}%')


if __name__ == "__main__":
    main()
