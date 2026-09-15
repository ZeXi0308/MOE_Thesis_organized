"""Quantify the target-versus-realised decode width mismatch.

The aligned-ladder run falsified the mechanism assumption behind it. The frozen
hypothesis was: put the ladder rungs on CUDA-graph capture points, and the batch
will execute at a capture point, removing padding waste. The run shows the rungs
are not what determines the executed width.

Reason: a non-preemptive admission cap only gates *new* admissions. Already
running requests keep decoding, and the realised width is set by the balance of
arrivals, completions and the cap, so it visits every integer on its way and
settles wherever that balance puts it -- not on the rung. Choosing rung values is
therefore control over an upper bound, not over the executed graph width.

This script measures that mismatch directly from the sealed and aligned runs:
for every scheduler step, how far the realised decode width sits from the target
cap and from the nearest capture point, and how much of the paid graph width was
padding. It also isolates the steps where the target itself *was* a capture point
yet the realised width was not.
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

CAPTURE_SIZES = (1, 2, 4, 8, 16, 24, 32)


def ceil_capture(width):
    for size in CAPTURE_SIZES:
        if width <= size:
            return size
    return CAPTURE_SIZES[-1]


def analyse(path):
    data = json.loads(Path(path).read_text())
    steps = data["scheduler_steps"]
    receipts = sorted({e["received_s"] for e in data["output_events"]})
    if len(steps) != len(receipts) or data["status"] != "COMPLETE":
        return None
    pure = []
    for k, step in enumerate(steps):
        if sum(r["prefill_tokens"] for r in step["scheduled"]) or step["decode_requests"] <= 0:
            continue
        width = step["decode_requests"]
        pure.append(dict(width=width, target=step["target_cap"],
                         exec_ms=(receipts[k] - step["start_s"]) * 1000.0,
                         padded=ceil_capture(width),
                         waste=1 - width / ceil_capture(width),
                         at_capture_point=width in CAPTURE_SIZES,
                         target_at_capture_point=step["target_cap"] in CAPTURE_SIZES,
                         gap_to_target=step["target_cap"] - width))
    if not pure:
        return None
    on_cp_target = [r for r in pure if r["target_at_capture_point"]]
    return dict(
        path=str(path), regime=data["regime"], policy=data.get("policy", "static"),
        cap=data["target_cap"], n_pure=len(pure),
        median_width=st.median(r["width"] for r in pure),
        median_target=st.median(r["target"] for r in pure),
        median_gap_to_target=st.median(r["gap_to_target"] for r in pure),
        mean_waste=st.mean(r["waste"] for r in pure),
        share_width_at_capture_point=st.mean(r["at_capture_point"] for r in pure),
        share_target_at_capture_point=st.mean(r["target_at_capture_point"] for r in pure),
        # The load-bearing number: target was aligned, width was not.
        share_target_aligned_but_width_not=(
            st.mean(not r["at_capture_point"] for r in on_cp_target) if on_cp_target else None),
        n_distinct_widths=len({r["width"] for r in pure}),
        width_histogram=dict(sorted(Counter(r["width"] for r in pure).items())),
        applied_targets=[a["target_cap"] for a in data.get("actions", [])],
        wasted_graph_slots=sum(r["padded"] - r["width"] for r in pure),
        paid_graph_slots=sum(r["padded"] for r in pure))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True,
                        help="label=glob, e.g. aligned=next_experiment/gpu_results")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)

    groups = {}
    for spec in args.run:
        label, root = spec.split("=", 1)
        cells = []
        for path in sorted(glob.glob(str(Path(root) / "**" / "cell-*.json"), recursive=True)):
            row = analyse(path)
            if row:
                cells.append(row)
        groups[label] = cells

    summary = {}
    for label, cells in groups.items():
        by_policy = defaultdict(list)
        for cell in cells:
            by_policy[(cell["regime"], cell["policy"])].append(cell)
        summary[label] = {
            f"{regime}_{policy}": dict(
                n_episodes=len(rows),
                median_realised_width=st.median([r["median_width"] for r in rows]),
                median_target=st.median([r["median_target"] for r in rows]),
                median_gap_to_target=st.median([r["median_gap_to_target"] for r in rows]),
                mean_waste=st.mean([r["mean_waste"] for r in rows]),
                share_width_at_capture_point=st.mean([r["share_width_at_capture_point"] for r in rows]),
                share_target_at_capture_point=st.mean([r["share_target_at_capture_point"] for r in rows]),
                share_target_aligned_but_width_not=st.mean(
                    [r["share_target_aligned_but_width_not"] for r in rows
                     if r["share_target_aligned_but_width_not"] is not None] or [float("nan")]),
                median_distinct_widths=st.median([r["n_distinct_widths"] for r in rows]),
                wasted_slot_fraction=(sum(r["wasted_graph_slots"] for r in rows)
                                      / max(1, sum(r["paid_graph_slots"] for r in rows))))
            for (regime, policy), rows in sorted(by_policy.items())}

    report = dict(
        evidence_type="NATIVE_VLLM_INPROCESS_STEP_LEVEL_RECONSTRUCTION",
        finding=("a non-preemptive admission cap bounds but does not set the executed decode "
                 "width; placing ladder rungs on capture points does not place the batch on "
                 "capture points"),
        capture_sizes=list(CAPTURE_SIZES), summary=summary,
        cells={label: cells for label, cells in groups.items()})
    (out / "mismatch.json").write_text(json.dumps(report, indent=2) + "\n")

    lines = ["# Target cap versus realised decode width", "",
             "`share_target_aligned_but_width_not` is the load-bearing column: the fraction of",
             "steps where the admission target *was* a capture point while the executed width",
             "was not. If placing rungs on capture points worked, it would be near zero.", ""]
    for label, block in summary.items():
        lines += [f"## {label}", "",
                  "| arm | episodes | median target | median realised width | median gap | "
                  "width on capture point | target on capture point | **target aligned but width not** | "
                  "distinct widths | wasted graph slots |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for arm, v in block.items():
            lines.append(
                f'| {arm} | {v["n_episodes"]} | {v["median_target"]:.0f} | '
                f'{v["median_realised_width"]:.0f} | {v["median_gap_to_target"]:.0f} | '
                f'{v["share_width_at_capture_point"]:.3f} | {v["share_target_at_capture_point"]:.3f} | '
                f'**{v["share_target_aligned_but_width_not"]:.3f}** | '
                f'{v["median_distinct_widths"]:.0f} | {v["wasted_slot_fraction"]:.3f} |')
        lines.append("")
    for label, cells in groups.items():
        feedback = [c for c in cells if c["policy"] == "feedback"]
        if not feedback:
            continue
        lines += [f"## {label}: feedback episodes in detail", ""]
        for c in feedback:
            lines.append(f'- `{Path(c["path"]).parent.name}/{Path(c["path"]).name}` {c["regime"]}: '
                         f'applied targets {c["applied_targets"]}, median width {c["median_width"]:.0f}, '
                         f'waste {c["mean_waste"]:.4f}, widths visited {c["n_distinct_widths"]}')
        lines.append("")
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")
    for label, block in summary.items():
        for arm, v in block.items():
            if "feedback" in arm or "static" in arm:
                print(f'  {label:8s} {arm:18s} target {v["median_target"]:5.0f} -> width '
                      f'{v["median_realised_width"]:5.0f}  aligned-target-but-not-width '
                      f'{v["share_target_aligned_but_width_not"]:.3f}')


if __name__ == "__main__":
    main()
