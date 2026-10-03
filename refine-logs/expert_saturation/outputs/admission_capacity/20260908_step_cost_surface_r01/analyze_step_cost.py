"""Rebuild the native MoE per-step cost surface from sealed episodes.

Observational post-hoc reconstruction over already-sealed native vLLM captures
(20260906_native_knee_r01). No new GPU execution, no policy comparison, no
counterfactual policy claim. Every step time comes from host timestamps that
were recorded during the original runs:

    step_exec_s[k] = received_s[k] - scheduler_steps[k].start_s

where received_s is the host receipt time of the outputs returned by the same
engine.step() call. Alignment (one scheduler step <-> one output receipt) is
verified per cell and any cell that fails is excluded and reported, never
silently dropped.
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics as st
from collections import defaultdict
from pathlib import Path


def load_cell(path):
    data = json.loads(Path(path).read_text())
    steps = data["scheduler_steps"]
    receipts = sorted({event["received_s"] for event in data["output_events"]})
    aligned = len(steps) == len(receipts) and all(
        receipts[k] > steps[k]["start_s"]
        and (k + 1 >= len(steps) or receipts[k] <= steps[k + 1]["start_s"] + 1e-9)
        for k in range(len(steps))
    )
    rows = []
    if aligned:
        for k, step in enumerate(steps):
            scheduled = step["scheduled"]
            prefill = sum(r["prefill_tokens"] for r in scheduled)
            decode = sum(r["decode_tokens"] for r in scheduled)
            width = step["decode_requests"]
            context = [r["computed_before"] for r in scheduled if r["decode_tokens"] > 0]
            rows.append(dict(
                step=k, exec_ms=(receipts[k] - step["start_s"]) * 1000.0,
                decode_width=width, decode_tokens=decode, prefill_tokens=prefill,
                prefill_requests=sum(r["prefill_tokens"] > 0 for r in scheduled),
                total_tokens=step["total_scheduled_tokens"],
                mean_decode_context=st.mean(context) if context else None,
                target_cap=step["target_cap"], actual_active=step["actual_active"],
                waiting=step["waiting_requests"]))
    return dict(
        path=str(path), regime=data["regime"], target_cap=data["target_cap"],
        policy=data.get("policy", "static"), status=data["status"],
        aligned=aligned, n_steps=len(steps), n_receipts=len(receipts),
        rows=rows, requests=data["requests"])


def quantiles(values):
    ordered = sorted(values)
    n = len(ordered)
    return dict(n=n, p10=ordered[int(0.10 * n)], p50=st.median(ordered),
                p90=ordered[min(n - 1, int(0.90 * n))], mean=st.mean(ordered))


def pure_decode_surface(cells):
    """Pure-decode step cost by decode width, with context-length stratification."""
    pooled = defaultdict(list)
    stratified = defaultdict(list)
    per_cell = defaultdict(lambda: defaultdict(list))
    for cell in cells:
        for row in cell["rows"]:
            if row["prefill_tokens"] or row["decode_width"] <= 0:
                continue
            width = row["decode_width"]
            pooled[width].append(row["exec_ms"])
            per_cell[width][cell["path"]].append(row["exec_ms"])
            context = row["mean_decode_context"]
            if context is not None:
                stratified[(width, int(context // 32) * 32)].append(row["exec_ms"])
    surface = {}
    for width, values in sorted(pooled.items()):
        stats = quantiles(values)
        stats["per_token_ms"] = stats["p50"] / width
        stats["decode_tokens_per_s"] = width / (stats["p50"] / 1000.0)
        cell_medians = [st.median(v) for v in per_cell[width].values() if len(v) >= 5]
        stats["n_cells_ge5"] = len(cell_medians)
        stats["cell_median_min"] = min(cell_medians) if cell_medians else None
        stats["cell_median_max"] = max(cell_medians) if cell_medians else None
        surface[width] = stats
    strata = {f"w{w}_ctx{c}": quantiles(v) for (w, c), v in sorted(stratified.items()) if len(v) >= 20}
    return surface, strata


def paired_prefill_tax(cells, max_distance=40):
    """Within-cell nearest-neighbour pairing of a mixed step to pure steps of
    the same decode width. Local pairing controls for context growth, engine
    identity, cap and arrival regime without pooling across runs."""
    pairs = []
    for cell in cells:
        by_width = defaultdict(list)
        for row in cell["rows"]:
            if not row["prefill_tokens"] and row["decode_width"] > 0:
                by_width[row["decode_width"]].append(row)
        for row in cell["rows"]:
            width = row["decode_width"]
            if not row["prefill_tokens"] or width <= 0 or width not in by_width:
                continue
            near = [c for c in by_width[width] if abs(c["step"] - row["step"]) <= max_distance]
            if len(near) < 3:
                continue
            baseline = st.median(c["exec_ms"] for c in near)
            pairs.append(dict(cell=cell["path"], regime=cell["regime"], cap=cell["target_cap"],
                              policy=cell["policy"], step=row["step"], decode_width=width,
                              prefill_tokens=row["prefill_tokens"],
                              prefill_requests=row["prefill_requests"],
                              mixed_ms=row["exec_ms"], pure_ms=baseline,
                              tax_ms=row["exec_ms"] - baseline, n_baseline=len(near)))
    return pairs


def summarize_tax(pairs, key, buckets):
    out = {}
    for label, predicate in buckets:
        selected = [p for p in pairs if predicate(p)]
        if len(selected) < 5:
            continue
        taxes = [p["tax_ms"] for p in selected]
        out[label] = dict(quantiles(taxes),
                          median_mixed_ms=st.median(p["mixed_ms"] for p in selected),
                          median_pure_ms=st.median(p["pure_ms"] for p in selected),
                          median_prefill_tokens=st.median(p["prefill_tokens"] for p in selected),
                          median_decode_width=st.median(p["decode_width"] for p in selected),
                          tax_per_prefill_token_ms=st.median(
                              p["tax_ms"] / p["prefill_tokens"] for p in selected),
                          tax_per_decode_request_ms=st.median(
                              p["tax_ms"] for p in selected))
        out[label]["key"] = key
    return out


def episode_itl_decomposition(cells):
    """How much of each episode's decode wall time sits in mixed steps."""
    out = []
    for cell in cells:
        pure = [r for r in cell["rows"] if not r["prefill_tokens"] and r["decode_width"] > 0]
        mixed = [r for r in cell["rows"] if r["prefill_tokens"] and r["decode_width"] > 0]
        prefill_only = [r for r in cell["rows"] if r["prefill_tokens"] and r["decode_width"] == 0]
        if not pure:
            continue
        pure_ms = sum(r["exec_ms"] for r in pure)
        mixed_ms = sum(r["exec_ms"] for r in mixed)
        # decode-request-weighted exposure: how many request-steps each class serves
        pure_req_steps = sum(r["decode_width"] for r in pure)
        mixed_req_steps = sum(r["decode_width"] for r in mixed)
        out.append(dict(
            cell=cell["path"], regime=cell["regime"], cap=cell["target_cap"], policy=cell["policy"],
            n_pure=len(pure), n_mixed=len(mixed), n_prefill_only=len(prefill_only),
            pure_ms=pure_ms, mixed_ms=mixed_ms,
            mixed_share_of_decode_time=mixed_ms / (pure_ms + mixed_ms),
            mixed_share_of_steps=len(mixed) / (len(pure) + len(mixed)),
            pure_req_steps=pure_req_steps, mixed_req_steps=mixed_req_steps,
            median_pure_ms=st.median(r["exec_ms"] for r in pure),
            median_mixed_ms=st.median(r["exec_ms"] for r in mixed) if mixed else None,
            mean_itl_all_ms=(pure_ms + mixed_ms) / max(1, len(pure) + len(mixed)),
            request_mean_itl_ms=(
                sum(r["exec_ms"] * r["decode_width"] for r in pure + mixed)
                / max(1, pure_req_steps + mixed_req_steps))))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=False)

    paths = []
    for root in args.results_dir:
        paths.extend(sorted(glob.glob(str(Path(root) / "**" / "cell-*.json"), recursive=True)))
    cells = [load_cell(p) for p in paths]
    excluded = [dict(path=c["path"], reason="alignment" if not c["aligned"] else c["status"])
                for c in cells if not c["aligned"] or c["status"] != "COMPLETE"]
    usable = [c for c in cells if c["aligned"] and c["status"] == "COMPLETE"]

    surface, strata = pure_decode_surface(usable)
    pairs = paired_prefill_tax(usable)
    width_buckets = [(f"width_{lo}_{hi}", (lambda p, lo=lo, hi=hi: lo <= p["decode_width"] <= hi))
                     for lo, hi in [(1, 2), (3, 6), (7, 10), (11, 14), (15, 18), (19, 22), (23, 28)]]
    chunk_buckets = [(f"prefill_{lo}_{hi}", (lambda p, lo=lo, hi=hi: lo <= p["prefill_tokens"] <= hi))
                     for lo, hi in [(1, 64), (65, 200), (201, 600), (601, 1100)]]
    regime_buckets = [(r, (lambda p, r=r: p["regime"] == r)) for r in ("steady", "bursty")]

    report = dict(
        evidence_type="OBSERVATIONAL_POSTHOC_RECONSTRUCTION_OF_SEALED_NATIVE_CAPTURES",
        source_dirs=args.results_dir, n_cells_found=len(cells), n_cells_used=len(usable),
        excluded=excluded,
        cells=[dict(path=c["path"], regime=c["regime"], cap=c["target_cap"], policy=c["policy"],
                    n_steps=c["n_steps"]) for c in usable],
        pure_decode_surface={str(k): v for k, v in surface.items()},
        pure_decode_context_strata=strata,
        prefill_tax_by_width=summarize_tax(pairs, "decode_width", width_buckets),
        prefill_tax_by_chunk=summarize_tax(pairs, "prefill_tokens", chunk_buckets),
        prefill_tax_by_regime=summarize_tax(pairs, "regime", regime_buckets),
        n_paired_mixed_steps=len(pairs),
        episodes=episode_itl_decomposition(usable))
    (out_dir / "step_cost_surface.json").write_text(json.dumps(report, indent=2) + "\n")

    lines = ["# Native MoE step cost surface (observational reconstruction)", "",
             f"cells used {len(usable)}/{len(cells)}; excluded {len(excluded)}",
             f"paired mixed steps {len(pairs)}", "",
             "## Pure-decode step cost by decode width", "",
             "| width | n | p50 ms | p10 | p90 | ms/token | decode tok/s | cells | cell p50 min | max |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for width, s in surface.items():
        if s["n"] < 20:
            continue
        lo = f'{s["cell_median_min"]:.3f}' if s["cell_median_min"] is not None else "-"
        hi = f'{s["cell_median_max"]:.3f}' if s["cell_median_max"] is not None else "-"
        lines.append(f'| {width} | {s["n"]} | {s["p50"]:.3f} | {s["p10"]:.3f} | {s["p90"]:.3f} | '
                     f'{s["per_token_ms"]:.4f} | {s["decode_tokens_per_s"]:.0f} | {s["n_cells_ge5"]} | {lo} | {hi} |')
    for title, block in (("Prefill tax by decode width", report["prefill_tax_by_width"]),
                         ("Prefill tax by chunk size", report["prefill_tax_by_chunk"]),
                         ("Prefill tax by arrival regime", report["prefill_tax_by_regime"])):
        lines += ["", f"## {title}", "",
                  "| bucket | n | median pure ms | median mixed ms | median tax ms | tax/prefill-token ms |",
                  "|---|---:|---:|---:|---:|---:|"]
        for label, s in block.items():
            lines.append(f'| {label} | {s["n"]} | {s["median_pure_ms"]:.3f} | {s["median_mixed_ms"]:.3f} | '
                         f'{s["p50"]:.3f} | {s["tax_per_prefill_token_ms"]:.5f} |')
    lines += ["", "## Per-episode mixed-step exposure", "",
              "| cell | regime | cap | policy | pure steps | mixed steps | median pure ms | median mixed ms | mixed share of decode time | request-weighted mean ITL ms |",
              "|---|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for e in report["episodes"]:
        mixed_ms = f'{e["median_mixed_ms"]:.3f}' if e["median_mixed_ms"] is not None else "-"
        lines.append(f'| {Path(e["cell"]).parent.name}/{Path(e["cell"]).name} | {e["regime"]} | {e["cap"]} | '
                     f'{e["policy"]} | {e["n_pure"]} | {e["n_mixed"]} | {e["median_pure_ms"]:.3f} | {mixed_ms} | '
                     f'{e["mixed_share_of_decode_time"]:.4f} | {e["request_mean_itl_ms"]:.3f} |')
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")
    print(f"cells used {len(usable)}/{len(cells)}, paired mixed steps {len(pairs)}")
    print(f"wrote {out_dir/'step_cost_surface.json'} and {out_dir/'report.md'}")


if __name__ == "__main__":
    main()
