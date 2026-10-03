"""Stage-2: attribute request TPOT to decode-width buckets and admission interference.

Observational reconstruction over the sealed native captures. Adds three things
the stage-1 surface did not resolve:

1. CUDA-graph capture buckets. `enforce_eager=false`, so a decode step of width w
   executes a graph captured at ceil_capture(w). Cost is therefore expected to be
   a staircase, and any width strictly inside a bucket pays the bucket price.

2. Within-cell, width-matched prefill tax decomposed into a fixed per-interleave
   component and a per-prefill-token component by OLS on matched pairs. This is
   the quantity that decides whether coalescing admissions can pay.

3. Per-request TPOT decomposition. For a non-preemptive engine every running
   request advances one token per step, so a request's mean TPOT is exactly the
   mean execution time of the contiguous step range it decodes in. Splitting that
   range into pure and mixed steps attributes each request's TPOT to steady-state
   decode cost versus admission interference, and shows which SLO failures are
   interference-caused.

All numbers are reconstructions of already-recorded host timestamps. No new
execution, no counterfactual run, no policy claim.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

# vLLM V1 default piecewise cudagraph capture sizes below/at max_num_seqs=32.
# `verify_capture_sizes` cross-checks this against what the engine itself logged,
# so the staircase claim never rests on an assumed bucket geometry.
CAPTURE_SIZES = (1, 2, 4, 8, 16, 24, 32)


def verify_capture_sizes(source_dirs, assumed=CAPTURE_SIZES):
    """Recover the engine's own capture-size list from the retained console logs.

    The staircase interpretation is only valid if the bucket edges used here are
    the edges the engine actually captured. vLLM logs the full list plus the
    number of decode graphs it captured; both are checked."""
    found, decode_graph_counts = set(), set()
    logs = []
    for root in source_dirs:
        logs.extend(glob.glob(str(Path(root).parent / "*.console.log")))
        logs.extend(glob.glob(str(Path(root) / "*.console.log")))
    for path in sorted(set(logs)):
        text = Path(path).read_text(errors="replace")
        for match in re.finditer(r"cudagraph_capture_sizes'?\s*:\s*\[([0-9,\s]+)\]", text):
            found.add(tuple(int(v) for v in match.group(1).replace(" ", "").split(",") if v))
        for match in re.finditer(r"Capturing CUDA graphs \(decode, FULL\):\s*100%\|[^|]*\|\s*(\d+)/(\d+)", text):
            decode_graph_counts.add(int(match.group(2)))
    logged = sorted(found)
    max_seqs_subset = {tuple(s for s in sizes if s <= max(assumed)) for sizes in logged}
    return dict(
        n_logs_scanned=len(set(logs)), logged_capture_size_lists=[list(s) for s in logged],
        decode_full_graph_counts=sorted(decode_graph_counts),
        assumed=list(assumed),
        assumed_matches_logged_subset=bool(max_seqs_subset) and max_seqs_subset == {tuple(assumed)},
        decode_graph_count_matches_assumed=(
            decode_graph_counts == {len(assumed)} if decode_graph_counts else None))



def capture_bucket(width):
    for size in CAPTURE_SIZES:
        if width <= size:
            return size
    return CAPTURE_SIZES[-1]


def ols(xs, ys):
    """Two-parameter least squares: y = intercept + slope * x."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return None
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    intercept = my - slope * mx
    ss_res = sum((y - intercept - slope * x) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    return dict(n=n, intercept=intercept, slope=slope,
                r2=1.0 - ss_res / ss_tot if ss_tot > 0 else None)


def load_cell(path):
    data = json.loads(Path(path).read_text())
    steps = data["scheduler_steps"]
    receipts = sorted({event["received_s"] for event in data["output_events"]})
    if len(steps) != len(receipts):
        return dict(path=str(path), aligned=False, status=data["status"])
    for k, step in enumerate(steps):
        if not (receipts[k] > step["start_s"]
                and (k + 1 >= len(steps) or receipts[k] <= steps[k + 1]["start_s"] + 1e-9)):
            return dict(path=str(path), aligned=False, status=data["status"])
    rows = []
    for k, step in enumerate(steps):
        scheduled = step["scheduled"]
        rows.append(dict(
            step=k, received_s=receipts[k], exec_ms=(receipts[k] - step["start_s"]) * 1000.0,
            decode_width=step["decode_requests"],
            prefill_tokens=sum(r["prefill_tokens"] for r in scheduled),
            prefill_requests=sum(r["prefill_tokens"] > 0 for r in scheduled),
            decode_ids=[r["request_id"] for r in scheduled if r["decode_tokens"] > 0],
            first_token_ids=[r["request_id"] for r in scheduled
                             if r["prefill_tokens"] > 0
                             and r["computed_after"] >= r["prompt_tokens"]]))
    return dict(path=str(path), aligned=True, status=data["status"], regime=data["regime"],
                target_cap=data["target_cap"], policy=data.get("policy", "static"),
                rows=rows, requests=data["requests"],
                observation_end_s=data["observation_end_s"])


def bucket_analysis(cells):
    pooled = defaultdict(list)
    for cell in cells:
        for row in cell["rows"]:
            if row["prefill_tokens"] or row["decode_width"] <= 0:
                continue
            pooled[row["decode_width"]].append(row["exec_ms"])
    by_width = {w: st.median(v) for w, v in pooled.items() if len(v) >= 20}
    buckets = defaultdict(dict)
    for width, med in sorted(by_width.items()):
        buckets[capture_bucket(width)][width] = med
    out = {}
    for size, members in sorted(buckets.items()):
        values = list(members.values())
        out[str(size)] = dict(
            widths=sorted(members), n_widths=len(members),
            median_ms_min=min(values), median_ms_max=max(values),
            within_bucket_spread_pct=100.0 * (max(values) - min(values)) / min(values),
            at_capture_point_ms=members.get(size))
    ordered = sorted(out)
    steps_across = []
    for a, b in zip(ordered, ordered[1:]):
        lo, hi = out[a]["at_capture_point_ms"], out[b]["at_capture_point_ms"]
        if lo and hi:
            steps_across.append(dict(from_width=int(a), to_width=int(b), delta_ms=hi - lo,
                                     jump_pct=100.0 * (hi - lo) / lo,
                                     ms_per_added_request=(hi - lo) / (int(b) - int(a))))
    per_request = {str(w): dict(step_ms=m, ms_per_request=m / w,
                                bucket=capture_bucket(w),
                                padded_waste_pct=100.0 * (1 - w / capture_bucket(w)))
                   for w, m in sorted(by_width.items())}
    return dict(buckets=out, across_bucket_steps=steps_across, per_request_cost=per_request)


def bucket_occupancy(cells):
    """Where each episode actually ran inside the capture staircase.

    `padding_waste` is the fraction of the paid graph width that carried no
    request: 1 - width / ceil_capture(width), averaged over pure-decode steps.
    It is a property of the admission pattern, not of the engine limit."""
    out = []
    for cell in cells:
        widths = [r["decode_width"] for r in cell["rows"]
                  if not r["prefill_tokens"] and r["decode_width"] > 0]
        if not widths:
            continue
        share = defaultdict(int)
        for w in widths:
            share[capture_bucket(w)] += 1
        out.append(dict(
            cell=cell["path"], regime=cell["regime"], cap=cell["target_cap"],
            policy=cell["policy"], n_pure_steps=len(widths),
            max_width=max(widths), median_width=st.median(widths),
            n_admission_events=sum(1 for r in cell["rows"] if r["prefill_requests"]),
            bucket_share={str(k): v / len(widths) for k, v in sorted(share.items())},
            mean_padding_waste=st.mean(1 - w / capture_bucket(w) for w in widths),
            share_of_steps_at_a_capture_point=sum(
                w == capture_bucket(w) for w in widths) / len(widths)))
    return out


def matched_prefill_tax(cells, max_distance=40):
    pairs = []
    for cell in cells:
        by_width = defaultdict(list)
        for row in cell["rows"]:
            if not row["prefill_tokens"] and row["decode_width"] > 0:
                by_width[row["decode_width"]].append(row)
        for row in cell["rows"]:
            width = row["decode_width"]
            if not row["prefill_tokens"] or width <= 0:
                continue
            near = [c for c in by_width.get(width, [])
                    if abs(c["step"] - row["step"]) <= max_distance]
            if len(near) < 3:
                continue
            pairs.append(dict(cell=cell["path"], regime=cell["regime"], cap=cell["target_cap"],
                              step=row["step"], decode_width=width,
                              prefill_tokens=row["prefill_tokens"],
                              prefill_requests=row["prefill_requests"],
                              mixed_ms=row["exec_ms"],
                              pure_ms=st.median(c["exec_ms"] for c in near),
                              tax_ms=row["exec_ms"] - st.median(c["exec_ms"] for c in near)))
    fit_all = ols([p["prefill_tokens"] for p in pairs], [p["tax_ms"] for p in pairs])
    wide = [p for p in pairs if p["decode_width"] >= 7]
    fit_wide = ols([p["prefill_tokens"] for p in wide], [p["tax_ms"] for p in wide])
    fit_width = ols([p["decode_width"] for p in pairs if 100 <= p["prefill_tokens"] <= 200],
                    [p["tax_ms"] for p in pairs if 100 <= p["prefill_tokens"] <= 200])
    by_chunk = defaultdict(list)
    for p in pairs:
        by_chunk[p["prefill_tokens"]].append(p["tax_ms"])
    chunk_table = {str(k): dict(n=len(v), median_tax_ms=st.median(v),
                                tax_per_token_ms=st.median(v) / k)
                   for k, v in sorted(by_chunk.items()) if len(v) >= 5}
    return dict(n_pairs=len(pairs), fit_tax_vs_prefill_tokens_all=fit_all,
                fit_tax_vs_prefill_tokens_width_ge7=fit_wide,
                fit_tax_vs_decode_width_at_chunk_128=fit_width,
                tax_by_exact_chunk_size=chunk_table, pairs=pairs)


def request_tpot_decomposition(cells, tax_lookup, ttft_slo_s, tpot_slo_s):
    out = []
    for cell in cells:
        rows = cell["rows"]
        pure_by_width = defaultdict(list)
        for row in rows:
            if not row["prefill_tokens"] and row["decode_width"] > 0:
                pure_by_width[row["decode_width"]].append(row["exec_ms"])
        decode_steps = defaultdict(list)
        for row in rows:
            for rid in row["decode_ids"]:
                decode_steps[rid].append(row)
        summary = []
        for request in cell["requests"]:
            rid = request["request_id"]
            times = request["token_times_s"]
            if request["status"] != "completed" or len(times) < 2:
                summary.append(dict(request_id=rid, status=request["status"], completed=False))
                continue
            steps = decode_steps.get(rid, [])
            span = [s for s in steps if s["received_s"] > times[0] + 1e-12]
            total_ms = sum(s["exec_ms"] for s in span)
            mixed = [s for s in span if s["prefill_tokens"]]
            interference_ms = 0.0
            for s in mixed:
                local = pure_by_width.get(s["decode_width"])
                baseline = st.median(local) if local and len(local) >= 3 else \
                    tax_lookup.get(s["decode_width"])
                if baseline is not None:
                    interference_ms += max(0.0, s["exec_ms"] - baseline)
            n_intervals = len(times) - 1
            measured_tpot_ms = (times[-1] - times[0]) * 1000.0 / n_intervals
            ttft_s = times[0] - request["arrival_s"]
            summary.append(dict(
                request_id=rid, completed=True, n_intervals=n_intervals,
                measured_tpot_ms=measured_tpot_ms,
                reconstructed_tpot_ms=total_ms / n_intervals if n_intervals else None,
                n_span_steps=len(span), n_mixed_steps=len(mixed),
                interference_ms=interference_ms,
                interference_share=interference_ms / total_ms if total_ms else 0.0,
                tpot_without_interference_ms=(total_ms - interference_ms) / n_intervals,
                ttft_s=ttft_s,
                ttft_pass=ttft_s <= ttft_slo_s,
                tpot_pass=measured_tpot_ms <= tpot_slo_s * 1000.0,
                tpot_pass_without_interference=(total_ms - interference_ms) / n_intervals
                <= tpot_slo_s * 1000.0))
        done = [r for r in summary if r.get("completed")]
        joint = [r for r in done if r["ttft_pass"] and r["tpot_pass"]]
        joint_no_int = [r for r in done if r["ttft_pass"] and r["tpot_pass_without_interference"]]
        rescued = [r for r in done if not r["tpot_pass"] and r["tpot_pass_without_interference"]]
        residual = [abs(r["measured_tpot_ms"] - r["reconstructed_tpot_ms"]) for r in done]
        out.append(dict(
            cell=cell["path"], regime=cell["regime"], cap=cell["target_cap"], policy=cell["policy"],
            n_requests=len(summary), n_completed=len(done),
            wall_s=cell["observation_end_s"],
            median_tpot_ms=st.median(r["measured_tpot_ms"] for r in done) if done else None,
            median_interference_share=st.median(r["interference_share"] for r in done) if done else None,
            median_tpot_without_interference_ms=st.median(
                r["tpot_without_interference_ms"] for r in done) if done else None,
            joint_pass=len(joint), joint_pass_without_interference=len(joint_no_int),
            tpot_failures_attributable_to_interference=len(rescued),
            goodput_req_per_s=len(joint) / cell["observation_end_s"],
            projected_goodput_zero_interference=len(joint_no_int) / cell["observation_end_s"],
            max_abs_reconstruction_residual_ms=max(residual) if residual else None,
            requests=summary))
    return out


def coalescing_projection(episodes, fit, cells):
    """Bounded arithmetic projection ONLY, using the measured fixed/marginal tax.

    It answers: if the same prefill work were delivered in groups of G instead of
    one interleave per arrival, how much interference time would remain? It does
    NOT model the TTFT cost of holding, queue dynamics, token-budget limits or any
    change in decode width, and is therefore an upper bound on the recoverable
    interference, not a predicted policy result.
    """
    fixed, per_token = fit["intercept"], fit["slope"]
    projections = []
    for cell in cells:
        mixed = [r for r in cell["rows"] if r["prefill_tokens"] and r["decode_width"] > 0]
        if len(mixed) < 5:
            continue
        widths = [r["decode_width"] for r in mixed]
        tokens = sum(r["prefill_tokens"] for r in mixed)
        observed = len(mixed) * fixed + per_token * tokens
        rows = dict(cell=cell["path"], regime=cell["regime"], cap=cell["target_cap"],
                    policy=cell["policy"], n_mixed_steps=len(mixed),
                    total_prefill_tokens=tokens, median_mixed_width=st.median(widths),
                    modelled_interference_ms=observed)
        for group in (2, 4, 8):
            events = -(-len(mixed) // group)
            rows[f"modelled_interference_group{group}_ms"] = events * fixed + per_token * tokens
            rows[f"recovered_ms_group{group}"] = observed - (events * fixed + per_token * tokens)
        projections.append(rows)
    return projections


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ttft-slo-s", type=float, default=0.20)
    parser.add_argument("--tpot-slo-s", type=float, default=0.009)
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=False)

    paths = []
    for root in args.results_dir:
        paths.extend(sorted(glob.glob(str(Path(root) / "**" / "cell-*.json"), recursive=True)))
    loaded = [load_cell(p) for p in paths]
    usable = [c for c in loaded if c["aligned"] and c["status"] == "COMPLETE"]
    excluded = [c["path"] for c in loaded if c not in usable]

    buckets = bucket_analysis(usable)
    capture_check = verify_capture_sizes(args.results_dir)
    occupancy = bucket_occupancy(usable)
    tax = matched_prefill_tax(usable)
    width_medians = {int(w): v["step_ms"] for w, v in buckets["per_request_cost"].items()}
    episodes = request_tpot_decomposition(usable, width_medians, args.ttft_slo_s, args.tpot_slo_s)
    fit = tax["fit_tax_vs_prefill_tokens_width_ge7"]
    projection = coalescing_projection(episodes, fit, usable)

    report = dict(
        evidence_type="OBSERVATIONAL_POSTHOC_RECONSTRUCTION_OF_SEALED_NATIVE_CAPTURES",
        source_dirs=args.results_dir, n_cells_used=len(usable), excluded=excluded,
        slo=dict(ttft_s=args.ttft_slo_s, tpot_s=args.tpot_slo_s),
        capture_sizes_assumed=list(CAPTURE_SIZES),
        capture_size_verification=capture_check,
        bucket_analysis=buckets,
        bucket_occupancy=occupancy,
        prefill_tax={k: v for k, v in tax.items() if k != "pairs"},
        episodes=[{k: v for k, v in e.items() if k != "requests"} for e in episodes],
        coalescing_projection=projection)
    (out_dir / "interference.json").write_text(json.dumps(report, indent=2) + "\n")
    (out_dir / "per_request.json").write_text(json.dumps(
        [dict(cell=e["cell"], regime=e["regime"], cap=e["cap"], policy=e["policy"],
              requests=e["requests"]) for e in episodes], indent=2) + "\n")

    lines = ["# Decode-width buckets and admission interference (reconstruction)", "",
             f"cells used {len(usable)}; excluded {len(excluded)}",
             f"matched mixed/pure pairs {tax['n_pairs']}", "",
             "## Capture-size verification (engine log vs assumed bucket edges)", "",
             f'- logs scanned: {capture_check["n_logs_scanned"]}',
             f'- engine-logged `cudagraph_capture_sizes`: {capture_check["logged_capture_size_lists"]}',
             f'- decode FULL graphs captured: {capture_check["decode_full_graph_counts"]}',
             f'- assumed edges {capture_check["assumed"]} match logged subset at or below 32: '
             f'**{capture_check["assumed_matches_logged_subset"]}**',
             f'- decode graph count equals number of assumed edges: '
             f'**{capture_check["decode_graph_count_matches_assumed"]}**', "",
             "## Pure-decode cost is a staircase over CUDA-graph capture buckets", "",
             "| capture bucket | widths present | median ms min | max | within-bucket spread % |",
             "|---:|---|---:|---:|---:|"]
    for size, b in buckets["buckets"].items():
        lines.append(f'| {size} | {b["widths"]} | {b["median_ms_min"]:.3f} | {b["median_ms_max"]:.3f} '
                     f'| {b["within_bucket_spread_pct"]:.2f} |')
    lines += ["", "| capture point transition | delta ms | jump % | ms per added request |",
              "|---|---:|---:|---:|"]
    for s in buckets["across_bucket_steps"]:
        lines.append(f'| {s["from_width"]} -> {s["to_width"]} | {s["delta_ms"]:.3f} | '
                     f'{s["jump_pct"]:.2f} | {s["ms_per_added_request"]:.4f} |')
    lines += ["", "## Per-request decode cost by width", "",
              "| width | step ms | ms per request | bucket | padded waste % |", "|---:|---:|---:|---:|---:|"]
    for w, v in buckets["per_request_cost"].items():
        lines.append(f'| {w} | {v["step_ms"]:.3f} | {v["ms_per_request"]:.4f} | {v["bucket"]} '
                     f'| {v["padded_waste_pct"]:.1f} |')
    lines += ["", "## Where each episode ran inside the staircase", "",
              "| cell | regime | cap | policy | admission events | max width | median width | "
              "mean padding waste | steps exactly at a capture point |",
              "|---|---|---:|---|---:|---:|---:|---:|---:|"]
    for o in occupancy:
        lines.append(f'| {Path(o["cell"]).parent.name}/{Path(o["cell"]).name} | {o["regime"]} | '
                     f'{o["cap"]} | {o["policy"]} | {o["n_admission_events"]} | {o["max_width"]} | '
                     f'{o["median_width"]:.0f} | {o["mean_padding_waste"]:.3f} | '
                     f'{o["share_of_steps_at_a_capture_point"]:.3f} |')
    lines += ["", "## Admission interference tax", ""]
    for name in ("fit_tax_vs_prefill_tokens_all", "fit_tax_vs_prefill_tokens_width_ge7",
                 "fit_tax_vs_decode_width_at_chunk_128"):
        f = tax[name]
        if f:
            lines.append(f'- `{name}`: n={f["n"]}, intercept={f["intercept"]:.4f}, '
                         f'slope={f["slope"]:.6f}, R2={f["r2"]:.4f}')
    lines += ["", "| exact prefill chunk | n | median tax ms | tax per token ms |", "|---:|---:|---:|---:|"]
    for chunk, v in tax["tax_by_exact_chunk_size"].items():
        lines.append(f'| {chunk} | {v["n"]} | {v["median_tax_ms"]:.3f} | {v["tax_per_token_ms"]:.5f} |')
    lines += ["", "## Per-episode request-level attribution", "",
              "| cell | regime | cap | policy | median TPOT ms | median interference share | "
              "median TPOT w/o interference ms | joint pass | joint pass w/o interference | "
              "TPOT failures attributable to interference | max reconstruction residual ms |",
              "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for e in episodes:
        lines.append(
            f'| {Path(e["cell"]).parent.name}/{Path(e["cell"]).name} | {e["regime"]} | {e["cap"]} | '
            f'{e["policy"]} | {e["median_tpot_ms"]:.3f} | {e["median_interference_share"]:.4f} | '
            f'{e["median_tpot_without_interference_ms"]:.3f} | {e["joint_pass"]} | '
            f'{e["joint_pass_without_interference"]} | {e["tpot_failures_attributable_to_interference"]} | '
            f'{e["max_abs_reconstruction_residual_ms"]:.4f} |')
    lines += ["", "## Bounded coalescing projection (arithmetic only, not a policy result)", "",
              "| cell | regime | cap | mixed steps | modelled interference ms | group2 | group4 | group8 |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for p in projection:
        lines.append(f'| {Path(p["cell"]).parent.name}/{Path(p["cell"]).name} | {p["regime"]} | '
                     f'{p["cap"]} | {p["n_mixed_steps"]} | {p["modelled_interference_ms"]:.1f} | '
                     f'{p["modelled_interference_group2_ms"]:.1f} | {p["modelled_interference_group4_ms"]:.1f} | '
                     f'{p["modelled_interference_group8_ms"]:.1f} |')
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")
    print(f"cells {len(usable)}, pairs {tax['n_pairs']}, episodes {len(episodes)}")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
