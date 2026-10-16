#!/usr/bin/env python3
"""Reproduce plot_results performance matrices with local PGFPlots, no matplotlib.

  python code/plot_matrix_pgf.py --analysis figures/F2_olmoe_ngram_r02 \
    --output figures/F2_olmoe_ngram_r02_reproduced --title 'OLMoE / n-gram'

Future domains use the identical two-stage interface:
  python code/plot_results.py runs/DOMAIN_reduced_r01 --no-plot --output runs/DOMAIN_matrix_analysis
  python code/plot_matrix_pgf.py --analysis runs/DOMAIN_matrix_analysis --output figures/F2_DOMAIN

--analysis accepts the analysis directory or one of its metadata/raw/summary
files. All three companion files and original source JSON must be available.
Original remote /runs/ paths are resolved against this local project. The
existing loader/statistics are reused and checked against exported raw/summary
values. Instrumented performance sources are rejected, not plotted. Native
direct drafting only; an adaptive configured gamma is not an action-grid cell.

Plots use complete original repeat sets (at least two). Incomplete cells or AR
denominators remain blank. Raw run values, exact plot cells, source hashes and
reproduction checks are exported. Energy is a separate measured-window J/token
figure, not inferred complete-population counter-span energy. No GPU or upload.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path

from compare_controller_policies import canonical, output_csv, sha
from plot_energy_progress import render_pgf, tex_escape
from plot_results import REGIME_FIELDS, _finite, _requests, load_runs, summarize


PROJECT = Path(__file__).resolve().parents[1]
BASE_METRICS = ("tokens_per_second", "effective_tpot_ms", "joules_per_token")
METRICS = {"speedup": "Throughput / same-B AR", "tpot_speedup": "Same-B AR TPOT / TPOT",
           "joules_per_token": "Measured-window energy (J / output token)"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def locate_source(name):
    path = Path(name)
    if path.is_file():
        return path.resolve()
    if "/runs/" in name:
        path = PROJECT / "runs" / name.split("/runs/", 1)[1]
    elif not path.is_absolute():
        path = PROJECT / path
    require(path.is_file(), f"original source unavailable; cannot check route instrumentation: {name}")
    return path.resolve()


def equal_number(left, right):
    left, right = _finite(left), _finite(right)
    return left == right if left is None or right is None else math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)


def load_analysis(directory):
    directory = Path(directory)
    if directory.is_file():
        directory = directory.parent
    metadata = read(directory / "analysis_metadata.json")
    exported = read_csv(directory / "raw_runs.csv")
    summaries = read_csv(directory / "summary.csv")
    require(metadata.get("evidence_kind") == "measured_generation_runs", "not a plot_results measured performance analysis")
    require(metadata.get("run_count") == len(exported), "analysis/raw run count differs")
    summary_by_sources = {}
    for row in summaries:
        key = frozenset(str(locate_source(s)) for s in json.loads(row["sources"]))
        require(key not in summary_by_sources, "duplicate summary source set")
        summary_by_sources[key] = row
    groups, raw, provenance, seen = {}, [], [], set()
    for row in exported:
        require(row.get("evidence_kind") == "measured_run", "raw row is not a measured generation run")
        source = locate_source(row["source"])
        require(source not in seen, "duplicate original run source")
        seen.add(source)
        loaded, excluded = load_runs([source])
        require(len(loaded) == 1 and not excluded, f"source rejected by performance loader: {source}: {excluded}")
        actual = loaded[0]
        require(actual["measurement_kind"] == "fixed_length_generation", "matrix requires fixed-length generation; separate other workloads")
        require(actual["strategy"] == "static" and actual["mode"] in ("native", "static") and actual["proposal_regime"] == "direct_draft",
                "matrix accepts only static direct drafting; separate tables/adaptive/post-truncation runs")
        for key in ("batch", "gamma", "repeat", "proposal_budget", "request_count"):
            require(int(row[key]) == actual[key], f"raw/source identity differs: {key}")
        require(row.get("strategy") == actual["strategy"] and row.get("proposal_regime") == actual["proposal_regime"], "raw/source execution strategy differs")
        for key in (*BASE_METRICS, "wall_s", "output_tokens"):
            require(equal_number(row.get(key), actual.get(key)), f"raw/source value differs: {key} in {source}")
        record, config = read(source), read(source.parent / "configuration.json")
        plan, args = config["plan"], config["args"]
        ids = sorted(r["source_index"] for r in _requests(record))
        require(ids and len(set(ids)) == len(ids) and record.get("all_finished") is True, "incomplete/duplicate request population")
        domain = {k: plan.get(k) for k in ("model_revision", "draft_model", "draft_revision", "speculative_overrides", "semantic_token_boundary")}
        domain.update({k: actual[k] for k in REGIME_FIELDS})
        domain.update(sample_indices_sha256=hashlib.sha256(canonical(ids).encode()).hexdigest(),
                      request_count=len(ids), output_tokens_per_request=actual["output_tokens"] / len(ids))
        tag = hashlib.sha256(canonical(domain).encode()).hexdigest()[:12]
        repeats = max(2, int(args.get("repeats") or plan["repeats"]))
        axes = {"batches": sorted(args.get("batches") or plan["batches"]), "gammas": sorted(plan["gammas"]), "expected_repeats": repeats}
        if tag in groups:
            require(groups[tag]["axes"] == axes, "same domain has different matrix grids/repeat plans; analyze separately")
        groups[tag] = {"domain": domain, "axes": axes}
        raw.append({**actual, "group": tag, "original_exported_source": row["source"], "source_sha256": sha(source)})
        provenance.append({"source": str(source), "sha256": sha(source), "configuration_sha256": sha(source.parent / "configuration.json"),
                           "engine_args_sha256": sha(source.parent / "engine_args.json")})
    cells, checks = [], []
    for tag, info in groups.items():
        observations = [r for r in raw if r["group"] == tag]
        by_cell = defaultdict(list)
        for row in observations:
            by_cell[row["batch"], row["gamma"]].append(row)
        for key, rows in by_cell.items():
            require(len({r["repeat"] for r in rows}) == len(rows), f"duplicate repeat IDs in cell {key}; do not pool campaigns")
        computed = {}
        for row in summarize(observations):
            key = row["batch"], row["gamma"]
            originals = by_cell[key]
            previous = summary_by_sources.get(frozenset(r["source"] for r in originals))
            require(previous is not None, f"summary does not identify original raw sources for {key}")
            require(int(previous["n_runs"]) == len(originals), "summary repeat count differs")
            for metric in (*BASE_METRICS, "speedup", "tpot_speedup"):
                require(equal_number(previous.get(metric), row.get(metric)), f"recomputed summary differs: {key} {metric}")
                a, b = _finite(previous.get(metric)), _finite(row.get(metric))
                checks.append({"group": tag, "batch": key[0], "gamma": key[1], "metric": metric,
                               "exported_value": a, "recomputed_value": b, "absolute_difference": abs(a-b) if a is not None and b is not None else None})
            computed[key] = row
        expected = info["axes"]["expected_repeats"]
        for batch in info["axes"]["batches"]:
            for gamma in info["axes"]["gammas"]:
                rows = sorted(by_cell[batch, gamma], key=lambda r: r["repeat"])
                base = sorted(by_cell[batch, 0], key=lambda r: r["repeat"])
                repeats = [r["repeat"] for r in rows]
                complete = repeats == list(range(expected))
                baseline_complete = [r["repeat"] for r in base] == list(range(expected)) and all(r["proposal_budget"] == 0 for r in base)
                cell = {"group": tag, "batch": batch, "gamma": gamma, "expected_repeats": expected,
                    "observed_repeats": repeats, "sources": [r["source"] for r in rows], "source_sha256": [r["source_sha256"] for r in rows],
                    "repeat_status": "complete" if complete else "pending", "ar_repeat_status": "complete" if baseline_complete else "pending"}
                for metric in BASE_METRICS:
                    cell[metric + "_raw"] = [r[metric] for r in rows]
                    cell[metric] = computed.get((batch, gamma), {}).get(metric) if complete and all(r[metric] is not None for r in rows) else None
                for ratio, metric in (("speedup", "tokens_per_second"), ("tpot_speedup", "effective_tpot_ms")):
                    valid = complete and baseline_complete and all(r[metric] is not None for r in rows + base)
                    cell[ratio] = computed.get((batch, gamma), {}).get(ratio) if valid else None
                cells.append(cell)
    return groups, raw, cells, checks, {"analysis_directory": str(directory.resolve()), "analysis_metadata": metadata,
        "analysis_sha256": {name: sha(directory / name) for name in ("analysis_metadata.json", "raw_runs.csv", "summary.csv")},
        "source_provenance": provenance}


def color(value, low, high, diverging):
    stops = ((178, 24, 43), (247, 247, 247), (33, 102, 172)) if diverging else ((247, 251, 255), (8, 48, 107))
    if diverging:
        left, right = (stops[0], stops[1]) if value <= 1 else (stops[1], stops[2])
        fraction = (value-low)/(1-low) if value <= 1 else (value-1)/(high-1)
    else:
        left, right = stops
        fraction = (value-low)/(high-low)
    return tuple(round(a + min(1, max(0, fraction)) * (b-a)) for a, b in zip(left, right))


def document(info, cells, metric, title):
    batches, gammas = info["axes"]["batches"], info["axes"]["gammas"]
    values = [r[metric] for r in cells if r[metric] is not None]
    diverging = metric in ("speedup", "tpot_speedup")
    low, high = (min(min(values), .99), max(max(values), 1.01)) if diverging else (min(values), max(values))
    if low == high:
        high = low * 1.01 + .001
    colors = "; ".join("rgb255=(%d,%d,%d)" % color(low + i * (high-low)/100, low, high, diverging) for i in range(101))
    domain = info["domain"]
    heading = title or f"{Path(domain['model']).name} | {domain['draft_source']} | {domain['execution_mode']}"
    width, height = max(8.5, len(gammas) * 1.05), max(4.0, len(batches) * .72)
    xlabel = "Synchronous wave batch B" if domain["admission_mode"] == "wave" else "Admission cap B"
    lines = [r"\documentclass[border=5pt]{standalone}", r"\usepackage[T1]{fontenc}\usepackage{lmodern}\usepackage{pgfplots}",
        r"\pgfplotsset{compat=1.18}\begin{document}\sffamily\begin{tikzpicture}",
        r"\begin{axis}[name=matrix,width=%.2fcm,height=%.2fcm,scale only axis,xmin=-.5,xmax=%.1f,ymin=-.5,ymax=%.1f,y dir=reverse,axis on top,tick align=outside,xtick={%s},xticklabels={%s},ytick={%s},yticklabels={%s},xlabel={Draft length $\gamma$},ylabel={%s},title={%s},title style={align=center,font=\small},tick label style={font=\small},label style={font=\small},point meta min=%.12g,point meta max=%.12g,colormap={matrixcolors}{%s},colorbar,colorbar style={ylabel={%s},label style={font=\small},tick label style={font=\small}}]" %
        (width, height, len(gammas)-.5, len(batches)-.5, ",".join(map(str, range(len(gammas)))), ",".join(map(str, gammas)),
         ",".join(map(str, range(len(batches)))), ",".join(map(str, batches)), tex_escape(xlabel),
         tex_escape(heading) + r"\\" + tex_escape(METRICS[metric]), low, high, colors, tex_escape(METRICS[metric]))]
    for row in cells:
        x, y = gammas.index(row["gamma"]), batches.index(row["batch"])
        value = row[metric]
        rgb = (232, 232, 232) if value is None else color(value, low, high, diverging)
        fill = "{rgb,255:red,%d;green,%d;blue,%d}" % rgb
        lines.append(r"\path[fill=%s,draw=white,line width=.6pt] (axis cs:%.1f,%.1f) rectangle (axis cs:%.1f,%.1f);" % (fill, x-.5, y-.5, x+.5, y+.5))
        if value is not None:
            label = f"{value:.2f}" if diverging else f"{value:.3f}"
            lines.append(r"\node[align=center,font=\scriptsize,fill=white,fill opacity=.76,text opacity=1,inner sep=1.5pt] at (axis cs:%d,%d) {%s\\$n=%d$};" % (x, y, label, len(row["observed_repeats"])))
    note = r"Complete run medians; $n$ = original repeats. Grey/blank = incomplete cell or missing AR denominator.\\Raw repeats retained in CSV; no confidence intervals; requests/steps are not independent system repeats."
    if metric == "speedup":
        note += r"\\Throughput uses summed measured windows, including prefill/setup/drain; inter-wave handling gaps are excluded."
    elif metric == "tpot_speedup":
        note += r"\\TPOT is the per-run request mean from first to last output receipt; per-request prefill is excluded."
    else:
        note += r"\\Energy sums measured windows including prefill; it is not the full counter span across inter-wave gaps."
    lines += [r"\end{axis}", r"\node[below=1.3cm,align=center,font=\scriptsize] at (matrix.south) {%s};" % note,
              r"\end{tikzpicture}\end{document}"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--analysis", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metric", choices=tuple(METRICS), nargs="+", default=list(METRICS))
    parser.add_argument("--title")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    require(not output.exists() or not any(output.iterdir()), "choose a fresh/empty derived output directory")
    groups, raw, cells, checks, provenance = load_analysis(args.analysis)
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("raw_runs.csv", raw), ("plot_cells.csv", cells), ("reproduction_checks.csv", checks)):
        output_csv(output / name, rows)
    figures, omitted = [], []
    for group, info in groups.items():
        values = [r for r in cells if r["group"] == group]
        for metric in args.metric:
            if not any(r[metric] is not None for r in values):
                omitted.append({"group": group, "metric": metric, "reason": "no_complete_measured_cells"})
            elif not args.no_plot:
                figures.extend(render_pgf(document(info, values, metric, args.title), output, f"{metric}_{group}", ["pdf", "svg", "png"]))
    report = {"schema_version": 1, "status": "data_exported" if args.no_plot else "rendered_measured_cells" if figures else "pending_no_complete_cells",
        **provenance, "groups": groups, "run_count": len(raw), "cell_count": len(cells),
        "reproduction_check_count": len(checks), "maximum_absolute_difference": max((r["absolute_difference"] or 0 for r in checks), default=None),
        "metrics": args.metric, "figures": figures, "omitted": omitted,
        "source_sha256": {name: sha(Path(__file__).parent / name) for name in (Path(__file__).name, "plot_results.py", "plot_energy_progress.py")},
        "boundaries": ["Original JSON is checked through plot_results.load_runs; route-instrumented sources are rejected.",
            "Statistics reuse plot_results.summarize and must reproduce the supplied CSV. Only complete original repeat sets (at least two) are colored.",
            "Execution/draft revisions and model paths, resource regime, request sample identities and output population remain separate. No cross-domain AR denominator.",
            "Throughput is summed-window throughput, not population throughput including inter-wave gaps. Energy is measured-window J/token, not full-population counter-span energy.",
            "Native direct drafting only. No adaptive configured-cap heatmap, best-repeat selection, confidence interval, GPU execution or upload."]}
    (output / "plot_manifest.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "runs": len(raw), "cells": len(cells), "max_difference": report["maximum_absolute_difference"], "output": str(output)}))


if __name__ == "__main__":
    main()
