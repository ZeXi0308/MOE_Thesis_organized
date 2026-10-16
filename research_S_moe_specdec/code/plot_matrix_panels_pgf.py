#!/usr/bin/env python3
"""Compose measured F2 matrices with one domain per panel and one shared scale.

  python code/plot_matrix_panels_pgf.py \
    'OLMoE / n-gram=runs/olmoe_matrix_analysis' \
    'Qwen3 AWQ / EAGLE3=runs/qwen3_matrix_analysis' \
    --output figures/F2_compiled_domains --columns 2

Each LABEL=ANALYSIS_DIR must identify a single-domain plot_results analysis.
The existing plot_matrix_pgf loader checks original sources and reproduces its
statistics once per input. No domains are pooled and no missing cells are
interpolated. Complete original repeat sets (at least two) are required. Each
ratio uses its own domain's same-B AR baseline. Energy retains measured-window
J/token, rather than a complete-population counter span. No GPU or upload.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from compare_controller_policies import output_csv, sha
from plot_energy_progress import render_pgf, tex_escape
from plot_matrix_pgf import METRICS, color, load_analysis, require


def shared_scale(cells, metric):
    values = [row[metric] for row in cells if row[metric] is not None]
    if not values:
        return None
    diverging = metric in ("speedup", "tpot_speedup")
    low, high = ((min(min(values), .99), max(max(values), 1.01))
                 if diverging else (min(values), max(values)))
    if low == high:
        high = low * 1.01 + .001
    return {"minimum": low, "maximum": high,
            "observed_minimum": min(values), "observed_maximum": max(values),
            "neutral_value": 1 if diverging else None,
            "complete_metric_cells": len(values),
            "blank_metric_cells": len(cells) - len(values)}


def domain_caption(domain):
    fields = [domain.get("execution_mode"), domain.get("dtype")]
    quantization = domain.get("quantization")
    if quantization and str(quantization).lower() != "none":
        fields.append(quantization)
    fields.append(domain.get("draft_source"))
    return " | ".join(str(value) for value in fields if value)


def document(panels, cells, metric, scale, columns, title):
    columns = min(columns, len(panels))
    rows = math.ceil(len(panels) / columns)
    low, high = scale["minimum"], scale["maximum"]
    diverging = metric in ("speedup", "tpot_speedup")
    colors = "; ".join("rgb255=(%d,%d,%d)" % color(low + i * (high-low)/100, low, high, diverging)
                       for i in range(101))
    width = max(5.6, max(len(p["axes"]["gammas"]) for p in panels) * .72)
    height = max(3.4, max(len(p["axes"]["batches"]) for p in panels) * .52)
    lines = [r"\documentclass[border=5pt]{standalone}",
             r"\usepackage[T1]{fontenc}\usepackage{lmodern}\usepackage{pgfplots}",
             r"\usepgfplotslibrary{groupplots}\pgfplotsset{compat=1.18}",
             r"\begin{document}\sffamily\begin{tikzpicture}",
             r"\begin{groupplot}[group style={group size=%d by %d,horizontal sep=1.25cm,vertical sep=2.0cm},width=%.2fcm,height=%.2fcm,scale only axis,y dir=reverse,axis on top,tick align=outside,xlabel={Draft length $\gamma$},title style={align=center,font=\small},tick label style={font=\scriptsize},label style={font=\small},point meta min=%.12g,point meta max=%.12g,colormap={sharedcolors}{%s}]"
             % (columns, rows, width, height, low, high, colors)]
    for index, panel in enumerate(panels):
        batches, gammas = panel["axes"]["batches"], panel["axes"]["gammas"]
        heading = f"({chr(97 + index)}) {panel['label']}"
        subtitle = domain_caption(panel["domain"])
        ylabel = "Wave batch B" if panel["domain"]["admission_mode"] == "wave" else "Admission cap B"
        options = ["xmin=-.5", f"xmax={len(gammas)-.5}", "ymin=-.5", f"ymax={len(batches)-.5}",
                   "xtick={%s}" % ",".join(map(str, range(len(gammas)))),
                   "xticklabels={%s}" % ",".join(map(str, gammas)),
                   "ytick={%s}" % ",".join(map(str, range(len(batches)))),
                   "yticklabels={%s}" % ",".join(map(str, batches)),
                   "ylabel={%s}" % tex_escape(ylabel),
                   "title={%s}" % (tex_escape(heading) + r"\\" + tex_escape(subtitle)
                                   + r"; $n=%d$" % panel["axes"]["expected_repeats"])]
        # A single bar applies to every panel; it is not rescaled by domain.
        if index == columns - 1:
            options += ["colorbar", r"colorbar style={y dir=normal,ylabel={%s},label style={font=\small},tick label style={font=\scriptsize}}" % tex_escape(METRICS[metric])]
        lines.append(r"\nextgroupplot[%s]" % ",".join(options))
        for row in (r for r in cells if r["panel_id"] == panel["panel_id"]):
            x, y = gammas.index(row["gamma"]), batches.index(row["batch"])
            value = row[metric]
            rgb = (232, 232, 232) if value is None else color(value, low, high, diverging)
            fill = "{rgb,255:red,%d;green,%d;blue,%d}" % rgb
            lines.append(r"\path[fill=%s,draw=white,line width=.6pt] (axis cs:%.1f,%.1f) rectangle (axis cs:%.1f,%.1f);"
                         % (fill, x-.5, y-.5, x+.5, y+.5))
            if value is not None:
                label = f"{value:.2f}" if diverging else f"{value:.3f}"
                text_color = "white" if sum(a*b for a, b in zip(rgb, (.2126, .7152, .0722))) < 145 else "black"
                lines.append(r"\node[align=center,font=\scriptsize,text=%s,inner sep=1pt] at (axis cs:%d,%d) {%s};"
                             % (text_color, x, y, label))
    for _ in range(columns * rows - len(panels)):
        lines.append(r"\nextgroupplot[hide axis,xmin=0,xmax=1,ymin=0,ymax=1]")
    lines.append(r"\end{groupplot}")
    heading = tex_escape(title) + r"\\" if title else ""
    heading += tex_escape(METRICS[metric]) + " --- common color scale"
    lines.append(r"\path (group c1r1.north west) -- (group c%dr1.north east) node[midway,above=1.15cm,align=center,font=\small] {%s};" % (columns, heading))
    note = r"Complete original-run medians; panel $n$ = planned repeats per cell. Grey/blank = missing measurement or incomplete repeats/AR."
    note += r"\\Models, draft sources, precisions and execution domains may differ; common colors are not a controlled cross-domain comparison."
    note += r"\\Raw repeats and source hashes retained in CSV/JSON; no confidence intervals or interpolation."
    if metric == "speedup":
        note += r"\\Throughput includes prefill/setup/drain within measured windows; inter-wave gaps are excluded. Each panel uses its own same-B AR."
    elif metric == "tpot_speedup":
        note += r"\\Per-run mean request TPOT spans first to last output receipt, excluding prefill. Each panel uses its own same-B AR."
    else:
        note += r"\\Energy sums measured windows including prefill; it is not the full-population counter span across inter-wave gaps."
    lines += [r"\path (group c1r%d.south west) -- (group c%dr%d.south east) node[midway,below=1.2cm,align=center,font=\scriptsize] {%s};" % (rows, columns, rows, note),
              r"\end{tikzpicture}\end{document}"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", metavar="LABEL=ANALYSIS_DIR")
    parser.add_argument("--output", required=True)
    parser.add_argument("--metric", choices=tuple(METRICS), nargs="+", default=list(METRICS))
    parser.add_argument("--columns", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--title")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    require(not output.exists() or (output.is_dir() and not any(output.iterdir())), "choose a fresh/empty derived output directory")
    panels, raw, cells, checks, provenance = [], [], [], [], []
    seen_labels, seen_directories, seen_sources = set(), set(), set()
    for index, item in enumerate(args.inputs):
        label, separator, name = item.partition("=")
        require(separator and label.strip() and name.strip(), "each input must be LABEL=ANALYSIS_DIR")
        label, directory = label.strip(), Path(name.strip()).resolve()
        if directory.is_file():
            directory = directory.parent
        require(label not in seen_labels and directory not in seen_directories, "duplicate panel label or analysis directory")
        seen_labels.add(label)
        seen_directories.add(directory)
        groups, observations, values, validations, origin = load_analysis(directory)
        require(len(groups) == 1, f"{label}: expected exactly one domain; split mixed-domain analyses first")
        sources = {row["source"] for row in observations}
        require(not sources.intersection(seen_sources), "an original run appears in more than one panel")
        seen_sources.update(sources)
        group, info = next(iter(groups.items()))
        panel_id = f"panel{index + 1}"
        prefix = {"panel_id": panel_id, "panel_label": label}
        panels.append({"panel_id": panel_id, "label": label, "group": group, **info,
                       "run_count": len(observations), "cell_count": len(values),
                       "complete_repeat_cells": sum(r["repeat_status"] == "complete" for r in values)})
        raw.extend({**prefix, **r} for r in observations)
        cells.extend({**prefix, **r} for r in values)
        checks.extend({**prefix, **r} for r in validations)
        provenance.append({**prefix, **origin})
    output.mkdir(parents=True, exist_ok=True)
    for name, data in (("raw_runs.csv", raw), ("plot_cells.csv", cells), ("reproduction_checks.csv", checks)):
        output_csv(output / name, data)
    figures, omitted, scales = [], [], {}
    for metric in dict.fromkeys(args.metric):
        scales[metric] = shared_scale(cells, metric)
        if scales[metric] is None:
            omitted.append({"metric": metric, "reason": "no_complete_measured_cells_in_any_panel"})
        elif not args.no_plot:
            figures.extend(render_pgf(document(panels, cells, metric, scales[metric], args.columns, args.title),
                                      output, f"panels_{metric}", ["pdf", "svg", "png"]))
    report = {"schema_version": 1,
              "status": "data_exported" if args.no_plot else "rendered_measured_cells" if figures else "pending_no_complete_cells",
              "panels": panels, "panel_count": len(panels), "input_provenance": provenance,
              "run_count": len(raw), "cell_count": len(cells), "reproduction_check_count": len(checks),
              "maximum_absolute_difference": max((r["absolute_difference"] or 0 for r in checks), default=None),
              "metrics": list(scales), "shared_scales": scales, "figures": figures, "omitted": omitted,
              "source_sha256": {name: sha(Path(__file__).parent / name) for name in
                                (Path(__file__).name, "plot_matrix_pgf.py", "plot_results.py", "plot_energy_progress.py")},
              "boundaries": [
                  "One explicit input per domain. Source runs and native B/gamma axes remain separate; no pooled statistics or cross-domain AR denominator.",
                  "The existing loader/statistics reproduce supplied analyses. Only complete original repeat sets (at least two) are colored; missing metric/AR cells remain grey and unlabelled.",
                  "A common color scale does not establish equivalent model, precision, execution, draft source, resource or workload conditions; full domain metadata is retained.",
                  "Measured-window throughput and energy exclude inter-wave gaps; they are not complete-population throughput or counter-span energy.",
                  "No best-repeat selection, confidence interval, interpolation, new model fit, GPU execution or upload. This composition adds no experimental evidence."]}
    (output / "plot_manifest.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "panels": len(panels), "runs": len(raw), "cells": len(cells),
                      "max_difference": report["maximum_absolute_difference"], "output": str(output)}))


if __name__ == "__main__":
    main()
