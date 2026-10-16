#!/usr/bin/env python3
"""Render measured F4 energy and F5 draft availability/progress curves.

Example (matplotlib required only for rendering)::

  python code/plot_energy_progress.py \
    --performance-summary figures/F2_olmoe_ngram_r02/summary.csv \
    --progress runs/derived/draft_progress_r02_complete/draft_progress_summary.csv \
    --output figures/F4F5_olmoe_ngram_r02 --model-label OLMoE

Use --backend pgfplots to render entirely with installed LaTeX/Poppler instead
of matplotlib. Inputs are summaries from plot_results.py and
summarize_draft_progress.py. The
progress input also accepts the latter's draft_progress.json. Each plotted dot
is a run value. Lines connect run medians; dots are neither confidence intervals
nor independent request-step repetitions. F4 ratios divide each energy run by
the same-B autoregressive run median; they are not paired-run estimates. F5 uses
actual emitted tokens, not the 1+accepted proxy, so terminal clipping is retained.
Separate execution/resource regimes are never combined. Missing values remain
gaps. --no-plot exports the exact plot data and provenance using only Python's
standard library; --self-test checks parsing, missingness and baseline matching.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import shutil
import subprocess
import unittest

from plot_results import REGIME_FIELDS, resolve_regime, write_csv as _write_csv


PROGRESS_METRICS = (
    ("proposal_positive_fraction", "Request-steps with a proposal (%)", 100.0),
    ("mean_draft_slots_per_request_step", "Available draft slots / request-step", 1.0),
    ("actual_emitted_tokens_per_request_step", "Emitted tokens / request-step", 1.0),
)


def write_csv(path, rows):
    _write_csv(path, [{k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v
                       for k, v in row.items()} for row in rows])


def finite(value):
    if value in (None, ""):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def raw_values(row, key):
    """Preserve missing repetitions; never reconstruct raw points from medians."""
    value = row.get(key, [])
    if isinstance(value, str):
        value = json.loads(value) if value else []
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a JSON array")
    return [finite(item) for item in value]


def complete_median(values):
    return statistics.median(values) if values and all(x is not None for x in values) else None


def read_summary(path):
    path = Path(path)
    if path.suffix.lower() == ".csv":
        with path.open(newline="") as handle:
            return list(csv.DictReader(handle))
    obj = json.loads(path.read_text())
    rows = obj if isinstance(obj, list) else obj.get("summary", [])
    if not isinstance(rows, list):
        raise ValueError(f"{path}: expected a summary array")
    return rows


def regime_key(row):
    resolved = resolve_regime(row, {})
    return tuple(str(resolved[key]) for key in REGIME_FIELDS)


def unique_cells(rows, gamma_key):
    result = {}
    for row in rows:
        key = int(row["batch"]), int(row[gamma_key])
        if key in result:
            raise ValueError(f"duplicate cell {key}; separate strategy/proposal budgets first")
        result[key] = row
    return result


def prepare_energy(rows):
    cells = unique_cells(rows, "gamma")
    prepared = []
    for (batch, gamma), row in sorted(cells.items()):
        raw = raw_values(row, "joules_per_token_raw")
        if any(value is not None and value <= 0 for value in raw):
            raise ValueError("energy per token must be positive")
        baseline = cells.get((batch, 0))
        baseline_raw = raw_values(baseline, "joules_per_token_raw") if baseline else []
        baseline_median = complete_median(baseline_raw)
        ratios = [value / baseline_median if value is not None and baseline_median else None for value in raw]
        prepared.append({"batch": batch, "gamma": gamma,
            "proposal_budget": int(row["proposal_budget"]), "n_runs": int(row["n_runs"]),
            "energy_raw": raw, "energy_median": complete_median(raw),
            "ar_energy_raw": baseline_raw, "ar_energy_median": baseline_median,
            "energy_ratio_raw": ratios, "energy_ratio_median": complete_median(ratios),
            "sources": row.get("sources", ""),
            "ratio_boundary": "each run divided by same-B AR run median; not paired; no CI"})
    return prepared


def prepare_progress(rows):
    cells = unique_cells(rows, "nominal_gamma")
    prepared = []
    for (batch, gamma), row in sorted(cells.items()):
        for metric, label, scale in PROGRESS_METRICS:
            raw = raw_values(row, metric + "_raw")
            prepared.append({"batch": batch, "gamma": gamma, "metric": metric,
                "metric_label": label, "display_scale": scale, "raw": raw,
                "median": complete_median(raw), "n_runs": int(row["n_runs"]),
                "population": row["population"], "raw_sources": row.get("raw_sources", ""),
                "decode_request_steps_raw": row.get("decode_request_steps_raw", ""),
                "aggregation_boundary": "run medians; request-steps are descriptive within-run observations"})
    return prepared


def grouped_inputs(performance, progress, strategy, proposal_regime, progress_mode, population):
    groups = defaultdict(lambda: {"performance": [], "progress": []})
    skipped = []
    for source_kind, rows in (("performance", performance), ("progress", progress)):
        for row in rows:
            if source_kind == "performance":
                include = row.get("strategy") == strategy and row.get("proposal_regime") == proposal_regime
                gamma_key = "gamma"
                if "prediction" in str(row.get("evidence_kind", "")).lower():
                    raise ValueError("predicted performance cannot be plotted as measured F4")
            else:
                include = row.get("mode") == progress_mode and row.get("population") == population
                gamma_key = "nominal_gamma"
            if not include:
                continue
            # These curves compare native direct-draft lengths. Post-proposer
            # truncation with a different generation budget is a separate study.
            if int(row["proposal_budget"]) != int(row[gamma_key]):
                skipped.append({"kind": source_kind, "batch": row["batch"], "gamma": row[gamma_key],
                                "reason": "proposal_budget differs from gamma; separate cost regime"})
                continue
            groups[regime_key(row)][source_kind].append(row)
    return groups, skipped


def configure_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "axes.spines.top": False, "axes.spines.right": False,
        "savefig.dpi": 180})
    return plt


def save_figure(fig, destination, formats):
    outputs = []
    for extension in formats:
        path = destination.with_suffix("." + extension)
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        outputs.append(path.name)
    return outputs


def raw_dots(ax, x, raw, color, *, log_x=False, scale=1.0):
    # Small deterministic horizontal offsets expose identical repetitions.
    # Offsets have no quantitative meaning and do not shift median lines.
    for index, value in enumerate(raw):
        if value is None:
            continue
        offset = 0.028 * (index - (len(raw) - 1) / 2)
        px = x * math.exp(offset) if log_x else x + offset
        ax.scatter([px], [scale * value], s=15, facecolors="none", edgecolors=[color],
                   linewidths=0.7, alpha=0.8, zorder=3)


def plot_energy(rows, out, label, formats, plt):
    if not any(row["energy_median"] is not None for row in rows):
        return []
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.65))
    batches = sorted({row["batch"] for row in rows})
    gammas = sorted({row["gamma"] for row in rows})
    colors = plt.get_cmap("tab10")
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    lookup = {(row["batch"], row["gamma"]): row for row in rows}
    for index, gamma in enumerate(gammas):
        color = "#222222" if gamma == 0 else colors((index - 1) % 10)
        for ax, median_key, raw_key in zip(axes, ("energy_median", "energy_ratio_median"),
                                          ("energy_raw", "energy_ratio_raw")):
            values = [lookup.get((b, gamma), {}).get(median_key) for b in batches]
            ax.plot(batches, [v if v is not None else float("nan") for v in values],
                    color=color, marker=markers[index % len(markers)], markersize=3.3,
                    linewidth=1.25, label=rf"$\gamma={gamma}$" + (" (AR)" if gamma == 0 else ""))
            for batch in batches:
                row = lookup.get((batch, gamma))
                if row:
                    raw_dots(ax, batch, row[raw_key], color, log_x=True)
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(batches, [str(batch) for batch in batches])
        ax.set_xlabel("Nominal batch size B")
        ax.grid(axis="both", color="#dddddd", linewidth=0.5, alpha=0.65)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Energy per generated token (J/token)")
    axes[0].set_title("(a) Absolute energy")
    axes[1].set_ylabel("Energy / same-B AR median")
    axes[1].set_title("(b) Relative energy (lower is better)")
    axes[1].axhline(1.0, color="#555555", linewidth=0.7, linestyle=":", zorder=0)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 0.91),
               ncol=min(8, len(handles)), frameon=False, columnspacing=1.0, handlelength=1.8)
    fig.suptitle(f"{label}: n-gram speculation energy", y=1.0, fontsize=11)
    fig.text(0.5, 0.005, "Lines: run medians. Open dots: individual runs (slight x offsets). No confidence intervals.\n"
             "Relative dots use the same-B AR run median; denominator uncertainty is not shown.",
             ha="center", va="bottom", fontsize=8)
    fig.subplots_adjust(top=0.73, bottom=0.23, wspace=0.29)
    outputs = save_figure(fig, out / "F4_energy_vs_batch", formats)
    plt.close(fig)
    return outputs


def plot_progress(rows, out, label, formats, plt, population):
    if not any(row["median"] is not None for row in rows):
        return []
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.75))
    batches = sorted({row["batch"] for row in rows})
    gammas = sorted({row["gamma"] for row in rows})
    lookup = {(row["batch"], row["gamma"], row["metric"]): row for row in rows}
    colors = plt.get_cmap("tab10")
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    for index, batch in enumerate(batches):
        color = colors(index % 10)
        for ax, (metric, ylabel, scale) in zip(axes, PROGRESS_METRICS):
            values = [lookup.get((batch, gamma, metric), {}).get("median") for gamma in gammas]
            ax.plot(gammas, [scale*v if v is not None else float("nan") for v in values],
                    color=color, marker=markers[index % len(markers)], markersize=3.5,
                    linewidth=1.2, label=f"B={batch}")
            for gamma in gammas:
                row = lookup.get((batch, gamma, metric))
                if row:
                    raw_dots(ax, gamma, row["raw"], color, scale=scale)
            ax.set_ylabel(ylabel)
    for index, ax in enumerate(axes):
        ax.set_xticks(gammas)
        ax.set_xlabel(r"Nominal draft length $\gamma$")
        ax.grid(color="#dddddd", linewidth=0.5, alpha=0.65)
        ax.set_title(("(a) Proposal availability", "(b) Draft work available", "(c) Actual decode progress")[index])
    axes[0].set_ylim(bottom=-0.5)
    axes[1].set_ylim(bottom=-0.02)
    axes[2].axhline(1.0, color="#777777", linestyle=":", linewidth=0.7)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 0.91),
               ncol=min(7, len(handles)), frameon=False, columnspacing=1.3)
    fig.suptitle(f"{label}: measured draft availability and decode progress", y=1.0, fontsize=11)
    population_note = {"all_decode": "All scheduled decode request-steps, including zero proposals, drain batches and terminal steps.",
        "nonterminal": "Scheduled nonterminal decode request-steps, including zero proposals and drain batches.",
        "terminal": "Terminal decode request-steps only; native end-of-generation clipping is retained."}[population]
    fig.text(0.5, 0.005, population_note + "\n"
             "Lines: medians of run-level descriptive rates. Open dots: individual runs; request-steps are not independent repeats.",
             ha="center", va="bottom", fontsize=8)
    fig.subplots_adjust(top=0.73, bottom=0.23, wspace=0.36)
    outputs = save_figure(fig, out / "F5_availability_progress", formats)
    plt.close(fig)
    return outputs


def tex_escape(value):
    return "".join({"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
                    "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}"}.get(c, c) for c in str(value))


def pgf_document(panels, legend_count, note, title):
    """A local, dependency-free data exporter to the installed PGFPlots engine."""
    count = len(panels)
    colors = ["222222", "1f77b4", "ff7f0e", "2ca02c", "d62728", "9467bd", "8c564b", "e377c2"]
    lines = [r"\documentclass[border=5pt]{standalone}", r"\usepackage[T1]{fontenc}",
        r"\usepackage{lmodern}", r"\usepackage{pgfplots}", r"\usepgfplotslibrary{groupplots}",
        r"\pgfplotsset{compat=1.18}", r"\begin{document}", r"\sffamily",
        r"\begin{tikzpicture}"]
    for i, color in enumerate(colors):
        lines.append(r"\definecolor{series%d}{HTML}{%s}" % (i, color))
    width = "9.0cm" if count == 2 else "6.5cm"
    lines.append(r"\begin{groupplot}[group style={group size=%d by 1,horizontal sep=1.8cm}," % count
        + "width=" + width + r",height=6.2cm,scale only axis,grid=major,grid style={gray!20},"
        + r"axis lines*=left,tick label style={font=\small},label style={font=\small},"
        + r"title style={font=\small},unbounded coords=jump]")
    markers = ["*", "square*", "triangle*", "diamond*", "triangle", "pentagon*", "x", "star"]
    for index, panel in enumerate(panels):
        options = ["title={" + panel["title"] + "}", "xlabel={" + panel["xlabel"] + "}",
                   "ylabel={" + panel["ylabel"] + "}",
                   "xtick={" + ",".join(str(x) for x in panel["xs"]) + "}",
                   "xticklabels={" + ",".join(str(x) for x in panel["xs"]) + "}"]
        if panel.get("log_x"):
            options += ["xmode=log", "log basis x=2", "minor x tick num=0"]
        if panel.get("log_y"):
            options += ["ymode=log"]
        else:
            options += ["scaled y ticks=false", r"yticklabel style={/pgf/number format/fixed,/pgf/number format/precision=2}"]
        if index == 0:
            options += ["legend columns=" + str(legend_count),
                r"legend to name=sharedlegend,legend style={draw=none,font=\small,/tikz/every even column/.append style={column sep=8pt}}"]
        lines.append(r"\nextgroupplot[" + ",".join(options) + "]")
        for series_index, series in enumerate(panel["series"]):
            coords = " ".join("(%s,%s)" % (x, "nan" if y is None else f"{y:.12g}") for x, y in series["medians"])
            lines.append(r"\addplot+[color=series%d,mark=%s,mark size=1.7pt,mark options={fill=series%d,draw=series%d},line width=0.8pt] coordinates {%s};"
                         % (series_index, markers[series_index % len(markers)], series_index, series_index, coords))
            if index == 0:
                lines.append(r"\addlegendentry{" + series["label"] + "}")
            raw_coords = []
            for x, raw in series["raw"]:
                for run_index, value in enumerate(raw):
                    if value is None:
                        continue
                    offset = 0.028 * (run_index - (len(raw) - 1) / 2)
                    px = x * math.exp(offset) if panel.get("log_x") else x + offset
                    raw_coords.append(f"({px:.12g},{value:.12g})")
            lines.append(r"\addplot+[only marks,forget plot,color=series%d,mark=o,mark size=2.1pt,mark options={fill=white,line width=0.45pt}] coordinates {%s};"
                         % (series_index, " ".join(raw_coords)))
        if panel.get("reference") is not None:
            lines.append(r"\addplot[forget plot,gray,densely dotted,domain=%s:%s] {%s};"
                         % (min(panel["xs"]), max(panel["xs"]), panel["reference"]))
    lines += [r"\end{groupplot}",
        r"\path (group c1r1.north west) -- node[above=1.2cm,font=\large] {%s} (group c%dr1.north east);" % (tex_escape(title), count),
        r"\path (group c1r1.north west) -- node[above=0.7cm] {\ref{sharedlegend}} (group c%dr1.north east);" % count,
        r"\path (group c1r1.south west) -- node[below=1.2cm,align=center,font=\small] {%s} (group c%dr1.south east);" % (note, count),
        r"\end{tikzpicture}", r"\end{document}"]
    return "\n".join(lines) + "\n"


def render_pgf(tex, out, name, formats):
    for executable in ["pdflatex"] + (["pdftoppm"] if "png" in formats else []) + (["pdftocairo"] if "svg" in formats else []):
        if shutil.which(executable) is None:
            raise RuntimeError(f"PGFPlots backend requires {executable}")
    path = out / (name + ".tex")
    path.write_text(tex)
    for _ in range(2):  # Resolve the shared legend reference.
        result = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", path.name],
                                cwd=out, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(result.stdout[-6000:])
    outputs = [name + ".pdf", name + ".tex"]
    if "png" in formats:
        subprocess.run(["pdftoppm", "-png", "-singlefile", "-r", "150", name + ".pdf", name], cwd=out, check=True)
        outputs.append(name + ".png")
    if "svg" in formats:
        subprocess.run(["pdftocairo", "-svg", name + ".pdf", name + ".svg"], cwd=out, check=True)
        outputs.append(name + ".svg")
    for suffix in (".aux", ".log"):
        (out / (name + suffix)).unlink(missing_ok=True)
    return outputs


def plot_pgf(energy, progress, out, label, formats, population):
    outputs = []
    if any(row["energy_median"] is not None for row in energy):
        batches = sorted({row["batch"] for row in energy})
        gammas = sorted({row["gamma"] for row in energy})
        lookup = {(row["batch"], row["gamma"]): row for row in energy}
        panels = []
        for index, (median_key, raw_key, title, ylabel) in enumerate([
            ("energy_median", "energy_raw", "(a) Absolute energy", "Energy per generated token (J/token)"),
            ("energy_ratio_median", "energy_ratio_raw", "(b) Relative energy (lower is better)", "Energy / same-B AR median")]):
            panel = {"xs": batches, "xlabel": "Nominal batch size $B$", "ylabel": ylabel,
                     "title": title, "log_x": True, "log_y": index == 0, "series": []}
            if index == 1:
                panel["reference"] = 1
            for gamma in gammas:
                panel["series"].append({"label": r"$\gamma=%d$" % gamma + (" (AR)" if gamma == 0 else ""),
                    "medians": [(b, lookup.get((b, gamma), {}).get(median_key)) for b in batches],
                    "raw": [(b, lookup.get((b, gamma), {}).get(raw_key, [])) for b in batches]})
            panels.append(panel)
        note = r"Lines: run medians. Open dots: individual runs (slight $x$ offsets). No confidence intervals.\\" \
               r"Relative dots use the same-$B$ AR run median; denominator uncertainty is not shown."
        outputs += render_pgf(pgf_document(panels, len(gammas), note, label + ": n-gram speculation energy"),
                              out, "F4_energy_vs_batch", formats)
    if any(row["median"] is not None for row in progress):
        batches = sorted({row["batch"] for row in progress})
        gammas = sorted({row["gamma"] for row in progress})
        lookup = {(row["batch"], row["gamma"], row["metric"]): row for row in progress}
        panels = []
        for index, (metric, ylabel, scale) in enumerate(PROGRESS_METRICS):
            panel = {"xs": gammas, "xlabel": r"Nominal draft length $\gamma$", "ylabel": tex_escape(ylabel),
                "title": ("(a) Proposal availability", "(b) Draft work available", "(c) Actual decode progress")[index], "series": []}
            if index == 2:
                panel["reference"] = 1
            for batch in batches:
                values = [lookup.get((batch, g, metric), {}).get("median") for g in gammas]
                panel["series"].append({"label": f"$B={batch}$",
                    "medians": [(g, value*scale if value is not None else None) for g, value in zip(gammas, values)],
                    "raw": [(g, [x*scale if x is not None else None for x in lookup.get((batch, g, metric), {}).get("raw", [])]) for g in gammas]})
            panels.append(panel)
        population_note = {"all_decode": "All scheduled decode request-steps: includes zero proposals, drain batches and terminal steps.",
            "nonterminal": "Nonterminal decode request-steps: includes zero proposals and drain batches.",
            "terminal": "Terminal decode request-steps only: native end-of-generation clipping is retained."}[population]
        note = tex_escape(population_note) + r"\\Lines: medians of run-level descriptive rates. Open dots: individual runs; request-steps are not independent repeats."
        outputs += render_pgf(pgf_document(panels, len(batches), note, label + ": measured draft availability and decode progress"),
                              out, "F5_availability_progress", formats)
    return outputs


class Tests(unittest.TestCase):
    def test_missing_raw_is_not_imputed(self):
        self.assertEqual(raw_values({"x": "[1.0, null]"}, "x"), [1.0, None])
        self.assertIsNone(complete_median([1, None]))
        self.assertEqual(raw_values({"median": 1}, "x"), [])

    def test_baseline_is_same_batch(self):
        rows = [{"batch": b, "gamma": g, "proposal_budget": g, "n_runs": 2,
                 "joules_per_token_raw": raw} for b, g, raw in
                [(1, 0, [2, 4]), (1, 1, [1, 2]), (4, 1, [0.3, 0.5])]]
        got = prepare_energy(rows)
        self.assertEqual(got[1]["energy_ratio_raw"], [1/3, 2/3])
        self.assertEqual(got[2]["energy_ratio_raw"], [None, None])

    def test_duplicate_cell_rejected(self):
        with self.assertRaises(ValueError):
            unique_cells([{"batch": 1, "gamma": 0}]*2, "gamma")

    def test_regime_separates_execution(self):
        self.assertNotEqual(regime_key({"execution_mode": "eager"}),
                            regime_key({"execution_mode": "compiled"}))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--performance-summary", type=Path)
    parser.add_argument("--progress", type=Path, help="draft_progress_summary.csv or draft_progress.json")
    parser.add_argument("--output", type=Path, default=Path("figures/F4F5"))
    parser.add_argument("--model-label", default="MoE")
    parser.add_argument("--strategy", default="static")
    parser.add_argument("--proposal-regime", default="direct_draft")
    parser.add_argument("--progress-mode", default="native")
    parser.add_argument("--population", choices=("all_decode", "nonterminal", "terminal"), default="all_decode")
    parser.add_argument("--formats", nargs="+", choices=("pdf", "svg", "png"), default=["pdf", "svg", "png"])
    parser.add_argument("--backend", choices=("matplotlib", "pgfplots"), default="matplotlib",
                        help="pgfplots uses installed LaTeX/Poppler only, without Python plotting packages")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
        return 0 if result.wasSuccessful() else 1
    if not args.performance_summary and not args.progress:
        parser.error("provide --performance-summary and/or --progress")
    performance = read_summary(args.performance_summary) if args.performance_summary else []
    progress = read_summary(args.progress) if args.progress else []
    groups, skipped = grouped_inputs(performance, progress, args.strategy, args.proposal_regime,
                                    args.progress_mode, args.population)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {"inputs": [{"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                           for path in (args.performance_summary, args.progress) if path],
        "population": args.population, "skipped": skipped, "regimes": [],
        "evidence_boundary": "measured run summaries only; missing cells remain gaps; no CI",
        "energy_ratio_boundary": "ratio of medians; raw points divided by same-B AR median, not paired estimates",
        "progress_boundary": "run-level descriptive rates over scheduled decode request-steps; actual client-emitted tokens",
        "not_observed_here": "no EWMA trajectory or cross-workload acceptance distribution is inferred"}
    manifest["render_backend"] = args.backend
    plt = None if args.no_plot or not groups or args.backend != "matplotlib" else configure_matplotlib()
    for key, group in sorted(groups.items()):
        regime = dict(zip(REGIME_FIELDS, key))
        digest = hashlib.sha256(json.dumps(regime, sort_keys=True).encode()).hexdigest()[:10]
        out = args.output if len(groups) == 1 else args.output / digest
        out.mkdir(parents=True, exist_ok=True)
        energy = prepare_energy(group["performance"])
        coverage = prepare_progress(group["progress"])
        write_csv(out / "F4_plot_values.csv", energy)
        write_csv(out / "F5_plot_values.csv", coverage)
        outputs = []
        if plt:
            outputs += plot_energy(energy, out, args.model_label, args.formats, plt)
            outputs += plot_progress(coverage, out, args.model_label, args.formats, plt, args.population)
        elif not args.no_plot and args.backend == "pgfplots":
            outputs += plot_pgf(energy, coverage, out, args.model_label, args.formats, args.population)
        manifest["regimes"].append({"regime": regime, "directory": str(out),
            "performance_cells": len(energy), "performance_run_records": sum(row["n_runs"] for row in energy),
            "progress_cells": len(group["progress"]), "progress_run_records": sum(int(row["n_runs"]) for row in group["progress"]),
            "energy_missing_or_partial_cells": sum(row["energy_median"] is None for row in energy),
            "progress_missing_or_partial_metrics": sum(row["median"] is None for row in coverage),
            "outputs": outputs})
    (args.output / "plot_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output), "regimes": len(groups),
                      "performance_cells": sum(len(g["performance"]) for g in groups.values()),
                      "progress_cells": sum(len(g["progress"]) for g in groups.values()), "plotted": not args.no_plot}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
