#!/usr/bin/env python3
"""Render measured F3/F5/F6 controller figures as standalone SVG/PDF.

  python plot_controller.py --performance runs/MATRIX runs/CONTROLLERS \
      --analysis runs/CALIBRATION --output figures/controllers

F3 shows request-mean effective TPOT, summarized across run records. Its per-B
best measured static curve is labelled a descriptive post-hoc lower envelope;
it is not a single fixed-gamma policy. Optional --static-selection JSON contains
selected_gamma and selection_source_files; these must not overlap evaluation
files. This provenance check alone does not establish statistical independence.

F5 uses measured acceptance_survival.csv and acceptance_tracking.csv. Per-step
records currently contain a conditional alpha estimate, not complete EWMA prefix
or availability snapshots; the plot says so. F6 uses timed CPU controller costs
and counts of discarded available drafts, never host-call time as GPU kernel
time. Missing measurements stay absent. No placeholder experiment is generated.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import statistics

from plot_results import REGIME_FIELDS, _context, _finite, load_runs, resolve_regime, summarize, write_csv


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [{**row, **resolve_regime(row, _context(Path(row["source"])) if row.get("source") else {})}
            for row in rows]


def series_label(row: dict) -> str:
    mode = row.get("strategy", row.get("mode", "unknown"))
    gamma, budget = int(row["gamma"]), int(row.get("proposal_budget", row["gamma"]))
    if mode in ("native", "static"):
        regime = row.get("proposal_regime", "direct_draft" if gamma == budget else f"full_draft_budget_{budget}")
        return f"static gamma={gamma}" + (f" ({regime})" if regime != "direct_draft" else "")
    if mode == "native_batch_table":
        return "vLLM native batch table"
    if mode == "dense_heuristic":
        return f"dense batch-disable heuristic (budget={budget})"
    return f"{mode} (budget={budget})"


def selected_static(path: str | None, runs: list[dict]) -> dict | None:
    if path is None:
        return None
    obj = json.loads(Path(path).read_text())
    gamma = int(obj["selected_gamma"])
    files = obj.get("selection_source_files", [])
    if gamma < 0 or not files:
        raise ValueError("static selection needs selected_gamma>=0 and selection_source_files")
    selection = {str(Path(p).resolve()) for p in files}
    evaluation = {str(Path(row["source"]).resolve()) for row in runs}
    if selection & evaluation:
        raise ValueError("static-selection files overlap evaluation; cannot label as held-out selection")
    return {"selected_gamma": gamma, "selection_source_files": sorted(selection),
            "label": f"fixed gamma={gamma} (selected on separate files)",
            "boundary": "disjoint source files checked; independent-run design must be checked separately"}


def _tag(key) -> str:
    return hashlib.sha256(json.dumps(key).encode()).hexdigest()[:9]


def _execution(key) -> str:
    return dict(zip(REGIME_FIELDS, key)).get("execution_mode", "unspecified")


def _save(fig, output: Path, name: str, formats: list[str]) -> list[str]:
    paths = []
    for ext in formats:
        path = output / f"{name}.{ext}"
        fig.savefig(path, dpi=200)
        paths.append(str(path))
    return paths


def _finish_axis(ax):
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Synchronous wave batch B")
    ax.grid(alpha=.2)


def plot_f3(rows: list[dict], output: Path, formats: list[str], selection: dict | None, plt) -> list[str]:
    groups = defaultdict(list)
    for row in rows:
        if row["effective_tpot_ms"] is not None:
            groups[tuple(row[k] for k in REGIME_FIELDS)].append(row)
    files = []
    for key, values in sorted(groups.items()):
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        series, envelope = defaultdict(list), defaultdict(list)
        selected = []
        for row in values:
            label = series_label(row)
            series[label].append((row["batch"], row["effective_tpot_ms"], row["effective_tpot_ms_raw"]))
            if row["strategy"] == "static" and row["proposal_regime"] == "direct_draft":
                envelope[row["batch"]].append(row["effective_tpot_ms"])
                if selection and row["gamma"] == selection["selected_gamma"]:
                    selected.append((row["batch"], row["effective_tpot_ms"]))
        for label, points in sorted(series.items()):
            points.sort()
            x, y = [p[0] for p in points], [p[1] for p in points]
            line, = ax.plot(x, y, marker="o", markersize=4, label=label, alpha=.85)
            for batch, _, raw in points:
                values_raw = json.loads(raw)
                ax.scatter([batch] * len(values_raw), values_raw, color=line.get_color(), s=13, alpha=.4)
        if envelope:
            points = sorted((b, min(v)) for b, v in envelope.items())
            ax.plot([p[0] for p in points], [p[1] for p in points], "k--", lw=1.5,
                    label="per-B static lower envelope (post hoc, descriptive)")
        if selected:
            selected.sort()
            ax.plot([p[0] for p in selected], [p[1] for p in selected], color="black", lw=2.5,
                    label=selection["label"])
        _finish_axis(ax)
        ax.set_ylabel("Effective TPOT (ms / output token / request)")
        ax.set_title(f"{Path(key[0]).name} | {key[3]} | {_execution(key)} | {key[4]}", fontsize=10)
        ax.legend(fontsize=7, loc="best")
        fig.text(.01, .01, "Lines: run medians; dots: raw run values. Envelope is not one fixed-gamma policy.", fontsize=8)
        fig.tight_layout(rect=(0, .035, 1, 1))
        files.extend(_save(fig, output, "F3_tpot_" + _tag(key), formats))
        plt.close(fig)
    return files


def plot_f5_survival(rows: list[dict], output: Path, formats: list[str], plt) -> list[str]:
    groups = defaultdict(list)
    for row in rows:
        if _finite(row.get("availability_survival")) is not None:
            key = tuple(row.get(k, "unspecified") for k in (*REGIME_FIELDS, "mode", "gamma", "proposal_budget"))
            groups[key].append(row)
    files = []
    for key, values in sorted(groups.items()):
        fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5), sharex=True, sharey=True)
        metrics = [("availability_survival", "Available full drafts P(P >= rank)"),
                   ("conditional_acceptance_survival", "Accepted prefix | scheduled to rank"),
                   ("unconditional_accepted_survival", "Accepted prefix / all decode requests")]
        for ax, (metric, title) in zip(axes, metrics):
            series = defaultdict(lambda: defaultdict(list))
            for row in values:
                value = _finite(row.get(metric))
                if value is not None:
                    series[int(row["batch"])][int(row["rank"])].append(value)
            for batch, ranks in sorted(series.items()):
                ranks_sorted = sorted(ranks)
                line, = ax.plot(ranks_sorted, [statistics.median(ranks[r]) for r in ranks_sorted],
                                marker="o", markersize=3, label=f"B={batch}")
                for rank in ranks_sorted:
                    ax.scatter([rank] * len(ranks[rank]), ranks[rank], s=10, alpha=.35, color=line.get_color())
            ax.set(title=title, xlabel="Draft rank", ylim=(-.03, 1.03))
            ax.grid(alpha=.2)
        axes[0].set_ylabel("Empirical survival probability")
        axes[-1].legend(fontsize=7)
        fig.suptitle(f"{Path(key[0]).name} | {key[3]} | {_execution(key)} | {key[-3]} | gamma={key[-2]}, budget={key[-1]}", fontsize=10)
        fig.text(.01, .01, "Uncensored full-B decode requests; lines are run medians, points are run values.", fontsize=8)
        fig.tight_layout(rect=(0, .04, 1, .94))
        files.extend(_save(fig, output, "F5_survival_" + _tag(key), formats))
        plt.close(fig)
    return files


def plot_f5_tracking(rows: list[dict], output: Path, formats: list[str], plt) -> list[str]:
    groups = defaultdict(list)
    for row in rows:
        # Native fallback alpha is cumulative conditional acceptance, not EWMA.
        if row.get("mode") == "adaptive" and _finite(row.get("alpha_after_update")) is not None:
            key = tuple(row.get(k, "unspecified") for k in (*REGIME_FIELDS, "batch", "configured_gamma"))
            groups[key].append(row)
    files = []
    for key, values in sorted(groups.items()):
        sources = defaultdict(list)
        for row in values:
            sources[row["source"]].append(row)
        fig, axes = plt.subplots(2, 1, figsize=(7.5, 5), sharex=True)
        for run_index, (source, observations) in enumerate(sorted(sources.items())):
            observed = [_finite(row.get("conditional_alpha_observed")) for row in observations]
            after = [_finite(row.get("alpha_after_update")) for row in observations]
            x = list(range(len(observations)))
            line, = axes[0].plot(x, after, label=f"run {run_index}: logged conditional EWMA", lw=1)
            valid_alpha = [(i, value) for i, value in enumerate(observed) if value is not None]
            axes[0].scatter([v[0] for v in valid_alpha], [v[1] for v in valid_alpha],
                            alpha=.15, s=5, color=line.get_color())
            errors = [_finite(row.get("output_prediction_error")) for row in observations]
            valid = [(i, error) for i, error in enumerate(errors) if error is not None]
            if valid:
                axes[1].plot([v[0] for v in valid], [v[1] for v in valid], alpha=.7, lw=.6,
                             label=f"run {run_index}: A prediction before step - observed A")
        axes[0].set(ylabel="Conditional alpha", ylim=(-.03, 1.03))
        axes[0].legend(fontsize=7)
        axes[1].axhline(0, color="black", lw=.5)
        axes[1].set(xlabel="Eligible decode step, concatenated waves within each run", ylabel="Output tokens / request error")
        if axes[1].get_legend_handles_labels()[0]:
            axes[1].legend(fontsize=7)
        for ax in axes:
            ax.grid(alpha=.2)
        fig.suptitle(f"{Path(key[0]).name} | {_execution(key)} | adaptive | B={key[-2]}", fontsize=10)
        fig.text(.01, .01, "Only conditional-alpha records are available; complete prefix/availability EWMA snapshots are not inferred.", fontsize=7)
        fig.tight_layout(rect=(0, .04, 1, .94))
        files.extend(_save(fig, output, "F5_tracking_" + _tag(key), formats))
        plt.close(fig)
    return files


def plot_f6(rows: list[dict], output: Path, formats: list[str], plt) -> list[str]:
    groups = defaultdict(list)
    for row in rows:
        if row.get("phase") == "decode":
            groups[tuple(row.get(k, "unspecified") for k in REGIME_FIELDS)].append(row)
    files = []
    metrics = [("policy_choose_us_mean", "Policy choose CPU (microseconds)"),
               ("control_us_mean", "Controller wrapper CPU (microseconds)"),
               ("discarded_available_token_fraction", "Discarded / available draft tokens")]
    for key, values in sorted(groups.items()):
        present = [(metric, title) for metric, title in metrics if any(_finite(r.get(metric)) is not None for r in values)]
        if not present:
            continue
        fig, axes = plt.subplots(1, len(present), figsize=(4.2 * len(present), 3.7), squeeze=False)
        for ax, (metric, title) in zip(axes[0], present):
            series = defaultdict(lambda: defaultdict(list))
            for row in values:
                value = _finite(row.get(metric))
                if value is not None:
                    series[series_label(row)][int(row["batch"])].append(value)
            for label, batches in sorted(series.items()):
                b = sorted(batches)
                line, = ax.plot(b, [statistics.median(batches[x]) for x in b], marker="o", markersize=3, label=label)
                for x in b:
                    ax.scatter([x] * len(batches[x]), batches[x], s=12, alpha=.35, color=line.get_color())
            _finish_axis(ax)
            ax.set_ylabel(title)
            if "fraction" in metric:
                ax.set_ylim(-.03, 1.03)
            ax.legend(fontsize=7)
        fig.suptitle(f"{Path(key[0]).name} | {key[3]} | {_execution(key)} | measured decode control", fontsize=10)
        fig.text(.01, .01, "CPU timing and discarded-token counts only; these do not measure wasted GPU draft-kernel time or energy.", fontsize=7)
        fig.tight_layout(rect=(0, .05, 1, .94))
        files.extend(_save(fig, output, "F6_control_" + _tag(key), formats))
        plt.close(fig)
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--performance", nargs="*", default=[])
    parser.add_argument("--analysis", nargs="*", default=[], help="build_calibration output directories")
    parser.add_argument("--static-selection", help="JSON selected_gamma + selection_source_files")
    parser.add_argument("--output", required=True)
    parser.add_argument("--formats", nargs="+", choices=["svg", "pdf", "png"], default=["svg", "pdf"])
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    runs, excluded = load_runs(args.performance) if args.performance else ([], [])
    selection = selected_static(args.static_selection, runs)
    summary = summarize(runs)
    acceptance, tracking, control = [], [], []
    for directory in sorted(set(map(Path, args.analysis))):
        acceptance.extend(read_csv(directory / "acceptance_survival.csv"))
        tracking.extend(read_csv(directory / "acceptance_tracking.csv"))
        control.extend(read_csv(directory / "control_overhead.csv"))
    write_csv(output / "F3_run_values.csv", runs)
    write_csv(output / "F3_summary.csv", summary)
    figures = {
        "F3": plot_f3(summary, output, args.formats, selection, plt),
        "F5_survival": plot_f5_survival(acceptance, output, args.formats, plt),
        "F5_tracking": plot_f5_tracking(tracking, output, args.formats, plt),
        "F6": plot_f6(control, output, args.formats, plt)}
    manifest = {"figures": figures, "missing_or_skipped": [k for k, v in figures.items() if not v],
        "performance_exclusions": excluded, "static_selection": selection,
        "evidence_kind": "plots_of_supplied_measured_runs_only",
        "F3_boundary": "per-B static envelope is descriptive post-hoc selection, not a single best fixed gamma",
        "F5_boundary": "conditional alpha logged; no invented prefix/availability EWMA trajectory",
        "F6_boundary": "CPU timings and token counts; no inferred draft GPU time or energy"}
    (output / "controller_figures.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
