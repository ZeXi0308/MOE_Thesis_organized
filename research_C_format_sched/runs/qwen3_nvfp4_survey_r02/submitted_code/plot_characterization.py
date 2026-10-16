#!/usr/bin/env python3
"""Plot observed two-format fused-MoE characterization, never serving results.

Default input: runs/twoformat_survey_r01/*_bf16.jsonl and *_fp8.jsonl.
Default grid: 1,8,32,128,512,2048,8192 input tokens; two models, two routes.
Only complete, status=ok records in the selected timing mode are plotted.
F1 shows actual medians and min/max of in-process latency repeats. F2 shows
total and idle-subtracted device energy per SINGLE MoE-layer input token;
the energy window is not repeated five times, so it has no invented whiskers.

Run again after completed remote records have been copied locally::

    python code/plot_characterization.py

Output: figures/F1_latency.{pdf,png}, F2_energy.{pdf,png}, summary.csv.
No fitted curves, interpolated crossovers, or serving gains are calculated.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import statistics
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
MODELS = ("olmoe", "qwen3")
MODEL_LABELS = {"olmoe": "OLMoE expert shape", "qwen3": "Qwen3-30B-A3B expert shape"}
FORMATS = ("bf16", "fp8")
ROUTES = ("uniform", "zipf")
MODEL_COUNTS = {"olmoe": (64, 8), "qwen3": (128, 8)}
COLORS = {"bf16": "#0072B2", "fp8": "#D55E00"}
STYLES = {"uniform": "-", "zipf": "--"}
MARKERS = {"uniform": "o", "zipf": "^"}
EXPECTED_TOKENS = (1, 8, 32, 128, 512, 2048, 8192)


def number(value, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    if not math.isfinite(value) or (positive and value <= 0):
        return None
    return value


@dataclass
class Observation:
    model: str
    fmt: str
    routing: str
    tokens: int
    mean: float
    latency: float
    latency_min: float
    latency_max: float
    total_j_per_input: float | None
    dynamic_j_per_input: float | None
    row: dict

    @property
    def key(self):
        return self.model, self.fmt, self.routing, self.tokens


def observation(record, source, line_number):
    model, fmt = record["model_shape"], record["format"]
    route, shape = record["routing"], record["shape"]
    kind = route["kind"]
    if model not in MODELS or fmt not in FORMATS or kind not in ROUTES:
        raise ValueError("Record is outside the requested two-model/two-format/two-route scope")
    tokens = record["tokens"]
    if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens < 1:
        raise ValueError("tokens must be a positive integer")
    experts, topk = MODEL_COUNTS[model]
    if (shape["experts"], shape["top_k"]) != (experts, topk):
        raise ValueError(f"Unexpected E/top-k for {model}: {shape}")
    counts = route["counts"]
    if len(counts) != experts or any(not isinstance(x, int) or x < 0 for x in counts):
        raise ValueError("Routing counts are not a complete expert histogram")
    if sum(counts) != tokens * topk:
        raise ValueError("Routing assignments disagree with input tokens * top-k")
    mean = sum(counts) / experts
    if not math.isclose(mean, route["mean_per_expert"], rel_tol=1e-10):
        raise ValueError("mean_per_expert does not include all experts")
    raw = record["latency_ms"]
    if not raw or any(number(x, positive=True) is None for x in raw):
        raise ValueError("Missing or invalid raw latency trials")
    if len(raw) != record.get("repeats", len(raw)):
        raise ValueError("Raw latency count disagrees with repeats")
    latency = statistics.median(raw)
    if not math.isclose(latency, record["latency_ms_median"], rel_tol=1e-7, abs_tol=1e-10):
        raise ValueError("Recorded median disagrees with raw trials")
    if record.get("output", {}).get("finite") is False:
        raise ValueError("Nonfinite output cannot be plotted as a valid measurement")

    power = record.get("power") or {}
    iterations = number(record.get("power_iterations"), positive=True)
    energy = number(power.get("energy_j"))
    counter = number(power.get("counter_energy_j"))
    sampled = number(power.get("sampled_energy_j"))
    method = power.get("method", "not_measured")
    # Use the benchmark's primary energy measurement. If a valid counter is
    # recorded, enforce that it is actually the primary quantity rather than
    # silently plotting the different sampled-power integral.
    if power.get("counter_valid") and counter is not None:
        if energy is None or not math.isclose(energy, counter, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("Valid energy counter disagrees with primary energy_j")
    if power.get("workload_failed"):
        energy = None
    dynamic = number(power.get("dynamic_energy_j")) if energy is not None else None
    total_per_input = energy / iterations / tokens if energy is not None and iterations else None
    dynamic_per_input = dynamic / iterations / tokens if dynamic is not None and iterations else None
    for field, calculated in (("energy_j_per_token", total_per_input),
                              ("dynamic_energy_j_per_token", dynamic_per_input)):
        reported = number(record.get(field))
        if reported is not None and calculated is not None and not math.isclose(
            reported, calculated, rel_tol=1e-8, abs_tol=1e-12
        ):
            raise ValueError(f"{field} disagrees with energy / calls / input tokens")
    difference = ((sampled - counter) / counter * 100
                  if sampled is not None and counter is not None and counter != 0 else None)
    backend = record.get("backend") or {}
    env = record.get("environment") or {}
    row = {
        "source_file": str(source.resolve()), "source_line": line_number,
        "started_utc": record.get("started_utc", ""), "status": "ok",
        "model_shape": model, "format": fmt, "routing": kind,
        "input_tokens": tokens, "experts": experts, "top_k": topk,
        "mean_per_expert_including_inactive": mean,
        "active_experts": sum(x > 0 for x in counts), "max_per_expert": max(counts),
        "latency_ms_median": latency, "latency_ms_min": min(raw),
        "latency_ms_max": max(raw), "latency_repeat_count": len(raw),
        "latency_ms_raw": json.dumps(raw, separators=(",", ":")),
        "repetition_scope": record.get("repetition_scope", "not_recorded"),
        "timing_mode": record.get("timing_mode", "not_recorded"),
        "iterations_per_repeat": record.get("iterations_per_repeat"),
        "total_j_per_layer_input_token": total_per_input,
        "idle_subtracted_j_per_layer_input_token": dynamic_per_input,
        "energy_method": method, "power_window_elapsed_s": power.get("elapsed_s"),
        "power_workload_elapsed_s": (power.get("boundary") or {}).get("workload_elapsed_s"),
        "power_iterations": record.get("power_iterations"),
        "primary_energy_j": energy, "idle_subtracted_energy_j": dynamic,
        "idle_power_w": power.get("idle_power_w"),
        "counter_valid": power.get("counter_valid"), "counter_energy_j": counter,
        "sampled_energy_j": sampled,
        "sampled_minus_counter_pct": difference,
        "absolute_sampled_counter_difference_pct": abs(difference) if difference is not None else None,
        "power_sample_count": (power.get("sampling") or {}).get("count"),
        "power_sample_interval_s": (power.get("sampling") or {}).get("requested_interval_s"),
        "power_warnings": json.dumps(power.get("warnings", []), ensure_ascii=False),
        "weight_source": backend.get("weight_source", "not_recorded"),
        "requested_backend": backend.get("requested_backend", "not_recorded"),
        "routing_provenance": route.get("provenance", "not_recorded"),
        "zipf_alpha": route.get("zipf_alpha"), "gpu": env.get("gpu", "not_recorded"),
        "torch": env.get("torch"), "vllm": env.get("vllm"), "triton": env.get("triton"),
        "artifact_sha256": backend.get("artifact_sha256", ""),
    }
    return Observation(model, fmt, kind, tokens, mean, latency, min(raw), max(raw),
                       total_per_input, dynamic_per_input, row)


def load_observations(paths, timing_mode):
    found, skipped, incomplete = {}, [], []
    for path in paths:
        # A transfer may end inside a JSON line. Ignore only an incomplete
        # final line and report it; malformed completed records are an error.
        lines = path.read_text().splitlines(keepends=True)
        for index, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                if index == len(lines) and not line.endswith("\n"):
                    incomplete.append(f"{path.name}:{index}")
                    continue
                raise ValueError(f"Malformed JSON at {path}:{index}") from exc
            if record.get("status") != "ok":
                skipped.append(f"{path.name}:{index}: status={record.get('status')}")
                continue
            if record.get("timing_mode") != timing_mode:
                skipped.append(f"{path.name}:{index}: timing_mode={record.get('timing_mode')}")
                continue
            try:
                point = observation(record, path, index)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid status=ok record at {path}:{index}: {exc}") from exc
            if point.key in found:
                other = found[point.key].row
                raise ValueError(
                    f"Duplicate measurement {point.key}: {other['source_file']}:{other['source_line']} "
                    f"and {path}:{index}; select one run rather than silently choosing a result"
                )
            found[point.key] = point
    return found, skipped, incomplete


def chart_style():
    # Keep optional plotting dependencies out of the reader/CSV code path.
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "format_sched_mpl"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11,
        "axes.labelsize": 10, "legend.fontsize": 9, "pdf.fonttype": 42,
        "ps.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
        "savefig.dpi": 180,
    })
    return plt


def coverage_text(points, model, expected, attribute):
    lines = []
    for fmt in FORMATS:
        for route in ROUTES:
            absent = [token for token in expected
                      if (model, fmt, route, token) not in points
                      or getattr(points[model, fmt, route, token], attribute) is None]
            if absent:
                lines.append(f"{fmt.upper()} {route}: missing {len(absent)}/{len(expected)}")
    return "\n".join(lines)


def draw_series(ax, points, model, expected, attribute, *, ranges=False):
    experts, topk = MODEL_COUNTS[model]
    plotted = 0
    for fmt in FORMATS:
        for route in ROUTES:
            series = [points.get((model, fmt, route, token)) for token in expected]
            xs = [token * topk / experts for token in expected]
            ys = [getattr(p, attribute) if p is not None and getattr(p, attribute) is not None
                  else math.nan for p in series]
            # NaN explicitly breaks a line across an unmeasured scheduled point.
            ax.plot(xs, ys, color=COLORS[fmt], linestyle=STYLES[route],
                    marker=MARKERS[route], markersize=4.1, linewidth=1.35,
                    label=f"{fmt.upper()} {route}")
            if ranges:
                valid = [p for p in series if p is not None]
                if valid:
                    ax.errorbar([p.mean for p in valid], [p.latency for p in valid],
                        yerr=[[p.latency - p.latency_min for p in valid],
                              [p.latency_max - p.latency for p in valid]],
                        fmt="none", color=COLORS[fmt], capsize=2.5,
                        elinewidth=0.85, alpha=0.85)
            plotted += sum(math.isfinite(y) for y in ys)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Mean routed tokens / expert (all experts, including inactive)")
    ax.grid(True, which="major", linewidth=0.5, alpha=0.25)
    ax.grid(True, which="minor", linewidth=0.3, alpha=0.08)
    missing = coverage_text(points, model, expected, attribute)
    if missing:
        ax.text(0.02, 0.02, missing, transform=ax.transAxes, fontsize=7.5,
                va="bottom", color="#555555",
                bbox={"facecolor": "white", "edgecolor": "#dddddd", "alpha": 0.9, "pad": 3})
    if not plotted:
        ax.text(0.5, 0.56, "No valid measurements available", transform=ax.transAxes,
                ha="center", va="center", color="#777777")
    return plotted


def save_figure(fig, out_dir, stem):
    for suffix in ("pdf", "png"):
        output = out_dir / f"{stem}.{suffix}"
        temporary = output.with_name(output.stem + ".tmp." + suffix)
        fig.savefig(temporary, bbox_inches="tight", facecolor="white")
        temporary.replace(output)


def make_figures(points, expected, out_dir, timing_mode, extra_note):
    plt = chart_style()
    repeat_counts = sorted({p.row["latency_repeat_count"] for p in points.values()})
    repeats = "/".join(map(str, repeat_counts))
    mode_text = {"cuda_graph_replay": "CUDA graph replay",
                 "eager_cuda_events": "Eager CUDA events"}.get(timing_mode, timing_mode)
    source_types = {p.row["weight_source"] for p in points.values()}
    weights_text = "Synthetic resident weights" if all(
        str(x).startswith("synthetic") for x in source_types
    ) else "Weight provenance is recorded in summary.csv"
    description = f"{mode_text}; {weights_text}; constructed uniform / Zipf routes"
    fig, axes = plt.subplots(1, 2, figsize=(11.9, 5.2), sharey=True)
    for ax, model in zip(axes, MODELS):
        draw_series(ax, points, model, expected, "latency", ranges=True)
        ax.set_yscale("log")
        ax.set_title(MODEL_LABELS[model], pad=9)
    axes[0].set_ylabel("Single MoE-layer latency (ms)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.91),
               ncol=4, frameon=False)
    fig.suptitle("F1 | Observed fused-MoE layer latency on SM120", fontsize=14, y=1.0)
    fig.text(0.5, 0.936, description, ha="center", fontsize=9, color="#555555")
    footer = (
        f"Markers: medians; whiskers: min-max of {repeats} in-process timing repeats, not independent runs.\n"
        "Segments join adjacent measured loads only; gaps remain missing. No fitted curves or crossover estimates."
    )
    if extra_note:
        footer += "\n" + extra_note
    fig.text(0.07, 0.013, footer, fontsize=8.1, color="#555555", va="bottom")
    fig.subplots_adjust(top=0.80, bottom=0.23, left=0.08, right=0.98, wspace=0.13)
    save_figure(fig, out_dir, "F1_latency")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11.9, 8.0), sharex=True, sharey="row")
    metrics = (("total_j_per_input", "Total device energy"),
               ("dynamic_j_per_input", "Idle-subtracted device energy"))
    for row_index, (attribute, metric_label) in enumerate(metrics):
        values = [getattr(p, attribute) for p in points.values()
                  if getattr(p, attribute) is not None]
        nonpositive = any(v <= 0 for v in values)
        for column, model in enumerate(MODELS):
            ax = axes[row_index, column]
            draw_series(ax, points, model, expected, attribute)
            if nonpositive:
                positive = [abs(v) for v in values if v != 0]
                ax.set_yscale("symlog", linthresh=min(positive) if positive else 1e-6)
                if column == 0:
                    ax.text(0.98, 0.98, "symlog; nonpositive values retained", transform=ax.transAxes,
                            ha="right", va="top", fontsize=7.5, color="#555555")
            else:
                ax.set_yscale("log")
            if row_index == 0:
                ax.set_title(MODEL_LABELS[model], pad=9)
                ax.set_xlabel("")
            if column == 0:
                ax.set_ylabel(metric_label + "\n(J / MoE-layer input token)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.935),
               ncol=4, frameon=False)
    fig.suptitle("F2 | Device energy per input token for one MoE layer", fontsize=14, y=1.0)
    fig.text(0.5, 0.957, description, ha="center", fontsize=9, color="#555555")
    methods = sorted({p.row["energy_method"] for p in points.values()
                      if p.total_j_per_input is not None})
    method_text = ", ".join(methods) if methods else "no energy measurements yet"
    footer = (
        "Each marker uses one measured power window / kernel calls / input tokens; energy has no repeat error bars.\n"
        "These are single-layer input tokens, not generated tokens or serving J/token. Missing points break lines.\n"
        f"Primary energy: {method_text}. Counter vs sampled-integral differences remain in summary.csv."
    )
    if extra_note:
        footer += "\n" + extra_note
    fig.text(0.07, 0.01, footer, fontsize=8.0, color="#555555", va="bottom")
    fig.subplots_adjust(top=0.865, bottom=0.18, left=0.085, right=0.98, hspace=0.12, wspace=0.13)
    save_figure(fig, out_dir, "F2_energy")
    plt.close(fig)


def write_summary(points, path):
    rows = [p.row for p in sorted(points.values(), key=lambda p: p.key)]
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs/twoformat_survey_r01")
    parser.add_argument("--inputs", type=Path, nargs="+", help="Explicit JSONL files instead of default globs")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "figures")
    parser.add_argument("--expected-tokens", default=",".join(map(str, EXPECTED_TOKENS)))
    parser.add_argument("--timing-mode", default="cuda_graph_replay",
                        choices=("cuda_graph_replay", "eager_cuda_events"))
    parser.add_argument("--csv-only", action="store_true", help="Validate/read and emit CSV without matplotlib")
    args = parser.parse_args()
    expected = sorted({int(x) for x in args.expected_tokens.split(",")})
    if not expected or min(expected) < 1:
        parser.error("Expected token counts must be positive")
    paths = sorted(set(args.inputs or [p for fmt in FORMATS for p in args.run_dir.glob(f"*_{fmt}.jsonl")]))
    if not paths:
        parser.error("No input JSONL files found; sync completed runs locally first")
    points, skipped, incomplete = load_observations(paths, args.timing_mode)
    if not points:
        parser.error("No valid status=ok records in the requested timing mode; no figures overwritten")
    unexpected = sorted({p.tokens for p in points.values()} - set(expected))
    if unexpected:
        parser.error(f"Observed token counts outside --expected-tokens: {unexpected}; declare the actual grid")
    environment_keys = {(p.row["gpu"], p.row["torch"], p.row["vllm"], p.row["triton"])
                        for p in points.values()}
    if len(environment_keys) != 1:
        parser.error("Input records mix GPU/software environments; select comparable files")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_summary(points, args.out_dir / "summary.csv")
    note = f"Coverage: {len(points)}/{len(expected) * len(MODELS) * len(FORMATS) * len(ROUTES)} planned cases."
    if skipped or incomplete:
        note += f" Excluded: {len(skipped)} non-ok/other-mode records, {len(incomplete)} incomplete trailing lines."
    if not args.csv_only:
        make_figures(points, expected, args.out_dir, args.timing_mode, note)
    for reason in [*skipped, *(f"Incomplete trailing JSON: {x}" for x in incomplete)]:
        print(reason, file=sys.stderr)
    print(json.dumps({"records": len(points), "expected_cases": len(expected) * 8,
                      "energy_records": sum(p.total_j_per_input is not None for p in points.values()),
                      "output_directory": str(args.out_dir.resolve()),
                      "csv_only": args.csv_only}, ensure_ascii=False))


if __name__ == "__main__":
    main()
