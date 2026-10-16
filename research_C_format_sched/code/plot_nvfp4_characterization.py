#!/usr/bin/env python3
"""Plot measured Qwen3 three-format latency and NVFP4-only energy v2.

BF16/FP8 use synthetic weights; NVFP4 uses calibrated checkpoint layer 0.
This is a same-shape kernel characterization, not a same-weight baseline or
quality/serving comparison. Only measured points are drawn; no crossover is
interpolated. Latency whiskers span five within-process repetitions. Energy
has one measurement window per point, so no uncertainty bars are invented.

Run: /private/tmp/format_sched_plot_venv/bin/python code/plot_nvfp4_characterization.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import tempfile


ROOT = Path(__file__).resolve().parents[1]
TOKENS = (1, 8, 32, 128, 512, 2048, 8192)
FORMATS = ("bf16", "fp8", "nvfp4")
ROUTES = ("uniform", "zipf")
SHAPE = {"d": 2048, "intermediate": 768, "experts": 128, "top_k": 8}
ENERGY_PROTOCOL = "idle_baseline_then_continuous_workload_warmup_v2"
COLORS = {"bf16": "#61768A", "fp8": "#1876AC", "nvfp4": "#BD561B"}
MARKERS = {"bf16": "s", "fp8": "o", "nvfp4": "D"}
LABELS = {"bf16": "BF16 Triton (synthetic)",
          "fp8": "FP8 W8A8 Triton (synthetic)",
          "nvfp4": "NVFP4 CUTLASS (calibrated layer 0)"}


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-8, abs_tol=1e-12)


def relative(path):
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def load_inputs(reference_dir, nvfp4_dir):
    """Require the complete grid and check source hashes and measurement units."""
    points, sources = {}, []
    for fmt in FORMATS:
        directory = nvfp4_dir if fmt == "nvfp4" else reference_dir
        hashes = json.loads((directory / "SHA256.json").read_text())
        for route in ROUTES:
            path = directory / f"qwen3_{route}_{fmt}.jsonl"
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if hashes.get(path.name) != digest:
                raise ValueError(f"Source SHA-256 mismatch: {path}")
            sources.append({"path": relative(path), "sha256": digest})
            for line_number, line in enumerate(path.read_text().splitlines(), 1):
                row = json.loads(line)
                if (row["status"] != "ok" or row["model_shape"] != "qwen3"
                        or row["format"] != fmt or row["routing"]["kind"] != route
                        or row["shape"] != SHAPE or row["timing_mode"] != "cuda_graph_replay"):
                    raise ValueError(f"Unexpected measurement contract: {path}:{line_number}")
                if not row["output"]["finite"] or not row["cuda_graph_eager_output_match"]:
                    raise ValueError(f"Invalid output/graph check: {path}:{line_number}")
                batch, latency = row["tokens"], row["latency_ms"]
                counts = row["routing"]["counts"]
                if len(counts) != 128 or sum(counts) != batch * 8 or any(c < 0 for c in counts):
                    raise ValueError("Invalid routing assignment counts")
                if not close(sum(counts) / 128, row["routing"]["mean_per_expert"]):
                    raise ValueError("X-axis mean does not include all experts")
                if len(latency) != 5 or row["repeats"] != 5 or any(
                        not math.isfinite(v) or v <= 0 for v in latency):
                    raise ValueError("Expected five positive within-process latency repeats")
                if not close(statistics.median(latency), row["latency_ms_median"]):
                    raise ValueError("Recorded latency median disagrees with repetitions")
                source = row["backend"]["weight_source"]
                if fmt != "nvfp4" and not source.startswith("synthetic"):
                    raise ValueError("Reference weights no longer match the synthetic figure labels")
                if fmt == "nvfp4":
                    backend, power = row["backend"], row["power"]
                    if (source != "calibrated_checkpoint" or backend["layer"] != 0
                            or backend["requested_backend"] != "cutlass"):
                        raise ValueError("Expected calibrated layer-0 CUTLASS NVFP4")
                    if power.get("energy_protocol") != ENERGY_PROTOCOL:
                        raise ValueError("F2 accepts energy v2 only; do not mix energy v1")
                    if power["workload_failed"] or not power["counter_valid"]:
                        raise ValueError("Expected a valid NVML total-energy counter window")
                    if not close(power["energy_j"], power["counter_energy_j"]):
                        raise ValueError("Primary energy differs from the valid counter")
                    for field, measured in (("energy_j_per_token", power["energy_j"]),
                                            ("dynamic_energy_j_per_token", power["dynamic_energy_j"])):
                        expected = measured / row["power_iterations"] / batch
                        if not math.isfinite(expected) or expected <= 0 or not close(expected, row[field]):
                            raise ValueError("Invalid per-layer-input-token energy normalization")
                key = (route, fmt, batch)
                if key in points:
                    raise ValueError(f"Duplicate observation: {key}")
                points[key] = row
    expected = {(r, f, b) for r in ROUTES for f in FORMATS for b in TOKENS}
    if set(points) != expected:
        raise ValueError(f"Incomplete or unexpected grid: {set(points) ^ expected}")
    for route in ROUTES:
        for batch in TOKENS:
            rows = [points[route, fmt, batch] for fmt in FORMATS]
            if any(r["routing"]["counts"] != rows[0]["routing"]["counts"] for r in rows[1:]):
                raise ValueError("Formats have different actual routing histograms")
    environments = [json.dumps(r["environment"], sort_keys=True) for r in points.values()]
    if len(set(environments)) != 1:
        raise ValueError("Measured hardware/software environments differ")
    return points, sources


def make_summary(points, sources):
    first = points["uniform", "nvfp4", 1]
    summary = {
        "schema": 1, "model": "Qwen3-30B-A3B expert shape", "shape": SHAPE,
        "timing_mode": "cuda_graph_replay", "environment": first["environment"],
        "sources": sources, "input_tokens": TOKENS,
        "mean_assignments_per_all_experts": [b / 16 for b in TOKENS],
        "routing_histograms_match_across_formats": True,
        "latency_repetition_scope": "five repeats within one process/residency; not five independent runs",
        "comparison_scope": "same shape and graph mode; synthetic BF16/FP8 versus calibrated layer-0 NVFP4; not same weights",
        "same_weight_comparison": False, "quality_claim": False, "serving_claim": False,
        "awq_status": "not measured in this figure",
        "crossover_estimation": "none; sampled winners and adjacent observed rank-change brackets only",
        "energy_scope": "NVFP4 only; energy v2; one device-energy window per point; no repeat-based uncertainty",
        "energy_protocol": ENERGY_PROTOCOL,
        "energy_v1_reference_comparison": "excluded",
        "nvfp4_checkpoint": {k: first["backend"][k] for k in (
            "checkpoint", "checkpoint_revision", "base_checkpoint", "base_checkpoint_revision",
            "exact_base_revision_known", "layer", "artifact_sha256")},
        "routes": {},
    }
    for route in ROUTES:
        entry = {"zipf_alpha": points[route, "nvfp4", 1]["routing"]["zipf_alpha"], "formats": {}}
        for fmt in FORMATS:
            rows = [points[route, fmt, b] for b in TOKENS]
            entry["formats"][fmt] = {
                "weight_source": rows[0]["backend"]["weight_source"],
                "latency_ms_median": [statistics.median(r["latency_ms"]) for r in rows],
                "latency_ms_within_run_min": [min(r["latency_ms"]) for r in rows],
                "latency_ms_within_run_max": [max(r["latency_ms"]) for r in rows],
            }
        entry["fastest_observed_format"] = [
            min(FORMATS, key=lambda f: points[route, f, b]["latency_ms_median"]) for b in TOKENS]
        entry["fp8_latency_divided_by_nvfp4"] = [
            points[route, "fp8", b]["latency_ms_median"] / points[route, "nvfp4", b]["latency_ms_median"]
            for b in TOKENS]
        winners = entry["fastest_observed_format"]
        entry["adjacent_observed_rank_change_token_brackets"] = [
            [a, b] for a, b, wa, wb in zip(TOKENS, TOKENS[1:], winners, winners[1:]) if wa != wb]
        rows = [points[route, "nvfp4", b] for b in TOKENS]
        entry["nvfp4_energy"] = {
            "total_j_per_layer_input_token": [r["energy_j_per_token"] for r in rows],
            "idle_subtracted_j_per_layer_input_token": [r["dynamic_energy_j_per_token"] for r in rows],
            "window_s": [r["power"]["elapsed_s"] for r in rows],
            "calls_per_window": [r["power_iterations"] for r in rows],
            "idle_power_w": [r["power"]["idle_power_w"] for r in rows],
            "counter_sampled_difference_pct": [
                100 * (r["power"]["sampled_energy_j"] / r["power"]["counter_energy_j"] - 1) for r in rows],
            "warnings": [r["power"].get("warnings", []) for r in rows],
            "method": "nvml_total_energy_counter",
        }
        summary["routes"][route] = entry
    return summary


def draw_figures(summary, output_dir):
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "format_sched_mpl"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.labelsize": 10, "axes.edgecolor": "#687483",
                         "axes.spines.right": False, "axes.spines.top": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 220})
    xs = summary["mean_assignments_per_all_experts"]

    def canvas(title, subtitle, ylabel):
        fig, axes = plt.subplots(1, 2, figsize=(12.6, 6.7), sharey=True)
        fig.subplots_adjust(left=.08, right=.98, bottom=.27, top=.71, wspace=.16)
        fig.suptitle(title, x=.08, y=.975, ha="left", fontsize=17, weight="bold")
        fig.text(.08, .917, subtitle, fontsize=10, color="#455161")
        for ax, route in zip(axes, ROUTES):
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_xlim(xs[0] / 1.55, xs[-1] * 1.55)
            ax.xaxis.set_major_locator(FixedLocator(xs))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda value, pos: f"{value:g}"))
            ax.xaxis.set_minor_locator(NullLocator())
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, pos: f"{value:g}"))
            ax.grid(axis="both", which="major", color="#DCE2E8", linewidth=.7, zorder=0)
            ax.tick_params(labelsize=9)
            ax.set_xlabel("Mean assignments / expert, B k / E (all E = 128)", labelpad=9)
            top = ax.secondary_xaxis("top", functions=(lambda x: x * 16, lambda x: x / 16))
            top.xaxis.set_major_locator(FixedLocator(TOKENS))
            top.xaxis.set_major_formatter(FuncFormatter(lambda value, pos: f"{value:g}"))
            top.xaxis.set_minor_locator(NullLocator())
            top.set_xlabel("Input tokens B (measured points)", labelpad=8, fontsize=9)
            top.tick_params(labelsize=9)
            text = "(a) Uniform routing" if route == "uniform" else "(b) Zipf routing, alpha = 1.2"
            ax.text(.035, .94, text, transform=ax.transAxes, va="top", fontsize=11, weight="bold",
                    bbox={"facecolor": "white", "alpha": .9, "edgecolor": "none", "pad": 3})
        axes[0].set_ylabel(ylabel, labelpad=10)
        return fig, axes

    fig, axes = canvas("Qwen3 expert latency: three measured formats",
                       "d = 2048, intermediate = 768, E = 128, top-k = 8  |  RTX PRO 6000 SM120  |  CUDA graph replay",
                       "Single MoE-layer latency (ms)")
    for ax, route in zip(axes, ROUTES):
        ax.set_ylim(.018, 5.0)
        ax.yaxis.set_major_locator(FixedLocator([.02, .05, .1, .2, .5, 1, 2, 5]))
        for fmt in FORMATS:
            data = summary["routes"][route]["formats"][fmt]
            med, lo, hi = (data[k] for k in ("latency_ms_median", "latency_ms_within_run_min", "latency_ms_within_run_max"))
            ax.errorbar(xs, med, yerr=[[m - l for m, l in zip(med, lo)], [h - m for h, m in zip(hi, med)]],
                        fmt=MARKERS[fmt], linestyle="none", color=COLORS[fmt], markersize=5,
                        capsize=3.5, capthick=1.1, elinewidth=1.2, label=LABELS[fmt], zorder=3)
        winners = summary["routes"][route]["fastest_observed_format"]
        win_text = [f"{fmt.upper()}: B = " + ", ".join(str(b) for b, f in zip(TOKENS, winners) if f == fmt)
                    for fmt in ("nvfp4", "fp8")]
        ax.text(.965, .055, "Fastest at sampled points\n" + "\n".join(win_text),
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5,
                color="#354150", bbox={"facecolor": "#F2F5F8", "edgecolor": "none", "pad": 6})
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.53, .88), ncol=3,
               frameon=False, fontsize=9.5, columnspacing=1.8, handletextpad=.45)
    fig.text(.08, .165, "Different weight sources: this is not a same-weight quality or serving comparison.", fontsize=10, weight="bold")
    fig.text(.08, .122, "Markers: medians; whiskers: min-max of five repeats within one process/residency (not independent runs).", fontsize=9)
    fig.text(.08, .080, "Measured B = 1-8192 only. Rank changes are bracketed by B = [1, 8] and [2048, 8192]; no crossover is fitted.", fontsize=9)
    fig.text(.08, .038, "Synthetic routing is identical across formats at each point. Calibrated NVFP4: NVIDIA Qwen3 layer 0. AWQ remains unmeasured.", fontsize=9, color="#455161")
    save(fig, output_dir / "F1_qwen_three_formats")
    plt.close(fig)

    fig, axes = canvas("Qwen3 calibrated NVFP4: device energy per input token",
                       "NVIDIA checkpoint layer 0  |  CUTLASS W4A4  |  Same Qwen3 shape and CUDA graph replay  |  Energy protocol v2 only",
                       "Single MoE-layer energy (mJ / input token)")
    for ax, route in zip(axes, ROUTES):
        ax.set_ylim(.065, 21)
        # Energy is largest at B=1; keep the panel label away from that point.
        ax.texts[0].set_position((.965, .94))
        ax.texts[0].set_horizontalalignment("right")
        ax.yaxis.set_major_locator(FixedLocator([.1, .2, .5, 1, 2, 5, 10, 20]))
        data = summary["routes"][route]["nvfp4_energy"]
        for field, color, marker, label in (
                ("total_j_per_layer_input_token", "#BD561B", "D", "Total device energy"),
                ("idle_subtracted_j_per_layer_input_token", "#167A73", "o", "Idle-subtracted device energy")):
            ax.plot(xs, [1000 * value for value in data[field]], linestyle="none", marker=marker,
                    color=color, markersize=5.5, label=label, zorder=3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.53, .88), ncol=2,
               frameon=False, fontsize=10, columnspacing=3)
    fig.text(.08, .165, "NVFP4 energy v2 only; older BF16/FP8 energy v1 is excluded from comparison.", fontsize=10, weight="bold")
    fig.text(.08, .122, "Primary measurement: NVML total-energy counter; per-case idle baseline, continuous-workload warmup, then a ~3 s window.", fontsize=9)
    fig.text(.08, .080, "One energy window per point; no repeated-window uncertainty is available. This is one MoE layer, not whole-model J/token.", fontsize=9)
    fig.text(.08, .038, "Points span measured B = 1-8192 (mean assignments 0.0625-512); no interpolation, quality claim, or serving gain. AWQ absent.", fontsize=9, color="#455161")
    save(fig, output_dir / "F2_qwen_nvfp4_energy")
    plt.close(fig)


def save(fig, stem):
    fig.savefig(stem.with_suffix(".pdf"), metadata={"Title": stem.name, "Creator": "plot_nvfp4_characterization.py"})
    fig.savefig(stem.with_suffix(".png"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, default=ROOT / "runs/twoformat_survey_r01")
    parser.add_argument("--nvfp4-dir", type=Path, default=ROOT / "runs/qwen3_nvfp4_survey_r02")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures")
    args = parser.parse_args()
    points, sources = load_inputs(args.reference_dir, args.nvfp4_dir)
    summary = make_summary(points, sources)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    draw_figures(summary, args.output_dir)
    out = args.output_dir / "qwen_nvfp4_characterization_summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(f"Wrote 42 latency points and 14 NVFP4 energy-v2 points: {out}")
    for route in ROUTES:
        print(route, "sampled winners", summary["routes"][route]["fastest_observed_format"])


if __name__ == "__main__":
    main()
