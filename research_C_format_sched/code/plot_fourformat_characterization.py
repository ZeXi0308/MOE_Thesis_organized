#!/usr/bin/env python3
"""Qwen3 four-format characterization from archived measurements only.

BF16/FP8 are synthetic; AWQ and NVFP4 are two different public calibrated
layer-0 checkpoints. This is not a same-weight, quality, or serving comparison.
Latency uses the same graph/event/repetition protocol; older BF16/FP8 energy
v1 is excluded from F2. F2 compares only the calibrated AWQ/NVFP4 energy v2.

Missing r06 observations remain null/blank and mark every output partial.
Failed rows remain in the summary. Duplicate successful points are rejected,
never resolved by selecting a faster result. Five raw latency repetitions,
actual profiler kernel names, source hashes, ratios and observed rank-change
brackets are retained. Repetition envelopes are not confidence intervals.

  /private/tmp/format_sched_plot_venv/bin/python code/plot_fourformat_characterization.py
"""
from __future__ import annotations

import argparse
import ast
import hashlib
from itertools import combinations
import json
import math
import os
from pathlib import Path
import statistics
import tempfile


ROOT = Path(__file__).resolve().parents[1]
TOKENS = (1, 8, 32, 128, 512, 2048, 8192)
ROUTES = ("uniform", "zipf")
FORMATS = ("bf16", "fp8", "awq", "nvfp4")
SHAPE = {"d": 2048, "intermediate": 768, "experts": 128, "top_k": 8}
ENERGY_V2 = "idle_baseline_then_continuous_workload_warmup_v2"
COLORS = {"bf16": "#6A7C8D", "fp8": "#1976AF", "awq": "#8754A7", "nvfp4": "#C26726"}
MARKERS = {"bf16": "s", "fp8": "o", "awq": "^", "nvfp4": "D"}
LABELS = {"bf16": "BF16 Triton (synthetic)", "fp8": "FP8 W8A8 Triton (synthetic)",
          "awq": "AWQ W4A16 Marlin (public calibrated)", "nvfp4": "NVFP4 W4A4 CUTLASS (public calibrated)"}
LATENCY_FIELDS = ("timing_mode", "repetition_scope", "warmup_calls", "repeats", "cache_policy")
BF16_EPSILON = 2 ** -7
ENERGY_KEYS = ("j_per_input_token", "idle_subtracted_j_per_input_token")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative(path):
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(Path(path).resolve())


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-8, abs_tol=1e-12)


def read_archive(directory, formats, *, optional=False):
    """Only locally archived, hash-verified rows are admitted to comparisons."""
    info = {"directory": relative(directory), "availability": "available", "files": []}
    if not directory.is_dir() or not (directory / "SHA256.json").is_file():
        if not optional:
            raise FileNotFoundError(f"Missing archived measurements/hash inventory: {directory}")
        info["availability"] = "missing_directory" if not directory.is_dir() else "awaiting_hash_inventory"
        return [], [], info
    hashes = json.loads((directory / "SHA256.json").read_text())
    info["hash_inventory_sha256"] = sha256(directory / "SHA256.json")
    if (directory / "status.json").is_file():
        if hashes.get("status.json") != sha256(directory / "status.json"):
            raise ValueError(f"Archived run status SHA-256 mismatch: {directory}")
        info["run_status"] = json.loads((directory / "status.json").read_text())
        command = info["run_status"].get("command", [])
        info["configured_graph_output_check"] = (command[command.index("--graph-output-check") + 1]
            if "--graph-output-check" in command else "exact")
    source = directory / "submitted_code/bench_expert_formats.py"
    if source.is_file():
        if hashes.get(str(source.relative_to(directory))) != sha256(source):
            raise ValueError(f"Archived benchmark source SHA-256 mismatch: {source}")
        tree = ast.parse(source.read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "event_time")
        info["event_time_ast_sha256"] = hashlib.sha256(ast.dump(function, include_attributes=False).encode()).hexdigest()
        info["benchmark_source_sha256"] = sha256(source)
        # The record's passed field incorporates graph-output finiteness in r06.
        # Verify that archived source fact instead of inventing an unlogged flag.
        expected = ast.dump(ast.parse("bool(torch.isfinite(graph_output).all()) and "
            "(relative_l2 <= eps / 8 and max_absolute <= eps * max_reference)", mode="eval").body)
        guards = [node.value for node in ast.walk(tree) if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "numeric_ok" for target in node.targets)]
        info["roundoff_guard_finite_source_verified"] = len(guards) == 1 and ast.dump(guards[0]) == expected
    rows, failures = [], []
    for fmt in formats:
        for route in ROUTES:
            path = directory / f"qwen3_{route}_{fmt}.jsonl"
            if not path.is_file():
                info["files"].append({"path": relative(path), "availability": "missing"})
                continue
            digest = sha256(path)
            if hashes.get(path.name) != digest:
                raise ValueError(f"Archived JSONL SHA-256 mismatch: {path}")
            info["files"].append({"path": relative(path), "sha256": digest, "availability": "verified"})
            for line_number, line in enumerate(path.read_text().splitlines(), 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                provenance = {"path": relative(path), "file_sha256": digest, "line": line_number,
                              "line_sha256": hashlib.sha256(line.encode()).hexdigest(), "archive": directory.name}
                if (row.get("model_shape") != "qwen3" or row.get("format") != fmt or
                        row.get("tokens") not in TOKENS or row.get("routing", {}).get("kind") != route):
                    raise ValueError(f"Unexpected row identity: {path}:{line_number}")
                if row.get("status") != "ok":
                    failures.append({"source": provenance, "raw_row": row})
                    continue
                profile = row.get("profile") or {}
                names = profile.get("kernel_names")
                if not isinstance(names, list) or not names or any(not isinstance(n, str) or not n for n in names):
                    raise ValueError(f"Successful row lacks actual profiler kernel names: {path}:{line_number}")
                original = Path(profile["trace"])
                trace = directory / original.parent.name / original.name
                if not trace.is_file() or hashes.get(str(trace.relative_to(directory))) != sha256(trace):
                    raise ValueError(f"Profiler trace is absent or differs from archived hash: {trace}")
                provenance.update(profile_trace=relative(trace), profile_trace_sha256=sha256(trace))
                rows.append((row, provenance))
    return rows, failures, info


def validate_graph_output(row, archive):
    exact = row.get("cuda_graph_eager_output_match")
    check = row.get("cuda_graph_output_check")
    if check is None:
        if exact is not True:
            raise ValueError("Legacy rows require the recorded exact graph/eager guard")
        return {"accepted": True, "policy": "legacy_exact", "bitwise_equal": True,
                "scope": "capture/replay numerical guard, not conversion correctness or task quality"}
    mode = check.get("mode")
    if (mode not in ("exact", "bf16-roundoff") or mode != archive.get("configured_graph_output_check") or
            check.get("passed") is not True or type(exact) is not bool or check.get("bitwise_equal") is not exact):
        raise ValueError("Graph output policy, configured command, passed flag or bitwise diagnostic disagrees")
    if mode == "exact":
        if not exact:
            raise ValueError("Exact graph output policy failed")
    else:
        values = [check.get(k) for k in ("relative_l2", "max_absolute_error", "max_reference_absolute")]
        if (not archive.get("roundoff_guard_finite_source_verified") or
                any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in values) or
                check.get("relative_l2_limit") != BF16_EPSILON / 8 or
                check.get("max_error_reference_scale_limit") != BF16_EPSILON):
            raise ValueError("Invalid/nonfinite graph error metrics or changed BF16-roundoff limits")
        rel_l2, max_abs, max_ref = values
        different = check.get("different_elements")
        if (rel_l2 > BF16_EPSILON / 8 or max_abs > BF16_EPSILON * max_ref or
                type(different) is not int or not 0 <= different <= row["tokens"] * SHAPE["d"] or
                exact != (different == 0) or exact != (max_abs == 0) or exact != (rel_l2 == 0)):
            raise ValueError("Recorded BF16-roundoff policy does not pass its original numerical bounds")
    return {"accepted": True, "policy": mode, "bitwise_equal": exact,
            "finite_graph_output_evidence": "passed=true plus hash-verified archived numeric_ok source guard",
            "independently_rechecked_recorded_error_bounds": mode == "bf16-roundoff",
            "scope": "capture/replay numerical guard, not conversion correctness or task quality"}


def validate_row(row, archive):
    latency = row["latency_ms"]
    fmt, batch = row["format"], row["tokens"]
    if (row["shape"] != SHAPE or row["timing_mode"] != "cuda_graph_replay" or
            row["repeats"] != 5 or row["warmup_calls"] != 10 or len(latency) != 5 or
            any(not math.isfinite(v) or v <= 0 for v in latency) or
            not close(statistics.median(latency), row["latency_ms_median"]) or
            row["output"]["finite"] is not True or row["output"]["shape"] != [batch, SHAPE["d"]]):
        raise ValueError(f"Invalid latency/shape/output contract: {fmt}/{batch}")
    validation = validate_graph_output(row, archive)
    counts = row["routing"]["counts"]
    if (len(counts) != 128 or any(type(c) is not int or c < 0 for c in counts) or
            sum(counts) != batch * 8 or not close(sum(counts) / 128, row["routing"]["mean_per_expert"])):
        raise ValueError("Invalid routed assignments; denominator must include all 128 experts")
    backend, power = row["backend"], row["power"]
    if fmt in ("bf16", "fp8"):
        if not backend["weight_source"].startswith("synthetic") or backend["requested_backend"] != "triton":
            raise ValueError("BF16/FP8 labels require the existing synthetic Triton measurements")
    else:
        if (backend["weight_source"] != "calibrated_checkpoint" or backend["layer"] != 0 or
                backend["requested_backend"] != ("marlin" if fmt == "awq" else "cutlass")):
            raise ValueError("Expected public calibrated layer-0 AWQ/Marlin or NVFP4/CUTLASS")
        if (power.get("energy_protocol") != ENERGY_V2 or power.get("workload_failed") is not False or
                power.get("counter_valid") is not True or not close(power["energy_j"], power["counter_energy_j"]) or
                power.get("energy_warmup_iterations", 0) <= 0):
            raise ValueError("F2 accepts valid counter-based, continuously warmed energy v2 only")
        for field, key in (("energy_j_per_token", "energy_j"), ("dynamic_energy_j_per_token", "dynamic_energy_j")):
            normalized = power[key] / row["power_iterations"] / batch
            if not math.isfinite(normalized) or normalized <= 0 or not close(normalized, row[field]):
                raise ValueError("Invalid per-layer-input-token energy normalization")
    return validation


def point_summary(row, source, validation):
    latency, power = row["latency_ms"], row["power"]
    energy = {key: power.get(key) for key in ("method", "elapsed_s", "energy_j", "dynamic_energy_j", "counter_energy_j",
        "sampled_energy_j", "counter_valid", "idle_power_w", "idle_energy_j", "warnings", "energy_protocol",
        "energy_warmup_iterations", "idle_settle_seconds", "boundary", "counter_start", "counter_end")}
    energy.update(calls_per_window=row["power_iterations"], j_per_input_token=row["energy_j_per_token"],
        idle_subtracted_j_per_input_token=row["dynamic_energy_j_per_token"],
        plotted_in_F2=row["format"] in ("awq", "nvfp4"),
        excluded_reason="older energy v1; not comparable to v2" if row["format"] in ("bf16", "fp8") else None,
        sampled_minus_counter_pct=100 * (power["sampled_energy_j"] / power["counter_energy_j"] - 1))
    return {"format": row["format"], "routing": row["routing"]["kind"], "tokens": row["tokens"],
        "mean_assignments_per_all_experts": row["tokens"] / 16, "source": source,
        "latency_ms_repetitions": latency, "wall_ms_per_call_repetitions": row["wall_ms_per_call"],
        "iterations_per_repeat": row["iterations_per_repeat"], "latency_ms_median": statistics.median(latency),
        "latency_ms_min": min(latency), "latency_ms_max": max(latency),
        "latency_protocol": {key: row[key] for key in LATENCY_FIELDS},
        "graph_output_validation": {key: value for key, value in row.items() if key.startswith("cuda_graph_")},
        "graph_output_acceptance": validation,
        "backend": row["backend"], "routing_detail": row["routing"],
        "profile": row["profile"], "energy": energy, "started_utc": row["started_utc"]}


def ratio(a, b):
    return {"latency_a_over_b": a["latency_ms_median"] / b["latency_ms_median"],
        "all_repeat_combination_envelope": [a["latency_ms_min"] / b["latency_ms_max"],
                                            a["latency_ms_max"] / b["latency_ms_min"]],
        "repeat_ranges_disjoint": a["latency_ms_max"] < b["latency_ms_min"] or b["latency_ms_max"] < a["latency_ms_min"],
        "scope": "ratio of medians and unpaired min/max envelope; not paired trials or a confidence interval"}


def failed_predecessor_runs(directories):
    records = []
    for directory in directories:
        if not directory.is_dir():
            continue
        path, inventory = directory / "status.json", directory / "SHA256.json"
        hashes = json.loads(inventory.read_text())
        if hashes.get(path.name) != sha256(path):
            raise ValueError(f"Failed-run status SHA-256 mismatch: {path}")
        if list(directory.glob("*.jsonl")):
            raise ValueError(f"Predecessor contains measurements; read explicitly instead of ignoring them: {directory}")
        status = json.loads(path.read_text())
        if status.get("state") != "FAILED":
            raise ValueError(f"Expected an archived failed predecessor, got {directory}")
        records.append({"path": relative(path), "sha256": sha256(path), "raw_status": status,
                        "measurement_rows": 0, "scope": "failed controller attempt, not a measured kernel result"})
    return records


def load_summary(args):
    points, failures, archives, environments = {}, [], [], []
    for directory, formats, optional in ((args.reference_dir, ("bf16", "fp8"), False),
                                         (args.nvfp4_dir, ("nvfp4",), False),
                                         (args.awq_initial_dir, ("awq",), False),
                                         (args.awq_resume_dir, ("awq",), True)):
        rows, failed, info = read_archive(directory, formats, optional=optional)
        failures.extend(failed)
        archives.append(info)
        for row, source in rows:
            validation = validate_row(row, info)
            key = row["routing"]["kind"], row["format"], row["tokens"]
            if key in points:
                raise ValueError(f"Duplicate successful point {key}; refusing to choose a favorable repeat")
            environments.append(row["environment"])
            points[key] = point_summary(row, source, validation)
    required = {(r, f, b) for r in ROUTES for f in ("bf16", "fp8", "nvfp4") for b in TOKENS}
    if required - points.keys():
        raise ValueError(f"Baseline/NVFP4 archived grid is incomplete: {sorted(required - points.keys())}")
    protocols = {json.dumps(p["latency_protocol"], sort_keys=True) for p in points.values()}
    if len(protocols) != 1 or len({json.dumps(env, sort_keys=True) for env in environments}) != 1:
        raise ValueError("Latency protocol or recorded hardware/software differs across sources")
    event_hashes = {a["event_time_ast_sha256"] for a in archives if "event_time_ast_sha256" in a}
    if len(event_hashes) > 1:
        raise ValueError("Native CUDA-event timing implementation changed across archives")
    for route in ROUTES:
        for batch in TOKENS:
            counts = [points[route, f, batch]["routing_detail"]["counts"] for f in FORMATS if (route, f, batch) in points]
            if any(c != counts[0] for c in counts[1:]):
                raise ValueError("Actual routing histograms differ across compared formats")
    for fmt in ("awq", "nvfp4"):
        identities = {json.dumps({k: p["backend"].get(k) for k in ("checkpoint", "checkpoint_revision", "artifact_sha256")}, sort_keys=True)
                      for p in points.values() if p["format"] == fmt}
        if len(identities) != 1:
            raise ValueError(f"{fmt} artifact identity changes across points; explicit analysis is required before merging")
    missing = [{"routing": r, "format": f, "tokens": b} for r in ROUTES for f in FORMATS for b in TOKENS if (r, f, b) not in points]
    nonexact = [{"routing": p["routing"], "tokens": p["tokens"], "format": p["format"],
                "source": p["source"], "raw_guard": p["graph_output_validation"],
                "acceptance": p["graph_output_acceptance"]} for p in points.values()
               if p["graph_output_acceptance"]["bitwise_equal"] is False]
    summary = {"schema": 2, "status": "partial" if missing else "complete", "shape": SHAPE,
        "analysis_script_sha256": sha256(__file__),
        "model": "Qwen3-30B-A3B expert layer shape", "environment": environments[0], "archives": archives,
        "expected_points": 56, "observed_successful_points": len(points), "missing_points": missing,
        "observed_latency_repetition_count": sum(len(p["latency_ms_repetitions"]) for p in points.values()),
        "observed_energy_v2_window_count": sum(p["energy"]["plotted_in_F2"] for p in points.values()),
        "failed_observations": failures, "latency_protocol": json.loads(next(iter(protocols))),
        "failed_predecessor_runs": failed_predecessor_runs(args.awq_failed_run_dirs),
        "graph_output_validation": {"accepted_nonbitwise_points": nonexact,
            "scope": "Exact legacy checks and explicitly configured r06 BF16-roundoff checks; numerical guard policy differs, CUDA-event latency protocol is unchanged",
            "bf16_relative_l2_limit": BF16_EPSILON / 8, "bf16_max_error_reference_scale_limit": BF16_EPSILON,
            "nonbitwise_is_not_exact": True},
        "latency_event_time_ast_sha256": next(iter(event_hashes), None),
        "latency_scope": "five raw within-process/residency repeats; not independent system repetitions",
        "comparison_scope": "synthetic BF16/FP8 versus two distinct public calibrated checkpoints; same shape/routes/timing protocol, not same weights",
        "same_weight_comparison": False, "quality_claim": False, "serving_claim": False,
        "energy_scope": "F2 AWQ/NVFP4 v2 only, one counter window per point; BF16/FP8 v1 excluded; no repeated-window error bars",
        "energy_protocol": ENERGY_V2, "no_interpolation_or_missing_point_imputation": True,
        "kernel_evidence_scope": "recorded real profiler names retained; names alone do not establish instruction counts or DRAM bandwidth",
        "routes": {}}
    for route in ROUTES:
        entry = {"points": {f: [points.get((route, f, b)) for b in TOKENS] for f in FORMATS}, "comparisons": []}
        for batch in TOKENS:
            observed = {f: points[route, f, batch] for f in FORMATS if (route, f, batch) in points}
            order = sorted(observed, key=lambda f: observed[f]["latency_ms_median"])
            comparison = {"tokens": batch, "all_four_observed": len(order) == 4,
                "observed_order_fast_to_slow": order, "fastest_among_observed": order[0],
                "four_format_winner": order[0] if len(order) == 4 else None,
                "latency_pairwise_ratios": {a + "_over_" + b: ratio(observed[a], observed[b]) for a, b in combinations(observed, 2)},
                "fp8_latency_over_format": {f: ratio(observed["fp8"], point) for f, point in observed.items() if f != "fp8"},
                "energy_awq_over_nvfp4": None}
            comparison["adjacent_latency_order_gaps"] = [{"faster": f, "slower": g,
                "slower_over_faster_minus_one_pct": 100 * (observed[g]["latency_ms_median"] / observed[f]["latency_ms_median"] - 1),
                **ratio(observed[g], observed[f])} for f, g in zip(order, order[1:])]
            if "awq" in observed:
                comparison["energy_awq_over_nvfp4"] = {key: observed["awq"]["energy"][key] / observed["nvfp4"]["energy"][key]
                    for key in ENERGY_KEYS}
                winners = {key: "awq" if comparison["energy_awq_over_nvfp4"][key] < 1 else "nvfp4"
                    if comparison["energy_awq_over_nvfp4"][key] > 1 else "equal" for key in ENERGY_KEYS}
                comparison["energy_lower_format_by_metric"] = winners
                comparison["total_and_idle_subtracted_energy_order_disagrees"] = len(set(winners.values())) > 1
            entry["comparisons"].append(comparison)
        entry["four_format_winner_change_observed_brackets"] = [
            {"tokens": [a["tokens"], b["tokens"]], "winners": [a["four_format_winner"], b["four_format_winner"]]}
            for a, b in zip(entry["comparisons"], entry["comparisons"][1:])
            if a["all_four_observed"] and b["all_four_observed"] and a["four_format_winner"] != b["four_format_winner"]]
        entry["pairwise_median_order_change_observed_brackets"] = {}
        for f, g in combinations(FORMATS, 2):
            brackets = []
            for a, b in zip(TOKENS, TOKENS[1:]):
                if all((route, x, y) in points for x in (f, g) for y in (a, b)):
                    ra = points[route, f, a]["latency_ms_median"] / points[route, g, a]["latency_ms_median"]
                    rb = points[route, f, b]["latency_ms_median"] / points[route, g, b]["latency_ms_median"]
                    if (ra - 1) * (rb - 1) < 0:
                        brackets.append([a, b])
            entry["pairwise_median_order_change_observed_brackets"][f + "_vs_" + g] = brackets
        entry["energy_order_change_observed_brackets"] = {key: [
            [a["tokens"], b["tokens"]] for a, b in zip(entry["comparisons"], entry["comparisons"][1:])
            if a["energy_awq_over_nvfp4"] and b["energy_awq_over_nvfp4"]
            and (a["energy_awq_over_nvfp4"][key] - 1) * (b["energy_awq_over_nvfp4"][key] - 1) < 0]
            for key in ENERGY_KEYS}
        summary["routes"][route] = entry
    return summary


def draw(summary, output):
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "format_sched_mpl"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator, MaxNLocator

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
        "axes.spines.right": False, "axes.edgecolor": "#6C7884", "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.dpi": 180})
    xs = [b / 16 for b in TOKENS]
    observed = [point for r in ROUTES for f in FORMATS for point in summary["routes"][r]["points"][f] if point]
    latency_limits = (min(.018, min(p["latency_ms_min"] for p in observed) * .8),
                      max(4., max(p["latency_ms_max"] for p in observed) * 1.2))
    ratio_upper = max(2.65, max(value["all_repeat_combination_envelope"][1]
        for r in ROUTES for row in summary["routes"][r]["comparisons"]
        for value in row["fp8_latency_over_format"].values()) * 1.18)
    energy_values = [1000 * p["energy"][key] for p in observed if p["format"] in ("awq", "nvfp4")
                     for key in ("j_per_input_token", "idle_subtracted_j_per_input_token")]
    energy_limits = min(.025, min(energy_values) * .8), max(25., max(energy_values) * 1.2)
    awq_count = sum(point is not None for r in ROUTES for point in summary["routes"][r]["points"]["awq"])
    badge = f"PARTIAL: AWQ {awq_count}/14 points" if summary["status"] == "partial" else "COMPLETE: 56 measured points"
    missing_note = "AWQ missing: " + "; ".join(r + " B=" + ",".join(str(m["tokens"]) for m in summary["missing_points"] if m["routing"] == r)
        for r in ROUTES if any(m["routing"] == r for m in summary["missing_points"])) if summary["missing_points"] else "All four formats observed at all 14 routing/load points."

    def axis(ax, *, top=False):
        ax.set_xscale("log", base=2)
        ax.set_xlim(xs[0] / 1.4, xs[-1] * 1.4)
        ax.xaxis.set_major_locator(FixedLocator(xs))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, pos: f"{value:g}"))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.tick_params(labelsize=9)
        ax.grid(which="major", color="#DEE4EA", linewidth=.65, zorder=0)
        if top:
            secondary = ax.secondary_xaxis("top", functions=(lambda x: x * 16, lambda x: x / 16))
            secondary.xaxis.set_major_locator(FixedLocator(TOKENS))
            secondary.xaxis.set_major_formatter(FuncFormatter(lambda value, pos: f"{value:g}"))
            secondary.xaxis.set_minor_locator(NullLocator())
            secondary.set_xlabel("Input tokens B", fontsize=9, labelpad=7)
            secondary.tick_params(labelsize=8)

    def save(fig, name):
        fig.savefig(output / (name + ".pdf"), metadata={"Title": name, "Creator": Path(__file__).name,
            "Subject": summary["comparison_scope"] + "; status=" + summary["status"]})
        fig.savefig(output / (name + ".png"))
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.1), sharex="col", gridspec_kw={"height_ratios": [1.3, 1]})
    fig.subplots_adjust(left=.085, right=.98, top=.775, bottom=.235, hspace=.20, wspace=.15)
    fig.suptitle("Qwen3 MoE expert latency across four formats", x=.085, y=.975, ha="left", fontsize=17, weight="bold")
    fig.text(.085, .933, "d=2048, intermediate=768, E=128, top-k=8 | RTX PRO 6000 SM120 | CUDA graph replay", fontsize=10, color="#44505B")
    fig.text(.98, .97, badge, ha="right", va="top", fontsize=9, color="#7C3B91", bbox={"facecolor": "#F3EBF7", "edgecolor": "none", "pad": 5})
    handles = [Line2D([], [], color=COLORS[f], marker=MARKERS[f], markersize=5, label=LABELS[f]) for f in FORMATS]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.08, .909), ncol=2, frameon=False, fontsize=9.5,
               columnspacing=3.4, handlelength=2.0)
    for col, route in enumerate(ROUTES):
        upper, lower = axes[:, col]
        axis(upper, top=True); axis(lower)
        upper.set_yscale("log")
        upper.set_ylim(*latency_limits)
        upper.yaxis.set_major_locator(FixedLocator([v for v in (.005, .01, .02, .05, .1, .2, .5, 1, 2, 4, 10, 20, 50, 100)
                                                   if latency_limits[0] <= v <= latency_limits[1]]))
        upper.yaxis.set_major_formatter(FuncFormatter(lambda value, pos: f"{value:g}"))
        upper.text(.03, .95, "Uniform routing" if route == "uniform" else "Zipf routing (alpha=1.2)",
            transform=upper.transAxes, va="top", fontsize=11, weight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": .92, "pad": 3})
        lower.axhline(1, color="#687684", linewidth=.9, linestyle="--")
        lower.set_ylim(0, ratio_upper)
        lower.yaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]))
        lower.set_xlabel("Mean assignments / expert, B k / E (all 128 experts)", fontsize=9, labelpad=8)
        for fmt in FORMATS:
            series = summary["routes"][route]["points"][fmt]
            median = [p["latency_ms_median"] if p else math.nan for p in series]
            lo = [p["latency_ms_median"] - p["latency_ms_min"] if p else math.nan for p in series]
            hi = [p["latency_ms_max"] - p["latency_ms_median"] if p else math.nan for p in series]
            upper.errorbar(xs, median, yerr=[lo, hi], marker=MARKERS[fmt], color=COLORS[fmt],
                           linewidth=1.1, markersize=4.8, capsize=3, elinewidth=.9, zorder=3)
            if fmt == "fp8":
                continue
            ratios = [p["fp8_latency_over_format"].get(fmt) for p in summary["routes"][route]["comparisons"]]
            y = [p["latency_a_over_b"] if p else math.nan for p in ratios]
            err = [[p["latency_a_over_b"] - p["all_repeat_combination_envelope"][0] if p else math.nan for p in ratios],
                   [p["all_repeat_combination_envelope"][1] - p["latency_a_over_b"] if p else math.nan for p in ratios]]
            lower.errorbar(xs, y, yerr=err, color=COLORS[fmt], marker=MARKERS[fmt], linewidth=1.1,
                           markersize=4.8, capsize=3, elinewidth=.9, zorder=3)
        if not any(summary["routes"][route]["points"]["awq"]):
            lower.text(.98, .94, "AWQ: no archived successful points", transform=lower.transAxes, ha="right", va="top", fontsize=9, color=COLORS["awq"])
    axes[0, 0].set_ylabel("Single MoE-layer latency (ms, log)", labelpad=8)
    axes[1, 0].set_ylabel("FP8 latency / format latency\n>1: format faster than FP8", labelpad=8)
    fig.text(.085, .158, "Different weight sources: synthetic BF16/FP8; dtometzki AWQ and NVIDIA NVFP4 calibrated layer 0.", fontsize=10, weight="bold")
    fig.text(.085, .117, "Medians and min-max of five within-process repetitions. Ratio whiskers are unpaired range envelopes, not confidence intervals.", fontsize=9)
    guard_count = len(summary["graph_output_validation"]["accepted_nonbitwise_points"])
    fig.text(.085, .078, f"AWQ numerical guard: {guard_count} non-bitwise points pass recorded BF16-roundoff bounds; not exact equality or task-quality validation.", fontsize=9)
    fig.text(.085, .039, "Lines join observed points only; crossings are bracketed, not fitted. " + missing_note, fontsize=9, color="#7C3B91")
    save(fig, "F1_qwen_four_formats")

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.1), sharex="col", gridspec_kw={"height_ratios": [1.3, 1]})
    fig.subplots_adjust(left=.085, right=.98, top=.775, bottom=.235, hspace=.20, wspace=.15)
    fig.suptitle("Qwen3 calibrated low-bit device energy", x=.085, y=.975, ha="left", fontsize=17, weight="bold")
    fig.text(.085, .933, "AWQ W4A16 Marlin vs NVFP4 W4A4 CUTLASS | Energy protocol v2 only | Same expert shape/routes", fontsize=10, color="#44505B")
    energy_badge = ("PARTIAL: " if summary["status"] == "partial" else "COMPLETE: ") + str(summary["observed_energy_v2_window_count"]) + "/28 energy windows"
    fig.text(.98, .97, energy_badge, ha="right", va="top", fontsize=9, color="#7C3B91", bbox={"facecolor": "#F3EBF7", "edgecolor": "none", "pad": 5})
    energy_handles = [Line2D([], [], color=COLORS[f], marker=MARKERS[f], linestyle=style,
        label=f.upper() + (" total" if key == "j_per_input_token" else " idle-subtracted"))
        for f in ("awq", "nvfp4") for key, style in (("j_per_input_token", "-"), ("idle_subtracted_j_per_input_token", "--"))]
    fig.legend(handles=energy_handles, loc="upper left", bbox_to_anchor=(.08, .909), ncol=4, frameon=False, fontsize=9.5, columnspacing=1.6)
    energy_ratios = [row["energy_awq_over_nvfp4"][key] for route in ROUTES
        for row in summary["routes"][route]["comparisons"] if row["energy_awq_over_nvfp4"] for key in ENERGY_KEYS]
    ratio_limits = min(.35, min(energy_ratios) * .9), max(1.6, max(energy_ratios) * 1.1)
    for col, route in enumerate(ROUTES):
        ax, lower = axes[:, col]
        axis(ax, top=True)
        ax.set_yscale("log"); ax.set_ylim(*energy_limits)
        ax.yaxis.set_major_locator(FixedLocator([v for v in (.003, .01, .03, .1, .3, 1, 3, 10, 20, 50, 100)
                                                if energy_limits[0] <= v <= energy_limits[1]]))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, pos: f"{value:g}"))
        ax.text(.98, .95, "Uniform routing" if route == "uniform" else "Zipf routing (alpha=1.2)",
            transform=ax.transAxes, ha="right", va="top", fontsize=11, weight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": .92, "pad": 3})
        for fmt in ("awq", "nvfp4"):
            series = summary["routes"][route]["points"][fmt]
            for key, style in (("j_per_input_token", "-"), ("idle_subtracted_j_per_input_token", "--")):
                ax.plot(xs, [1000 * p["energy"][key] if p else math.nan for p in series], color=COLORS[fmt],
                        marker=MARKERS[fmt], markersize=5, linestyle=style, linewidth=1.2)
        if not any(summary["routes"][route]["points"]["awq"]):
            ax.text(.98, .82, "AWQ missing at all 7 points", transform=ax.transAxes, ha="right", fontsize=9, color=COLORS["awq"])
        axis(lower)
        lower.axhline(1, color="#687684", linewidth=.9, linestyle=":")
        lower.set_ylim(*ratio_limits)
        lower.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
        for key, style, color, marker, label in ((ENERGY_KEYS[0], "-", "#34495E", "o", "Total"),
                (ENERGY_KEYS[1], "--", "#0F8C94", "s", "Idle-subtracted")):
            ys = [row["energy_awq_over_nvfp4"][key] if row["energy_awq_over_nvfp4"] else math.nan
                  for row in summary["routes"][route]["comparisons"]]
            lower.plot(xs, ys, color=color, linestyle=style, marker=marker, markersize=4.5, linewidth=1.2, label=label)
        lower.legend(loc="upper left", ncol=2, fontsize=8, frameon=True, facecolor="white", edgecolor="none")
        lower.set_xlabel("Mean assignments / expert, B k / E (all 128 experts)", fontsize=9, labelpad=8)
    axes[0, 0].set_ylabel("Single MoE-layer energy\n(mJ / input token, log)", labelpad=10)
    axes[1, 0].set_ylabel("AWQ energy / NVFP4 energy\n>1: NVFP4 uses less", labelpad=8)
    fig.text(.085, .158, "Only matched energy v2 is compared. Older synthetic BF16/FP8 energy v1 is excluded.", fontsize=10, weight="bold")
    fig.text(.085, .117, "NVML counter after idle baseline and continuous-load warmup; one ~3 s window per point. Small energy gaps remain unconfirmed.", fontsize=9)
    fig.text(.085, .078, "Two public checkpoints; not same weights or whole-model J/token. No repeated-window error bars, quality or serving claim.", fontsize=9)
    energy_coverage = missing_note if summary["missing_points"] else "Both low-bit formats observed at all 14 routing/load points."
    fig.text(.085, .039, "Ratios use measured total and idle-subtracted energy separately; no crossover interpolation. " + energy_coverage, fontsize=9, color="#7C3B91")
    save(fig, "F2_qwen_awq_nvfp4_energy_v2")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference-dir", type=Path, default=ROOT / "runs/twoformat_survey_r01")
    parser.add_argument("--nvfp4-dir", type=Path, default=ROOT / "runs/qwen3_nvfp4_survey_r02")
    parser.add_argument("--awq-initial-dir", type=Path, default=ROOT / "runs/qwen3_awq_survey_r01")
    parser.add_argument("--awq-resume-dir", type=Path, default=ROOT / "runs/qwen3_awq_survey_r06")
    parser.add_argument("--awq-failed-run-dirs", type=Path, nargs="*",
        default=[ROOT / f"runs/qwen3_awq_survey_r{n:02d}" for n in range(2, 6)])
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures")
    args = parser.parse_args()
    summary = load_summary(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    draw(summary, args.output_dir)
    target = args.output_dir / "qwen_fourformat_characterization_summary.json"
    target.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"status": summary["status"], "observed_successful_points": summary["observed_successful_points"],
        "missing_points": summary["missing_points"], "failed_observations_retained": len(summary["failed_observations"]),
        "failed_controller_attempts_retained": len(summary["failed_predecessor_runs"]),
        "nonbitwise_accepted_points": len(summary["graph_output_validation"]["accepted_nonbitwise_points"]),
        "summary": str(target)}, indent=2))


if __name__ == "__main__":
    main()
