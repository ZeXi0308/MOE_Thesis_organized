#!/usr/bin/env python3
"""Fit a small physical MoE kernel-cost model on a fixed batch holdout split.

T_ms(D,S) = max(a*D, b*S) + c, with a,b,c >= 0.
Fit batches are {1,4,16,64}; test batches are {2,8,32}. S is the actual
verification row count from the microbenchmark, never B*(gamma+1) by assumption.
Only calibration observations select the r=b/a grid and coefficients. For
each r, x=max(D,r*S) gives a two-variable nonnegative least-squares problem.
Pure-S and constant boundaries are included explicitly. No scipy or learning
library is required; matplotlib is imported only with --plot-dir.

The input is bench_moe_routes.py JSONL. Each route snapshot contributes one
median of its repeat means. Raw repeat/call values are retained in the output;
five repeats within one process are not five independent service experiments.

Example::

  python code/fit_kernel_cost.py --inputs 'runs/kernel_routes_*.jsonl' \
    --out-dir runs/kernel_cost_fit_r01 --plot-dir figures/kernel_cost_r01

The fit concerns one BF16 MoE layer with synthetic activations and supplied
top-k routes, including prepare/align/GEMM/activation/reduction. Histogram
replay does not preserve token co-occurrence. CUDA-event intervals may include
host launch gaps. Neither coefficients nor selected storage/time ratios are
measured HBM bandwidth, and this is not full-step/controller calibration.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import glob
import hashlib
import json
import math
from pathlib import Path
import statistics


FIT_BATCHES = (1, 4, 16, 64)
TEST_BATCHES = (2, 8, 32)
METRICS = {"cuda_event_ms_median": "cuda_event_ms_mean", "wall_ms_median": "wall_ms_mean"}
BOUNDARIES = [
    "Single real BF16 OLMoE expert layer; synthetic unit-RMS normal activations.",
    "Top-k weights are uniform unless supplied; route ID replay may be exact or histogram-realized, as grouped below.",
    "Histogram replay preserves expert counts and D but loses token co-occurrence and rank ordering.",
    "CUDA-event time is a stream interval including possible host launch gaps; it is not a sum of GPU kernel durations.",
    "No router, attention, drafter, scheduler, end-to-end latency or task quality is modeled.",
    "a, b and c are effective local cost coefficients; a is not a direct HBM-bandwidth measurement.",
    "Cache turnover is a controlled unrelated-buffer operation, not an exact reconstruction of layer-to-layer cache state.",
    "Repeat/call timings share a process and snapshot; errors are descriptive, with no independent-system confidence interval.",
    "The max-form is a hypothesis, not an established causal decomposition; residual structure can reflect omitted load imbalance, alignment, and launch effects.",
    "Predictions use measured D; this holdout tests D,S-to-time cost, not the independent route-occupancy prediction of D.",
]


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def sse(predicted, observed):
    return math.fsum((p - y) ** 2 for p, y in zip(predicted, observed))


def nonnegative_line(x, y):
    """Exact two-variable NNLS: unconstrained solution plus both boundaries."""
    if not x or len(x) != len(y) or any(v < 0 or not math.isfinite(v) for v in (*x, *y)):
        raise ValueError("Require paired finite nonnegative x and y")
    n, sx, sy = len(x), math.fsum(x), math.fsum(y)
    sxx, sxy = math.fsum(v * v for v in x), math.fsum(a * b for a, b in zip(x, y))
    candidates = [(0.0, sy / n)]
    if sxx:
        candidates.append((max(0.0, sxy / sxx), 0.0))
    denominator = n * sxx - sx * sx
    if denominator > 1e-14 * max(n * sxx, 1e-30):
        slope = (n * sxy - sx * sy) / denominator
        intercept = (sy - slope * sx) / n
        if slope >= 0 and intercept >= 0:
            candidates.append((slope, intercept))
    scored = [(sse([slope * v + intercept for v in x], y), slope, intercept)
              for slope, intercept in candidates]
    error, slope, intercept = min(scored)
    return slope, intercept, error


def prediction(parameters, distinct, tokens):
    return max(parameters["a_ms_per_distinct_expert"] * distinct,
               parameters["b_ms_per_token_slot"] * tokens) + parameters["c_ms"]


def make_ratio_grid(observations, grid_size):
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2")
    ratios = sorted({row["distinct_experts"] / row["tokens"] for row in observations})
    if not ratios:
        raise ValueError("No calibration ratios")
    lo, hi = ratios[0], ratios[-1]
    # Outside these thresholds the entire calibration set is on one branch;
    # explicit pure-D and pure-S models cover both unbounded ratio regions.
    grid = {0.0, *ratios}
    if hi > lo:
        grid.update(math.exp(math.log(lo) + i * math.log(hi / lo) / (grid_size - 1))
                    for i in range(grid_size))
    return sorted(grid)


def fit_model(observations, grid_size=801, ratio_grid=None):
    if len(observations) < 3:
        raise ValueError("At least three calibration snapshot observations are required")
    if len({(row["distinct_experts"], row["tokens"]) for row in observations}) < 2:
        raise ValueError("At least two distinct calibration (D,S) conditions are required")
    d = [row["distinct_experts"] for row in observations]
    slots = [row["tokens"] for row in observations]
    y = [row["observed_ms"] for row in observations]
    ratios = make_ratio_grid(observations, grid_size) if ratio_grid is None else sorted(set(ratio_grid) | {0.0})
    if any(r < 0 or not math.isfinite(r) for r in ratios):
        raise ValueError("Ratio grid must be finite and nonnegative")
    candidates = []
    for ratio in ratios:
        x = [max(di, ratio * si) for di, si in zip(d, slots)]
        a, c, error = nonnegative_line(x, y)
        candidates.append({"a_ms_per_distinct_expert": a,
                           "b_ms_per_token_slot": a * ratio, "c_ms": c,
                           "r_token_to_expert_ratio": ratio, "fit_sse_ms2": error,
                           "candidate_kind": "constant" if a == 0 else "pure_D" if ratio == 0 else "max_D_S"})
    b, c, error = nonnegative_line(slots, y)
    candidates.append({"a_ms_per_distinct_expert": 0.0, "b_ms_per_token_slot": b,
                       "c_ms": c, "r_token_to_expert_ratio": None,
                       "fit_sse_ms2": error, "candidate_kind": "pure_S" if b else "constant"})
    best_error = min(row["fit_sse_ms2"] for row in candidates)
    tolerance = 1e-12 * max(math.fsum(v * v for v in y), 1e-30)
    tied = [row for row in candidates if row["fit_sse_ms2"] <= best_error + tolerance]
    # Exact/roundoff ties prefer fewer nonzero coefficients, then the explicit
    # boundary. This keeps an inactive branch from receiving arbitrary weight.
    result = min(tied, key=lambda row: (
        sum(row[key] > 0 for key in ("a_ms_per_distinct_expert", "b_ms_per_token_slot", "c_ms")),
        row["candidate_kind"] == "max_D_S", row["fit_sse_ms2"],
        row["r_token_to_expert_ratio"] if row["r_token_to_expert_ratio"] is not None else math.inf))
    result = dict(result)
    result["ratio_grid"] = {"count": len(ratios), "minimum": min(ratios), "maximum": max(ratios),
                            "construction": "calibration-only log grid plus observed D/S breakpoints and pure-S boundary"}
    result["calibration_active_terms"] = dict(Counter(active_term(result, di, si) for di, si in zip(d, slots)))
    result["parameter_interpretation"] = (
        "Effective nonnegative coefficients. A branch never active on calibration is not separately identified; "
        "the ratio is selected on a finite grid, not claimed to be a continuous global optimum.")
    return result


def active_term(parameters, d, slots):
    memory = parameters["a_ms_per_distinct_expert"] * d
    compute = parameters["b_ms_per_token_slot"] * slots
    if math.isclose(memory, compute, rel_tol=1e-10, abs_tol=1e-14):
        return "tie"
    return "D_term" if memory > compute else "S_term"


def fit_domain(rows, grid_size=801):
    calibration = [row for row in rows if row["batch"] in FIT_BATCHES]
    test = [row for row in rows if row["batch"] in TEST_BATCHES]
    other = [row for row in rows if row["batch"] not in FIT_BATCHES + TEST_BATCHES]
    result = {"calibration_batches_planned": list(FIT_BATCHES), "test_batches_planned": list(TEST_BATCHES),
              "calibration_batches_observed": sorted({r["batch"] for r in calibration}),
              "test_batches_observed": sorted({r["batch"] for r in test}),
              "calibration_batches_unmeasured": sorted(set(FIT_BATCHES) - {r["batch"] for r in calibration}),
              "test_batches_unmeasured": sorted(set(TEST_BATCHES) - {r["batch"] for r in test}),
              "calibration_snapshot_count": len(calibration), "test_snapshot_count": len(test),
              "outside_split_snapshot_count": len(other), "parameters": None}
    predictions = []
    for row in rows:
        split = "calibration" if row["batch"] in FIT_BATCHES else "test" if row["batch"] in TEST_BATCHES else "outside_split"
        predictions.append({**row, "split": split, "predicted_ms": None,
                            "signed_error_ms": None, "absolute_error_ms": None,
                            "relative_error": None, "active_term": None})
    try:
        parameters = fit_model(calibration, grid_size)
    except ValueError as exc:
        result.update(status="unmeasured_or_insufficient_calibration", reason=str(exc),
                      calibration_metrics=None, test_metrics=None,
                      next_action="Collect the missing calibration batch conditions with the same kernel, weights and cache policy.")
        return result, predictions
    result.update(status="fit_available" if test else "fit_available_test_unmeasured", parameters=parameters)
    d_range = (min(row["distinct_experts"] for row in calibration), max(row["distinct_experts"] for row in calibration))
    s_range = (min(row["tokens"] for row in calibration), max(row["tokens"] for row in calibration))
    result["calibration_measured_ranges"] = {"distinct_experts": list(d_range), "tokens": list(s_range)}
    for row in predictions:
        if row["split"] == "outside_split":
            continue
        value = prediction(parameters, row["distinct_experts"], row["tokens"])
        error = value - row["observed_ms"]
        row.update(predicted_ms=value, signed_error_ms=error, absolute_error_ms=abs(error),
                   relative_error=error / row["observed_ms"],
                   active_term=active_term(parameters, row["distinct_experts"], row["tokens"]),
                   outside_calibration_D_range=not d_range[0] <= row["distinct_experts"] <= d_range[1],
                   outside_calibration_S_range=not s_range[0] <= row["tokens"] <= s_range[1])
    result["calibration_metrics"] = error_metrics([r for r in predictions if r["split"] == "calibration"])
    result["test_metrics"] = error_metrics([r for r in predictions if r["split"] == "test"])
    result["next_action"] = (
        "Use held-out residuals versus actual S, D and per-expert load imbalance to decide whether alignment/launch terms need refinement; retain the same route and kernel family."
        if test else "Measure held-out batches 2, 8 and 32 without refitting or changing this calibration split.")
    return result, predictions


def error_metrics(rows):
    if not rows:
        return {"status": "unmeasured", "snapshot_count": 0}
    errors = [row["signed_error_ms"] for row in rows]
    relative = [row["relative_error"] for row in rows]
    worst_relative = max(rows, key=lambda row: abs(row["relative_error"]))
    worst_absolute = max(rows, key=lambda row: row["absolute_error_ms"])
    fields = ("case_id", "source_file", "source_line", "batch", "gamma", "tokens", "distinct_experts",
              "observed_ms", "predicted_ms", "signed_error_ms", "relative_error")
    return {"status": "measured_descriptive_error", "snapshot_count": len(rows),
            "mae_ms": statistics.mean(abs(v) for v in errors),
            "rmse_ms": math.sqrt(statistics.mean(v * v for v in errors)),
            "mean_signed_error_ms": statistics.mean(errors),
            "mape_percent": 100 * statistics.mean(abs(v) for v in relative),
            "median_absolute_percentage_error": 100 * statistics.median(abs(v) for v in relative),
            "maximum_absolute_percentage_error": 100 * max(abs(v) for v in relative),
            "worst_relative_point": {key: worst_relative.get(key) for key in fields},
            "worst_absolute_point": {key: worst_absolute.get(key) for key in fields},
            "uncertainty": "No independent-system confidence interval; within-snapshot raw repetitions retained."}


def domain_identity(record):
    weights, environment, backend = record.get("weights", {}), record.get("environment", {}), record.get("backend", {})
    identity = weights.get("selected_weights_sha256")
    if not identity:
        raise ValueError("Missing selected_weights_sha256; cannot establish a homogeneous physical cost domain")
    return {"selected_weights_sha256": identity, "weight_layer": weights.get("layer"),
            "shape": weights.get("shape"), "bytes_per_expert": weights.get("bytes_per_expert"),
            "gpu": environment.get("gpu"), "compute_capability": environment.get("compute_capability"),
            "vllm": environment.get("vllm"), "triton": environment.get("triton"),
            "entrypoint": backend.get("entrypoint"), "experts_class": backend.get("experts_class"),
            "prepare_finalize_class": backend.get("prepare_finalize_class"),
            "kernel_source_hashes": {key: value.get("source_file_sha256")
                                     for key, value in backend.get("source_files", {}).items()},
            "cache_policy": record.get("cache_policy"), "cache_turnover_bytes": record.get("cache_turnover_bytes"),
            "execution_mode": record.get("execution_mode", "eager"),
            "route_kind": record.get("route_kind"), "input_source": record.get("input_source"),
            "topk_weight_source": record.get("topk_weight_source")}


def read_observations(patterns, metric):
    paths = sorted({Path(path).resolve() for pattern in patterns for path in glob.glob(pattern, recursive=True)})
    observations, domains, ignored, files = [], {}, [], []
    seen = set()
    for path in paths:
        content = path.read_bytes()
        files.append({"path": str(path), "sha256": hashlib.sha256(content).hexdigest()})
        for line_number, line in enumerate(content.decode().splitlines(), 1):
            if not line.strip():
                continue
            record = json.loads(line)
            provenance = {"source_file": str(path), "source_line": line_number, "case_id": record.get("case_id")}
            if record.get("status") != "ok":
                ignored.append({**provenance, "reason": "input case not measured successfully", "input_status": record.get("status")})
                continue
            fingerprint = digest_json(record)
            if fingerprint in seen:
                ignored.append({**provenance, "reason": "byte-equivalent parsed result already present; not another measurement"})
                continue
            seen.add(fingerprint)
            try:
                d, slots, batch, gamma = (int(record[key]) for key in ("distinct_experts", "tokens", "batch", "gamma"))
                y = float(record[metric])
                if min(d, slots, batch) < 1 or gamma < 0 or not math.isfinite(y) or y <= 0:
                    raise ValueError("Nonpositive or nonfinite measured cost/shape")
                counts = record.get("expert_token_counts")
                if counts is not None and sum(count > 0 for count in counts) != d:
                    raise ValueError("distinct_experts differs from actual histogram")
                repeats = record.get("timing_repeats", [])
                repeat_means = [float(repeat[METRICS[metric]]) for repeat in repeats]
                if not repeat_means or not math.isclose(statistics.median(repeat_means), y, rel_tol=1e-8, abs_tol=1e-10):
                    raise ValueError("Measured median missing or inconsistent with retained repeat means")
                identity = domain_identity(record)
            except (KeyError, TypeError, ValueError) as exc:
                ignored.append({**provenance, "reason": str(exc), "input_status": "invalid_for_fit"})
                continue
            domain_id = digest_json(identity)[:16]
            domains[domain_id] = identity
            observations.append({**provenance, "domain_id": domain_id,
                "batch": batch, "gamma": gamma, "tokens": slots, "distinct_experts": d,
                "observed_ms": y, "metric": metric, "repeat_count": len(repeats),
                "repeat_means_ms": repeat_means, "raw_timing_repeats": repeats,
                "actual_gamma_counts": record.get("actual_gamma_counts"),
                "route_kind": record.get("route_kind"),
                "histogram_max_load": max(counts) if counts else None,
                "expert_token_counts": counts,
                "source_route_file": record.get("source_file"), "source_route_step": record.get("source_step")})
    return observations, domains, ignored, files


def write_predictions(path, rows):
    fields = ("domain_id", "case_id", "split", "source_file", "source_line", "batch", "gamma", "tokens",
              "distinct_experts", "observed_ms", "predicted_ms", "signed_error_ms", "absolute_error_ms",
              "relative_error", "active_term", "outside_calibration_D_range", "outside_calibration_S_range",
              "repeat_count", "metric", "route_kind", "histogram_max_load")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def make_plots(directory, domain_results, predictions, metric):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return {"status": "not_rendered_missing_matplotlib", "files": []}
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for domain_id, fit in domain_results.items():
        rows = [row for row in predictions if row["domain_id"] == domain_id and row["predicted_ms"] is not None]
        if not rows:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.7), constrained_layout=True)
        for split, label, marker, color in (("calibration", "Calibration B=1,4,16,64", "o", "#3478a5"),
                                             ("test", "Held out B=2,8,32", "^", "#d76832")):
            selected = [row for row in rows if row["split"] == split]
            if not selected:
                continue
            axes[0].scatter([r["observed_ms"] for r in selected], [r["predicted_ms"] for r in selected],
                            label=label, marker=marker, color=color, alpha=0.8)
            axes[1].scatter([r["batch"] for r in selected], [100 * r["relative_error"] for r in selected],
                            marker=marker, color=color, alpha=0.8)
        maximum = max(max(r["observed_ms"], r["predicted_ms"]) for r in rows) * 1.05
        axes[0].plot([0, maximum], [0, maximum], "--", color="0.5", linewidth=1)
        axes[0].set(xlabel="Measured layer time (ms)", ylabel="Predicted layer time (ms)",
                    xlim=(0, maximum), ylim=(0, maximum))
        axes[0].legend(fontsize=8)
        axes[1].axhline(0, color="0.5", linewidth=1)
        axes[1].set_xscale("log", base=2)
        axes[1].set_xticks([1, 2, 4, 8, 16, 32, 64], [1, 2, 4, 8, 16, 32, 64])
        axes[1].set(xlabel="Observed decode batch B", ylabel="Signed prediction error (%)")
        fig.suptitle(f"max(aD, bS) + c | {domain_id[:8]} | {metric}", fontsize=10)
        for extension in ("png", "pdf"):
            path = directory / f"kernel_cost_{domain_id}.{extension}"
            fig.savefig(path, dpi=200)
            paths.append(str(path))
        plt.close(fig)
    return {"status": "rendered" if paths else "unmeasured_no_fitted_plot", "files": paths}


def self_test():
    minimal = {"weights": {"selected_weights_sha256": "synthetic"}}
    assert domain_identity(minimal)["execution_mode"] == "eager"
    assert digest_json(domain_identity(minimal)) != digest_json(domain_identity({**minimal, "execution_mode": "cudagraph"}))
    a, c, error = nonnegative_line([1, 2, 3], [3, 5, 7])
    assert abs(a - 2) < 1e-12 and abs(c - 1) < 1e-12 and error < 1e-20
    a, c, _ = nonnegative_line([1, 2, 3], [3, 2, 1])
    assert a == 0 and c == 2
    a, c, _ = nonnegative_line([1, 2, 3], [1, 4, 7])
    assert a >= 0 and c == 0  # unconstrained negative intercept is disallowed
    truth = {"a_ms_per_distinct_expert": 0.02, "b_ms_per_token_slot": 0.005, "c_ms": 0.04}
    calibration = [{"batch": batch, "gamma": 0, "distinct_experts": d, "tokens": slots,
                    "observed_ms": prediction(truth, d, slots), "case_id": str(index)}
                   for index, (batch, d, slots) in enumerate(((1, 8, 1), (4, 25, 4), (16, 64, 128), (64, 64, 512)))]
    fitted = fit_model(calibration, ratio_grid=[0.125, 0.25, 0.5, 1, 8])
    assert all(math.isclose(fitted[key], value, abs_tol=1e-12) for key, value in truth.items())
    first, _ = fit_domain(calibration + [{"batch": 8, "gamma": 3, "tokens": 32,
        "distinct_experts": 48, "observed_ms": 0.9, "case_id": "test"}])
    second, _ = fit_domain(calibration + [{"batch": 8, "gamma": 3, "tokens": 32,
        "distinct_experts": 48, "observed_ms": 900.0, "case_id": "test"}])
    assert first["parameters"] == second["parameters"]  # held-out outcomes cannot tune fit
    insufficient, rows = fit_domain(calibration[:2])
    assert insufficient["status"] == "unmeasured_or_insufficient_calibration"
    assert all(row["predicted_ms"] is None for row in rows)
    pure_s = [{**row, "observed_ms": 0.003 * row["tokens"] + 0.2} for row in calibration]
    fitted = fit_model(pure_s)
    assert fitted["candidate_kind"] == "pure_S" and fitted["fit_sse_ms2"] < 1e-20
    print("CPU self-test passed: NNLS boundaries, max-model recovery, held-out isolation, missing data, pure-S boundary")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inputs", nargs="+")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--plot-dir", type=Path)
    parser.add_argument("--metric", choices=tuple(METRICS), default="cuda_event_ms_median")
    parser.add_argument("--grid-size", type=int, default=801)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.inputs or args.out_dir is None:
        parser.error("--inputs and --out-dir are required")
    if args.grid_size < 2:
        parser.error("--grid-size must be at least 2")
    names = ("kernel_cost_fit.json", "kernel_cost_predictions.csv", "kernel_cost_observations.jsonl")
    if any((args.out_dir / name).exists() for name in names):
        raise FileExistsError("Choose a fresh output directory to preserve prior fits")
    observations, domains, ignored, files = read_observations(args.inputs, args.metric)
    grouped = defaultdict(list)
    for observation in observations:
        grouped[observation["domain_id"]].append(observation)
    fits, predictions = {}, []
    for domain_id, rows in sorted(grouped.items()):
        fit, output = fit_domain(rows, args.grid_size)
        fits[domain_id] = {"domain": domains[domain_id], **fit}
        predictions.extend(output)
    report = {"schema_version": 1, "model": "T_ms=max(a*D,b*S)+c; a,b,c>=0",
        "status": "measured_inputs_analyzed" if observations else "unmeasured_no_valid_kernel_data",
        "metric": args.metric, "input_files": files, "ignored_input_records": ignored,
        "fit_batches": list(FIT_BATCHES), "test_batches": list(TEST_BATCHES),
        "fit_unit": "one snapshot's median of repeat means; each snapshot weighted equally",
        "coefficient_units": {"a": "ms/distinct expert", "b": "ms/actual token slot", "c": "ms"},
        "snapshot_count": len(observations), "domains": fits, "boundaries": BOUNDARIES,
        "data_leakage_rule": "Test batches do not choose ratio grid, coefficients or domain fitting decisions.",
        "plots": {"status": "not_requested", "files": []}}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_predictions(args.out_dir / names[1], predictions)
    with (args.out_dir / names[2]).open("w") as stream:
        for row in observations:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    if args.plot_dir:
        report["plots"] = make_plots(args.plot_dir, fits, predictions, args.metric)
    (args.out_dir / names[0]).write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "snapshot_count": len(observations),
                      "domains": len(fits), "out_dir": str(args.out_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
