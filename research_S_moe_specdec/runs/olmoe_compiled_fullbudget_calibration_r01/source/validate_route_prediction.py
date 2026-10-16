#!/usr/bin/env python3
"""Fit gamma=0 expert marginals and predict held-out executed gamma=1..7 routes.

Input JSON/JSONL may contain run_matrix episodes, plugin dictionaries, or route
records with explicit nominal_gamma. Only target, pure-decode histograms are
used. Histograms estimate inclusion probabilities; they never become token
replays. Exact topk_ids, when present, are checked against the histogram.

Example (quote globs so the script can record unmatched inputs):
  python code/validate_route_prediction.py \
    --fit-routes 'runs/matrix/gamma0/routes-b*.json' \
    --test-routes 'runs/matrix/gamma[1-7]/routes-b*.json' \
    --top-k-without-replacement --nominal-comparison \
    --out-dir runs/route_prediction_r01 --plot-dir figures/route_prediction_r01

Fit/test files must have disjoint resolved paths AND SHA256 values. The split
holds out gamma execution conditions, not prompts: the same prompt subset may
appear in both partitions. Route events are not independent system repetitions.
No torch, NumPy or GPU is needed; plotting lazily imports matplotlib.
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
import statistics
import tempfile

from cost_model import expected_distinct


HISTOGRAM_KEYS = ("expert_token_counts", "expert_counts", "tokens_per_expert")
BOUNDARIES = [
    "Held-out gamma execution conditions, with the same prompt subset permitted; not independent workload validation.",
    "q_e is an expert's inclusion probability per token (sum q_e = top_k), not its fraction of route slots.",
    "Analytic occupancy assumes independent token expert sets and unchanged gamma=0 marginals; token dependence and changed draft contexts can violate it.",
    "Observed histograms come from executed verification positions, including rejected drafts; no histogram is converted into a token replay.",
    "Pure target decode events only; prefills, mixed steps and draft-model forwards are excluded and counted.",
    "Every gamma=0 token has equal weight within its layer, so larger fit batches contribute more tokens.",
    "Per-step/layer observations and shared prompts are correlated; error counts are descriptive, not independent system repetitions or confidence-interval sample sizes.",
    "Distinct experts and analytic occupancy do not measure HBM traffic, kernel time or end-to-end performance.",
]


def integer(value, name, minimum=0):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        result = int(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc
    if isinstance(value, float) and result != value:
        raise ValueError(f"{name} must be integral")
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def inventory(patterns):
    files, unmatched, seen_paths, seen_hashes = [], [], set(), {}
    for pattern in patterns:
        matches = sorted(glob.glob(pattern, recursive=True))
        if not matches:
            unmatched.append(pattern)
        for name in matches:
            path = Path(name).resolve()
            if not path.is_file():
                raise ValueError(f"Input is not a file: {path}")
            if str(path) in seen_paths:
                continue  # Overlapping glob patterns need not duplicate observations.
            payload = path.read_bytes()
            digest = hashlib.sha256(payload).hexdigest()
            if digest in seen_hashes:
                raise ValueError(f"Duplicate input content: {path} and {seen_hashes[digest]}")
            seen_paths.add(str(path))
            seen_hashes[digest] = str(path)
            files.append({"path": str(path), "sha256": digest, "bytes": len(payload)})
    return files, unmatched


def check_partition_separation(fit_files, test_files):
    for field in ("path", "sha256"):
        overlap = {row[field] for row in fit_files} & {row[field] for row in test_files}
        if overlap:
            raise ValueError(f"Fit/test {field} overlap: {sorted(overlap)}")


def walk_routes(obj, context=None):
    """Carry episode metadata without confusing actual row gamma with its cap."""
    context = dict(context or {})
    if isinstance(obj, list):
        for child in obj:
            yield from walk_routes(child, context)
        return
    if not isinstance(obj, dict):
        return
    if any(key in obj for key in HISTOGRAM_KEYS):
        yield obj, context
        return
    for key in ("model", "draft_source", "execution_mode", "proposal_budget", "label"):
        if key in obj:
            context[key] = obj[key]
    if "nominal_gamma" in obj or "gamma" in obj:
        context["nominal_gamma"] = obj.get("nominal_gamma", obj.get("gamma"))
    if "batch" in obj:
        context["nominal_batch"] = obj["batch"]
    if "requests" in obj and isinstance(obj["requests"], list):
        context["source_indices"] = sorted({str(r["source_index"]) for r in obj["requests"]
                                            if isinstance(r, dict) and "source_index" in r})
    summary = obj.get("summary", {})
    if summary.get("trace_complete") is False or summary.get("route_errors"):
        raise ValueError(f"Incomplete route episode: {context.get('label')}: {summary.get('route_errors')}")
    # A standalone plugin has no episode cap, but static/native config declares it.
    config = obj.get("config", {})
    if "nominal_gamma" not in context and isinstance(config, dict):
        if config.get("mode") == "native" and "max_gamma" in config:
            context["nominal_gamma"] = config["max_gamma"]
        elif config.get("mode") == "static" and "static_gamma" in config:
            context["nominal_gamma"] = config["static_gamma"]
    for key in ("episodes", "plugin", "routes", "records"):
        if key in obj:
            yield from walk_routes(obj[key], context)


def normalize_route(row, context, source, ordinal, asserted_unique):
    layer = str(row["layer"])
    tokens = integer(row.get("tokens", row.get("num_tokens", row.get("token_count"))), "tokens", 1)
    experts = integer(row.get("num_experts"), "num_experts", 1)
    top_k = integer(row.get("top_k"), "top_k", 1)
    if top_k > experts:
        raise ValueError(f"{layer}: top_k exceeds num_experts")
    raw = next(row[key] for key in HISTOGRAM_KEYS if key in row)
    if isinstance(raw, dict):
        sparse = {integer(key, "expert id"): integer(value, "expert count") for key, value in raw.items()}
        if any(e >= experts for e in sparse):
            raise ValueError(f"{layer}: expert id outside num_experts")
        histogram = [sparse.get(e, 0) for e in range(experts)]
    else:
        histogram = [integer(count, "expert count") for count in raw]
    if len(histogram) != experts or any(count > tokens for count in histogram):
        raise ValueError(f"{layer}: malformed inclusion histogram")
    if sum(histogram) != tokens * top_k or integer(row.get("invalid_expert_assignments", 0), "invalid assignments"):
        raise ValueError(f"{layer}: histogram does not cover all valid top-k edges")
    distinct = sum(count > 0 for count in histogram)
    if "distinct_experts" in row and integer(row["distinct_experts"], "distinct experts") != distinct:
        raise ValueError(f"{layer}: distinct count disagrees with histogram")
    exact = row.get("topk_ids")
    if exact is not None:
        counts = Counter()
        if len(exact) != tokens:
            raise ValueError(f"{layer}: topk_ids row count disagrees with tokens")
        for ids in exact:
            ids = [integer(e, "topk expert") for e in ids]
            if len(ids) != top_k or len(set(ids)) != top_k or any(e >= experts for e in ids):
                raise ValueError(f"{layer}: invalid or duplicate expert within token")
            counts.update(ids)
        if [counts[e] for e in range(experts)] != histogram:
            raise ValueError(f"{layer}: topk_ids disagree with histogram")
    elif not asserted_unique:
        raise ValueError("Histogram-only inputs require --top-k-without-replacement; token uniqueness cannot be inferred from a histogram")
    nominal_gamma = integer(row.get("nominal_gamma", context.get("nominal_gamma")), "nominal_gamma")
    batch = integer(row.get("batch", row.get("decode_batch")), "actual batch", 1)
    nominal_batch = integer(row.get("nominal_batch", context.get("nominal_batch", batch)), "nominal batch", 1)
    if tokens < batch or batch > nominal_batch:
        raise ValueError(f"{layer}: invalid decode token/batch population")
    gamma_counts = row.get("actual_gamma_counts")
    if gamma_counts is not None:
        gamma_counts = {str(integer(key, "actual gamma")): integer(value, "gamma count")
                        for key, value in gamma_counts.items()}
        gamma_counts = {key: count for key, count in gamma_counts.items() if count}
        if (sum(gamma_counts.values()) != batch
                or sum((int(gamma) + 1) * count for gamma, count in gamma_counts.items()) != tokens):
            raise ValueError(f"{layer}: actual_gamma_counts disagree with B or S")
    if tokens > batch * (nominal_gamma + 1):
        raise ValueError(f"{layer}: actual S exceeds nominal per-request cap")
    return {"source_file": source["path"], "source_sha256": source["sha256"],
            "source_route_ordinal": ordinal, "episode": row.get("episode", context.get("label", "")),
            "step": integer(row.get("step", row.get("step_id")), "step"),
            "layer": layer, "model": row.get("model", context.get("model")),
            "draft_source": context.get("draft_source"), "execution_mode": context.get("execution_mode"),
            "source_indices": context.get("source_indices", []),
            "actual_batch": batch, "nominal_batch": nominal_batch, "nominal_gamma": nominal_gamma,
            "recorded_step_gamma": row.get("gamma"), "actual_gamma_counts": gamma_counts,
            "actual_tokens": tokens, "num_experts": experts, "top_k": top_k,
            "observed_distinct": distinct, "expert_token_counts": histogram,
            "route_path": row.get("route_path"), "exact_topk_checked": exact is not None}


def load_partition(files, partition, asserted_unique):
    result, exclusions, seen = [], Counter(), set()
    for source in files:
        path = Path(source["path"])
        if path.suffix.lower() in {".jsonl", ".ndjson"}:
            objects = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        else:
            objects = [json.loads(path.read_text())]
        for ordinal, (row, context) in enumerate(walk_routes(objects)):
            if row.get("model_phase") != "target" or row.get("phase") != "decode":
                exclusions[f"{row.get('model_phase', 'missing')}:{row.get('phase', 'missing')}"] += 1
                continue
            event = normalize_route(row, context, source, ordinal, asserted_unique)
            gamma = event["nominal_gamma"]
            if partition == "fit" and (gamma != 0 or event["actual_tokens"] != event["actual_batch"]):
                raise ValueError("Fit routes must come from nominal gamma=0, one verification token per decode request")
            if partition == "test" and not 1 <= gamma <= 7:
                raise ValueError("Test routes must come from nominal gamma=1..7")
            key = (source["path"], event["episode"], event["step"], event["layer"])
            if key in seen:
                raise ValueError(f"Duplicate layer/step route event: {key}")
            seen.add(key)
            result.append(event)
    return result, dict(exclusions)


def fit_probabilities(events):
    grouped = defaultdict(list)
    for event in events:
        grouped[event["layer"]].append(event)
    fitted = {}
    for layer, rows in sorted(grouped.items()):
        shapes = {(row["num_experts"], row["top_k"]) for row in rows}
        if len(shapes) != 1:
            raise ValueError(f"Fit shape changes within layer {layer}")
        experts, top_k = shapes.pop()
        tokens = sum(row["actual_tokens"] for row in rows)
        counts = [sum(row["expert_token_counts"][e] for row in rows) for e in range(experts)]
        fitted[layer] = {"layer": layer, "num_experts": experts, "top_k": top_k,
                         "q_e": [count / tokens for count in counts], "expert_token_counts": counts,
                         "fit_tokens": tokens, "fit_route_events": len(rows),
                         "fit_batches": sorted({row["actual_batch"] for row in rows}),
                         "fit_source_sha256": sorted({row["source_sha256"] for row in rows})}
    return fitted


def make_predictions(fitted, events, nominal_comparison):
    predictions, uncovered = [], Counter()
    for event in events:
        estimate = fitted.get(event["layer"])
        if estimate is None:
            uncovered[event["layer"]] += 1
            continue
        if (estimate["num_experts"], estimate["top_k"]) != (event["num_experts"], event["top_k"]):
            raise ValueError(f"Test/fit expert shape mismatch for {event['layer']}")
        row = dict(event)
        batch, gamma, tokens = row["actual_batch"], row["nominal_gamma"], row["actual_tokens"]
        row["cap_tokens_at_actual_batch"] = batch * (gamma + 1)
        row["full_nominal_tokens"] = row["nominal_batch"] * (gamma + 1)
        row["scheduled_draft_fraction_of_cap"] = (tokens - batch) / (batch * gamma)
        row["predicted_distinct"] = expected_distinct(estimate["q_e"], tokens, 0)
        row["error"] = row["predicted_distinct"] - row["observed_distinct"]
        row["absolute_error"] = abs(row["error"])
        if nominal_comparison:
            row["predicted_cap_distinct"] = expected_distinct(estimate["q_e"], batch, gamma)
            row["predicted_nominal_distinct"] = expected_distinct(estimate["q_e"], row["nominal_batch"], gamma)
        predictions.append(row)
    return predictions, dict(uncovered)


def metrics(rows, prediction="predicted_distinct"):
    if not rows:
        return {"status": "unmeasured", "route_events": 0}
    errors = [row[prediction] - row["observed_distinct"] for row in rows]
    abs_errors = list(map(abs, errors))
    worst = rows[max(range(len(rows)), key=lambda i: abs_errors[i])]
    return {"status": "measured", "route_events": len(rows), "bias": statistics.mean(errors),
            "mae": statistics.mean(abs_errors), "rmse": math.sqrt(statistics.mean(e * e for e in errors)),
            "median_absolute_error": statistics.median(abs_errors), "max_absolute_error": max(abs_errors),
            "mape_percent": 100 * statistics.mean(abs(e) / row["observed_distinct"] for e, row in zip(errors, rows)),
            "mae_fraction_of_num_experts": statistics.mean(abs(e) / row["num_experts"] for e, row in zip(errors, rows)),
            "mean_observed_distinct": statistics.mean(row["observed_distinct"] for row in rows),
            "mean_predicted_distinct": statistics.mean(row[prediction] for row in rows),
            "worst_point": {key: worst[key] for key in ("source_file", "source_sha256", "episode", "step", "layer",
                            "actual_batch", "nominal_gamma", "actual_tokens", "observed_distinct", prediction)}}


def summarize_groups(predictions):
    grouped = defaultdict(list)
    for row in predictions:
        grouped[row["layer"], row["actual_batch"], row["nominal_gamma"]].append(row)
    result = []
    for (layer, batch, gamma), rows in sorted(grouped.items()):
        summary = {"layer": layer, "actual_batch": batch, "nominal_gamma": gamma, **metrics(rows),
                   "actual_tokens_mean": statistics.mean(row["actual_tokens"] for row in rows),
                   "actual_tokens_histogram": dict(sorted(Counter(row["actual_tokens"] for row in rows).items())),
                   "recorded_step_gamma_histogram": dict(Counter(str(row["recorded_step_gamma"]) for row in rows)),
                   "scheduled_draft_fraction_mean": statistics.mean(row["scheduled_draft_fraction_of_cap"] for row in rows)}
        counts = Counter()
        for row in rows:
            counts.update(row["actual_gamma_counts"] or {})
        summary["actual_gamma_request_counts"] = dict(sorted(counts.items(), key=lambda x: int(x[0])))
        summary["missing_actual_gamma_counts_events"] = sum(row["actual_gamma_counts"] is None for row in rows)
        if "predicted_nominal_distinct" in rows[0]:
            summary["nominal_comparison"] = metrics(rows, "predicted_nominal_distinct")
            summary["actual_batch_cap_comparison"] = metrics(rows, "predicted_cap_distinct")
        result.append(summary)
    return result


def unique_steps(events):
    steps = {}
    fields = ("source_file", "source_sha256", "episode", "step", "actual_batch", "nominal_batch",
              "nominal_gamma", "recorded_step_gamma", "actual_gamma_counts", "actual_tokens")
    for row in events:
        key = (row["source_file"], row["episode"], row["step"])
        step = {field: row[field] for field in fields}
        if key in steps and steps[key] != step:
            raise ValueError(f"Layers disagree about verification population at {key}")
        steps[key] = step
    return list(steps.values())


def analyze(fit_events, test_events, nominal_comparison=False):
    models = {row["model"] for row in fit_events + test_events if row["model"]}
    if len(models) > 1:
        raise ValueError(f"Different target model identities in input: {sorted(models)}")
    fitted = fit_probabilities(fit_events)
    predictions, uncovered = make_predictions(fitted, test_events, nominal_comparison)
    steps = unique_steps(test_events)
    actual_counts = Counter()
    for step in steps:
        actual_counts.update(step["actual_gamma_counts"] or {})
    fit_indices = {s for row in fit_events for s in row["source_indices"]}
    test_indices = {s for row in test_events for s in row["source_indices"]}
    groups = summarize_groups(predictions)
    result = {"schema_version": 1, "status": "measured" if predictions else "unmeasured",
              "split_kind": "heldout_gamma_execution_same_prompt_subset_permitted",
              "evidence_kind": "actual_verification_histogram_prediction_no_token_replay",
              "model": next(iter(models), None), "prediction_formula": "sum_e [1 - (1-q_e)**actual_S]",
              "nominal_comparison_formula": "sum_e [1 - (1-q_e)**(nominal_B*(nominal_gamma+1))]",
              "actual_batch_cap_comparison_formula": "sum_e [1 - (1-q_e)**(actual_B*(nominal_gamma+1))]",
              "boundaries": BOUNDARIES, "fit_layers": list(fitted.values()), "fit_route_events": len(fit_events),
              "test_route_events": len(test_events), "predicted_test_route_events": len(predictions),
              "uncovered_test_layers": uncovered, "metrics": metrics(predictions), "groups": groups,
              "prompt_source_indices": {"fit": sorted(fit_indices), "test": sorted(test_indices),
                                        "overlap": sorted(fit_indices & test_indices),
                                        "identity_boundary": "Source indices are metadata only; no independent-workload split is claimed."},
              "test_steps": {"count": len(steps), "actual_gamma_request_counts": dict(actual_counts),
                             "actual_tokens_histogram": dict(Counter(step["actual_tokens"] for step in steps)),
                             "missing_actual_gamma_counts_steps": sum(step["actual_gamma_counts"] is None for step in steps)},
              "observed_fit_batches": sorted({row["actual_batch"] for row in fit_events}),
              "observed_test_batches": sorted({row["actual_batch"] for row in test_events}),
              "observed_test_nominal_gammas": sorted({row["nominal_gamma"] for row in test_events}),
              "unmeasured_nominal_gammas": sorted(set(range(1, 8)) - {row["nominal_gamma"] for row in test_events})}
    if nominal_comparison and predictions:
        result["nominal_comparison_metrics"] = metrics(predictions, "predicted_nominal_distinct")
        result["actual_batch_cap_comparison_metrics"] = metrics(predictions, "predicted_cap_distinct")
    return result, predictions, steps


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})


def make_plot(directory, predictions, groups, nominal_comparison):
    if not predictions:
        return {"status": "unmeasured", "files": []}
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return {"status": "not_rendered_missing_matplotlib", "files": []}
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    panels = 2 if nominal_comparison else 1
    fig, axes = plt.subplots(1, panels, figsize=(5 * panels, 4.3), squeeze=False, constrained_layout=True)
    for index, (field, title) in enumerate(([('predicted_distinct', 'Actual verification S')]
            + ([('predicted_nominal_distinct', 'Nominal B × (γ + 1)')] if nominal_comparison else []))):
        axis = axes[0][index]
        for gamma in sorted({row["nominal_gamma"] for row in predictions}):
            rows = [row for row in predictions if row["nominal_gamma"] == gamma]
            axis.scatter([row["observed_distinct"] for row in rows], [row[field] for row in rows],
                         s=8, alpha=.22, rasterized=True, label=f"γ={gamma}", linewidths=0)
        maximum = max(max(row["observed_distinct"], row[field]) for row in predictions)
        axis.plot([0, maximum], [0, maximum], "--", color="0.4", linewidth=1)
        axis.set(xlabel="Observed distinct experts", ylabel="Predicted distinct experts", title=title,
                 xlim=(0, maximum * 1.03), ylim=(0, maximum * 1.03))
        axis.set_aspect("equal")
        axis.legend(markerscale=2, fontsize=8)
    fig.suptitle("γ=0 marginal fit; held-out executed γ conditions\nSame prompt subset permitted; layer-step observations are correlated", fontsize=10)
    paths = []
    for suffix in ("png", "pdf"):
        path = directory / f"route_prediction_scatter.{suffix}"
        fig.savefig(path, dpi=180)
        paths.append(str(path))
    plt.close(fig)
    # Compact aggregate diagnostic, keeping layers in the calculation rather
    # than declaring them independent repetitions.
    fig, axis = plt.subplots(figsize=(6, 3.8), constrained_layout=True)
    for gamma in sorted({row["nominal_gamma"] for row in groups}):
        batches = sorted({row["actual_batch"] for row in groups if row["nominal_gamma"] == gamma})
        values = [statistics.mean(row["mae"] for row in groups
                                 if row["nominal_gamma"] == gamma and row["actual_batch"] == batch)
                  for batch in batches]
        axis.plot(batches, values, "o-", label=f"γ={gamma}")
    axis.set(xlabel="Actual decode batch", ylabel="MAE (equal mean over layers)",
             title="Actual-S prediction error by held-out gamma")
    axis.set_xscale("log", base=2)
    axis.legend(fontsize=8)
    for suffix in ("png", "pdf"):
        path = directory / f"route_prediction_mae.{suffix}"
        fig.savefig(path, dpi=180)
        paths.append(str(path))
    plt.close(fig)
    return {"status": "rendered", "files": paths}


def self_test():
    def raw(gamma, counts, tokens, actual_counts, step=1):
        return {"layer": "layer0", "tokens": tokens, "num_experts": 4, "top_k": 2,
                "expert_token_counts": counts, "batch": 2, "nominal_gamma": gamma,
                "actual_gamma_counts": actual_counts, "step": step, "gamma": max(map(int, actual_counts)),
                "model_phase": "target", "phase": "decode"}
    source = {"path": "fit", "sha256": "fit-sha"}
    fit = normalize_route(raw(0, [1, 1, 1, 1], 2, {"0": 2}), {}, source, 0, True)
    test = normalize_route(raw(3, [2, 2, 1, 1], 3, {"0": 1, "1": 1}), {},
                           {"path": "test", "sha256": "test-sha"}, 0, True)
    report, predictions, steps = analyze([fit], [test], True)
    assert report["fit_layers"][0]["q_e"] == [.5] * 4
    assert predictions[0]["predicted_distinct"] == 3.5
    assert predictions[0]["predicted_nominal_distinct"] == 4 * (1 - .5 ** 8)
    assert predictions[0]["scheduled_draft_fraction_of_cap"] == 1 / 6
    assert report["metrics"]["mae"] == .5 and len(steps) == 1
    changed = dict(test, expert_token_counts=[3, 3, 0, 0], observed_distinct=2)
    assert analyze([fit], [changed])[0]["fit_layers"] == report["fit_layers"]
    assert analyze([], [test])[0]["status"] == "unmeasured"
    second_layer = dict(test, layer="layer1")
    assert len(unique_steps([test, second_layer])) == 1
    for fit_manifest, test_manifest in (([source], [source]),
            ([source], [{"path": "alias", "sha256": source["sha256"]}])):
        try:
            check_partition_separation(fit_manifest, test_manifest)
        except ValueError:
            pass
        else:
            raise AssertionError("Partition overlap accepted")
    invalids = [dict(raw(0, [1, 1, 1, 1], 2, {"0": 2}), distinct_experts=3),
                dict(raw(0, [1, 1, 1, 1], 2, {"0": 2}), topk_ids=[[0, 0], [2, 3]]),
                raw(3, [2, 2, 1, 1], 3, {"0": 2})]
    for row in invalids:
        try:
            normalize_route(row, {}, source, 0, True)
        except ValueError:
            pass
        else:
            raise AssertionError("Malformed route accepted")
    with tempfile.TemporaryDirectory() as name:
        path = Path(name) / "routes.json"
        event = raw(0, [1, 1, 1, 1], 2, {"0": 2})
        event.pop("nominal_gamma")
        path.write_text(json.dumps({"gamma": 0, "batch": 2, "plugin": {"routes": [event]}}))
        files, unmatched = inventory([str(path), str(path)])
        loaded, excluded = load_partition(files, "fit", True)
        assert len(files) == len(loaded) == 1 and not unmatched and not excluded
        event["gamma"] = 0
        path.write_text(json.dumps({"gamma": 3, "batch": 2, "plugin": {"routes": [event]}}))
        files, _ = inventory([str(path)])
        assert load_partition(files, "test", True)[0][0]["nominal_gamma"] == 3
        try:
            load_partition(files, "fit", True)
        except ValueError:
            pass
        else:
            raise AssertionError("Nonzero nominal gamma fit accepted")
    print("validate_route_prediction self-test: passed (CPU only)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fit-routes", nargs="+", default=[])
    parser.add_argument("--test-routes", nargs="+", default=[])
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--plot-dir", type=Path, help="Default: OUT_DIR/figures; missing matplotlib is recorded")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--nominal-comparison", action="store_true")
    parser.add_argument("--top-k-without-replacement", action="store_true",
                        help="Assert the router selects distinct experts within a token for histogram-only inputs")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.out_dir is None:
        parser.error("--out-dir is required")
    fit_files, fit_missing = inventory(args.fit_routes)
    test_files, test_missing = inventory(args.test_routes)
    check_partition_separation(fit_files, test_files)
    fit_events, fit_excluded = load_partition(fit_files, "fit", args.top_k_without_replacement)
    test_events, test_excluded = load_partition(test_files, "test", args.top_k_without_replacement)
    report, predictions, steps = analyze(fit_events, test_events, args.nominal_comparison)
    report.update(input_provenance={"fit": fit_files, "test": test_files,
                                   "fit_unmatched_patterns": fit_missing, "test_unmatched_patterns": test_missing,
                                   "resolved_path_overlap": [], "sha256_overlap": []},
                  excluded_route_events={"fit": fit_excluded, "test": test_excluded},
                  histogram_topk_uniqueness_asserted=args.top_k_without_replacement,
                  plots={"status": "not_requested", "files": []})
    args.out_dir.mkdir(parents=True, exist_ok=False)
    # Raw predictions retain histograms and per-request gamma-count summaries.
    for name, rows in (("route_prediction_points.jsonl", predictions), ("route_prediction_steps.jsonl", steps)):
        with (args.out_dir / name).open("w") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    write_csv(args.out_dir / "route_prediction_points.csv",
              [{k: v for k, v in row.items() if k not in {"expert_token_counts", "source_indices"}} for row in predictions])
    write_csv(args.out_dir / "route_prediction_groups.csv", report["groups"])
    if not args.no_plot:
        report["plots"] = make_plot(args.plot_dir or args.out_dir / "figures", predictions, report["groups"], args.nominal_comparison)
    (args.out_dir / "route_prediction.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "fit_route_events": len(fit_events),
                      "test_route_events": len(test_events), "metrics": report["metrics"],
                      "plots": report["plots"], "out_dir": str(args.out_dir)}, allow_nan=False))


if __name__ == "__main__":
    main()
