#!/usr/bin/env python3
"""Fixed AR-marginal occupancy check for the compiled route diagnostic.

Only CPU postprocessing. Gamma=0 B={1,8,32} all decode tokens fit q_e;
gamma=4 B={1,8,32} all decode layer-steps are held out. No selected subset,
new predictor, or performance inference is introduced.
"""
from collections import Counter, defaultdict
import csv
import gc
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RUN = ROOT / "runs/olmoe_compiled_routes_r01"
VERIFIED = ROOT / "runs/compiled_route_summary_r01/summary.json"
sys.path.insert(0, str(ROOT / "code"))
from validate_route_prediction import fit_probabilities, normalize_route
from cost_model import expected_distinct


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v
                         for k, v in row.items()} for row in rows)


def metric(rows, predictor, target):
    errors = [r[predictor] - r[target] for r in rows]
    worst = rows[max(range(len(rows)), key=lambda i: abs(errors[i]))]
    return {
        "decode_layer_steps": len(rows),
        "mean_prediction": statistics.mean(r[predictor] for r in rows),
        "mean_observed": statistics.mean(r[target] for r in rows),
        "bias": statistics.mean(errors),
        "mae": statistics.mean(abs(e) for e in errors),
        "rmse": math.sqrt(statistics.mean(e * e for e in errors)),
        "mape_percent": 100 * statistics.mean(abs(e) / r[target] for e, r in zip(errors, rows)),
        "max_absolute_error": max(abs(e) for e in errors),
        "worst_point": {k: worst[k] for k in ("source_file", "episode", "step", "layer", "batch",
                                                 "actual_batch", "step_gamma", "S", "P", predictor, target)},
    }


def main():
    verified = json.loads(VERIFIED.read_text())
    assert verified["validation_all_passed"]
    known = {r["file"]: r for r in verified["cases"]}
    sources, fit, test, padding_b1 = [], [], [], []
    excluded = Counter()
    for gamma in (0, 4):
        for batch in (1, 8, 32):
            path = RUN / f"gamma{gamma}/routes-b{batch}.json"
            source = {"path": str(path.relative_to(ROOT)), "sha256": sha256(path),
                      "gamma": gamma, "nominal_batch": batch}
            assert source["sha256"] == known[str(path.relative_to(RUN))]["sha256"]
            sources.append(source)
            obj = json.loads(path.read_text())
            assert obj["gamma"] == gamma and obj["batch"] == batch and obj["population"] == 64
            requests = [r for ep in obj["episodes"] for r in ep["requests"]]
            indices = sorted({r["source_index"] for r in requests})
            assert len(requests) == len(indices) == 64
            source["source_indices"] = indices
            assert indices == sources[0]["source_indices"]
            assert all(r["finished"] and len(r["output_token_ids"]) == 32 for r in requests)
            ordinal = 0
            for ep in obj["episodes"]:
                plugin = ep["plugin"]
                assert plugin["summary"]["trace_complete"] and not plugin["summary"]["route_errors"]
                physical = {(r["step"], r["layer"]): r for r in plugin["graph_padding_routes"]}
                for raw in plugin["routes"]:
                    ordinal += 1
                    if raw["model_phase"] != "target" or raw["phase"] != "decode":
                        excluded[f"gamma{gamma}:{raw['model_phase']}:{raw['phase']}"] += 1
                        continue
                    event = normalize_route(raw, {"nominal_gamma": gamma, "nominal_batch": batch},
                                            source, ordinal, asserted_unique=False)
                    if gamma == 0:
                        assert event["actual_tokens"] == event["actual_batch"]
                        fit.append(event)
                        continue
                    phys = physical[raw["step"], raw["layer"]]
                    assert phys["complete"] and phys["scheduled_tokens"] == event["actual_tokens"]
                    assert phys["physical_distinct_experts"] == sum(n > 0 for n in phys["physical_expert_token_counts"])
                    test.append({"source_file": source["path"], "source_sha256": source["sha256"],
                        "episode": ep["label"], "step": raw["step"], "layer": event["layer"],
                        "batch": batch, "actual_batch": event["actual_batch"], "step_gamma": event["recorded_step_gamma"],
                        "S": event["actual_tokens"], "P": phys["physical_tokens"],
                        "valid_D": event["observed_distinct"], "physical_D": phys["physical_distinct_experts"]})
                    if batch == 1 and phys["padding_tokens"] > 0:
                        assert event["actual_tokens"] == 1 and phys["physical_tokens"] == 5
                        padding_b1.append({"episode": ep["label"], "step": raw["step"], "layer": event["layer"],
                            "histogram": tuple(phys["padding_expert_token_counts"]),
                            "support": frozenset(i for i, n in enumerate(phys["padding_expert_token_counts"]) if n),
                            "valid_histogram": tuple(raw["expert_token_counts"]),
                            "valid_support": frozenset(i for i, n in enumerate(raw["expert_token_counts"]) if n)})
            del obj, requests, ep, plugin, physical
            gc.collect()
    fit_hashes = {s["sha256"] for s in sources if s["gamma"] == 0}
    test_hashes = {s["sha256"] for s in sources if s["gamma"] == 4}
    assert fit_hashes.isdisjoint(test_hashes)
    fitted = fit_probabilities(fit)
    assert len(fitted) == 16
    for row in fitted.values():
        assert row["fit_tokens"] == 5952 and math.isclose(sum(row["q_e"]), 8)
    prediction_cache = {}
    for row in test:
        for slots, name in ((row["S"], "q_S"), (row["P"], "q_P")):
            key = row["layer"], slots
            if key not in prediction_cache:
                prediction_cache[key] = expected_distinct(fitted[row["layer"]]["q_e"], slots, 0)
            row[name] = prediction_cache[key]
    comparisons = (("q+S_to_validD", "q_S", "valid_D"),
                   ("q+P_to_physicalD", "q_P", "physical_D"),
                   ("q+S_to_physicalD", "q_S", "physical_D"))
    summary, grouped = [], []
    for name, predictor, target in comparisons:
        for batch in (None, 1, 8, 32):
            rows = [r for r in test if batch is None or r["batch"] == batch]
            summary.append({"comparison": name, "nominal_batch": batch if batch is not None else "all",
                            **metric(rows, predictor, target)})
        for batch in (1, 8, 32):
            for layer in sorted(fitted):
                rows = [r for r in test if r["batch"] == batch and r["layer"] == layer]
                grouped.append({"comparison": name, "nominal_batch": batch, "layer": layer,
                                **metric(rows, predictor, target)})

    repetition = []
    for layer in sorted(fitted):
        rows = [r for r in padding_b1 if r["layer"] == layer]
        supports = Counter(r["support"] for r in rows)
        histograms = Counter(r["histogram"] for r in rows)
        by_step = {(r["episode"], r["step"]): r for r in rows}
        pairs = [(by_step[r["episode"], r["step"] - 1], r) for r in rows
                 if (r["episode"], r["step"] - 1) in by_step]
        def pair_mean(fn):
            return statistics.mean(fn(a, b) for a, b in pairs) if pairs else None
        modal, modal_count = supports.most_common(1)[0]
        counts = [sum(r["histogram"][i] for r in rows) for i in range(64)]
        # Descriptive comparison only. This padding marginal is never used to
        # refit or predict D. Divide by top-k to compare distributions over edges.
        pad_q = [n / (4 * len(rows)) for n in counts]
        ar_q = fitted[layer]["q_e"]
        repetition.append({"layer": layer, "padded_steps": len(rows), "padding_rows_per_step": 4,
            "distinct_padding_support_sets": len(supports), "distinct_padding_histograms": len(histograms),
            "modal_support_fraction": modal_count / len(rows), "modal_support_ids": sorted(modal),
            "modal_histogram_fraction": histograms.most_common(1)[0][1] / len(rows),
            "mean_padding_D": statistics.mean(len(r["support"]) for r in rows),
            "consecutive_padded_step_pairs_within_wave": len(pairs),
            "consecutive_same_padding_support_fraction": pair_mean(lambda a,b: a["support"] == b["support"]),
            "consecutive_same_padding_histogram_fraction": pair_mean(lambda a,b: a["histogram"] == b["histogram"]),
            "consecutive_padding_support_jaccard_mean": pair_mean(lambda a,b: len(a["support"] & b["support"]) / len(a["support"] | b["support"])),
            "consecutive_same_valid_support_fraction": pair_mean(lambda a,b: a["valid_support"] == b["valid_support"]),
            "consecutive_valid_support_jaccard_mean": pair_mean(lambda a,b: len(a["valid_support"] & b["valid_support"]) / len(a["valid_support"] | b["valid_support"])),
            "padding_vs_AR_route_edge_marginal_total_variation": sum(abs(a-b) for a,b in zip(pad_q,ar_q)) / 16})

    boundaries = [
        "Fixed fit partition: gamma=0, nominal B={1,8,32}, all 64 requests and all target decode rows. q_e=expert inclusion count/token count, pooled by layer. Each B contributes 1984 decode tokens per layer, total 5952; no fitting to gamma=4.",
        "Fixed test partition: gamma=4, nominal B={1,8,32}, all target decode layer-steps. SHA-disjoint files but the same 64 prompt subset: held-out executed gamma conditions, not an independent workload validation.",
        "Comparison predictions all use the identical fitted q_e and formula sum_e[1-(1-q_e)^n], with n=S or P. P is model-forward row count, excluding expert GEMM tile padding.",
        "Metrics weight decode layer-steps equally; small B therefore contributes more steps to the pooled summary. Per-B metrics are the primary comparison. Terminal shrinking batches and every executed valid verification row are retained.",
        "Steps, layers and waves are correlated, with one instrumented condition run; no independent confidence interval or significance claim is made.",
        "B1 padding repetition uses exact recorded padding histograms/support sets, not token-level padding routes. Adjacent means consecutive integer steps within the same wave, both padded; no across-wave or across-gap pair is included.",
        "Padding route-edge marginal distance is a descriptive comparison to AR, not a fitted alternate predictor. Repeated support/histogram does not establish identical ordered token routes or their mechanism.",
        "Instrumented physical working-set prediction only, not performance or measured HBM traffic; route-return/native histogram overhead remains active. Main eager F1 data and analysis are unchanged.",
    ]
    interpretation = (
        "Using P instead of S removes much of the omitted physical expert-set work, "
        "especially at B=1. It supports P as a necessary correction for a physical-row "
        "occupancy approximation, but does not establish that padding rows obey the AR "
        "marginals or within-step independence assumptions. At B=1 q+P still has MAE "
        "3.132707 and positive bias 2.561535 experts (MAPE 12.863844%). The recorded "
        "padding histograms recur exactly over consecutive padded steps and their "
        "edge marginals differ from AR; temporal recurrence alone does not determine "
        "within-step independence. This is descriptive evidence about route sets, "
        "not a CPU/GPU cache-hit observation, timing decomposition, or service benefit."
    )
    report = {"sources": sources, "verified_summary": str(VERIFIED.relative_to(ROOT)),
        "verified_summary_sha256": sha256(VERIFIED), "analysis_code_sha256": sha256(Path(__file__)),
        "fit_test_sha_disjoint": True, "fit_layer_count": len(fitted), "fit_layer_step_events": len(fit),
        "test_layer_step_events": len(test), "excluded_non_decode_events": dict(excluded),
        "fit_tokens_per_layer": 5952, "fit_nominal_batches": [1,8,32], "test_nominal_gamma": 4,
        "boundaries": boundaries, "interpretation": interpretation,
        "metrics": summary, "padding_repetition_by_layer": repetition,
        "implementation_sources": [{"path":str((ROOT/'code'/name).relative_to(ROOT)),"sha256":sha256(ROOT/'code'/name)}
                                   for name in ('validate_route_prediction.py','cost_model.py')]}
    (OUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    write_csv(OUT / "metrics.csv", summary)
    write_csv(OUT / "metrics_by_layer.csv", grouped)
    write_csv(OUT / "padding_repetition_by_layer.csv", repetition)
    write_csv(OUT / "fitted_q.csv", [{"layer": layer, "expert": i, "q_e": q,
        "fit_tokens": model["fit_tokens"], "fit_inclusions": model["expert_token_counts"][i]}
        for layer, model in sorted(fitted.items()) for i, q in enumerate(model["q_e"])])
    lines = ["B | q+S -> valid D MAE | q+P -> physical D MAE | q+S -> physical D MAE | q+P bias | q+P MAPE (%)"]
    for batch in (1, 8, 32):
        by_name = {r["comparison"]:r for r in summary if r["nominal_batch"] == batch}
        physical = by_name["q+P_to_physicalD"]
        lines.append(f"{batch} | {by_name['q+S_to_validD']['mae']:.6f} | "
                     f"{physical['mae']:.6f} | {by_name['q+S_to_physicalD']['mae']:.6f} | "
                     f"{physical['bias']:.6f} | {physical['mape_percent']:.6f}")
    recurrence = (
        "B1 padding diagnostic: 1569 padded steps per layer, four padding rows per step. "
        "Each of 16 layers has 134 distinct padding histograms/support sets; its modal "
        "histogram accounts for 46.653920% of padded steps. All 1370 adjacent padded-step "
        "pairs within waves per layer have identical padding histograms (support Jaccard=1). "
        "For the valid row on those same pairs, equal-support fraction ranges from 0 to "
        "1.094891%, with mean Jaccard by layer 0.177427 to 0.355602. Padding versus AR "
        "route-edge marginal total variation ranges from 0.404341 to 0.572274. These "
        "are histogram/support observations, not a reconstruction of padding token routes."
    )
    (OUT / "README.txt").write_text(
        "Reproduce: python3 research_S_moe_specdec/runs/compiled_route_prediction_r01/analyze.py\n\n"
        + interpretation + "\n\n" + "\n".join(lines) + "\n\n" + recurrence
        + "\n\n" + "\n\n".join(boundaries) + "\n")
    for r in summary:
        print(json.dumps({k: r[k] for k in ("comparison","nominal_batch","decode_layer_steps","mean_prediction","mean_observed","bias","mae","rmse","mape_percent")}))
    print("PADDING REPETITION", json.dumps(repetition))


if __name__ == "__main__":
    main()
