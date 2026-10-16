"""Reproduce the descriptive eager/graph pairing; no fitting or GPU execution."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from fit_kernel_cost import FIT_BATCHES, TEST_BATCHES, domain_identity
from plot_kernel_diagnostics import coords, document, scatter, write_csv
from plot_energy_progress import render_pgf

OUTPUT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures/kernel_cost_graph_r01"
INPUTS = {"eager": ROOT / "runs/olmoe_kernel_r03/measurements.jsonl",
          "cudagraph": ROOT / "runs/olmoe_kernel_graph_r01/measurements.jsonl"}
MATCH_FIELDS = ("batch", "gamma", "nominal_batch", "step_gamma", "tokens", "distinct_experts",
    "expert_token_counts", "actual_gamma_counts", "source_file", "source_file_sha256", "source_record_index",
    "source_step", "source_episode", "cell_snapshot_population", "selected_rank_in_sorted_population",
    "weights", "backend", "environment", "seed", "input_source", "route_kind", "cooccurrence_preserved",
    "topk_weight_source", "cache_policy", "cache_turnover_bytes", "resolved_tile_configuration",
    "warmup_calls", "repeats", "iterations_per_repeat")


def describe(values):
    return {"count": len(values), "min": min(values), "median": statistics.median(values),
            "mean": statistics.mean(values), "max": max(values)}


raw, provenance = {}, {}
for mode, path in INPUTS.items():
    data = path.read_bytes()
    rows = [json.loads(line) for line in data.splitlines() if line]
    raw[mode] = {r["case_id"]: r for r in rows}
    assert len(rows) == len(raw[mode]) == 154
    provenance[mode] = {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(),
        "status": json.loads((path.parent / "status.json").read_text())["status"],
        "domain": domain_identity(rows[0])}
    assert provenance[mode]["status"] == "complete"
    for row in rows:
        assert row["status"] == "ok" and row.get("execution_mode", "eager") == mode
        assert len(row["timing_repeats"]) == row["repeats"] == 5
        for name in ("cuda_event_ms", "wall_ms"):
            means = []
            for repeat in row["timing_repeats"]:
                calls = repeat[name + "_calls"]
                assert len(calls) == row["iterations_per_repeat"] == 20
                assert all(math.isfinite(v) and v > 0 for v in calls)
                assert math.isclose(statistics.mean(calls), repeat[name + "_mean"], rel_tol=1e-9)
                means.append(repeat[name + "_mean"])
            assert math.isclose(statistics.median(means), row[name + "_median"], rel_tol=1e-9)
assert raw["eager"].keys() == raw["cudagraph"].keys()

pairs = []
for case, eager in raw["eager"].items():
    graph = raw["cudagraph"][case]
    assert all(eager.get(key) == graph.get(key) for key in MATCH_FIELDS), case
    assert graph["execution_mode_realized"] and graph["graph_capture"]["capture_succeeded"]
    assert graph["graph_capture"]["matches_eager_warmup"]
    assert graph["output_finite_before_timing"] and graph["output_finite_after_timing"]
    assert graph["profile"]["cache_turnover_before_profile"]
    pair = {"case_id": case, "batch": eager["batch"], "gamma": eager["gamma"],
        "tokens": eager["tokens"], "distinct_experts": eager["distinct_experts"],
        "split": "calibration" if eager["batch"] in FIT_BATCHES else "test",
        "route_source_sha256": eager["source_file_sha256"], "route_source_record_index": eager["source_record_index"],
        "tile": eager["resolved_tile_configuration"]["parameters"],
        "graph_capture_wall_ms": graph["graph_capture"]["capture_wall_ms"],
        "graph_first_replay_wall_ms": graph["graph_capture"]["first_replay_wall_ms"]}
    for metric, short in (("cuda_event_ms", "event"), ("wall_ms", "wall")):
        e, g = eager[metric + "_median"], graph[metric + "_median"]
        pair.update({f"eager_{short}_ms": e, f"graph_{short}_ms": g,
            f"graph_over_eager_{short}": g / e, f"graph_minus_eager_{short}_ms": g - e,
            f"eager_{short}_repeat_means_ms": [r[metric + "_mean"] for r in eager["timing_repeats"]],
            f"graph_{short}_repeat_means_ms": [r[metric + "_mean"] for r in graph["timing_repeats"]]})
    pairs.append(pair)

report = {"status": "matched_measured_execution_modes", "inputs": provenance,
    "matched_case_count": len(pairs), "matched_fields": list(MATCH_FIELDS), "mismatches": [],
    "fit_test_counts": dict(Counter(r["split"] for r in pairs)),
    "actual_S": describe([r["tokens"] for r in pairs]), "actual_D": describe([r["distinct_experts"] for r in pairs]),
    "per_case_statistic": "median of five within-process means, each retaining twenty raw calls",
    "raw_repeats_per_execution_mode": 770, "raw_calls_per_execution_mode": 15400,
    "paired_metrics": {key: describe([r[key] for r in pairs]) for key in (
        "graph_over_eager_event", "graph_over_eager_wall", "graph_minus_eager_event_ms", "graph_minus_eager_wall_ms")},
    "graph_lower_cases": {name: sum(r[f"graph_over_eager_{name}"] < 1 for r in pairs) for name in ("event", "wall")},
    "graph_startup_excluded_from_timings": {key: describe([r[key] for r in pairs])
        for key in ("graph_capture_wall_ms", "graph_first_replay_wall_ms")},
    "profile_kernel_count_distributions": {mode: dict(Counter(r["profile"]["kernel_count"] for r in rows.values())) for mode, rows in raw.items()},
    "boundaries": [
        "Execution domains are fitted separately; this table pairs raw measured cases and fits no new model.",
        "Both event/wall campaigns use 256MiB unrelated-buffer turnover outside every timed call; matched metadata does not prove identical physical cache contents.",
        "These are separate execution windows, not randomized/interleaved independent service repeats. Timing deltas are descriptive execution-mode contrasts, not isolated CPU launch costs.",
        "Graph captures one fixed-input MoE layer, not model graph padding/router/attention or end-to-end inference. Capture and first replay are excluded from the measured repeats and recorded separately.",
        "Legacy r03 profiling omitted cache turnover; graph profiling explicitly repeated it. Profiler values are not subtracted across runs or from event intervals to infer host cost.",
        "Histograms and D/S match, but token co-occurrence was synthetically realized. Selected expert storage bytes are not measured HBM traffic.",
        "Five repeats and twenty calls share the snapshot/process; no independent-system confidence interval is inferred."],
    "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
write_csv(OUTPUT / "paired_case_comparison.csv", pairs)
maximum = 1.05 * max(r["eager_event_ms"] for r in pairs)
first = (r"\nextgroupplot[title={(a) Same-case event intervals},xlabel={Eager event interval (ms)},ylabel={CUDA graph event interval (ms)},xmin=0,xmax=%s,ymin=0,ymax=%s,legend pos=north west]" % (maximum, maximum)
    + scatter(pairs, lambda r: r["eager_event_ms"], lambda r: r["graph_event_ms"])
    + r"\addplot[gray,dashed,forget plot,domain=0:%s] {x};" % maximum)
second = r"\nextgroupplot[title={(b) Same-case graph/eager ratio},xlabel={Measured distinct experts $D$},ylabel={Graph time / eager time},ymin=0.45,ymax=1.02,legend pos=south east]"
for field, label, color, mark in (("event", "Event interval", "cal", "o"), ("wall", "Wall interval", "test", "triangle")):
    second += r"\addplot[only marks,color=%s,mark=%s,mark size=1.8pt,opacity=0.65] coordinates {%s};\addlegendentry{%s}" % (color, mark, coords([(r["distinct_experts"], r[f"graph_over_eager_{field}"]) for r in pairs]), label)
second += r"\addplot[gray,dashed,forget plot,domain=8:64] {1};"
note = (r"154 paired route snapshots; identical weights, routes, tiles, seeds and 256 MiB turnover policy; five within-process repeats per point.\\"
        r"Separate execution windows; ratios are descriptive, not isolated launch overhead or end-to-end speedup. Profiler cache conditions differ.")
report["figures"] = render_pgf(document([first, second], "Matched OLMoE layer snapshots: eager versus CUDA graph", note),
    FIGURES, "kernel_eager_graph_paired", ["pdf", "svg", "png"])
(OUTPUT / "paired_case_comparison.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
print(json.dumps({"cases": len(pairs), "paired_metrics": report["paired_metrics"], "figures": report["figures"]}))
