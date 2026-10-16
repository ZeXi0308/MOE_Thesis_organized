#!/usr/bin/env python3
"""CPU-only component accounting for the existing Qwen3 FP8/NVFP4 traces.

Reads the two completed local surveys; never imports torch or starts a worker.
Chrome trace ts/dur values are microseconds (displayTimeUnit is only a viewer
setting). GPU kernel events are graph nodes, not separate host launches.
--awq-focus adds a separate B2048/8192 AWQ comparison JSON from r06, without
overwriting the original FP8/NVFP4 summary or notes.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path


TOKENS = (1, 8, 32, 128, 512, 2048, 8192)
FOCUS = (1, 8, 2048, 8192)
RUNS = {"fp8": "twoformat_survey_r01", "nvfp4": "qwen3_nvfp4_survey_r02"}
GROUPS = ("projection_kernels", "routing_metadata", "grouped_gemm_setup",
          "activation_and_quantization", "row_shuffles", "output_operations",
          "unattributed_copy_kernel", "unclassified")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def interval_union(events):
    intervals = sorted((float(e["ts"]), float(e["ts"]) + float(e["dur"])) for e in events)
    if not intervals:
        return 0.0
    start, end = intervals[0]
    total = 0.0
    for a, b in intervals[1:]:
        if a > end:
            total += end - start
            start, end = a, b
        else:
            end = max(end, b)
    return total + end - start


def classify(name, fmt, occurrences):
    """Map observed symbols and source-call order; do not infer traffic bytes."""
    if (name == "fused_moe_kernel" or "MainloopSm120ArrayTmaWarpSpecializedBlockScaled" in name
            or (fmt == "awq" and "marlin_moe_wna16::Marlin<" in name)):
        occurrences["projection"] += 1
        return "projection_kernels", f"projection_{occurrences['projection']}"
    if "FillFunctor<int>" in name:
        return "routing_metadata", "integer_fill"
    if any(s in name for s in ("moe_align_block_size_kernel", "count_and_sort_expert_tokens_kernel",
                              "compute_problem_sizes", "compute_expert_blockscale_offsets", "compute_arg_sorts")):
        return "routing_metadata", name.split("(")[0].split("<")[0].removeprefix("void ")
    if "__get_group_gemm_starts" in name:
        return "grouped_gemm_setup", "group_gemm_starts"
    if "shuffleInputRowsKernel" in name:
        occurrences["shuffle"] += 1
        return "row_shuffles", f"row_shuffle_{occurrences['shuffle']}"
    if "cvt_fp16_to_fp4" in name:
        occurrences["fp4_quant"] += 1
        # The local cutlass_moe.py calls input quantization, then the fused
        # SiLU+mul+quantization op. This label uses order, not a guessed meaning
        # of the C++ template's boolean arguments.
        stage = "input_fp4_quantization" if occurrences["fp4_quant"] == 1 else "intermediate_silu_mul_fp4_quantization"
        return "activation_and_quantization", stage
    if "segmented_max_reduction_strided" in name:
        occurrences["scale_reduce"] += 1
        return "activation_and_quantization", f"fp8_scale_reduction_{occurrences['scale_reduce']}"
    if "scaled_fp8_quant_kernel_strided_dynamic" in name:
        occurrences["fp8_quant"] += 1
        return "activation_and_quantization", f"fp8_quantization_{occurrences['fp8_quant']}"
    if "act_and_mul_kernel" in name:
        return "activation_and_quantization", "silu_and_mul"
    if "moe_sum_vec_kernel" in name:
        return "output_operations", "fp8_moe_sum"
    if fmt == "nvfp4" and "bfloat16_copy_kernel_cuda" in name:
        return "output_operations", "bf16_dtype_copy"
    if fmt == "nvfp4" and "MulFunctor<float>" in name:
        return "output_operations", "output_weight_multiply"
    if fmt == "nvfp4" and "sum_functor" in name:
        return "output_operations", "output_reduce"
    if fmt == "awq" and "sum_functor" in name:
        # fused_marlin_moe with expert_map=None returns torch.sum(..., dim=1).
        return "output_operations", "awq_moe_sum"
    if name == "memcpy32_post":
        # Its symbol and position alone do not establish whether this is an
        # output copy, runtime helper, or profiler artifact. Retain separately.
        return "unattributed_copy_kernel", "memcpy32_post"
    return "unclassified", "unclassified"


def analyze_trace(root, row, path, catalog):
    trace = json.loads(path.read_text())
    events = trace["traceEvents"]
    kernels = sorted((e for e in events if e.get("cat") == "kernel" and e.get("ph") == "X"), key=lambda e: e["ts"])
    if not kernels or any(not math.isfinite(e["dur"]) or e["dur"] < 0 for e in kernels):
        raise ValueError(f"invalid CUDA durations: {path}")
    runtime = [e for e in events if e.get("cat") == "cuda_runtime" and e.get("ph") == "X"]
    runtime_counts = Counter(e["name"] for e in runtime)
    if runtime_counts["cudaGraphLaunch"] != 1:
        raise ValueError(f"expected exactly one profiled graph replay: {path}")
    memory = [e for e in events if e.get("cat") in ("gpu_memcpy", "gpu_memset")]
    occurrences = Counter()
    components = {g: {"duration_sum_us": 0.0, "kernel_event_count": 0} for g in GROUPS}
    details = []
    for e in kernels:
        group, stage = classify(e["name"], row["format"], occurrences)
        components[group]["duration_sum_us"] += e["dur"]
        components[group]["kernel_event_count"] += 1
        args = e.get("args", {})
        details.append({"kernel_id": catalog[e["name"]], "component": group, "stage": stage,
                        "duration_us": e["dur"], "start_offset_us": round(e["ts"] - kernels[0]["ts"], 3),
                        "stream": args.get("stream"), "grid": args.get("grid"), "block": args.get("block")})
    if occurrences["projection"] != 2:
        raise ValueError(f"expected two projection kernels: {path}")
    if row["format"] == "nvfp4" and (occurrences["fp4_quant"] != 2 or occurrences["shuffle"] != 2):
        raise ValueError(f"NVFP4 call order changed: {path}")
    total = sum(e["dur"] for e in kernels)
    recorded = row["profile"]
    if len(kernels) != recorded["kernel_event_count"] or not math.isclose(total, recorded["kernel_duration_sum_us"], abs_tol=1e-6):
        raise ValueError(f"trace and measurement record disagree: {path}")
    union = interval_union(kernels)
    span = max(e["ts"] + e["dur"] for e in kernels) - min(e["ts"] for e in kernels)
    projection = components["projection_kernels"]["duration_sum_us"]
    for value in components.values():
        value["duration_sum_us"] = round(value["duration_sum_us"], 6)
        value["fraction_of_kernel_sum"] = value["duration_sum_us"] / total
    latency_us = row["latency_ms_median"] * 1000
    return {
        "format": row["format"], "routing": row["routing"]["kind"], "tokens": row["tokens"],
        "trace": str(path.relative_to(root)), "trace_sha256": digest(path),
        "status": row["status"], "timing_mode": row["timing_mode"],
        "weight_source": row["backend"].get("weight_source"),
        "routing_counts": row["routing"]["counts"], "active_experts": row["routing"]["active_experts"],
        "max_per_expert": row["routing"]["max_per_expert"],
        "unprofiled_graph_event_median_us": latency_us,
        "unprofiled_graph_event_trials_us": [x * 1000 for x in row["latency_ms"]],
        "profiled_kernel_event_count": len(kernels), "profiled_kernel_duration_sum_us": round(total, 6),
        "profiled_projection_kernel_sum_us": round(projection, 6),
        "profiled_other_kernel_sum_us": round(total - projection, 6),
        "profiled_kernel_interval_union_us": round(union, 3),
        "profiled_first_kernel_start_to_last_kernel_end_us": round(span, 3),
        "profiled_span_minus_union_us": round(max(0.0, span - union), 3),
        "profiled_kernel_overlap_us": round(max(0.0, total - union), 3),
        "profiled_streams": sorted({str(e.get("args", {}).get("stream")) for e in kernels}),
        "sum_vs_unprofiled_median_difference_pct": 100 * (total / latency_us - 1),
        "host_runtime_call_counts": dict(runtime_counts),
        "host_runtime_duration_sum_us": {n: round(sum(e["dur"] for e in runtime if e["name"] == n), 6) for n in runtime_counts},
        "gpu_memcpy_memset_event_count": len(memory),
        "components": components, "kernel_events": details,
    }


def comparison(fp8, nvfp4):
    if fp8["routing_counts"] != nvfp4["routing_counts"]:
        raise ValueError("paired routing histograms differ")
    def difference(key):
        return round(nvfp4[key] - fp8[key], 6)
    return {"routing": fp8["routing"], "tokens": fp8["tokens"], "routing_histograms_match": True,
            "unprofiled_median_winner": min((fp8, nvfp4), key=lambda x: x["unprofiled_graph_event_median_us"])["format"],
            "profiled_kernel_sum_winner": min((fp8, nvfp4), key=lambda x: x["profiled_kernel_duration_sum_us"])["format"],
            "nvfp4_minus_fp8_profiled_total_us": difference("profiled_kernel_duration_sum_us"),
            "nvfp4_minus_fp8_profiled_projection_us": difference("profiled_projection_kernel_sum_us"),
            "nvfp4_minus_fp8_profiled_other_us": difference("profiled_other_kernel_sum_us"),
            "nvfp4_minus_fp8_gpu_kernel_count": nvfp4["profiled_kernel_event_count"] - fp8["profiled_kernel_event_count"],
            "component_duration_deltas_us": {g: round(nvfp4["components"][g]["duration_sum_us"] - fp8["components"][g]["duration_sum_us"], 6) for g in GROUPS}}


def make_notes(summary):
    rows = summary["traces"]
    index = {(r["routing"], r["tokens"], r["format"]): r for r in rows}
    lines = [
        "Local Qwen3 profiler component analysis, r01",
        "28 existing traces: 14 NVFP4/CUTLASS and 14 matched FP8/Triton. CPU analysis only.",
        "Each trace is one CUDA graph replay, not a repeated profiler experiment.",
        "Chrome ts/dur units are us; displayTimeUnit='ms' is a viewer preference.",
        "",
        "KEY OBSERVATIONS (all durations below are profiler CUDA kernel sums unless explicitly called median)",
        "B=1: NVFP4 has both longer projection kernels and more non-projection work than FP8.",
        "B=8 and 2048: NVFP4's shorter projection-kernel durations outweigh its higher auxiliary sum.",
        "B=8192: NVFP4 projection kernels remain shorter, but its larger auxiliary sum reverses the full-kernel-sum ranking.",
        "This local decomposition is consistent with the unprofiled ranking; it does not prove a causal performance model.",
        "",
        "Focus table: route B format | median_us | GPU kernels | projection_us | other_us | sum_us | graph_calls",
    ]
    for route in ("uniform", "zipf"):
        for b in FOCUS:
            for fmt in ("fp8", "nvfp4"):
                r = index[route, b, fmt]
                lines.append(f"{route:7s} {b:4d} {fmt:5s} | {r['unprofiled_graph_event_median_us']:9.3f} | {r['profiled_kernel_event_count']:2d} | "
                             f"{r['profiled_projection_kernel_sum_us']:9.3f} | {r['profiled_other_kernel_sum_us']:9.3f} | "
                             f"{r['profiled_kernel_duration_sum_us']:9.3f} | {r['host_runtime_call_counts']['cudaGraphLaunch']}")
    lines += ["", "Focus component sums (us; kernel-event counts in parentheses):"]
    for route in ("uniform", "zipf"):
        for b in FOCUS:
            for fmt in ("fp8", "nvfp4"):
                r = index[route, b, fmt]
                active = [f"{g}={v['duration_sum_us']:.3f}({v['kernel_event_count']})" for g, v in r["components"].items() if v["kernel_event_count"]]
                lines.append(f"{route} B={b} {fmt}: " + "; ".join(active))
    lines += ["", "B=8192 COMPONENT EVIDENCE:"]
    for route in ("uniform", "zipf"):
        a, b = index[route, 8192, "fp8"], index[route, 8192, "nvfp4"]
        for fmt, r in (("FP8", a), ("NVFP4", b)):
            c = r["components"]
            lines.append(f"{route} {fmt}: projection={r['profiled_projection_kernel_sum_us']:.3f} us; "
                         f"row_shuffles={c['row_shuffles']['duration_sum_us']:.3f}; "
                         f"activation_and_quantization={c['activation_and_quantization']['duration_sum_us']:.3f}; "
                         f"output_operations={c['output_operations']['duration_sum_us']:.3f}.")
        lines.append(f"{route}: NVFP4 other-kernel share={100*b['profiled_other_kernel_sum_us']/b['profiled_kernel_duration_sum_us']:.2f}%; "
                     f"FP8={100*a['profiled_other_kernel_sum_us']/a['profiled_kernel_duration_sum_us']:.2f}%.")
    lines += [
        "", "INTERPRETATION AND LIMITS:",
        "- Across B=8..8192 the kernel count stays 15 for NVFP4 versus 10 for FP8. Thus increasing launch count is not the observed cause of the high-B reversal; existing auxiliary kernels become longer.",
        "- At B=1 counts are 16 versus 9, including an unattributed NVFP4 memcpy32_post kernel (0.992 us). Its cause is unresolved and it is not silently excluded.",
        "- All traces contain one host cudaGraphLaunch. GPU kernel-event counts are graph-node executions, not that many CPU API launches. Host runtime durations and synchronization durations are retained separately; they are never added to CUDA sums or interpreted as normal serving overhead.",
        "- Projection labels are whole kernel durations, not isolated tensor-core time. FP8 fused_moe_kernel includes addressing/scales and routed-weight work; the NVFP4 path exposes explicit shuffle/weighting/reduction kernels. Component boundaries differ.",
        "- NVFP4's second FP4 conversion is labeled SiLU+mul+quantization from the local source call order, not from guessing C++ template boolean semantics. The dtype-copy label is deliberately coarse; memcpy32_post stays unattributed.",
        "- Kernel duration sums, interval unions and spans come from an instrumented single replay. They are not substituted for the separately collected five-repeat CUDA-event medians and have no independent repeat uncertainty.",
        "- No hardware DRAM counters, memory bandwidth, arithmetic utilization, cache-miss rates, or CPU launch-cost intervention are present. Explicit memory-related kernels suggest a candidate explanation, not measured traffic or a proven bandwidth bottleneck.",
        "- FP8 weights are synthetic; NVFP4 is calibrated Qwen3 layer 0. These are untuned paths from different runs, with synthetic matching route histograms, not same-weight quality/serving experiments.",
        "- The existing traces support separating small-B projection/setup cost from large-B auxiliary growth. They do not identify a unique cause, establish a fitted crossover, or predict the benefit of fusion/retuning without running it.",
    ]
    mismatches = [p for p in summary["paired_comparisons"] if p["unprofiled_median_winner"] != p["profiled_kernel_sum_winner"]]
    duration_differences = [r["sum_vs_unprofiled_median_difference_pct"] for r in rows]
    lines += ["", f"Profile/unprofiled winner disagreements across 14 pairs: {len(mismatches)}.",
              f"Profiler kernel sums differ from separate unprofiled medians by {min(duration_differences):+.3f}% to {max(duration_differences):+.3f}%; this is not an estimate of profiler overhead.",
              "All 28 trace event counts and duration sums matched their original JSONL profile receipts.",
              f"Unclassified kernel events: {summary['unclassified_kernel_events']}.",
              "Reproduce: python3 code/analyze_profile_components.py (run from project root)."]
    return "\n".join(lines) + "\n"


def analyze_awq_focus(root, output):
    """Reuse the same accounting on twelve existing, hash-verified traces."""
    focus, formats = (2048, 8192), ("fp8", "awq", "nvfp4")
    runs = RUNS | {"awq": "qwen3_awq_survey_r06"}
    records, sources, names = [], [], set()
    for fmt in formats:
        directory = root / "runs" / runs[fmt]
        hashes = json.loads((directory / "SHA256.json").read_text())
        for route in ("uniform", "zipf"):
            path = directory / f"qwen3_{route}_{fmt}.jsonl"
            if hashes.get(path.name) != digest(path):
                raise ValueError(f"measurement SHA mismatch: {path}")
            sources.append({"path": str(path.relative_to(root)), "sha256": digest(path)})
            selected = [row for line in path.read_text().splitlines() if line.strip()
                        if (row := json.loads(line))["tokens"] in focus]
            if sorted(row["tokens"] for row in selected) != list(focus):
                raise ValueError(f"Missing or duplicate focus observations: {path}")
            for row in selected:
                if row["status"] != "ok" or row["format"] != fmt or row["routing"]["kind"] != route:
                    raise ValueError(f"Unexpected focus observation: {path}")
                original = Path(row["profile"]["trace"])
                trace_path = directory / original.parent.name / original.name
                if hashes.get(str(trace_path.relative_to(directory))) != digest(trace_path):
                    raise ValueError(f"trace SHA mismatch: {trace_path}")
                trace = json.loads(trace_path.read_text())
                names.update(e["name"] for e in trace["traceEvents"] if e.get("cat") == "kernel")
                records.append((row, trace_path))
    catalog = {name: f"k{i:02d}" for i, name in enumerate(sorted(names))}
    traces = []
    for row, path in records:
        trace = analyze_trace(root, row, path, catalog)
        trace["graph_output_validation"] = {k: v for k, v in row.items() if k.startswith("cuda_graph_")}
        trace["checkpoint_identity"] = {k: row["backend"].get(k) for k in
            ("checkpoint", "checkpoint_revision", "artifact_sha256", "requested_backend", "quantization")}
        traces.append(trace)
    index = {(r["routing"], r["tokens"], r["format"]): r for r in traces}
    comparisons = []
    for route in ("uniform", "zipf"):
        for batch in focus:
            for alternative in ("awq", "nvfp4"):
                base, other = index[route, batch, "fp8"], index[route, batch, alternative]
                if base["routing_counts"] != other["routing_counts"]:
                    raise ValueError("Focus routing histograms differ")
                comparisons.append({"routing": route, "tokens": batch, "alternative": alternative,
                    "baseline": "fp8", "routing_histograms_match": True,
                    "alternative_minus_fp8_projection_us": round(other["profiled_projection_kernel_sum_us"] - base["profiled_projection_kernel_sum_us"], 6),
                    "alternative_minus_fp8_other_us": round(other["profiled_other_kernel_sum_us"] - base["profiled_other_kernel_sum_us"], 6),
                    "alternative_minus_fp8_kernel_sum_us": round(other["profiled_kernel_duration_sum_us"] - base["profiled_kernel_duration_sum_us"], 6),
                    "alternative_projection_over_fp8": other["profiled_projection_kernel_sum_us"] / base["profiled_projection_kernel_sum_us"],
                    "profiled_kernel_sum_winner": min((base, other), key=lambda r: r["profiled_kernel_duration_sum_us"])["format"],
                    "unprofiled_median_winner": min((base, other), key=lambda r: r["unprofiled_graph_event_median_us"])["format"]})
    for trace in traces:
        trace["projection_fraction_of_kernel_sum"] = trace["profiled_projection_kernel_sum_us"] / trace["profiled_kernel_duration_sum_us"]
        trace["routing_histogram_sha256"] = hashlib.sha256(json.dumps(trace.pop("routing_counts"), separators=(",", ":")).encode()).hexdigest()
    old_path = root / "runs/profile_component_analysis_r01/summary.json"
    old = {(r["routing"], r["tokens"], r["format"]): r for r in json.loads(old_path.read_text())["traces"]}
    for trace in traces:
        if trace["format"] in RUNS:
            previous = old[trace["routing"], trace["tokens"], trace["format"]]["components"]
            # Compare source quantities exactly. Python 3.12+ changed float sum
            # rounding, so the derived fractions can differ by a final bit.
            if any(value[key] != previous[group][key] for group, value in trace["components"].items()
                   for key in ("duration_sum_us", "kernel_event_count")):
                raise ValueError("Reused FP8/NVFP4 accounting differs from the existing component result")
    code = ["runs/source_snapshot/vllm/model_executor/layers/fused_moe/experts/marlin_moe.py",
            "runs/source_snapshot/vllm/model_executor/layers/fused_moe/experts/cutlass_moe.py",
            "runs/source_snapshot/vllm/model_executor/layers/fused_moe/fused_moe.py"]
    summary = {"schema_version": 1, "scope": "CPU-only focus comparison of existing single instrumented CUDA-graph replays",
        "script_sha256": digest(Path(__file__)), "trace_count": len(traces), "focus_tokens": list(focus),
        "duration_unit": "microseconds", "measurement_sources": sources,
        "reused_component_summary": {"path": str(old_path.relative_to(root)), "sha256": digest(old_path), "eight_baseline_component_durations_and_counts_match_exactly": True},
        "classification_sources": [{"path": p, "sha256": digest(root / p)} for p in code],
        "awq_call_order_evidence": "marlin_moe.py _fused_marlin_moe: two moe_wna16_marlin_gemm calls with activation between; fused_marlin_moe: route alignment then torch.sum with expert_map=None",
        "kernel_catalog": {value: name for name, value in catalog.items()},
        "unclassified_kernel_events": sum(t["components"]["unclassified"]["kernel_event_count"] for t in traces),
        "traces": traces, "comparisons_to_fp8": comparisons,
        "limitations": ["Each point has one instrumented replay; separate unprofiled CUDA-event measurements retain five within-process repetitions, not independent system repetitions",
            "Projection durations cover whole kernels with different fused work, including scales/addressing/reduction; they are not isolated Tensor Core instruction time",
            "No dynamic hardware counters; no established bandwidth, utilization, occupancy, or dequantization bottleneck",
            "Kernel-duration arithmetic is an observed decomposition, not an achievable speedup or a causal fusion/retuning result",
            "One host cudaGraphLaunch per trace; GPU kernel nodes are not separate host API launches; host timings are not added to CUDA sums",
            "FP8 weights are synthetic; AWQ and NVFP4 are two public calibrated checkpoints; matching shape/routing does not establish same-weight quality or serving behavior",
            "AWQ accepted BF16-roundoff diagnostics are preserved; accepted non-bitwise output is not exact equality or conversion-quality validation"]}
    if summary["unclassified_kernel_events"]:
        raise ValueError("Unclassified focus kernels require explicit handling")
    output.mkdir(parents=True, exist_ok=True)
    target = output / "awq_focus_r06.json"
    target.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": "complete", "trace_count": len(traces), "output": str(target)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("runs/profile_component_analysis_r01"))
    parser.add_argument("--awq-focus", action="store_true", help="Write only awq_focus_r06.json for B2048/8192; preserve original summary/notes")
    args = parser.parse_args()
    root = args.project_root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    if args.awq_focus:
        analyze_awq_focus(root, output)
        return
    records, sources, names = [], [], set()
    for fmt, run in RUNS.items():
        for routing in ("uniform", "zipf"):
            path = root / "runs" / run / f"qwen3_{routing}_{fmt}.jsonl"
            sources.append({"path": str(path.relative_to(root)), "sha256": digest(path)})
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            if sorted(r["tokens"] for r in rows) != list(TOKENS):
                raise ValueError(f"unexpected point set: {path}")
            for row in rows:
                if row["status"] != "ok" or row["timing_mode"] != "cuda_graph_replay":
                    raise ValueError(f"invalid run record: {path}, B={row['tokens']}")
                trace_path = root / row["profile"]["trace"]
                trace = json.loads(trace_path.read_text())
                names.update(e["name"] for e in trace["traceEvents"] if e.get("cat") == "kernel")
                records.append((row, trace_path))
    catalog = {name: f"k{i:02d}" for i, name in enumerate(sorted(names))}
    traces = [analyze_trace(root, row, path, catalog) for row, path in records]
    index = {(r["routing"], r["tokens"], r["format"]): r for r in traces}
    pairs = [comparison(index[route, b, "fp8"], index[route, b, "nvfp4"]) for route in ("uniform", "zipf") for b in TOKENS]
    code_sources = ["runs/source_snapshot/vllm/model_executor/layers/fused_moe/experts/cutlass_moe.py",
                    "runs/source_snapshot/vllm/model_executor/layers/fused_moe/fused_moe.py"]
    code_sources += [f"runs/{run}/submitted_code/bench_expert_formats.py" for run in RUNS.values()]
    summary = {"schema_version": 1, "scope": "CPU analysis of existing single-replay profiler traces only",
               "script_sha256": digest(Path(__file__)), "trace_count": len(traces), "focus_tokens": list(FOCUS),
               "duration_unit": "microseconds; Chrome trace displayTimeUnit does not rescale ts/dur",
               "measurement_sources": sources,
               "classification_source_snapshots": [{"path": s, "sha256": digest(root / s)} for s in code_sources],
               "classification_source_note": "NVFP4 call order: cutlass_moe.py lines 598-674; FP8 fused_experts_impl lines 1706-1790; benchmark profile_call profiles one fn() then synchronize.",
               "kernel_catalog": {kid: name for name, kid in catalog.items()},
               "unclassified_kernel_events": sum(r["components"]["unclassified"]["kernel_event_count"] for r in traces),
               "component_boundary_warning": "Whole projection kernels include different fused work; auxiliary groups are not equivalent primitive sets.",
               "claims_excluded": ["actual DRAM traffic", "bandwidth bottleneck proof", "causal fusion benefit", "fitted crossover", "quality", "serving gain", "independent-run uncertainty"],
               "traces": traces, "paired_comparisons": pairs}
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (output / "notes.txt").write_text(make_notes(summary))
    print(json.dumps({"trace_count": len(traces), "unclassified_kernel_events": summary["unclassified_kernel_events"],
                      "output": str(output), "status": "complete"}))


if __name__ == "__main__":
    main()
