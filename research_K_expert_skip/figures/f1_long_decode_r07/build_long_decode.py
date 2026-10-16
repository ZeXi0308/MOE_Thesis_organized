"""Build native long-decode occupancy from completed counted graph replays.

CPU-only; imports the existing plot_results parsers and never writes raw inputs.
The explicit E/k are checked against each routed layer's recorded installation.
The uniform iid curve is a reference, not a fit. No per-expert skew is inferred.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
PROJECT = Path(__file__).resolve().parents[2]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value):
    require(isinstance(value, (int, float)) and value >= 0 and int(value) == value,
            f"Expected a nonnegative integer counter, got {value!r}")
    return int(value)


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path, help="run root, bench directory, or measurements.jsonl")
    parser.add_argument("--model", required=True, help="human-readable checkpoint identity")
    parser.add_argument("--experts", required=True, type=int)
    parser.add_argument("--top-k", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--weight-format", help="explicit label; defaults to quantization plus nonquantized dtype")
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    args = parser.parse_args()
    require(0 < args.top_k <= args.experts, "Require 0 < k <= E")
    require(len(set(args.batches)) == len(args.batches) and min(args.batches) > 0, "Invalid batch list")

    module_path = PROJECT / "code/plot_results.py"
    spec = importlib.util.spec_from_file_location("existing_plot_results", module_path)
    existing = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(existing)
    source = args.run.resolve()
    if source.is_dir():
        source = source / ("measurements.jsonl" if (source / "measurements.jsonl").exists() else "bench/measurements.jsonl")
    root = source.parent
    sources = existing.Sources()
    for path in (source, root / "config.json", root / "status.json", module_path):
        sources.use(path)
    config = existing.read_json(root / "config.json")
    status = existing.read_json(root / "status.json")
    require(status.get("status") == "COMPLETE", "Require a complete bench, not a partial matrix")
    run_args = config["args"]
    prefix, outputs = integer(run_args["prompt_tokens"]), integer(run_args["newtokens"])
    require(outputs > 1, "No decode predictions in this run")
    require(run_args.get("execution") == "graph", "This builder expects counted graph replay inputs")
    quant = run_args.get("quantization")
    dtype = run_args.get("dtype", "unknown")
    weight_format = args.weight_format or ("BF16" if not quant and dtype == "bfloat16" else
        f"{quant} weights/activations; nonquantized dtype {dtype}" if quant else dtype)
    population = [r for r in existing.read_rows(source) if r.get("arm") == "native" and r.get("batch") in args.batches]
    require(population, "No requested native measurements")
    repeats = sorted({integer(r["repeat"]) for r in population})
    require(len(population) == len(repeats) * len(args.batches), "Missing or duplicated native cells")
    aggregate, layers, seen = [], [], set()
    additive = ("events", "tokens", "slots_before", "slots_after", "skipped_slots", "singleton_slots_before", "distinct_before", "distinct_after")
    for row in sorted(population, key=lambda r: (r["repeat"], r["batch"])):
        repeat, batch = integer(row["repeat"]), integer(row["batch"])
        require((repeat, batch) not in seen, "Duplicated native cell")
        seen.add((repeat, batch))
        require(row.get("spec") is None, "Native measurement must not apply a policy")
        require(row.get("logical_counter_source") == "separate_graph_process_identical_output_tokens"
                and row.get("counter_identity_verified") is True, "Missing counted/performance output identity")
        totals, counter_path = existing.bench_counters(row, root, sources)
        snapshot = existing.unwrap_snapshot(existing.read_json(counter_path))
        layer_counts = snapshot["layers"]["layers"]
        require(layer_counts and snapshot.get("spec") is None, "Missing native routed-layer counters")
        decode = integer(snapshot["phase_steps"]["decode"])
        require(decode == outputs - 1, "Decode population differs from fixed output protocol")
        require(integer(row["decode_output_tokens"]) == batch * decode, "Decode output denominator mismatch")
        runtime = [v for v in snapshot["graph"]["runtime_rows"] if v["phase"] == "decode"]
        require(len(runtime) == decode and all(v["actual_tokens"] == batch and v["padded_tokens"] == batch
                and v["runtime_mode"] == "FULL" for v in runtime), "Require complete, unpadded FULL decode execution")
        for key in additive:
            require(sum(integer(v[key]) for v in layer_counts.values()) == integer(totals[key]), f"Layer/total disagreement: {key}")
        require(integer(totals["skipped_slots"]) == 0 and totals["slots_before"] == totals["slots_after"], "Native routes were changed")
        require(integer(totals["events"]) == len(layer_counts) * decode, "Layer-step denominator mismatch")
        require(integer(totals["tokens"]) == batch * integer(totals["events"]), "Actual decode batch mismatch")
        require(integer(totals["slots_before"]) == args.top_k * integer(totals["tokens"]), "Routed k differs")
        uniform = (1 - args.top_k / args.experts) ** (batch - 1)
        base = {"model": args.model, "weight_format": weight_format, "repeat": repeat, "batch": batch,
                "experts": args.experts, "top_k": args.top_k, "prefix_tokens": prefix,
                "output_tokens_per_request": outputs, "decode_steps": decode}
        for name, values in sorted(layer_counts.items(), key=lambda item: [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", item[0])]):
            installed = snapshot["execution"][name.split("@", 1)[0]]
            require(installed["num_experts"] == args.experts and installed["original_top_k"] == args.top_k
                    and installed["top_k"] == args.top_k, "Explicit E/k disagree with routed layer installation")
            require(integer(values["events"]) == decode and integer(values["tokens"]) == batch * decode,
                    "Layer population mismatch")
            slots, singletons = integer(values["slots_before"]), integer(values["singleton_slots_before"])
            require(slots == batch * decode * args.top_k and 0 <= singletons <= slots, "Invalid layer slot counts")
            layers.append({**base, "layer": name, "events": decode, "tokens": integer(values["tokens"]),
                "singleton_slots": singletons, "slots": slots, "singleton_fraction": singletons / slots,
                "uniform_iid_fraction": uniform, "counter_source": counter_path})
        slots, singletons = integer(totals["slots_before"]), integer(totals["singleton_slots_before"])
        aggregate.append({**base, "routed_layers": len(layer_counts), "layer_events": integer(totals["events"]),
            "token_layer_observations": integer(totals["tokens"]), "singleton_slots": singletons, "slots": slots,
            "singleton_fraction": singletons / slots, "singleton_percent": 100 * singletons / slots,
            "uniform_iid_fraction": uniform, "uniform_iid_percent": 100 * uniform,
            "measured_minus_uniform_percentage_points": 100 * (singletons / slots - uniform),
            "counter_performance_identity_verified": True, "output_tokens_sha256": row["output_tokens_sha256"],
            "counter_source": counter_path, "performance_folder": row["folder"]})

    out = args.output_dir.resolve()
    require(not out.is_relative_to(root), "Derived output must not be inside raw bench input")
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "occupancy_by_batch.csv", aggregate)
    write_csv(out / "occupancy_by_layer.csv", layers)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for repeat in repeats:
        rows = [r for r in aggregate if r["repeat"] == repeat]
        label = "Measured native, long decode" if len(repeats) == 1 else f"Measured native, repeat {repeat}"
        ax.plot([r["batch"] for r in rows], [r["singleton_percent"] for r in rows], "o-", label=label, linewidth=1.8)
    bs = sorted(args.batches)
    ax.plot(bs, [100 * (1 - args.top_k / args.experts) ** (b - 1) for b in bs], "--", color="#dd8030",
            label=r"Uniform iid reference: $(1-k/E)^{B-1}$", linewidth=1.6)
    ax.set_xscale("log", base=2)
    ax.set_xticks(bs)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set(xlabel="Actual decode batch, B", ylabel="Singleton assignments / all routed slots (%)", ylim=(-2, 105))
    ax.grid(alpha=.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right", frameon=False)
    fig.suptitle(f"F1: {args.model} | {weight_format} | E={args.experts}, k={args.top_k}", fontsize=11)
    ax.set_title(f"Native-generated fixed-length trajectories: prefix {prefix}, outputs {outputs}", fontsize=9)
    fig.subplots_adjust(left=.11, right=.98, top=.85, bottom=.24)
    fig.text(.11, .13, f"{outputs - 1} decode steps per batch; counted replay matches each performance transcript.\n"
             f"{len(repeats)} source execution(s) per batch; tokens/layers are not independent runs. No skew fit or confidence interval.\n"
             "EOS ignored; occupancy is neither natural task quality nor a physical DRAM/time saving.", fontsize=7.5, va="top")
    for extension in ("pdf", "png", "svg"):
        fig.savefig(out / f"F1_long_decode.{extension}", dpi=220)
    plt.close(fig)
    interpretation = (
        "The CSV numerator counts singleton token-expert assignments; its denominator counts all native routed slots.\n"
        "Each per-layer ratio pools the complete generated decode trajectory; layers and tokens are not independent system repetitions.\n"
        "Uniform iid routing is an analytical reference only. These summaries do not contain complete per-expert inclusion marginals, "
        "so there is no fitted skew curve and no separation of routing skew from within-batch correlation.\n"
        "Compared with a short route trace, this measurement covers later states of long native-generated fixed-length continuations. "
        "Different prefix lengths, observation lengths, and content trajectories prevent a causal comparison based on trace duration alone.\n"
        "A future FP8 run with the same external prefix/output protocol is comparable as a format-specific generated-trajectory observation; "
        "it does not hold content/routing fixed or isolate the direct causal effect of quantization.\n"
        "The benchmark ignores EOS. It is not natural completion, and singleton opportunity alone is not observed policy savings, DRAM traffic, or latency.\n")
    (out / "interpretation.txt").write_text(interpretation)
    manifest = {"status": "DERIVED", "source_status": status["status"], "model": args.model,
        "weight_format": weight_format, "run": str(source), "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "source_executions_per_batch": len(repeats), "aggregate_rows": len(aggregate), "layer_rows": len(layers),
        "sources": list(sources.paths.values()), "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "parser_reuse": "plot_results.read_rows/bench_counters/unwrap_snapshot/Sources",
        "counter_identity_scope": "Joined measurement identity flag plus raw counted snapshot equality; no raw generation replay performed by this CPU builder",
        "limitations": interpretation.splitlines(), "warnings": sources.warnings}
    (out / "source.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": "DERIVED", "aggregate_rows": len(aggregate), "layer_rows": len(layers),
        "singleton_percent": {r["batch"]: r["singleton_percent"] for r in aggregate}, "output_dir": str(out)}, indent=2))


if __name__ == "__main__":
    main()
