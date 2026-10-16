#!/usr/bin/env python3
"""Analyze run_matrix outputs and plot measured batch x gamma heatmaps.

Usage:
  python plot_results.py runs/CAMPAIGN --output figures/CAMPAIGN
  python plot_results.py runs/CAMPAIGN --output figures/CAMPAIGN --no-plot
  python plot_results.py probabilities_comparison.csv --kind routes --output figures/F1

CSV/JSON analysis uses the standard library. Rendering uses matplotlib/numpy,
imported only when requested. Missing cells stay blank. All original run values
are exported; a shared request/step population is never counted as independent
system repeats. Default speedup is the ratio of run-median end-to-end generated
token throughput to gamma=0, including prefill, wave setup and drain. The optional
tpot_speedup uses the median of per-run request-mean effective TPOT instead.
These two metrics must not be confused with a pure decode-kernel speedup.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import textwrap
from collections import defaultdict
from pathlib import Path


def _finite(value):
    if value in (None, ""):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _bool(value) -> bool:
    return value is True or str(value).lower() in ("true", "1")


EXECUTION_FIELDS = ("execution_mode", "dtype", "quantization", "kv_cache_memory_bytes",
                    "max_model_len", "attention_backend", "kernel_backend",
                    "admission_mode", "engine_max_num_seqs")
REGIME_FIELDS = ("model", "device", "workload_sha256", "draft_source", "measurement_kind", *EXECUTION_FIELDS)


def _stable_json(value) -> str:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            pass
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def resolve_regime(record: dict, context: dict) -> dict:
    """Keep missing provenance separate from known values; canonicalize JSON."""
    result = {}
    for key in REGIME_FIELDS:
        default = "wave" if key == "admission_mode" else "unspecified"
        value = record.get(key, context.get(key, default))
        if key in ("attention_backend", "kernel_backend"):
            value = _stable_json(value)
        elif key in EXECUTION_FIELDS:
            value = "none" if value is None else str(value)
        result[key] = value
    return result


def route_instrumented(record: dict, context: dict | None = None) -> bool:
    context = context or {}
    # Routed-expert return instruments the whole engine, even when a particular
    # episode forgot to set trace_enabled or did not enable the Python hook.
    enabled = (_bool(record.get("trace_enabled", False))
               or _bool(record.get("native_route_return", False))
               or _bool(record.get("enable_return_routed_experts", False))
               or _bool(context.get("native_route_return", False))
               or record.get("measurement_kind") == "route_diagnostic")
    return enabled or any(route_instrumented(e, context) for e in record.get("episodes", []))


def _requests(record: dict) -> list[dict]:
    if "episodes" in record:
        return [request for episode in record["episodes"] for request in episode.get("requests", [])]
    return record.get("requests", [])


def _context(path: Path) -> dict:
    result = {}
    config = path.parent / "configuration.json"
    if config.exists():
        obj = json.loads(config.read_text())
        result["workload_sha256"] = obj.get("workload_sha256", "unspecified")
        result["workload"] = obj.get("plan", {}).get("workload", "unspecified")
        result["admission_mode"] = (obj.get("admission_mode") or obj.get("args", {}).get("admission_mode")
                                    or obj.get("plan", {}).get("admission_mode", "wave"))
        if "engine_max_num_seqs" in obj:
            result["engine_max_num_seqs"] = obj["engine_max_num_seqs"]
    environment = path.parent / "environment.json"
    if environment.exists():
        obj = json.loads(environment.read_text())
        result["device"] = obj.get("gpu", "unspecified")
    engine_args = path.parent / "engine_args.json"
    if engine_args.exists():
        obj = json.loads(engine_args.read_text())
        if "enforce_eager" in obj:
            result["execution_mode"] = "eager" if _bool(obj["enforce_eager"]) else "compiled"
        for key in ("dtype", "quantization", "kv_cache_memory_bytes", "max_model_len"):
            result[key] = obj.get(key, "none" if key == "quantization" else "unspecified")
        if "max_num_seqs" in obj:
            result["engine_max_num_seqs"] = obj["max_num_seqs"]
        attention = {key: obj[key] for key in ("attention_backend", "attention_config", "attn_backend") if key in obj}
        kernel = {key: obj[key] for key in ("kernel_backend", "kernel_config", "moe_backend") if key in obj}
        result["attention_backend"] = _stable_json(attention or {"selection": "auto_unspecified"})
        result["kernel_backend"] = _stable_json(kernel or {"selection": "auto_unspecified"})
        result["native_route_return"] = _bool(obj.get("enable_return_routed_experts", False))
        table = (obj.get("speculative_config") or {}).get("num_speculative_tokens_per_batch_size")
        if table is not None:
            result["native_batch_table"] = json.dumps(table)
    return result


def load_runs(paths: list[str | Path]) -> tuple[list[dict], list[dict]]:
    """Read per-run JSON, CSV, or JSON {records:[...]}; retain exclusion reasons."""
    inputs = []
    for path in map(Path, paths):
        inputs.extend(sorted(path.rglob("perf-*.json")) if path.is_dir() else [path])
    runs, excluded = [], []
    for path in sorted(set(p.resolve() for p in inputs)):
        if path.suffix == ".csv":
            with path.open(newline="") as f:
                records = list(csv.DictReader(f))
        else:
            obj = json.loads(path.read_text())
            records = obj if isinstance(obj, list) else obj.get("records", [obj])
        context = _context(path)
        for index, record in enumerate(records):
            label = str(record.get("label", path.stem))
            source = str(path.resolve())
            reason = None
            if route_instrumented(record, context):
                reason = "route instrumentation enabled"
            elif label.startswith("warmup"):
                reason = "warmup"
            elif "prediction" in str(record.get("evidence_kind", "")).lower():
                reason = "analytic predictions are not measured performance"
            elif record.get("all_finished") is not None and not _bool(record["all_finished"]):
                reason = "incomplete population; report separately"
            if reason:
                excluded.append({"source": source, "label": label, "reason": reason})
                continue
            if "batch" not in record or "gamma" not in record:
                excluded.append({"source": source, "label": label, "reason": "missing batch/gamma"})
                continue
            batch, gamma = int(record["batch"]), int(record["gamma"])
            wall = _finite(record.get("wall_s"))
            tokens = _finite(record.get("output_tokens"))
            tps = _finite(record.get("tokens_per_second"))
            if tps is None and wall and tokens is not None:
                tps = tokens / wall
            if batch <= 0 or gamma < 0 or tps is None or tps <= 0:
                excluded.append({"source": source, "label": label, "reason": "invalid or missing measured throughput"})
                continue
            requests = _requests(record)
            tpots = [v for r in requests if (v := _finite(r.get("effective_tpot_ms"))) is not None]
            tpot = _finite(record.get("effective_tpot_ms"))
            if tpot is None and tpots:
                tpot = statistics.mean(tpots)
            energy = _finite(record.get("joules_per_token"))
            if energy is None and tokens:
                total_energy = _finite(record.get("energy_j"))
                energy = total_energy / tokens if total_energy is not None else None
            repeat = re.search(r"(?:^|-)r(\d+)(?:-|$)", label)
            proposal_budget = int(record.get("proposal_budget", gamma))
            mode = record.get("mode", "static")
            regime = "direct_draft" if proposal_budget == gamma else f"full_draft_budget_{proposal_budget}"
            runs.append({**context, **resolve_regime(record, context), "source": source, "record_index": index, "label": label,
                "repeat": int(repeat.group(1)) if repeat else record.get("repeat"),
                "mode": mode, "strategy": ("native_batch_table" if mode == "native" and
                    (record.get("native_batch_table") or context.get("native_batch_table"))
                    else "static" if mode == "native" else mode),
                "proposal_regime": regime, "proposal_budget": proposal_budget,
                "batch": batch, "gamma": gamma, "wall_s": wall, "output_tokens": tokens,
                "request_count": len(requests), "tokens_per_second": tps,
                "effective_tpot_ms": tpot, "joules_per_token": energy,
                "energy_source": (record.get("energy") or {}).get("source", "see raw energy record"),
                "evidence_kind": "measured_run"})
    return runs, excluded


def summarize(runs: list[dict]) -> list[dict]:
    baselines, cells = defaultdict(list), defaultdict(list)
    for row in runs:
        regime = tuple(row[key] for key in REGIME_FIELDS)
        if row["gamma"] == 0 and row["proposal_budget"] == 0 and row["strategy"] == "static":
            baselines[(regime, row["batch"])].append(row)
        key = (*regime, row["strategy"], row["proposal_regime"], row["batch"], row["gamma"])
        cells[key].append(row)
    summary = []
    for key, observations in sorted(cells.items()):
        row = dict(zip((*REGIME_FIELDS, "strategy", "proposal_regime", "batch", "gamma"), key))
        row["proposal_budget"] = observations[0]["proposal_budget"]
        base = baselines.get((key[:len(REGIME_FIELDS)], row["batch"]), [])
        row.update(n_runs=len(observations), baseline_n_runs=len(base),
                   sources=json.dumps([r["source"] for r in observations]),
                   evidence_kind="measured_run_summary")
        for metric in ("tokens_per_second", "effective_tpot_ms", "joules_per_token"):
            values = [r[metric] for r in observations if r[metric] is not None]
            row[metric] = statistics.median(values) if values else None
            row[metric + "_min"] = min(values) if values else None
            row[metric + "_max"] = max(values) if values else None
            row[metric + "_raw"] = json.dumps(values)
        base_tps = [r["tokens_per_second"] for r in base]
        base_tpot = [r["effective_tpot_ms"] for r in base if r["effective_tpot_ms"] is not None]
        row["speedup"] = row["tokens_per_second"] / statistics.median(base_tps) if base_tps else None
        row["tpot_speedup"] = statistics.median(base_tpot) / row["effective_tpot_ms"] \
            if base_tpot and row["effective_tpot_ms"] else None
        row["baseline_tokens_per_second_raw"] = json.dumps(base_tps)
        row["baseline_effective_tpot_ms_raw"] = json.dumps(base_tpot)
        row["aggregation"] = "median across run records; ratio of medians; no CI inferred"
        summary.append(row)
    return summary


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot_heatmaps(summary: list[dict], output: Path, metric: str, formats: list[str], title: str | None = None) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    import numpy as np

    panels = defaultdict(list)
    for row in summary:
        # An adaptive run's configured gamma is a cap, not its chosen gamma.
        if row["strategy"] == "static":
            key = tuple(row[k] for k in (*REGIME_FIELDS, "proposal_regime"))
            panels[key].append(row)
    labels = {"speedup": "Token throughput / gamma=0",
              "tpot_speedup": "TPOT(gamma=0) / TPOT",
              "tokens_per_second": "Generated tokens/s",
              "joules_per_token": "GPU joules / generated token"}
    files = []
    for index, (key, rows) in enumerate(sorted(panels.items())):
        batches, gammas = sorted({r["batch"] for r in rows}), sorted({r["gamma"] for r in rows})
        matrix = np.full((len(batches), len(gammas)), np.nan)
        counts = {}
        for row in rows:
            i, j = batches.index(row["batch"]), gammas.index(row["gamma"])
            if row[metric] is not None:
                matrix[i, j] = row[metric]
                counts[i, j] = row["n_runs"]
        valid = matrix[np.isfinite(matrix)]
        if not len(valid):
            continue
        fig, ax = plt.subplots(figsize=(max(6, len(gammas) * .85), max(3.8, len(batches) * .65)))
        norm = TwoSlopeNorm(vmin=min(float(valid.min()), .99), vcenter=1,
                            vmax=max(float(valid.max()), 1.01)) if "speedup" in metric else None
        color = plt.get_cmap("RdBu" if "speedup" in metric else "viridis").copy()
        color.set_bad("#e8e8e8")
        plot = ax.imshow(matrix, cmap=color, norm=norm, aspect="auto")
        for i in range(len(batches)):
            for j in range(len(gammas)):
                value = matrix[i, j]
                ax.text(j, i, f"{value:.2f}\nn={counts[i,j]}" if np.isfinite(value) else "missing",
                        ha="center", va="center", fontsize=9,
                        bbox={"facecolor": "white", "alpha": .65, "edgecolor": "none", "pad": 1})
                if "speedup" in metric and np.isfinite(value) and value < 1:
                    from matplotlib.patches import Rectangle
                    ax.add_patch(Rectangle((j-.5, i-.5), 1, 1, fill=False, hatch="//", edgecolor="#555", lw=.5))
        ax.set_xticks(range(len(gammas)), gammas)
        ax.set_yticks(range(len(batches)), batches)
        ax.set_xlabel("Draft length gamma")
        ax.set_ylabel("Synchronous wave batch B")
        regime = dict(zip((*REGIME_FIELDS, "proposal_regime"), key))
        device = str(regime['device']).replace('NVIDIA ', '').replace(' Blackwell Server Edition', '')
        heading = title or f"{Path(regime['model']).name} | {regime['draft_source']} | {regime['proposal_regime']}"
        ax.set_title(textwrap.fill(heading, 72) + f"\n{device} | {regime['execution_mode']}", fontsize=10)
        bar = fig.colorbar(plot, ax=ax)
        bar.set_label(labels[metric], fontsize=10)
        fig.text(.02, .01, "Run medians; n = repeats. Throughput/energy include prefill.\n"
                 "Grey = unavailable; hatching = speedup < 1. Full provenance in CSV.", fontsize=8)
        fig.tight_layout(rect=(0, .065, 1, 1))
        tag = hashlib.sha256(json.dumps(key).encode()).hexdigest()[:8]
        for ext in formats:
            filename = output / f"{metric}_{index}_{tag}.{ext}"
            fig.savefig(filename, dpi=200, bbox_inches="tight", metadata={"Title": labels[metric]})
            files.append(str(filename))
        plt.close(fig)
    return files


def plot_routes(path: Path, output: Path, formats: list[str]) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with path.open(newline="") as f:
        records = list(csv.DictReader(f))
    groups = defaultdict(list)
    for row in records:
        observed = _finite(row.get("observed_window_distinct_mean", row.get("observed_distinct")))
        predicted = _finite(row.get("predicted_distinct"))
        if observed is not None and predicted is not None:
            groups[row.get("partition", "unspecified")].append((predicted, observed))
    if not groups:
        raise ValueError("no observed distinct-expert values to plot")
    fig, ax = plt.subplots(figsize=(5, 4.5))
    maximum = max(max(pair) for pairs in groups.values() for pair in pairs)
    for partition, pairs in sorted(groups.items()):
        ax.scatter([p[0] for p in pairs], [p[1] for p in pairs], label=partition, alpha=.65, s=16)
    ax.plot([0, maximum], [0, maximum], color="black", linestyle="--", lw=1)
    ax.set(xlabel="Predicted distinct experts (token-inclusion q)", ylabel="Observed distinct experts",
           title="Route-model diagnostic (see CSV evidence_kind)")
    ax.legend()
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    files = []
    for ext in formats:
        filename = output / f"distinct_prediction.{ext}"
        fig.savefig(filename, dpi=200)
        files.append(str(filename))
    plt.close(fig)
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--kind", choices=["performance", "routes"], default="performance")
    parser.add_argument("--metric", choices=["speedup", "tpot_speedup", "joules_per_token", "tokens_per_second"], default="speedup")
    parser.add_argument("--formats", nargs="+", choices=["svg", "pdf", "png"], default=["svg", "pdf"])
    parser.add_argument("--no-plot", action="store_true", help="write analysis without matplotlib")
    parser.add_argument("--title", help="Human-readable figure title; full regime remains in CSV")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    files = []
    if args.kind == "routes":
        if len(args.inputs) != 1:
            parser.error("route plot accepts one comparison CSV")
        if not args.no_plot:
            files = plot_routes(Path(args.inputs[0]), output, args.formats)
    else:
        runs, excluded = load_runs(args.inputs)
        if not runs:
            raise ValueError(f"no completed measured performance runs; exclusions={excluded}")
        summary = summarize(runs)
        write_csv(output / "raw_runs.csv", runs)
        write_csv(output / "summary.csv", summary)
        (output / "exclusions.json").write_text(json.dumps(excluded, indent=2))
        if not args.no_plot:
            files = plot_heatmaps(summary, output, args.metric, args.formats, args.title)
        (output / "analysis_metadata.json").write_text(json.dumps({
            "metric": args.metric, "title": args.title, "inputs": args.inputs, "run_count": len(runs), "excluded_count": len(excluded),
            "aggregation": "ratio of run medians; raw repeated values retained; no inferred confidence interval",
            "uncertainty": "few repeats characterize local variation only",
            "boundary": "throughput and energy include prefill/wave setup/drain; TPOT is request-mean per run",
            "evidence_kind": "measured_generation_runs", "figures": files}, indent=2))
    print(json.dumps({"output": str(output), "figures": files}))


if __name__ == "__main__":
    main()
