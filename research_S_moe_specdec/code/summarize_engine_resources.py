#!/usr/bin/env python3
"""Read saved engine resources, without loading a model or inferring missing memory.

Accepts campaign directories or individual engine directories. The CSV contains
one row per engine; log_evidence.jsonl preserves every matched startup report and
its nearby context, including ambiguous multi-model loading reports.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

GIB = 1024 ** 3
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
PATTERNS = {
    "model_loading_memory_gib": re.compile(r"Model loading took ([0-9.]+) GiB memory"),
    "graph_capture_memory_gib": re.compile(r"Graph capturing finished in [0-9.]+ secs, took ([0-9.]+) GiB"),
    "kv_reserved_memory_gib": re.compile(r"reserved ([0-9.]+) GiB memory for KV Cache"),
    "kv_cache_tokens": re.compile(r"GPU KV cache size: ([0-9,]+) tokens"),
}
BOUNDARIES = [
    "allocated_bytes is the recorded current-process torch.cuda.memory_allocated() snapshot, not NVML device-resident memory, allocator-reserved memory, a peak, or full resident model weights.",
    "A confirmed snapshot follows engine construction and runtime attachment, before application workload warmup. Engine-internal loading, profiling and graph capture may already have occurred.",
    "kv_cache_memory_bytes is the explicit engine configuration cap. It is not a second independent resident-memory measurement and must not be added to the allocated snapshot.",
    "Model-loading and graph-capture values retain the logger's rounded GiB units and measurement scope. They are not interchangeable with the allocated snapshot; their sums or differences do not partition memory.",
    "Every matching loading/capture report is preserved with context. Multiple loading reports are not summed or assigned to target/draft models without evidence; the single-report scalar is null when ambiguous.",
    "Missing log reports are null, including graph memory for eager engines; absence of a report is not a measured zero.",
    "compiled means enforce_eager=false. Startup graph modes do not establish the fraction of measured requests executed in each graph mode.",
]


def read(path: Path) -> dict:
    return json.loads(path.read_text()) if path.is_file() else {}


def sha(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def single(reports: list[dict], metric: str):
    values = [r["value"] for r in reports if r["metric"] == metric]
    return values[0] if len(values) == 1 else None


def log_reports(path: Path, engine: Path) -> tuple[list[dict], list[str]]:
    if not path.is_file():
        return [], []
    # Split on newline only to preserve source line numbers despite progress CRs.
    lines = path.read_text(errors="replace").split("\n")
    reports = []
    for index, original in enumerate(lines):
        line = ANSI.sub("", original)
        for metric, pattern in PATTERNS.items():
            for match in pattern.finditer(line):
                value = match.group(1).replace(",", "")
                reports.append({"engine": str(engine), "log": str(path),
                    "log_sha256": sha(path), "line": index + 1, "metric": metric,
                    "value": int(value) if metric == "kv_cache_tokens" else float(value),
                    "model_attribution": None,
                    "context": [{"line": j + 1, "text": lines[j]}
                                for j in range(max(0, index - 2), min(len(lines), index + 2))]})
    modes = sorted(set(re.findall(r"Capturing CUDA graphs \(([^)]+)\)", "\n".join(lines))))
    return reports, modes


def snapshot_proof(campaign: Path, env: dict) -> dict:
    source = campaign / "source/run_matrix.py"
    result = {"snapshot_source": str(source) if source.is_file() else None,
              "snapshot_source_sha256": sha(source), "snapshot_source_line": None,
              "snapshot_boundary_confirmed": False}
    expected = env.get("source_sha256", {}).get("run_matrix.py")
    if not expected or result["snapshot_source_sha256"] != expected:
        return result
    text = source.read_text()
    allocation = re.search(r"['\"]allocated_bytes['\"]\s*:\s*torch\.cuda\.memory_allocated\(\)", text)
    if allocation is None:
        return result
    result["snapshot_source_line"] = text[:allocation.start()].count("\n") + 1
    construct = text.find("engine = LLMEngine.from_engine_args(")
    attach = text.find("runtime.attach_engine(engine)")
    environment = text.find("dump(out/'environment.json'")
    episode = text.find("def episode(")
    result["snapshot_boundary_confirmed"] = (0 <= construct < attach < environment
                                              < allocation.start() < episode)
    return result


def engine_row(engine: Path) -> tuple[dict, list[dict]]:
    campaign = engine.parent
    envpath, argpath, cfgpath = [engine / name for name in
                              ("environment.json", "engine_args.json", "configuration.json")]
    env, kwargs, cfg = map(read, (envpath, argpath, cfgpath))
    args, plan = cfg.get("args", {}), cfg.get("plan", {})
    log = campaign / (engine.name + ".log")
    reports, modes = log_reports(log, engine)
    spec = kwargs.get("speculative_config")
    allocated = env.get("allocated_bytes")
    kv = kwargs.get("kv_cache_memory_bytes")
    eager = kwargs.get("enforce_eager")
    row = {"campaign": campaign.name, "engine": engine.name,
        "campaign_status": read(campaign / "status.json").get("status"),
        "engine_status": read(engine / "status.json").get("status"),
        "model": kwargs.get("model"), "model_revision": plan.get("model_revision"),
        "dtype": kwargs.get("dtype"), "quantization_argument": kwargs.get("quantization"),
        "draft_source_plan": plan.get("draft_source"),
        "draft_source_engine": spec.get("method") if isinstance(spec, dict) else ("none" if kwargs else None),
        "draft_model_engine": spec.get("model") if isinstance(spec, dict) else None,
        "proposal_budget_engine": spec.get("num_speculative_tokens") if isinstance(spec, dict) else (0 if kwargs else None),
        "verification_cap_argument": args.get("gamma"), "policy_mode": args.get("mode"),
        "execution_mode": "eager" if eager is True else "compiled" if eager is False else None,
        "enforce_eager": eager, "allocated_bytes": allocated,
        "allocated_gib": allocated / GIB if allocated is not None else None,
        "kv_cache_memory_bytes_config": kv, "kv_cache_cap_gib": kv / GIB if kv is not None else None,
        "max_num_seqs": kwargs.get("max_num_seqs"), "max_model_len": kwargs.get("max_model_len"),
        "kernel_config": kwargs.get("kernel_config"), "gpu": env.get("gpu"),
        "vllm": env.get("vllm"), "torch": env.get("torch"), "cuda": env.get("cuda"),
        "model_runner_version": env.get("model_runner_version"),
        "log_model_loading_memory_gib_single_report": single(reports, "model_loading_memory_gib"),
        "log_model_loading_report_count": sum(r["metric"] == "model_loading_memory_gib" for r in reports),
        "log_graph_capture_memory_gib_single_report": single(reports, "graph_capture_memory_gib"),
        "log_graph_capture_report_count": sum(r["metric"] == "graph_capture_memory_gib" for r in reports),
        "log_kv_reserved_memory_gib_single_report": single(reports, "kv_reserved_memory_gib"),
        "log_kv_cache_tokens_single_report": single(reports, "kv_cache_tokens"),
        "log_capture_modes": modes,
        "environment_path": str(envpath), "environment_sha256": sha(envpath),
        "engine_args_path": str(argpath), "engine_args_sha256": sha(argpath),
        "configuration_path": str(cfgpath), "configuration_sha256": sha(cfgpath),
        "log_path": str(log) if log.is_file() else None, "log_sha256": sha(log),
        **snapshot_proof(campaign, env)}
    row["missing_observations"] = [key for key in (
        "allocated_bytes", "kv_cache_memory_bytes_config", "log_model_loading_memory_gib_single_report",
        "log_graph_capture_memory_gib_single_report") if row[key] is None]
    return row, reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Saved campaign or engine directories")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    engines = set()
    for path in args.inputs:
        path = path.resolve()
        if (path / "environment.json").is_file() or (path / "engine_args.json").is_file():
            engines.add(path)
        else:
            engines.update(p.parent for p in path.glob("*/engine_args.json"))
            engines.update(p.parent for p in path.glob("*/environment.json"))
    if not engines:
        parser.error("No saved engines found; no measurements are inferred.")
    rows, reports = [], []
    for engine in sorted(engines):
        row, evidence = engine_row(engine)
        rows.append(row)
        reports.extend(evidence)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "engine_resources.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v
                         for k, v in row.items()} for row in rows)
    (args.output / "log_evidence.jsonl").write_text("".join(json.dumps(r) + "\n" for r in reports))
    campaigns = []
    for name in sorted({r["campaign"] for r in rows}):
        group = [r for r in rows if r["campaign"] == name]
        def values(key):
            return sorted({r[key] for r in group if r[key] is not None})
        campaigns.append({"campaign": name, "engine_count": len(group),
            "engine_statuses": values("engine_status"), "execution_modes": values("execution_mode"),
            "snapshot_boundary_confirmed_count": sum(r["snapshot_boundary_confirmed"] for r in group),
            "allocated_bytes_values": values("allocated_bytes"), "allocated_gib_values": values("allocated_gib"),
            "kv_cap_gib_values": values("kv_cache_cap_gib"),
            "model_loading_report_gib_values": values("log_model_loading_memory_gib_single_report"),
            "graph_capture_report_gib_by_engine": {r["engine"]: r["log_graph_capture_memory_gib_single_report"] for r in group},
            "proposal_budget_by_engine": {r["engine"]: r["proposal_budget_engine"] for r in group}})
    summary = {"status": "saved_observations_summarized", "engine_count": len(rows),
        "inputs": [str(p.resolve()) for p in args.inputs], "campaigns": campaigns,
        "log_evidence_records": len(reports), "boundaries": BOUNDARIES,
        "unmeasured_fields": ["NVML resident bytes", "peak allocated bytes", "full resident weight bytes"],
        "reader_sha256": sha(Path(__file__)),
        "outputs": ["engine_resources.csv", "log_evidence.jsonl"]}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"engine_count": len(rows), "campaigns": campaigns}, indent=2))


if __name__ == "__main__":
    main()
