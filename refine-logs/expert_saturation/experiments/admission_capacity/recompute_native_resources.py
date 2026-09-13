#!/usr/bin/env python3
"""Recompute the retained four-cell native32/safe29 campaign from raw ledgers.

Uses the campaign's existing identity, request-metric and executed-work APIs.
Supports raw.json or raw.json.gz; missing/inconsistent evidence fails explicitly.
This is an analysis of completed executions, never an action counterfactual.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def unique_pairs(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def reject_constant(value):
    raise ValueError(f"non-finite JSON constant: {value}")


def read(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return json.load(stream, object_pairs_hook=unique_pairs, parse_constant=reject_constant)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_api(campaign):
    path = campaign / "analyze_native_preemption.py"
    sys.path.insert(0, str(campaign))
    spec = importlib.util.spec_from_file_location("retained_native_preemption_analysis", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load campaign API: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def recompute_cell(campaign, label, api):
    directory = campaign / "gpu_results" / label
    candidates = [directory / name for name in ("raw.json", "raw.json.gz")]
    paths = [path for path in candidates if path.is_file()]
    require(bool(paths), f"{label}: missing raw.json and raw.json.gz")
    if len(paths) == 2:
        with paths[0].open("rb") as plain, gzip.open(paths[1], "rb") as compressed:
            require(hashlib.file_digest(plain, "sha256").digest() == hashlib.file_digest(compressed, "sha256").digest(),
                    f"{label}: plain and compressed raw ledgers differ")
    raw, config = read(paths[0]), read(directory / "config.json")
    terminal, qualification = read(directory / "status.json"), read(directory / "safe-cap-qualification.json")
    arm = label.split("-", 1)[1]
    cap = 29 if arm == "safe" else 32
    require(config["requested_arm"] == arm and config["cap"] == cap, f"{label}: arm/cap mismatch")
    require(qualification["status"] == "QUALIFIED" and qualification["safe_cap"] == 29,
            f"{label}: safe29 layout was not qualified")
    reserved = (config["prompt_tokens"] + config["output_tokens"] + qualification["block_size"] - 1) // qualification["block_size"]
    require(reserved == qualification["per_request_reserved_blocks"]
            and min(32, qualification["usable_blocks"] // reserved) == 29,
            f"{label}: safe-cap arithmetic mismatch")
    mode = "stop_before_preemption" if arm == "safe" else "native_recompute"
    require(raw["preemption_mode"] == config["preemption_mode"] == mode, f"{label}: preemption mode mismatch")
    identity = api.base.validate_identity(raw, config, campaign / "inputs_preparation" / "prepared", "long", cap)
    require(raw["status"] == terminal["status"] == "COMPLETE" and identity["all_requests_completed"],
            f"{label}: incomplete execution cannot be compared as complete throughput")
    require(raw["capacity_boundary"] is None, f"{label}: capacity-stopped episode")
    metrics = api.base.summarize_episode_requests(raw["requests"], observation_end_s=raw["observation_end_s"],
                                                ttft_slo_s=5.0, tpot_slo_s=0.2)
    require(metrics["n_completed"] == config["requests"] == 32, f"{label}: incomplete request denominator")
    saved = read(directory / "metrics.json")
    require(all(saved[key] == value for key, value in metrics.items()), f"{label}: saved metric mismatch")
    work, calls = api.execution_accounting(raw)
    events = api.preemptions(raw, calls)
    effects = api.request_effects(raw, events, work)
    require(arm != "safe" or raw["actual_preemption_count"] == 0, f"{label}: safe arm preempted")
    phases = []
    for victim in effects["victim_requests"]:
        gap = victim["longest_itl"]
        own = [item for item in work["recomputed_intervals"]
               if item["request_id"] == victim["request_id"]
               and gap["start_s"] <= item["call_start_s"]
               and item["call_returned_s"] <= gap["end_s"]]
        require(bool(own), f"{label}: longest victim gap does not contain its recomputation")
        first, last = min(item["call_start_s"] for item in own), max(item["call_returned_s"] for item in own)
        require(gap["start_s"] <= first <= last <= gap["end_s"], f"{label}: invalid recovery timeline")
        phases.append(dict(request_id=victim["request_id"], gap_start_s=gap["start_s"], gap_end_s=gap["end_s"],
                           gap_s=gap["itl_s"], before_first_recompute_call_s=first-gap["start_s"],
                           recompute_calls_span_s=last-first, after_last_recompute_call_s=gap["end_s"]-last,
                           recompute_step_ids=[item["step"] for item in own]))
    return dict(label=label, status=raw["status"], identity_checked_requests=identity["checked_requests"],
                workload_sha256=identity["workload_sha256"], raw_path=str(paths[0].relative_to(campaign)),
                saved_metrics_exact_match=True, n_completed=metrics["n_completed"],
                duration_s=metrics["observation_duration_s"], throughput_rps=metrics["throughput_rps"],
                ttft_s=metrics["latency_s"]["ttft"], tpot_s=metrics["latency_s"]["tpot"],
                pooled_itl_s=metrics["latency_s"]["itl"], request_max_itl_s=effects["request_max_itl_s"],
                max_itl_s=effects["longest_itl_requests"][0]["longest_itl"]["itl_s"],
                completion_latency_s=effects["completion_latency_s"],
                actual_preemptions=raw["actual_preemption_count"], work=work["totals"],
                longest_gap_before_recompute_call_s=max((p["before_first_recompute_call_s"] for p in phases), default=None),
                recovery_phases=phases)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "output exists; refusing to overwrite")
    campaign = args.campaign.resolve()
    api = load_api(campaign)
    rows = [recompute_cell(campaign, label, api) for label in api.LABELS]
    require(len(rows) == 4, "expected the complete four-cell frozen campaign")
    comparisons = []
    for repeat in (0, 1):
        native = next(row for row in rows if row["label"] == f"repeat{repeat}-native32")
        safe = next(row for row in rows if row["label"] == f"repeat{repeat}-safe")
        comparisons.append(dict(repeat=repeat, native_vs_safe_throughput_relative=native["throughput_rps"] / safe["throughput_rps"] - 1,
                                native_minus_safe_ttft_p99_s=native["ttft_s"]["p99"] - safe["ttft_s"]["p99"]))
    result = dict(status="MEASUREMENT_ONLY", source_campaign=campaign.name, new_gpu_executions=0,
                  cells=rows, comparisons=comparisons,
                  scope="Full host request denominator includes waiting, recomputation and instrumentation. Recovery phases run from the preceding token receipt to the next receipt; before-first-recompute is a host interval, not isolated scheduler waiting. Recompute-call span includes concurrent work and instrumentation, not pure GPU time. Do not sum overlapping victims or add phases again to request latency. Same 32 documents repeated, not independent population tail estimates. No counterfactual action result.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(status=result["status"], cells=len(rows), completed=sum(row["n_completed"] for row in rows), comparisons=comparisons), indent=2))


if __name__ == "__main__":
    main()
