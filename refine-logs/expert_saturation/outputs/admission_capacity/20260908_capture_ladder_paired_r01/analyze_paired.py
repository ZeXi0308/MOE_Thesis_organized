#!/usr/bin/env python3
"""Recompute frozen paired-ladder cells; retain missing and invalid attempts."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "experiments/admission_capacity"))
from metrics import summarize_episode_requests


def read(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def arm(plan):
    return f"{plan['policy']}:{plan['cap']}:{plan['ladder_name']}"


def inspect_cell(directory, index, plan, config, workload):
    path = directory / f"cell-{index:03d}.json"
    row = dict(index=index, plan=plan, path=str(path), status="UNRUN", eligible=False, errors=[])
    if not path.exists():
        return row
    try:
        raw = read(path)
        row.update(raw_status=raw.get("status"), raw_error=raw.get("error"), status="INVALID")
        # Retain recoverable metrics even when identity or post-cell checks fail.
        try:
            row["metrics"] = summarize_episode_requests(raw["requests"],
                observation_end_s=raw["observation_end_s"],
                ttft_slo_s=config["ttft_slo_s"], tpot_slo_s=config["tpot_slo_s"])
            m = row["metrics"]
            m["completion_fraction_planned"] = m["n_completed"] / m["n_planned"]
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
            row["errors"].append(f"metrics: {exc}")
        require(raw["plan"] == plan, "raw plan differs from frozen plan")
        for key, raw_key in (("regime", "regime"), ("cap", "target_cap"),
                             ("policy", "policy"), ("arrival_scale", "arrival_scale")):
            require(raw[raw_key] == plan[key], f"raw {raw_key} differs from plan")
        require(read(directory / "config.json") == config, "run config differs from frozen config")
        require(read(directory / "workload.json") == workload, "run workload differs from frozen workload")
        require(read(directory / f"checks-{index:03d}.json")["status"] == "PASS", "post-cell check failed")
        sources = workload["source_requests"]
        arrivals = workload["arrival_traces_s"][plan["regime"]]
        expected = {s["request_id"]: (s, a * plan["arrival_scale"])
                    for s, a in zip(sources, arrivals)}
        requests = raw["requests"]
        require(len(requests) == len(expected), "request population differs")
        require({r["request_id"] for r in requests} == set(expected), "request identities differ")
        for request in requests:
            source, arrival = expected[request["request_id"]]
            require(all(request[k] == source[k] for k in ("document_id", "prompt_token_ids_sha256")),
                    "document or prompt hash differs")
            require(request["arrival_s"] == arrival and request["prompt_tokens"] == config["prompt_tokens"],
                    "arrival or prompt length differs")
            if request["status"] == "completed":
                require(len(request["output_token_ids"]) == config["output_tokens"], "completed output length differs")
        require(raw["host_chunk_diagnostics"]["token_level_itl_resolved"] is True, "unresolved host chunks")
        require(bool(raw["output_events"]) and all(e["chunk_size"] in (0, 1) for e in raw["output_events"]),
                "host events absent or contain unresolved chunks")
        steps = raw["scheduler_steps"]
        require(bool(steps), "scheduler trace absent")
        row["exposure"] = dict(max_actual_active=max(s["actual_active"] for s in steps),
            max_decode_requests=max(s["decode_requests"] for s in steps),
            max_waiting_requests=max(s["waiting_requests"] for s in steps),
            steps_with_waiting=sum(s["waiting_requests"] > 0 for s in steps),
            n_scheduler_steps=len(steps), queue_fraction_semantics="step snapshots, not wall-time fraction")
        require(not row["errors"], "metric reconstruction failed")
        complete = raw["status"] == "COMPLETE" and all(r["status"] == "completed" for r in requests)
        row.update(status="COMPLETE" if complete else "INCOMPLETE", eligible=complete)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        row["errors"].append(f"{type(exc).__name__}: {exc}")
        row["status"] = "INVALID"
    return row


def compare(rows, engine, regime):
    group = [r for r in rows if r["plan"]["regime"] == regime]
    expected = {f"static:{c}:static" for c in (8, 12, 16, 24, 32)} | {
        "shadow:32:legacy", "feedback:32:legacy", "feedback:32:aligned"}
    valid = {arm(r["plan"]): r for r in group if r["eligible"]}
    missing = sorted(expected - valid.keys())
    result = dict(engine=engine, regime=regime, status="INCOMPLETE", missing_or_ineligible_arms=missing)
    if missing or len(group) != len(expected) or {arm(r["plan"]) for r in group} != expected:
        return result
    statics = [valid[f"static:{c}:static"] for c in (8, 12, 16, 24, 32)]
    best = max(r["metrics"]["goodput_rps"] for r in statics)
    result.update(status="COMPLETE", best_static_goodput_rps=best,
        best_static_caps=[r["plan"]["cap"] for r in statics if r["metrics"]["goodput_rps"] == best],
        baseline_scope="within-engine exploratory hindsight best static; not an action Oracle")
    old, new = (valid[f"feedback:32:{name}"]["metrics"] for name in ("legacy", "aligned"))
    result["arms"] = {key: dict(goodput_rps=r["metrics"]["goodput_rps"],
        completion_fraction_planned=r["metrics"]["completion_fraction_planned"],
        n_slo_pass=r["metrics"]["n_slo_pass"], latency_s=r["metrics"]["latency_s"])
        for key, r in sorted(valid.items())}
    result.update(aligned_minus_legacy_goodput_rps=new["goodput_rps"] - old["goodput_rps"],
        aligned_relative_legacy_goodput=None if old["goodput_rps"] == 0 else new["goodput_rps"] / old["goodput_rps"] - 1,
        aligned_minus_best_static_goodput_rps=new["goodput_rps"] - best,
        aligned_minus_legacy_median_tpot_s=new["latency_s"]["tpot"]["p50"] - old["latency_s"]["tpot"]["p50"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("results-dir", "plans-dir", "output-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output_dir.exists(), "output directory must be new")
    rows, comparisons = [], []
    for engine in ("forward", "reverse"):
        folder = args.plans_dir / engine
        config = read(folder / "config.json")
        workload = read((folder if (folder / "workload.json").exists() else args.plans_dir) / "workload.json")
        digest = hashlib.sha256(json.dumps(workload, sort_keys=True).encode()).hexdigest()
        require(digest == config["workload_sha256"], "frozen workload hash differs")
        sources, prompts = workload["source_requests"], workload["actual_prompt_token_ids"]
        require(len(sources) == len(prompts) == config["requests"] and len({s["request_id"] for s in sources}) == len(sources),
                "frozen workload population differs")
        for source, tokens in zip(sources, prompts):
            require(len(tokens) == config["prompt_tokens"] and hashlib.sha256(json.dumps(tokens,
                separators=(",", ":")).encode()).hexdigest() == source["prompt_token_ids_sha256"], "frozen prompt differs")
        for regime in ("steady", "bursty"):
            require(len(workload["arrival_traces_s"][regime]) == len(sources), "frozen arrivals differ")
        engine_rows = [dict(inspect_cell(args.results_dir / engine, i, p, config, workload), engine=engine)
                       for i, p in enumerate(config["plans"])]
        rows.extend(engine_rows)
        comparisons.extend(compare(engine_rows, engine, regime) for regime in ("steady", "bursty"))
    unrun = all(r["status"] == "UNRUN" for r in rows)
    report = dict(status="UNRUN" if unrun else ("MEASUREMENT_ONLY" if all(c["status"] == "COMPLETE" for c in comparisons) else "INCOMPLETE"),
        evidence_type="NO_REQUEST_MEASUREMENTS" if unrun else "NATIVE_VLLM_INPROCESS_REQUEST_MEASUREMENT", cells=rows, comparisons=comparisons,
        limits=["Incomplete or invalid cells retain metrics but never enter comparisons.",
                "Forward is canonical; reverse is the controlled repeat, with neither selected by results.",
                "Finite repeated cohort, host delivery timing; no kernel/padding inference, Oracle, or method GO."])
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "analysis.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(status=report["status"], cells=len(rows), eligible=sum(r["eligible"] for r in rows))))


if __name__ == "__main__":
    main()
