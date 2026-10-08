#!/usr/bin/env python3
"""Complete-cohort outcomes for arbitrary native A cells; no action benefit inferred.

Reuse the established clock/goodput definitions. Raw lease qualification is a
separate report: this entry reports every arm including failed or partial ones.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from evaluate_goodput import summarize, pair


def read_cell(directory, expected_requests, timeout_s):
    archive = directory / "archive"
    if not archive.exists():
        archive = directory / "output"
    if not (archive / "raw.json").exists():
        return dict(status="NO_MEASUREMENT", directory=str(directory)), None
    raw = json.loads((archive / "raw.json").read_text())
    status = json.loads((archive / "status.json").read_text())
    config = json.loads((archive / "config.json").read_text())
    store = json.loads((archive / "selective-store.json").read_text()) if (archive / "selective-store.json").exists() else {}
    # Refuse to silently turn a partial submission into a smaller cohort.
    if len(raw.get("requests", [])) != expected_requests:
        return dict(status="INCOMPLETE_COHORT", observed_requests=len(raw.get("requests", [])),
                    expected_requests=expected_requests, capture_status=raw.get("status"),
                    engine_status=status, directory=str(directory)), raw
    metrics = summarize(raw, expected_requests, timeout_s)
    complete = (raw.get("status") == status.get("status") == "COMPLETE"
                and raw.get("error") is None and status.get("error") is None
                and metrics["completed"] == expected_requests
                and metrics["failed"] == metrics["unfinished"] == 0)
    events = store.get("events", [])
    return dict(status="COMPLETE" if complete else "INCOMPLETE", metrics=metrics,
                config=config, engine_status=status,
                raw_sha256=hashlib.sha256((archive / "raw.json").read_bytes()).hexdigest(),
                policy_summary={k: v for k, v in store.items()
                                if "lease" in k or k in ("status", "store_scope", "applied_rotations", "oldest_episode_count")},
                event_counts=dict(Counter(e.get("event", "UNKNOWN") for e in events)),
                actual_preemption_events=sum(e.get("original_preemption_called") is True
                    and e.get("original_preemption_returned") is True for e in raw.get("preemption_events", [])),
                host_transfer_accounting="Not inferred from overlapping elapsed intervals",
                directory=str(directory)), raw


def analyze(session, expected_requests, timeout_s, reference):
    rows = {}; raws = {}
    for directory in sorted(session.glob("cell-[0-9][0-9]-*")):
        name = directory.name[8:]
        if name in rows:
            raise ValueError("Duplicate arm identity")
        rows[name], raws[name] = read_cell(directory, expected_requests, timeout_s)
    if not rows:
        raise ValueError("No cells found")
    comparisons = {}
    if reference not in rows:
        raise ValueError("Reference arm not present")
    for name, row in rows.items():
        if name == reference or "metrics" not in row or "metrics" not in rows[reference]:
            continue
        a, b = rows[reference]["metrics"], row["metrics"]
        comparison = pair(a, b)
        old = {r["request_id"]: r for r in raws[reference]["requests"]}
        new = {r["request_id"]: r for r in raws[name]["requests"]}
        comparison["sequence_differences"] = [rid for rid in sorted(old)
            if old[rid]["output_token_ids"] != new[rid]["output_token_ids"]]
        comparison["stop_differences"] = [rid for rid in sorted(old)
            if old[rid].get("stop_reason") != new[rid].get("stop_reason")]
        comparison["exploratory_tradeoff"] = dict(
            both_complete=rows[reference]["status"] == row["status"] == "COMPLETE",
            p95_gap_lower=(b["max_gap_request_p95_s"] is not None
                and a["max_gap_request_p95_s"] is not None
                and b["max_gap_request_p95_s"] < a["max_gap_request_p95_s"]),
            rate_at_least_97pct=b["actual_output_tokens_s"] >= .97 * a["actual_output_tokens_s"],
            mean_flow_at_most_105pct=b["mean_flow_with_incomplete_penalty_s"] <=
                1.05 * a["mean_flow_with_incomplete_penalty_s"])
        comparisons[name + "_vs_" + reference] = comparison
    return dict(status="SERVICE_OUTCOMES_ONLY", session=str(session), arms=rows, comparisons=comparisons,
        scope="Independent policy trajectories on same inputs; no equal-work, causal-action, novelty or stability inference. Lease events require raw joins separately.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--expected-requests", type=int, required=True)
    parser.add_argument("--timeout-s", type=float, default=180)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.session, args.expected_requests, args.timeout_s, args.reference)
    with args.output.open("x") as output:
        json.dump(result, output, indent=2, ensure_ascii=False, allow_nan=False)
        output.write("\n")
