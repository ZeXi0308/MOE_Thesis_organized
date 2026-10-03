"""Locate request-gap regressions in the completed saving ABBA; no policy replay."""
import argparse
from collections import Counter
import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "execution_weste_26862/readback/results"


def read_cell(name):
    folder = RESULTS / name
    raw = json.loads((folder / "raw.json").read_text())
    status = json.loads((folder / "status.json").read_text())
    if status["status"] != "COMPLETE" or raw["status"] != "COMPLETE":
        raise ValueError(f"{name}: this completed-group diagnostic cannot compare an incomplete cell")
    # Native diagnostic events share the exact timestamp captured after engine.step.
    # The lightweight recorder instead writes the call index directly.
    calls = {}
    for call in raw.get("engine_steps", []):
        if call["completed"]:
            if call["returned_s"] in calls:
                raise ValueError("Ambiguous diagnostic engine-return timestamp")
            calls[call["returned_s"]] = call
    events = {r["request_id"]: [] for r in raw["requests"]}
    for event in raw["output_events"]:
        if event["prefix_valid"] and event["chunk_size"]:
            if event["chunk_size"] != 1:
                raise ValueError("Unresolved intra-chunk intervals require a different analysis")
            if "engine_call_index" not in event:
                call = calls[event["received_s"]]
                if event["external_request_id"] not in call["output_request_ids"]:
                    raise ValueError("Diagnostic output not returned by the matched call")
                event = dict(event, engine_call_index=call["call_index"])
            events[event["request_id"]].append(event)
    rows = {}
    for r in raw["requests"]:
        es = events[r["request_id"]]
        if len(es) != len(r["token_times_s"]) or len(es) != 1024:
            raise ValueError("Planned output and event identity mismatch")
        if ([e["received_s"] for e in es] != r["token_times_s"] or
                [e["cumulative_tokens"] for e in es] != list(range(1, 1025))):
            raise ValueError("Request token times or output counts disagree with events")
        a, b = max(zip(es, es[1:]), key=lambda pair: pair[1]["received_s"] - pair[0]["received_s"])
        rows[r["request_id"]] = dict(
            completion_latency_s=r["completion_s"] - r["arrival_s"],
            gap_s=b["received_s"] - a["received_s"],
            before_call=a["engine_call_index"], after_call=b["engine_call_index"],
            call_span=b["engine_call_index"] - a["engine_call_index"],
            before_output_count=a["cumulative_tokens"], after_output_count=b["cumulative_tokens"])
    signatures = {rid: [(e["engine_call_index"], e["cumulative_tokens"], e["new_token_ids"]) for e in es]
                  for rid, es in events.items()}
    return raw, rows, signatures


def analyze():
    diagnostics = {arm: read_cell("diag-" + arm) for arm in ("off", "on")}
    result = dict(sources=[], pairs=[], timing="Host receipt of actual new output; not client delivery",
        scope="Per-request outcomes and observed output paths. Diagnostic correspondence is not proof of hidden KV equivalence or a same-state causal fork.")
    for block in (0, 1):
        cells = {arm: read_cell(f"block{block}-{arm}") for arm in ("off", "on")}
        if set(cells["off"][1]) != set(cells["on"][1]):
            raise ValueError("Request identities differ")
        matches = {arm: {rid: sig == diagnostics[arm][2][rid] for rid, sig in cells[arm][2].items()}
                   for arm in ("off", "on")}
        rows = []
        for rid, a in cells["off"][1].items():
            b = cells["on"][1][rid]
            identity_keys = ("before_call", "after_call", "before_output_count", "after_output_count")
            kind = ("consecutive_calls" if a["call_span"] == b["call_span"] == 1 else
                    "same_output_path_interval" if all(a[k] == b[k] for k in identity_keys) else
                    "changed_output_path_interval")
            row = dict(request_id=rid, off=a, on=b, observed_kind=kind,
                gap_on_minus_off_s=b["gap_s"] - a["gap_s"],
                completion_on_minus_off_s=b["completion_latency_s"] - a["completion_latency_s"],
                diagnostic_output_path_matches={arm: matches[arm][rid] for arm in ("off", "on")})
            if row["gap_on_minus_off_s"] > 0:
                row["diagnostic_preemptions"] = {}
                for arm in ("off", "on"):
                    raw = diagnostics[arm][0]
                    row["diagnostic_preemptions"][arm] = [dict(step=p["attempted_step"],
                        outputs=p["victim_state"]["output_tokens"], computed=p["victim_state"]["computed_tokens"])
                        for p in raw["preemption_events"]
                        if raw["internal_to_source"][p["victim_internal_request_id"]] == rid] if matches[arm][rid] else None
            rows.append(row)
        result["pairs"].append(dict(block=block, requests=rows,
            worsened_by_kind=dict(Counter(r["observed_kind"] for r in rows if r["gap_on_minus_off_s"] > 0)),
            diagnostic_matching_request_counts={arm: sum(x.values()) for arm, x in matches.items()}))
    result["sources"] = [str((RESULTS / name / "raw.json").relative_to(Path(__file__).resolve().parents[6]))
                         for name in ("diag-off", "diag-on", "block0-off", "block0-on", "block1-on", "block1-off")]
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze()
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps([{k: v for k, v in p.items() if k != "requests"} for p in result["pairs"]]))
