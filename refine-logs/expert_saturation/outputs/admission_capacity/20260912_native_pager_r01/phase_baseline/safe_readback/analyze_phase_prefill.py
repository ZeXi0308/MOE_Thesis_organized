"""Read actual phase-baseline cells remotely; export metrics without prompt data."""
import argparse
import hashlib
import json
from pathlib import Path

from analyze_prefill_midpoint import prestate
from analyze_wisp_injection import comparison, hardware_summary, summarize


def decision_checks(raw):
    issues = []
    action = raw["action"]["engine_call"]
    for step in raw["steps"]:
        decision = step.get("phase_prefill", {})
        ready = sorted(rid for rid in step["before"]["running"]
            if any(t < step["start_s"] for t in raw["requests"][rid]["token_received_s"])
            and raw["requests"][rid]["completion_s"] >= step["start_s"])
        enabled = step["index"] >= action
        expected = 32 if not enabled else (8 if raw["args"]["policy"] == "static8" or ready else 0)
        actual_rows = step["scheduled"]
        if (sorted(decision.get("ready_decode_ids", [])) != ready
                or decision.get("enabled") != enabled
                or decision.get("chosen_threshold") != expected
                or decision.get("status") != "applied"
                or decision.get("actual_prefill_rows") != sum(r["prefill_tokens"] for r in actual_rows)
                or decision.get("actual_decode_rows") != sum(r["decode_tokens"] for r in actual_rows)
                or decision.get("actual_scheduled") != actual_rows):
            issues.append(step["index"])
        for rid in ready:
            matches = [r for r in actual_rows if r["request_id"] == rid]
            if len(matches) != 1 or matches[0]["tokens"] != 1 or matches[0]["decode_tokens"] != 1:
                issues.append(step["index"])
    return sorted(set(issues))


def evaluate(base, config):
    execution = json.loads((base / "execution.json").read_text())
    states = {row["cell"]: row for row in execution["cells"]}
    raw, summaries = {}, {}
    result = dict(status="MEASUREMENT_ONLY", execution=execution, cells={}, comparisons=[],
        scope="Exploratory same-document phase baseline, not independent-document or online-model evidence.")
    for cell in config["cells"]:
        name = cell["cell"]
        path = base / name / "result.json"
        if not path.exists():
            result["cells"][name] = dict(status="UNRUN_OR_MISSING")
            result["status"] = "INCOMPLETE"
            continue
        data = path.read_bytes()
        raw[name] = value = json.loads(data)
        entry = result["cells"][name] = dict(status=value.get("status"),
            raw_sha256=hashlib.sha256(data).hexdigest())
        if value.get("status") != "COMPLETED":
            result["status"] = "INCOMPLETE"
            entry["error"] = value.get("error")
            continue
        summaries[name] = summary = summarize(value, states.get(name, {}))
        issues = decision_checks(value)
        entry.update(measured=summary, decision_issues=issues,
            prestate_sha256=prestate(value),
            decisions=[dict(engine_call=s["index"], **s["phase_prefill"]) for s in value["steps"]],
            whole_episode_copy_payload_bytes=sum(layer["weight_copy_payload_bytes"]
                for worker in value["final_worker"] for layer in worker["pager_layers"]),
            output_sha256={rid:hashlib.sha256(json.dumps(row["output_token_ids"]).encode()).hexdigest()
                           for rid, row in value["requests"].items()},
            hardware=hardware_summary(base / name / "hardware.jsonl", value, list(range(8))))
        if (issues or entry["prestate_sha256"] != config["preaction_sha256"]
                or not summary["complete_expected_requests"] or not summary["identity_valid"]
                or not summary["first_shape_valid"] or summary["scheduler_violations"]
                or not all(r["timing_valid"] for r in summary["requests"].values())):
            entry["status"] = result["status"] = "INVALID_COMPARISON"
    repeats = sorted({c["repeat"] for c in config["cells"]})
    for repeat in repeats:
        names = {c["policy"]: c["cell"] for c in config["cells"] if c["repeat"] == repeat}
        a, b = names["static8"], names["phase8"]
        if a not in summaries or b not in summaries:
            continue
        pair = comparison(raw[a], raw[b], summaries[a], summaries[b], "same-task phase rule")
        pair.update(static=a, phase=b, repeat=repeat)
        result["comparisons"].append(pair)
        if not all(pair[k] for k in ("resources_same", "same_input_tasks", "old_tokens_same",
                                     "scheduler_prestate_same", "full_pager_state_same")):
            result["status"] = "INVALID_COMPARISON"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("results", "config", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    result = evaluate(args.results, json.loads(args.config.read_text()))
    with args.out.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
