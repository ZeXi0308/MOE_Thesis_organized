"""Safe numeric readback for the map-write-only same-engine comparison."""
import argparse
import hashlib
import json
from pathlib import Path

from analyze_phase_prefill import decision_checks
from analyze_prefill_midpoint import prestate
from analyze_wisp_injection import comparison, hardware_summary, summarize


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def evaluate(base):
    config = json.loads((base / "config.json").read_text())
    execution = json.loads((base / "execution.json").read_text())
    out = dict(status="MEASUREMENT_ONLY", execution=execution, cells={}, comparisons=[],
        scope="Same-document static8, map-only engineering intervention; no profiler or SLO claim")
    raws, summaries = {}, {}
    for row in execution["cells"]:
        name = row["cell"]
        path = base/name/"result.json"
        data = path.read_bytes()
        raw = raws[name] = json.loads(data)
        cell = out["cells"][name] = dict(status=raw["status"], raw_sha256=hashlib.sha256(data).hexdigest())
        if raw["status"] != "COMPLETED":
            cell["error"] = raw.get("error")
            out["status"] = "INCOMPLETE"
            continue
        summary = summaries[name] = summarize(raw, row)
        cell.update(measured=summary, prestate_sha256=prestate(raw), decision_issues=decision_checks(raw),
            map_initial=raw["map_initial"], map_measurement=raw["map_measurement"],
            map_validation=raw["map_validation"],
            whole_episode_copy_payload_bytes=sum(s["weight_copy_payload_bytes"]
                for w in raw["final_worker"] for s in w["pager_layers"]),
            output_sha256={rid:digest(r["output_token_ids"]) for rid,r in raw["requests"].items()},
            final_pager_sha256=digest([w["pager_execution_state"] for w in raw["final_worker"]]),
            schedule_sha256=digest([s["scheduled"] for s in raw["steps"]]),
            hardware=hardware_summary(base/"hardware.jsonl",raw,list(range(8))))
        valid = (cell["prestate_sha256"] == execution["reference_prestate_sha256"]
            and not cell["decision_issues"] and summary["complete_expected_requests"]
            and summary["identity_valid"] and summary["first_shape_valid"]
            and not summary["scheduler_violations"] and raw["map_validation"]["status"] == "PASS"
            and all(r["timing_valid"] for r in summary["requests"].values()))
        if not valid:
            cell["status"] = out["status"] = "INVALID_COMPARISON"
    for repeat in sorted({r["repeat"] for r in config["cells"]}):
        names = {r["map_mode"]:r["cell"] for r in config["cells"] if r["repeat"] == repeat}
        a,b = names["original"],names["batched"]
        if a not in summaries or b not in summaries:
            continue
        pair = comparison(raws[a],raws[b],summaries[a],summaries[b],"same-task map writes")
        pair.update(original=a,batched=b,repeat=repeat)
        for key in ("output_sha256","final_pager_sha256","schedule_sha256","whole_episode_copy_payload_bytes"):
            pair[key+"_same"] = out["cells"][a][key] == out["cells"][b][key]
        keys = ("resources_same","same_input_tasks","old_tokens_same","scheduler_prestate_same",
                "full_pager_state_same","output_sha256_same","final_pager_sha256_same",
                "schedule_sha256_same","whole_episode_copy_payload_bytes_same")
        if not all(pair[k] for k in keys):
            out["status"] = "INVALID_COMPARISON"
        out["comparisons"].append(pair)
    if execution["status"] != "COMPLETED" or len(out["cells"]) != len(config["cells"]):
        out["status"] = "INCOMPLETE"
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results",required=True,type=Path)
    parser.add_argument("--out",required=True,type=Path)
    args = parser.parse_args()
    result = evaluate(args.results)
    with args.out.open("x") as handle:
        json.dump(result,handle,indent=2,allow_nan=False)
        handle.write("\n")
