"""Evaluate frozen unseen-action predictions against actual, retained GPU cells."""
import argparse
import hashlib
import json
from pathlib import Path

from analyze_wisp_injection import hardware_summary, resources, summarize


def sha(data):
    return hashlib.sha256(data).hexdigest()


def prestate(raw):
    action = raw["action"]
    return sha(json.dumps(dict(tokens=action["old_output_tokens"],
        scheduler=action["before_scheduler"],
        pager=[w["pager_execution_state"] for w in action["before_worker"]]),
        sort_keys=True).encode())


def evaluate(config, reference, results):
    execution = json.loads((results / "execution.json").read_text())
    rows = {r["cell"]: r for r in execution["cells"]}
    calibration = {}
    for name, expected in config["calibration"].items():
        data = (reference / name / "result.json").read_bytes()
        if sha(data) != expected["raw_sha256"]:
            raise ValueError("Calibration changed: " + name)
        calibration[name] = json.loads(data)
    out = dict(status="MEASUREMENT_ONLY", execution=execution, cells=[],
        scope=config["holdout_scope"], first_action_payload=config["first_action_payload"],
        boundary="Prediction errors are observed, not a threshold-selected method verdict. "
                 "Request and elapsed times include intervention observation costs.")
    for prediction in config["frozen_predictions"]:
        name = prediction["cell"]
        path = results / name / "result.json"
        if not path.exists():
            out["cells"].append(dict(cell=name, status="UNRUN_OR_MISSING"))
            out["status"] = "INCOMPLETE"
            continue
        data = path.read_bytes()
        raw = json.loads(data)
        entry = dict(cell=name, raw_sha256=sha(data), status=raw.get("status"))
        out["cells"].append(entry)
        if raw.get("status") != "COMPLETED":
            entry["error"] = raw.get("error")
            out["status"] = "INCOMPLETE"
            continue
        repeat = prediction["calibration_repeat"]
        endpoints = [calibration[f"r{repeat}-long{chunk}"] for chunk in (8, 32)]
        summary = summarize(raw, rows.get(name, {}))
        checks = dict(
            prestate=prestate(raw) == config["preaction_sha256"],
            resources=all(resources(raw) == resources(endpoint) for endpoint in endpoints),
            same_workload=all(raw["workload_sha256"] == endpoint["workload_sha256"]
                              for endpoint in endpoints),
            actual_action=raw["args"]["chunk"] == prediction["chunk"],
            first_shape=summary["first_shape_valid"] and
                        summary["first_action_tokens"] == prediction["expected_first_rows"],
            requests_complete=summary["complete_expected_requests"],
            request_identity=summary["identity_valid"] and summary["preaction_old_prefix_valid"],
            token_timing=all(r["timing_valid"] for r in summary["requests"].values()),
            no_preemption_or_recompute=not summary["scheduler_violations"])
        entry.update(checks=checks, measured=summary,
            hardware=hardware_summary(results / name / "hardware.jsonl", raw,
                                      config["resources"]["cpu_affinity"]))
        if not all(checks.values()):
            entry["status"] = "INVALID_COMPARISON"
            out["status"] = "INVALID_COMPARISON"
            continue
        expected = prediction["first_itl_s"]
        entry["first_itl_prediction_s"] = expected
        entry["first_itl_errors"] = {rid: dict(observed_s=value,
            signed_error_s=value-expected, absolute_error_s=abs(value-expected),
            relative_error=(value-expected)/expected)
            for rid, value in summary["old_cross_action_itl_s"].items()}
        entry["whole_episode_copy_payload_bytes"] = sum(
            layer["weight_copy_payload_bytes"] for worker in raw["final_worker"]
            for layer in worker["pager_layers"])
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "reference", "results", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    output = evaluate(config, args.reference, args.results)
    output["frozen_config_sha256"] = sha(args.config.read_bytes())
    with args.out.open("x") as handle:
        json.dump(output, handle, indent=2, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
