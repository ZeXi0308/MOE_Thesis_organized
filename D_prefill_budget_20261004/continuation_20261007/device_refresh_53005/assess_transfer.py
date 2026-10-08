#!/usr/bin/env python3
"""Score OLD frozen host-cost models on NEW-device formal steps; never refit."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / "niyama_component"
sys.path[:0] = [str(OLD), str(ROOT.parent / "mixed_arrivals")]
from calibrate import REGIMES, describe, error_stats, load_run, regime
from analyze_fixed import signature


def sha(data):
    return hashlib.sha256(data).hexdigest()


def assess(path, expected_policy, expected_signature, calibration):
    out = dict(cell=path.parent.name, raw_path=str(path), valid=False, errors=[])
    try:
        blob = path.read_bytes(); raw = json.loads(blob)
        out.update(raw_sha256=sha(blob), policy=raw["policy"], observed_raw_steps=len(raw["steps"]),
            observed_requests=len(raw["requests"]),
            observed_output_tokens=sum(len(q["output_token_ids"]) for q in raw["requests"]))
        out["input_contract_matches"] = signature(raw["requests"]) == expected_signature
        if not out["input_contract_matches"]: out["errors"].append("Frozen request/prompt/arrival/output-count input mismatch")
        if raw["policy"] != expected_policy: out["errors"].append("Policy differs from the frozen formal cell")
        # Absolute name resolves through the existing helper's DATA/name path.
        samples, provenance = load_run(str(path.parent.resolve()))
        out["checks"] = provenance["checks"]
        out["all_completed_step_counts_and_features"] = describe(samples)
        out["steps_over_1s_retained_in_transfer_metrics"] = [
            dict(step_index=q["step_index"], cost_s=q["cost_s"], features=q["features"])
            for q in provenance["excluded_from_artifact_fit_or_filtered_metrics"]]
        models, ranges = calibration["models"], calibration["training_ranges"]
        out["all_completed_steps"] = error_stats(samples, models, ranges)
        out["by_actual_regime"] = {
            name: dict(counts_and_features=describe([s for s in samples if regime(s["x"]) == name]),
                       errors=error_stats([s for s in samples if regime(s["x"]) == name], models, ranges))
            for name in REGIMES}
        out["valid"] = not out["errors"] and provenance["checks"]["passed"]
    except Exception as exc:
        out["errors"].append(repr(exc))
    out["status"] = "DESCRIPTIVE_TRANSFER_SCORED" if out["valid"] else "INVALID_OR_INCOMPLETE_OBSERVATION"
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--design", type=Path, default=ROOT / "design.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    design_blob = args.design.read_bytes(); design = json.loads(design_blob)
    workload = (args.design.parent / design["workload"]).read_bytes()
    if sha(workload) != design["workload_sha256"]: parser.error("Frozen refresh workload hash mismatch")
    old_design_blob = (OLD / "design.json").read_bytes(); old_design = json.loads(old_design_blob)
    calibration_blob = (OLD / "calibration.json").read_bytes(); calibration = json.loads(calibration_blob)
    if sha(calibration_blob) != old_design["calibration_sha256"]: parser.error("OLD frozen calibration hash mismatch; refusing different coefficients")
    expected = [f"{i:02d}_{p}" for i, p in enumerate(design["policies"])]
    sig = signature(json.loads(workload)); rows, missing = [], []
    for cell, policy in zip(expected, design["policies"]):
        path = args.run_directory / cell / "raw.json"
        if path.exists(): rows.append(assess(path, policy, sig, calibration))
        else: missing.append(cell)
    status_path = args.run_directory / "status.json"
    try: controller = json.loads(status_path.read_text()) if status_path.exists() else dict(status="MISSING")
    except Exception as exc: controller = dict(status="UNREADABLE", error=repr(exc))
    environment = args.run_directory / "environment.json"
    try: hardware = json.loads(environment.read_text()) if environment.exists() else None
    except Exception as exc: hardware = dict(read_error=repr(exc))
    ready = not missing and all(r["valid"] for r in rows) and controller.get("status") == "COMPLETE"
    status = "COMPLETE_DESCRIPTIVE_TRANSFER" if ready else "UNRUN" if not rows and controller.get("status") not in ("COMPLETE", "FAILED", "UNREADABLE") else "INCOMPLETE_OR_INVALID"
    result = dict(schema="d-old-model-new-device-transfer-v1", status=status,
        observed_formal_runs=len(rows), valid_complete_formal_runs=sum(r["valid"] for r in rows),
        missing_formal_cells=missing, controller_status=controller, new_environment=hardware,
        old_calibration_path=str(OLD / "calibration.json"), old_calibration_sha256=sha(calibration_blob),
        old_component_design_sha256=sha(old_design_blob), refresh_design_sha256=sha(design_blob),
        unchanged_models=calibration["models"], fixed_margin=1.2, runs=rows,
        scope="OLD coefficients and training ranges only; no fitting, clipping, margin/threshold search, or filtering of completed host durations above 1s. Actual features are P,D,sum(computed_start+1) over scheduled decode rows,sum(computed_start) over scheduled prefill rows; regimes use actual P+D<=512 and >512. Warm runs are excluded. These are host engine.step prediction errors on new hardware/driver, not GPU-only costs, service effects, same-hardware causal evidence, or an online-SLO guarantee. Full service comparison is separate in analyze_fixed.py.")
    output = args.output or args.run_directory / "transfer_results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    print(f"{len(rows)} formal, {result['valid_complete_formal_runs']} valid complete; {status}; {output}")


if __name__ == "__main__":
    main()
