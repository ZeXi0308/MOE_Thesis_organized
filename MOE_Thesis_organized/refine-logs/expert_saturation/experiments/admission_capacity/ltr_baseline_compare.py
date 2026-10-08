"""Compare complete native request traces for the frozen LTR-style development grid.

This reads each arm's own executed trace. It never estimates a policy's outcome
from another policy's future queue, KV state, output or route. A common cohort
and physical-resource receipt are required before selecting a calibration point.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean


def _number(value, label):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return float(value)


def _load(path):
    return json.loads(Path(path).read_text())


def _identity(raw):
    rows = raw["requests"]
    ids = [r["request_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate request identity")
    return {r["request_id"]: (r["document_id"], r["prompt_token_ids_sha256"],
            _number(r["arrival_s"], "arrival_s"), r["max_output_tokens"])
            for r in rows}


def _percentile(values, fraction):
    values = sorted(values)
    if not values:
        return None
    index = (len(values) - 1) * fraction
    lo, hi = math.floor(index), math.ceil(index)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def summarize(raw, *, df_s, dg_s, horizon_s, fronts=()):
    """All planned requests remain in the denominator, including failures."""
    end = _number(raw["observation_end_s"], "observation_end_s")
    if end > horizon_s + 1e-6:
        raise ValueError("capture exceeded the declared common horizon")
    per_request = []
    for row in raw["requests"]:
        arrival = _number(row["arrival_s"], "arrival_s")
        times = [_number(t, "token_time_s") for t in row["token_times_s"]]
        if len(times) != len(row["output_token_ids"]):
            raise ValueError("output tokens and times differ")
        if arrival > horizon_s or any(t < arrival or t > end for t in times):
            raise ValueError("token outside arrival/capture boundary")
        if any(b < a for a, b in zip(times, times[1:])):
            raise ValueError("nonmonotonic output times")
        completion = row.get("completion_s")
        if completion is not None:
            completion = _number(completion, "completion_s")
            if completion < (times[-1] if times else arrival) or completion > end:
                raise ValueError("completion outside output/capture boundary")
        status = row["status"]
        if status not in ("completed", "failed", "unfinished"):
            raise ValueError("unknown request status")
        if (status == "completed") != (completion is not None):
            raise ValueError("completion/status mismatch")
        unique_returns = sorted(set(times))
        gaps = [b - a for a, b in zip(unique_returns, unique_returns[1:])]
        ttft = times[0] - arrival if times else None
        max_gap = max(gaps) if gaps else None
        passed = (status == "completed" and ttft is not None and ttft <= df_s
                  and (max_gap is None or max_gap <= dg_s))
        per_request.append(dict(request_id=row["request_id"], status=status,
            outputs=len(times), stop_reason=row.get("stop_reason"),
            output_sha256=hashlib.sha256(json.dumps(row["output_token_ids"],
                separators=(",", ":")).encode()).hexdigest(),
            ttft_s=ttft, max_gap_s=max_gap,
            flow_s=completion - arrival if completion is not None else None,
            goodput_pass=passed))
    completed = [r for r in per_request if r["status"] == "completed"]
    flows = [r["flow_s"] for r in completed]
    ttfts = [r["ttft_s"] for r in per_request if r["ttft_s"] is not None]
    gaps = [r["max_gap_s"] for r in per_request if r["max_gap_s"] is not None]
    outputs = sum(r["outputs"] for r in per_request)
    front = []
    for limits in fronts:
        ttft_limit, gap_limit = (_number(limits[k], k) for k in
                                  ("ttft_limit_s", "max_gap_limit_s"))
        front.append(dict(ttft_limit_s=ttft_limit, max_gap_limit_s=gap_limit,
            goodput_rps=sum(r["status"] == "completed" and r["ttft_s"] is not None
                and r["ttft_s"] <= ttft_limit and
                (r["max_gap_s"] is None or r["max_gap_s"] <= gap_limit)
                for r in per_request) / horizon_s))
    return dict(status=raw.get("status"), request_count=len(per_request),
        completed=len(completed), failed=sum(r["status"] == "failed" for r in per_request),
        unfinished=sum(r["status"] == "unfinished" for r in per_request),
        observation_end_s=end, common_horizon_s=horizon_s,
        actual_output_rate_s=outputs / end if end > 0 else None,
        actual_output_tokens=outputs, request_throughput_s=len(completed) / horizon_s,
        goodput_rps=sum(r["goodput_pass"] for r in per_request) / horizon_s,
        goodput_front=front,
        mean_flow_s=mean(flows) if flows else None,
        flow_population="completed requests; failed/unfinished reported separately and disqualify calibration",
        ttft=dict(median=_percentile(ttfts, .5), p90=_percentile(ttfts, .9),
                  p95=_percentile(ttfts, .95)),
        per_request_max_gap=dict(median=_percentile(gaps, .5),
            p90=_percentile(gaps, .9), p95=_percentile(gaps, .95),
            maximum=max(gaps) if gaps else None,
            single_output_or_no_output=len(per_request) - len(gaps)),
        requests=per_request)


def _resource_receipt(receipt, expected, spec, attestation):
    for key in ("usable_gpu_blocks", "gpu_kv_bytes", "host_kv_bytes",
                "block_tokens", "offload_backend"):
        if receipt.get(key) != expected[key]:
            raise ValueError(f"physical resource mismatch: {key}")
    if receipt.get("save_scope") != spec["save_scope"]:
        raise ValueError("save scope mismatch")
    if receipt.get("policy_id") != spec["name"]:
        raise ValueError("policy identity mismatch")
    if "policy_config" in spec and receipt.get("policy_config") != spec["policy_config"]:
        raise ValueError("policy configuration mismatch")
    for key, value in attestation.items():
        if receipt.get(key) != value:
            raise ValueError(f"runtime/workload attestation mismatch: {key}")
    if receipt.get("runtime_verified") is not True:
        raise ValueError("resource receipt is not a runtime observation")


def _pair(reference, treatment):
    left = {r["request_id"]: r for r in reference["requests"]}
    right = {r["request_id"]: r for r in treatment["requests"]}
    if left.keys() != right.keys():
        raise ValueError("paired request identities differ")
    result = {}
    for field in ("ttft_s", "max_gap_s", "flow_s"):
        differences = [right[rid][field] - left[rid][field] for rid in left
            if left[rid][field] is not None and right[rid][field] is not None]
        result[field] = dict(n=len(differences), improved=sum(x < 0 for x in differences),
            worsened=sum(x > 0 for x in differences), tied=sum(x == 0 for x in differences),
            median_delta_s=_percentile(differences, .5),
            max_harm_s=max(differences) if differences else None)
    result["outputs"] = dict(total_delta=sum(right[rid]["outputs"] - left[rid]["outputs"]
        for rid in left), equal_length=sum(right[rid]["outputs"] == left[rid]["outputs"]
        for rid in left), equal_sequence=sum(right[rid]["output_sha256"] ==
        left[rid]["output_sha256"] for rid in left),
        changed_stop=sum(right[rid]["stop_reason"] != left[rid]["stop_reason"]
        for rid in left))
    result["goodput"] = dict(gained=sum(right[rid]["goodput_pass"] and
        not left[rid]["goodput_pass"] for rid in left),
        lost=sum(left[rid]["goodput_pass"] and not right[rid]["goodput_pass"]
        for rid in left))
    return result


def compare(manifest, *, base_dir=Path(".")):
    horizon = _number(manifest["common_horizon_s"], "common_horizon_s")
    df_s, dg_s = (_number(manifest[k], k) for k in ("ttft_limit_s", "max_gap_limit_s"))
    if min(horizon, df_s, dg_s) <= 0:
        raise ValueError("horizon and front thresholds must be positive")
    specs = manifest["cells"]
    if len({s["name"] for s in specs}) != len(specs):
        raise ValueError("duplicate arm name")
    summaries, identities = {}, []
    for spec in specs:
        raw = _load(base_dir / spec["raw"])
        _resource_receipt(_load(base_dir / spec["resource_receipt"]),
            manifest["physical_resources"], spec,
            manifest.get("runtime_attestation", {}))
        identities.append(_identity(raw))
        summaries[spec["name"]] = summarize(raw, df_s=df_s, dg_s=dg_s,
            horizon_s=horizon, fronts=manifest.get("slo_front", ()))
    if not identities or any(identity != identities[0] for identity in identities[1:]):
        raise ValueError("cohort/prompt/arrival/output-cap mismatch")
    expected_count = manifest.get("development_cohort", {}).get("requests")
    if expected_count is not None and len(identities[0]) != expected_count:
        raise ValueError("planned development cohort count differs")
    reference = summaries[manifest["calibration_reference"]]
    grid_specs = [spec for spec in specs if spec.get("role") == "ltr_calibration"]
    grid = manifest.get("calibration_grid", {})
    if grid:
        expected_grid = {(t, q) for t in grid["threshold_calls"]
                         for q in grid["positive_allocation_call_quantum"]}
        actual_grid = {(spec["threshold"], spec["quantum"]) for spec in grid_specs}
        if actual_grid != expected_grid or len(grid_specs) != len(expected_grid):
            raise ValueError("calibration grid differs from the frozen plan")
    complete_grid = (reference["status"] == "COMPLETE" and
        reference["completed"] == reference["request_count"] and
        all(summaries[spec["name"]]["status"] == "COMPLETE" and
            summaries[spec["name"]]["completed"] == summaries[spec["name"]]["request_count"]
            for spec in grid_specs))
    candidates = []
    for spec in specs:
        if spec.get("role") != "ltr_calibration":
            continue
        s = summaries[spec["name"]]
        eligible = (complete_grid and s["status"] == "COMPLETE"
            and s["actual_output_rate_s"] is not None
            and reference["actual_output_rate_s"] is not None
            and s["mean_flow_s"] is not None and reference["mean_flow_s"] is not None
            and s["per_request_max_gap"]["maximum"] is not None)
        if eligible:
            eligible = (s["actual_output_rate_s"] >=
                    manifest["min_output_rate_ratio"] * reference["actual_output_rate_s"]
                and s["mean_flow_s"] <=
                    manifest["max_mean_flow_ratio"] * reference["mean_flow_s"])
        candidates.append(dict(name=spec["name"], threshold=spec["threshold"],
            quantum=spec["quantum"], eligible=bool(eligible),
            max_gap_s=s["per_request_max_gap"]["maximum"]))
    winners = [c for c in candidates if c["eligible"]]
    winners.sort(key=lambda c: (c["max_gap_s"],
        -summaries[c["name"]]["actual_output_rate_s"],
        summaries[c["name"]]["mean_flow_s"], c["threshold"], c["quantum"]))
    contrasts = {name: _pair(reference, summary) for name, summary in summaries.items()
                 if name != manifest["calibration_reference"]}
    return dict(status="COMPLETE_COMPARISON" if all(
        s["status"] == "COMPLETE" for s in summaries.values()) else "INCOMPLETE_COMPARISON",
        cohort_size=len(identities[0]), common_horizon_s=horizon,
        ttft_limit_s=df_s, max_gap_limit_s=dg_s,
        evidence="actual per-policy complete request traces; in-process host return times",
        arms=summaries, paired_vs_reference=contrasts,
        calibration_candidates=candidates,
        selected_ltr=winners[0]["name"] if winners else None,
        selected_ltr_status=("NOT_APPLICABLE" if not grid_specs else
            "INCOMPLETE_GRID" if not complete_grid else
            "QUALIFIED" if winners else "NO_QUALIFYING_POINT"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(_load(args.manifest), base_dir=args.manifest.resolve().parent)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(dict(status=result["status"],
        selected_ltr=result["selected_ltr"], output=str(args.output))))


if __name__ == "__main__":
    main()
