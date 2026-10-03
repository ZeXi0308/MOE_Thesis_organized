"""Freeze a workload-only first-action interpolation before either cap16 run."""
import hashlib
import json
import statistics
import time


def freeze(base, cells):
    if [c["chunk"] for c in cells] != [8, 32, 32, 8]:
        raise RuntimeError("Expected the four predeclared endpoint cells, without cap16")
    rows = []
    for cell in cells:
        data = (base/cell["cell"]/"result.json").read_bytes()
        raw = json.loads(data)
        if raw["status"] != "COMPLETED":
            raise RuntimeError("Incomplete calibration endpoint")
        action = raw["action"]
        step = raw["steps"][action["engine_call"]]
        intervals = []
        for rid, prefix in action["old_output_tokens"].items():
            times = raw["requests"][rid]["token_received_s"]
            intervals.append(times[len(prefix)]-times[len(prefix)-1])
        rows.append(dict(cell=cell["cell"], chunk=cell["chunk"], end_unix=cell["end_unix"],
            raw_sha256=hashlib.sha256(data).hexdigest(),
            first_step_s=step["return_s"]-step["start_s"], first_old_itl_s=max(intervals)))
    means = {str(cap): {metric:statistics.mean(r[metric] for r in rows if r["chunk"] == cap)
                       for metric in ("first_step_s","first_old_itl_s")} for cap in (8,32)}
    prediction = {metric:means["8"][metric]+(means["32"][metric]-means["8"][metric])/3
                  for metric in means["8"]}
    result = dict(status="FROZEN_BEFORE_CAP16", created_unix=time.time(), calibration=rows,
        endpoint_means=means, chunk=16, prediction=prediction,
        model="Linear interpolation in prefill tokens: alpha=(16-8)/(32-8)=1/3",
        scope="First mixed step and incumbent cross-action interval only; no future routes, "
              "whole-request prediction, independent-document holdout or online method claim")
    with (base/"midpoint_prediction.json").open("x") as handle:
        json.dump(result,handle,indent=2,allow_nan=False)
        handle.write("\n")
    return result
