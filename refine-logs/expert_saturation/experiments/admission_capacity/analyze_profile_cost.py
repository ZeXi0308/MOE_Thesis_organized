"""Attribute recorded GPU activities within engine-call markers; no latency simulator."""
import argparse
import json
from pathlib import Path

from analyze_wisp_injection import summarize, hardware_summary
from analyze_prefill_midpoint import prestate


def union_us(intervals):
    end, total = float("-inf"), 0.0
    for start, stop in sorted(intervals):
        total += max(0.0, stop-max(start, end))
        end = max(end, stop)
    return total


def classify(event):
    category, name = event.get("cat", ""), event.get("name", "").lower()
    if category == "kernel":
        return "kernel"
    if category == "gpu_memcpy":
        return "h2d" if "htod" in name else "d2h" if "dtoh" in name else "other_copy"
    if category == "gpu_memset":
        return "memset"
    if category in ("cuda_runtime", "cuda_driver"):
        return "cpu_cuda_api"
    return None


def profile_calls(trace, steps):
    events = [e for e in trace["traceEvents"] if e.get("ph") == "X" and e.get("dur", 0) >= 0]
    markers = [e for e in events if e.get("cat") == "user_annotation"
               and e.get("name", "").startswith("ENGINE_CALL/")]
    indices = [int(e["name"].split("/")[-1]) for e in markers]
    if sorted(indices) != [s["index"] for s in steps] or len(set(indices)) != len(indices):
        raise ValueError("Profiler engine markers do not align with recorded calls")
    markers.sort(key=lambda e: e["ts"])
    if [int(e["name"].split("/")[-1]) for e in markers] != [s["index"] for s in steps]:
        raise ValueError("Profiler markers are out of engine-call order")
    if any(a["ts"]+a["dur"] > b["ts"] for a,b in zip(markers,markers[1:])):
        raise ValueError("Profiler engine-call markers overlap")
    activities = [(e, classify(e)) for e in events if classify(e)]
    rows = []
    gpu_seen = False
    for marker, step in zip(markers, steps):
        start, stop = marker["ts"], marker["ts"]+marker["dur"]
        bins = {k: [] for k in ("kernel", "h2d", "d2h", "other_copy", "memset", "cpu_cuda_api")}
        byte_counts = dict(h2d=0, d2h=0, other_copy=0)
        copy_sizes = {}
        names = {}
        for event, kind in activities:
            a, b = max(start, event["ts"]), min(stop, event["ts"]+event["dur"])
            if b <= a:
                continue
            bins[kind].append((a, b))
            gpu_seen |= kind != "cpu_cuda_api"
            if kind in byte_counts and start <= event["ts"] < stop:
                value = event.get("args", {}).get("bytes")
                if isinstance(value, (int, float)):
                    byte_counts[kind] += value
                    key = kind+":"+str(value)
                    copy_sizes[key] = copy_sizes.get(key,0)+1
            key = kind+":"+event["name"]
            count, duration = names.get(key, (0, 0.0))
            names[key] = (count+1, duration+b-a)
        gpu_intervals = [interval for kind, intervals in bins.items()
                         if kind != "cpu_cuda_api" for interval in intervals]
        gpu_union = union_us(gpu_intervals)
        rows.append(dict(engine_call=int(marker["name"].split("/")[-1]),
            activity_status="INCOMPLETE_ACTIVITY" if step.get("total_scheduled_tokens",0) > 0
                and not bins["kernel"] else "RECORDED",
            marked_wall_ms=marker["dur"]/1000, gpu_busy_union_ms=gpu_union/1000,
            marker_minus_step_wall_ms=marker["dur"]/1000-(step["return_s"]-step["start_s"])*1000
                if "return_s" in step else None,
            gap_without_recorded_gpu_activity_ms=(marker["dur"]-gpu_union)/1000,
            activity_union_ms={k:union_us(v)/1000 for k, v in bins.items()},
            activity_counts={k:len(v) for k, v in bins.items()}, copy_bytes=byte_counts,
            copy_size_counts=copy_sizes,
            activity_overlap_ms=(sum(union_us(v) for k,v in bins.items()
                                     if k != "cpu_cuda_api")-gpu_union)/1000,
            named_activities={n:dict(count=c, clipped_sum_ms=d/1000) for n,(c,d) in names.items()}))
    status = "RECORDED_GPU_ACTIVITIES" if gpu_seen else "GPU_ACTIVITY_UNAVAILABLE"
    if any(r["activity_status"] == "INCOMPLETE_ACTIVITY" for r in rows):
        status = "INCOMPLETE_ACTIVITY"
    return dict(status=status, calls=rows,
        boundary="GPU busy and gaps partition each marker. Individual GPU classes may overlap; CPU API time overlaps GPU activity. "
                 "Gaps include submission delay, dependencies and unrecorded activity, not an automatic CPU-cause attribution. "
                 "Copy bytes assigned by start timestamp; extra metadata transfers are not expert weight bytes.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    execution = json.loads((args.results / "execution.json").read_text())
    result = dict(status="DIAGNOSTIC_PROFILE", execution=execution, cells={})
    for row in execution["cells"]:
        name = row["cell"]
        raw = json.loads((args.results/name/"result.json").read_text())
        cell = result["cells"][name] = dict(status=raw["status"])
        if raw["status"] != "COMPLETED":
            cell["error"] = raw.get("error")
            result["status"] = "INCOMPLETE"
            continue
        cell.update(prestate_sha256=prestate(raw), measured=summarize(raw,row),
            profile=profile_calls(json.loads((args.results/name/"profile.json").read_text()), raw["steps"]),
            expert_copy_payload_bytes=sum(s["weight_copy_payload_bytes"] for w in raw["final_worker"] for s in w["pager_layers"]),
            hardware=hardware_summary(args.results/"hardware.jsonl",raw,list(range(8))))
        copied = sum(r["copy_bytes"]["h2d"] for r in cell["profile"]["calls"])
        cell["profile"]["h2d_byte_check"] = dict(recorded_bytes=copied,
            expert_counter_bytes=cell["expert_copy_payload_bytes"],
            status="LOWER_BOUND_CLOSED" if copied >= cell["expert_copy_payload_bytes"]
                else "CAPTURE_BYTE_FIELD_OR_BOUNDARY_INCOMPLETE")
        if copied < cell["expert_copy_payload_bytes"]:
            cell["profile"]["status"] = "INCOMPLETE_ACTIVITY"
    with args.out.open("x") as handle:
        json.dump(result,handle,indent=2,allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
