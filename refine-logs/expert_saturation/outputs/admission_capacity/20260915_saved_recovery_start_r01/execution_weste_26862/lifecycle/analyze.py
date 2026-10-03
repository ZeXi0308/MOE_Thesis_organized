"""Describe recovery lifecycles; primary deltas are consumed, never recalculated."""
import json
from pathlib import Path
from statistics import median

here = Path(__file__).resolve().parent
current_path = here.parent / "analysis.json"
reference_path = here.parents[2] / "20260915_repeated_kv_service_r01/analysis/analysis.json"
current = json.loads(current_path.read_text())
reference = json.loads(reference_path.read_text())
eager = current["cells"]["diagnostic-eager"]["diagnostic"]
old = reference["cells"]["diag-on"]["diagnostic"]

def summarize(diag):
    segments = diag["segments"]
    repeated = [s for s in segments if s["end"] == "repreempted"]
    return dict(counts=diag["counts"], distinct_preempted_requests=len({s["request_id"] for s in segments}),
        median_L_to_S_s=median(s["S"]["time_s"]-s["L_s"] for s in segments),
        median_S_to_F_s=median(s["F_s"]-s["S"]["time_s"] for s in segments),
        summed_request_gaps_s=sum(s["F_s"]-s["L_s"] for s in segments),
        median_repreempted_outputs=median(s["useful_outputs"] for s in repeated),
        repreempted_output_counts=sorted(s["useful_outputs"] for s in repeated),
        recompute_positions=sum(s["confirmed_recompute_tokens"] for s in segments),
        completed_store_jobs=sum(j["count"] for e in diag["offload"]["completed_jobs"] for j in e["jobs"] if j["is_store"]),
        completed_load_jobs=sum(j["count"] for e in diag["offload"]["completed_jobs"] for j in e["jobs"] if not j["is_store"]),
        measured_store_bytes=sum(e["store"]["bytes"] for e in diag["offload"]["transfers"]),
        measured_load_bytes=sum(e["load"]["bytes"] for e in diag["offload"]["transfers"]))

extra = sorted({s["request_id"] for s in eager["segments"]} - {s["request_id"] for s in old["segments"]})
blocks = []
for comparison in current["performance_comparisons"]:
    deltas = comparison["request_deltas"]
    counts = {}
    for metric in ("max_gap_eager_minus_current_s", "completion_latency_eager_minus_current_s"):
        counts[metric] = {"better": sum(d[metric] < 0 for d in deltas), "worse": sum(d[metric] > 0 for d in deltas)}
    blocks.append(dict(block=comparison["block"], counts=counts,
        gap_worse_requests=[d for d in deltas if d["max_gap_eager_minus_current_s"] > 0]))

examples = []
for segment in sorted((s for s in eager["segments"] if s["end"] == "repreempted"), key=lambda s: s["useful_outputs"])[:2]:
    successor = next(s for s in eager["segments"] if s["request_id"] == segment["request_id"] and s["preempt_step"] == segment["next_preempt_step"])
    first = successor["executed_steps"][0]
    examples.append(dict(request_id=segment["request_id"], recovery_preempt_step=segment["preempt_step"],
        first_output_s=segment["F_s"], useful_outputs=segment["useful_outputs"],
        next_preempt_step=segment["next_preempt_step"], gpu_history_positions_at_release=segment["discarded_computed_tokens"],
        gpu_blocks_at_release=segment["discarded_block_counts"],
        successor_accepted_load_jobs=[d["job_id"] for d in successor["actual_load_dispatches"]],
        successor_execution_start_computed=first["scheduled_start_computed"],
        successor_recompute_to_first_output=successor["recompute_tokens_to_first_output"],
        successor_useful_outputs=successor["useful_outputs"]))

result = dict(sources=dict(current=str(current_path), historical_reference=str(reference_path)),
    reference_scope="Separate diagnostic groups; descriptive timing reference, not a paired primary comparison.",
    reference=summarize(old), eager=summarize(eager), primary_request_deltas=blocks,
    requests_newly_preempted_relative_to_reference=extra, short_service_examples=examples,
    limits=["Diagnostic sums overlap across requests and are not episode wall-time decomposition.",
        "Primary deltas are copied from the completed current analysis; diagnostic timing is never pooled into them.",
        "GPU release does not imply host invalidation; the examples establish reuse, not universal absence of eviction.",
        "S is a host boundary; earlier policy evolution can change every subsequent lifecycle."])
with (here / "summary.json").open("x") as stream:
    json.dump(result, stream, indent=2)
    stream.write("\n")
print(here / "summary.json")
