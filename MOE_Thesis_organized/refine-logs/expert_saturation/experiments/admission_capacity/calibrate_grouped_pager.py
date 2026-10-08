"""Validate grouped cache accounting against each policy's actual GPU trace.

POST_ROUTER/OFFLINE: routes come from the same executed policy. This cannot
predict an unexecuted action or establish online feature value. CUDA load spans
are reported separately from overlapping host time. Hold is a two-request
negative control, not a same-task throughput baseline.
"""
import argparse
import hashlib
import json
from pathlib import Path

from resource_transition_model import GroupedSlotState, grouped_slot_dry_run


def state(row):
    return GroupedSlotState(tuple(row["slot_to_expert"]), tuple(row["lru_tick"]), row["lru_clock"])


def analyze(path):
    data = json.loads(path.read_text())
    assert data["status"] == "COMPLETED"
    action = data["action"]
    before = data["action"]["before_worker"][0]["pager_execution_state"]
    current = {s["layer_idx"]: state(s) for s in before}
    layer_counts = {i: 0 for i in current}
    actual_bytes = predicted_bytes = lower_bytes = group_count = 0
    spans = 0.0
    for call in data["paging_trace"]["layers"]:
        if call["engine_call_index"] < action["engine_call"]:
            continue
        assert call["successful"]
        layer = call["layer_idx"]
        entry = {e for e in current[layer].slot_to_expert if e >= 0}
        assert entry == set(call["resident_at_entry"])
        groups = call["subgroups"]
        result = grouped_slot_dry_run(current[layer],
            [g["required_experts"] for g in groups],
            {e: call["expert_weight_bytes"] for e in range(64)})
        for actual, predicted in zip(groups, result.groups):
            assert set(actual["loaded_experts"]) == set(predicted.loaded_experts)
            assert set(actual["evicted_experts"]) == set(predicted.evicted_experts)
            assert actual["load_bytes"] == predicted.load_bytes
            spans += actual["load_section_cuda_ms"]
        assert len(result.groups) == len(groups)
        current[layer] = result.final_state
        assert {e for e in current[layer].slot_to_expert if e >= 0} == set(call["resident_at_exit"])
        required = set().union(*(set(g["required_experts"]) for g in groups))
        lower = len(required - entry) * call["expert_weight_bytes"]
        assert lower == call["unique_lower_bound_bytes"]
        assert result.total_load_bytes == call["actual_load_bytes"]
        assert result.total_load_bytes - lower == call["extra_load_bytes"]
        layer_counts[layer] += 1
        group_count += len(groups)
        actual_bytes += call["actual_load_bytes"]
        predicted_bytes += result.total_load_bytes
        lower_bytes += lower
    final = data["final_worker"][0]["pager_execution_state"]
    assert all(current[s["layer_idx"]] == state(s) for s in final)
    assert len(current) == 16 and len(set(layer_counts.values())) == 1
    requests = data["requests"]
    old = [requests[f"injection-{i}"] for i in range(2)]
    new = requests["injection-2"]
    for row in old:
        assert row["status"] == "COMPLETED" and len(row["output_token_ids"]) == row["max_tokens"]
    return dict(cell=path.parent.name, input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        layer_calls=sum(layer_counts.values()), groups=group_count, exact_final_slots_ticks_clocks=True,
        actual_weight_copy_bytes=actual_bytes, modeled_weight_copy_bytes=predicted_bytes,
        unique_missing_lower_bound_bytes=lower_bytes, extra_load_bytes=actual_bytes-lower_bytes,
        load_section_cuda_ms=spans, old_cross_action_itl_s=[r["token_received_s"][4]-r["token_received_s"][3] for r in old],
        old_post_action_max_itl_s=[max(b-a for a,b in zip(r["token_received_s"][3:], r["token_received_s"][4:])) for r in old],
        new_ttft_s=new["token_received_s"][0]-new["arrival_s"] if new["token_received_s"] else None,
        wall_s=data["wall_s"], same_task=new["status"] == "COMPLETED")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    cells = [analyze(path) for path in sorted(args.results.glob("r*-*/result.json"))]
    if len(cells) != 8:
        raise ValueError("expected all eight declared injection cells")
    report = dict(status="PASS", evidence="POST_ROUTER_TRACE_MODEL_VALIDATION",
        claim="Each actual policy trace reproduces group loads, eviction sets and final slots/ticks/clocks; not action-conditioned counterfactual prediction",
        cells=cells, total_groups=sum(c["groups"] for c in cells),
        source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                       [Path(__file__), Path(__file__).with_name("resource_transition_model.py")]})
    with args.output.open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps(dict(status=report["status"], cells=len(cells), groups=report["total_groups"])))


if __name__ == "__main__":
    main()
