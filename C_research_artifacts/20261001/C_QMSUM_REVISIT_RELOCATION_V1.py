#!/usr/bin/env python3
"""All single-request relocations under a one-path prompt-cache proxy."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from C_QMSUM_ANALYZE_V1 import load, sha

HERE = Path(__file__).resolve().parent
N = 200
BLOCK = 16


def common_prefix(a, b):
    length = 0
    for x, y in zip(a, b):
        if x != y:
            break
        length += 1
    return length


def direct_cost(prompt, order):
    costs = [len(prompt[order[0]])]
    for previous, current in zip(order, order[1:]):
        reusable = min(common_prefix(prompt[previous], prompt[current]),
                       len(prompt[current]) - 1) // BLOCK
        costs.append(len(prompt[current]) - BLOCK * reusable)
    return costs


def score_from_costs(costs):
    cumulative = 0
    objective = 0
    for cost in costs:
        cumulative += cost
        objective += cumulative
    return cumulative, objective


def incoming_edges(prompt, order, matrix):
    previous = None
    edges = {}
    for current in order:
        cost = len(prompt[current]) if previous is None else matrix[previous][current]
        edges[current] = dict(previous_source_index=previous, prompt_work_tokens=cost,
                              reused_prompt_tokens=len(prompt[current]) - cost)
        previous = current
    return edges


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir", type=Path, default=HERE / "20261002_c_qmsum_inputs_v1")
    ap.add_argument("--whole-order", type=Path, default=HERE / "qmsum_density_order_v1.json")
    ap.add_argument("--horn-order", type=Path, default=HERE / "qmsum_horn_compact_v1.json")
    ap.add_argument("--gpu-actions", type=Path, default=HERE / "qmsum_horn_actions_v1.json")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    started_wall, started_cpu = time.perf_counter(), time.process_time()
    workload_path = args.input_dir / "workload.json"
    workload = load(workload_path)
    whole_receipt, horn_receipt = load(args.whole_order), load(args.horn_order)
    rows = workload["requests"]
    if len(rows) != N or [r["source_index"] for r in rows] != list(range(N)):
        raise ValueError("expected all 200 frozen source-order prompts")
    prompt = [r["prompt_token_ids"] for r in rows]
    workload_sha = sha(workload_path)
    for receipt in (whole_receipt, horn_receipt):
        if (receipt["workload_sha256"] != workload_sha
                or receipt["request_count"] != N
                or sorted(receipt["source_indices_in_submission_order"]) != list(range(N))):
            raise ValueError("frozen order/workload differs")
    whole = whole_receipt["source_indices_in_submission_order"]
    horn = horn_receipt["source_indices_in_submission_order"]
    # This matrix uses prompts only. No output, EOS, quality or actual GPU
    # result is loaded until all relocation scores and the winner are fixed.
    edge = [[0] * N for _ in range(N)]
    for previous in range(N):
        for current in range(N):
            reusable = min(common_prefix(prompt[previous], prompt[current]),
                           len(prompt[current]) - 1) // BLOCK
            edge[previous][current] = len(prompt[current]) - BLOCK * reusable

    def score(order):
        work = objective = 0
        previous = None
        for current in order:
            work += (len(prompt[current]) if previous is None else edge[previous][current])
            objective += work
            previous = current
        return work, objective

    base_work, base_objective = score(whole)
    if base_work != 347028:
        raise ValueError(f"whole one-path prompt work differs: {base_work}")
    best_order = list(whole)
    best_work, best_objective = base_work, base_objective
    best_move = (0, 0)  # Representative no-op; all 200 no-ops are enumerated.
    improving_moves = 0
    for source_position in range(N):
        remaining = whole[:source_position] + whole[source_position + 1:]
        for destination_position in range(N):
            candidate = remaining[:destination_position] + [whole[source_position]] + remaining[destination_position:]
            work, objective = score(candidate)
            if objective < base_objective:
                improving_moves += 1
            # Equal objective keeps the no-op; later ties favor lower work,
            # then the smaller source/destination positions.
            key = (objective, source_position != destination_position,
                   work, source_position, destination_position)
            incumbent = (best_objective, best_move[0] != best_move[1],
                         best_work, *best_move)
            if key < incumbent:
                best_order, best_work, best_objective = candidate, work, objective
                best_move = (source_position, destination_position)
    horn_work, horn_objective = score(horn)
    if (score_from_costs(direct_cost(prompt, whole)) != (base_work, base_objective)
            or score_from_costs(direct_cost(prompt, best_order)) != (best_work, best_objective)
            or score_from_costs(direct_cost(prompt, horn)) != (horn_work, horn_objective)):
        raise ValueError("matrix score differs from independent direct LCP recomputation")
    old_edges = incoming_edges(prompt, whole, edge)
    new_edges = incoming_edges(prompt, best_order, edge)
    changed = []
    for index in range(N):
        if old_edges[index] != new_edges[index]:
            changed.append(dict(source_index=index, before=old_edges[index],
                after=new_edges[index],
                prompt_work_delta_tokens=(new_edges[index]["prompt_work_tokens"]
                                          - old_edges[index]["prompt_work_tokens"])))
    if sum(item["prompt_work_delta_tokens"] for item in changed) != best_work - base_work:
        raise ValueError("changed incoming-edge work does not sum to total delta")
    model_result = dict(scope="One previous request's block-aligned reusable prompt path remains; all other prefixes forgotten; serial prompt execution, no output/decode work",
        formula="cost(first)=P_first; cost(i,j)=P_j-16*floor(min(LCP(i,j),P_j-1)/16)",
        candidate_operations=N*N, no_op_operations=N,
        improving_relocation_operations=improving_moves,
        whole=dict(prompt_work_tokens=base_work,
                   sum_equal_weight_cumulative_prompt_completion_tokens=base_objective,
                   mean_cumulative_prompt_completion_tokens=base_objective/N),
        horn=dict(prompt_work_tokens=horn_work,
                  sum_equal_weight_cumulative_prompt_completion_tokens=horn_objective,
                  mean_cumulative_prompt_completion_tokens=horn_objective/N),
        best_single_relocation=dict(moved_source_index=whole[best_move[0]],
            from_position=best_move[0], to_position=best_move[1],
            is_no_op=best_move[0] == best_move[1],
            source_indices_in_submission_order=best_order,
            prompt_work_tokens=best_work,
            sum_equal_weight_cumulative_prompt_completion_tokens=best_objective,
            mean_cumulative_prompt_completion_tokens=best_objective/N,
            objective_improvement_tokens=base_objective-best_objective,
            objective_improvement_fraction=(base_objective-best_objective)/base_objective,
            changed_incoming_edge_costs=changed,
            added_transition_work_tokens=sum(max(0, x["prompt_work_delta_tokens"]) for x in changed),
            saved_transition_work_tokens=-sum(min(0, x["prompt_work_delta_tokens"]) for x in changed),
            net_transition_work_delta_tokens=best_work-base_work),
        any_improving_single_relocation=improving_moves > 0)
    planning_cpu_s = time.process_time()-started_cpu
    planning_wall_s = time.perf_counter()-started_wall
    # Descriptive actual GPU work is appended after the prompt-only winner is fixed.
    actual = load(args.gpu_actions)
    if (actual.get("schema") != "c-qmsum-horn-whole-native512-actions-v1"
            or actual["arms"]["whole"]["initial_prompt_compute_tokens"] != 347028):
        raise ValueError("completed GPU work receipt differs")
    gpu_work = dict(whole_initial_prompt_compute_tokens=actual["arms"]["whole"]["initial_prompt_compute_tokens"],
        horn_initial_prompt_compute_tokens=actual["arms"]["horn"]["initial_prompt_compute_tokens"],
        horn_minus_whole_initial_prompt_compute_tokens=(actual["arms"]["horn"]["initial_prompt_compute_tokens"]
                                                       - actual["arms"]["whole"]["initial_prompt_compute_tokens"]),
        source_sha256=sha(args.gpu_actions),
        scope="Prior measured GPU work only; not fed into the relocation search")
    result = dict(schema="c-qmsum-one-path-single-relocation-proxy-v1",
        input_sha256={"workload.json": workload_sha, "whole_order": sha(args.whole_order),
                      "horn_order": sha(args.horn_order)},
        planning_cpu_s=planning_cpu_s, planning_wall_s=planning_wall_s,
        one_path_proxy=model_result, prior_actual_gpu_prompt_work=gpu_work,
        limits=[
            "The one-path cache is a restricted CPU proxy, not a GPU upper bound or an actual cache oracle; it forgets all prefixes except the immediately previous request's path.",
            "Only one relocation of the fixed whole order was searched. No improving move means no improvement in this neighborhood under this proxy, not global impossibility.",
            "All objective values are prompt-work token-time proxies with unit request weights and zero decode/output work; they are not observed latency or quality."])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(improving_moves=improving_moves,
        moved_source_index=whole[best_move[0]], from_position=best_move[0],
        to_position=best_move[1], objective_improvement_tokens=base_objective-best_objective,
        extra_prompt_work_tokens=best_work-base_work,
        planning_cpu_s=planning_cpu_s), sort_keys=True))


if __name__ == "__main__":
    main()
