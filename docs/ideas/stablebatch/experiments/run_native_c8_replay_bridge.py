#!/usr/bin/env python3
"""Run the frozen native-background fixed-C8 canonical replay bridge Gate.

Every target MoE block first executes all native expert groups.  A treatment
then runs one additional zero-padded C8 call and replaces only one target raw
contribution before the original gate-weighted index-add combine.  This is a
single-GPU target-MoE-stage bridge/cost experiment, not a serving benchmark.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
import types
from typing import Any, Iterable, Mapping, Sequence


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_single_contribution_pilot as base  # noqa: E402


ProtocolError = base.ProtocolError
CONFIG_RELATIVE = (
    "docs/ideas/stablebatch/experiments/configs/native_c8_replay_bridge_v1.json"
)


def cell_key(row: Mapping[str, Any]) -> str:
    return f"{row['victim_id']}|layer={int(row['layer']):02d}"


def public_trace(trace: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in trace.items() if not key.startswith("_")}


def fraction_payload(value: Fraction | int) -> dict[str, Any]:
    item = value if isinstance(value, Fraction) else Fraction(value, 1)
    return {
        "numerator": item.numerator,
        "denominator": item.denominator,
        "value": float(item),
    }


def fraction_value(payload: Mapping[str, Any]) -> Fraction:
    return Fraction(int(payload["numerator"]), int(payload["denominator"]))


def write_jsonl_new(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def validate_bound_file(repo_root: Path, binding: Mapping[str, Any]) -> Path:
    path = (repo_root / str(binding["path"])).resolve()
    if not path.is_file():
        raise ProtocolError(f"bound input is absent: {path}")
    observed = base.sha256_file(path)
    if observed != str(binding["sha256"]):
        raise ProtocolError(f"bound input hash mismatch: {path}: {observed}")
    return path


def verify_source_model(source_config: Mapping[str, Any]) -> dict[str, Any]:
    model_cfg = source_config["model"]
    model_root = Path(str(model_cfg["local_path"])).resolve()
    observed: dict[str, str] = {}
    for relative, expected in model_cfg["file_sha256"].items():
        path = model_root / str(relative)
        if not path.is_file():
            raise ProtocolError(f"source model file is absent: {path}")
        digest = base.sha256_file(path)
        if digest != str(expected):
            raise ProtocolError(f"source model hash mismatch for {relative}")
        observed[str(relative)] = digest
    return {
        "model_path": str(model_root),
        "model_file_sha256": observed,
        "workload_source": "window_token_ids_bound_inside_frozen_proxy_ledger",
    }


def route_decomposition(
    baseline_changed: Sequence[int], action_changed: Sequence[int]
) -> dict[str, Any]:
    old = set(map(int, baseline_changed))
    new = set(map(int, action_changed))
    recovered = sorted(old - new)
    harmed = sorted(new - old)
    persistent = sorted(old & new)
    return {
        "route_recovered_layers": recovered,
        "route_recovered_count": len(recovered),
        "route_harmed_layers": harmed,
        "route_harmed_count": len(harmed),
        "route_persistent_layers": persistent,
        "route_persistent_count": len(persistent),
        "route_net_reward": len(recovered) - len(harmed),
    }


def recover_frozen_cells(
    model: Any,
    sealed_rows: Sequence[Mapping[str, Any]],
    source_config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Recover live hidden tensors from sealed token windows without reselection."""

    import torch

    hidden_size = int(source_config["model"]["hidden_size"])
    top_k = int(source_config["model"]["num_experts_per_tok"])
    cache: dict[str, Mapping[str, Any]] = {}
    cells: list[dict[str, Any]] = []
    for sealed in sealed_rows:
        window_hash = str(sealed["window_token_ids_sha256"])
        if window_hash not in cache:
            input_ids = torch.tensor(
                [sealed["window_token_ids"]], dtype=torch.long, device="cuda"
            )
            cache[window_hash] = base.run_native_capture(
                model, input_ids, source_config
            )
        capture = cache[window_hash]
        layer = int(sealed["layer"])
        token_idx = int(sealed["flat_token_idx"])
        flat_hidden = capture["moe_inputs"][layer].reshape(-1, hidden_size)
        hidden = flat_hidden[token_idx]
        native_logits = capture["output"].router_logits[layer].reshape(
            -1, int(source_config["model"]["num_experts"])
        )
        with torch.inference_mode():
            replay_logits = model.model.layers[layer].mlp.gate(flat_hidden)
        if not torch.equal(replay_logits, native_logits):
            raise ProtocolError(f"sealed gate replay mismatch for {sealed['cell_key']}")
        logits = native_logits[token_idx]
        weights, experts = base.topk_from_logits(logits, top_k)
        cell = {
            "victim_id": str(sealed["victim_id"]),
            "document_index": int(sealed["document_index"]),
            "window_token_ids": list(map(int, sealed["window_token_ids"])),
            "window_token_ids_sha256": window_hash,
            "layer": layer,
            "flat_token_idx": token_idx,
            "target_hidden_sha256": base.tensor_sha256(hidden),
            "target_router_logits_sha256": base.tensor_sha256(logits),
            "gate_weights": list(map(float, weights.detach().cpu().tolist())),
            "expert_ids": list(map(int, experts.detach().cpu().tolist())),
            "_hidden_cpu": hidden.detach().cpu().clone(),
        }
        for field in (
            "victim_id",
            "document_index",
            "layer",
            "flat_token_idx",
            "target_hidden_sha256",
            "target_router_logits_sha256",
            "window_token_ids_sha256",
            "expert_ids",
            "gate_weights",
        ):
            if cell[field] != sealed[field]:
                raise ProtocolError(
                    f"fresh frozen-cell {field} differs for {sealed['cell_key']}"
                )
        cells.append(cell)
    # The bridge cohort is sealed to eight unique request windows; do not
    # silently accept duplicated or expanded workload identity.
    if len(cache) != 8:
        raise ProtocolError(f"frozen cohort has {len(cache)} unique windows, expected 8")
    return cells


def _target_fields(
    flat_hidden: Any,
    router_logits: Any,
    routing_weights: Any,
    selected_experts: Any,
    cell: Mapping[str, Any],
) -> tuple[Any, Any, list[int], list[float]]:
    token_idx = int(cell["flat_token_idx"])
    victim_logits = router_logits[token_idx]
    victim_experts = list(map(int, selected_experts[token_idx].detach().cpu().tolist()))
    victim_weights = list(map(float, routing_weights[token_idx].detach().cpu().tolist()))
    return flat_hidden[token_idx], victim_logits, victim_experts, victim_weights


@contextlib.contextmanager
def native_then_c8_replay(
    model: Any,
    cell: Mapping[str, Any],
    replay_rank: int | None,
    canonical_m: int = 8,
    focal_slot: int = 5,
    *,
    measure: bool = False,
    detail: bool = True,
    capture_hidden: bool = False,
):
    """Copy OLMoE forward with a strict all-native -> replay -> combine order."""

    import torch
    import torch.nn.functional as F

    layer = int(cell["layer"])
    token_idx = int(cell["flat_token_idx"])
    expert_ids = tuple(map(int, cell["expert_ids"]))
    top_k = len(expert_ids)
    if len(set(expert_ids)) != top_k:
        raise ProtocolError("target top-k expert identities are not unique")
    if replay_rank is not None and replay_rank not in range(top_k):
        raise ProtocolError(f"replay rank is outside [0,{top_k}): {replay_rank}")
    if canonical_m != 8 or focal_slot != 5:
        raise ProtocolError("bridge is frozen to zero C8 with focal slot 5")
    rank_by_expert = {expert_id: rank for rank, expert_id in enumerate(expert_ids)}
    block = model.model.layers[layer].mlp
    original_forward = block.forward
    trace: dict[str, Any] = {}
    start_event = torch.cuda.Event(enable_timing=True) if measure else None
    end_event = torch.cuda.Event(enable_timing=True) if measure else None

    def injected_forward(this: Any, hidden_states: Any):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        flat_hidden = hidden_states.view(-1, hidden_dim)
        router_logits = this.gate(flat_hidden)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, this.top_k, dim=-1
        )
        if this.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        selector_weights = routing_weights
        routing_weights = routing_weights.to(flat_hidden.dtype)

        target_hidden = None
        victim_logits = None
        victim_experts: list[int] | None = None
        victim_weights: list[float] | None = None
        if detail:
            target_hidden, victim_logits, victim_experts, victim_weights = _target_fields(
                flat_hidden,
                router_logits,
                selector_weights,
                selected_experts,
                cell,
            )
            if base.tensor_sha256(target_hidden) != str(cell["target_hidden_sha256"]):
                raise ProtocolError("live target hidden differs from frozen cell")
            if base.tensor_sha256(victim_logits) != str(
                cell["target_router_logits_sha256"]
            ):
                raise ProtocolError("live target router logits differ from frozen cell")
            if victim_experts != list(expert_ids):
                raise ProtocolError("live target experts differ from frozen cell")
            if victim_weights != list(map(float, cell["gate_weights"])):
                raise ProtocolError("live target gate weights differ from frozen cell")

        if start_event is not None:
            start_event.record(torch.cuda.current_stream())
        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=flat_hidden.dtype,
            device=flat_hidden.device,
        )
        expert_mask = F.one_hot(
            selected_experts, num_classes=this.num_experts
        ).permute(2, 1, 0)

        # Phase 1: execute every stock native expert group, including M=0 groups.
        records: list[dict[str, Any]] = []
        for expert_idx in range(this.num_experts):
            idx, top_x = torch.where(expert_mask[expert_idx])
            current_state = flat_hidden[None, top_x].reshape(-1, hidden_dim)
            raw_outputs = this.experts[expert_idx](current_state)
            target_rank = rank_by_expert.get(expert_idx)
            match_mask = None
            if target_rank is not None:
                match_mask = (top_x == token_idx) & (idx == target_rank)
            records.append(
                {
                    "expert_idx": expert_idx,
                    "idx": idx,
                    "top_x": top_x,
                    "current_state": current_state,
                    "native_raw": raw_outputs,
                    "applied_raw": raw_outputs,
                    "target_rank": target_rank,
                    "match_mask": match_mask,
                }
            )

        # Phase 2: only after all native calls, optionally replay the same focal
        # row through the frozen zero-padded C8 state and patch one raw row.
        replay_output = None
        lane = None
        if replay_rank is not None:
            replay_expert = expert_ids[replay_rank]
            lane = torch.zeros(
                (canonical_m, hidden_dim),
                dtype=flat_hidden.dtype,
                device=flat_hidden.device,
            )
            lane[focal_slot] = flat_hidden[token_idx]
            replay_output = this.experts[replay_expert](lane)[focal_slot]
            record = records[replay_expert]
            match_mask = record["match_mask"]
            record["applied_raw"] = torch.where(
                match_mask[:, None], replay_output[None, :], record["native_raw"]
            )

        # Phase 3: retain the stock expert order, gate weights and index_add.
        for record in records:
            current_hidden_states = record["applied_raw"] * routing_weights[
                record["top_x"], record["idx"], None
            ]
            final_hidden_states.index_add_(
                0,
                record["top_x"],
                current_hidden_states.to(flat_hidden.dtype),
            )
        final_hidden_states = final_hidden_states.reshape(
            batch_size, sequence_length, hidden_dim
        )
        if end_event is not None:
            end_event.record(torch.cuda.current_stream())

        if detail:
            pair_counts: dict[int, int] = {}
            local_offsets: dict[int, int] = {}
            natural_m: dict[int, int] = {}
            native_raw: dict[int, Any] = {}
            applied_raw: dict[int, Any] = {}
            gate_weights: dict[int, Any] = {}
            non_target_hasher = hashlib.sha256()
            for record in records:
                target_rank = record["target_rank"]
                local_index: int | None = None
                if target_rank is not None:
                    matches = torch.nonzero(
                        record["match_mask"], as_tuple=False
                    ).reshape(-1)
                    pair_counts[target_rank] = int(matches.numel())
                    if matches.numel() == 1:
                        local_index = int(matches[0].item())
                        local_offsets[target_rank] = local_index
                        natural_m[target_rank] = int(record["current_state"].shape[0])
                        native_raw[target_rank] = record["native_raw"][
                            local_index
                        ].detach().clone()
                        applied_raw[target_rank] = record["applied_raw"][
                            local_index
                        ].detach().clone()
                        gate_weights[target_rank] = routing_weights[
                            token_idx, target_rank
                        ].detach().clone()
                hashable = record["applied_raw"].detach().clone()
                if local_index is not None:
                    hashable[local_index].zero_()
                non_target_hasher.update(
                    int(record["expert_idx"]).to_bytes(4, "little")
                )
                non_target_hasher.update(base.tensor_bytes(record["idx"]))
                non_target_hasher.update(base.tensor_bytes(record["top_x"]))
                non_target_hasher.update(base.tensor_bytes(hashable))
            if any(pair_counts.get(rank, 0) != 1 for rank in range(top_k)):
                raise ProtocolError(f"target pair counts are {pair_counts}")
            trace.update(
                {
                    "execution_order": "all_native_then_optional_c8_then_combine",
                    "layer": layer,
                    "flat_token_idx": token_idx,
                    "hidden_dim": hidden_dim,
                    "replay_rank": replay_rank,
                    "replay_expert_id": (
                        None if replay_rank is None else expert_ids[replay_rank]
                    ),
                    "canonical_m": canonical_m,
                    "focal_slot": focal_slot,
                    "dummy_rows": canonical_m - 1,
                    "logical_native_expert_forward_invocations": this.num_experts,
                    "logical_replay_expert_forward_invocations": int(
                        replay_rank is not None
                    ),
                    "cuda_kernel_launches": "UNKNOWN_NOT_PROFILED",
                    "pair_match_count_by_rank": {
                        str(rank): pair_counts[rank] for rank in range(top_k)
                    },
                    "local_group_row_offset_by_rank": {
                        str(rank): local_offsets[rank] for rank in range(top_k)
                    },
                    "natural_m_by_rank": {
                        str(rank): natural_m[rank] for rank in range(top_k)
                    },
                    "target_input_sha256": base.tensor_sha256(target_hidden),
                    "target_router_logits_sha256": base.tensor_sha256(victim_logits),
                    "target_selected_experts": victim_experts,
                    "target_routing_weights_sha256": base.tensor_sha256(
                        selector_weights[token_idx]
                    ),
                    "target_native_raw_sha256_by_rank": {
                        str(rank): base.tensor_sha256(native_raw[rank])
                        for rank in range(top_k)
                    },
                    "target_applied_raw_sha256_by_rank": {
                        str(rank): base.tensor_sha256(applied_raw[rank])
                        for rank in range(top_k)
                    },
                    "target_gate_weight_sha256_by_rank": {
                        str(rank): base.tensor_sha256(gate_weights[rank])
                        for rank in range(top_k)
                    },
                    "non_target_contributions_sha256": (
                        non_target_hasher.hexdigest()
                    ),
                    "target_moe_output_sha256": base.tensor_sha256(
                        final_hidden_states
                    ),
                    "lane_input_sha256": (
                        None if lane is None else base.tensor_sha256(lane)
                    ),
                    "c8_replay_raw_sha256": (
                        None
                        if replay_output is None
                        else base.tensor_sha256(replay_output)
                    ),
                }
            )
            if capture_hidden:
                trace["_hidden_states_cpu"] = hidden_states.detach().cpu().clone()
        return final_hidden_states, router_logits

    block.forward = types.MethodType(injected_forward, block)
    try:
        trace["_start_event"] = start_event
        trace["_end_event"] = end_event
        yield trace
    finally:
        block.forward = original_forward


def elapsed_ms(trace: Mapping[str, Any]) -> float:
    import torch

    start = trace.get("_start_event")
    end = trace.get("_end_event")
    if start is None or end is None:
        raise ProtocolError("timing trace has no CUDA events")
    torch.cuda.synchronize()
    return float(start.elapsed_time(end))


def assert_observations_equal(
    native: Mapping[str, Any], copied: Mapping[str, Any]
) -> None:
    fields = (
        "input_ids_sha256",
        "attention_mask_sha256",
        "target_input_sha256",
        "target_router_logits_sha256",
        "target_moe_output_sha256",
        "router_logits_sha256_by_layer",
        "topk_experts_by_layer",
        "final_logits_sha256",
        "greedy_token_id",
    )
    for field in fields:
        if native[field] != copied[field]:
            raise ProtocolError(f"copied native path differs from stock at {field}")


def baseline_and_plan(
    model: Any,
    cell: Mapping[str, Any],
    sealed: Mapping[str, Any],
    source_config: Mapping[str, Any],
    bridge_config: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], Any]:
    import torch

    representative = base.PairIdentity(
        layer=int(cell["layer"]),
        flat_token_idx=int(cell["flat_token_idx"]),
        topk_rank=0,
        expert_id=int(cell["expert_ids"][0]),
    )
    input_ids = torch.tensor(
        [cell["window_token_ids"]], dtype=torch.long, device="cuda"
    )
    stock = base.run_observation(model, input_ids, source_config, representative)
    replay_cfg = bridge_config["canonical_replay"]
    with native_then_c8_replay(
        model,
        cell,
        None,
        int(replay_cfg["canonical_m"]),
        int(replay_cfg["focal_slot"]),
        detail=True,
        capture_hidden=True,
    ) as trace:
        copied = base.run_observation(model, input_ids, source_config, representative)
    assert_observations_equal(stock, copied)
    if trace["target_moe_output_sha256"] != copied["target_moe_output_sha256"]:
        raise ProtocolError("baseline trace/output target MoE hash mismatch")
    sealed_native = sealed["reference_arm"]["arm"]
    for key in (
        "target_native_raw_sha256_by_rank",
        "target_gate_weight_sha256_by_rank",
    ):
        if trace[key] != sealed_native[key]:
            raise ProtocolError(f"fresh native trace differs from sealed {key}")

    reference_routes = sealed["reference_arm"]["arm"]["topk_experts_by_layer"]
    start_layer = int(cell["layer"]) + 1
    changed = base.changed_membership_layers(
        reference_routes, copied["topk_experts_by_layer"], start_layer
    )
    cell_id = str(sealed["cell_id"])
    rows: list[dict[str, Any]] = []
    for rank in range(len(cell["expert_ids"])):
        rows.append(
            {
                "schema_version": "stablebatch-native-c8-action-plan-v1",
                "action_id": f"{cell_id}-rank-{rank}",
                "cell_id": cell_id,
                "cell_key": str(sealed["cell_key"]),
                "victim_id": str(cell["victim_id"]),
                "document_index": int(cell["document_index"]),
                "window_token_ids_sha256": str(cell["window_token_ids_sha256"]),
                "layer": int(cell["layer"]),
                "flat_token_idx": int(cell["flat_token_idx"]),
                "rank": rank,
                "expert_id": int(cell["expert_ids"][rank]),
                "natural_m": int(trace["natural_m_by_rank"][str(rank)]),
                "local_group_row_offset": int(
                    trace["local_group_row_offset_by_rank"][str(rank)]
                ),
                "hidden_dim": int(trace["hidden_dim"]),
                "gate_weight": float(cell["gate_weights"][rank]),
                "gate_weight_sha256": str(
                    trace["target_gate_weight_sha256_by_rank"][str(rank)]
                ),
                "target_hidden_sha256": str(cell["target_hidden_sha256"]),
                "target_router_logits_sha256": str(
                    cell["target_router_logits_sha256"]
                ),
                "target_native_raw_sha256": str(
                    trace["target_native_raw_sha256_by_rank"][str(rank)]
                ),
                "canonical_m": int(replay_cfg["canonical_m"]),
                "focal_slot": int(replay_cfg["focal_slot"]),
                "c8_shape": [int(replay_cfg["canonical_m"]), int(trace["hidden_dim"])],
                "c8_expected_raw_sha256": str(
                    sealed["c8_side_calls"]["ranks"][str(rank)]["c8_sha256"]
                ),
                "is_frozen_same_rank": rank == int(sealed["frozen_m1_rank"]),
            }
        )
    baseline = {
        "cell_id": cell_id,
        "cell_key": str(sealed["cell_key"]),
        "victim_id": str(cell["victim_id"]),
        "document_index": int(cell["document_index"]),
        "layer": int(cell["layer"]),
        "frozen_m1_rank": int(sealed["frozen_m1_rank"]),
        "changed_layers_vs_proxy_R": changed,
        "distance_vs_proxy_R": len(changed),
        "proxy_unprotected_changed_layers_vs_R": list(
            sealed["unprotected_arm"]["changed_layers_vs_R"]
        ),
        "observation": base.public_observation(copied),
        "trace": public_trace(trace),
    }
    return baseline, rows, trace["_hidden_states_cpu"]


def freeze_timing_plan(
    action_plan: Sequence[Mapping[str, Any]], max_actions: int
) -> list[dict[str, Any]]:
    selected: dict[str, Mapping[str, Any]] = {}
    ranks = sorted({int(row["rank"]) for row in action_plan})
    for rank in ranks:
        candidates = sorted(
            (row for row in action_plan if int(row["rank"]) == rank),
            key=lambda row: (
                int(row["natural_m"]),
                str(row["cell_id"]),
                str(row["action_id"]),
            ),
        )
        for row in (candidates[0], candidates[-1]):
            selected[str(row["action_id"])] = row
    ordered = sorted(
        selected.values(), key=lambda row: (int(row["rank"]), str(row["action_id"]))
    )
    if len(ordered) > max_actions:
        raise ProtocolError("frozen timing coverage exceeds configured max_actions")
    return [
        {
            "action_id": str(row["action_id"]),
            "cell_id": str(row["cell_id"]),
            "rank": int(row["rank"]),
            "expert_id": int(row["expert_id"]),
            "natural_m": int(row["natural_m"]),
            "selection_used_utility": False,
            "selection_used_latency": False,
        }
        for row in ordered
    ]


def assert_action_trace(
    plan: Mapping[str, Any],
    baseline_trace: Mapping[str, Any],
    action_trace: Mapping[str, Any],
    sealed: Mapping[str, Any],
) -> None:
    rank = int(plan["rank"])
    key = str(rank)
    scalar_checks = {
        "layer": int(plan["layer"]),
        "flat_token_idx": int(plan["flat_token_idx"]),
        "hidden_dim": int(plan["hidden_dim"]),
        "canonical_m": int(plan["canonical_m"]),
        "focal_slot": int(plan["focal_slot"]),
        "replay_rank": rank,
        "replay_expert_id": int(plan["expert_id"]),
    }
    for field, expected in scalar_checks.items():
        if action_trace[field] != expected:
            raise ProtocolError(
                f"{plan['action_id']} trace {field}={action_trace[field]} != {expected}"
            )
    if int(action_trace["natural_m_by_rank"][key]) != int(plan["natural_m"]):
        raise ProtocolError(f"{plan['action_id']} natural M drifted")
    if int(action_trace["local_group_row_offset_by_rank"][key]) != int(
        plan["local_group_row_offset"]
    ):
        raise ProtocolError(f"{plan['action_id']} local native row drifted")
    if action_trace["target_input_sha256"] != str(plan["target_hidden_sha256"]):
        raise ProtocolError(f"{plan['action_id']} target hidden drifted")
    if action_trace["target_router_logits_sha256"] != str(
        plan["target_router_logits_sha256"]
    ):
        raise ProtocolError(f"{plan['action_id']} target router drifted")
    if action_trace["target_selected_experts"] != list(
        map(int, sealed["expert_ids"])
    ):
        raise ProtocolError(f"{plan['action_id']} target experts drifted")
    if action_trace["target_gate_weight_sha256_by_rank"][key] != str(
        plan["gate_weight_sha256"]
    ):
        raise ProtocolError(f"{plan['action_id']} gate weight drifted")
    if action_trace["target_native_raw_sha256_by_rank"] != baseline_trace[
        "target_native_raw_sha256_by_rank"
    ]:
        raise ProtocolError(f"{plan['action_id']} native raw background drifted")
    if action_trace["non_target_contributions_sha256"] != baseline_trace[
        "non_target_contributions_sha256"
    ]:
        raise ProtocolError(f"{plan['action_id']} non-target contribution drifted")
    if action_trace["c8_replay_raw_sha256"] != str(
        plan["c8_expected_raw_sha256"]
    ):
        raise ProtocolError(f"{plan['action_id']} C8 replay hash drifted")
    if action_trace["target_applied_raw_sha256_by_rank"][key] != str(
        plan["c8_expected_raw_sha256"]
    ):
        raise ProtocolError(f"{plan['action_id']} C8 raw was not applied")
    for other in range(len(sealed["expert_ids"])):
        other_key = str(other)
        if other == rank:
            continue
        if action_trace["target_applied_raw_sha256_by_rank"][other_key] != (
            baseline_trace["target_native_raw_sha256_by_rank"][other_key]
        ):
            raise ProtocolError(
                f"{plan['action_id']} unselected rank {other} is not native"
            )


def run_action(
    model: Any,
    cell: Mapping[str, Any],
    sealed: Mapping[str, Any],
    baseline: Mapping[str, Any],
    plan: Mapping[str, Any],
    source_config: Mapping[str, Any],
    bridge_config: Mapping[str, Any],
) -> dict[str, Any]:
    import torch

    rank = int(plan["rank"])
    identity_checks = {
        "cell_id": (str(plan["cell_id"]), str(sealed["cell_id"])),
        "cell_key": (str(plan["cell_key"]), str(sealed["cell_key"])),
        "victim_id": (str(plan["victim_id"]), str(cell["victim_id"])),
        "document_index": (
            int(plan["document_index"]),
            int(cell["document_index"]),
        ),
        "window_token_ids_sha256": (
            str(plan["window_token_ids_sha256"]),
            str(cell["window_token_ids_sha256"]),
        ),
        "layer": (int(plan["layer"]), int(cell["layer"])),
        "flat_token_idx": (
            int(plan["flat_token_idx"]),
            int(cell["flat_token_idx"]),
        ),
        "expert_id": (
            int(plan["expert_id"]),
            int(cell["expert_ids"][rank]),
        ),
        "gate_weight": (
            float(plan["gate_weight"]),
            float(cell["gate_weights"][rank]),
        ),
    }
    for field, (observed, expected) in identity_checks.items():
        if observed != expected:
            raise ProtocolError(f"action {field} mismatch: {observed} != {expected}")
    if int(plan["expert_id"]) != int(sealed["expert_ids"][rank]):
        raise ProtocolError("action expert/rank mismatch")
    if list(map(int, plan["c8_shape"])) != [
        int(plan["canonical_m"]),
        int(plan["hidden_dim"]),
    ]:
        raise ProtocolError("action C8 shape is inconsistent with ledger M/H")
    representative = base.PairIdentity(
        layer=int(cell["layer"]),
        flat_token_idx=int(cell["flat_token_idx"]),
        topk_rank=0,
        expert_id=int(cell["expert_ids"][0]),
    )
    input_ids = torch.tensor(
        [cell["window_token_ids"]], dtype=torch.long, device="cuda"
    )
    replay_cfg = bridge_config["canonical_replay"]
    with native_then_c8_replay(
        model,
        cell,
        rank,
        int(replay_cfg["canonical_m"]),
        int(replay_cfg["focal_slot"]),
        detail=True,
    ) as trace:
        observation = base.run_observation(model, input_ids, source_config, representative)
    assert_action_trace(plan, baseline["trace"], trace, sealed)
    if trace["target_moe_output_sha256"] != observation["target_moe_output_sha256"]:
        raise ProtocolError("action trace/output target MoE hash mismatch")
    for layer in range(int(cell["layer"]) + 1):
        if observation["router_logits_sha256_by_layer"][layer] != baseline[
            "observation"
        ]["router_logits_sha256_by_layer"][layer]:
            raise ProtocolError(f"action differs before intervention at layer {layer}")
    reference_routes = sealed["reference_arm"]["arm"]["topk_experts_by_layer"]
    changed = base.changed_membership_layers(
        reference_routes,
        observation["topk_experts_by_layer"],
        int(cell["layer"]) + 1,
    )
    parts = route_decomposition(
        baseline["changed_layers_vs_proxy_R"], changed
    )
    return {
        "schema_version": "stablebatch-native-c8-replay-action-v1",
        "action_id": str(plan["action_id"]),
        "cell_id": str(plan["cell_id"]),
        "rank": rank,
        "expert_id": int(plan["expert_id"]),
        "natural_m": int(plan["natural_m"]),
        "changed_layers_vs_proxy_R": changed,
        "distance_vs_proxy_R": len(changed),
        **parts,
        "observation": base.public_observation(observation),
        "trace": public_trace(trace),
        "integrity_status": "PASS",
    }


def action_sum(
    actions: Sequence[Mapping[str, Any]], divisor: int = 1
) -> dict[str, Any]:
    return {
        "recovered": fraction_payload(
            Fraction(sum(int(row["route_recovered_count"]) for row in actions), divisor)
        ),
        "harmed": fraction_payload(
            Fraction(sum(int(row["route_harmed_count"]) for row in actions), divisor)
        ),
        "net": fraction_payload(
            Fraction(sum(int(row["route_net_reward"]) for row in actions), divisor)
        ),
        "action_rows": len(actions),
        "expectation_divisor": divisor,
    }


def choose_oracle(actions: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    best = min(
        actions,
        key=lambda row: (
            -int(row["route_net_reward"]),
            -int(row["route_recovered_count"]),
            int(row["route_harmed_count"]),
            int(row["rank"]),
        ),
    )
    return best if int(best["route_net_reward"]) > 0 else None


def proxy_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    same = [row["c8_actions"][str(int(row["frozen_m1_rank"]))] for row in rows]
    all_actions = [
        row["c8_actions"][str(rank)] for row in rows for rank in range(8)
    ]
    oracle_actions = [
        chosen
        for row in rows
        if (chosen := choose_oracle(list(row["c8_actions"].values()))) is not None
    ]
    return {
        "same_rank": action_sum(same),
        "matched_random": action_sum(all_actions, 8),
        "abstaining_oracle": action_sum(oracle_actions),
    }


def native_metrics(cell_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    same: list[Mapping[str, Any]] = []
    all_actions: list[Mapping[str, Any]] = []
    oracle_actions: list[Mapping[str, Any]] = []
    cell_signs = {"positive": 0, "zero": 0, "negative": 0}
    per_document: dict[int, list[Mapping[str, Any]]] = {}
    for row in cell_rows:
        actions = list(row["actions"].values())
        all_actions.extend(actions)
        frozen = row["actions"][str(int(row["frozen_m1_rank"]))]
        same.append(frozen)
        net = int(frozen["route_net_reward"])
        cell_signs["positive" if net > 0 else "negative" if net < 0 else "zero"] += 1
        per_document.setdefault(int(row["document_index"]), []).append(frozen)
        chosen = choose_oracle(actions)
        if chosen is not None:
            oracle_actions.append(chosen)
    document_rows: list[dict[str, Any]] = []
    for document, actions in sorted(per_document.items()):
        metric = action_sum(actions)
        document_rows.append(
            {"document_index": document, "cell_count": len(actions), **metric}
        )
    document_signs = {"positive": 0, "zero": 0, "negative": 0}
    for row in document_rows:
        value = fraction_value(row["net"])
        document_signs[
            "positive" if value > 0 else "negative" if value < 0 else "zero"
        ] += 1
    action_signs = {
        "positive": sum(int(int(row["route_net_reward"]) > 0) for row in all_actions),
        "zero": sum(int(int(row["route_net_reward"]) == 0) for row in all_actions),
        "negative": sum(int(int(row["route_net_reward"]) < 0) for row in all_actions),
    }
    return {
        "same_rank": action_sum(same),
        "matched_random": action_sum(all_actions, 8),
        "abstaining_oracle": action_sum(oracle_actions),
        "same_rank_cell_signs": cell_signs,
        "all_action_signs": action_signs,
        "same_rank_document_signs": document_signs,
        "per_document_same_rank": document_rows,
        "oracle_selected_cell_count": len(oracle_actions),
    }


def percentile(values: Sequence[float], percent: float) -> float:
    if not values:
        raise ProtocolError("cannot take percentile of empty values")
    ordered = sorted(map(float, values))
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percent / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def run_direct_timing(
    model: Any,
    cell: Mapping[str, Any],
    hidden_cpu: Any,
    rank: int,
    bridge_config: Mapping[str, Any],
) -> dict[str, Any]:
    import torch

    timing_cfg = bridge_config["timing"]
    replay_cfg = bridge_config["canonical_replay"]
    hidden = hidden_cpu.to(device="cuda", dtype=torch.bfloat16)
    block = model.model.layers[int(cell["layer"])].mlp

    def one(replay: int | None, measure: bool) -> tuple[float | None, str]:
        with native_then_c8_replay(
            model,
            cell,
            replay,
            int(replay_cfg["canonical_m"]),
            int(replay_cfg["focal_slot"]),
            measure=measure,
            detail=False,
        ) as trace:
            with torch.inference_mode():
                output, _router = block(hidden)
        value = elapsed_ms(trace) if measure else None
        return value, base.tensor_sha256(output)

    # One detailed, untimed equivalence preflight precedes repeated measurements.
    with native_then_c8_replay(
        model,
        cell,
        None,
        int(replay_cfg["canonical_m"]),
        int(replay_cfg["focal_slot"]),
        detail=True,
    ) as native_trace:
        with torch.inference_mode():
            native_output, _ = block(hidden)
    with native_then_c8_replay(
        model,
        cell,
        rank,
        int(replay_cfg["canonical_m"]),
        int(replay_cfg["focal_slot"]),
        detail=True,
    ) as replay_trace:
        with torch.inference_mode():
            replay_output, _ = block(hidden)
    torch.cuda.synchronize()
    if native_trace["target_moe_output_sha256"] != base.tensor_sha256(native_output):
        raise ProtocolError("timing native preflight output mismatch")
    if replay_trace["target_moe_output_sha256"] != base.tensor_sha256(replay_output):
        raise ProtocolError("timing replay preflight output mismatch")

    for pair in range(int(timing_cfg["warmup_pairs"])):
        order = (None, rank) if pair % 2 == 0 else (rank, None)
        for arm in order:
            one(arm, False)
    torch.cuda.synchronize()

    native_ms: list[float] = []
    replay_ms: list[float] = []
    deltas: list[float] = []
    relatives: list[float] = []
    for pair in range(int(timing_cfg["measurement_pairs"])):
        values: dict[str, float] = {}
        order = (("native", None), ("replay", rank))
        if pair % 2:
            order = tuple(reversed(order))
        for label, arm in order:
            measured, _hash = one(arm, True)
            assert measured is not None
            values[label] = measured
        native_ms.append(values["native"])
        replay_ms.append(values["replay"])
        delta = values["replay"] - values["native"]
        deltas.append(delta)
        relatives.append(delta / values["native"])
    low, high = map(float, timing_cfg["range_percentiles"])
    return {
        "target_moe_stage_native_ms": {
            "median": statistics.median(native_ms),
            f"p{int(low)}": percentile(native_ms, low),
            f"p{int(high)}": percentile(native_ms, high),
        },
        "target_moe_stage_native_plus_replay_ms": {
            "median": statistics.median(replay_ms),
            f"p{int(low)}": percentile(replay_ms, low),
            f"p{int(high)}": percentile(replay_ms, high),
        },
        "paired_direct_patch_delta_ms": {
            "median": statistics.median(deltas),
            f"p{int(low)}": percentile(deltas, low),
            f"p{int(high)}": percentile(deltas, high),
        },
        "paired_relative_direct_overhead": {
            "median": statistics.median(relatives),
            f"p{int(low)}": percentile(relatives, low),
            f"p{int(high)}": percentile(relatives, high),
        },
        "raw_native_ms": native_ms,
        "raw_native_plus_replay_ms": replay_ms,
        "raw_paired_delta_ms": deltas,
        "raw_paired_relative_overhead": relatives,
        "warmup_pairs": int(timing_cfg["warmup_pairs"]),
        "measurement_pairs": int(timing_cfg["measurement_pairs"]),
        "event_boundary": str(timing_cfg["event_boundary"]),
        "dummy_rows": int(replay_cfg["dummy_rows"]),
        "canonical_m": int(replay_cfg["canonical_m"]),
    }


def aggregate_timing(
    timing_rows: Sequence[Mapping[str, Any]],
    action_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    native = [value for row in timing_rows for value in row["raw_native_ms"]]
    replay = [
        value for row in timing_rows for value in row["raw_native_plus_replay_ms"]
    ]
    delta = [value for row in timing_rows for value in row["raw_paired_delta_ms"]]
    relative = [
        value for row in timing_rows for value in row["raw_paired_relative_overhead"]
    ]
    low, high = 10.0, 90.0
    selected_net = sum(
        int(action_by_id[str(row["action_id"])]["route_net_reward"])
        for row in timing_rows
    )
    sum_action_median_delta = sum(
        float(row["paired_direct_patch_delta_ms"]["median"])
        for row in timing_rows
    )
    per_recovered = (
        sum_action_median_delta / selected_net if selected_net > 0 else None
    )
    return {
        "timed_action_count": len(timing_rows),
        "paired_sample_count": len(delta),
        "target_moe_stage_native_ms": {
            "median": statistics.median(native),
            "p10": percentile(native, low),
            "p90": percentile(native, high),
        },
        "target_moe_stage_native_plus_replay_ms": {
            "median": statistics.median(replay),
            "p10": percentile(replay, low),
            "p90": percentile(replay, high),
        },
        "paired_direct_patch_delta_ms": {
            "median": statistics.median(delta),
            "p10": percentile(delta, low),
            "p90": percentile(delta, high),
        },
        "paired_relative_direct_overhead": {
            "median": statistics.median(relative),
            "p10": percentile(relative, low),
            "p90": percentile(relative, high),
        },
        "per_protected_action_delta_ms": statistics.median(delta),
        "timing_subset_net_route_reward": selected_net,
        "timing_subset_sum_action_median_delta_ms": sum_action_median_delta,
        "timing_subset_per_net_recovered_route_ms": per_recovered,
        "full_policy_per_recovered_route_cost": "UNRESOLVED_TIMING_SUBSET_NOT_EXTRAPOLATED",
        "dummy_ratio": Fraction(7, 8).__float__(),
        "cost_scope": "target_moe_stage_cuda_event_not_ttft_tpot_or_serving_slo",
    }


def classify_bridge(
    opportunity_routes: int,
    native: Mapping[str, Any],
    timing: Mapping[str, Any],
    high_threshold: float,
) -> str:
    if opportunity_routes == 0:
        return "NO_NATIVE_OPPORTUNITY"
    oracle_net = fraction_value(native["abstaining_oracle"]["net"])
    if oracle_net <= 0:
        return "PROXY_BACKGROUND_DEPENDENT"
    if float(timing["paired_relative_direct_overhead"]["median"]) >= high_threshold:
        return "NATIVE_ACTION_VALID_DIRECT_COST_HIGH"
    same_net = fraction_value(native["same_rank"]["net"])
    random_net = fraction_value(native["matched_random"]["net"])
    if same_net > 0 and same_net > random_net:
        return "NATIVE_REPLAY_AND_RANK_SPECIFICITY_TRANSFER"
    return "NATIVE_REPLAY_TRANSFERS_RANK_SIGNAL_WEAK"


def metric_number(payload: Mapping[str, Any]) -> str:
    value = fraction_value(payload)
    return str(value.numerator) if value.denominator == 1 else f"{float(value):.4f}"


def build_report(summary: Mapping[str, Any]) -> str:
    classification = str(summary["bridge_classification"])
    proxy_metric = summary["proxy_metrics"]["same_rank"]
    native = summary["native_metrics"]
    timing = summary["direct_cost"]
    transfer = summary["bridge_metrics"]
    implication = {
        "NATIVE_REPLAY_AND_RANK_SPECIFICITY_TRANSFER": (
            "Canonical ShapePatch + profile/witness policy"
        ),
        "NATIVE_REPLAY_TRANSFERS_RANK_SIGNAL_WEAK": (
            "Canonical ShapePatch + profile/witness policy"
        ),
        "NATIVE_ACTION_VALID_DIRECT_COST_HIGH": "high-risk opportunistic replay",
        "PROXY_BACKGROUND_DEPENDENT": "\u6269\u5927 protection unit",
        "NO_NATIVE_OPPORTUNITY": "\u505c\u6b62\u5f53\u524d sparse action",
    }[classification]
    next_experiment = {
        "NATIVE_REPLAY_AND_RANK_SPECIFICITY_TRANSFER": (
            "\u5728\u4e00\u4e2a\u9884\u5148\u51bb\u7ed3\u7684 profile/witness \u7b56\u7565\u4e0a\uff0c\u7528\u7ea6\u675f\u5f0f StabilityBudget "
            "\u6d4b\u4e00\u6b21 held-out sparse replay \u6548\u7528\u4e0e\u76f4\u63a5\u6210\u672c\u3002"
        ),
        "NATIVE_REPLAY_TRANSFERS_RANK_SIGNAL_WEAK": (
            "\u4ec5\u7528 native-background trace \u51bb\u7ed3\u4e00\u5f20 per-regime static rank map\uff0c\u7136\u540e\u5728 held-out cells \u4e0a\u6d4b\u4e00\u6b21 replay \u8f6c\u79fb\u3002"
        ),
        "NATIVE_ACTION_VALID_DIRECT_COST_HIGH": (
            "\u53ea\u5bf9\u51bb\u7ed3\u7684 high-risk \u5c0f\u961f\u5217\u505a\u4e00\u6b21 opportunistic replay\uff0c\u68c0\u67e5\u7ea6\u675f\u9884\u7b97\u4e0b\u662f\u5426\u5b58\u5728\u975e\u7a7a\u6b63\u6548\u7528\u96c6\u5408\u3002"
        ),
        "PROXY_BACKGROUND_DEPENDENT": (
            "\u5728\u539f 33 cells \u4e0a\u53ea\u6bd4\u8f83\u201c\u5355 rank\u201d\u4e0e\u201c\u6574\u4e2a cell \u591a rank\u201d\u4e24\u79cd protection unit \u7684 native replay \u51c0\u6548\u7528\u3002"
        ),
        "NO_NATIVE_OPPORTUNITY": (
            "\u53ea\u91cd\u653e\u4e00\u6b21 proxy U \u4e0e native N \u7684 baseline semantic alignment\uff0c\u5b9a\u4f4d 41 \u4e2a proxy opportunity \u4e3a\u4f55\u6d88\u5931\u3002"
        ),
    }[classification]
    lines = [
        "## Bridge classification",
        "",
        classification,
        "",
        "## Result table",
        "",
        "| Condition | Recovered | Harmed | Net |",
        "|---|---:|---:|---:|",
        f"| proxy-background same-rank | {metric_number(proxy_metric['recovered'])} | {metric_number(proxy_metric['harmed'])} | {metric_number(proxy_metric['net'])} |",
        f"| native-background same-rank | {metric_number(native['same_rank']['recovered'])} | {metric_number(native['same_rank']['harmed'])} | {metric_number(native['same_rank']['net'])} |",
        f"| native-background matched random | {metric_number(native['matched_random']['recovered'])} | {metric_number(native['matched_random']['harmed'])} | {metric_number(native['matched_random']['net'])} |",
        f"| native-background oracle | {metric_number(native['abstaining_oracle']['recovered'])} | {metric_number(native['abstaining_oracle']['harmed'])} | {metric_number(native['abstaining_oracle']['net'])} |",
        "",
        "## Bridge metrics",
        "",
        f"same-rank bridge transfer = {transfer['same_rank_bridge_transfer']:.4f}; oracle bridge transfer = {transfer['oracle_bridge_transfer']:.4f}; native specificity gap = {transfer['native_specificity_gap']:.4f}. Same-rank cells positive/zero/negative = {native['same_rank_cell_signs']['positive']}/{native['same_rank_cell_signs']['zero']}/{native['same_rank_cell_signs']['negative']}; positive document coverage = {native['same_rank_document_signs']['positive']}/{summary['document_count']}.",
        "",
        "## Direct cost",
        "",
        f"Target-MoE-stage native total latency median {timing['target_moe_stage_native_ms']['median']:.6f} ms (p10-p90 {timing['target_moe_stage_native_ms']['p10']:.6f}-{timing['target_moe_stage_native_ms']['p90']:.6f}); native + replay {timing['target_moe_stage_native_plus_replay_ms']['median']:.6f} ms ({timing['target_moe_stage_native_plus_replay_ms']['p10']:.6f}-{timing['target_moe_stage_native_plus_replay_ms']['p90']:.6f}). Paired delta {timing['paired_direct_patch_delta_ms']['median']:.6f} ms ({timing['paired_direct_patch_delta_ms']['p10']:.6f}-{timing['paired_direct_patch_delta_ms']['p90']:.6f}), relative overhead {timing['paired_relative_direct_overhead']['median']:.4f}, per protected action {timing['per_protected_action_delta_ms']:.6f} ms, dummy ratio 7/8 = 87.5%. Per-recovered-route cost is reported only for the exact frozen timing subset: {timing['timing_subset_per_net_recovered_route_ms']} ms; it is not extrapolated to the full policy.",
        "",
        "## Mechanistic interpretation",
        "",
        f"The strict bridge executed every native expert group before issuing the selected C8 replay, so unselected rows retained their native batch shape and raw outputs. The native baseline exposed {summary['native_opportunity']['route_count']} downstream route opportunities across {summary['native_opportunity']['cell_count']} cells. The replay result is classified as {classification} under the pre-latency decision order. Fixed C8 remains a canonical arithmetic state rather than M1, M64, FP32, or ground truth. The measured delta includes duplicate expert compute, seven dummy rows, replay launch, replacement, and the unchanged gate/scatter/combine path.",
        "",
        "## System implication",
        "",
        implication,
        "",
        "## Scope",
        "",
        "This is one frozen OLMoE revision on one RTX 5090 with 33 proxy-selected cells. Route recovery is not model-quality improvement, and target-MoE-stage CUDA-event cost is not TTFT, TPOT, queueing, fragmentation, lost batching opportunity, or a serving SLO. No ridge-selector outcome was used.",
        "",
        "## Next minimal experiment",
        "",
        next_experiment,
        "",
    ]
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=HERE.parents[3])
    parser.add_argument(
        "--config",
        type=Path,
        default=HERE / "configs/native_c8_replay_bridge_v1.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-wall-seconds", type=int, default=2400)
    parser.add_argument("--layer", type=int)
    parser.add_argument("--expert", type=int)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    config_path = args.config.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise ProtocolError(f"refusing to reuse output directory {output_dir}")
    if str(config_path.relative_to(repo_root)) != CONFIG_RELATIVE:
        raise ProtocolError("bridge config path differs from canonical path")
    config = base.load_json(config_path)
    if config.get("schema_version") != "stablebatch-native-c8-replay-bridge-v1":
        raise ProtocolError("wrong bridge config schema")
    if config.get("status") != "FROZEN_BEFORE_NATIVE_LATENCY":
        raise ProtocolError("bridge config is not frozen before native latency")
    source_path = validate_bound_file(repo_root, config["source_config"])
    proxy_path = validate_bound_file(repo_root, config["proxy_ledger"])
    source_config = base.load_json(source_path)
    sealed_rows = base.load_jsonl(proxy_path)
    expected_cells = int(config["proxy_ledger"]["expected_cells"])
    if len(sealed_rows) != expected_cells:
        raise ProtocolError("proxy ledger cell count mismatch")
    if len({int(row["document_index"]) for row in sealed_rows}) != int(
        config["proxy_ledger"]["expected_documents"]
    ):
        raise ProtocolError("proxy ledger document count mismatch")
    if any(row.get("integrity_status") != "PASS" for row in sealed_rows):
        raise ProtocolError("proxy ledger contains a failed row")
    proxy_summary = proxy_metrics(sealed_rows)
    for label, key in (("proxy_same_rank", "same_rank"), ("proxy_oracle", "abstaining_oracle")):
        expected = config["proxy_ledger"][label]
        observed = proxy_summary[key]
        for field in ("recovered", "harmed", "net"):
            if fraction_value(observed[field]) != int(expected[field]):
                raise ProtocolError(f"sealed {label} {field} closure mismatch")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    started = time.time()
    base.write_json_new(
        output_dir / "run_request.json",
        {
            "schema_version": "stablebatch-native-c8-replay-request-v1",
            "started_at": base.utc_now(),
            "argv": sys.argv,
            "runner_sha256": base.sha256_file(Path(__file__).resolve()),
            "config_sha256": base.sha256_file(config_path),
            "source_config_sha256": base.sha256_file(source_path),
            "proxy_ledger_sha256": base.sha256_file(proxy_path),
            "max_wall_seconds": int(args.max_wall_seconds),
            "cli_layer_assertion": args.layer,
            "cli_expert_assertion": args.expert,
        },
    )
    try:
        pre_gpu = base.gpu_snapshot()
        runtime_source_config = copy.deepcopy(source_config)
        source_uuid = str(runtime_source_config["environment"]["gpu_uuid"])
        runtime_source_config["environment"]["gpu_uuid"] = str(pre_gpu["uuid"])
        environment = base.verify_environment(runtime_source_config, pre_gpu)
        environment["source_gpu_uuid"] = source_uuid
        environment["bridge_gpu_uuid"] = str(pre_gpu["uuid"])
        environment["gpu_uuid_change_authorized_by_new_user_host"] = True
        base.write_json_new(output_dir / "environment.json", environment)
        static_inputs = verify_source_model(source_config)
        base.write_json_new(output_dir / "static_inputs.json", static_inputs)
        model, _tokenizer = base.load_model(source_config)
        first_ids = __import__("torch").tensor(
            [sealed_rows[0]["window_token_ids"]],
            dtype=__import__("torch").long,
            device="cuda",
        )
        base.run_native_capture(model, first_ids, source_config)
        __import__("torch").cuda.synchronize()
        recovered_cells = recover_frozen_cells(model, sealed_rows, source_config)
        recovered_by_key = {cell_key(row): row for row in recovered_cells}
        sealed_by_key = {str(row["cell_key"]): row for row in sealed_rows}
        cells: list[dict[str, Any]] = []
        for sealed in sealed_rows:
            key = str(sealed["cell_key"])
            cell = recovered_by_key.get(key)
            if cell is None:
                raise ProtocolError(f"frozen cell is absent from fresh replay: {key}")
            for field in (
                "victim_id",
                "document_index",
                "layer",
                "flat_token_idx",
                "target_hidden_sha256",
                "target_router_logits_sha256",
                "window_token_ids_sha256",
                "expert_ids",
                "gate_weights",
            ):
                if cell[field] != sealed[field]:
                    raise ProtocolError(f"fresh {field} differs for {key}")
            if args.layer is not None and int(cell["layer"]) != args.layer:
                raise ProtocolError(f"CLI layer assertion mismatches {key}")
            if args.expert is not None and any(
                int(expert_id) != args.expert for expert_id in cell["expert_ids"]
            ):
                raise ProtocolError(
                    f"CLI expert assertion does not match every action in {key}"
                )
            cells.append(cell)

        baselines: list[dict[str, Any]] = []
        action_plan: list[dict[str, Any]] = []
        hidden_by_cell: dict[str, Any] = {}
        for cell in cells:
            if time.time() - started > int(args.max_wall_seconds):
                raise TimeoutError("bridge run exceeded max wall time in baseline pass")
            sealed = sealed_by_key[cell_key(cell)]
            baseline, plans, hidden = baseline_and_plan(
                model, cell, sealed, source_config, config
            )
            baselines.append(baseline)
            action_plan.extend(plans)
            hidden_by_cell[str(sealed["cell_id"])] = hidden
        if len(action_plan) != expected_cells * int(
            config["proxy_ledger"]["expected_ranks"]
        ):
            raise ProtocolError("native action plan is not the full 33x8 surface")
        write_jsonl_new(output_dir / "native_baselines.jsonl", baselines)
        write_jsonl_new(output_dir / "native_action_plan.jsonl", action_plan)
        timing_plan = freeze_timing_plan(
            action_plan, int(config["timing"]["max_actions"])
        )
        timing_plan_payload = {
            "schema_version": "stablebatch-native-c8-timing-plan-v1",
            "status": "FROZEN_BEFORE_ANY_DIRECT_COST_TIMING_WARMUP_OR_EVENT",
            "selection": config["timing"]["selection"],
            "actions": timing_plan,
        }
        timing_plan_payload["deterministic_content_sha256"] = hashlib.sha256(
            base.canonical_json_bytes(timing_plan_payload)
        ).hexdigest()
        base.write_json_new(output_dir / "TIMING_PLAN.json", timing_plan_payload)

        baseline_by_cell = {str(row["cell_id"]): row for row in baselines}
        plan_by_id = {str(row["action_id"]): row for row in action_plan}
        cell_by_id = {
            str(sealed_by_key[cell_key(cell)]["cell_id"]): cell for cell in cells
        }
        action_rows: list[dict[str, Any]] = []
        cell_results: list[dict[str, Any]] = []
        for cell in cells:
            sealed = sealed_by_key[cell_key(cell)]
            cell_id = str(sealed["cell_id"])
            baseline = baseline_by_cell[cell_id]
            actions: dict[str, Any] = {}
            for rank in range(int(config["proxy_ledger"]["expected_ranks"])):
                if time.time() - started > int(args.max_wall_seconds):
                    raise TimeoutError("bridge run exceeded max wall time in action pass")
                plan = plan_by_id[f"{cell_id}-rank-{rank}"]
                action = run_action(
                    model, cell, sealed, baseline, plan, source_config, config
                )
                action_rows.append(action)
                actions[str(rank)] = action
            cell_results.append(
                {
                    "schema_version": "stablebatch-native-c8-replay-cell-v1",
                    "cell_id": cell_id,
                    "cell_key": str(sealed["cell_key"]),
                    "victim_id": str(cell["victim_id"]),
                    "document_index": int(cell["document_index"]),
                    "layer": int(cell["layer"]),
                    "frozen_m1_rank": int(sealed["frozen_m1_rank"]),
                    "native_baseline": baseline,
                    "actions": actions,
                    "integrity_status": "PASS",
                }
            )
        write_jsonl_new(output_dir / "action_results.jsonl", action_rows)
        write_jsonl_new(output_dir / "cell_results.jsonl", cell_results)

        action_by_id = {str(row["action_id"]): row for row in action_rows}
        timing_rows: list[dict[str, Any]] = []
        for item in timing_plan:
            action_id = str(item["action_id"])
            cell = cell_by_id[str(item["cell_id"])]
            measured = run_direct_timing(
                model,
                cell,
                hidden_by_cell[str(item["cell_id"])],
                int(item["rank"]),
                config,
            )
            timing_rows.append(
                {
                    "schema_version": "stablebatch-native-c8-direct-cost-v1",
                    **item,
                    **measured,
                }
            )
        write_jsonl_new(output_dir / "timing_results.jsonl", timing_rows)

        native_summary = native_metrics(cell_results)
        direct_cost = aggregate_timing(timing_rows, action_by_id)
        opportunity_routes = sum(
            int(row["distance_vs_proxy_R"]) for row in baselines
        )
        opportunity_cells = sum(
            int(int(row["distance_vs_proxy_R"]) > 0) for row in baselines
        )
        opportunity_docs = len(
            {
                int(row["document_index"])
                for row in baselines
                if int(row["distance_vs_proxy_R"]) > 0
            }
        )
        retained = proxy_only = native_only = exact_cells = 0
        for row in baselines:
            native_set = set(map(int, row["changed_layers_vs_proxy_R"]))
            proxy_set = set(map(int, row["proxy_unprotected_changed_layers_vs_R"]))
            retained += len(native_set & proxy_set)
            proxy_only += len(proxy_set - native_set)
            native_only += len(native_set - proxy_set)
            exact_cells += int(native_set == proxy_set)
        same_net = fraction_value(native_summary["same_rank"]["net"])
        random_net = fraction_value(native_summary["matched_random"]["net"])
        oracle_net = fraction_value(native_summary["abstaining_oracle"]["net"])
        same_denominator = fraction_value(proxy_summary["same_rank"]["net"])
        oracle_denominator = fraction_value(
            proxy_summary["abstaining_oracle"]["net"]
        )
        bridge_metrics = {
            "same_rank_bridge_transfer": float(same_net / same_denominator),
            "oracle_bridge_transfer": float(oracle_net / oracle_denominator),
            "native_specificity_gap": float(same_net - random_net),
        }
        classification = classify_bridge(
            opportunity_routes,
            native_summary,
            direct_cost,
            float(config["timing"]["clearly_high_relative_overhead"]),
        )
        summary = {
            "schema_version": "stablebatch-native-c8-replay-bridge-summary-v1",
            "status": "COMPLETE",
            "bridge_classification": classification,
            "cell_count": len(cell_results),
            "document_count": len(
                {int(row["document_index"]) for row in cell_results}
            ),
            "correctness_action_count": len(action_rows),
            "proxy_metrics": proxy_summary,
            "native_metrics": native_summary,
            "bridge_metrics": bridge_metrics,
            "native_opportunity": {
                "route_count": opportunity_routes,
                "cell_count": opportunity_cells,
                "document_count": opportunity_docs,
                "retained_proxy_opportunity": retained,
                "proxy_only_opportunity": proxy_only,
                "native_only_opportunity": native_only,
                "exact_baseline_cells": exact_cells,
            },
            "direct_cost": direct_cost,
            "direct_cost_high_threshold": float(
                config["timing"]["clearly_high_relative_overhead"]
            ),
            "research_boundary": config["research_boundary"],
            "completed_at": base.utc_now(),
            "wall_seconds": time.time() - started,
        }
        base.write_json_new(output_dir / "summary.json", summary)
        (output_dir / "report.md").write_text(
            build_report(summary), encoding="utf-8"
        )
        base.write_json_new(
            output_dir / "RUN_STATUS.json",
            {
                "status": "COMPLETE",
                "scientific_result_eligible": True,
                "bridge_classification": classification,
                "correctness_action_count": len(action_rows),
                "timed_action_count": len(timing_rows),
                "completed_at": base.utc_now(),
                "wall_seconds": time.time() - started,
            },
        )
        return 0
    except BaseException as error:
        failure = {
            "status": "FAILED",
            "scientific_result_eligible": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "failed_at": base.utc_now(),
            "wall_seconds": time.time() - started,
        }
        if not (output_dir / "FAILURE.json").exists():
            base.write_json_new(output_dir / "FAILURE.json", failure)
        if not (output_dir / "RUN_STATUS.json").exists():
            base.write_json_new(output_dir / "RUN_STATUS.json", failure)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
