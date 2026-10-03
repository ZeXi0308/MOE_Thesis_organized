#!/usr/bin/env python3
"""StableBatch MaxGate-v1 versus matched-shuffle action-value pilot.

Each held-out victim/layer cell has one shared top-8 action surface.  R replaces
all eight raw expert contributions with M=1 side-call outputs; U uses M=64 for
all eight; O and S each protect exactly one rank with M=1 inside the identical
seven-M64 background.  MaxGate-v1 sees only current-layer gate weights.  The
    balanced shuffled rank, arm order, and side-call order are sealed before any
    M1/M64 side-call or intervention outcome is extracted.

This is a single-GPU, same-cell, offline action-value replay.  It is not a
dynamic batching controller, a serving benchmark, an EP experiment, or a
population-prevalence estimate.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import types
from typing import Any, Iterable, Mapping, Sequence


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_single_contribution_pilot as base  # noqa: E402


ProtocolError = base.ProtocolError
ARM_LABELS = ("R", "U", "O", "S")
RUNNER_RELATIVE = "docs/ideas/stablebatch/experiments/run_observable_selector_pilot.py"
CONFIG_RELATIVE = (
    "docs/ideas/stablebatch/experiments/configs/observable_selector_pilot_v1.json"
)
TEST_RELATIVE = "docs/ideas/stablebatch/experiments/test_observable_selector_pilot.py"
BASE_RUNNER_RELATIVE = (
    "docs/ideas/stablebatch/experiments/run_single_contribution_pilot.py"
)
LOCK_RELATIVE = (
    "docs/ideas/stablebatch/experiments/configs/"
    "FROZEN_OBSERVABLE_SELECTOR_LOCK_V1.json"
)
LOCK_SCHEMA = "stablebatch-observable-selector-frozen-lock-v1"


@dataclasses.dataclass(frozen=True)
class ObservableCellView:
    """The complete allowlisted selector input for one victim/layer cell."""

    gate_weights: tuple[float, ...]
    expert_ids: tuple[int, ...]


def cell_key(row: Mapping[str, Any]) -> str:
    return f"{row['victim_id']}|layer={int(row['layer']):02d}"


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def select_maxgate_rank(view: ObservableCellView) -> int:
    if len(view.gate_weights) != len(view.expert_ids) or not view.gate_weights:
        raise ProtocolError("MaxGate selector received malformed top-k inputs")
    if any(not math.isfinite(value) for value in view.gate_weights):
        raise ProtocolError("MaxGate selector received non-finite gate weight")
    return min(
        range(len(view.gate_weights)),
        key=lambda rank: (-view.gate_weights[rank], rank, view.expert_ids[rank]),
    )


def public_cell(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def verify_workload_digest(
    workloads: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> str:
    observed = sha256_text(
        "".join(str(row["window_token_ids_sha256"]) for row in workloads)
    )
    expected = str(config["data"]["ordered_window_hash_digest"])
    if observed != expected:
        raise ProtocolError(f"ordered held-out window digest {observed} != {expected}")
    if len({str(row["window_token_ids_sha256"]) for row in workloads}) != len(
        workloads
    ):
        raise ProtocolError("held-out windows are not unique")
    return observed


def expected_lock_files(config: Mapping[str, Any]) -> set[str]:
    return {
        RUNNER_RELATIVE,
        CONFIG_RELATIVE,
        TEST_RELATIVE,
        BASE_RUNNER_RELATIVE,
        str(config["data"]["manifest"]),
    }


def expected_frozen_semantics(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "independent_documents": len(config["data"]["document_indices"]),
        "window_level_fresh_only": True,
        "token_offset": int(config["data"]["token_offset"]),
        "window_tokens": int(config["data"]["window_tokens"]),
        "victim_position": int(config["data"]["victim_position"]),
        "target_layers": list(map(int, config["selection"]["target_layers"])),
        "cell_count": int(config["selection"]["cell_count"]),
        "actions_per_policy_per_cell": int(
            config["selection"]["actions_per_policy_per_cell"]
        ),
        "observable_selector": str(config["selection"]["observable_selector_name"]),
        "selector_allowed_signal_fields": list(
            config["selection"]["allowed_signal_fields"]
        ),
        "selector_identity_fields": list(
            config["selection"]["selector_identity_fields"]
        ),
        "shuffled_selector": str(config["selection"]["shuffle_name"]),
        "shuffle_rank_count_each": int(
            config["selection"]["shuffle_rank_count_each"]
        ),
        "baseline_m": int(config["intervention"]["baseline_m"]),
        "treatment_m": int(config["intervention"]["treatment_m"]),
        "repeats_per_arm": int(config["intervention"]["repeats_per_arm"]),
        "opportunity_min_cells": int(
            config["gate"]["opportunity_min_unprotected_divergent_cells"]
        ),
        "opportunity_min_distinct_victims": int(
            config["gate"]["opportunity_min_distinct_victims"]
        ),
        "support_min_observable_total_reward": int(
            config["gate"]["support_min_observable_total_reward"]
        ),
        "support_min_ratio_vs_shuffle_when_positive": float(
            config["gate"]["support_min_ratio_vs_shuffle_when_shuffle_positive"]
        ),
        "support_min_victims_observable_above_shuffle": int(
            config["gate"]["support_min_victims_with_observable_reward_above_shuffle"]
        ),
    }


def validate_selector_lock_document(
    lock: Mapping[str, Any], config: Mapping[str, Any]
) -> None:
    if lock.get("schema_version") != LOCK_SCHEMA:
        raise ProtocolError("wrong frozen-lock schema for observable-selector pilot")
    if lock.get("status") != "FROZEN_PRE_RUN":
        raise ProtocolError("observable-selector frozen lock is not FROZEN_PRE_RUN")
    files = lock.get("files")
    if not isinstance(files, dict) or set(map(str, files)) != expected_lock_files(config):
        raise ProtocolError("observable-selector frozen lock does not bind the exact file set")
    if lock.get("frozen_semantics") != expected_frozen_semantics(config):
        raise ProtocolError("observable-selector frozen semantics differ from config")
    if lock.get("claim_boundary") != config["research_boundary"]:
        raise ProtocolError("observable-selector frozen claim boundary differs from config")


def verify_selector_static_inputs(
    config: Mapping[str, Any],
    repo_root: Path,
    runner_path: Path,
    config_path: Path,
    lock_path: Path,
) -> dict[str, Any]:
    expected_paths = {
        "runner": RUNNER_RELATIVE,
        "config": CONFIG_RELATIVE,
        "lock": LOCK_RELATIVE,
        "base_runner": BASE_RUNNER_RELATIVE,
        "test": TEST_RELATIVE,
    }
    actual_paths = {
        "runner": runner_path,
        "config": config_path,
        "lock": lock_path,
        "base_runner": HERE / "run_single_contribution_pilot.py",
        "test": HERE / "test_observable_selector_pilot.py",
    }
    for role, path in actual_paths.items():
        try:
            relative = str(path.resolve().relative_to(repo_root.resolve()))
        except ValueError as error:
            raise ProtocolError(f"{role} path is outside repo root") from error
        if relative != expected_paths[role]:
            raise ProtocolError(
                f"{role} path {relative!r} != frozen path {expected_paths[role]!r}"
            )
    lock = base.load_json(lock_path)
    validate_selector_lock_document(lock, config)
    static = base.verify_static_inputs(
        config, repo_root, runner_path, config_path, lock_path
    )
    static["observable_selector_lock_contract"] = {
        "schema_version": LOCK_SCHEMA,
        "exact_files": sorted(expected_lock_files(config)),
        "frozen_semantics": expected_frozen_semantics(config),
        "claim_boundary": config["research_boundary"],
    }
    return static


def warmup_native_only(
    model: Any, input_ids: Any, config: Mapping[str, Any]
) -> None:
    import torch

    base.run_native_capture(model, input_ids, config)
    torch.cuda.synchronize()


def scan_observable_cells(
    model: Any,
    workloads: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Capture only current-layer selector fields; never compute M outcomes."""

    import torch

    model_cfg = config["model"]
    data_cfg = config["data"]
    target_layers = tuple(int(value) for value in config["selection"]["target_layers"])
    top_k = int(model_cfg["num_experts_per_tok"])
    hidden_size = int(model_cfg["hidden_size"])
    victim = int(data_cfg["victim_position"])
    cells: list[dict[str, Any]] = []
    for workload in workloads:
        input_ids = torch.tensor(
            [workload["window_token_ids"]], dtype=torch.long, device="cuda"
        )
        capture = base.run_native_capture(model, input_ids, config)
        output = capture["output"]
        for layer_idx in target_layers:
            block = model.model.layers[layer_idx].mlp
            flat_hidden = capture["moe_inputs"][layer_idx].reshape(-1, hidden_size)
            hidden = flat_hidden[victim]
            native_full_logits = output.router_logits[layer_idx].reshape(
                -1, int(model_cfg["num_experts"])
            )
            with torch.inference_mode():
                replay_logits = block.gate(flat_hidden)
            if not torch.equal(replay_logits, native_full_logits):
                raise ProtocolError(
                    f"{workload['victim_id']} layer {layer_idx} gate replay mismatch"
                )
            logits = native_full_logits[victim]
            weights, experts = base.topk_from_logits(logits, top_k)
            expert_ids = tuple(map(int, experts.detach().cpu().tolist()))
            gate_weights = tuple(map(float, weights.detach().cpu().tolist()))
            if len(set(expert_ids)) != top_k:
                raise ProtocolError("top-k expert identities are not unique")
            cells.append(
                {
                    "victim_id": workload["victim_id"],
                    "document_index": int(workload["document_index"]),
                    "window_token_ids": list(workload["window_token_ids"]),
                    "window_token_ids_sha256": workload["window_token_ids_sha256"],
                    "layer": layer_idx,
                    "flat_token_idx": victim,
                    "target_hidden_sha256": base.tensor_sha256(hidden),
                    "target_router_logits_sha256": base.tensor_sha256(logits),
                    "current_layer_topk_cutoff_margin": base.topk_margin(logits, top_k),
                    "gate_weights": list(gate_weights),
                    "expert_ids": list(expert_ids),
                    "_hidden_cpu": hidden.detach().cpu().clone(),
                }
            )
    expected = int(config["selection"]["cell_count"])
    if len(cells) != expected:
        raise ProtocolError(f"observable scan produced {len(cells)} cells, expected {expected}")
    if len({cell_key(row) for row in cells}) != len(cells):
        raise ProtocolError("observable cells are not unique")
    return cells


def build_assignment_ledger(
    cells: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    selection = config["selection"]
    forbidden = set(map(str, selection["forbidden_signal_fields"]))
    for row in cells:
        leaked = forbidden.intersection(row)
        if leaked:
            raise ProtocolError(f"selector cell contains forbidden fields: {sorted(leaked)}")

    shuffle_seed = str(selection["shuffle_seed"])
    shuffle_order = sorted(
        cells,
        key=lambda row: (sha256_text(f"{shuffle_seed}|{cell_key(row)}"), cell_key(row)),
    )
    shuffle_rank_by_cell = {
        cell_key(row): index % int(config["model"]["num_experts_per_tok"])
        for index, row in enumerate(shuffle_order)
    }
    schedule_seed = str(config["intervention"]["sidecall_schedule_seed"])
    schedule_order = sorted(
        cells,
        key=lambda row: (sha256_text(f"arm-order|{schedule_seed}|{cell_key(row)}"), cell_key(row)),
    )
    rotation_by_cell = {
        cell_key(row): index % len(ARM_LABELS)
        for index, row in enumerate(schedule_order)
    }

    rows: list[dict[str, Any]] = []
    repeats = int(config["intervention"]["repeats_per_arm"])
    if repeats != 3:
        raise ProtocolError("Balanced side-call schedule v1 requires exactly 3 repeats")
    m1 = int(config["intervention"]["baseline_m"])
    m64 = int(config["intervention"]["treatment_m"])
    for row in sorted(cells, key=lambda item: (str(item["victim_id"]), int(item["layer"]))):
        key = cell_key(row)
        view = ObservableCellView(
            gate_weights=tuple(map(float, row["gate_weights"])),
            expert_ids=tuple(map(int, row["expert_ids"])),
        )
        observable_rank = select_maxgate_rank(view)
        shuffled_rank = int(shuffle_rank_by_cell[key])
        rotation = int(rotation_by_cell[key])
        arm_orders: list[list[str]] = []
        for repeat in range(repeats):
            offset = (rotation + repeat) % len(ARM_LABELS)
            arm_orders.append(list(ARM_LABELS[offset:] + ARM_LABELS[:offset]))
        reverse_sidecalls = int(sha256_text(f"sidecall|{schedule_seed}|{key}")[-1], 16) % 2 == 1
        sidecall_order = [m1, m64, m64, m1, m1, m64]
        if reverse_sidecalls:
            sidecall_order = list(reversed(sidecall_order))
        rows.append(
            {
                **public_cell(row),
                "cell_key": key,
                "selector_input": {
                    "gate_weights": list(view.gate_weights),
                    "expert_ids": list(view.expert_ids),
                },
                "observable_rank": observable_rank,
                "observable_expert_id": view.expert_ids[observable_rank],
                "shuffled_rank": shuffled_rank,
                "shuffled_expert_id": view.expert_ids[shuffled_rank],
                "shuffle_identity_hash": sha256_text(f"{shuffle_seed}|{key}"),
                "arm_orders_by_repeat": arm_orders,
                "sidecall_m_order_per_rank": sidecall_order,
            }
        )

    top_k = int(config["model"]["num_experts_per_tok"])
    shuffle_counts = [sum(int(row["shuffled_rank"]) == rank for row in rows) for rank in range(top_k)]
    expected_each = int(selection["shuffle_rank_count_each"])
    if shuffle_counts != [expected_each] * top_k:
        raise ProtocolError(f"shuffled rank balance mismatch: {shuffle_counts}")
    if any(int(row["observable_rank"]) != 0 for row in rows):
        raise ProtocolError("MaxGate-v1 did not select top-k rank 0")
    if len(rows) != int(selection["cell_count"]):
        raise ProtocolError("assignment ledger cell count mismatch")

    work_signature = {
        "schema_version": "stablebatch-cell-action-work-signature-v1",
        "cells": len(rows),
        "actions_per_policy_per_cell": 1,
        "target_token_topk_contributions": top_k,
        "observable_surface_m_multiset": [m1] + [m64] * (top_k - 1),
        "shuffled_surface_m_multiset": [m1] + [m64] * (top_k - 1),
        "sidecall_m1_per_rank": repeats,
        "sidecall_m64_per_rank": repeats,
        "full_forward_repeats_per_arm": repeats,
        "full_forward_input_batch": 1,
        "full_forward_sequence_tokens": int(config["data"]["window_tokens"]),
        "hidden_size": int(config["model"]["hidden_size"]),
        "intermediate_size": int(config["model"]["intermediate_size"]),
        "dtype": str(config["environment"]["dtype"]),
        "attention_mask": "all_ones",
        "use_cache": False,
    }
    if work_signature["observable_surface_m_multiset"] != work_signature[
        "shuffled_surface_m_multiset"
    ]:
        raise ProtocolError("observable and shuffled work surfaces differ")
    deterministic_content = {
        "schema_version": "stablebatch-observable-selector-assignment-ledger-v1",
        "selector": str(selection["observable_selector_name"]),
        "selector_allowed_signal_fields": list(selection["allowed_signal_fields"]),
        "selector_identity_fields": list(selection["selector_identity_fields"]),
        "selector_forbidden_signal_fields": list(selection["forbidden_signal_fields"]),
        "shuffle": str(selection["shuffle_name"]),
        "shuffle_seed": shuffle_seed,
        "shuffle_rank_counts": shuffle_counts,
        "work_signature": work_signature,
        "work_signature_sha256": hashlib.sha256(
            base.canonical_json_bytes(work_signature)
        ).hexdigest(),
        "cells": rows,
    }
    return {
        **deterministic_content,
        "created_at": base.utc_now(),
        "assignment_content_sha256": hashlib.sha256(
            base.canonical_json_bytes(deterministic_content)
        ).hexdigest(),
    }


def precompute_cell_replacements(
    model: Any,
    cell: Mapping[str, Any],
    assignment: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[dict[int, dict[int, Any]], dict[str, Any]]:
    import torch

    layer = int(cell["layer"])
    hidden = cell["_hidden_cpu"].to(device="cuda", dtype=torch.bfloat16)
    if base.tensor_sha256(hidden) != str(cell["target_hidden_sha256"]):
        raise ProtocolError("cell hidden hash changed before side-calls")
    m1 = int(config["intervention"]["baseline_m"])
    m64 = int(config["intervention"]["treatment_m"])
    schedule = list(map(int, assignment["sidecall_m_order_per_rank"]))
    expected_repeats = int(config["intervention"]["repeats_per_arm"])
    if schedule.count(m1) != expected_repeats or schedule.count(m64) != expected_repeats:
        raise ProtocolError(f"invalid side-call schedule: {schedule}")

    replacements: dict[int, dict[int, Any]] = {}
    metadata: dict[str, Any] = {"sidecall_m_order_per_rank": schedule, "ranks": {}}
    for rank, expert_id in enumerate(map(int, cell["expert_ids"])):
        expert = model.model.layers[layer].mlp.experts[expert_id]
        outputs: dict[int, list[Any]] = {m1: [], m64: []}
        hashes: dict[int, list[str]] = {m1: [], m64: []}
        for m_value in schedule:
            with torch.inference_mode():
                output = expert(hidden.reshape(1, -1).repeat(m_value, 1))[0]
            outputs[m_value].append(output.detach().clone())
            hashes[m_value].append(base.tensor_sha256(output))
        if len(set(hashes[m1])) != 1 or len(set(hashes[m64])) != 1:
            raise ProtocolError(
                f"same-M side-call unstable at {cell_key(cell)} rank {rank}: {hashes}"
            )
        replacements[rank] = {m1: outputs[m1][0], m64: outputs[m64][0]}
        changed = base.bitwise_changed_elements(outputs[m1][0], outputs[m64][0])
        l2 = float(
            torch.linalg.vector_norm(outputs[m64][0].float() - outputs[m1][0].float()).item()
        )
        metadata["ranks"][str(rank)] = {
            "rank": rank,
            "expert_id": expert_id,
            "m1_sha256_by_repeat": hashes[m1],
            "m64_sha256_by_repeat": hashes[m64],
            "m1_sha256": hashes[m1][0],
            "m64_sha256": hashes[m64][0],
            "changed_bf16_elements": changed,
            "local_l2": l2,
        }
    torch.cuda.synchronize()
    metadata["changed_rank_count"] = sum(
        int(row["changed_bf16_elements"] > 0) for row in metadata["ranks"].values()
    )
    return replacements, metadata


@contextlib.contextmanager
def patched_topk_contributions(
    model: Any,
    cell: Mapping[str, Any],
    replacements_by_rank: Mapping[int, Any] | None,
    mode: str,
):
    """Copy native OLMoE combine and replace every target-token top-k row."""

    import torch
    import torch.nn.functional as F

    if mode not in {"self", "replacement"}:
        raise ValueError(mode)
    top_k = len(cell["expert_ids"])
    if mode == "replacement" and set(replacements_by_rank or {}) != set(range(top_k)):
        raise ProtocolError("replacement arm must bind every target top-k rank")
    layer = int(cell["layer"])
    token_idx = int(cell["flat_token_idx"])
    expert_ids = tuple(map(int, cell["expert_ids"]))
    if len(set(expert_ids)) != top_k:
        raise ProtocolError("target top-k experts must be unique")
    rank_by_expert = {expert_id: rank for rank, expert_id in enumerate(expert_ids)}
    block = model.model.layers[layer].mlp
    original_forward = block.forward
    trace: dict[str, Any] = {}

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
        routing_weights = routing_weights.to(flat_hidden.dtype)
        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=flat_hidden.dtype,
            device=flat_hidden.device,
        )
        expert_mask = F.one_hot(
            selected_experts, num_classes=this.num_experts
        ).permute(2, 1, 0)
        pair_counts = {rank: 0 for rank in range(top_k)}
        native_raw: dict[int, Any] = {}
        applied_raw: dict[int, Any] = {}
        gate_weights: dict[int, Any] = {}
        non_target_hasher = hashlib.sha256()
        for expert_idx in range(this.num_experts):
            expert_layer = this.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])
            current_state = flat_hidden[None, top_x].reshape(-1, hidden_dim)
            raw_outputs = expert_layer(current_state)
            local_index: int | None = None
            target_rank = rank_by_expert.get(expert_idx)
            if target_rank is not None:
                matches = torch.where(
                    (top_x == token_idx) & (idx == target_rank)
                )[0]
                pair_counts[target_rank] += int(matches.numel())
                if matches.numel() == 1:
                    local_index = int(matches[0].item())
                    native = raw_outputs[local_index].detach().clone()
                    native_raw[target_rank] = native
                    gate_weights[target_rank] = routing_weights[
                        token_idx, target_rank
                    ].detach().clone()
                    applied = (
                        native
                        if mode == "self"
                        else replacements_by_rank[target_rank].to(
                            device=raw_outputs.device, dtype=raw_outputs.dtype
                        )
                    )
                    if tuple(applied.shape) != (hidden_dim,):
                        raise ProtocolError(
                            f"replacement rank {target_rank} shape {tuple(applied.shape)}"
                        )
                    raw_outputs = raw_outputs.clone()
                    raw_outputs[local_index] = applied
                    applied_raw[target_rank] = applied.detach().clone()
            hashable = raw_outputs.detach().clone()
            if local_index is not None:
                hashable[local_index].zero_()
            non_target_hasher.update(int(expert_idx).to_bytes(4, "little"))
            non_target_hasher.update(base.tensor_bytes(idx))
            non_target_hasher.update(base.tensor_bytes(top_x))
            non_target_hasher.update(base.tensor_bytes(hashable))
            current_hidden_states = raw_outputs * routing_weights[top_x, idx, None]
            final_hidden_states.index_add_(
                0, top_x, current_hidden_states.to(flat_hidden.dtype)
            )
        if any(pair_counts[rank] != 1 for rank in range(top_k)):
            raise ProtocolError(f"target top-k pair counts are {pair_counts}")
        final_hidden_states = final_hidden_states.reshape(
            batch_size, sequence_length, hidden_dim
        )
        victim_logits = router_logits.reshape(-1, router_logits.shape[-1])[token_idx]
        victim_weights, victim_experts = base.topk_from_logits(victim_logits, this.top_k)
        trace.update(
            {
                "pair_match_count_by_rank": {str(k): v for k, v in pair_counts.items()},
                "routing_weight_apply_count_by_rank": {
                    str(rank): 1 for rank in range(top_k)
                },
                "layer": layer,
                "flat_token_idx": token_idx,
                "target_input_sha256": base.tensor_sha256(flat_hidden[token_idx]),
                "target_router_logits_sha256": base.tensor_sha256(victim_logits),
                "target_selected_experts": victim_experts.detach().cpu().tolist(),
                "target_routing_weights_sha256": base.tensor_sha256(victim_weights),
                "target_native_raw_sha256_by_rank": {
                    str(rank): base.tensor_sha256(native_raw[rank]) for rank in range(top_k)
                },
                "target_applied_raw_sha256_by_rank": {
                    str(rank): base.tensor_sha256(applied_raw[rank]) for rank in range(top_k)
                },
                "target_gate_weight_sha256_by_rank": {
                    str(rank): base.tensor_sha256(gate_weights[rank]) for rank in range(top_k)
                },
                "non_target_contributions_sha256": non_target_hasher.hexdigest(),
                "target_moe_output_sha256": base.tensor_sha256(final_hidden_states),
            }
        )
        return final_hidden_states, router_logits

    block.forward = types.MethodType(injected_forward, block)
    try:
        yield trace
    finally:
        block.forward = original_forward


def surface_for_arm(
    arm: str,
    observable_rank: int,
    shuffled_rank: int,
    top_k: int,
    baseline_m: int = 1,
    treatment_m: int = 64,
) -> dict[int, int]:
    if arm == "R":
        return {rank: baseline_m for rank in range(top_k)}
    if arm == "U":
        return {rank: treatment_m for rank in range(top_k)}
    if arm == "O":
        return {
            rank: (baseline_m if rank == observable_rank else treatment_m)
            for rank in range(top_k)
        }
    if arm == "S":
        return {
            rank: (baseline_m if rank == shuffled_rank else treatment_m)
            for rank in range(top_k)
        }
    raise ValueError(arm)


def multi_arm_signature(row: Mapping[str, Any]) -> bytes:
    observation = row["observation"]
    trace = row["intervention_trace"]
    return base.canonical_json_bytes(
        {
            "surface_m_by_rank": row["surface_m_by_rank"],
            "input_ids_sha256": observation["input_ids_sha256"],
            "attention_mask_sha256": observation["attention_mask_sha256"],
            "target_input_sha256": observation["target_input_sha256"],
            "target_router_logits_sha256": observation["target_router_logits_sha256"],
            "target_moe_output_sha256": observation["target_moe_output_sha256"],
            "router_logits_sha256_by_layer": observation["router_logits_sha256_by_layer"],
            "topk_experts_by_layer": observation["topk_experts_by_layer"],
            "final_logits_sha256": observation["final_logits_sha256"],
            "greedy_token_id": observation["greedy_token_id"],
            "target_native_raw_sha256_by_rank": trace[
                "target_native_raw_sha256_by_rank"
            ],
            "target_applied_raw_sha256_by_rank": trace[
                "target_applied_raw_sha256_by_rank"
            ],
            "non_target_contributions_sha256": trace[
                "non_target_contributions_sha256"
            ],
        }
    )


def run_cell(
    model: Any,
    cell_index: int,
    cell: Mapping[str, Any],
    assignment: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    import torch

    top_k = int(config["model"]["num_experts_per_tok"])
    m1 = int(config["intervention"]["baseline_m"])
    m64 = int(config["intervention"]["treatment_m"])
    observable_rank = int(assignment["observable_rank"])
    shuffled_rank = int(assignment["shuffled_rank"])
    representative = base.PairIdentity(
        layer=int(cell["layer"]),
        flat_token_idx=int(cell["flat_token_idx"]),
        topk_rank=0,
        expert_id=int(cell["expert_ids"][0]),
    )
    input_ids = torch.tensor(
        [cell["window_token_ids"]], dtype=torch.long, device="cuda"
    )
    replacements, local = precompute_cell_replacements(
        model, cell, assignment, config
    )

    native = base.run_observation(model, input_ids, config, representative)
    if native["target_input_sha256"] != cell["target_hidden_sha256"]:
        raise ProtocolError("native cell input differs from sealed selector input")
    if native["target_router_logits_sha256"] != cell["target_router_logits_sha256"]:
        raise ProtocolError("native cell router differs from sealed selector input")
    with patched_topk_contributions(model, cell, None, "self") as noop_trace:
        noop = base.run_observation(model, input_ids, config, representative)
    noop_checks = {
        "input_equal": native["input_ids_sha256"] == noop["input_ids_sha256"],
        "attention_mask_equal": native["attention_mask_sha256"]
        == noop["attention_mask_sha256"],
        "target_input_equal": native["target_input_sha256"]
        == noop["target_input_sha256"],
        "target_router_equal": native["target_router_logits_sha256"]
        == noop["target_router_logits_sha256"],
        "target_moe_output_equal": native["target_moe_output_sha256"]
        == noop["target_moe_output_sha256"],
        "all_routes_equal": native["topk_experts_by_layer"]
        == noop["topk_experts_by_layer"],
        "final_logits_equal": native["final_logits_sha256"]
        == noop["final_logits_sha256"],
    }
    if not all(noop_checks.values()):
        raise ProtocolError(f"multi-replacement native no-op failed: {noop_checks}")
    if noop_trace["target_moe_output_sha256"] != noop[
        "target_moe_output_sha256"
    ]:
        raise ProtocolError("no-op trace and observation disagree on MoE output")
    if noop_trace["target_selected_experts"] != list(map(int, cell["expert_ids"])):
        raise ProtocolError("no-op top-k identity differs from assignment ledger")
    for rank in range(top_k):
        key = str(rank)
        if noop_trace["pair_match_count_by_rank"][key] != 1:
            raise ProtocolError("no-op did not match every target rank exactly once")
        if noop_trace["target_native_raw_sha256_by_rank"][key] != noop_trace[
            "target_applied_raw_sha256_by_rank"
        ][key]:
            raise ProtocolError("self replacement changed a target raw output")

    repeats = int(config["intervention"]["repeats_per_arm"])
    arm_rows: dict[str, list[dict[str, Any]]] = {label: [] for label in ARM_LABELS}
    for repeat in range(repeats):
        order = list(map(str, assignment["arm_orders_by_repeat"][repeat]))
        if sorted(order) != sorted(ARM_LABELS):
            raise ProtocolError(f"invalid arm order {order}")
        for arm in order:
            surface = surface_for_arm(
                arm, observable_rank, shuffled_rank, top_k, m1, m64
            )
            replacement_map = {
                rank: replacements[rank][m_value]
                for rank, m_value in surface.items()
            }
            with patched_topk_contributions(
                model, cell, replacement_map, "replacement"
            ) as trace:
                observation = base.run_observation(
                    model, input_ids, config, representative
                )
            if trace["target_input_sha256"] != cell["target_hidden_sha256"]:
                raise ProtocolError("arm target input differs from sealed selector input")
            if trace["target_router_logits_sha256"] != cell[
                "target_router_logits_sha256"
            ]:
                raise ProtocolError("arm target router differs from sealed selector input")
            if trace["target_selected_experts"] != list(map(int, cell["expert_ids"])):
                raise ProtocolError("arm target experts differ from selector ledger")
            for rank in range(top_k):
                key = str(rank)
                if trace["pair_match_count_by_rank"][key] != 1:
                    raise ProtocolError("arm target rank was not matched exactly once")
                if trace["routing_weight_apply_count_by_rank"][key] != 1:
                    raise ProtocolError("arm routing weight was not applied exactly once")
                expected_hash = local["ranks"][key][
                    "m1_sha256" if surface[rank] == m1 else "m64_sha256"
                ]
                if trace["target_applied_raw_sha256_by_rank"][key] != expected_hash:
                    raise ProtocolError("arm applied raw output differs from side-call")
                if trace["target_native_raw_sha256_by_rank"][key] != noop_trace[
                    "target_native_raw_sha256_by_rank"
                ][key]:
                    raise ProtocolError("arm native raw output differs from no-op")
                if trace["target_gate_weight_sha256_by_rank"][key] != noop_trace[
                    "target_gate_weight_sha256_by_rank"
                ][key]:
                    raise ProtocolError("arm gate weight differs from no-op")
            if trace["target_routing_weights_sha256"] != noop_trace[
                "target_routing_weights_sha256"
            ]:
                raise ProtocolError("arm routing weights differ from no-op")
            if trace["non_target_contributions_sha256"] != noop_trace[
                "non_target_contributions_sha256"
            ]:
                raise ProtocolError("arm non-target contributions differ from no-op")
            if trace["target_moe_output_sha256"] != observation[
                "target_moe_output_sha256"
            ]:
                raise ProtocolError("arm trace and observation disagree on MoE output")
            if observation["input_ids_sha256"] != native["input_ids_sha256"]:
                raise ProtocolError("arm input IDs differ from native")
            if observation["attention_mask_sha256"] != native["attention_mask_sha256"]:
                raise ProtocolError("arm attention mask differs from native")
            for layer_idx in range(int(cell["layer"]) + 1):
                if observation["router_logits_sha256_by_layer"][layer_idx] != native[
                    "router_logits_sha256_by_layer"
                ][layer_idx]:
                    raise ProtocolError(
                        f"arm differs before intervention at layer {layer_idx}"
                    )
            arm_rows[arm].append(
                {
                    "repeat": repeat,
                    "execution_position": order.index(arm),
                    "surface_m_by_rank": {str(k): v for k, v in surface.items()},
                    "intervention_trace": dict(trace),
                    "observation": observation,
                }
            )

    for arm, rows in arm_rows.items():
        if len({multi_arm_signature(row) for row in rows}) != 1:
            raise ProtocolError(f"same-arm full-forward output unstable for {arm}")
    expected_os_surface = sorted([m1] + [m64] * (top_k - 1))
    if sorted(map(int, arm_rows["O"][0]["surface_m_by_rank"].values())) != expected_os_surface:
        raise ProtocolError("observable work surface is not 1xM1 + 7xM64")
    if sorted(map(int, arm_rows["S"][0]["surface_m_by_rank"].values())) != expected_os_surface:
        raise ProtocolError("shuffled work surface is not 1xM1 + 7xM64")
    if observable_rank == shuffled_rank:
        for repeat in range(repeats):
            if multi_arm_signature(arm_rows["O"][repeat]) != multi_arm_signature(
                arm_rows["S"][repeat]
            ):
                raise ProtocolError("O/S differ despite selecting the same rank")

    reference_routes = arm_rows["R"][0]["observation"]["topk_experts_by_layer"]
    start_layer = int(cell["layer"]) + 1
    changed_by_arm: dict[str, list[list[int]]] = {}
    for arm in ("U", "O", "S"):
        changed_by_arm[arm] = [
            base.changed_membership_layers(
                reference_routes,
                arm_rows[arm][repeat]["observation"]["topk_experts_by_layer"],
                start_layer,
            )
            for repeat in range(repeats)
        ]
        if len({tuple(value) for value in changed_by_arm[arm]}) != 1:
            raise ProtocolError(f"{arm}/R route difference is unstable")
        if changed_by_arm[arm][0] and arm_rows[arm][0]["observation"][
            "target_moe_output_sha256"
        ] == arm_rows["R"][0]["observation"]["target_moe_output_sha256"]:
            raise ProtocolError(f"{arm} route changed without target MoE combine change")

    d_u = len(changed_by_arm["U"][0])
    d_o = len(changed_by_arm["O"][0])
    d_s = len(changed_by_arm["S"][0])
    reward_o = d_u - d_o
    reward_s = d_u - d_s
    serial_arms: dict[str, Any] = {}
    for arm, rows in arm_rows.items():
        serial_arms[arm] = [
            {
                "repeat": row["repeat"],
                "execution_position": row["execution_position"],
                "surface_m_by_rank": row["surface_m_by_rank"],
                "intervention_trace": row["intervention_trace"],
                "observation": base.public_observation(row["observation"]),
            }
            for row in rows
        ]
    return {
        "cell_index": cell_index,
        "cell_id": f"cell-{cell_index:03d}",
        **public_cell(cell),
        "assignment": {
            "observable_rank": observable_rank,
            "observable_expert_id": int(cell["expert_ids"][observable_rank]),
            "shuffled_rank": shuffled_rank,
            "shuffled_expert_id": int(cell["expert_ids"][shuffled_rank]),
            "arm_orders_by_repeat": assignment["arm_orders_by_repeat"],
            "sidecall_m_order_per_rank": assignment["sidecall_m_order_per_rank"],
        },
        "local_side_calls": local,
        "native_noop_checks": noop_checks,
        "native_observation": base.public_observation(native),
        "noop_observation": base.public_observation(noop),
        "noop_intervention_trace": dict(noop_trace),
        "arms": serial_arms,
        "changed_downstream_membership_layers_vs_R_by_arm_and_repeat": changed_by_arm,
        "d_unprotected_vs_R": d_u,
        "d_observable_vs_R": d_o,
        "d_shuffled_vs_R": d_s,
        "observable_reward": reward_o,
        "shuffled_reward": reward_s,
        "observable_full_restoration": bool(d_u > 0 and d_o == 0),
        "shuffled_full_restoration": bool(d_u > 0 and d_s == 0),
        "observable_harm": bool(reward_o < 0),
        "shuffled_harm": bool(reward_s < 0),
        "observable_shuffled_same_rank": bool(observable_rank == shuffled_rank),
        "integrity_status": "PASS",
    }


def classify_results(
    rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    gate = config["gate"]
    opportunity = [row for row in rows if int(row["d_unprotected_vs_R"]) > 0]
    opportunity_victims = sorted({str(row["victim_id"]) for row in opportunity})
    a_o = sum(int(row["observable_reward"]) for row in rows)
    a_s = sum(int(row["shuffled_reward"]) for row in rows)
    per_victim: dict[str, dict[str, int]] = {}
    for row in rows:
        victim = str(row["victim_id"])
        per_victim.setdefault(victim, {"observable_reward": 0, "shuffled_reward": 0})
        per_victim[victim]["observable_reward"] += int(row["observable_reward"])
        per_victim[victim]["shuffled_reward"] += int(row["shuffled_reward"])
    victims_o_above_s = sorted(
        victim
        for victim, values in per_victim.items()
        if values["observable_reward"] > values["shuffled_reward"]
    )
    opportunity_pass = len(opportunity) >= int(
        gate["opportunity_min_unprotected_divergent_cells"]
    ) and len(opportunity_victims) >= int(gate["opportunity_min_distinct_victims"])
    absolute_pass = a_o >= int(gate["support_min_observable_total_reward"])
    ratio_threshold = (
        math.ceil(float(gate["support_min_ratio_vs_shuffle_when_shuffle_positive"]) * a_s)
        if a_s > 0
        else int(gate["support_min_observable_total_reward"])
    )
    ratio_pass = a_o >= ratio_threshold
    victim_pass = len(victims_o_above_s) >= int(
        gate["support_min_victims_with_observable_reward_above_shuffle"]
    )
    if not opportunity_pass:
        verdict = "UNABLE_TO_DECIDE_INSUFFICIENT_OPPORTUNITY"
    elif a_o <= a_s:
        verdict = "WEAKENS_MAXGATE_V1_NOT_BETTER_THAN_SHUFFLE"
    elif absolute_pass and ratio_pass and victim_pass:
        verdict = "SUPPORT_MAXGATE_V1_ACTION_VALUE"
    else:
        verdict = "UNABLE_TO_DECIDE_BELOW_FROZEN_MAGNITUDE_OR_COVERAGE"
    return {
        "verdict": verdict,
        "cell_count": len(rows),
        "independent_document_count": len({str(row["victim_id"]) for row in rows}),
        "opportunity_cell_count": len(opportunity),
        "opportunity_distinct_victim_count": len(opportunity_victims),
        "opportunity_victims": opportunity_victims,
        "observable_total_reward": a_o,
        "shuffled_total_reward": a_s,
        "observable_reward_per_action": a_o / len(rows),
        "shuffled_reward_per_action": a_s / len(rows),
        "observable_positive_tie_negative": {
            "positive": sum(int(row["observable_reward"]) > 0 for row in rows),
            "tie": sum(int(row["observable_reward"]) == 0 for row in rows),
            "negative": sum(int(row["observable_reward"]) < 0 for row in rows),
        },
        "shuffled_positive_tie_negative": {
            "positive": sum(int(row["shuffled_reward"]) > 0 for row in rows),
            "tie": sum(int(row["shuffled_reward"]) == 0 for row in rows),
            "negative": sum(int(row["shuffled_reward"]) < 0 for row in rows),
        },
        "observable_full_restoration_count": sum(
            bool(row["observable_full_restoration"]) for row in rows
        ),
        "shuffled_full_restoration_count": sum(
            bool(row["shuffled_full_restoration"]) for row in rows
        ),
        "observable_harm_count": sum(bool(row["observable_harm"]) for row in rows),
        "shuffled_harm_count": sum(bool(row["shuffled_harm"]) for row in rows),
        "observable_shuffled_same_rank_cell_count": sum(
            bool(row["observable_shuffled_same_rank"]) for row in rows
        ),
        "per_victim_rewards": per_victim,
        "victims_observable_reward_above_shuffle": victims_o_above_s,
        "victims_observable_reward_above_shuffle_count": len(victims_o_above_s),
        "frozen_ratio_threshold": ratio_threshold,
        "gate_checks": {
            "opportunity_pass": opportunity_pass,
            "observable_absolute_reward_pass": absolute_pass,
            "observable_ratio_pass": ratio_pass,
            "victim_coverage_pass": victim_pass,
        },
    }


def assert_native_stability_across_cells(rows: Sequence[Mapping[str, Any]]) -> None:
    by_victim: dict[str, bytes] = {}
    for row in rows:
        observation = row["native_observation"]
        signature = base.canonical_json_bytes(
            {
                "input_ids_sha256": observation["input_ids_sha256"],
                "attention_mask_sha256": observation["attention_mask_sha256"],
                "router_logits_sha256_by_layer": observation[
                    "router_logits_sha256_by_layer"
                ],
                "topk_experts_by_layer": observation["topk_experts_by_layer"],
                "final_logits_sha256": observation["final_logits_sha256"],
                "greedy_token_id": observation["greedy_token_id"],
            }
        )
        victim = str(row["victim_id"])
        previous = by_victim.setdefault(victim, signature)
        if previous != signature:
            raise ProtocolError(f"native trajectory drifted across cells for {victim}")


def validate_formal_rows(
    rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> None:
    expected = int(config["selection"]["cell_count"])
    if len(rows) != expected:
        raise ProtocolError(f"formal result has {len(rows)} cells, expected {expected}")
    identities = {(str(row["victim_id"]), int(row["layer"])) for row in rows}
    if len(identities) != expected:
        raise ProtocolError("formal result contains duplicate victim/layer cells")
    if any(row.get("integrity_status") != "PASS" for row in rows):
        raise ProtocolError("formal result contains a non-PASS cell")


def json_artifact_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8")


def build_manifest(
    output_dir: Path, pending_run_status: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(output_dir.iterdir()):
        if not path.is_file() or path.name in {"MANIFEST.json", "RUN_STATUS.json"}:
            continue
        files[path.name] = {
            "size_bytes": path.stat().st_size,
            "sha256": base.sha256_file(path),
        }
    if pending_run_status is not None:
        if (output_dir / "RUN_STATUS.json").exists():
            raise ProtocolError("RUN_STATUS exists before success manifest is sealed")
        status_bytes = json_artifact_bytes(pending_run_status)
        files["RUN_STATUS.json"] = {
            "size_bytes": len(status_bytes),
            "sha256": hashlib.sha256(status_bytes).hexdigest(),
        }
    return {
        "schema_version": "stablebatch-observable-selector-manifest-v1",
        "created_at": base.utc_now(),
        "files": files,
    }


def verify_output_manifest(output_dir: Path) -> dict[str, Any]:
    manifest_path = output_dir / "MANIFEST.json"
    manifest = base.load_json(manifest_path)
    if manifest.get("schema_version") != "stablebatch-observable-selector-manifest-v1":
        raise ProtocolError("unexpected observable-selector output manifest schema")
    expected = manifest.get("files")
    if not isinstance(expected, dict):
        raise ProtocolError("output manifest has no files map")
    actual_names = {
        path.name
        for path in output_dir.iterdir()
        if path.is_file() and path.name != "MANIFEST.json"
    }
    if actual_names != set(map(str, expected)):
        raise ProtocolError("output manifest file set does not match directory")
    for name, binding in expected.items():
        path = output_dir / str(name)
        if path.stat().st_size != int(binding["size_bytes"]):
            raise ProtocolError(f"output artifact size mismatch for {name}")
        if base.sha256_file(path) != str(binding["sha256"]):
            raise ProtocolError(f"output artifact hash mismatch for {name}")
    return manifest


def write_bound_run_status(
    output_dir: Path, run_status: Mapping[str, Any]
) -> None:
    final_path = output_dir / "RUN_STATUS.json"
    pending_path = output_dir / ".RUN_STATUS.json.pending"
    if final_path.exists() or pending_path.exists():
        raise ProtocolError("refusing to reuse a run-status path")
    base.write_json_new(pending_path, run_status)
    expected = hashlib.sha256(json_artifact_bytes(run_status)).hexdigest()
    if base.sha256_file(pending_path) != expected:
        raise ProtocolError("pending RUN_STATUS bytes differ from sealed manifest binding")
    os.rename(pending_path, final_path)


def verify_acceptance_evidence(
    acceptance_dir: Path,
    runner_path: Path,
    base_runner_path: Path,
    config_path: Path,
    lock_path: Path,
) -> dict[str, Any]:
    acceptance_dir = acceptance_dir.resolve()
    if not acceptance_dir.is_dir():
        raise ProtocolError(f"acceptance evidence directory is absent: {acceptance_dir}")
    verify_output_manifest(acceptance_dir)
    status = base.load_json(acceptance_dir / "RUN_STATUS.json")
    acceptance = base.load_json(acceptance_dir / "REAL_GPU_ACCEPTANCE.json")
    request = base.load_json(acceptance_dir / "run_request.json")
    static = base.load_json(acceptance_dir / "static_bindings.json")
    if status.get("status") != "COMPLETE_ACCEPTANCE_ONLY" or status.get(
        "scientific_result_eligible"
    ) is not False:
        raise ProtocolError("acceptance RUN_STATUS is not a completed smoke-only result")
    if acceptance.get("status") != "PASS" or acceptance.get("integrity_status") != "PASS":
        raise ProtocolError("real-GPU acceptance did not pass")
    expected_hashes = {
        "runner_sha256": base.sha256_file(runner_path),
        "base_runner_sha256": base.sha256_file(base_runner_path),
        "config_sha256": base.sha256_file(config_path),
        "frozen_lock_sha256": base.sha256_file(lock_path),
    }
    if request.get("acceptance_only") is not True:
        raise ProtocolError("acceptance run_request is not acceptance-only")
    for field, expected_hash in expected_hashes.items():
        if request.get(field) != expected_hash:
            raise ProtocolError(f"acceptance binding differs for {field}")
    if static.get("frozen_lock_sha256") != expected_hashes["frozen_lock_sha256"]:
        raise ProtocolError("acceptance static lock binding differs")
    return {
        "acceptance_dir": str(acceptance_dir),
        "manifest_sha256": base.sha256_file(acceptance_dir / "MANIFEST.json"),
        "run_status_sha256": base.sha256_file(acceptance_dir / "RUN_STATUS.json"),
        "real_gpu_acceptance_sha256": base.sha256_file(
            acceptance_dir / "REAL_GPU_ACCEPTANCE.json"
        ),
        **expected_hashes,
    }


def run_acceptance(
    model: Any,
    workloads: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    output_dir: Path,
) -> None:
    import torch

    input_ids = torch.tensor(
        [workloads[0]["window_token_ids"]], dtype=torch.long, device="cuda"
    )
    warmup_native_only(model, input_ids, config)
    cells = scan_observable_cells(model, workloads[:1], {
        **config,
        "selection": {**config["selection"], "cell_count": 15},
    })
    cell = cells[0]
    observable_rank = select_maxgate_rank(
        ObservableCellView(
            tuple(map(float, cell["gate_weights"])),
            tuple(map(int, cell["expert_ids"])),
        )
    )
    assignment = {
        "observable_rank": observable_rank,
        "shuffled_rank": (observable_rank + 1) % int(config["model"]["num_experts_per_tok"]),
        "arm_orders_by_repeat": [
            list(ARM_LABELS[index:] + ARM_LABELS[:index])
            for index in range(int(config["intervention"]["repeats_per_arm"]))
        ],
        "sidecall_m_order_per_rank": [
            int(config["intervention"]["baseline_m"]),
            int(config["intervention"]["treatment_m"]),
            int(config["intervention"]["treatment_m"]),
            int(config["intervention"]["baseline_m"]),
            int(config["intervention"]["baseline_m"]),
            int(config["intervention"]["treatment_m"]),
        ],
    }
    row = run_cell(model, 0, cell, assignment, config)
    base.write_json_new(
        output_dir / "REAL_GPU_ACCEPTANCE.json",
        {
            "schema_version": "stablebatch-observable-selector-acceptance-v1",
            "status": "PASS",
            "evidence_boundary": "harness_and_one_real_same_cell_smoke_only_not_scientific_result",
            "victim_id": row["victim_id"],
            "layer": row["layer"],
            "observable_rank": row["assignment"]["observable_rank"],
            "shuffled_rank": row["assignment"]["shuffled_rank"],
            "d_unprotected_vs_R": row["d_unprotected_vs_R"],
            "observable_reward": row["observable_reward"],
            "shuffled_reward": row["shuffled_reward"],
            "integrity_status": row["integrity_status"],
        },
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frozen-lock", type=Path, required=True)
    parser.add_argument("--acceptance-only", action="store_true")
    parser.add_argument("--acceptance-dir", type=Path)
    parser.add_argument("--max-wall-seconds", type=int, default=7200)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    runner_path = Path(__file__).resolve()
    config_path = args.config.resolve()
    lock_path = args.frozen_lock.resolve()
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise ProtocolError(f"refusing to reuse output directory {output_dir}")
    base_runner_path = HERE / "run_single_contribution_pilot.py"
    for role, path in {
        "runner": runner_path,
        "base runner": base_runner_path,
        "config": config_path,
        "frozen lock": lock_path,
    }.items():
        if not path.is_file():
            raise ProtocolError(f"{role} is absent before run start: {path}")
    if args.acceptance_only and args.acceptance_dir is not None:
        raise ProtocolError("--acceptance-dir is forbidden with --acceptance-only")
    if not args.acceptance_only and args.acceptance_dir is None:
        raise ProtocolError("formal mode requires --acceptance-dir")
    config = base.load_json(config_path)
    acceptance_binding = None
    if not args.acceptance_only:
        acceptance_binding = verify_acceptance_evidence(
            args.acceptance_dir,
            runner_path,
            base_runner_path,
            config_path,
            lock_path,
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    started = time.time()
    try:
        base.write_json_new(
            output_dir / "run_request.json",
            {
                "schema_version": "stablebatch-observable-selector-run-request-v1",
                "started_at": base.utc_now(),
                "argv": sys.argv,
                "pid": os.getpid(),
                "acceptance_only": bool(args.acceptance_only),
                "acceptance_binding": acceptance_binding,
                "max_wall_seconds": args.max_wall_seconds,
                "runner_path": str(runner_path),
                "runner_sha256": base.sha256_file(runner_path),
                "base_runner_path": str(base_runner_path),
                "base_runner_sha256": base.sha256_file(base_runner_path),
                "config_path": str(config_path),
                "config_sha256": base.sha256_file(config_path),
                "frozen_lock_path": str(lock_path),
                "frozen_lock_sha256": base.sha256_file(lock_path),
                "repo_root": str(repo_root),
                "git_head": base.command_output(
                    ["git", "-C", str(repo_root), "rev-parse", "HEAD"]
                ),
                "git_status_short": base.command_output(
                    ["git", "-C", str(repo_root), "status", "--short"]
                ),
            },
        )
        pre_import_gpu = base.gpu_snapshot()
        environment = base.verify_environment(config, pre_import_gpu)
        static = verify_selector_static_inputs(
            config, repo_root, runner_path, config_path, lock_path
        )
        base.write_json_new(output_dir / "environment.json", environment)
        base.write_json_new(output_dir / "static_bindings.json", static)
        base.write_json_new(output_dir / "config_snapshot.json", config)
        model, tokenizer = base.load_model(config)
        workloads = base.load_workloads(config, repo_root, tokenizer)
        workload_digest = verify_workload_digest(workloads, config)
        base.write_jsonl_new(output_dir / "workloads.jsonl", workloads)
        if args.acceptance_only:
            run_acceptance(model, workloads, config, output_dir)
            base.write_json_new(
                output_dir / "runtime_final.json", base.verify_final_runtime(config)
            )
            run_status = {
                "status": "COMPLETE_ACCEPTANCE_ONLY",
                "scientific_result_eligible": False,
                "completed_at": base.utc_now(),
                "wall_seconds": time.time() - started,
            }
            base.write_json_new(
                output_dir / "MANIFEST.json",
                build_manifest(output_dir, pending_run_status=run_status),
            )
            write_bound_run_status(output_dir, run_status)
            verify_output_manifest(output_dir)
            return 0

        first_ids = __import__("torch").tensor(
            [workloads[0]["window_token_ids"]],
            dtype=__import__("torch").long,
            device="cuda",
        )
        warmup_native_only(model, first_ids, config)
        cells = scan_observable_cells(model, workloads, config)
        candidate_path = output_dir / "observable_cells.jsonl"
        base.write_jsonl_new(candidate_path, (public_cell(row) for row in cells))
        ledger = build_assignment_ledger(cells, config)
        ledger_path = output_dir / "assignment_ledger.json"
        base.write_json_new(ledger_path, ledger)
        base.write_json_new(
            output_dir / "POLICY_SELECTION_LOCK.json",
            {
                "schema_version": "stablebatch-policy-selection-lock-v1",
                "status": (
                    "SEALED_BEFORE_M1_M64_SIDECALLS_AND_"
                    "INTERVENTION_OUTCOME_EXTRACTION"
                ),
                "sealed_at": base.utc_now(),
                "observable_cells_sha256": base.sha256_file(candidate_path),
                "assignment_ledger_sha256": base.sha256_file(ledger_path),
                "assignment_content_sha256": ledger["assignment_content_sha256"],
                "ordered_window_hash_digest": workload_digest,
                "selector_allowed_signal_fields": config["selection"][
                    "allowed_signal_fields"
                ],
                "selector_identity_fields": config["selection"][
                    "selector_identity_fields"
                ],
                "selector_forbidden_signal_fields": config["selection"][
                    "forbidden_signal_fields"
                ],
                "target_results_existed_at_seal": False,
            },
        )
        assignment_by_key = {
            str(row["cell_key"]): row for row in ledger["cells"]
        }
        result_rows: list[dict[str, Any]] = []
        result_path = output_dir / "cell_results.jsonl"
        with result_path.open("x", encoding="utf-8") as stream:
            for cell_index, cell in enumerate(
                sorted(cells, key=lambda row: (str(row["victim_id"]), int(row["layer"])))
            ):
                if time.time() - started > args.max_wall_seconds:
                    raise TimeoutError("selector pilot exceeded --max-wall-seconds")
                row = run_cell(
                    model,
                    cell_index,
                    cell,
                    assignment_by_key[cell_key(cell)],
                    config,
                )
                result_rows.append(row)
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
        assert_native_stability_across_cells(result_rows)
        validate_formal_rows(result_rows, config)
        summary = {
            "schema_version": "stablebatch-observable-selector-summary-v1",
            "status": "COMPLETE",
            "evaluation_type": config["evaluation_type"],
            "evidence_boundary": config["research_boundary"],
            "ordered_window_hash_digest": workload_digest,
            "acceptance_binding": acceptance_binding,
            "policy_selection_lock_sha256": base.sha256_file(
                output_dir / "POLICY_SELECTION_LOCK.json"
            ),
            "work_signature_sha256": ledger["work_signature_sha256"],
            **classify_results(result_rows, config),
            "support_rule": config["interpretation"]["support"],
            "weakening_rule": config["interpretation"]["weakens"],
            "unable_rule": config["interpretation"]["unable"],
            "all_cell_integrity_pass": all(
                row["integrity_status"] == "PASS" for row in result_rows
            ),
            "wall_seconds": time.time() - started,
            "completed_at": base.utc_now(),
        }
        base.write_json_new(output_dir / "summary.json", summary)
        base.write_json_new(
            output_dir / "runtime_final.json", base.verify_final_runtime(config)
        )
        run_status = {
            "status": "COMPLETE",
            "scientific_result_eligible": True,
            "verdict": summary["verdict"],
            "acceptance_manifest_sha256": acceptance_binding["manifest_sha256"],
            "completed_at": base.utc_now(),
            "wall_seconds": time.time() - started,
        }
        base.write_json_new(
            output_dir / "MANIFEST.json",
            build_manifest(output_dir, pending_run_status=run_status),
        )
        write_bound_run_status(output_dir, run_status)
        verify_output_manifest(output_dir)
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
