#!/usr/bin/env python3
"""Run the frozen M1-positive-cohort fixed-C8 action-transfer Gate.

The treatment surface is exactly one zero-padded fixed-C8 contribution inside
the same seven-M64 background used by the earlier M1 oracle.  The runner
executes fresh R, U, and frozen-rank M1 closure arms before inspecting the C8
outcomes.  It is a bounded, retrospective action-transfer test, not an online
selector or a model-quality evaluation.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_observable_selector_pilot as observable  # noqa: E402
import run_oracle_action_sweep as oracle  # noqa: E402
import run_shape_lane_correctness_pilot as shape_lane  # noqa: E402
import run_single_contribution_pilot as base  # noqa: E402


ProtocolError = base.ProtocolError
RUNNER_RELATIVE = "docs/ideas/stablebatch/experiments/run_c8_action_transfer.py"
TEST_RELATIVE = "docs/ideas/stablebatch/experiments/test_c8_action_transfer.py"
RECOMPUTE_RELATIVE = (
    "docs/ideas/stablebatch/experiments/recompute_c8_action_transfer.py"
)
CONFIG_RELATIVE = (
    "docs/ideas/stablebatch/experiments/configs/c8_action_transfer_v1.json"
)
LOCK_RELATIVE = (
    "docs/ideas/stablebatch/experiments/configs/"
    "FROZEN_C8_ACTION_TRANSFER_LOCK_V1.json"
)
LOCK_SCHEMA = "stablebatch-c8-action-transfer-frozen-lock-v1"


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def cell_key(row: Mapping[str, Any]) -> str:
    return f"{row['victim_id']}|layer={int(row['layer']):02d}"


def fraction_payload(value: Fraction | int) -> dict[str, Any]:
    fraction = value if isinstance(value, Fraction) else Fraction(value, 1)
    return {
        "numerator": fraction.numerator,
        "denominator": fraction.denominator,
        "value": float(fraction),
    }


def fraction_from_payload(value: Mapping[str, Any]) -> Fraction:
    return Fraction(int(value["numerator"]), int(value["denominator"]))


def write_json_new(path: Path, value: Any) -> None:
    base.write_json_new(path, value)


def write_jsonl_stream(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())


def bound_path(
    repo_root: Path, section: Mapping[str, Any], key: str
) -> tuple[Path, str]:
    value = section[key]
    if isinstance(value, Mapping):
        relative = str(value["path"])
        expected = str(value["sha256"])
    else:
        relative = str(value)
        expected = str(section[f"{key}_sha256"])
    path = (repo_root / relative).resolve()
    if not path.is_file():
        raise ProtocolError(f"bound file is absent: {path}")
    observed = base.sha256_file(path)
    if observed != expected:
        raise ProtocolError(
            f"bound file hash mismatch for {key}: {observed} != {expected}"
        )
    return path, observed


def validate_lock(
    lock: Mapping[str, Any], config: Mapping[str, Any], repo_root: Path
) -> dict[str, str]:
    if lock.get("schema_version") != LOCK_SCHEMA:
        raise ProtocolError("wrong C8 transfer frozen-lock schema")
    if lock.get("status") != "FROZEN_PRE_RUN":
        raise ProtocolError("C8 transfer lock is not FROZEN_PRE_RUN")
    if lock.get("action_space") != config.get("action_space"):
        raise ProtocolError("C8 transfer action-space lock mismatch")
    if lock.get("thresholds") != config.get("thresholds"):
        raise ProtocolError("C8 transfer threshold lock mismatch")
    files = lock.get("files")
    if not isinstance(files, Mapping) or not files:
        raise ProtocolError("C8 transfer lock has no file bindings")
    observed: dict[str, str] = {}
    for relative, expected in files.items():
        path = (repo_root / str(relative)).resolve()
        if not path.is_file():
            raise ProtocolError(f"locked file is absent: {path}")
        digest = base.sha256_file(path)
        if digest != str(expected):
            raise ProtocolError(f"locked file mismatch for {relative}: {digest}")
        observed[str(relative)] = digest
    return observed


def verify_source_model_and_data(
    source_config: Mapping[str, Any], repo_root: Path
) -> dict[str, Any]:
    if source_config.get("status") != "FROZEN_PRE_RUN":
        raise ProtocolError("source config is not FROZEN_PRE_RUN")
    model_cfg = source_config["model"]
    model_root = Path(str(model_cfg["local_path"])).resolve()
    model_hashes: dict[str, str] = {}
    for relative, expected in model_cfg["file_sha256"].items():
        path = model_root / str(relative)
        if not path.is_file():
            raise ProtocolError(f"source model file is absent: {path}")
        observed = base.sha256_file(path)
        if observed != str(expected):
            raise ProtocolError(f"source model hash mismatch for {relative}")
        model_hashes[str(relative)] = observed
    manifest = (repo_root / str(source_config["data"]["manifest"])).resolve()
    expected_manifest = str(source_config["data"]["manifest_sha256"])
    if not manifest.is_file() or base.sha256_file(manifest) != expected_manifest:
        raise ProtocolError("source workload manifest hash mismatch")
    return {
        "model_path": str(model_root),
        "model_file_sha256": model_hashes,
        "workload_manifest": str(manifest),
        "workload_manifest_sha256": expected_manifest,
    }


def route_decomposition(
    unprotected_changed: Sequence[int], action_changed: Sequence[int]
) -> dict[str, Any]:
    old = set(map(int, unprotected_changed))
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


def select_primary_m1_rank(row: Mapping[str, Any]) -> int:
    candidates: list[dict[str, int]] = []
    for rank_text, action in row["actions"].items():
        if int(action["reward"]) <= 0:
            continue
        parts = route_decomposition(
            row["unprotected_changed_layers_vs_R"], action["changed_layers_vs_R"]
        )
        candidates.append(
            {
                "rank": int(rank_text),
                "recovered": int(parts["route_recovered_count"]),
                "harmed": int(parts["route_harmed_count"]),
            }
        )
    if not candidates:
        raise ProtocolError(f"positive oracle row has no positive rank: {cell_key(row)}")
    selected = min(
        candidates,
        key=lambda item: (-item["recovered"], item["harmed"], item["rank"]),
    )
    return int(selected["rank"])


def derive_source_cohort(
    oracle_rows: Sequence[Mapping[str, Any]], expected_cells: int, expected_docs: int
) -> list[dict[str, Any]]:
    positive = [row for row in oracle_rows if int(row["forced_oracle_reward"]) > 0]
    if len(positive) != expected_cells:
        raise ProtocolError(
            f"forced-positive cohort has {len(positive)} cells, expected {expected_cells}"
        )
    if any(row.get("integrity_status") != "PASS" for row in positive):
        raise ProtocolError("forced-positive cohort contains a failed source row")
    if any(row.get("selected_positive_action_confirmation") is None for row in positive):
        raise ProtocolError("forced-positive source action lacks confirmation")
    if len({cell_key(row) for row in positive}) != len(positive):
        raise ProtocolError("forced-positive cohort contains duplicate cells")
    if len({int(row["document_index"]) for row in positive}) != expected_docs:
        raise ProtocolError("forced-positive cohort has wrong document count")
    derived: list[dict[str, Any]] = []
    for row in sorted(positive, key=cell_key):
        rank = select_primary_m1_rank(row)
        if rank != int(row["forced_oracle_rank"]):
            raise ProtocolError(
                f"prescribed primary rank differs from source oracle at {cell_key(row)}"
            )
        action = row["actions"][str(rank)]
        parts = route_decomposition(
            row["unprotected_changed_layers_vs_R"], action["changed_layers_vs_R"]
        )
        derived.append(
            {
                "cell_key": cell_key(row),
                "victim_id": str(row["victim_id"]),
                "document_index": int(row["document_index"]),
                "layer": int(row["layer"]),
                "flat_token_idx": int(row["flat_token_idx"]),
                "frozen_m1_rank": rank,
                "m1_route_recovered_count": int(parts["route_recovered_count"]),
                "m1_route_harmed_count": int(parts["route_harmed_count"]),
                "m1_route_net_reward": int(parts["route_net_reward"]),
                "source_row": row,
            }
        )
    return derived


def _manifest_rank(cell: Mapping[str, Any]) -> int:
    primary = cell.get("m1_primary")
    if isinstance(primary, Mapping):
        return int(primary["rank"])
    for key in ("frozen_m1_rank", "frozen_rank", "rank"):
        if key in cell:
            return int(cell[key])
    raise ProtocolError("cohort manifest cell has no frozen M1 rank")


def validate_cohort_manifest(
    manifest: Mapping[str, Any],
    derived: Sequence[Mapping[str, Any]],
    expected_cells: int,
    expected_docs: int,
) -> None:
    cells = manifest.get("cells")
    if not isinstance(cells, list) or len(cells) != expected_cells:
        raise ProtocolError("cohort manifest has wrong cell count")
    by_key = {str(row["cell_key"]): row for row in cells}
    if len(by_key) != len(cells):
        raise ProtocolError("cohort manifest has duplicate cell keys")
    if len({int(row["document_index"]) for row in cells}) != expected_docs:
        raise ProtocolError("cohort manifest has wrong document count")
    for source in derived:
        key = str(source["cell_key"])
        if key not in by_key:
            raise ProtocolError(f"cohort manifest omitted source cell {key}")
        sealed = by_key[key]
        if _manifest_rank(sealed) != int(source["frozen_m1_rank"]):
            raise ProtocolError(f"cohort manifest rank mismatch for {key}")
        for field in ("victim_id", "document_index", "layer", "flat_token_idx"):
            if field in sealed and sealed[field] != source[field]:
                raise ProtocolError(f"cohort manifest {field} mismatch for {key}")
        primary = sealed.get("m1_primary")
        if isinstance(primary, Mapping):
            expected_fields = {
                "route_recovered_count": source["m1_route_recovered_count"],
                "route_harmed_count": source["m1_route_harmed_count"],
                "route_net_reward": source["m1_route_net_reward"],
            }
            for field, expected in expected_fields.items():
                if field in primary and int(primary[field]) != int(expected):
                    raise ProtocolError(f"cohort manifest {field} mismatch for {key}")
    digest = manifest.get("deterministic_content_sha256")
    if digest is not None:
        payload = {
            key: value
            for key, value in manifest.items()
            if key
            not in {
                "created_at",
                "sealed_at",
                "frozen_at",
                "deterministic_content_sha256",
            }
        }
        observed = hashlib.sha256(base.canonical_json_bytes(payload)).hexdigest()
        if observed != str(digest):
            raise ProtocolError("cohort deterministic content hash mismatch")


def final_mismatch_bitset(reference: Any, candidate: Any) -> dict[str, Any]:
    import torch

    left = reference.detach().cpu().contiguous()
    right = candidate.detach().cpu().contiguous()
    if left.dtype != right.dtype or tuple(left.shape) != tuple(right.shape):
        raise ProtocolError("final logits differ in dtype or shape")
    numel = int(left.numel())
    width = int(left.element_size())
    left_bytes = left.view(torch.uint8).reshape(numel, width)
    right_bytes = right.view(torch.uint8).reshape(numel, width)
    mismatch = (left_bytes != right_bytes).any(dim=1)
    packed = bytearray((numel + 7) // 8)
    for index in torch.nonzero(mismatch, as_tuple=False).reshape(-1).tolist():
        packed[int(index) // 8] |= 1 << (int(index) % 8)
    return {
        "encoding": "packed-bitset-lsb0-v1",
        "num_elements": numel,
        "dtype": str(left.dtype),
        "packed_hex": bytes(packed).hex(),
        "set_bit_count": int(mismatch.sum().item()),
        "vector_bitwise_mismatch": bool(mismatch.any().item()),
    }


def bitset_int(payload: Mapping[str, Any]) -> int:
    return int.from_bytes(bytes.fromhex(str(payload["packed_hex"])), "little")


def final_decomposition(
    unprotected: Mapping[str, Any], action: Mapping[str, Any]
) -> dict[str, int]:
    if int(unprotected["num_elements"]) != int(action["num_elements"]):
        raise ProtocolError("final-logit bitsets have different lengths")
    count = int(unprotected["num_elements"])
    valid_mask = (1 << count) - 1
    old = bitset_int(unprotected) & valid_mask
    new = bitset_int(action) & valid_mask
    recovered = old & ~new & valid_mask
    harmed = new & ~old & valid_mask
    persistent = old & new
    recovered_count = bin(recovered).count("1")
    harmed_count = bin(harmed).count("1")
    persistent_count = bin(persistent).count("1")
    return {
        "final_logit_recovered_count": recovered_count,
        "final_logit_harmed_count": harmed_count,
        "final_logit_persistent_count": persistent_count,
        "final_logit_net_reward": recovered_count - harmed_count,
    }


def precompute_c8_replacements(
    model: Any,
    cell: Mapping[str, Any],
    canonical_m: int,
    context: Mapping[str, Any],
    repeats: int,
) -> tuple[dict[int, Any], dict[str, Any]]:
    import torch

    if canonical_m != 8:
        raise ProtocolError("this Gate is frozen to canonical C=8")
    if context.get("kind") != "zero" or int(context.get("focal_slot", -1)) != 5:
        raise ProtocolError("this Gate is frozen to zero_pad_slot5")
    if repeats != 3:
        raise ProtocolError("this Gate requires exactly three C8 side-calls")
    focal = cell["_hidden_cpu"]
    lane_cpu, identities = shape_lane.build_lane_batch(
        focal, [], context, canonical_m
    )
    if identities.count("FOCAL") != 1:
        raise ProtocolError("C8 lane does not contain exactly one focal row")
    lane = lane_cpu.to(device="cuda", dtype=torch.bfloat16)
    slot = int(context["focal_slot"])
    replacements: dict[int, Any] = {}
    metadata: dict[str, Any] = {
        "canonical_m": canonical_m,
        "context": dict(context),
        "lane_input_sha256": base.tensor_sha256(lane_cpu),
        "lane_row_identities": identities,
        "repeats": repeats,
        "ranks": {},
    }
    for rank, expert_id in enumerate(map(int, cell["expert_ids"])):
        expert = model.model.layers[int(cell["layer"])].mlp.experts[expert_id]
        outputs: list[Any] = []
        hashes: list[str] = []
        for _repeat in range(repeats):
            with torch.inference_mode():
                output = expert(lane)[slot]
            outputs.append(output.detach().clone())
            hashes.append(base.tensor_sha256(output))
        if len(set(hashes)) != 1:
            raise ProtocolError(f"C8 side-call is unstable at {cell_key(cell)} rank {rank}")
        replacements[rank] = outputs[0]
        metadata["ranks"][str(rank)] = {
            "rank": rank,
            "expert_id": expert_id,
            "c8_sha256": hashes[0],
            "c8_sha256_by_repeat": hashes,
        }
    torch.cuda.synchronize()
    return replacements, metadata


def execute_replacement_arm(
    model: Any,
    input_ids: Any,
    cell: Mapping[str, Any],
    representative: Any,
    source_config: Mapping[str, Any],
    replacement_map: Mapping[int, Any],
    expected_hashes: Mapping[int, str],
    surface_descriptor: Mapping[int, int],
    native: Mapping[str, Any],
    noop_trace: Mapping[str, Any],
) -> dict[str, Any]:
    top_k = int(source_config["model"]["num_experts_per_tok"])
    if set(replacement_map) != set(range(top_k)):
        raise ProtocolError("replacement arm does not bind all target ranks")
    with observable.patched_topk_contributions(
        model, cell, replacement_map, "replacement"
    ) as trace:
        observation = base.run_observation(
            model, input_ids, source_config, representative
        )
    if trace["target_input_sha256"] != str(cell["target_hidden_sha256"]):
        raise ProtocolError("arm target input differs from sealed cell")
    if trace["target_router_logits_sha256"] != str(
        cell["target_router_logits_sha256"]
    ):
        raise ProtocolError("arm target router differs from sealed cell")
    if trace["target_selected_experts"] != list(map(int, cell["expert_ids"])):
        raise ProtocolError("arm target experts differ from sealed cell")
    for rank in range(top_k):
        key = str(rank)
        if int(trace["pair_match_count_by_rank"][key]) != 1:
            raise ProtocolError("arm missed a target rank")
        if int(trace["routing_weight_apply_count_by_rank"][key]) != 1:
            raise ProtocolError("arm did not apply one target gate weight")
        if trace["target_applied_raw_sha256_by_rank"][key] != str(
            expected_hashes[rank]
        ):
            raise ProtocolError("arm applied unexpected replacement hash")
        if trace["target_native_raw_sha256_by_rank"][key] != noop_trace[
            "target_native_raw_sha256_by_rank"
        ][key]:
            raise ProtocolError("arm native target raw output drifted")
        if trace["target_gate_weight_sha256_by_rank"][key] != noop_trace[
            "target_gate_weight_sha256_by_rank"
        ][key]:
            raise ProtocolError("arm target gate weight drifted")
    if trace["target_routing_weights_sha256"] != noop_trace[
        "target_routing_weights_sha256"
    ]:
        raise ProtocolError("arm target routing weights drifted")
    if trace["non_target_contributions_sha256"] != noop_trace[
        "non_target_contributions_sha256"
    ]:
        raise ProtocolError("arm changed non-target contributions")
    if trace["target_moe_output_sha256"] != observation["target_moe_output_sha256"]:
        raise ProtocolError("arm trace and observation target output disagree")
    if observation["input_ids_sha256"] != native["input_ids_sha256"]:
        raise ProtocolError("arm input IDs differ from native")
    if observation["attention_mask_sha256"] != native["attention_mask_sha256"]:
        raise ProtocolError("arm attention mask differs from native")
    for layer in range(int(cell["layer"]) + 1):
        if observation["router_logits_sha256_by_layer"][layer] != native[
            "router_logits_sha256_by_layer"
        ][layer]:
            raise ProtocolError(f"arm differs before intervention at layer {layer}")
    return {
        "surface_m_by_rank": {
            str(rank): int(surface_descriptor[rank]) for rank in range(top_k)
        },
        "surface_sha256_by_rank": {
            str(rank): str(expected_hashes[rank]) for rank in range(top_k)
        },
        "intervention_trace": dict(trace),
        "observation": observation,
    }


def arm_signature(row: Mapping[str, Any]) -> bytes:
    return base.canonical_json_bytes(
        {
            "compact_arm": oracle.compact_arm(row),
            "router_logits_sha256_by_layer": row["observation"][
                "router_logits_sha256_by_layer"
            ],
        }
    )


def assert_source_arm_closure(
    current: Mapping[str, Any], expected: Mapping[str, Any], label: str
) -> None:
    observed = oracle.compact_arm(current)
    if observed != expected:
        raise ProtocolError(f"fresh {label} arm differs from frozen oracle ledger")


def public_arm(
    raw: Mapping[str, Any],
    reference_logits: Any,
    reference_routes: Sequence[Sequence[int]],
    unprotected_changed: Sequence[int],
    start_layer: int,
) -> dict[str, Any]:
    changed = base.changed_membership_layers(
        reference_routes,
        raw["observation"]["topk_experts_by_layer"],
        start_layer,
    )
    final_bits = final_mismatch_bitset(
        reference_logits, raw["observation"]["_final_logits_cpu"]
    )
    return {
        "changed_layers_vs_R": changed,
        "distance_vs_R": len(changed),
        **route_decomposition(unprotected_changed, changed),
        "final_logits_mismatch_vs_R": final_bits,
        "arm": oracle.compact_arm(raw),
    }


def add_final_decomposition(
    unprotected: Mapping[str, Any], action: dict[str, Any]
) -> dict[str, Any]:
    return {
        **action,
        **final_decomposition(
            unprotected["final_logits_mismatch_vs_R"],
            action["final_logits_mismatch_vs_R"],
        ),
    }


def run_cell(
    model: Any,
    cell_index: int,
    cell: Mapping[str, Any],
    source_row: Mapping[str, Any],
    source_config: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    import torch

    top_k = int(source_config["model"]["num_experts_per_tok"])
    ranks = list(map(int, config["action_space"]["candidate_ranks"]))
    if ranks != list(range(top_k)):
        raise ProtocolError("C8 transfer action space must enumerate every rank")
    m1 = int(config["action_space"]["m1"])
    m64 = int(config["action_space"]["m64"])
    frozen_rank = int(source_row["forced_oracle_rank"])
    assignment = {
        "sidecall_m_order_per_rank": source_row["sidecall_m_order_per_rank"]
    }
    m_replacements, m_metadata = observable.precompute_cell_replacements(
        model, cell, assignment, source_config
    )
    c8_replacements, c8_metadata = precompute_c8_replacements(
        model,
        cell,
        int(config["action_space"]["canonical_m"]),
        config["action_space"]["c8_context"],
        int(config["action_space"]["c8_sidecall_repeats"]),
    )
    for rank in ranks:
        key = str(rank)
        if m_metadata["ranks"][key]["m1_sha256"] != source_row[
            "local_side_calls"
        ]["ranks"][key]["m1_sha256"]:
            raise ProtocolError("fresh M1 side-call differs from oracle ledger")
        if m_metadata["ranks"][key]["m64_sha256"] != source_row[
            "local_side_calls"
        ]["ranks"][key]["m64_sha256"]:
            raise ProtocolError("fresh M64 side-call differs from oracle ledger")
        c8_metadata["ranks"][key].update(
            {
                "equals_m1": c8_metadata["ranks"][key]["c8_sha256"]
                == m_metadata["ranks"][key]["m1_sha256"],
                "equals_m64": c8_metadata["ranks"][key]["c8_sha256"]
                == m_metadata["ranks"][key]["m64_sha256"],
            }
        )

    representative = base.PairIdentity(
        layer=int(cell["layer"]),
        flat_token_idx=int(cell["flat_token_idx"]),
        topk_rank=0,
        expert_id=int(cell["expert_ids"][0]),
    )
    input_ids = torch.tensor(
        [cell["window_token_ids"]], dtype=torch.long, device="cuda"
    )
    native = base.run_observation(model, input_ids, source_config, representative)
    with observable.patched_topk_contributions(
        model, cell, None, "self"
    ) as noop_trace:
        noop = base.run_observation(model, input_ids, source_config, representative)
    noop_checks = oracle.validate_noop(native, noop, noop_trace, cell, top_k)

    surfaces: dict[
        str, tuple[dict[int, Any], dict[int, str], dict[int, int]]
    ] = {}
    r_map = {rank: m_replacements[rank][m1] for rank in ranks}
    r_hash = {rank: m_metadata["ranks"][str(rank)]["m1_sha256"] for rank in ranks}
    u_map = {rank: m_replacements[rank][m64] for rank in ranks}
    u_hash = {rank: m_metadata["ranks"][str(rank)]["m64_sha256"] for rank in ranks}
    m1_map = dict(u_map)
    m1_hash = dict(u_hash)
    m1_map[frozen_rank] = m_replacements[frozen_rank][m1]
    m1_hash[frozen_rank] = m_metadata["ranks"][str(frozen_rank)]["m1_sha256"]
    surfaces["R"] = (r_map, r_hash, {rank: m1 for rank in ranks})
    surfaces["U"] = (u_map, u_hash, {rank: m64 for rank in ranks})
    surfaces["M1"] = (
        m1_map,
        m1_hash,
        {rank: (m1 if rank == frozen_rank else m64) for rank in ranks},
    )
    for rank in ranks:
        replacement_map = dict(u_map)
        expected_hashes = dict(u_hash)
        replacement_map[rank] = c8_replacements[rank]
        expected_hashes[rank] = c8_metadata["ranks"][str(rank)]["c8_sha256"]
        surfaces[f"C8-{rank}"] = (
            replacement_map,
            expected_hashes,
            {item: (8 if item == rank else m64) for item in ranks},
        )

    labels = list(surfaces)
    arm_order = oracle.deterministic_arm_order(
        cell_key(cell), labels, str(config["action_space"]["arm_order_seed"])
    )
    raw_arms: dict[str, dict[str, Any]] = {}
    for label in arm_order:
        replacement_map, expected_hashes, surface_descriptor = surfaces[label]
        raw_arms[label] = execute_replacement_arm(
            model,
            input_ids,
            cell,
            representative,
            source_config,
            replacement_map,
            expected_hashes,
            surface_descriptor,
            native,
            noop_trace,
        )

    assert_source_arm_closure(raw_arms["R"], source_row["reference_arm"], "R")
    assert_source_arm_closure(raw_arms["U"], source_row["unprotected_arm"], "U")
    assert_source_arm_closure(
        raw_arms["M1"], source_row["actions"][str(frozen_rank)]["arm"], "M1"
    )
    confirmation: dict[str, Any] | None = None
    if bool(config["action_space"].get("confirm_same_rank_repeat", True)):
        replacement_map, expected_hashes, surface_descriptor = surfaces[
            f"C8-{frozen_rank}"
        ]
        repeated = execute_replacement_arm(
            model,
            input_ids,
            cell,
            representative,
            source_config,
            replacement_map,
            expected_hashes,
            surface_descriptor,
            native,
            noop_trace,
        )
        first_signature = arm_signature(raw_arms[f"C8-{frozen_rank}"])
        second_signature = arm_signature(repeated)
        if first_signature != second_signature:
            raise ProtocolError("frozen-rank C8 full-forward repeat is unstable")
        confirmation = {
            "rank": frozen_rank,
            "status": "PASS",
            "signature_sha256": hashlib.sha256(first_signature).hexdigest(),
        }

    reference_routes = raw_arms["R"]["observation"]["topk_experts_by_layer"]
    reference_logits = raw_arms["R"]["observation"]["_final_logits_cpu"]
    start_layer = int(cell["layer"]) + 1
    current_u_changed = base.changed_membership_layers(
        reference_routes,
        raw_arms["U"]["observation"]["topk_experts_by_layer"],
        start_layer,
    )
    if current_u_changed != list(source_row["unprotected_changed_layers_vs_R"]):
        raise ProtocolError("fresh U route distance differs from oracle ledger")
    reference = public_arm(
        raw_arms["R"], reference_logits, reference_routes, current_u_changed, start_layer
    )
    unprotected = public_arm(
        raw_arms["U"], reference_logits, reference_routes, current_u_changed, start_layer
    )
    m1_arm = add_final_decomposition(
        unprotected,
        public_arm(
            raw_arms["M1"],
            reference_logits,
            reference_routes,
            current_u_changed,
            start_layer,
        ),
    )
    if m1_arm["changed_layers_vs_R"] != source_row["actions"][str(frozen_rank)][
        "changed_layers_vs_R"
    ]:
        raise ProtocolError("fresh M1 route result differs from oracle ledger")
    c8_actions: dict[str, dict[str, Any]] = {}
    for rank in ranks:
        action = public_arm(
            raw_arms[f"C8-{rank}"],
            reference_logits,
            reference_routes,
            current_u_changed,
            start_layer,
        )
        action = add_final_decomposition(unprotected, action)
        action.update(
            {
                "rank": rank,
                "expert_id": int(cell["expert_ids"][rank]),
                "is_frozen_m1_rank": rank == frozen_rank,
            }
        )
        c8_actions[str(rank)] = action

    return {
        "schema_version": "stablebatch-c8-action-transfer-cell-v1",
        "cell_index": cell_index,
        "cell_id": f"cell-{cell_index:03d}",
        **observable.public_cell(cell),
        "cell_key": cell_key(cell),
        "frozen_m1_rank": frozen_rank,
        "arm_order": arm_order,
        "m1_m64_side_calls": m_metadata,
        "c8_side_calls": c8_metadata,
        "native_noop_checks": noop_checks,
        "reference_arm": reference,
        "unprotected_arm": unprotected,
        "m1_same_rank_arm": m1_arm,
        "c8_actions": c8_actions,
        "same_rank_repeat_confirmation": confirmation,
        "integrity_status": "PASS",
    }


def action_metric_sum(
    actions: Sequence[Mapping[str, Any]], divisor: int = 1
) -> dict[str, Any]:
    fields = (
        "route_recovered_count",
        "route_harmed_count",
        "route_persistent_count",
        "route_net_reward",
        "distance_vs_R",
        "final_logit_recovered_count",
        "final_logit_harmed_count",
        "final_logit_persistent_count",
        "final_logit_net_reward",
    )
    result = {
        field: fraction_payload(Fraction(sum(int(row[field]) for row in actions), divisor))
        for field in fields
    }
    final_mismatch_bits = sum(
        int(row["final_logits_mismatch_vs_R"]["set_bit_count"]) for row in actions
    )
    final_mismatch_vectors = sum(
        int(bool(row["final_logits_mismatch_vs_R"]["vector_bitwise_mismatch"]))
        for row in actions
    )
    result.update(
        {
            "final_logit_mismatch_element_count": fraction_payload(
                Fraction(final_mismatch_bits, divisor)
            ),
            "final_logit_mismatch_vector_count": fraction_payload(
                Fraction(final_mismatch_vectors, divisor)
            ),
            "action_count": len(actions),
            "expectation_divisor": divisor,
        }
    )
    return result


def choose_c8_oracle_action(row: Mapping[str, Any]) -> Mapping[str, Any]:
    actions = list(row["c8_actions"].values())
    return min(
        actions,
        key=lambda item: (
            -int(item["route_net_reward"]),
            -int(item["route_recovered_count"]),
            int(item["route_harmed_count"]),
            int(item["rank"]),
        ),
    )


def core_aggregates(
    rows: Sequence[Mapping[str, Any]], ranks: Sequence[int]
) -> dict[str, Any]:
    m1 = [row["m1_same_rank_arm"] for row in rows]
    same = [row["c8_actions"][str(int(row["frozen_m1_rank"]))] for row in rows]
    all_c8 = [row["c8_actions"][str(rank)] for row in rows for rank in ranks]
    oracle_actions = [choose_c8_oracle_action(row) for row in rows]
    abstaining = [
        action for action in oracle_actions if int(action["route_net_reward"]) > 0
    ]
    baseline_route_distance = sum(
        int(row["unprotected_arm"]["distance_vs_R"]) for row in rows
    )
    baseline_final_elements = sum(
        int(row["unprotected_arm"]["final_logits_mismatch_vs_R"]["set_bit_count"])
        for row in rows
    )
    baseline_final_vectors = sum(
        int(
            bool(
                row["unprotected_arm"]["final_logits_mismatch_vs_R"][
                    "vector_bitwise_mismatch"
                ]
            )
        )
        for row in rows
    )
    return {
        "baseline": {
            "route_mismatch_count": fraction_payload(baseline_route_distance),
            "route_recovered_count": fraction_payload(0),
            "route_harmed_count": fraction_payload(0),
            "route_net_reward": fraction_payload(0),
            "final_logit_mismatch_element_count": fraction_payload(
                baseline_final_elements
            ),
            "final_logit_mismatch_vector_count": fraction_payload(
                baseline_final_vectors
            ),
            "action_count": 0,
        },
        "m1_same_rank": action_metric_sum(m1),
        "c8_same_rank": action_metric_sum(same),
        "c8_exact_uniform_random_rank": action_metric_sum(all_c8, len(ranks)),
        "c8_forced_best_rank_oracle": action_metric_sum(oracle_actions),
        "c8_abstaining_best_rank_oracle": action_metric_sum(abstaining),
    }


def single_cell_summary(
    row: Mapping[str, Any], ranks: Sequence[int]
) -> dict[str, Any]:
    core = core_aggregates([row], ranks)
    best = choose_c8_oracle_action(row)
    return {
        "cell_key": str(row["cell_key"]),
        "document_index": int(row["document_index"]),
        "frozen_m1_rank": int(row["frozen_m1_rank"]),
        "c8_best_rank": int(best["rank"]),
        "core": core,
    }


def decision_metrics(
    core: Mapping[str, Any],
    lodo: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    m1_recovered = fraction_from_payload(
        core["m1_same_rank"]["route_recovered_count"]
    )
    same_recovered = fraction_from_payload(
        core["c8_same_rank"]["route_recovered_count"]
    )
    same_net = fraction_from_payload(core["c8_same_rank"]["route_net_reward"])
    random_net = fraction_from_payload(
        core["c8_exact_uniform_random_rank"]["route_net_reward"]
    )
    oracle_net = fraction_from_payload(
        core["c8_forced_best_rank_oracle"]["route_net_reward"]
    )
    oracle_recovered = fraction_from_payload(
        core["c8_forced_best_rank_oracle"]["route_recovered_count"]
    )
    transfer = Fraction(same_recovered, m1_recovered) if m1_recovered else Fraction(0)
    oracle_transfer = (
        Fraction(oracle_recovered, m1_recovered) if m1_recovered else Fraction(0)
    )
    rank_gap = same_net - random_net
    oracle_gap = oracle_net - same_net
    lodo_positive = all(
        fraction_from_payload(item["core"]["c8_same_rank"]["route_net_reward"])
        > 0
        for item in lodo
    )
    lodo_rank_specific = all(
        fraction_from_payload(item["core"]["c8_same_rank"]["route_net_reward"])
        > fraction_from_payload(
            item["core"]["c8_exact_uniform_random_rank"]["route_net_reward"]
        )
        for item in lodo
    )
    low = Fraction(str(thresholds["low_transfer_ratio"]))
    go = Fraction(str(thresholds["go_transfer_ratio"]))
    same_harmed = fraction_from_payload(
        core["c8_same_rank"]["route_harmed_count"]
    )
    if (
        transfer >= go
        and rank_gap > 0
        and same_net > 0
        and same_harmed <= same_recovered
        and lodo_positive
        and lodo_rank_specific
    ):
        candidate = "GO_SHAPEABI_PLUS_STABILITYBUDGET"
    elif transfer <= low and oracle_transfer >= go and oracle_gap > 0:
        candidate = "CONDITIONAL_C8_SPECIFIC_RANK_SPACE"
    elif transfer <= low:
        candidate = "STOP_FIXED_C8_AS_QUALITY_ACTION"
    elif rank_gap <= 0 and same_net > 0:
        candidate = "CONDITIONAL_GLOBAL_SHAPEABI_NOT_SPARSE_SELECTOR"
    else:
        candidate = "CONDITIONAL_ONE_MECHANISM_DIAGNOSTIC_ONLY"
    return {
        "c8_transfer_ratio": fraction_payload(transfer),
        "rank_specificity_gap": fraction_payload(rank_gap),
        "c8_oracle_gap": fraction_payload(oracle_gap),
        "c8_oracle_transfer_ratio": fraction_payload(oracle_transfer),
        "lodo_same_rank_net_positive_all": lodo_positive,
        "lodo_rank_specificity_positive_all": lodo_rank_specific,
        "low_transfer_ratio": fraction_payload(low),
        "go_transfer_ratio": fraction_payload(go),
        "gate_candidate": candidate,
    }


def summarize_rows(
    rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    ranks = list(map(int, config["action_space"]["candidate_ranks"]))
    expected_cells = int(config["cohort"]["expected_unique_cells"])
    expected_docs = int(config["cohort"]["expected_documents"])
    if len(rows) != expected_cells:
        raise ProtocolError("C8 summary received wrong cell count")
    if len({str(row["cell_key"]) for row in rows}) != len(rows):
        raise ProtocolError("C8 summary received duplicate cells")
    if len({int(row["document_index"]) for row in rows}) != expected_docs:
        raise ProtocolError("C8 summary received wrong document count")
    if any(row.get("integrity_status") != "PASS" for row in rows):
        raise ProtocolError("C8 summary received a failed row")
    if any(set(map(int, row["c8_actions"])) != set(ranks) for row in rows):
        raise ProtocolError("C8 summary received an incomplete rank surface")
    core = core_aggregates(rows, ranks)
    all_actions = [row["c8_actions"][str(rank)] for row in rows for rank in ranks]
    action_level = {
        **action_metric_sum(all_actions),
        "positive_net_action_count": sum(
            int(int(action["route_net_reward"]) > 0) for action in all_actions
        ),
        "zero_net_action_count": sum(
            int(int(action["route_net_reward"]) == 0) for action in all_actions
        ),
        "negative_net_action_count": sum(
            int(int(action["route_net_reward"]) < 0) for action in all_actions
        ),
    }
    documents = sorted({int(row["document_index"]) for row in rows})
    per_document = [
        {
            "document_index": document,
            "cell_count": sum(
                int(int(row["document_index"]) == document) for row in rows
            ),
            "core": core_aggregates(
                [row for row in rows if int(row["document_index"]) == document],
                ranks,
            ),
        }
        for document in documents
    ]
    lodo = [
        {
            "left_out_document_index": document,
            "remaining_cell_count": sum(
                int(int(row["document_index"]) != document) for row in rows
            ),
            "core": core_aggregates(
                [row for row in rows if int(row["document_index"]) != document],
                ranks,
            ),
        }
        for document in documents
    ]
    return {
        "cell_count": len(rows),
        "document_count": len(documents),
        "candidate_action_count": len(all_actions),
        "core": core,
        "action_level": action_level,
        "unique_cell_level": [single_cell_summary(row, ranks) for row in rows],
        "per_document": per_document,
        "leave_one_document_out": lodo,
        "decision": decision_metrics(core, lodo, config["thresholds"]),
    }


def build_manifest(output_dir: Path, names: Sequence[str]) -> dict[str, Any]:
    files: dict[str, Any] = {}
    for name in names:
        path = output_dir / name
        files[name] = {
            "size_bytes": path.stat().st_size,
            "sha256": base.sha256_file(path),
        }
    return {
        "schema_version": "stablebatch-c8-action-transfer-output-manifest-v1",
        "files": files,
    }


def verify_manifest(output_dir: Path) -> None:
    manifest = base.load_json(output_dir / "MANIFEST.json")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or not files:
        raise ProtocolError("output manifest has no files")
    for name, binding in files.items():
        path = output_dir / str(name)
        if not path.is_file():
            raise ProtocolError(f"manifest-bound output is absent: {name}")
        if path.stat().st_size != int(binding["size_bytes"]):
            raise ProtocolError(f"manifest size mismatch: {name}")
        if base.sha256_file(path) != str(binding["sha256"]):
            raise ProtocolError(f"manifest hash mismatch: {name}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=HERE.parents[3])
    parser.add_argument("--config", type=Path, default=HERE / "configs/c8_action_transfer_v1.json")
    parser.add_argument(
        "--frozen-lock",
        type=Path,
        default=HERE / "configs/FROZEN_C8_ACTION_TRANSFER_LOCK_V1.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-wall-seconds", type=int, default=1800)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    runner_path = Path(__file__).resolve()
    config_path = args.config.resolve()
    lock_path = args.frozen_lock.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise ProtocolError(f"refusing to reuse output directory {output_dir}")
    if str(runner_path.relative_to(repo_root)) != RUNNER_RELATIVE:
        raise ProtocolError("C8 transfer runner path differs from frozen path")
    if str(config_path.relative_to(repo_root)) != CONFIG_RELATIVE:
        raise ProtocolError("C8 transfer config path differs from frozen path")
    if str(lock_path.relative_to(repo_root)) != LOCK_RELATIVE:
        raise ProtocolError("C8 transfer lock path differs from frozen path")
    config = base.load_json(config_path)
    lock = base.load_json(lock_path)
    if config.get("schema_version") != "stablebatch-c8-action-transfer-v1":
        raise ProtocolError("wrong C8 transfer config schema")
    if config.get("status") != "FROZEN_PRE_RUN":
        raise ProtocolError("C8 transfer config is not FROZEN_PRE_RUN")
    locked_files = validate_lock(lock, config, repo_root)

    source_config_path, source_config_sha = bound_path(
        repo_root, config, "source_config"
    )
    oracle_paths: dict[str, Path] = {}
    oracle_hashes: dict[str, str] = {}
    for key in ("cell_results", "summary", "config", "frozen_lock", "assignment_ledger"):
        oracle_paths[key], oracle_hashes[key] = bound_path(
            repo_root, config["oracle"], key
        )
    cohort_path, cohort_sha = bound_path(repo_root, config["cohort"], "path") if isinstance(config["cohort"].get("path"), Mapping) else ((repo_root / str(config["cohort"]["path"])).resolve(), str(config["cohort"]["sha256"]))
    if not cohort_path.is_file() or base.sha256_file(cohort_path) != cohort_sha:
        raise ProtocolError("cohort manifest hash mismatch")
    source_config = base.load_json(source_config_path)
    oracle_rows = base.load_jsonl(oracle_paths["cell_results"])
    oracle_summary = base.load_json(oracle_paths["summary"])
    oracle_config = base.load_json(oracle_paths["config"])
    assignment_ledger = base.load_json(oracle_paths["assignment_ledger"])
    cohort_manifest = base.load_json(cohort_path)
    expected_cells = int(config["cohort"]["expected_unique_cells"])
    expected_docs = int(config["cohort"]["expected_documents"])
    derived = derive_source_cohort(oracle_rows, expected_cells, expected_docs)
    validate_cohort_manifest(cohort_manifest, derived, expected_cells, expected_docs)
    if int(oracle_summary["positive_oracle_cell_count"]) != expected_cells:
        raise ProtocolError("oracle summary positive-cell closure mismatch")
    if oracle_config.get("status") != "FROZEN_PRE_RUN":
        raise ProtocolError("oracle source config is not frozen")
    assignment_by_key = {
        str(row["cell_key"]): row for row in assignment_ledger["cells"]
    }
    for item in derived:
        source_row = item["source_row"]
        assignment = assignment_by_key.get(str(item["cell_key"]))
        if assignment is None or assignment["sidecall_m_order_per_rank"] != source_row[
            "sidecall_m_order_per_rank"
        ]:
            raise ProtocolError("source assignment ledger closure mismatch")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    started = time.time()
    try:
        write_json_new(
            output_dir / "run_request.json",
            {
                "schema_version": "stablebatch-c8-action-transfer-request-v1",
                "started_at": base.utc_now(),
                "argv": sys.argv,
                "pid": os.getpid(),
                "runner_sha256": base.sha256_file(runner_path),
                "config_sha256": base.sha256_file(config_path),
                "lock_sha256": base.sha256_file(lock_path),
                "source_config_sha256": source_config_sha,
                "cohort_sha256": cohort_sha,
                "oracle_source_sha256": oracle_hashes,
                "repo_root": str(repo_root),
                "max_wall_seconds": int(args.max_wall_seconds),
                "git_head": base.command_output(
                    ["git", "-C", str(repo_root), "rev-parse", "HEAD"]
                ),
                "git_status_short": base.command_output(
                    ["git", "-C", str(repo_root), "status", "--short"]
                ),
            },
        )
        write_json_new(
            output_dir / "COHORT_LOCK.json",
            {
                "schema_version": "stablebatch-c8-action-transfer-cohort-lock-v1",
                "status": "SEALED_BEFORE_C8_OUTCOMES",
                "sealed_at": base.utc_now(),
                "cohort_manifest_path": str(cohort_path.relative_to(repo_root)),
                "cohort_manifest_sha256": cohort_sha,
                "oracle_cell_results_sha256": oracle_hashes["cell_results"],
                "cell_count": expected_cells,
                "document_count": expected_docs,
                "ordered_cells": [
                    {
                        "cell_key": item["cell_key"],
                        "frozen_m1_rank": item["frozen_m1_rank"],
                    }
                    for item in derived
                ],
                "result_rows_existed_at_seal": False,
            },
        )
        pre_import_gpu = base.gpu_snapshot()
        environment = base.verify_environment(source_config, pre_import_gpu)
        static_source = verify_source_model_and_data(source_config, repo_root)
        write_json_new(output_dir / "environment.json", environment)
        write_json_new(
            output_dir / "static_bindings.json",
            {
                "runner_path": str(runner_path),
                "runner_sha256": base.sha256_file(runner_path),
                "config_path": str(config_path),
                "config_sha256": base.sha256_file(config_path),
                "lock_path": str(lock_path),
                "lock_sha256": base.sha256_file(lock_path),
                "locked_files": locked_files,
                "source_config_path": str(source_config_path),
                "source_config_sha256": source_config_sha,
                "oracle_source_sha256": oracle_hashes,
                "cohort_path": str(cohort_path),
                "cohort_sha256": cohort_sha,
                **static_source,
            },
        )
        write_json_new(output_dir / "config_snapshot.json", config)
        write_json_new(output_dir / "source_config_snapshot.json", source_config)
        write_json_new(output_dir / "cohort_snapshot.json", cohort_manifest)

        model, tokenizer = base.load_model(source_config)
        workloads = base.load_workloads(source_config, repo_root, tokenizer)
        observable.verify_workload_digest(workloads, source_config)
        base.write_jsonl_new(output_dir / "workloads.jsonl", workloads)
        first_ids = __import__("torch").tensor(
            [workloads[0]["window_token_ids"]],
            dtype=__import__("torch").long,
            device="cuda",
        )
        observable.warmup_native_only(model, first_ids, source_config)
        scanned = observable.scan_observable_cells(model, workloads, source_config)
        base.write_jsonl_new(
            output_dir / "observable_cells.jsonl",
            (observable.public_cell(row) for row in scanned),
        )
        source_observable = repo_root / str(oracle_config["source"]["observable_cells"])
        if base.sha256_file(source_observable) != str(
            oracle_config["source"]["observable_cells_sha256"]
        ):
            raise ProtocolError("oracle observable-cell source hash drifted")
        if base.sha256_file(output_dir / "observable_cells.jsonl") != base.sha256_file(
            source_observable
        ):
            raise ProtocolError("fresh observable scan differs from oracle source")
        scanned_by_key = {cell_key(row): row for row in scanned}
        source_by_key = {cell_key(row): row for row in oracle_rows}
        cohort_cells: list[dict[str, Any]] = []
        for item in derived:
            key = str(item["cell_key"])
            if key not in scanned_by_key or key not in source_by_key:
                raise ProtocolError(f"cohort cell is absent from fresh scan: {key}")
            scanned_cell = scanned_by_key[key]
            source_row = source_by_key[key]
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
                if scanned_cell[field] != source_row[field]:
                    raise ProtocolError(f"fresh cell {field} differs for {key}")
            cohort_cells.append(scanned_cell)

        rows: list[dict[str, Any]] = []
        result_path = output_dir / "cell_results.jsonl"
        with result_path.open("x", encoding="utf-8") as stream:
            for index, cell in enumerate(cohort_cells):
                if time.time() - started > int(args.max_wall_seconds):
                    raise TimeoutError("C8 transfer run exceeded max wall time")
                row = run_cell(
                    model,
                    index,
                    cell,
                    source_by_key[cell_key(cell)],
                    source_config,
                    config,
                )
                rows.append(row)
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())

        summary = {
            "schema_version": "stablebatch-c8-action-transfer-summary-v1",
            "status": "COMPLETE",
            "evaluation_type": "m1_positive_cohort_c8_action_transfer",
            "research_boundary": config.get("research_boundary"),
            "cohort_lock_sha256": base.sha256_file(output_dir / "COHORT_LOCK.json"),
            "metrics": summarize_rows(rows, config),
            "wall_seconds": time.time() - started,
            "completed_at": base.utc_now(),
        }
        write_json_new(output_dir / "summary.json", summary)
        recompute_path = repo_root / RECOMPUTE_RELATIVE
        recompute_output = output_dir / "INDEPENDENT_RECOMPUTE.json"
        subprocess.run(
            [
                sys.executable,
                str(recompute_path),
                "--cell-results",
                str(result_path),
                "--summary",
                str(output_dir / "summary.json"),
                "--config",
                str(config_path),
                "--output",
                str(recompute_output),
            ],
            check=True,
        )
        recomputed = base.load_json(recompute_output)
        if recomputed.get("status") != "PASS" or int(
            recomputed.get("mismatch_count", -1)
        ) != 0:
            raise ProtocolError("independent C8 summary recompute failed")
        write_json_new(
            output_dir / "runtime_final.json", base.verify_final_runtime(source_config)
        )
        core_names = [
            "run_request.json",
            "COHORT_LOCK.json",
            "environment.json",
            "static_bindings.json",
            "config_snapshot.json",
            "source_config_snapshot.json",
            "cohort_snapshot.json",
            "workloads.jsonl",
            "observable_cells.jsonl",
            "cell_results.jsonl",
            "summary.json",
            "INDEPENDENT_RECOMPUTE.json",
            "runtime_final.json",
        ]
        write_json_new(output_dir / "MANIFEST.json", build_manifest(output_dir, core_names))
        status = {
            "status": "COMPLETE",
            "scientific_result_eligible": True,
            "gate_candidate": summary["metrics"]["decision"]["gate_candidate"],
            "completed_at": base.utc_now(),
            "wall_seconds": time.time() - started,
        }
        write_json_new(output_dir / "RUN_STATUS.json", status)
        verify_manifest(output_dir)
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
            write_json_new(output_dir / "FAILURE.json", failure)
        if not (output_dir / "RUN_STATUS.json").exists():
            write_json_new(output_dir / "RUN_STATUS.json", failure)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
