#!/usr/bin/env python3
"""Exploratory D10 fixed-C=8 shape-lane correctness gate.

This runner asks one narrow question on the already enriched StableBatch cells:
does a fixed expert-row shape make a focal row and its downstream trajectory
independent of real companion identity, focal slot, zero padding, and expert
call order?  It does not measure continuous-decode prevalence, queueing,
serving latency, vLLM Batch Invariance, EP, or production behavior.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Mapping, Sequence


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_observable_selector_pilot as observable  # noqa: E402
import run_single_contribution_pilot as base  # noqa: E402


ProtocolError = base.ProtocolError


def cell_key(row: Mapping[str, Any]) -> str:
    return f"{row['victim_id']}|layer={int(row['layer']):02d}"


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_bound_file(repo_root: Path, binding: Mapping[str, Any]) -> Path:
    path = (repo_root / str(binding["path"])).resolve()
    if not path.is_file():
        raise ProtocolError(f"bound file is absent: {path}")
    observed = base.sha256_file(path)
    if observed != str(binding["sha256"]):
        raise ProtocolError(f"bound file hash mismatch: {path}")
    return path


def load_cells(target_path: Path, config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    targets = base.load_jsonl(target_path)
    expected_rows = int(config["selected_targets"]["expected_rows"])
    if len(targets) != expected_rows:
        raise ProtocolError(f"selected target rows {len(targets)} != {expected_rows}")
    by_key: dict[str, list[dict[str, Any]]] = {}
    for target in targets:
        by_key.setdefault(cell_key(target), []).append(target)
    expected_cells = int(config["selected_targets"]["expected_unique_victim_layer_cells"])
    if len(by_key) != expected_cells:
        raise ProtocolError(f"unique cells {len(by_key)} != {expected_cells}")
    cells: list[dict[str, Any]] = []
    for key in sorted(by_key):
        rows = by_key[key]
        first = rows[0]
        identity_fields = (
            "victim_id",
            "document_index",
            "layer",
            "flat_token_idx",
            "target_hidden_sha256",
            "target_router_logits_sha256",
            "window_token_ids_sha256",
        )
        for row in rows[1:]:
            if any(row[field] != first[field] for field in identity_fields):
                raise ProtocolError(f"selected target cell identity disagreement: {key}")
            if row["window_token_ids"] != first["window_token_ids"]:
                raise ProtocolError(f"selected target token window disagreement: {key}")
        cells.append(
            {
                "cell_key": key,
                "victim_id": first["victim_id"],
                "document_index": int(first["document_index"]),
                "layer": int(first["layer"]),
                "flat_token_idx": int(first["flat_token_idx"]),
                "target_hidden_sha256": first["target_hidden_sha256"],
                "target_router_logits_sha256": first["target_router_logits_sha256"],
                "window_token_ids": list(map(int, first["window_token_ids"])),
                "window_token_ids_sha256": first["window_token_ids_sha256"],
                "prior_sensitive_targets": rows,
            }
        )
    return targets, cells


def workload_rows(cells: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_victim: dict[str, dict[str, Any]] = {}
    for cell in cells:
        row = {
            "victim_id": str(cell["victim_id"]),
            "document_index": int(cell["document_index"]),
            "window_token_ids": list(map(int, cell["window_token_ids"])),
            "window_token_ids_sha256": str(cell["window_token_ids_sha256"]),
        }
        previous = by_victim.setdefault(row["victim_id"], row)
        if previous != row:
            raise ProtocolError(f"victim workload disagreement: {row['victim_id']}")
    return [by_victim[key] for key in sorted(by_victim)]


def companion_identity(row: Mapping[str, Any]) -> str:
    return (
        f"{row['victim_id']}|token={int(row['token_idx']):03d}|"
        f"rank={int(row['topk_rank'])}|expert={int(row['expert_id'])}"
    )


def ordered_companion_indices(
    identities: Sequence[str], seed: str, key: str
) -> list[int]:
    if len(set(identities)) != len(identities):
        raise ProtocolError("companion identities are not unique")
    return sorted(
        range(len(identities)),
        key=lambda index: (
            sha256_text(f"{seed}|{key}|{identities[index]}"), identities[index]
        ),
    )


def select_companion_segments(
    identities: Sequence[str], seed: str, key: str, width: int, segments: int
) -> list[list[int]]:
    required = width * segments
    if len(identities) < required:
        raise ProtocolError(
            f"companion pool {len(identities)} is smaller than required {required}"
        )
    order = ordered_companion_indices(identities, seed, key)
    return [order[index * width : (index + 1) * width] for index in range(segments)]


def rank_order(top_k: int, context_index: int, repeat: int) -> list[int]:
    offset = (context_index + repeat) % top_k
    return list(range(offset, top_k)) + list(range(offset))


def build_capture_and_pools(
    model: Any,
    workloads: Sequence[Mapping[str, Any]],
    cells: Sequence[Mapping[str, Any]],
    base_config: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[tuple[int, int], list[dict[str, Any]]]]:
    import torch

    hidden_size = int(base_config["model"]["hidden_size"])
    num_experts = int(base_config["model"]["num_experts"])
    top_k = int(base_config["model"]["num_experts_per_tok"])
    target_layers = sorted({int(cell["layer"]) for cell in cells})
    captures: dict[str, dict[str, Any]] = {}
    pools: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for workload in workloads:
        input_ids = torch.tensor(
            [workload["window_token_ids"]], dtype=torch.long, device="cuda"
        )
        captured = base.run_native_capture(model, input_ids, base_config)
        victim_id = str(workload["victim_id"])
        layer_rows: dict[str, Any] = {}
        for layer in target_layers:
            hidden = captured["moe_inputs"][layer].reshape(-1, hidden_size)
            logits = captured["output"].router_logits[layer].reshape(-1, num_experts)
            _weights, experts = base.topk_from_logits(logits, top_k)
            layer_rows[str(layer)] = {
                "hidden": hidden.detach().cpu().clone(),
                "router_logits": logits.detach().cpu().clone(),
                "experts": experts.detach().cpu().clone(),
            }
            for token_idx in range(hidden.shape[0]):
                for topk_rank in range(top_k):
                    expert_id = int(experts[token_idx, topk_rank].item())
                    pools.setdefault((layer, expert_id), []).append(
                        {
                            "victim_id": victim_id,
                            "token_idx": token_idx,
                            "topk_rank": topk_rank,
                            "expert_id": expert_id,
                            "hidden": hidden[token_idx].detach().cpu().clone(),
                        }
                    )
        captures[victim_id] = layer_rows
    return captures, pools


def prepare_cells(
    cells: Sequence[Mapping[str, Any]],
    captures: Mapping[str, Mapping[str, Any]],
    pools: Mapping[tuple[int, int], Sequence[Mapping[str, Any]]],
    config: Mapping[str, Any],
    base_config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    top_k = int(base_config["model"]["num_experts_per_tok"])
    minimum = int(config["lane"]["minimum_distinct_real_companions_per_rank"])
    eligible: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for source in cells:
        cell = dict(source)
        layer = int(cell["layer"])
        token_idx = int(cell["flat_token_idx"])
        captured = captures[str(cell["victim_id"])][str(layer)]
        hidden = captured["hidden"][token_idx]
        logits = captured["router_logits"][token_idx]
        experts = list(map(int, captured["experts"][token_idx].tolist()))
        if len(experts) != top_k or len(set(experts)) != top_k:
            raise ProtocolError(f"target top-k malformed: {cell['cell_key']}")
        if base.tensor_sha256(hidden) != str(cell["target_hidden_sha256"]):
            raise ProtocolError(f"target hidden drift: {cell['cell_key']}")
        if base.tensor_sha256(logits) != str(cell["target_router_logits_sha256"]):
            raise ProtocolError(f"target router drift: {cell['cell_key']}")
        filtered_by_rank: dict[int, list[dict[str, Any]]] = {}
        counts: dict[str, int] = {}
        for rank, expert_id in enumerate(experts):
            filtered = [
                dict(row)
                for row in pools.get((layer, expert_id), [])
                if not (
                    str(row["victim_id"]) == str(cell["victim_id"])
                    and int(row["token_idx"]) == token_idx
                )
            ]
            # One hidden row can route to the same expert only once; retain a
            # unique route identity so companion segments are genuinely distinct.
            unique = {companion_identity(row): row for row in filtered}
            filtered_by_rank[rank] = [unique[key] for key in sorted(unique)]
            counts[str(rank)] = len(filtered_by_rank[rank])
        if any(value < minimum for value in counts.values()):
            rejected.append(
                {
                    "cell_key": cell["cell_key"],
                    "victim_id": cell["victim_id"],
                    "layer": layer,
                    "companion_count_by_rank": counts,
                    "reason": "INSUFFICIENT_DISTINCT_REAL_COMPANIONS",
                }
            )
            continue
        cell["expert_ids"] = experts
        cell["_hidden_cpu"] = hidden.detach().clone()
        cell["_companions_by_rank"] = filtered_by_rank
        cell["companion_count_by_rank"] = counts
        eligible.append(cell)
    return eligible, rejected


def verify_prior_sensitive_hashes(
    model: Any, cell: Mapping[str, Any], base_config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    import torch

    hidden = cell["_hidden_cpu"].to(device="cuda", dtype=torch.bfloat16)
    rows: list[dict[str, Any]] = []
    for target in cell["prior_sensitive_targets"]:
        rank = int(target["topk_rank"])
        expert_id = int(target["expert_id"])
        if int(cell["expert_ids"][rank]) != expert_id:
            raise ProtocolError("prior sensitive target expert/rank drift")
        expert = model.model.layers[int(cell["layer"])].mlp.experts[expert_id]
        with torch.inference_mode():
            m1 = expert(hidden.reshape(1, -1))[0]
            m64 = expert(hidden.reshape(1, -1).repeat(64, 1))[0]
        m1_hash = base.tensor_sha256(m1)
        m64_hash = base.tensor_sha256(m64)
        if m1_hash != str(target["local_m1_sha256"]):
            raise ProtocolError("prior M1 hash did not reproduce")
        if m64_hash != str(target["local_m64_sha256"]):
            raise ProtocolError("prior M64 hash did not reproduce")
        if m1_hash == m64_hash:
            raise ProtocolError("prior shape-sensitive target no longer differs")
        rows.append(
            {
                "topk_rank": rank,
                "expert_id": expert_id,
                "m1_sha256": m1_hash,
                "m64_sha256": m64_hash,
            }
        )
    return rows


def build_lane_batch(
    focal: Any,
    companions: Sequence[Mapping[str, Any]],
    context: Mapping[str, Any],
    canonical_m: int,
) -> tuple[Any, list[str]]:
    import torch

    slot = int(context["focal_slot"])
    if slot < 0 or slot >= canonical_m:
        raise ProtocolError("focal slot is outside the lane")
    if context["kind"] == "zero":
        other_rows = [torch.zeros_like(focal) for _ in range(canonical_m - 1)]
        identities = [f"zero-{index}" for index in range(canonical_m - 1)]
    else:
        if len(companions) != canonical_m - 1:
            raise ProtocolError("real lane has wrong companion count")
        other_rows = [row["hidden"] for row in companions]
        identities = [companion_identity(row) for row in companions]
    tensors = list(other_rows)
    labels = list(identities)
    tensors.insert(slot, focal)
    labels.insert(slot, "FOCAL")
    return torch.stack(tensors, dim=0), labels


def observation_signature(row: Mapping[str, Any]) -> bytes:
    observation = row["observation"]
    trace = row["trace"]
    return base.canonical_json_bytes(
        {
            "target_moe_output_sha256": observation["target_moe_output_sha256"],
            "router_logits_sha256_by_layer": observation["router_logits_sha256_by_layer"],
            "topk_experts_by_layer": observation["topk_experts_by_layer"],
            "final_logits_sha256": observation["final_logits_sha256"],
            "greedy_token_id": observation["greedy_token_id"],
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
    index: int,
    cell: Mapping[str, Any],
    config: Mapping[str, Any],
    base_config: Mapping[str, Any],
) -> dict[str, Any]:
    import torch

    canonical_m = int(config["lane"]["canonical_m"])
    repeats = int(config["lane"]["repeats"])
    contexts = list(config["lane"]["contexts"])
    seed = str(config["lane"]["assignment_seed"])
    top_k = int(base_config["model"]["num_experts_per_tok"])
    focal_cpu = cell["_hidden_cpu"]
    focal = focal_cpu.to(device="cuda", dtype=torch.bfloat16)
    prior_hashes = verify_prior_sensitive_hashes(model, cell, base_config)

    context_batches: dict[tuple[int, int], tuple[Any, list[str]]] = {}
    context_meta: dict[str, Any] = {}
    for rank in range(top_k):
        pool = cell["_companions_by_rank"][rank]
        identities = [companion_identity(row) for row in pool]
        segments = select_companion_segments(
            identities,
            seed,
            f"{cell['cell_key']}|rank={rank}",
            canonical_m - 1,
            3,
        )
        for context_index, context in enumerate(contexts):
            selected = (
                []
                if context["kind"] == "zero"
                else [pool[item] for item in segments[int(context["segment"])]]
            )
            batch, labels = build_lane_batch(
                focal_cpu, selected, context, canonical_m
            )
            context_batches[(context_index, rank)] = (
                batch.to(device="cuda", dtype=torch.bfloat16), labels
            )
            context_meta.setdefault(str(context_index), {"name": context["name"], "ranks": {}})[
                "ranks"
            ][str(rank)] = {
                "focal_slot": int(context["focal_slot"]),
                "lane_input_sha256": base.tensor_sha256(batch),
                "lane_row_identities": labels,
            }

    outputs: dict[int, dict[int, list[Any]]] = {
        context_index: {rank: [] for rank in range(top_k)}
        for context_index in range(len(contexts))
    }
    elapsed_ms: dict[int, list[float]] = {
        context_index: [] for context_index in range(len(contexts))
    }
    for context_index in range(len(contexts)):
        for repeat in range(repeats):
            total_ms = 0.0
            for rank in rank_order(top_k, context_index, repeat):
                expert_id = int(cell["expert_ids"][rank])
                expert = model.model.layers[int(cell["layer"])].mlp.experts[expert_id]
                batch = context_batches[(context_index, rank)][0]
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                with torch.inference_mode():
                    lane_output = expert(batch)
                end.record()
                end.synchronize()
                slot = int(contexts[context_index]["focal_slot"])
                outputs[context_index][rank].append(lane_output[slot].detach().clone())
                total_ms += float(start.elapsed_time(end))
            elapsed_ms[context_index].append(total_ms)

    raw_hashes: dict[str, dict[str, list[str]]] = {}
    within_context_repeat_stable = True
    cross_context_raw_stable = True
    for context_index in range(len(contexts)):
        context_hashes: dict[str, list[str]] = {}
        for rank in range(top_k):
            hashes = [base.tensor_sha256(value) for value in outputs[context_index][rank]]
            context_hashes[str(rank)] = hashes
            within_context_repeat_stable &= len(set(hashes)) == 1
        raw_hashes[str(context_index)] = context_hashes
    for rank in range(top_k):
        representatives = [raw_hashes[str(context)][str(rank)][0] for context in range(len(contexts))]
        cross_context_raw_stable &= len(set(representatives)) == 1

    representative = base.PairIdentity(
        layer=int(cell["layer"]),
        flat_token_idx=int(cell["flat_token_idx"]),
        topk_rank=0,
        expert_id=int(cell["expert_ids"][0]),
    )
    input_ids = torch.tensor(
        [cell["window_token_ids"]], dtype=torch.long, device="cuda"
    )
    native = base.run_observation(model, input_ids, base_config, representative)
    with observable.patched_topk_contributions(model, cell, None, "self") as noop_trace:
        noop = base.run_observation(model, input_ids, base_config, representative)
    noop_checks = {
        "target_input_equal": native["target_input_sha256"] == noop["target_input_sha256"],
        "target_router_equal": native["target_router_logits_sha256"] == noop[
            "target_router_logits_sha256"
        ],
        "target_moe_equal": native["target_moe_output_sha256"] == noop[
            "target_moe_output_sha256"
        ],
        "routes_equal": native["topk_experts_by_layer"] == noop["topk_experts_by_layer"],
        "final_logits_equal": native["final_logits_sha256"] == noop["final_logits_sha256"],
    }
    if not all(noop_checks.values()):
        raise ProtocolError(f"native no-op failed: {cell['cell_key']} {noop_checks}")

    full_rows: dict[str, list[dict[str, Any]]] = {}
    for context_index, context in enumerate(contexts):
        replacement_map = {
            rank: outputs[context_index][rank][0] for rank in range(top_k)
        }
        rows: list[dict[str, Any]] = []
        for repeat in range(repeats):
            with observable.patched_topk_contributions(
                model, cell, replacement_map, "replacement"
            ) as trace:
                observation = base.run_observation(
                    model, input_ids, base_config, representative
                )
            for rank in range(top_k):
                expected = raw_hashes[str(context_index)][str(rank)][0]
                if trace["target_applied_raw_sha256_by_rank"][str(rank)] != expected:
                    raise ProtocolError("full-forward applied raw differs from C8 side-call")
            if trace["non_target_contributions_sha256"] != noop_trace[
                "non_target_contributions_sha256"
            ]:
                raise ProtocolError("non-target contribution closure failed")
            rows.append(
                {
                    "repeat": repeat,
                    "trace": dict(trace),
                    "observation": base.public_observation(observation),
                }
            )
        if len({observation_signature(row) for row in rows}) != 1:
            raise ProtocolError("same C8 context full-forward output is unstable")
        full_rows[str(context_index)] = rows

    start_layer = int(cell["layer"]) + 1
    representatives = [full_rows[str(index)][0]["observation"] for index in range(len(contexts))]
    target_moe_stable = len({row["target_moe_output_sha256"] for row in representatives}) == 1
    route_payloads = [
        base.canonical_json_bytes(row["topk_experts_by_layer"][start_layer:])
        for row in representatives
    ]
    downstream_routes_stable = len(set(route_payloads)) == 1
    final_logits_stable = len({row["final_logits_sha256"] for row in representatives}) == 1
    greedy_token_stable = len({int(row["greedy_token_id"]) for row in representatives}) == 1
    return {
        "cell_index": index,
        "cell_key": cell["cell_key"],
        "victim_id": cell["victim_id"],
        "document_index": int(cell["document_index"]),
        "layer": int(cell["layer"]),
        "flat_token_idx": int(cell["flat_token_idx"]),
        "expert_ids": list(map(int, cell["expert_ids"])),
        "prior_sensitive_hashes": prior_hashes,
        "companion_count_by_rank": cell["companion_count_by_rank"],
        "context_metadata": context_meta,
        "raw_sha256_by_context_rank_repeat": raw_hashes,
        "expert_ms_by_context_repeat": {str(key): value for key, value in elapsed_ms.items()},
        "within_context_repeat_raw_bitwise": within_context_repeat_stable,
        "cross_context_raw_bitwise": cross_context_raw_stable,
        "cross_context_target_moe_bitwise": target_moe_stable,
        "cross_context_downstream_routes_equal": downstream_routes_stable,
        "cross_context_final_logits_bitwise": final_logits_stable,
        "cross_context_greedy_token_equal": greedy_token_stable,
        "native_noop_checks": noop_checks,
        "full_forward_by_context": full_rows,
        "integrity_status": "PASS",
    }


def classify_results(
    rows: Sequence[Mapping[str, Any]],
    rejected: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    victims = {str(row["victim_id"]) for row in rows}
    enough = len(rows) >= int(config["gate"]["minimum_eligible_cells"]) and len(
        victims
    ) >= int(config["gate"]["minimum_distinct_victims"])
    raw_failures = [
        row["cell_key"]
        for row in rows
        if not row["within_context_repeat_raw_bitwise"]
        or not row["cross_context_raw_bitwise"]
    ]
    route_failures = [
        row["cell_key"]
        for row in rows
        if not row["cross_context_target_moe_bitwise"]
        or not row["cross_context_downstream_routes_equal"]
        or not row["cross_context_final_logits_bitwise"]
    ]
    if not enough:
        verdict = str(config["decision"]["invalid"])
    elif raw_failures or route_failures:
        verdict = str(config["decision"]["kill"])
    else:
        verdict = str(config["decision"]["pass"])
    return {
        "verdict": verdict,
        "paper_result": False,
        "claim_boundary": config["research_boundary"],
        "eligible_cell_count": len(rows),
        "eligible_distinct_victim_count": len(victims),
        "rejected_cell_count": len(rejected),
        "raw_failure_cell_count": len(raw_failures),
        "raw_failure_cells": raw_failures,
        "route_or_final_failure_cell_count": len(route_failures),
        "route_or_final_failure_cells": route_failures,
        "all_prior_sensitive_hashes_reproduced": all(
            bool(row["prior_sensitive_hashes"]) for row in rows
        ),
        "gate_checks": {
            "minimum_coverage": enough,
            "zero_raw_context_mismatch": not raw_failures,
            "zero_route_or_final_context_mismatch": not route_failures,
        },
        "next_gate_if_pass": "continuous_decode_lane_cost_vs_serial_M1_and_vLLM_batch_invariance",
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-wall-seconds", type=int, default=3600)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    config_path = args.config.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise ProtocolError(f"refusing to reuse output directory: {output_dir}")
    config = base.load_json(config_path)
    if config.get("schema_version") != "stablebatch-shape-lane-correctness-pilot-v1":
        raise ProtocolError("unexpected shape-lane config schema")
    if config.get("status") != "FROZEN_PRE_RUN":
        raise ProtocolError("shape-lane config is not frozen")
    base_config_path = load_bound_file(repo_root, config["base_config"])
    target_path = load_bound_file(repo_root, config["selected_targets"])
    base_config = base.load_json(base_config_path)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    started = time.time()
    try:
        base.write_json_new(
            output_dir / "run_request.json",
            {
                "schema_version": "stablebatch-shape-lane-run-request-v1",
                "started_at": base.utc_now(),
                "argv": sys.argv,
                "runner_sha256": base.sha256_file(Path(__file__).resolve()),
                "config_sha256": base.sha256_file(config_path),
                "base_config_sha256": base.sha256_file(base_config_path),
                "selected_targets_sha256": base.sha256_file(target_path),
                "max_wall_seconds": args.max_wall_seconds,
            },
        )
        pre_import_gpu = base.gpu_snapshot()
        environment = base.verify_environment(base_config, pre_import_gpu)
        base.write_json_new(output_dir / "environment.json", environment)
        base.write_json_new(output_dir / "config_snapshot.json", config)
        targets, cells = load_cells(target_path, config)
        workloads = workload_rows(cells)
        base.write_jsonl_new(output_dir / "workloads.jsonl", workloads)
        base.write_jsonl_new(output_dir / "source_targets.jsonl", targets)
        model, _tokenizer = base.load_model(base_config)
        first_ids = __import__("torch").tensor(
            [workloads[0]["window_token_ids"]],
            dtype=__import__("torch").long,
            device="cuda",
        )
        observable.warmup_native_only(model, first_ids, base_config)
        captures, pools = build_capture_and_pools(
            model, workloads, cells, base_config
        )
        eligible, rejected = prepare_cells(
            cells, captures, pools, config, base_config
        )
        base.write_jsonl_new(output_dir / "rejected_cells.jsonl", rejected)
        base.write_jsonl_new(
            output_dir / "eligible_cells.jsonl",
            (
                {
                    key: value
                    for key, value in row.items()
                    if not key.startswith("_") and key != "prior_sensitive_targets"
                }
                for row in eligible
            ),
        )
        result_rows: list[dict[str, Any]] = []
        result_path = output_dir / "cell_results.jsonl"
        with result_path.open("x", encoding="utf-8") as stream:
            for index, cell in enumerate(eligible):
                if time.time() - started > args.max_wall_seconds:
                    raise TimeoutError("shape-lane pilot exceeded wall-time cap")
                row = run_cell(model, index, cell, config, base_config)
                result_rows.append(row)
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                stream.flush()
        summary = classify_results(result_rows, rejected, config)
        summary["completed_at"] = base.utc_now()
        summary["wall_seconds"] = time.time() - started
        base.write_json_new(output_dir / "summary.json", summary)
        base.write_json_new(
            output_dir / "runtime_final.json", base.verify_final_runtime(base_config)
        )
        base.write_json_new(
            output_dir / "RUN_STATUS.json",
            {
                "status": "COMPLETE",
                "scientific_result_eligible": True,
                "paper_result": False,
                "verdict": summary["verdict"],
                "completed_at": base.utc_now(),
                "wall_seconds": time.time() - started,
            },
        )
        return 0
    except Exception as error:
        if not (output_dir / "RUN_STATUS.json").exists():
            base.write_json_new(
                output_dir / "RUN_STATUS.json",
                {
                    "status": "FAILED",
                    "scientific_result_eligible": False,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "failed_at": base.utc_now(),
                },
            )
        raise


if __name__ == "__main__":
    raise SystemExit(main())

