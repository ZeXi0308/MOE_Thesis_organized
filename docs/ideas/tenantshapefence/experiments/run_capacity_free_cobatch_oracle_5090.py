#!/usr/bin/env python3
"""Minimal capacity-free co-batch oracle for TenantShapeFence.

The search keeps batch size, sequence length, victim slot, model, and all
non-MoE execution surfaces fixed.  It changes only the foreign request token
content.  In ordinary OLMoE execution those rows alter each expert GEMM's M.

The defense arm keeps the same full batch and splits only expert execution by
security domain: each expert is invoked separately for foreign and victim
rows, while routing, attention, dense layers, normalization, and LM head stay
batched.  This is an exploratory single-model/GPU oracle, not a serving or
whole-model noninterference result.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import os
from pathlib import Path
import sys
import time
import types
from typing import Any, Iterable, Mapping, Sequence


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
STABLEBATCH_DIR = REPO_ROOT / "docs" / "ideas" / "stablebatch" / "experiments"
if str(STABLEBATCH_DIR) not in sys.path:
    sys.path.insert(0, str(STABLEBATCH_DIR))

import run_single_contribution_pilot as base  # noqa: E402


DEFAULT_CONFIG = STABLEBATCH_DIR / "configs" / "single_contribution_pilot_v1.json"
DEFAULT_VICTIM_RESULTS = (
    STABLEBATCH_DIR
    / "outputs"
    / "single_contribution_20260810_run01"
    / "target_results.jsonl"
)
DEFAULT_ATTACKER_SOURCES = (
    STABLEBATCH_DIR
    / "outputs"
    / "observable_selector_20260810_run01"
    / "workloads.jsonl",
    STABLEBATCH_DIR
    / "outputs"
    / "selectability_decomposition_20260810_run02"
    / "workloads.jsonl",
)

SCHEMA = "tenantshapefence-capacity-free-cobatch-oracle-v1"


class PilotError(RuntimeError):
    """The exploratory result cannot be interpreted."""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def parse_int_list(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values or any(item < 2 for item in values):
        raise argparse.ArgumentTypeError("batch sizes must be integers >= 2")
    if len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("batch sizes must be unique")
    return tuple(sorted(values))


def public_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in observation.items() if not key.startswith("_")}


def token_ids_sha256(tokens: Sequence[int]) -> str:
    return __import__("hashlib").sha256(base.canonical_json_bytes(list(map(int, tokens)))).hexdigest()


def verify_upstream_artifact(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    manifest_path = path.parent / "MANIFEST.json"
    status_path = path.parent / "RUN_STATUS.json"
    if not manifest_path.is_file() or not status_path.is_file():
        raise PilotError(f"upstream artifact lacks MANIFEST/RUN_STATUS: {path}")
    manifest = base.load_json(manifest_path)
    status = base.load_json(status_path)
    entry = manifest.get("files", {}).get(path.name)
    observed = base.sha256_file(path)
    if not isinstance(entry, dict) or str(entry.get("sha256")) != observed:
        raise PilotError(f"upstream manifest does not bind {path}")
    if status.get("status") != "COMPLETE" or not bool(
        status.get("scientific_result_eligible")
    ):
        raise PilotError(f"upstream source is not a completed eligible run: {path}")
    return {
        "path": str(path),
        "sha256": observed,
        "manifest_path": str(manifest_path),
        "manifest_sha256": base.sha256_file(manifest_path),
        "run_status_path": str(status_path),
        "run_status_sha256": base.sha256_file(status_path),
        "upstream_status": str(status["status"]),
        "scientific_result_eligible": True,
    }


def verify_model_binding(config: Mapping[str, Any]) -> dict[str, Any]:
    if config.get("status") != "FROZEN_PRE_RUN":
        raise PilotError("base config is not FROZEN_PRE_RUN")
    model_cfg = config["model"]
    model_root = Path(str(model_cfg["local_path"])).resolve()
    observed: dict[str, str] = {}
    for relative, expected in model_cfg["file_sha256"].items():
        path = model_root / str(relative)
        if not path.is_file():
            raise PilotError(f"missing pinned model file {path}")
        digest = base.sha256_file(path)
        if digest != str(expected):
            raise PilotError(f"model hash mismatch for {relative}: {digest}")
        observed[str(relative)] = digest
    return {"model_root": str(model_root), "model_file_sha256": observed}


def load_victim(path: Path) -> dict[str, Any]:
    rows = [
        row
        for row in base.load_jsonl(Path(path))
        if bool(row.get("reproducible_token_flip"))
    ]
    if len(rows) != 1:
        raise PilotError(f"expected exactly one reproducible token-flip victim, got {len(rows)}")
    row = rows[0]
    tokens = list(map(int, row["window_token_ids"]))
    if len(tokens) != 16:
        raise PilotError("frozen victim window is not 16 tokens")
    if token_ids_sha256(tokens) != str(row["window_token_ids_sha256"]):
        raise PilotError("frozen victim token hash does not match its bytes")
    if row.get("integrity_status") != "PASS":
        raise PilotError("frozen victim did not pass its source integrity checks")
    return {
        "victim_id": str(row["victim_id"]),
        "window_token_ids": tokens,
        "window_token_ids_sha256": str(row["window_token_ids_sha256"]),
        "source_target_id": str(row["target_id"]),
        "source_intervention": {
            "layer": int(row["layer"]),
            "expert_id": int(row["expert_id"]),
            "topk_rank": int(row["topk_rank"]),
            "m1_greedy": int(row["greedy_token_pairs_by_repeat"][0][0]),
            "m64_greedy": int(row["greedy_token_pairs_by_repeat"][0][1]),
        },
    }


def load_attackers(paths: Sequence[Path], victim_hash: str) -> list[dict[str, Any]]:
    attackers: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in paths:
        for row in base.load_jsonl(Path(source)):
            digest = str(row["window_token_ids_sha256"])
            tokens = list(map(int, row["window_token_ids"]))
            if len(tokens) != 16:
                raise PilotError(f"attacker {row.get('victim_id')} is not 16 tokens")
            if token_ids_sha256(tokens) != digest:
                raise PilotError(f"attacker {row.get('victim_id')} token hash mismatch")
            if digest == victim_hash or digest in seen:
                continue
            seen.add(digest)
            attackers.append(
                {
                    "attacker_id": str(row["victim_id"]),
                    "window_token_ids": tokens,
                    "window_token_ids_sha256": digest,
                    "source": str(source),
                }
            )
    attackers.sort(key=lambda row: (row["window_token_ids_sha256"], row["attacker_id"]))
    if len(attackers) < 2:
        raise PilotError("need at least two distinct foreign workloads")
    return attackers


def _ledger_from_routes(
    selected_experts: Any,
    request_domains: Sequence[int],
    *,
    sequence_length: int,
    num_experts: int,
    victim_domain: int,
) -> dict[str, Any]:
    import torch

    flat_domains = torch.as_tensor(
        list(map(int, request_domains)), device=selected_experts.device, dtype=torch.long
    ).repeat_interleave(int(sequence_length))
    contribution_domains = flat_domains[:, None].expand_as(selected_experts).reshape(-1)
    flat_experts = selected_experts.reshape(-1)
    total = torch.bincount(flat_experts, minlength=int(num_experts))
    victim_mask = contribution_domains == int(victim_domain)
    victim = torch.bincount(flat_experts[victim_mask], minlength=int(num_experts))
    foreign = total - victim
    victim_token_index = len(request_domains) * int(sequence_length) - 1
    return {
        "total_m_by_expert": list(map(int, total.detach().cpu().tolist())),
        "foreign_m_by_expert": list(map(int, foreign.detach().cpu().tolist())),
        "victim_domain_m_by_expert": list(map(int, victim.detach().cpu().tolist())),
        "victim_last_token_experts": list(
            map(int, selected_experts[victim_token_index].detach().cpu().tolist())
        ),
        "processed_contributions": int(selected_experts.numel()),
    }


@contextlib.contextmanager
def expert_execution_mode(
    model: Any,
    *,
    mode: str,
    request_domains: Sequence[int],
    victim_domain: int = 1,
):
    """Instrument native execution or split only each expert call by domain."""

    if mode not in {"unprotected", "domain_split"}:
        raise ValueError(f"unknown execution mode {mode!r}")
    if list(request_domains).count(int(victim_domain)) != 1:
        raise PilotError("exactly one victim request domain is required")

    originals: list[tuple[Any, Any]] = []
    trace: dict[str, Any] = {
        "mode": mode,
        "request_domains": list(map(int, request_domains)),
        "victim_domain": int(victim_domain),
        "layers": {},
    }
    try:
        for layer_index, decoder_layer in enumerate(model.model.layers):
            block = decoder_layer.mlp
            original_forward = block.forward

            if mode == "unprotected":

                def observed_forward(
                    this: Any,
                    hidden_states: Any,
                    *args: Any,
                    _original: Any = original_forward,
                    _layer: int = layer_index,
                    **kwargs: Any,
                ) -> Any:
                    result = _original(hidden_states, *args, **kwargs)
                    if not isinstance(result, tuple) or len(result) != 2:
                        raise PilotError("unexpected OLMoE block output")
                    final_hidden_states, router_logits = result
                    _, sequence_length, _ = hidden_states.shape
                    _, selected = base.topk_from_logits(router_logits, int(this.top_k))
                    ledger = _ledger_from_routes(
                        selected,
                        request_domains,
                        sequence_length=int(sequence_length),
                        num_experts=int(this.num_experts),
                        victim_domain=int(victim_domain),
                    )
                    ledger["expert_call_count"] = sum(
                        int(value > 0) for value in ledger["total_m_by_expert"]
                    )
                    trace["layers"][str(_layer)] = ledger
                    return final_hidden_states, router_logits

                replacement = types.MethodType(observed_forward, block)
            else:

                def split_forward(
                    this: Any,
                    hidden_states: Any,
                    *args: Any,
                    _layer: int = layer_index,
                    **kwargs: Any,
                ) -> Any:
                    import torch
                    import torch.nn.functional as F

                    if args or kwargs:
                        raise PilotError("unexpected arguments to OLMoE expert block")
                    batch_size, sequence_length, hidden_dim = hidden_states.shape
                    if int(batch_size) != len(request_domains):
                        raise PilotError("domain vector does not match batch size")
                    flat_hidden = hidden_states.view(-1, hidden_dim)
                    router_logits = this.gate(flat_hidden)
                    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
                    routing_weights, selected = torch.topk(
                        routing_weights, int(this.top_k), dim=-1
                    )
                    if bool(this.norm_topk_prob):
                        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
                    routing_weights = routing_weights.to(flat_hidden.dtype)
                    final_hidden = torch.zeros(
                        (int(batch_size) * int(sequence_length), int(hidden_dim)),
                        dtype=flat_hidden.dtype,
                        device=flat_hidden.device,
                    )
                    expert_mask = F.one_hot(
                        selected, num_classes=int(this.num_experts)
                    ).permute(2, 1, 0)
                    flat_domains = torch.as_tensor(
                        list(map(int, request_domains)),
                        device=flat_hidden.device,
                        dtype=torch.long,
                    ).repeat_interleave(int(sequence_length))
                    expert_call_count = 0
                    domain_call_count = {str(domain): 0 for domain in sorted(set(request_domains))}
                    processed = 0
                    for expert_idx in range(int(this.num_experts)):
                        idx, top_x = torch.where(expert_mask[expert_idx])
                        expert_layer = this.experts[expert_idx]
                        for domain in sorted(set(map(int, request_domains))):
                            keep = flat_domains[top_x] == int(domain)
                            domain_idx = idx[keep]
                            domain_top_x = top_x[keep]
                            if int(domain_top_x.numel()) == 0:
                                continue
                            current_state = flat_hidden[None, domain_top_x].reshape(
                                -1, hidden_dim
                            )
                            current_hidden = expert_layer(current_state) * routing_weights[
                                domain_top_x, domain_idx, None
                            ]
                            final_hidden.index_add_(
                                0,
                                domain_top_x,
                                current_hidden.to(flat_hidden.dtype),
                            )
                            expert_call_count += 1
                            domain_call_count[str(domain)] += 1
                            processed += int(domain_top_x.numel())
                    ledger = _ledger_from_routes(
                        selected,
                        request_domains,
                        sequence_length=int(sequence_length),
                        num_experts=int(this.num_experts),
                        victim_domain=int(victim_domain),
                    )
                    if processed != int(selected.numel()):
                        raise PilotError("domain split dropped or duplicated expert contributions")
                    ledger.update(
                        {
                            "expert_call_count": int(expert_call_count),
                            "domain_call_count": domain_call_count,
                        }
                    )
                    trace["layers"][str(_layer)] = ledger
                    return (
                        final_hidden.reshape(batch_size, sequence_length, hidden_dim),
                        router_logits,
                    )

                replacement = types.MethodType(split_forward, block)

            originals.append((block, original_forward))
            block.forward = replacement
        yield trace
    finally:
        for block, original in reversed(originals):
            block.forward = original


def run_observation(
    model: Any,
    input_ids: Any,
    *,
    mode: str,
    request_domains: Sequence[int],
    victim_domain: int = 1,
) -> dict[str, Any]:
    import torch

    batch_size, sequence_length = map(int, input_ids.shape)
    if batch_size != len(request_domains):
        raise PilotError("input batch and domain vector differ")
    victim_slots = [
        index for index, domain in enumerate(request_domains) if int(domain) == int(victim_domain)
    ]
    if len(victim_slots) != 1:
        raise PilotError("expected one victim slot")
    victim_slot = victim_slots[0]
    victim_position = sequence_length - 1
    attention_mask = torch.ones_like(input_ids)
    with expert_execution_mode(
        model,
        mode=mode,
        request_domains=request_domains,
        victim_domain=victim_domain,
    ) as execution_trace:
        with torch.inference_mode():
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_router_logits=True,
                return_dict=True,
            )
    if getattr(output, "past_key_values", None) is not None:
        raise PilotError("observation unexpectedly returned a cache")
    if len(execution_trace["layers"]) != len(model.model.layers):
        raise PilotError("not every expert layer was observed")
    expected_contributions = batch_size * sequence_length * int(model.config.num_experts_per_tok)
    for layer, ledger in execution_trace["layers"].items():
        if int(ledger["processed_contributions"]) != expected_contributions:
            raise PilotError(f"layer {layer} did not process every routed contribution")

    victim_flat_index = victim_slot * sequence_length + victim_position
    routes: list[list[int]] = []
    router_hashes: list[str] = []
    for logits in output.router_logits:
        victim_logits = logits.reshape(-1, logits.shape[-1])[victim_flat_index]
        _, selected = base.topk_from_logits(
            victim_logits, int(model.config.num_experts_per_tok)
        )
        routes.append(list(map(int, selected.detach().cpu().tolist())))
        router_hashes.append(base.tensor_sha256(victim_logits))
    final_logits = output.logits[victim_slot, victim_position].detach().float().cpu()
    return {
        "mode": mode,
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "victim_slot": victim_slot,
        "victim_position": victim_position,
        "victim_input_sha256": base.tensor_sha256(input_ids[victim_slot]),
        "batch_input_sha256": base.tensor_sha256(input_ids),
        "victim_routes_by_layer": routes,
        "victim_router_logits_sha256_by_layer": router_hashes,
        "victim_final_logits_sha256": base.tensor_sha256(final_logits),
        "victim_greedy_token_id": int(torch.argmax(final_logits).item()),
        "execution_trace": execution_trace,
        "_final_logits_cpu": final_logits,
    }


def compare_observations(
    reference: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    import torch

    if reference["victim_input_sha256"] != candidate["victim_input_sha256"]:
        raise PilotError("victim input changed between paired arms")
    if reference["batch_size"] != candidate["batch_size"]:
        raise PilotError("paired arms use different batch sizes")
    ref_routes = reference["victim_routes_by_layer"]
    candidate_routes = candidate["victim_routes_by_layer"]
    if len(ref_routes) != len(candidate_routes):
        raise PilotError("route trace layer counts differ")
    ordered_layers: list[int] = []
    membership_layers: list[int] = []
    shape_delta: list[dict[str, Any]] = []
    for layer, (left, right) in enumerate(zip(ref_routes, candidate_routes)):
        if list(left) != list(right):
            ordered_layers.append(layer)
        if set(map(int, left)) != set(map(int, right)):
            membership_layers.append(layer)
        ref_ledger = reference["execution_trace"]["layers"][str(layer)]
        candidate_ledger = candidate["execution_trace"]["layers"][str(layer)]
        victim_used = {
            expert
            for expert, count in enumerate(ref_ledger["victim_domain_m_by_expert"])
            if int(count) > 0
        } | {
            expert
            for expert, count in enumerate(candidate_ledger["victim_domain_m_by_expert"])
            if int(count) > 0
        }
        changed = [
            expert
            for expert in sorted(victim_used)
            if int(ref_ledger["total_m_by_expert"][expert])
            != int(candidate_ledger["total_m_by_expert"][expert])
            and int(ref_ledger["foreign_m_by_expert"][expert])
            != int(candidate_ledger["foreign_m_by_expert"][expert])
        ]
        if changed:
            shape_delta.append(
                {
                    "layer": layer,
                    "victim_used_experts_with_total_m_change": changed,
                    "reference_total_m": {
                        str(expert): int(ref_ledger["total_m_by_expert"][expert])
                        for expert in changed
                    },
                    "candidate_total_m": {
                        str(expert): int(candidate_ledger["total_m_by_expert"][expert])
                        for expert in changed
                    },
                    "reference_foreign_m": {
                        str(expert): int(ref_ledger["foreign_m_by_expert"][expert])
                        for expert in changed
                    },
                    "candidate_foreign_m": {
                        str(expert): int(candidate_ledger["foreign_m_by_expert"][expert])
                        for expert in changed
                    },
                }
            )
    delta = reference["_final_logits_cpu"] - candidate["_final_logits_cpu"]
    return {
        "router_logits_equal": bool(
            reference["victim_router_logits_sha256_by_layer"]
            == candidate["victim_router_logits_sha256_by_layer"]
        ),
        "final_logits_equal": bool(
            reference["victim_final_logits_sha256"]
            == candidate["victim_final_logits_sha256"]
        ),
        "final_logits_l2": float(torch.linalg.vector_norm(delta).item()),
        "final_logits_max_abs": float(torch.max(torch.abs(delta)).item()),
        "greedy_token_pair": [
            int(reference["victim_greedy_token_id"]),
            int(candidate["victim_greedy_token_id"]),
        ],
        "greedy_token_changed": bool(
            reference["victim_greedy_token_id"]
            != candidate["victim_greedy_token_id"]
        ),
        "ordered_route_changed_layers": ordered_layers,
        "membership_changed_layers": membership_layers,
        "foreign_shape_delta_at_victim_experts": shape_delta,
        "first_membership_divergence_layer": (
            int(membership_layers[0]) if membership_layers else None
        ),
        "foreign_m_change_precedes_first_victim_divergence": bool(
            foreign_m_precedes_divergence(shape_delta, membership_layers)
        ),
        "has_route_or_token_effect": bool(
            membership_layers
            or reference["victim_greedy_token_id"]
            != candidate["victim_greedy_token_id"]
        ),
        "has_any_numeric_effect": bool(
            reference["victim_final_logits_sha256"]
            != candidate["victim_final_logits_sha256"]
        ),
    }


def observation_signature(observation: Mapping[str, Any]) -> bytes:
    return base.canonical_json_bytes(
        {
            "final": observation["victim_final_logits_sha256"],
            "routers": observation["victim_router_logits_sha256_by_layer"],
            "routes": observation["victim_routes_by_layer"],
            "greedy": observation["victim_greedy_token_id"],
        }
    )


def rank_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    comparison = row["comparison_vs_reference"]
    return (
        int(bool(comparison["greedy_token_changed"])),
        len(comparison["membership_changed_layers"]),
        len(comparison["ordered_route_changed_layers"]),
        float(comparison["final_logits_l2"]),
        str(row["attacker_id"]),
    )


def foreign_m_precedes_divergence(
    shape_delta: Sequence[Mapping[str, Any]], membership_layers: Sequence[int]
) -> bool:
    if not shape_delta:
        return False
    if not membership_layers:
        return True
    first = int(membership_layers[0])
    return any(int(row["layer"]) < first for row in shape_delta)


def decide_verdict(validation_rows: Sequence[Mapping[str, Any]]) -> str:
    if any(bool(row["validated_strong_hit"]) for row in validation_rows):
        return "GO_EXPERT_GEMM_SHAPE_NONINTERFERENCE"
    if any(not bool(row["all_arms_stable"]) for row in validation_rows):
        return "INCONCLUSIVE_UNSTABLE"
    if any(bool(row["validated_numeric_hit"]) for row in validation_rows):
        return "WEAK_NUMERIC_CHANNEL_ONLY"
    if any(
        bool(row["unprotected_effect_stable"])
        and not bool(row["protected_exact_noninterference"])
        for row in validation_rows
    ):
        return "INCONCLUSIVE_RESIDUAL_OUTSIDE_EXPERT_SPLIT"
    return "NO_GO_NO_NATURAL_CAPACITY_FREE_CHANNEL"


def build_batch(
    attacker: Mapping[str, Any],
    victim: Mapping[str, Any],
    batch_size: int,
) -> list[list[int]]:
    return [list(attacker["window_token_ids"]) for _ in range(batch_size - 1)] + [
        list(victim["window_token_ids"])
    ]


def execute(
    model: Any,
    attacker: Mapping[str, Any],
    victim: Mapping[str, Any],
    *,
    batch_size: int,
    mode: str,
) -> dict[str, Any]:
    import torch

    input_ids = torch.tensor(
        build_batch(attacker, victim, batch_size), dtype=torch.long, device="cuda"
    )
    domains = [0] * (batch_size - 1) + [1]
    return run_observation(
        model,
        input_ids,
        mode=mode,
        request_domains=domains,
        victim_domain=1,
    )


def validate_candidate(
    model: Any,
    reference_attacker: Mapping[str, Any],
    candidate: Mapping[str, Any],
    victim: Mapping[str, Any],
    *,
    batch_size: int,
    repeats: int,
) -> dict[str, Any]:
    arms: dict[str, list[dict[str, Any]]] = {
        "unprotected_reference": [],
        "unprotected_candidate": [],
        "protected_reference": [],
        "protected_candidate": [],
    }
    comparisons: dict[str, list[dict[str, Any]]] = {
        "unprotected": [],
        "protected": [],
    }
    for _repeat in range(int(repeats)):
        unprotected_reference = execute(
            model,
            reference_attacker,
            victim,
            batch_size=batch_size,
            mode="unprotected",
        )
        unprotected_candidate = execute(
            model,
            candidate,
            victim,
            batch_size=batch_size,
            mode="unprotected",
        )
        protected_reference = execute(
            model,
            reference_attacker,
            victim,
            batch_size=batch_size,
            mode="domain_split",
        )
        protected_candidate = execute(
            model,
            candidate,
            victim,
            batch_size=batch_size,
            mode="domain_split",
        )
        arms["unprotected_reference"].append(unprotected_reference)
        arms["unprotected_candidate"].append(unprotected_candidate)
        arms["protected_reference"].append(protected_reference)
        arms["protected_candidate"].append(protected_candidate)
        comparisons["unprotected"].append(
            compare_observations(unprotected_reference, unprotected_candidate)
        )
        comparisons["protected"].append(
            compare_observations(protected_reference, protected_candidate)
        )

    arm_stable = {
        name: len({observation_signature(row) for row in rows}) == 1
        for name, rows in arms.items()
    }
    unprotected_effect_stable = all(
        bool(row["has_any_numeric_effect"])
        and bool(row["foreign_shape_delta_at_victim_experts"])
        and bool(row["foreign_m_change_precedes_first_victim_divergence"])
        for row in comparisons["unprotected"]
    )
    unprotected_strong_effect = all(
        bool(row["has_route_or_token_effect"])
        for row in comparisons["unprotected"]
    )
    protected_exact = all(
        bool(row["router_logits_equal"])
        and bool(row["final_logits_equal"])
        and not row["ordered_route_changed_layers"]
        and not row["membership_changed_layers"]
        for row in comparisons["protected"]
    )
    all_arms_stable = all(arm_stable.values())
    return {
        "schema_version": SCHEMA,
        "batch_size": int(batch_size),
        "attacker_id": str(candidate["attacker_id"]),
        "attacker_sha256": str(candidate["window_token_ids_sha256"]),
        "reference_attacker_id": str(reference_attacker["attacker_id"]),
        "reference_attacker_sha256": str(reference_attacker["window_token_ids_sha256"]),
        "repeats": int(repeats),
        "arm_stable": arm_stable,
        "all_arms_stable": bool(all_arms_stable),
        "unprotected_effect_stable": bool(unprotected_effect_stable),
        "unprotected_strong_effect": bool(unprotected_strong_effect),
        "protected_exact_noninterference": bool(protected_exact),
        "validated_strong_hit": bool(
            all_arms_stable
            and unprotected_effect_stable
            and unprotected_strong_effect
            and protected_exact
        ),
        "validated_numeric_hit": bool(
            all_arms_stable and unprotected_effect_stable and protected_exact
        ),
        "comparisons": comparisons,
        "arms": {
            name: [public_observation(row) for row in rows]
            for name, rows in arms.items()
        },
    }


def run_pilot(
    model: Any,
    victim: Mapping[str, Any],
    attackers: Sequence[Mapping[str, Any]],
    *,
    batch_sizes: Sequence[int],
    validation_top_k: int,
    repeats: int,
    output_dir: Path,
    deadline: float,
) -> dict[str, Any]:
    reference_attacker = attackers[0]
    search_rows: list[dict[str, Any]] = []
    reference_rows: list[dict[str, Any]] = []
    for batch_size in batch_sizes:
        if time.monotonic() >= deadline:
            raise PilotError("wall-time budget exhausted during search")
        reference = execute(
            model,
            reference_attacker,
            victim,
            batch_size=int(batch_size),
            mode="unprotected",
        )
        reference_rows.append(
            {
                "batch_size": int(batch_size),
                "attacker_id": reference_attacker["attacker_id"],
                "attacker_sha256": reference_attacker["window_token_ids_sha256"],
                "observation": public_observation(reference),
            }
        )
        for attacker in attackers[1:]:
            if time.monotonic() >= deadline:
                raise PilotError("wall-time budget exhausted during search")
            observation = execute(
                model,
                attacker,
                victim,
                batch_size=int(batch_size),
                mode="unprotected",
            )
            search_rows.append(
                {
                    "schema_version": SCHEMA,
                    "batch_size": int(batch_size),
                    "attacker_id": attacker["attacker_id"],
                    "attacker_sha256": attacker["window_token_ids_sha256"],
                    "reference_attacker_id": reference_attacker["attacker_id"],
                    "reference_attacker_sha256": reference_attacker[
                        "window_token_ids_sha256"
                    ],
                    "comparison_vs_reference": compare_observations(
                        reference, observation
                    ),
                    "observation": public_observation(observation),
                }
            )
    base.write_jsonl_new(output_dir / "reference_observations.jsonl", reference_rows)
    base.write_jsonl_new(output_dir / "search_results.jsonl", search_rows)

    selected: list[dict[str, Any]] = []
    for batch_size in batch_sizes:
        rows = [row for row in search_rows if int(row["batch_size"]) == int(batch_size)]
        rows.sort(key=rank_key, reverse=True)
        selected.extend(rows[: min(int(validation_top_k), len(rows))])
    validation_rows: list[dict[str, Any]] = []
    attacker_by_hash = {
        str(attacker["window_token_ids_sha256"]): attacker for attacker in attackers
    }
    for selected_row in selected:
        if time.monotonic() >= deadline:
            raise PilotError("wall-time budget exhausted during validation")
        validation_rows.append(
            validate_candidate(
                model,
                reference_attacker,
                attacker_by_hash[str(selected_row["attacker_sha256"])],
                victim,
                batch_size=int(selected_row["batch_size"]),
                repeats=int(repeats),
            )
        )
    base.write_jsonl_new(output_dir / "validation_results.jsonl", validation_rows)

    strong = [row for row in validation_rows if row["validated_strong_hit"]]
    numeric = [row for row in validation_rows if row["validated_numeric_hit"]]
    unstable = [row for row in validation_rows if not row["all_arms_stable"]]
    verdict = decide_verdict(validation_rows)
    return {
        "schema_version": SCHEMA,
        "verdict": verdict,
        "victim": {key: value for key, value in victim.items() if key != "window_token_ids"},
        "reference_attacker": {
            key: value
            for key, value in reference_attacker.items()
            if key != "window_token_ids"
        },
        "batch_sizes": list(map(int, batch_sizes)),
        "attacker_count": len(attackers),
        "search_pair_count": len(search_rows),
        "validation_candidate_count": len(validation_rows),
        "validated_strong_hit_count": len(strong),
        "validated_numeric_hit_count": len(numeric),
        "unstable_validation_count": len(unstable),
        "best_validated_strong_hit": (
            {
                "batch_size": strong[0]["batch_size"],
                "attacker_id": strong[0]["attacker_id"],
                "attacker_sha256": strong[0]["attacker_sha256"],
                "comparisons": strong[0]["comparisons"],
            }
            if strong
            else None
        ),
        "gate": {
            "strong_go_requires": [
                "same fixed batch shape and victim slot",
                "stable route-membership or greedy-token change across foreign workloads",
                "a changed total M for at least one expert used by the victim domain",
                "the foreign-row M change precedes the first victim route-membership divergence",
                "all selected contributions processed with no capacity or drop path",
                "exact removal by expert-only (expert,domain) split",
                "all four arms repeat-stable",
            ],
            "weak_numeric_is_not_paper_go": True,
        },
        "evidence_boundary": (
            "single OLMoE model, one RTX 5090 software stack, fixed 16-token windows, "
            "coarse repeated-request attackers; expert-GEMM shape channel only; not whole-model "
            "noninterference, production overhead, cross-model prevalence, or serving proof"
        ),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--victim-results", type=Path, default=DEFAULT_VICTIM_RESULTS)
    parser.add_argument(
        "--attacker-sources",
        type=Path,
        nargs="+",
        default=list(DEFAULT_ATTACKER_SOURCES),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-sizes", type=parse_int_list, default=(8, 16))
    parser.add_argument("--max-attackers", type=int, default=32)
    parser.add_argument("--validation-top-k", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--max-wall-seconds", type=int, default=3600)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise PilotError(f"refusing to reuse output directory {output_dir}")
    if args.max_attackers < 2 or args.validation_top_k < 1 or args.repeats < 2:
        raise PilotError("need >=2 attackers, >=1 validation candidate, and >=2 repeats")
    output_dir.mkdir(parents=True, exist_ok=False)
    started_at = utc_now()
    start_monotonic = time.monotonic()
    deadline = start_monotonic + int(args.max_wall_seconds)
    config = base.load_json(args.config.resolve())
    source_bindings = {
        "victim_results": verify_upstream_artifact(args.victim_results.resolve()),
        "attacker_sources": [
            verify_upstream_artifact(path.resolve()) for path in args.attacker_sources
        ],
    }
    victim = load_victim(args.victim_results.resolve())
    attackers = load_attackers(
        [path.resolve() for path in args.attacker_sources],
        str(victim["window_token_ids_sha256"]),
    )[: int(args.max_attackers)]
    if len(attackers) < 2:
        raise PilotError("max-attackers left fewer than two workloads")
    run_request = {
        "schema_version": SCHEMA,
        "started_at": started_at,
        "runner": str(Path(__file__).resolve()),
        "runner_sha256": base.sha256_file(Path(__file__).resolve()),
        "base_runner": str((STABLEBATCH_DIR / "run_single_contribution_pilot.py").resolve()),
        "base_runner_sha256": base.sha256_file(
            STABLEBATCH_DIR / "run_single_contribution_pilot.py"
        ),
        "config": str(args.config.resolve()),
        "config_sha256": base.sha256_file(args.config.resolve()),
        "victim_results": str(args.victim_results.resolve()),
        "victim_results_sha256": base.sha256_file(args.victim_results.resolve()),
        "attacker_sources": [str(path.resolve()) for path in args.attacker_sources],
        "attacker_source_sha256": {
            str(path.resolve()): base.sha256_file(path.resolve())
            for path in args.attacker_sources
        },
        "batch_sizes": list(map(int, args.batch_sizes)),
        "max_attackers": int(args.max_attackers),
        "validation_top_k": int(args.validation_top_k),
        "repeats": int(args.repeats),
        "max_wall_seconds": int(args.max_wall_seconds),
    }
    base.write_json_new(output_dir / "run_request.json", run_request)
    base.write_jsonl_new(
        output_dir / "attacker_pool.jsonl",
        (
            {key: value for key, value in attacker.items() if key != "window_token_ids"}
            for attacker in attackers
        ),
    )
    base.write_json_new(
        output_dir / "victim.json",
        {key: value for key, value in victim.items() if key != "window_token_ids"},
    )

    static_bindings = {
        "schema_version": SCHEMA,
        "runner_sha256": base.sha256_file(Path(__file__).resolve()),
        "config_sha256": base.sha256_file(args.config.resolve()),
        "model": verify_model_binding(config),
        "upstream_sources": source_bindings,
    }
    base.write_json_new(output_dir / "static_bindings.json", static_bindings)

    pre_import_gpu = base.gpu_snapshot()
    environment = base.verify_environment(config, pre_import_gpu)
    base.write_json_new(output_dir / "environment.json", environment)
    import torch

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    model, _tokenizer = base.load_model(config)
    summary = run_pilot(
        model,
        victim,
        attackers,
        batch_sizes=args.batch_sizes,
        validation_top_k=int(args.validation_top_k),
        repeats=int(args.repeats),
        output_dir=output_dir,
        deadline=deadline,
    )
    runtime_final = base.verify_final_runtime(config)
    base.write_json_new(output_dir / "runtime_final.json", runtime_final)
    summary.update(
        {
            "started_at": started_at,
            "completed_at": utc_now(),
            "wall_seconds": float(time.monotonic() - start_monotonic),
        }
    )
    base.write_json_new(output_dir / "summary.json", summary)
    manifest_files = [
        "run_request.json",
        "environment.json",
        "static_bindings.json",
        "runtime_final.json",
        "victim.json",
        "attacker_pool.jsonl",
        "reference_observations.jsonl",
        "search_results.jsonl",
        "validation_results.jsonl",
        "summary.json",
    ]
    base.write_json_new(
        output_dir / "COMPLETE.json",
        {
            "schema_version": SCHEMA,
            "status": "COMPLETE",
            "verdict": summary["verdict"],
            "files": {
                name: base.sha256_file(output_dir / name) for name in manifest_files
            },
        },
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
