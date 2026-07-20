"""Owner-local MoE combine semantics for offline quality experiments.

This module models a *non-expanded / local-reduction* EP combine path: it
returns one locally reduced vector per ``(token origin, expert owner, token)``
instead of one vector per routed expert pair.  DeepEP-style backends can also
use expanded layouts, and LL kernels commonly transmit per-expert responses;
the grouped path here must therefore be treated as a backend-specific HT/local-
reduction counterfactual rather than the universal EP wire unit.  We multiply
every expert output by its router weight and accumulate colocated experts in
BF16 before applying fake FP8/MXFP4.

The implementation is deliberately a numerical reference.  It does not model
GPU packing, alignment, overlap, collectives, or kernel latency.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from fake_quant import apply_precision


GROUPED_POLICIES = (
    "grouped_bf16",
    "uniform_fp8",
    "uniform_mxfp4",
    "mixed_rank",
    "mixed_gate_mass",
    "mixed_inputnorm_gate",
    "mixed_profiled_gain",
    "mixed_pair_contribution",
    "mixed_contribution",
    "global_contribution",
    "token_contribution",
    "mixed_qerr",
    "mixed_oracle",
    "mixed_random",
)


@dataclass
class GroupedOwnerBatch:
    """Owner-local vectors and the statistics needed by fixed-quota selectors."""

    vectors: torch.Tensor  # [tokens, owners, hidden], BF16/model dtype
    present: torch.Tensor  # [tokens, owners]
    gate_mass: torch.Tensor  # [tokens, owners], float32
    rank_mass: torch.Tensor  # [tokens, owners], float32
    pair_contribution: torch.Tensor  # sum_i g_i ||o_i||, float32
    profiled_gain_mass: torch.Tensor | None  # sum_i g_i alpha_e, float32
    pair_count: torch.Tensor  # [tokens, owners], int64


def expert_owner_ids(
    expert_ids: torch.Tensor,
    num_experts: int,
    ep_size: int,
    mapping: str = "contiguous",
) -> torch.Tensor:
    """Map global expert ids to owners for common static EP placements."""
    if ep_size < 1:
        raise ValueError("ep_size must be positive")
    if num_experts < 1:
        raise ValueError("num_experts must be positive")
    if ep_size > num_experts:
        raise ValueError(f"ep_size={ep_size} exceeds num_experts={num_experts}")
    if expert_ids.numel() and (
        int(expert_ids.min().item()) < 0
        or int(expert_ids.max().item()) >= num_experts
    ):
        raise ValueError("expert id outside [0, num_experts)")
    if mapping == "contiguous":
        # This also handles num_experts not divisible by ep_size.
        return torch.div(
            expert_ids * ep_size, num_experts, rounding_mode="floor"
        ).clamp_max(ep_size - 1)
    if mapping in ("round_robin", "mod"):
        return expert_ids.remainder(ep_size)
    raise ValueError(f"unknown expert-owner mapping: {mapping}")


def group_owner_outputs(
    raw_outputs: torch.Tensor,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    num_experts: int,
    ep_size: int,
    mapping: str,
    expert_gain_profile: torch.Tensor | None = None,
) -> GroupedOwnerBatch:
    """BF16-reduce routed expert outputs into token-owner wire vectors.

    Experts are visited in global expert-id order, matching the reference
    Transformers combine loop within each owner.  The owner grouping changes
    BF16 associativity when ``ep_size > 1``; callers must compare against the
    returned grouped-BF16 reference rather than silently treating it as the
    original single-process accumulation order.
    """
    if raw_outputs.ndim != 3:
        raise ValueError("raw_outputs must have shape [tokens, top_k, hidden]")
    if routing_weights.shape != raw_outputs.shape[:2]:
        raise ValueError("routing_weights shape does not match raw_outputs")
    if selected_experts.shape != raw_outputs.shape[:2]:
        raise ValueError("selected_experts shape does not match raw_outputs")

    token_count, top_k, hidden_dim = raw_outputs.shape
    owners = expert_owner_ids(selected_experts, num_experts, ep_size, mapping)
    vectors = torch.zeros(
        (token_count, ep_size, hidden_dim),
        dtype=raw_outputs.dtype,
        device=raw_outputs.device,
    )
    gate_mass = torch.zeros(
        (token_count, ep_size), dtype=torch.float32, device=raw_outputs.device
    )
    rank_mass = torch.zeros_like(gate_mass)
    pair_contribution = torch.zeros_like(gate_mass)
    profiled_gain_mass = (
        torch.zeros_like(gate_mass) if expert_gain_profile is not None else None
    )
    pair_count = torch.zeros(
        (token_count, ep_size), dtype=torch.int64, device=raw_outputs.device
    )

    # A fixed, gate-free rank prior.  Summing reciprocal ranks accounts for a
    # token-owner group containing multiple routed ranks without observing
    # expert outputs.  This is the grouped analogue of a fixed-rank baseline,
    # not the old per-pair tail layout.
    reciprocal_rank = 1.0 / torch.arange(
        1, top_k + 1, dtype=torch.float32, device=raw_outputs.device
    )
    raw_norm = raw_outputs.float().norm(dim=-1)
    if expert_gain_profile is not None:
        if expert_gain_profile.shape != (num_experts,):
            raise ValueError(
                "expert_gain_profile must have shape [num_experts]"
            )
        expert_gain_profile = expert_gain_profile.to(
            device=raw_outputs.device, dtype=torch.float32
        )

    for expert_idx in range(num_experts):
        token_idx, rank_idx = torch.where(selected_experts == expert_idx)
        if token_idx.numel() == 0:
            continue
        owner = int(
            expert_owner_ids(
                torch.tensor(expert_idx, device=selected_experts.device),
                num_experts,
                ep_size,
                mapping,
            ).item()
        )
        weighted = raw_outputs[token_idx, rank_idx] * routing_weights[
            token_idx, rank_idx, None
        ]
        vectors[:, owner, :].index_add_(
            0, token_idx, weighted.to(dtype=raw_outputs.dtype)
        )
        gate_mass[:, owner].index_add_(
            0, token_idx, routing_weights[token_idx, rank_idx].float()
        )
        rank_mass[:, owner].index_add_(
            0, token_idx, reciprocal_rank[rank_idx]
        )
        pair_contribution[:, owner].index_add_(
            0,
            token_idx,
            routing_weights[token_idx, rank_idx].float()
            * raw_norm[token_idx, rank_idx],
        )
        if profiled_gain_mass is not None and expert_gain_profile is not None:
            profiled_gain_mass[:, owner].index_add_(
                0,
                token_idx,
                routing_weights[token_idx, rank_idx].float()
                * expert_gain_profile[expert_idx],
            )
        pair_count[:, owner].index_add_(
            0, token_idx, torch.ones_like(token_idx, dtype=torch.int64)
        )

    return GroupedOwnerBatch(
        vectors=vectors,
        present=pair_count > 0,
        gate_mass=gate_mass,
        rank_mass=rank_mass,
        pair_contribution=pair_contribution,
        profiled_gain_mass=profiled_gain_mass,
        pair_count=pair_count,
    )


def combine_owner_order(vectors: torch.Tensor) -> torch.Tensor:
    """Accumulate owner vectors in owner-id order using model dtype."""
    if vectors.ndim != 3:
        raise ValueError("vectors must have shape [tokens, owners, hidden]")
    final = torch.zeros(
        (vectors.shape[0], vectors.shape[2]),
        dtype=vectors.dtype,
        device=vectors.device,
    )
    for owner in range(vectors.shape[1]):
        final.add_(vectors[:, owner, :])
    return final


def _quantize_present(
    vectors: torch.Tensor, present: torch.Tensor, precision: str
) -> torch.Tensor:
    out = vectors.clone()
    if bool(present.any()):
        out[present] = apply_precision(vectors[present], precision)
    return out


def deterministic_random_scores(reference: torch.Tensor) -> torch.Tensor:
    """Stable rank/output-independent anti-control scores."""
    indices = torch.arange(
        reference.numel(), dtype=torch.int64, device=reference.device
    )
    hashed = (indices * 1103515245 + 12345) % 2147483647
    return hashed.reshape_as(reference).float()


def fixed_quota_high_mask(
    scores: torch.Tensor,
    present: torch.Tensor,
    tile_vectors: int,
    high_fraction: float,
) -> tuple[torch.Tensor, list[dict[str, int | float | str]]]:
    """Select an exact high-precision cardinality in every owner stream tile.

    A tile is a contiguous chunk of *present* token-owner vectors for one owner.
    Every selector receives the same deterministic ``floor(n * high_fraction)``
    FP8 slots for a tile of ``n`` vectors. Stable sorting plus token order breaks
    ties. Production decode needs an explicit short-tile fallback/carry rule;
    this numerical reference deliberately does not wait to fill a tile.
    """
    if scores.shape != present.shape:
        raise ValueError("scores and present must have identical shape")
    if tile_vectors < 1:
        raise ValueError("tile_vectors must be positive")
    if not 0.0 <= high_fraction <= 1.0:
        raise ValueError("high_fraction must be in [0, 1]")

    high = torch.zeros_like(present, dtype=torch.bool)
    tile_rows: list[dict[str, int | float | str]] = []
    for owner in range(present.shape[1]):
        token_positions = torch.nonzero(
            present[:, owner], as_tuple=False
        ).reshape(-1)
        for tile_index, start in enumerate(
            range(0, int(token_positions.numel()), tile_vectors)
        ):
            positions = token_positions[start : start + tile_vectors]
            high_count = int(math.floor(int(positions.numel()) * high_fraction))
            if high_count > 0:
                values = scores[positions, owner]
                order = torch.argsort(values, descending=True, stable=True)
                high[positions[order[:high_count]], owner] = True
            tile_rows.append(
                {
                    "scope": "peer",
                    "owner": owner,
                    "tile_index": tile_index,
                    "vectors": int(positions.numel()),
                    "high_vectors": high_count,
                    "low_vectors": int(positions.numel()) - high_count,
                    "target_high_fraction": float(high_fraction),
                }
            )
    return high, tile_rows


def global_high_mask(
    scores: torch.Tensor,
    present: torch.Tensor,
    high_fraction: float,
) -> tuple[torch.Tensor, list[dict[str, int | float | str]]]:
    """Global exact-cardinality quality upper bound with variable peer counts."""
    if scores.shape != present.shape:
        raise ValueError("scores and present must have identical shape")
    if not 0.0 <= high_fraction <= 1.0:
        raise ValueError("high_fraction must be in [0, 1]")
    positions = torch.nonzero(present.reshape(-1), as_tuple=False).reshape(-1)
    high_count = int(math.floor(int(positions.numel()) * high_fraction))
    high_flat = torch.zeros_like(present.reshape(-1), dtype=torch.bool)
    if high_count > 0:
        values = scores.reshape(-1)[positions]
        order = torch.argsort(values, descending=True, stable=True)
        high_flat[positions[order[:high_count]]] = True
    return high_flat.reshape_as(present), [
        {
            "scope": "global",
            "owner": -1,
            "tile_index": 0,
            "vectors": int(positions.numel()),
            "high_vectors": high_count,
            "low_vectors": int(positions.numel()) - high_count,
            "target_high_fraction": float(high_fraction),
        }
    ]


def token_high_mask(
    scores: torch.Tensor,
    present: torch.Tensor,
    high_fraction: float,
) -> tuple[torch.Tensor, list[dict[str, int | float | str]]]:
    """Per-token exact quota; a quality baseline with variable peer lane sizes."""
    if scores.shape != present.shape:
        raise ValueError("scores and present must have identical shape")
    if not 0.0 <= high_fraction <= 1.0:
        raise ValueError("high_fraction must be in [0, 1]")
    high = torch.zeros_like(present, dtype=torch.bool)
    rows: list[dict[str, int | float | str]] = []
    for token in range(present.shape[0]):
        owners = torch.nonzero(present[token], as_tuple=False).reshape(-1)
        high_count = int(math.floor(int(owners.numel()) * high_fraction))
        if high_count > 0:
            order = torch.argsort(
                scores[token, owners], descending=True, stable=True
            )
            high[token, owners[order[:high_count]]] = True
        rows.append(
            {
                "scope": "token",
                "owner": -1,
                "tile_index": token,
                "vectors": int(owners.numel()),
                "high_vectors": high_count,
                "low_vectors": int(owners.numel()) - high_count,
                "target_high_fraction": float(high_fraction),
            }
        )
    return high, rows


def _score_owner_vectors(
    mode: str,
    grouped: GroupedOwnerBatch,
    low_vectors: torch.Tensor,
    high_vectors: torch.Tensor,
    input_norm: torch.Tensor | None,
) -> torch.Tensor:
    if mode == "rank":
        return grouped.rank_mass
    if mode == "gate_mass":
        return grouped.gate_mass
    if mode == "inputnorm_gate":
        if input_norm is None:
            raise ValueError("inputnorm_gate requires input_norm")
        return input_norm.float()[:, None] * grouped.gate_mass
    if mode == "profiled_gain":
        if input_norm is None or grouped.profiled_gain_mass is None:
            raise ValueError(
                "profiled_gain requires input_norm and a calibration-only "
                "expert_gain_profile"
            )
        return input_norm.float()[:, None] * grouped.profiled_gain_mass
    if mode == "pair_contribution":
        # Sum_i g_i ||o_i|| is the pre-aggregation upper bound. Comparing it
        # with ||sum_i g_i o_i|| isolates the value of owner-local cancellation.
        return grouped.pair_contribution
    if mode == "contribution":
        # The norm of the actual owner-local wire vector captures cancellation
        # among experts colocated on the same owner.  pair_contribution remains
        # available in GroupedOwnerBatch for future ablations.
        return grouped.vectors.float().norm(dim=-1)
    if mode == "qerr":
        return (low_vectors.float() - grouped.vectors.float()).square().sum(dim=-1)
    if mode == "random":
        return deterministic_random_scores(grouped.gate_mass)
    if mode == "oracle":
        # Exact one-group local intervention from all-MXFP4 to FP8.  It includes
        # same-token cross-owner error terms.  Selecting multiple groups remains
        # a scored approximation because their intervention gains interact.
        reference = combine_owner_order(grouped.vectors).float()
        all_low = combine_owner_order(low_vectors).float()
        baseline_error = (all_low - reference).square().sum(dim=-1)
        score = torch.zeros_like(grouped.gate_mass)
        for owner in range(grouped.vectors.shape[1]):
            upgraded = low_vectors.clone()
            upgraded[:, owner, :] = high_vectors[:, owner, :]
            upgraded_final = combine_owner_order(upgraded).float()
            upgraded_error = (upgraded_final - reference).square().sum(dim=-1)
            score[:, owner] = baseline_error - upgraded_error
        return score
    raise ValueError(f"unknown grouped selector mode: {mode}")


def grouped_owner_combine(
    raw_outputs: torch.Tensor,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    num_experts: int,
    ep_size: int,
    mapping: str,
    policy: str,
    tile_vectors: int = 64,
    high_fraction: float = 0.5,
    input_norm: torch.Tensor | None = None,
    expert_gain_profile: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    """Return grouped-BF16 reference, approximation, and diagnostics."""
    if policy not in GROUPED_POLICIES:
        raise ValueError(f"unknown grouped policy: {policy}")
    grouped = group_owner_outputs(
        raw_outputs,
        routing_weights,
        selected_experts,
        num_experts,
        ep_size,
        mapping,
        expert_gain_profile=expert_gain_profile,
    )
    reference = combine_owner_order(grouped.vectors)
    low_vectors = _quantize_present(grouped.vectors, grouped.present, "mxfp4")
    high_vectors = _quantize_present(grouped.vectors, grouped.present, "fp8")
    tile_rows: list[dict[str, int | float | str]] = []

    if policy == "grouped_bf16":
        encoded = grouped.vectors
        high_mask = torch.zeros_like(grouped.present)
        low_mask = torch.zeros_like(grouped.present)
        bf16_mask = grouped.present
    elif policy == "uniform_fp8":
        encoded = high_vectors
        high_mask = grouped.present
        low_mask = torch.zeros_like(grouped.present)
        bf16_mask = torch.zeros_like(grouped.present)
    elif policy == "uniform_mxfp4":
        encoded = low_vectors
        high_mask = torch.zeros_like(grouped.present)
        low_mask = grouped.present
        bf16_mask = torch.zeros_like(grouped.present)
    else:
        if policy in ("global_contribution", "token_contribution"):
            score_mode = "contribution"
        else:
            score_mode = policy[len("mixed_") :]
        scores = _score_owner_vectors(
            score_mode, grouped, low_vectors, high_vectors, input_norm
        )
        if policy == "global_contribution":
            high_mask, tile_rows = global_high_mask(
                scores, grouped.present, high_fraction
            )
        elif policy == "token_contribution":
            high_mask, tile_rows = token_high_mask(
                scores, grouped.present, high_fraction
            )
        else:
            high_mask, tile_rows = fixed_quota_high_mask(
                scores, grouped.present, tile_vectors, high_fraction
            )
        low_mask = grouped.present & ~high_mask
        bf16_mask = torch.zeros_like(grouped.present)
        encoded = grouped.vectors.clone()
        if bool(high_mask.any()):
            encoded[high_mask] = high_vectors[high_mask]
        if bool(low_mask.any()):
            encoded[low_mask] = low_vectors[low_mask]

    approximation = combine_owner_order(encoded)
    routed_pairs = int(grouped.pair_count.sum().item())
    grouped_vectors = int(grouped.present.sum().item())
    multi_expert_vectors = int((grouped.pair_count > 1).sum().item())
    diagnostics: dict[str, object] = {
        "routed_pairs": routed_pairs,
        "grouped_vectors": grouped_vectors,
        "collision_pairs": routed_pairs - grouped_vectors,
        "multi_expert_vectors": multi_expert_vectors,
        "max_pairs_per_vector": int(grouped.pair_count.max().item()),
        "bf16_vectors": int(bf16_mask.sum().item()),
        "high_vectors": int(high_mask.sum().item()),
        "low_vectors": int(low_mask.sum().item()),
        "tile_count": len(tile_rows),
        "tile_rows": tile_rows,
        "pair_contribution_sum": float(
            grouped.pair_contribution[grouped.present].sum().item()
        ),
    }
    return reference, approximation, diagnostics


def estimate_expert_gain_profile(
    raw_outputs: torch.Tensor,
    input_norm: torch.Tensor,
    selected_experts: torch.Tensor,
    num_experts: int,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return calibration sums/counts for alpha_e = E[||o_e|| / ||h||].

    Callers must aggregate the returned sums and counts on a calibration split,
    freeze ``alpha = sums / counts``, and only then evaluate ``profiled_gain``
    on disjoint data.  Keeping this helper stateless makes split provenance an
    explicit responsibility of the experiment driver.
    """
    if input_norm.shape != (raw_outputs.shape[0],):
        raise ValueError("input_norm must have shape [tokens]")
    ratios = raw_outputs.float().norm(dim=-1) / input_norm.float()[:, None].clamp_min(
        eps
    )
    sums = torch.zeros(
        num_experts, dtype=torch.float64, device=raw_outputs.device
    )
    counts = torch.zeros_like(sums)
    sums.index_add_(0, selected_experts.reshape(-1), ratios.reshape(-1).double())
    counts.index_add_(
        0,
        selected_experts.reshape(-1),
        torch.ones_like(ratios.reshape(-1), dtype=torch.float64),
    )
    return sums, counts


def grouped_wire_bytes(
    hidden_size: int,
    bf16_vectors: int,
    high_vectors: int,
    low_vectors: int,
) -> int:
    """Numerical-format payload plus per-vector/block scales, no padding."""
    bf16_bytes = 2 * hidden_size
    fp8_bytes = hidden_size + 4
    mxfp4_bytes = math.ceil(hidden_size / 2) + math.ceil(hidden_size / 32)
    return (
        bf16_vectors * bf16_bytes
        + high_vectors * fp8_bytes
        + low_vectors * mxfp4_bytes
    )
