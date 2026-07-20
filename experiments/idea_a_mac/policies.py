from __future__ import annotations

from dataclasses import dataclass
import re

import torch

from fake_quant import apply_precision

BYTE_SIZES = {
    "bf16": 2.0,
    "full": 2.0,
    "fp8": 1.0,
    "int8": 1.0,
    "int4": 0.5,
    "mxfp4": 0.5,
    "nvfp4": 0.5,
    "drop": 0.0,
}
# Precisions that lower byte cost vs BF16. ``drop`` is handled separately because
# it zeros the output and optionally renormalizes gate weights.
REDUCE_PRECISIONS = ("fp8", "int8", "int4", "mxfp4", "nvfp4")


def _decode_policy_float(value: str) -> float:
    return float(value.replace("p", "."))


def _decode_milli_fraction(value: str) -> float:
    fraction = int(value) / 1000.0
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"encoded fraction must be in [0, 1000], got {value}")
    return fraction


def _gate_threshold_mask(routing_weights: torch.Tensor, threshold: float) -> torch.Tensor:
    return routing_weights.float() <= threshold


def _gate_tailmass_mask(routing_weights: torch.Tensor, tail_mass: float) -> torch.Tensor:
    weights = routing_weights.float()
    suffix_mass = weights.flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
    return suffix_mass <= tail_mass


def _block_budget_mask(
    scores: torch.Tensor,
    block_tokens: int,
    low_bit_fraction: float = 0.5,
) -> torch.Tensor:
    """Select a fixed number of lowest-score pairs in each contiguous token block.

    The output has exactly ``round(block_pairs * low_bit_fraction)`` selected
    entries per block (including the final short block).  This is the quality-side
    proxy for a fixed-rate two-lane communication tile: selection is dynamic,
    while the FP8/low-bit composition is deterministic conditional on the number
    of routed pairs.  It does not make the total EP message volume constant.
    """
    if scores.ndim != 2:
        raise ValueError(f"scores must be [tokens, top_k], got {tuple(scores.shape)}")
    if block_tokens < 1:
        raise ValueError("block_tokens must be positive")
    if not 0.0 <= low_bit_fraction <= 1.0:
        raise ValueError("low_bit_fraction must be in [0, 1]")

    mask = torch.zeros_like(scores, dtype=torch.bool)
    token_count = scores.shape[0]
    for start in range(0, token_count, block_tokens):
        stop = min(start + block_tokens, token_count)
        flat_scores = scores[start:stop].reshape(-1)
        low_count = int(round(flat_scores.numel() * low_bit_fraction))
        if low_count <= 0:
            continue
        if low_count >= flat_scores.numel():
            mask[start:stop] = True
            continue
        # Deterministic lexicographic tie-break.  Router top-k is ordered from
        # high to low rank, so reversing before a stable ascending sort makes
        # equal weights prefer the later (lower-importance) rank.  In
        # particular, block_tokens=1 is then exactly equivalent to a fixed
        # rank-tail split even when BF16 routing weights contain ties.
        reverse_order = torch.argsort(flat_scores.flip(0), stable=True)
        selected = flat_scores.numel() - 1 - reverse_order[:low_count]
        flat_mask = mask[start:stop].reshape(-1)
        flat_mask[selected] = True
    return mask


def _deterministic_random_scores(reference: torch.Tensor) -> torch.Tensor:
    """Return a reproducible rank-independent control score for each pair.

    The integer hash avoids consuming global RNG state, so the mask is identical
    in the numerical path and the byte-accounting path.  It is an anti-control,
    not a claim about production randomization.
    """
    indices = torch.arange(
        reference.numel(), dtype=torch.int64, device=reference.device
    )
    hashed = (indices * 1103515245 + 12345) % 2147483647
    return hashed.reshape_as(reference).to(dtype=torch.float32)


def _pair_criticality_scores(
    mode: str,
    raw_outputs: torch.Tensor,
    routing_weights: torch.Tensor,
    low_precision: str,
) -> torch.Tensor:
    """Score the benefit of keeping/refining a routed expert-output pair.

    Larger values mean "more critical".  `_block_budget_mask` therefore assigns
    the lowest-scoring pairs to the low-bit-only lane.  The error-based scores
    are deliberately separated into cheap proxies and expensive upper bounds:

    - qenergy/resenergy: ||o-Q4(o)||^2, owner-local and gate-free;
    - qerr/reserr: g^2 ||o-Q4(o)||^2, available after the all-pair base quantizer;
    - qbenefit: exact additive error-energy reduction from Q4 to fake FP8;
    - resbenefit: exact additive error-energy reduction from Q4 to Q4+Q4(residual).

    These are pair-additive proxies.  They do not include cross-expert error
    cancellation inside a token or downstream nonlinear propagation.
    """
    weights = routing_weights.float()
    if mode == "gate":
        return weights
    if mode == "reversegate":
        return -weights
    if mode == "random":
        return _deterministic_random_scores(weights)
    if mode == "contrib":
        return weights * raw_outputs.float().norm(dim=-1)

    base = apply_precision(raw_outputs, low_precision)
    base_error = raw_outputs.float() - base.float()
    base_energy = base_error.square().sum(dim=-1)
    if mode in ("qenergy", "resenergy"):
        return base_energy
    weighted_base_energy = weights.square() * base_energy
    if mode in ("qerr", "reserr"):
        return weighted_base_energy
    if mode == "qbenefit":
        high = apply_precision(raw_outputs, "fp8")
        high_error = raw_outputs.float() - high.float()
        high_energy = high_error.square().sum(dim=-1)
        return weights.square() * (base_energy - high_energy)
    if mode == "resbenefit":
        residual_q = apply_precision(base_error, low_precision)
        refined_error = base_error - residual_q.float()
        refined_energy = refined_error.square().sum(dim=-1)
        return weights.square() * (base_energy - refined_energy)
    raise ValueError(f"unknown criticality score mode: {mode}")


def _peer_block_budget_mask(
    scores: torch.Tensor,
    group_ids: torch.Tensor,
    block_pairs: int,
    low_bit_fraction: float = 0.5,
) -> torch.Tensor:
    """Fixed-budget selection independently inside each peer's pair stream."""
    if scores.shape != group_ids.shape:
        raise ValueError("scores and group_ids must have identical [tokens, top_k] shape")
    if block_pairs < 1:
        raise ValueError("block_pairs must be positive")
    if not 0.0 <= low_bit_fraction <= 1.0:
        raise ValueError("low_bit_fraction must be in [0, 1]")

    flat_scores = scores.reshape(-1)
    flat_groups = group_ids.reshape(-1)
    flat_mask = torch.zeros_like(flat_scores, dtype=torch.bool)
    for group in torch.unique(flat_groups, sorted=True):
        positions = torch.nonzero(flat_groups == group, as_tuple=False).reshape(-1)
        for start in range(0, positions.numel(), block_pairs):
            block_positions = positions[start : start + block_pairs]
            low_count = int(round(block_positions.numel() * low_bit_fraction))
            if low_count <= 0:
                continue
            if low_count >= block_positions.numel():
                flat_mask[block_positions] = True
                continue
            values = flat_scores[block_positions]
            reverse_order = torch.argsort(values.flip(0), stable=True)
            selected_local = values.numel() - 1 - reverse_order[:low_count]
            flat_mask[block_positions[selected_local]] = True
    return flat_mask.reshape_as(scores)


def receiver_group_ids(
    expert_ids: torch.Tensor,
    num_experts: int,
    num_receiver_groups: int,
    receiver_mapping: str = "contiguous",
) -> torch.Tensor:
    if num_receiver_groups <= 1:
        return torch.zeros_like(expert_ids)
    if receiver_mapping == "mod":
        return expert_ids % num_receiver_groups
    if receiver_mapping == "contiguous":
        return torch.div(expert_ids * num_receiver_groups, num_experts, rounding_mode="floor").clamp_max(
            num_receiver_groups - 1
        )
    raise ValueError(f"unknown receiver mapping: {receiver_mapping}")


@dataclass(frozen=True)
class ApproxPolicy:
    name: str
    drop_renorm: bool = False
    layer_lut: dict | None = None  # {(receiver_group, rank) -> precision}

    def rank_index(self, top_k: int, suffix: str) -> int | None:
        if suffix == "k":
            return top_k - 1
        value = int(suffix)
        if value < 1 or value > top_k:
            raise ValueError(f"rank {value} out of range for top_k={top_k}")
        return value - 1

    def bytes_per_element_by_rank(self, top_k: int) -> list[float]:
        if self.name == "full":
            return [2.0] * top_k
        for prec in REDUCE_PRECISIONS:
            if self.name == f"uniform_{prec}":
                return [BYTE_SIZES[prec]] * top_k
            match = re.fullmatch(rf"rank(\d+|k)_{prec}", self.name)
            if match:
                values = [2.0] * top_k
                values[self.rank_index(top_k, match.group(1))] = BYTE_SIZES[prec]
                return values
        match = re.fullmatch(r"rank(\d+|k)_drop(_renorm)?", self.name)
        if match:
            values = [2.0] * top_k
            values[self.rank_index(top_k, match.group(1))] = 0.0
            return values
        match = re.fullmatch(r"keep(\d+)_drop(_renorm)?", self.name)
        if match:
            n = min(int(match.group(1)), top_k)
            return [2.0] * n + [0.0] * (top_k - n)
        match = re.fullmatch(r"keep(\d+)_bf16_rest_(fp8|int8|int4|mxfp4|nvfp4)", self.name)
        if match:
            n = int(match.group(1))
            prec = match.group(2)
            return [2.0] * min(n, top_k) + [BYTE_SIZES[prec]] * max(top_k - n, 0)
        match = re.fullmatch(r"fp8top(\d+)_rest_(fp8|int8|int4|mxfp4|nvfp4)", self.name)
        if match:
            n = int(match.group(1))
            prec = match.group(2)
            return [1.0] * min(n, top_k) + [BYTE_SIZES[prec]] * max(top_k - n, 0)
        match = re.fullmatch(r"contrib_tail(\d+)_(fp8|int8|int4|mxfp4|nvfp4)", self.name)
        if match:
            n = int(match.group(1))
            prec = match.group(2)
            avg = (n * BYTE_SIZES[prec] + (top_k - n) * 1.0) / top_k
            return [avg] * top_k
        if re.fullmatch(
            r"gate_(threshold|tailmass)_\d+p\d+_(int4|mxfp4|nvfp4)",
            self.name,
        ):
            # Dynamic policies need observed routing weights for exact accounting.
            # Return the FP8 base here; MoeRecorder computes actual bytes per pair.
            return [1.0] * top_k
        match = re.fullmatch(r"block_gate(\d+)_(int4|mxfp4|nvfp4)", self.name)
        if match:
            # Exact accounting needs observed routing weights.  At 50/50 the
            # aggregate average is known, but return the FP8 base here so callers
            # cannot mistake this for a rank-static layout.
            return [1.0] * top_k
        match = re.fullmatch(
            r"block_gate(\d+)_f(\d+)_(int4|mxfp4|nvfp4)", self.name
        )
        if match:
            return [1.0] * top_k
        match = re.fullmatch(
            r"block_gate(\d+)_residual_(int4|mxfp4|nvfp4)", self.name
        )
        if match:
            # Every pair carries one 4-bit base; the critical half carries one
            # additional 4-bit residual.  Aggregate raw payload matches the
            # 50/50 FP8/FP4 policy, but metadata differs and is reported by the
            # experiment driver.
            return [0.75] * top_k
        if re.fullmatch(
            r"block_(contrib|qenergy|qerr|qbenefit|random|reversegate)\d+"
            r"(?:_f\d+)?_(int4|mxfp4|nvfp4)",
            self.name,
        ):
            return [1.0] * top_k
        if re.fullmatch(
            r"block_(contrib|resenergy|reserr|resbenefit|random|reversegate)\d+"
            r"_residual_(int4|mxfp4|nvfp4)",
            self.name,
        ):
            return [0.75] * top_k
        if re.fullmatch(
            r"peerblock_gate\d+_residual_(int4|mxfp4|nvfp4)", self.name
        ):
            return [0.75] * top_k
        if re.fullmatch(r"peerblock_gate\d+_(int4|mxfp4|nvfp4)", self.name):
            return [1.0] * top_k
        if re.fullmatch(
            r"peerblock_gate\d+_f\d+_(int4|mxfp4|nvfp4)", self.name
        ):
            return [1.0] * top_k
        if re.fullmatch(
            r"peerblock_(contrib|qenergy|qerr|qbenefit|random|reversegate)\d+"
            r"(?:_f\d+)?_(int4|mxfp4|nvfp4)",
            self.name,
        ):
            return [1.0] * top_k
        if re.fullmatch(
            r"peerblock_(contrib|resenergy|reserr|resbenefit|random|reversegate)\d+"
            r"_residual_(int4|mxfp4|nvfp4)",
            self.name,
        ):
            return [0.75] * top_k
        match = re.fullmatch(r"group\d+_rank(\d+|k)_(fp8|int[48]|mxfp4|nvfp4)", self.name)
        if match:
            return [2.0] * top_k
        match = re.fullmatch(r"group\d+_rank(\d+|k)_drop(_renorm)?", self.name)
        if match:
            return [2.0] * top_k
        if self.name == "rankk_int8":
            return [2.0] * (top_k - 1) + [1.0]
        if self.name == "rankk_int4":
            return [2.0] * (top_k - 1) + [0.5]
        if self.name == "rank1_int4":
            return [0.5] + [2.0] * (top_k - 1)
        if self.name == "rankk_drop":
            return [2.0] * (top_k - 1) + [0.0]
        if self.name == "rankk_drop_renorm":
            return [2.0] * (top_k - 1) + [0.0]
        if self.name == "rank1_drop":
            return [0.0] + [2.0] * (top_k - 1)
        raise ValueError(f"unknown strategy: {self.name}")

    def bytes_per_element_for_selected(
        self,
        selected_experts: torch.Tensor,
        num_experts: int,
        num_receiver_groups: int = 1,
        receiver_mapping: str = "contiguous",
        routing_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        top_k = selected_experts.shape[1]
        values = torch.full(selected_experts.shape, 2.0, dtype=torch.float32, device=selected_experts.device)

        if self.layer_lut is not None:
            group_ids = receiver_group_ids(
                selected_experts, num_experts, num_receiver_groups, receiver_mapping
            )
            for rank_idx in range(top_k):
                rank = rank_idx + 1
                for group in range(num_receiver_groups):
                    mask = group_ids[:, rank_idx] == group
                    if mask.any():
                        precision = self.layer_lut.get((group, rank), "bf16")
                        values[mask, rank_idx] = BYTE_SIZES.get(precision, 2.0)
            return values

        if self.name == "full":
            return values
        for prec in REDUCE_PRECISIONS:
            if self.name == f"uniform_{prec}":
                return torch.full_like(values, BYTE_SIZES[prec])
            match = re.fullmatch(rf"rank(\d+|k)_{prec}", self.name)
            if match:
                values[:, self.rank_index(top_k, match.group(1))] = BYTE_SIZES[prec]
                return values
        match = re.fullmatch(r"rank(\d+|k)_drop(_renorm)?", self.name)
        if match:
            values[:, self.rank_index(top_k, match.group(1))] = 0.0
            return values
        match = re.fullmatch(r"keep(\d+)_drop(_renorm)?", self.name)
        if match:
            n = min(int(match.group(1)), top_k)
            values[:, n:] = 0.0
            return values

        match = re.fullmatch(r"keep(\d+)_bf16_rest_(fp8|int8|int4|mxfp4|nvfp4)", self.name)
        if match:
            n = int(match.group(1))
            prec = match.group(2)
            for rank_idx in range(top_k):
                if rank_idx >= n:
                    values[:, rank_idx] = BYTE_SIZES[prec]
            return values

        match = re.fullmatch(r"fp8top(\d+)_rest_(fp8|int8|int4|mxfp4|nvfp4)", self.name)
        if match:
            n = int(match.group(1))
            prec = match.group(2)
            values = torch.full_like(values, 1.0)  # FP8 baseline for all
            for rank_idx in range(top_k):
                if rank_idx >= n:
                    values[:, rank_idx] = BYTE_SIZES[prec]
            return values

        # contrib_tail: per-token selection varies, but aggregate byte saving is
        # fixed (n at prec + rest at FP8). Return uniform average for all positions.
        match = re.fullmatch(r"contrib_tail(\d+)_(fp8|int8|int4|mxfp4|nvfp4)", self.name)
        if match:
            n = int(match.group(1))
            prec = match.group(2)
            avg = (n * BYTE_SIZES[prec] + (top_k - n) * 1.0) / top_k
            return torch.full_like(values, avg)

        match = re.fullmatch(
            r"gate_threshold_(\d+p\d+)_(int4|mxfp4|nvfp4)", self.name
        )
        if match:
            if routing_weights is None:
                raise ValueError(f"{self.name} requires routing weights")
            threshold = _decode_policy_float(match.group(1))
            values = torch.full_like(values, BYTE_SIZES["fp8"])
            values[_gate_threshold_mask(routing_weights, threshold)] = BYTE_SIZES[match.group(2)]
            return values

        match = re.fullmatch(
            r"gate_tailmass_(\d+p\d+)_(int4|mxfp4|nvfp4)", self.name
        )
        if match:
            if routing_weights is None:
                raise ValueError(f"{self.name} requires routing weights")
            tail_mass = _decode_policy_float(match.group(1))
            values = torch.full_like(values, BYTE_SIZES["fp8"])
            values[_gate_tailmass_mask(routing_weights, tail_mass)] = BYTE_SIZES[match.group(2)]
            return values

        match = re.fullmatch(r"block_gate(\d+)_(int4|mxfp4|nvfp4)", self.name)
        if match:
            if routing_weights is None:
                raise ValueError(f"{self.name} requires routing weights")
            block_tokens = int(match.group(1))
            values = torch.full_like(values, BYTE_SIZES["fp8"])
            mask = _block_budget_mask(routing_weights.float(), block_tokens, 0.5)
            values[mask] = BYTE_SIZES[match.group(2)]
            return values

        match = re.fullmatch(
            r"block_gate(\d+)_f(\d+)_(int4|mxfp4|nvfp4)", self.name
        )
        if match:
            if routing_weights is None:
                raise ValueError(f"{self.name} requires routing weights")
            block_tokens = int(match.group(1))
            low_fraction = _decode_milli_fraction(match.group(2))
            precision = match.group(3)
            values = torch.full_like(values, BYTE_SIZES["fp8"])
            mask = _block_budget_mask(routing_weights.float(), block_tokens, low_fraction)
            values[mask] = BYTE_SIZES[precision]
            return values

        match = re.fullmatch(
            r"block_gate(\d+)_residual_(int4|mxfp4|nvfp4)", self.name
        )
        if match:
            if routing_weights is None:
                raise ValueError(f"{self.name} requires routing weights")
            block_tokens = int(match.group(1))
            precision = match.group(2)
            low_mask = _block_budget_mask(routing_weights.float(), block_tokens, 0.5)
            values = torch.full_like(values, 2.0 * BYTE_SIZES[precision])
            values[low_mask] = BYTE_SIZES[precision]
            return values

        match = re.fullmatch(
            r"block_(contrib|qenergy|qerr|qbenefit|random|reversegate)(\d+)"
            r"(?:_f(\d+))?_(int4|mxfp4|nvfp4)",
            self.name,
        )
        if match:
            if routing_weights is None:
                raise ValueError(f"{self.name} requires routing weights")
            # Byte accounting only needs the fixed cardinality.  Error-aware
            # score values are intentionally not recomputed in this path.
            block_tokens = int(match.group(2))
            low_fraction = (
                _decode_milli_fraction(match.group(3))
                if match.group(3) is not None
                else 0.5
            )
            dummy_scores = (
                _deterministic_random_scores(routing_weights)
                if match.group(1) == "random"
                else routing_weights.float()
            )
            mask = _block_budget_mask(dummy_scores, block_tokens, low_fraction)
            values = torch.full_like(values, BYTE_SIZES["fp8"])
            values[mask] = BYTE_SIZES[match.group(4)]
            return values

        match = re.fullmatch(
            r"block_(contrib|resenergy|reserr|resbenefit|random|reversegate)(\d+)"
            r"_residual_(int4|mxfp4|nvfp4)",
            self.name,
        )
        if match:
            if routing_weights is None:
                raise ValueError(f"{self.name} requires routing weights")
            block_tokens = int(match.group(2))
            dummy_scores = (
                _deterministic_random_scores(routing_weights)
                if match.group(1) == "random"
                else routing_weights.float()
            )
            low_mask = _block_budget_mask(dummy_scores, block_tokens, 0.5)
            values = torch.full_like(values, 2.0 * BYTE_SIZES[match.group(3)])
            values[low_mask] = BYTE_SIZES[match.group(3)]
            return values

        match = re.fullmatch(r"peerblock_gate(\d+)_(int4|mxfp4|nvfp4)", self.name)
        if match:
            if routing_weights is None:
                raise ValueError(f"{self.name} requires routing weights")
            block_pairs = int(match.group(1))
            precision = match.group(2)
            group_ids = receiver_group_ids(
                selected_experts, num_experts, num_receiver_groups, receiver_mapping
            )
            mask = _peer_block_budget_mask(
                routing_weights.float(), group_ids, block_pairs, 0.5
            )
            values = torch.full_like(values, BYTE_SIZES["fp8"])
            values[mask] = BYTE_SIZES[precision]
            return values

        match = re.fullmatch(
            r"peerblock_gate(\d+)_f(\d+)_(int4|mxfp4|nvfp4)", self.name
        )
        if match:
            if routing_weights is None:
                raise ValueError(f"{self.name} requires routing weights")
            block_pairs = int(match.group(1))
            low_fraction = _decode_milli_fraction(match.group(2))
            precision = match.group(3)
            group_ids = receiver_group_ids(
                selected_experts, num_experts, num_receiver_groups, receiver_mapping
            )
            mask = _peer_block_budget_mask(
                routing_weights.float(), group_ids, block_pairs, low_fraction
            )
            values = torch.full_like(values, BYTE_SIZES["fp8"])
            values[mask] = BYTE_SIZES[precision]
            return values

        match = re.fullmatch(
            r"peerblock_gate(\d+)_residual_(int4|mxfp4|nvfp4)", self.name
        )
        if match:
            if routing_weights is None:
                raise ValueError(f"{self.name} requires routing weights")
            block_pairs = int(match.group(1))
            precision = match.group(2)
            group_ids = receiver_group_ids(
                selected_experts, num_experts, num_receiver_groups, receiver_mapping
            )
            low_mask = _peer_block_budget_mask(
                routing_weights.float(), group_ids, block_pairs, 0.5
            )
            values = torch.full_like(values, 2.0 * BYTE_SIZES[precision])
            values[low_mask] = BYTE_SIZES[precision]
            return values

        match = re.fullmatch(
            r"peerblock_(contrib|qenergy|qerr|qbenefit|random|reversegate)(\d+)"
            r"(?:_f(\d+))?_(int4|mxfp4|nvfp4)",
            self.name,
        )
        if match:
            if routing_weights is None:
                raise ValueError(f"{self.name} requires routing weights")
            block_pairs = int(match.group(2))
            low_fraction = (
                _decode_milli_fraction(match.group(3))
                if match.group(3) is not None
                else 0.5
            )
            group_ids = receiver_group_ids(
                selected_experts, num_experts, num_receiver_groups, receiver_mapping
            )
            dummy_scores = (
                _deterministic_random_scores(routing_weights)
                if match.group(1) == "random"
                else routing_weights.float()
            )
            mask = _peer_block_budget_mask(
                dummy_scores, group_ids, block_pairs, low_fraction
            )
            values = torch.full_like(values, BYTE_SIZES["fp8"])
            values[mask] = BYTE_SIZES[match.group(4)]
            return values

        match = re.fullmatch(
            r"peerblock_(contrib|resenergy|reserr|resbenefit|random|reversegate)(\d+)"
            r"_residual_(int4|mxfp4|nvfp4)",
            self.name,
        )
        if match:
            if routing_weights is None:
                raise ValueError(f"{self.name} requires routing weights")
            block_pairs = int(match.group(2))
            group_ids = receiver_group_ids(
                selected_experts, num_experts, num_receiver_groups, receiver_mapping
            )
            dummy_scores = (
                _deterministic_random_scores(routing_weights)
                if match.group(1) == "random"
                else routing_weights.float()
            )
            low_mask = _peer_block_budget_mask(
                dummy_scores, group_ids, block_pairs, 0.5
            )
            precision = match.group(3)
            values = torch.full_like(values, 2.0 * BYTE_SIZES[precision])
            values[low_mask] = BYTE_SIZES[precision]
            return values

        match = re.fullmatch(r"group(\d+)_rank(\d+|k)_(fp8|int8|int4|mxfp4|nvfp4)", self.name)
        if match:
            group = int(match.group(1))
            idx = self.rank_index(top_k, match.group(2))
            prec = match.group(3)
            group_ids = receiver_group_ids(selected_experts, num_experts, num_receiver_groups, receiver_mapping)
            values[:, idx] = torch.where(group_ids[:, idx] == group, BYTE_SIZES[prec], values[:, idx])
            return values
        match = re.fullmatch(r"group(\d+)_rank(\d+|k)_drop(_renorm)?", self.name)
        if match:
            group = int(match.group(1))
            idx = self.rank_index(top_k, match.group(2))
            group_ids = receiver_group_ids(selected_experts, num_experts, num_receiver_groups, receiver_mapping)
            values[:, idx] = torch.where(group_ids[:, idx] == group, 0.0, values[:, idx])
            return values

        raise ValueError(f"unknown strategy: {self.name}")

    def byte_saving(self, top_k: int) -> float:
        baseline = 2.0 * top_k
        actual = sum(self.bytes_per_element_by_rank(top_k))
        return 1.0 - (actual / baseline)

    def _apply_layer_lut(
        self,
        raw_outputs: torch.Tensor,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
        num_experts: int,
        num_receiver_groups: int,
        receiver_mapping: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = raw_outputs.clone()
        weights = routing_weights
        top_k = raw_outputs.shape[1]
        group_ids = receiver_group_ids(
            selected_experts, num_experts, num_receiver_groups, receiver_mapping
        )
        for rank_idx in range(top_k):
            rank = rank_idx + 1
            for group in range(num_receiver_groups):
                mask = group_ids[:, rank_idx] == group
                if not mask.any():
                    continue
                precision = self.layer_lut.get((group, rank), "bf16")
                if precision in REDUCE_PRECISIONS:
                    out[mask, rank_idx, :] = apply_precision(out[mask, rank_idx, :], precision)
                elif precision == "drop":
                    out[mask, rank_idx, :] = 0
        return out, weights

    def apply(
        self,
        raw_outputs: torch.Tensor,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor | None = None,
        num_experts: int | None = None,
        num_receiver_groups: int = 1,
        receiver_mapping: str = "contiguous",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return possibly approximated outputs and possibly adjusted routing weights.

        raw_outputs: [tokens, top_k, hidden]
        routing_weights: [tokens, top_k]
        """
        top_k = raw_outputs.shape[1]
        out = raw_outputs
        weights = routing_weights

        if self.layer_lut is not None and selected_experts is not None and num_experts is not None:
            return self._apply_layer_lut(
                raw_outputs, routing_weights, selected_experts,
                num_experts, num_receiver_groups, receiver_mapping,
            )

        if self.name == "full":
            return out, weights
        for prec in REDUCE_PRECISIONS:
            if self.name == f"uniform_{prec}":
                return apply_precision(out, prec), weights
            match = re.fullmatch(rf"rank(\d+|k)_{prec}", self.name)
            if match:
                out = out.clone()
                idx = self.rank_index(top_k, match.group(1))
                out[:, idx, :] = apply_precision(out[:, idx, :], prec)
                return out, weights

        match = re.fullmatch(r"rank(\d+|k)_drop(_renorm)?", self.name)
        if match:
            out = out.clone()
            idx = self.rank_index(top_k, match.group(1))
            out[:, idx, :] = 0
            if match.group(2):
                weights = weights.clone()
                weights[:, idx] = 0
                weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            return out, weights
        match = re.fullmatch(r"keep(\d+)_drop(_renorm)?", self.name)
        if match:
            n = min(int(match.group(1)), top_k)
            out = out.clone()
            out[:, n:, :] = 0
            if match.group(2):
                weights = weights.clone()
                weights[:, n:] = 0
                weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            return out, weights

        match = re.fullmatch(r"keep(\d+)_bf16_rest_(fp8|int8|int4|mxfp4|nvfp4)", self.name)
        if match:
            n = int(match.group(1))
            prec = match.group(2)
            out = out.clone()
            for rank_idx in range(top_k):
                if rank_idx >= n:
                    out[:, rank_idx, :] = apply_precision(out[:, rank_idx, :], prec)
            return out, weights

        match = re.fullmatch(r"fp8top(\d+)_rest_(fp8|int8|int4|mxfp4|nvfp4)", self.name)
        if match:
            n = int(match.group(1))
            prec = match.group(2)
            out = out.clone()
            # FP8 to all ranks first, then prec to tail ranks (by position)
            out = apply_precision(out, "fp8")
            for rank_idx in range(top_k):
                if rank_idx >= n:
                    out[:, rank_idx, :] = apply_precision(raw_outputs[:, rank_idx, :], prec)
            return out, weights

        # Contribution-based tail selection: pick the n (token, rank) pairs with
        # lowest g*||o|| per token, apply `prec` to those, FP8 to the rest.
        # Differs from rank-tail when ||o|| varies across ranks.
        match = re.fullmatch(r"contrib_tail(\d+)_(fp8|int8|int4|mxfp4|nvfp4)", self.name)
        if match:
            n = int(match.group(1))
            prec = match.group(2)
            out = raw_outputs.clone()
            contrib = routing_weights.float() * raw_outputs.float().norm(dim=-1)  # [tokens, top_k]
            _, bottom_idx = contrib.topk(min(n, top_k), dim=1, largest=False)  # [tokens, n]
            tail_mask = torch.zeros_like(contrib, dtype=torch.bool)
            tail_mask.scatter_(1, bottom_idx, True)
            # FP8 to all, then prec to contribution-tail
            out = apply_precision(out, "fp8")
            for rank_idx in range(top_k):
                mask = tail_mask[:, rank_idx]
                if mask.any():
                    out[mask, rank_idx, :] = apply_precision(raw_outputs[mask, rank_idx, :], prec)
            return out, weights

        match = re.fullmatch(
            r"gate_threshold_(\d+p\d+)_(int4|mxfp4|nvfp4)", self.name
        )
        if match:
            threshold = _decode_policy_float(match.group(1))
            precision = match.group(2)
            mask = _gate_threshold_mask(routing_weights, threshold)
            out = apply_precision(raw_outputs, "fp8")
            for rank_idx in range(top_k):
                rank_mask = mask[:, rank_idx]
                if rank_mask.any():
                    out[rank_mask, rank_idx, :] = apply_precision(
                        raw_outputs[rank_mask, rank_idx, :], precision
                    )
            return out, weights

        match = re.fullmatch(
            r"gate_tailmass_(\d+p\d+)_(int4|mxfp4|nvfp4)", self.name
        )
        if match:
            tail_mass = _decode_policy_float(match.group(1))
            precision = match.group(2)
            mask = _gate_tailmass_mask(routing_weights, tail_mass)
            out = apply_precision(raw_outputs, "fp8")
            for rank_idx in range(top_k):
                rank_mask = mask[:, rank_idx]
                if rank_mask.any():
                    out[rank_mask, rank_idx, :] = apply_precision(
                        raw_outputs[rank_mask, rank_idx, :], precision
                    )
            return out, weights

        match = re.fullmatch(r"block_gate(\d+)_(int4|mxfp4|nvfp4)", self.name)
        if match:
            block_tokens = int(match.group(1))
            precision = match.group(2)
            mask = _block_budget_mask(routing_weights.float(), block_tokens, 0.5)
            out = apply_precision(raw_outputs, "fp8")
            for rank_idx in range(top_k):
                rank_mask = mask[:, rank_idx]
                if rank_mask.any():
                    out[rank_mask, rank_idx, :] = apply_precision(
                        raw_outputs[rank_mask, rank_idx, :], precision
                    )
            return out, weights

        match = re.fullmatch(
            r"block_gate(\d+)_f(\d+)_(int4|mxfp4|nvfp4)", self.name
        )
        if match:
            block_tokens = int(match.group(1))
            low_fraction = _decode_milli_fraction(match.group(2))
            precision = match.group(3)
            mask = _block_budget_mask(routing_weights.float(), block_tokens, low_fraction)
            out = apply_precision(raw_outputs, "fp8")
            for rank_idx in range(top_k):
                rank_mask = mask[:, rank_idx]
                if rank_mask.any():
                    out[rank_mask, rank_idx, :] = apply_precision(
                        raw_outputs[rank_mask, rank_idx, :], precision
                    )
            return out, weights

        match = re.fullmatch(
            r"block_gate(\d+)_residual_(int4|mxfp4|nvfp4)", self.name
        )
        if match:
            block_tokens = int(match.group(1))
            precision = match.group(2)
            low_mask = _block_budget_mask(routing_weights.float(), block_tokens, 0.5)
            base = apply_precision(raw_outputs, precision)
            residual = raw_outputs.float() - base.float()
            residual_q = apply_precision(residual, precision)
            out = base.clone()
            critical_mask = ~low_mask
            for rank_idx in range(top_k):
                rank_mask = critical_mask[:, rank_idx]
                if rank_mask.any():
                    refined = base[rank_mask, rank_idx, :].float() + residual_q[
                        rank_mask, rank_idx, :
                    ].float()
                    out[rank_mask, rank_idx, :] = refined.to(dtype=raw_outputs.dtype)
            return out, weights

        match = re.fullmatch(
            r"block_(contrib|qenergy|qerr|qbenefit|random|reversegate)(\d+)"
            r"(?:_f(\d+))?_(int4|mxfp4|nvfp4)",
            self.name,
        )
        if match:
            mode = match.group(1)
            block_tokens = int(match.group(2))
            low_fraction = (
                _decode_milli_fraction(match.group(3))
                if match.group(3) is not None
                else 0.5
            )
            precision = match.group(4)
            scores = _pair_criticality_scores(
                mode, raw_outputs, routing_weights, precision
            )
            low_mask = _block_budget_mask(scores, block_tokens, low_fraction)
            out = apply_precision(raw_outputs, "fp8")
            for rank_idx in range(top_k):
                rank_mask = low_mask[:, rank_idx]
                if rank_mask.any():
                    out[rank_mask, rank_idx, :] = apply_precision(
                        raw_outputs[rank_mask, rank_idx, :], precision
                    )
            return out, weights

        match = re.fullmatch(
            r"block_(contrib|resenergy|reserr|resbenefit|random|reversegate)(\d+)"
            r"_residual_(int4|mxfp4|nvfp4)",
            self.name,
        )
        if match:
            mode = match.group(1)
            block_tokens = int(match.group(2))
            precision = match.group(3)
            scores = _pair_criticality_scores(
                mode, raw_outputs, routing_weights, precision
            )
            low_mask = _block_budget_mask(scores, block_tokens, 0.5)
            base = apply_precision(raw_outputs, precision)
            residual_q = apply_precision(raw_outputs.float() - base.float(), precision)
            out = base.clone()
            critical_mask = ~low_mask
            for rank_idx in range(top_k):
                rank_mask = critical_mask[:, rank_idx]
                if rank_mask.any():
                    refined = base[rank_mask, rank_idx, :].float() + residual_q[
                        rank_mask, rank_idx, :
                    ].float()
                    out[rank_mask, rank_idx, :] = refined.to(dtype=raw_outputs.dtype)
            return out, weights

        match = re.fullmatch(r"peerblock_gate(\d+)_(int4|mxfp4|nvfp4)", self.name)
        if match:
            if selected_experts is None or num_experts is None:
                raise ValueError(f"{self.name} requires selected experts")
            block_pairs = int(match.group(1))
            precision = match.group(2)
            group_ids = receiver_group_ids(
                selected_experts, num_experts, num_receiver_groups, receiver_mapping
            )
            mask = _peer_block_budget_mask(
                routing_weights.float(), group_ids, block_pairs, 0.5
            )
            out = apply_precision(raw_outputs, "fp8")
            for rank_idx in range(top_k):
                rank_mask = mask[:, rank_idx]
                if rank_mask.any():
                    out[rank_mask, rank_idx, :] = apply_precision(
                        raw_outputs[rank_mask, rank_idx, :], precision
                    )
            return out, weights

        match = re.fullmatch(
            r"peerblock_gate(\d+)_f(\d+)_(int4|mxfp4|nvfp4)", self.name
        )
        if match:
            if selected_experts is None or num_experts is None:
                raise ValueError(f"{self.name} requires selected experts")
            block_pairs = int(match.group(1))
            low_fraction = _decode_milli_fraction(match.group(2))
            precision = match.group(3)
            group_ids = receiver_group_ids(
                selected_experts, num_experts, num_receiver_groups, receiver_mapping
            )
            low_mask = _peer_block_budget_mask(
                routing_weights.float(), group_ids, block_pairs, low_fraction
            )
            out = apply_precision(raw_outputs, "fp8")
            for rank_idx in range(top_k):
                rank_mask = low_mask[:, rank_idx]
                if rank_mask.any():
                    out[rank_mask, rank_idx, :] = apply_precision(
                        raw_outputs[rank_mask, rank_idx, :], precision
                    )
            return out, weights

        match = re.fullmatch(
            r"peerblock_gate(\d+)_residual_(int4|mxfp4|nvfp4)", self.name
        )
        if match:
            if selected_experts is None or num_experts is None:
                raise ValueError(f"{self.name} requires selected experts")
            block_pairs = int(match.group(1))
            precision = match.group(2)
            group_ids = receiver_group_ids(
                selected_experts, num_experts, num_receiver_groups, receiver_mapping
            )
            low_mask = _peer_block_budget_mask(
                routing_weights.float(), group_ids, block_pairs, 0.5
            )
            base = apply_precision(raw_outputs, precision)
            residual = raw_outputs.float() - base.float()
            residual_q = apply_precision(residual, precision)
            out = base.clone()
            critical_mask = ~low_mask
            for rank_idx in range(top_k):
                rank_mask = critical_mask[:, rank_idx]
                if rank_mask.any():
                    refined = base[rank_mask, rank_idx, :].float() + residual_q[
                        rank_mask, rank_idx, :
                    ].float()
                    out[rank_mask, rank_idx, :] = refined.to(dtype=raw_outputs.dtype)
            return out, weights

        match = re.fullmatch(
            r"peerblock_(contrib|qenergy|qerr|qbenefit|random|reversegate)(\d+)"
            r"(?:_f(\d+))?_(int4|mxfp4|nvfp4)",
            self.name,
        )
        if match:
            if selected_experts is None or num_experts is None:
                raise ValueError(f"{self.name} requires selected experts")
            mode = match.group(1)
            block_pairs = int(match.group(2))
            low_fraction = (
                _decode_milli_fraction(match.group(3))
                if match.group(3) is not None
                else 0.5
            )
            precision = match.group(4)
            scores = _pair_criticality_scores(
                mode, raw_outputs, routing_weights, precision
            )
            group_ids = receiver_group_ids(
                selected_experts, num_experts, num_receiver_groups, receiver_mapping
            )
            low_mask = _peer_block_budget_mask(
                scores, group_ids, block_pairs, low_fraction
            )
            out = apply_precision(raw_outputs, "fp8")
            for rank_idx in range(top_k):
                rank_mask = low_mask[:, rank_idx]
                if rank_mask.any():
                    out[rank_mask, rank_idx, :] = apply_precision(
                        raw_outputs[rank_mask, rank_idx, :], precision
                    )
            return out, weights

        match = re.fullmatch(
            r"peerblock_(contrib|resenergy|reserr|resbenefit|random|reversegate)(\d+)"
            r"_residual_(int4|mxfp4|nvfp4)",
            self.name,
        )
        if match:
            if selected_experts is None or num_experts is None:
                raise ValueError(f"{self.name} requires selected experts")
            mode = match.group(1)
            block_pairs = int(match.group(2))
            precision = match.group(3)
            scores = _pair_criticality_scores(
                mode, raw_outputs, routing_weights, precision
            )
            group_ids = receiver_group_ids(
                selected_experts, num_experts, num_receiver_groups, receiver_mapping
            )
            low_mask = _peer_block_budget_mask(scores, group_ids, block_pairs, 0.5)
            base = apply_precision(raw_outputs, precision)
            residual_q = apply_precision(raw_outputs.float() - base.float(), precision)
            out = base.clone()
            critical_mask = ~low_mask
            for rank_idx in range(top_k):
                rank_mask = critical_mask[:, rank_idx]
                if rank_mask.any():
                    refined = base[rank_mask, rank_idx, :].float() + residual_q[
                        rank_mask, rank_idx, :
                    ].float()
                    out[rank_mask, rank_idx, :] = refined.to(dtype=raw_outputs.dtype)
            return out, weights

        match = re.fullmatch(r"group(\d+)_rank(\d+|k)_(fp8|int8|int4|mxfp4|nvfp4)", self.name)
        if match:
            if selected_experts is None or num_experts is None:
                raise ValueError(f"{self.name} requires selected experts")
            out = out.clone()
            group = int(match.group(1))
            idx = self.rank_index(top_k, match.group(2))
            prec = match.group(3)
            group_ids = receiver_group_ids(selected_experts, num_experts, num_receiver_groups, receiver_mapping)
            mask = group_ids[:, idx] == group
            out[mask, idx, :] = apply_precision(out[mask, idx, :], prec)
            return out, weights
        match = re.fullmatch(r"group(\d+)_rank(\d+|k)_drop(_renorm)?", self.name)
        if match:
            if selected_experts is None or num_experts is None:
                raise ValueError(f"{self.name} requires selected experts")
            out = out.clone()
            group = int(match.group(1))
            idx = self.rank_index(top_k, match.group(2))
            group_ids = receiver_group_ids(selected_experts, num_experts, num_receiver_groups, receiver_mapping)
            mask = group_ids[:, idx] == group
            out[mask, idx, :] = 0
            if match.group(3):
                weights = weights.clone()
                weights[mask, idx] = 0
                weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            return out, weights

        raise ValueError(f"unknown strategy: {self.name}")


def make_policy(name: str, layer_lut: dict | None = None) -> ApproxPolicy:
    return ApproxPolicy(name=name, drop_renorm=name.endswith("_renorm"), layer_lut=layer_lut)
