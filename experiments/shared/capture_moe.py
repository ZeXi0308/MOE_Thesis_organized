from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from types import MethodType
import re
import sys

import torch
import torch.nn.functional as F

from fake_quant import apply_precision
from policies import ApproxPolicy, make_policy
from policies import receiver_group_ids

# These two numerical references were moved into killed-idea archives while
# capture_moe remained shared by active experiments.  Keep the historical
# branches importable without making every active entrypoint modify sys.path.
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _compat_dir in (
    _REPO_ROOT / "docs/archive/killed_ideas/creditreduce/scripts",
    _REPO_ROOT / "docs/archive/killed_ideas/quotaep/scripts",
):
    if str(_compat_dir) not in sys.path:
        sys.path.append(str(_compat_dir))

from creditreduce_reference import creditreduce_reference
from grouped_owner_combine import grouped_owner_combine


def parse_dispatch_policy(name: str | None, top_k: int) -> list[str] | None:
    """Parse dispatch policy name into per-rank precision list (rank1 .. rank_k)."""
    if name is None or name == "full":
        return None
    if name.startswith("dispatch_uniform_"):
        prec = name[len("dispatch_uniform_"):]
        return [prec] * top_k
    match = re.fullmatch(r"dispatch_rank(\d+|k)_(fp8|int8|int4)", name)
    if match:
        suffix, prec = match.group(1), match.group(2)
        values = ["bf16"] * top_k
        idx = (top_k - 1) if suffix == "k" else (int(suffix) - 1)
        values[idx] = prec
        return values
    return None


@dataclass
class LayerRankStats:
    shares: list[float] = field(default_factory=list)
    rank1_over_rankk: list[float] = field(default_factory=list)


@dataclass
class LayerErrorStats:
    sq_error: float = 0.0
    sq_full: float = 0.0


@dataclass
class ReceiverRankStats:
    shares: list[float] = field(default_factory=list)
    token_count: int = 0
    full_bytes: float = 0.0
    policy_bytes: float = 0.0


@dataclass
class GroupedOwnerStats:
    routed_pairs: int = 0
    grouped_vectors: int = 0
    collision_pairs: int = 0
    multi_expert_vectors: int = 0
    max_pairs_per_vector: int = 0
    bf16_vectors: int = 0
    high_vectors: int = 0
    low_vectors: int = 0
    tile_count: int = 0
    association_sq_error: float = 0.0
    pair_full_sq: float = 0.0


class MoeRecorder:
    def __init__(
        self,
        num_receiver_groups: int = 1,
        receiver_mapping: str = "contiguous",
        record_routes: bool = False,
        audit_pair_scores: bool = False,
    ) -> None:
        self.rank_stats: dict[tuple[int, int], LayerRankStats] = defaultdict(LayerRankStats)
        self.error_stats: dict[int, LayerErrorStats] = defaultdict(LayerErrorStats)
        self.receiver_rank_stats: dict[tuple[int, int, int], ReceiverRankStats] = defaultdict(ReceiverRankStats)
        self.top_k: int | None = None
        self.num_layers_seen: set[int] = set()
        self.num_receiver_groups = num_receiver_groups
        self.receiver_mapping = receiver_mapping
        self.routing_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.record_routes = record_routes
        self.audit_pair_scores = audit_pair_scores
        self.current_sample_id = -1
        self.routing_weight_batches: list[torch.Tensor] = []
        self.route_batches: list[dict[str, int | torch.Tensor]] = []
        self.pair_audit_batches: list[dict[str, int | torch.Tensor]] = []
        self.cross_term_batches: list[dict[str, int | torch.Tensor]] = []
        self.grouped_owner_stats: dict[int, GroupedOwnerStats] = defaultdict(
            GroupedOwnerStats
        )
        self.grouped_tile_rows: list[dict[str, int | float]] = []
        self.creditreduce_layer_rows: list[dict[str, int | float | bool | str]] = []
        self.creditreduce_token_rows: list[dict[str, int | float | bool | str]] = []
        self.creditreduce_group_rows: list[dict[str, int | float | bool | str]] = []

    def set_sample_id(self, sample_id: int) -> None:
        self.current_sample_id = int(sample_id)

    def update_routing(
        self,
        layer_id: int,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> None:
        if not self.record_routes:
            return
        weights_cpu = routing_weights.detach().float().cpu()
        self.routing_weight_batches.append(weights_cpu)
        self.route_batches.append(
            {
                "sample_id": self.current_sample_id,
                "layer": int(layer_id),
                "selected_experts": selected_experts.detach().cpu(),
                "routing_weights": weights_cpu,
            }
        )

    def routing_weights_tensor(self) -> torch.Tensor:
        if not self.routing_weight_batches:
            return torch.empty((0, self.top_k or 0), dtype=torch.float32)
        return torch.cat(self.routing_weight_batches, dim=0)

    def route_rows(self) -> list[dict[str, float | int]]:
        rows: list[dict[str, float | int]] = []
        for batch in self.route_batches:
            experts = batch["selected_experts"]
            weights = batch["routing_weights"]
            if not isinstance(experts, torch.Tensor) or not isinstance(weights, torch.Tensor):
                continue
            for token_position in range(experts.shape[0]):
                for rank_idx in range(experts.shape[1]):
                    rows.append(
                        {
                            "sample_id": int(batch["sample_id"]),
                            "layer": int(batch["layer"]),
                            "token_position": token_position,
                            "rank": rank_idx + 1,
                            "expert_id": int(experts[token_position, rank_idx].item()),
                            "gate_weight": float(weights[token_position, rank_idx].item()),
                        }
                    )
        return rows

    def update_contrib(self, layer_id: int, shares: torch.Tensor) -> None:
        shares_cpu = shares.detach().float().cpu()
        top_k = shares_cpu.shape[1]
        self.top_k = top_k
        self.num_layers_seen.add(layer_id)
        for rank in range(top_k):
            self.rank_stats[(layer_id, rank + 1)].shares.extend(shares_cpu[:, rank].tolist())
        ratio = shares_cpu[:, 0] / shares_cpu[:, top_k - 1].clamp_min(1e-12)
        self.rank_stats[(layer_id, top_k)].rank1_over_rankk.extend(ratio.tolist())

    def update_pair_audit(
        self,
        layer_id: int,
        raw_outputs: torch.Tensor,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
    ) -> None:
        """Record pair scores and exact one-pair local FP4->FP8 interventions.

        The intervention target is the squared error of the current MoE combine
        output in float32, with every routed output initially encoded as MXFP4.
        Upgrading one pair at a time includes its cross term with all other
        low-bit pair errors for that token.  It is a local causal diagnostic,
        not an end-to-end downstream-loss oracle.
        """
        if not self.audit_pair_scores:
            return
        raw = raw_outputs.detach().float()
        weights = routing_weights.detach().float()
        low = apply_precision(raw_outputs.detach(), "mxfp4").float()
        high = apply_precision(raw_outputs.detach(), "fp8").float()
        low_delta = low - raw
        high_delta = high - raw
        weighted_low = weights[..., None] * low_delta
        weighted_high = weights[..., None] * high_delta
        total_low_error = weighted_low.sum(dim=1)
        total_energy = total_low_error.square().sum(dim=-1)
        pair_change = weighted_high - weighted_low
        intervention_energy = (
            total_low_error[:, None, :] + pair_change
        ).square().sum(dim=-1)
        causal_gain = total_energy[:, None] - intervention_energy

        low_energy = low_delta.square().sum(dim=-1)
        high_energy = high_delta.square().sum(dim=-1)
        output_energy = raw.square().sum(dim=-1)
        diagonal = weighted_low.square().sum(dim=(-1, -2))
        cross = total_energy - diagonal
        contribution = weights * raw.norm(dim=-1)
        qerr = weights.square() * low_energy
        qbenefit = weights.square() * (low_energy - high_energy)

        self.pair_audit_batches.append(
            {
                "sample_id": int(self.current_sample_id),
                "layer": int(layer_id),
                "selected_experts": selected_experts.detach().cpu(),
                "gate": weights.cpu(),
                "contribution": contribution.cpu(),
                "qerr": qerr.cpu(),
                "qbenefit": qbenefit.cpu(),
                "causal_local_gain": causal_gain.cpu(),
                "output_energy": output_energy.cpu(),
                "low_error_energy": low_energy.cpu(),
                "high_error_energy": high_energy.cpu(),
            }
        )
        self.cross_term_batches.append(
            {
                "sample_id": int(self.current_sample_id),
                "layer": int(layer_id),
                "diagonal_energy": diagonal.cpu(),
                "cross_energy": cross.cpu(),
                "total_energy": total_energy.cpu(),
            }
        )

    def update_error(self, layer_id: int, approx: torch.Tensor, full: torch.Tensor) -> None:
        err = (approx.detach().float() - full.detach().float()).pow(2).sum().item()
        denom = full.detach().float().pow(2).sum().item()
        stat = self.error_stats[layer_id]
        stat.sq_error += err
        stat.sq_full += denom

    def update_grouped_owner(
        self,
        layer_id: int,
        diagnostics: dict[str, object],
        grouped_full: torch.Tensor,
        pair_full: torch.Tensor,
    ) -> None:
        """Aggregate DeepEP-wire-unit collision and quota diagnostics."""
        stat = self.grouped_owner_stats[layer_id]
        for name in (
            "routed_pairs",
            "grouped_vectors",
            "collision_pairs",
            "multi_expert_vectors",
            "bf16_vectors",
            "high_vectors",
            "low_vectors",
            "tile_count",
        ):
            setattr(stat, name, getattr(stat, name) + int(diagnostics[name]))
        stat.max_pairs_per_vector = max(
            stat.max_pairs_per_vector,
            int(diagnostics["max_pairs_per_vector"]),
        )
        association = grouped_full.detach().float() - pair_full.detach().float()
        stat.association_sq_error += float(association.square().sum().item())
        stat.pair_full_sq += float(pair_full.detach().float().square().sum().item())

        tile_rows = diagnostics.get("tile_rows", [])
        if isinstance(tile_rows, list):
            for row in tile_rows:
                if not isinstance(row, dict):
                    continue
                copied = dict(row)
                copied.update(
                    {
                        "sample_id": int(self.current_sample_id),
                        "layer": int(layer_id),
                    }
                )
                self.grouped_tile_rows.append(copied)

    def update_creditreduce(
        self,
        layer_id: int,
        endpoint: str,
        records: dict[str, object],
        *,
        record_detail: bool,
    ) -> None:
        """Store JSON-ready CreditReduce arithmetic and payload diagnostics."""

        aggregate = records.get("aggregate")
        if not isinstance(aggregate, dict):
            raise TypeError("CreditReduce aggregate diagnostics must be a dict")
        self.creditreduce_layer_rows.append(
            {
                **aggregate,
                "sample_id": int(self.current_sample_id),
                "layer": int(layer_id),
                "selected_endpoint": endpoint,
            }
        )
        if not record_detail:
            return
        for name, sink in (
            ("token_rows", self.creditreduce_token_rows),
            ("group_rows", self.creditreduce_group_rows),
        ):
            rows = records.get(name)
            if not isinstance(rows, list):
                raise TypeError(f"CreditReduce {name} diagnostics must be a list")
            for row in rows:
                if not isinstance(row, dict):
                    raise TypeError(f"CreditReduce {name} row must be a dict")
                sink.append(
                    {
                        **row,
                        "sample_id": int(self.current_sample_id),
                        "layer": int(layer_id),
                        "selected_endpoint": endpoint,
                    }
                )

    def update_receiver(
        self,
        layer_id: int,
        selected_experts: torch.Tensor,
        shares: torch.Tensor,
        hidden_dim: int,
        num_experts: int,
        policy: ApproxPolicy,
        routing_weights: torch.Tensor,
    ) -> None:
        group_ids = receiver_group_ids(
            selected_experts.detach(),
            num_experts,
            self.num_receiver_groups,
            self.receiver_mapping,
        )
        bytes_per_element = policy.bytes_per_element_for_selected(
            selected_experts.detach(),
            num_experts,
            self.num_receiver_groups,
            self.receiver_mapping,
            routing_weights=routing_weights.detach(),
        )
        shares_cpu = shares.detach().float().cpu()
        group_cpu = group_ids.detach().cpu()
        bytes_cpu = bytes_per_element.detach().float().cpu()

        top_k = selected_experts.shape[1]
        for rank_idx in range(top_k):
            for group in range(self.num_receiver_groups):
                mask = group_cpu[:, rank_idx] == group
                count = int(mask.sum().item())
                if count == 0:
                    continue
                stat = self.receiver_rank_stats[(layer_id, group, rank_idx + 1)]
                stat.token_count += count
                stat.shares.extend(shares_cpu[mask, rank_idx].tolist())
                stat.full_bytes += float(count * hidden_dim * 2.0)
                stat.policy_bytes += float(bytes_cpu[mask, rank_idx].sum().item() * hidden_dim)

    def rank_rows(self) -> list[dict[str, float]]:
        rows: list[dict[str, float]] = []
        for (layer_id, rank), stat in sorted(self.rank_stats.items()):
            if not stat.shares:
                continue
            values = torch.tensor(stat.shares, dtype=torch.float)
            ratio_values = torch.tensor(stat.rank1_over_rankk, dtype=torch.float) if stat.rank1_over_rankk else None
            row = {
                "layer": layer_id,
                "rank": rank,
                "count": int(values.numel()),
                "mean_share": float(values.mean().item()),
                "median_share": float(values.median().item()),
                "p75_share": float(torch.quantile(values, 0.75).item()),
                "p90_share": float(torch.quantile(values, 0.90).item()),
            }
            if ratio_values is not None and ratio_values.numel() > 0:
                row["rank1_over_rankk_median"] = float(ratio_values.median().item())
            rows.append(row)
        return rows

    def error_rows(self) -> list[dict[str, float]]:
        rows: list[dict[str, float]] = []
        for layer_id, stat in sorted(self.error_stats.items()):
            rows.append(
                {
                    "layer": layer_id,
                    "sq_error": stat.sq_error,
                    "sq_full": stat.sq_full,
                    "relative_mse": stat.sq_error / max(stat.sq_full, 1e-12),
                }
            )
        return rows

    def receiver_rank_rows(self) -> list[dict[str, float]]:
        rows: list[dict[str, float]] = []
        for (layer_id, group, rank), stat in sorted(self.receiver_rank_stats.items()):
            if stat.token_count == 0:
                continue
            values = torch.tensor(stat.shares, dtype=torch.float)
            rows.append(
                {
                    "layer": layer_id,
                    "receiver_group": group,
                    "rank": rank,
                    "count": stat.token_count,
                    "mean_share": float(values.mean().item()),
                    "median_share": float(values.median().item()),
                    "p75_share": float(torch.quantile(values, 0.75).item()),
                    "p90_share": float(torch.quantile(values, 0.90).item()),
                    "full_bytes": stat.full_bytes,
                    "policy_bytes": stat.policy_bytes,
                    "policy_byte_saving": 1.0 - stat.policy_bytes / max(stat.full_bytes, 1e-12),
                }
            )
        return rows

    def total_byte_saving(self) -> float:
        full = sum(stat.full_bytes for stat in self.receiver_rank_stats.values())
        policy = sum(stat.policy_bytes for stat in self.receiver_rank_stats.values())
        if full <= 0:
            return 0.0
        return 1.0 - policy / full

    def grouped_owner_rows(self) -> list[dict[str, int | float]]:
        rows: list[dict[str, int | float]] = []
        for layer_id, stat in sorted(self.grouped_owner_stats.items()):
            rows.append(
                {
                    "layer": layer_id,
                    "routed_pairs_n": stat.routed_pairs,
                    "grouped_vectors_m": stat.grouped_vectors,
                    "n_over_m": stat.routed_pairs
                    / max(stat.grouped_vectors, 1),
                    "collision_pair_fraction": stat.collision_pairs
                    / max(stat.routed_pairs, 1),
                    "multi_expert_vector_fraction": stat.multi_expert_vectors
                    / max(stat.grouped_vectors, 1),
                    "max_pairs_per_vector": stat.max_pairs_per_vector,
                    "bf16_vectors": stat.bf16_vectors,
                    "high_vectors": stat.high_vectors,
                    "low_vectors": stat.low_vectors,
                    "observed_high_fraction": stat.high_vectors
                    / max(stat.high_vectors + stat.low_vectors, 1),
                    "tile_count": stat.tile_count,
                    "association_relative_mse": stat.association_sq_error
                    / max(stat.pair_full_sq, 1e-12),
                }
            )
        return rows

    def grouped_owner_summary(self) -> dict[str, int | float]:
        stats = list(self.grouped_owner_stats.values())
        routed_pairs = sum(row.routed_pairs for row in stats)
        grouped_vectors = sum(row.grouped_vectors for row in stats)
        collision_pairs = sum(row.collision_pairs for row in stats)
        multi_expert_vectors = sum(row.multi_expert_vectors for row in stats)
        bf16_vectors = sum(row.bf16_vectors for row in stats)
        high_vectors = sum(row.high_vectors for row in stats)
        low_vectors = sum(row.low_vectors for row in stats)
        association_sq_error = sum(row.association_sq_error for row in stats)
        pair_full_sq = sum(row.pair_full_sq for row in stats)
        return {
            "routed_pairs_n": routed_pairs,
            "grouped_vectors_m": grouped_vectors,
            "n_over_m": routed_pairs / max(grouped_vectors, 1),
            "collision_pair_fraction": collision_pairs / max(routed_pairs, 1),
            "multi_expert_vector_fraction": multi_expert_vectors
            / max(grouped_vectors, 1),
            "max_pairs_per_vector": max(
                (row.max_pairs_per_vector for row in stats), default=0
            ),
            "bf16_vectors": bf16_vectors,
            "high_vectors": high_vectors,
            "low_vectors": low_vectors,
            "observed_high_fraction": high_vectors
            / max(high_vectors + low_vectors, 1),
            "tile_count": sum(row.tile_count for row in stats),
            "association_relative_mse": association_sq_error
            / max(pair_full_sq, 1e-12),
        }


def patch_mixtral_moe(
    model,
    policy_name: str = "full",
    num_receiver_groups: int = 1,
    receiver_mapping: str = "contiguous",
    target_layer: int | None = None,
    cache_routing: bool = False,
    lock_routing: bool = False,
    routing_cache: dict | None = None,
    lut: dict | None = None,
    dispatch_policy_name: str | None = None,
    record_routes: bool = False,
    audit_pair_scores: bool = False,
    grouped_owner_policy: str | None = None,
    grouped_ep_size: int = 1,
    grouped_owner_mapping: str = "contiguous",
    grouped_tile_vectors: int = 64,
    grouped_high_fraction: float = 0.5,
    grouped_expert_gain_profile: dict[int, torch.Tensor] | None = None,
    creditreduce_endpoint: str | None = None,
    creditreduce_ep_size: int = 8,
    creditreduce_ranks_per_domain: int = 1,
    creditreduce_placement: str = "contiguous",
    creditreduce_residual_rms_threshold: float = float("inf"),
    creditreduce_record_detail: bool = False,
    record_diagnostics: bool = True,
) -> MoeRecorder:
    full_policy = make_policy("full")
    actual_policy = make_policy(policy_name)
    recorder = MoeRecorder(
        num_receiver_groups=num_receiver_groups,
        receiver_mapping=receiver_mapping,
        record_routes=record_routes,
        audit_pair_scores=audit_pair_scores,
    )
    if routing_cache is not None:
        recorder.routing_cache = routing_cache

    top_k = int(getattr(model.config, "num_experts_per_tok", 8))
    dispatch_precisions = parse_dispatch_policy(dispatch_policy_name, top_k)

    for layer_id, layer in enumerate(model.model.layers):
        if hasattr(layer, "block_sparse_moe"):
            moe = layer.block_sparse_moe
            forward_impl = _patched_mixtral_sparse_moe_forward
        elif (
            hasattr(layer, "mlp")
            and hasattr(layer.mlp, "experts")
            and hasattr(layer.mlp, "gate")
            and hasattr(layer.mlp, "shared_expert")
        ):
            moe = layer.mlp
            forward_impl = _patched_qwen2_moe_sparse_moe_forward
        elif hasattr(layer, "mlp") and hasattr(layer.mlp, "experts") and hasattr(layer.mlp, "gate"):
            moe = layer.mlp
            forward_impl = _patched_olmoe_sparse_moe_forward
        else:
            # Hybrid dense/MoE models legitimately contain non-MoE layers.
            # Preserve their original forward and patch only layers with experts.
            continue
        moe._idea_layer_id = layer_id
        if lut is not None:
            layer_lut = {(r, R): p for (l, r, R), p in lut.items() if l == layer_id}
            moe._idea_policy = make_policy("lut", layer_lut=layer_lut)
        else:
            moe._idea_policy = actual_policy if (target_layer is None or layer_id == target_layer) else full_policy
        moe._idea_recorder = recorder
        moe._idea_num_receiver_groups = num_receiver_groups
        moe._idea_receiver_mapping = receiver_mapping
        moe._idea_cache_routing = cache_routing
        moe._idea_lock_routing = lock_routing
        moe._idea_dispatch_precisions = dispatch_precisions
        moe._idea_grouped_owner_policy = grouped_owner_policy
        moe._idea_grouped_ep_size = grouped_ep_size
        moe._idea_grouped_owner_mapping = grouped_owner_mapping
        moe._idea_grouped_tile_vectors = grouped_tile_vectors
        moe._idea_grouped_high_fraction = grouped_high_fraction
        moe._idea_grouped_expert_gain_profile = (
            grouped_expert_gain_profile.get(layer_id)
            if grouped_expert_gain_profile is not None
            else None
        )
        moe._idea_creditreduce_endpoint = creditreduce_endpoint
        moe._idea_creditreduce_ep_size = creditreduce_ep_size
        moe._idea_creditreduce_ranks_per_domain = creditreduce_ranks_per_domain
        moe._idea_creditreduce_placement = creditreduce_placement
        moe._idea_creditreduce_residual_rms_threshold = (
            creditreduce_residual_rms_threshold
        )
        moe._idea_creditreduce_record_detail = creditreduce_record_detail
        moe._idea_record_diagnostics = record_diagnostics
        moe.forward = MethodType(forward_impl, moe)

    return recorder


def _combine_idea_paths(
    moe,
    raw_outputs: torch.Tensor,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    input_norm: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Choose the legacy per-pair path or owner-grouped combine path."""
    creditreduce_endpoint = getattr(moe, "_idea_creditreduce_endpoint", None)
    grouped_policy = getattr(moe, "_idea_grouped_owner_policy", None)
    record_diagnostics = bool(getattr(moe, "_idea_record_diagnostics", True))

    if not record_diagnostics and creditreduce_endpoint is None and grouped_policy is None:
        approx_outputs, approx_weights = moe._idea_policy.apply(
            raw_outputs,
            routing_weights,
            selected_experts=selected_experts,
            num_experts=moe.num_experts,
            num_receiver_groups=moe._idea_num_receiver_groups,
            receiver_mapping=moe._idea_receiver_mapping,
        )
        approximation = _combine_expert_order(
            approx_outputs, approx_weights, selected_experts, moe.num_experts
        )
        return approximation, approximation

    pair_full = _combine_expert_order(
        raw_outputs, routing_weights, selected_experts, moe.num_experts
    )
    if creditreduce_endpoint is not None:
        token_count = raw_outputs.shape[0]
        home_ranks = (
            torch.arange(
                token_count,
                dtype=torch.int64,
                device=selected_experts.device,
            )
            + int(moe._idea_recorder.current_sample_id)
            + int(moe._idea_layer_id)
        ).remainder(int(moe._idea_creditreduce_ep_size))
        result = creditreduce_reference(
            raw_outputs,
            routing_weights,
            selected_experts,
            num_experts=moe.num_experts,
            ep_size=int(moe._idea_creditreduce_ep_size),
            ranks_per_domain=int(moe._idea_creditreduce_ranks_per_domain),
            placement=str(moe._idea_creditreduce_placement),
            home_ranks=home_ranks,
            residual_rms_threshold=float(
                moe._idea_creditreduce_residual_rms_threshold
            ),
        )
        if creditreduce_endpoint not in result.outputs:
            raise ValueError(
                f"unknown CreditReduce endpoint: {creditreduce_endpoint}"
            )
        record_detail = bool(moe._idea_creditreduce_record_detail)
        records = (
            result.diagnostics.recorder_records()
            if record_detail
            else {"aggregate": result.diagnostics.aggregate()}
        )
        moe._idea_recorder.update_creditreduce(
            moe._idea_layer_id,
            creditreduce_endpoint,
            records,
            record_detail=record_detail,
        )
        return result.outputs["late_bf16"], result.outputs[creditreduce_endpoint]

    grouped_policy = getattr(moe, "_idea_grouped_owner_policy", None)
    if grouped_policy is not None:
        grouped_full, approximation, diagnostics = grouped_owner_combine(
            raw_outputs,
            routing_weights,
            selected_experts,
            num_experts=moe.num_experts,
            ep_size=moe._idea_grouped_ep_size,
            mapping=moe._idea_grouped_owner_mapping,
            policy=grouped_policy,
            tile_vectors=moe._idea_grouped_tile_vectors,
            high_fraction=moe._idea_grouped_high_fraction,
            input_norm=input_norm,
            expert_gain_profile=moe._idea_grouped_expert_gain_profile,
        )
        moe._idea_recorder.update_grouped_owner(
            moe._idea_layer_id,
            diagnostics,
            grouped_full,
            pair_full,
        )
        return grouped_full, approximation

    approx_outputs, approx_weights = moe._idea_policy.apply(
        raw_outputs,
        routing_weights,
        selected_experts=selected_experts,
        num_experts=moe.num_experts,
        num_receiver_groups=moe._idea_num_receiver_groups,
        receiver_mapping=moe._idea_receiver_mapping,
    )
    approximation = _combine_expert_order(
        approx_outputs, approx_weights, selected_experts, moe.num_experts
    )
    return pair_full, approximation


def _patched_mixtral_sparse_moe_forward(self, hidden_states: torch.Tensor):
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    if self.training and self.jitter_noise > 0:
        hidden_states *= torch.empty_like(hidden_states).uniform_(1.0 - self.jitter_noise, 1.0 + self.jitter_noise)
    hidden_states = hidden_states.view(-1, hidden_dim)
    needs_input_norm = bool(getattr(self, "_idea_record_diagnostics", True)) or (
        getattr(self, "_idea_grouped_owner_policy", None) is not None
    )
    input_norm = (
        hidden_states.float().norm(dim=-1)
        if needs_input_norm
        else torch.empty(0, device=hidden_states.device, dtype=torch.float32)
    )

    router_logits = self.gate(hidden_states)
    if getattr(self, "_idea_lock_routing", False):
        selected_experts, routing_weights = self._idea_recorder.routing_cache[self._idea_layer_id]
    else:
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)
        if getattr(self, "_idea_cache_routing", False):
            self._idea_recorder.routing_cache[self._idea_layer_id] = (
                selected_experts.clone(),
                routing_weights.clone(),
            )

    # Route capture is a cheap, independent capability.  It must not inherit
    # the expensive contribution/error diagnostics gate: identity-complete
    # route producers intentionally run with record_diagnostics=False.
    self._idea_recorder.update_routing(
        self._idea_layer_id, selected_experts, routing_weights
    )

    total_tokens = batch_size * sequence_length
    raw_outputs = torch.zeros(
        (total_tokens, self.top_k, hidden_dim),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
    expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
    for expert_idx_tensor in expert_hit:
        expert_idx = int(expert_idx_tensor.item())
        expert_layer = self.experts[expert_idx]
        idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
        current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
        raw_outputs[top_x, idx, :] = expert_layer(current_state)

    if bool(getattr(self, "_idea_record_diagnostics", True)):
        contrib = routing_weights.float() * raw_outputs.float().norm(dim=-1)
        shares = contrib / contrib.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        self._idea_recorder.update_pair_audit(
            self._idea_layer_id, raw_outputs, routing_weights, selected_experts
        )
        self._idea_recorder.update_contrib(self._idea_layer_id, shares)
        if (
            getattr(self, "_idea_grouped_owner_policy", None) is None
            and getattr(self, "_idea_creditreduce_endpoint", None) is None
        ):
            self._idea_recorder.update_receiver(
                self._idea_layer_id,
                selected_experts,
                shares,
                hidden_dim,
                self.num_experts,
                self._idea_policy,
                routing_weights,
            )

    full_final, approx_final = _combine_idea_paths(
        self, raw_outputs, routing_weights, selected_experts, input_norm
    )
    if bool(getattr(self, "_idea_record_diagnostics", True)):
        self._idea_recorder.update_error(self._idea_layer_id, approx_final, full_final)

    return approx_final.reshape(batch_size, sequence_length, hidden_dim), router_logits


def _combine_expert_order(
    outputs: torch.Tensor,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """Reproduce Transformers' expert-id-ordered BF16 accumulation.

    Transformers visits experts in expert-id order and performs one
    ``index_add_`` per expert.  Summing the top-k dimension changes BF16
    associativity, which is enough to perturb downstream routing.  Quality
    experiments need the patched ``full`` policy to be an exact reference.
    """
    total_tokens, _top_k, hidden_dim = outputs.shape
    final = torch.zeros(
        (total_tokens, hidden_dim), dtype=outputs.dtype, device=outputs.device
    )
    expert_mask = F.one_hot(
        selected_experts, num_classes=num_experts
    ).permute(2, 1, 0)
    for expert_idx in range(num_experts):
        rank_idx, token_idx = torch.where(expert_mask[expert_idx])
        current = outputs[token_idx, rank_idx, :] * routing_weights[
            token_idx, rank_idx, None
        ]
        final.index_add_(0, token_idx, current.to(outputs.dtype))
    return final


def _patched_olmoe_sparse_moe_forward(self, hidden_states: torch.Tensor):
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)
    needs_input_norm = bool(getattr(self, "_idea_record_diagnostics", True)) or (
        getattr(self, "_idea_grouped_owner_policy", None) is not None
    )
    input_norm = (
        hidden_states.float().norm(dim=-1)
        if needs_input_norm
        else torch.empty(0, device=hidden_states.device, dtype=torch.float32)
    )

    router_logits = self.gate(hidden_states)
    if getattr(self, "_idea_lock_routing", False):
        selected_experts, routing_weights = self._idea_recorder.routing_cache[self._idea_layer_id]
    else:
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)
        if getattr(self, "_idea_cache_routing", False):
            self._idea_recorder.routing_cache[self._idea_layer_id] = (
                selected_experts.clone(),
                routing_weights.clone(),
            )

    self._idea_recorder.update_routing(
        self._idea_layer_id, selected_experts, routing_weights
    )

    total_tokens = batch_size * sequence_length
    raw_outputs = torch.zeros(
        (total_tokens, self.top_k, hidden_dim),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
    expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
    dispatch_precisions = getattr(self, "_idea_dispatch_precisions", None)
    for expert_idx_tensor in expert_hit:
        expert_idx = int(expert_idx_tensor.item())
        expert_layer = self.experts[expert_idx]
        idx, top_x = torch.where(expert_mask[expert_idx])
        current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
        if dispatch_precisions is not None:
            for rank_val in idx.unique():
                rank_mask = idx == rank_val
                prec = dispatch_precisions[int(rank_val.item())]
                if prec != "bf16":
                    current_state[rank_mask] = apply_precision(current_state[rank_mask], prec)
        raw_outputs[top_x, idx, :] = expert_layer(current_state)

    if bool(getattr(self, "_idea_record_diagnostics", True)):
        contrib = routing_weights.float() * raw_outputs.float().norm(dim=-1)
        shares = contrib / contrib.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        self._idea_recorder.update_pair_audit(
            self._idea_layer_id, raw_outputs, routing_weights, selected_experts
        )
        self._idea_recorder.update_contrib(self._idea_layer_id, shares)
        if (
            getattr(self, "_idea_grouped_owner_policy", None) is None
            and getattr(self, "_idea_creditreduce_endpoint", None) is None
        ):
            self._idea_recorder.update_receiver(
                self._idea_layer_id,
                selected_experts,
                shares,
                hidden_dim,
                self.num_experts,
                self._idea_policy,
                routing_weights,
            )

    full_final, approx_final = _combine_idea_paths(
        self, raw_outputs, routing_weights, selected_experts, input_norm
    )
    if bool(getattr(self, "_idea_record_diagnostics", True)):
        self._idea_recorder.update_error(self._idea_layer_id, approx_final, full_final)

    return approx_final.reshape(batch_size, sequence_length, hidden_dim), router_logits


def _patched_qwen2_moe_sparse_moe_forward(self, hidden_states: torch.Tensor):
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)
    needs_input_norm = bool(getattr(self, "_idea_record_diagnostics", True)) or (
        getattr(self, "_idea_grouped_owner_policy", None) is not None
    )
    input_norm = (
        hidden_states.float().norm(dim=-1)
        if needs_input_norm
        else torch.empty(0, device=hidden_states.device, dtype=torch.float32)
    )

    router_logits = self.gate(hidden_states)
    if getattr(self, "_idea_lock_routing", False):
        selected_experts, routing_weights = self._idea_recorder.routing_cache[self._idea_layer_id]
    else:
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)
        if getattr(self, "_idea_cache_routing", False):
            self._idea_recorder.routing_cache[self._idea_layer_id] = (
                selected_experts.clone(),
                routing_weights.clone(),
            )

    self._idea_recorder.update_routing(
        self._idea_layer_id, selected_experts, routing_weights
    )

    total_tokens = batch_size * sequence_length
    raw_outputs = torch.zeros(
        (total_tokens, self.top_k, hidden_dim),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
    expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
    for expert_idx_tensor in expert_hit:
        expert_idx = int(expert_idx_tensor.item())
        expert_layer = self.experts[expert_idx]
        idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
        current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
        raw_outputs[top_x, idx, :] = expert_layer(current_state)

    shared_expert_output = self.shared_expert(hidden_states)
    shared_expert_output = F.sigmoid(self.shared_expert_gate(hidden_states)) * shared_expert_output

    if bool(getattr(self, "_idea_record_diagnostics", True)):
        contrib = routing_weights.float() * raw_outputs.float().norm(dim=-1)
        shares = contrib / contrib.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        self._idea_recorder.update_pair_audit(
            self._idea_layer_id, raw_outputs, routing_weights, selected_experts
        )
        self._idea_recorder.update_contrib(self._idea_layer_id, shares)
        if (
            getattr(self, "_idea_grouped_owner_policy", None) is None
            and getattr(self, "_idea_creditreduce_endpoint", None) is None
        ):
            self._idea_recorder.update_receiver(
                self._idea_layer_id,
                selected_experts,
                shares,
                hidden_dim,
                self.num_experts,
                self._idea_policy,
                routing_weights,
            )

    full_routed, approx_routed = _combine_idea_paths(
        self, raw_outputs, routing_weights, selected_experts, input_norm
    )
    full_final = full_routed + shared_expert_output
    approx_final = approx_routed + shared_expert_output
    if bool(getattr(self, "_idea_record_diagnostics", True)):
        self._idea_recorder.update_error(self._idea_layer_id, approx_final, full_final)

    return approx_final.reshape(batch_size, sequence_length, hidden_dim), router_logits
