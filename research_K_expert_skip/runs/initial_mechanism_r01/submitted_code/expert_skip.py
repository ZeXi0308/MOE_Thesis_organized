"""Post-selection expert skipping for the vLLM 0.26 Triton MoE path.

The policy accepts the router's actual weights, not an assumed sorted top-k.
Thresholds and the budget are fractions of the *selected* gate mass. Rank is
zero-based, descending by actual weight; ties keep the earlier original slot.
`global` with mass_budget=1 is exact top-(k-drop_j), `gate` ignores batch and
rank, and `marginal` additionally requires tail rank and singleton ownership.

Statistics are device tensors and synchronize only on explicit snapshot().
Logical weight bytes are an access-footprint proxy, NOT measured HBM traffic.
Install safe alignment before emitting -1 IDs. Only the single-GPU Triton
BF16/FP8 path is in the initial integration scope; no EP/EPLB/LoRA/monolithic
backend support is implied. Install before graph capture; changing a Python
configuration after capture does not update captured kernel arguments.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib
import math
from typing import Any


@dataclass(frozen=True)
class SkipConfig:
    mode: str = "dense"
    drop_j: int = 1
    tail_rank: int = 4
    gate_threshold: float = 0.05
    mass_budget: float = 1.0
    backend: str = "torch"

    def __post_init__(self):
        if self.mode not in {"dense", "global", "gate", "marginal"}:
            raise ValueError(f"unknown policy mode: {self.mode}")
        if self.backend not in {"torch", "triton"}:
            raise ValueError(f"unknown backend: {self.backend}")
        if self.drop_j < 0 or self.tail_rank < 0:
            raise ValueError("drop_j and tail_rank must be nonnegative")
        if not 0 <= self.gate_threshold <= 1 or not 0 <= self.mass_budget <= 1:
            raise ValueError("gate_threshold and mass_budget must be in [0, 1]")


STAT_NAMES = (
    "events", "tokens", "slots_before", "slots_after", "skipped_slots",
    "singleton_slots_before", "distinct_before", "distinct_after",
    "removed_gate_mass_sum", "logical_weight_bytes_before",
    "logical_weight_bytes_after", "tokens_with_skips", "removed_gate_mass_max",
)


def reference_policy(weights, ids, num_experts, config: SkipConfig,
                     renormalize=True, weight_bytes_per_expert=0):
    """Dependency-free, strict CPU oracle; returns (weights, ids, stats).

    Valid router rows must contain distinct nonnegative expert IDs. Previously
    skipped slots (-1, weight=0) are accepted for replay and diagnostic use.
    """
    if len(weights) != len(ids) or num_experts <= 0:
        raise ValueError("invalid routing shape or expert count")
    counts = [0] * num_experts
    k = len(weights[0]) if weights else 0
    for row_w, row_i in zip(weights, ids):
        if len(row_w) != k or len(row_i) != k or not k:
            raise ValueError("routing rows must be rectangular and nonempty")
        valid_ids = [e for e in row_i if e >= 0]
        if len(set(valid_ids)) != len(valid_ids) or not valid_ids:
            raise ValueError("each token needs distinct valid experts")
        for w, e in zip(row_w, row_i):
            if not math.isfinite(w) or w < 0 or e < -1 or e >= num_experts:
                raise ValueError("invalid weight or expert ID")
            if e == -1 and w != 0:
                raise ValueError("invalid slots must have zero weight")
            if e >= 0:
                counts[e] += 1
    out_w, out_i, removed_fractions = [], [], []
    for row_w, row_i in zip(weights, ids):
        valid = [j for j, e in enumerate(row_i) if e >= 0]
        order = sorted(valid, key=lambda j: (-row_w[j], j))
        ranks = {j: r for r, j in enumerate(order)}
        total = sum(row_w)
        fractions = [w / total if total else 0.0 for w in row_w]
        candidates = []
        for j in valid:
            r, e, g = ranks[j], row_i[j], fractions[j]
            if r == 0:
                continue
            take = config.mode == "global" and r >= max(1, len(valid)-config.drop_j)
            take |= config.mode == "gate" and g <= config.gate_threshold
            take |= (config.mode == "marginal" and r >= config.tail_rank
                     and g <= config.gate_threshold and counts[e] == 1)
            if take:
                candidates.append(j)
        # Spend the budget on the weakest eligible contributions first.
        candidates.sort(key=lambda j: (row_w[j], -j))
        removed = 0.0
        drop = set()
        for j in candidates:
            next_mass = removed + fractions[j]
            if next_mass <= config.mass_budget:
                drop.add(j)
                removed = next_mass
            else:
                break
        new_w = [0.0 if j in drop else float(w) for j, w in enumerate(row_w)]
        kept = sum(new_w)
        if renormalize and drop and kept > 0:
            # Preserve routed_scaling_factor and any other model-level scaling.
            new_w = [w * total / kept for w in new_w]
        out_w.append(new_w)
        out_i.append([-1 if j in drop else e for j, e in enumerate(row_i)])
        removed_fractions.append(removed)
    after = [0] * num_experts
    for row in out_i:
        for e in row:
            if e >= 0:
                after[e] += 1
    before_n, after_n = sum(counts), sum(after)
    distinct_before, distinct_after = sum(c > 0 for c in counts), sum(c > 0 for c in after)
    stats = dict(zip(STAT_NAMES, (
        1, len(ids), before_n, after_n, before_n-after_n,
        sum(c == 1 for c in counts), distinct_before, distinct_after,
        sum(removed_fractions), distinct_before*weight_bytes_per_expert,
        distinct_after*weight_bytes_per_expert,
        sum(a != b for a, b in zip(ids, out_i)),
        max(removed_fractions, default=0.0),
    )))
    return out_w, out_i, stats


class ExpertSkipPolicy:
    """Callable policy; a single object can hold separate per-layer counters."""

    def __init__(self, config: SkipConfig | dict | None = None):
        self.config = (SkipConfig(**config) if isinstance(config, dict)
                       else config or SkipConfig())
        self._stats: dict[tuple[str, str], Any] = {}

    def _counter(self, name, device):
        import torch
        key = (name, str(device))
        if key not in self._stats:
            self._stats[key] = torch.zeros(len(STAT_NAMES), device=device, dtype=torch.float64)
        return self._stats[key]

    def __call__(self, name, weights, ids, num_experts, renormalize,
                 weight_bytes_per_expert=0):
        import torch
        if weights.ndim != 2 or ids.shape != weights.shape or ids.shape[1] < 1:
            raise ValueError("expected matching [tokens, topk] weights and IDs")
        if num_experts <= 0 or weight_bytes_per_expert < 0:
            raise ValueError("invalid expert count or weight byte size")
        if ids.device != weights.device:
            raise ValueError("weights and IDs must share a device")
        counter = self._counter(name, weights.device)
        if self.config.backend == "triton":
            return _triton_policy(weights, ids, num_experts, self.config,
                                  renormalize, weight_bytes_per_expert, counter)
        valid = (ids >= 0) & (ids < num_experts)
        safe_ids = ids.clamp(0, num_experts).long()
        # Fixed shape scatter avoids bincount's data-dependent output shape.
        counts = torch.zeros(num_experts+1, dtype=torch.int32, device=ids.device)
        counts.scatter_add_(0, safe_ids.flatten(), valid.flatten().to(torch.int32))
        total = weights.float().sum(dim=-1, keepdim=True)
        fractions = weights.float() / total.clamp_min(torch.finfo(torch.float32).tiny)
        order = torch.argsort(weights.masked_fill(~valid, -torch.inf), dim=-1,
                              descending=True, stable=True)
        rank = torch.empty_like(order)
        rank.scatter_(1, order, torch.arange(ids.shape[1], device=ids.device).expand_as(order))
        cfg = self.config
        if cfg.mode == "global":
            cutoff = (valid.sum(dim=1, keepdim=True)-cfg.drop_j).clamp_min(1)
            eligible = valid & (rank >= cutoff)
        elif cfg.mode == "gate":
            eligible = valid & (fractions <= cfg.gate_threshold)
        elif cfg.mode == "marginal":
            eligible = (valid & (rank >= cfg.tail_rank)
                        & (fractions <= cfg.gate_threshold) & (counts[safe_ids] == 1))
        else:
            eligible = torch.zeros_like(valid)
        eligible &= rank > 0
        weak_order = order.flip(1)
        eligible_weak = eligible.gather(1, weak_order)
        mass_weak = (fractions.gather(1, weak_order) * eligible_weak).cumsum(1)
        drop_weak = eligible_weak & (mass_weak <= cfg.mass_budget)
        drop = torch.zeros_like(valid).scatter_(1, weak_order, drop_weak)
        out_ids = ids.masked_fill(drop, -1)
        out_weights = weights.masked_fill(drop, 0)
        if renormalize:
            keep_mass = out_weights.float().sum(dim=-1, keepdim=True)
            scale = torch.where(drop.any(dim=1, keepdim=True) & (keep_mass > 0),
                                total / keep_mass.clamp_min(torch.finfo(torch.float32).tiny),
                                torch.ones_like(total))
            out_weights = (out_weights * scale).to(weights.dtype)
        after = torch.zeros_like(counts)
        after.scatter_add_(0, safe_ids.flatten(), (valid & ~drop).flatten().to(torch.int32))
        distinct_before = (counts[:num_experts] > 0).sum()
        distinct_after = (after[:num_experts] > 0).sum()
        removed_mass = (fractions * drop).sum(1)
        values = torch.stack([
            counter.new_tensor(1), counter.new_tensor(ids.shape[0]),
            valid.sum(), (valid & ~drop).sum(), drop.sum(),
            (counts[:num_experts] == 1).sum(), distinct_before, distinct_after,
            removed_mass.sum(), distinct_before * weight_bytes_per_expert,
            distinct_after * weight_bytes_per_expert, drop.any(dim=1).sum(),
        ]).to(counter.dtype)
        counter[:-1].add_(values)
        if ids.shape[0]:
            counter[-1].copy_(torch.maximum(counter[-1], removed_mass.max()))
        # Preserve exact dense/zero-drop values; multiplication by one is exact.
        return out_weights, out_ids

    def reset(self):
        """Zero existing tensors in place (captured graphs retain their pointers)."""
        for tensor in self._stats.values():
            tensor.zero_()

    def snapshot(self, reset=False):
        """The only host synchronization in this module; call outside timing."""
        layers = {}
        totals = dict.fromkeys(STAT_NAMES, 0.0)
        for (name, device), tensor in self._stats.items():
            values = dict(zip(STAT_NAMES, tensor.detach().cpu().tolist()))
            layers[f"{name}@{device}"] = values
            for key, value in values.items():
                if key == "removed_gate_mass_max":
                    totals[key] = max(totals[key], value)
                else:
                    totals[key] += value
        if reset:
            self.reset()
        return {"config": asdict(self.config), "totals": totals, "layers": layers,
                "byte_semantics": "distinct expert logical weight footprint; not HBM traffic"}


class AlignmentPatch:
    def __init__(self, originals, wrapper):
        self.originals, self.wrapper = originals, wrapper

    def restore(self):
        for module, name, original in self.originals:
            if getattr(module, name) is self.wrapper:
                setattr(module, name, original)


def install_safe_alignment():
    """Map negative IDs to an extra sentinel expert before native histogram.

    Keep the sentinel block in the assignment so both native Triton GEMMs hit
    `off_experts == -1`, explicitly write output zeros, and return before any
    expert-weight access. Filtering slots out altogether would leave reused
    workspaces unwritten. Native low-token naive assignment already performs
    that same zero-writing -1 branch and does not need modification.

    The wrapper intentionally sets ignore_invalid_experts=False even if the
    caller requested True: skipped slots need explicit zero-output blocks.
    """
    import torch
    base_name = "vllm.model_executor.layers.fused_moe"
    align_module = importlib.import_module(base_name+".moe_align_block_size")
    native = align_module.moe_align_block_size
    if getattr(native, "_expert_skip_safe_alignment", False):
        raise RuntimeError("safe alignment is already installed")
    maps = {}

    def safe_align(topk_ids, block_size, num_experts, expert_map=None,
                   pad_sorted_ids=False, ignore_invalid_experts=False):
        # No tensor->host predicate: this is safe to record in CUDA graphs.
        routed_ids = torch.where(topk_ids < 0, num_experts, topk_ids)
        if expert_map is None:
            key = (num_experts, str(topk_ids.device))
            if key not in maps:
                maps[key] = torch.cat((
                    torch.arange(num_experts, dtype=torch.int32, device=topk_ids.device),
                    torch.full((1,), -1, dtype=torch.int32, device=topk_ids.device)))
            extended_map = maps[key]
        else:
            extended_map = torch.cat((expert_map, expert_map.new_full((1,), -1)))
        return native(routed_ids, block_size, num_experts+1, extended_map,
                      pad_sorted_ids=pad_sorted_ids, ignore_invalid_experts=False)

    safe_align._expert_skip_safe_alignment = True
    originals = []
    for suffix in (".moe_align_block_size", ".fused_moe", ".experts.triton_moe"):
        module = importlib.import_module(base_name+suffix)
        original = module.moe_align_block_size
        originals.append((module, "moe_align_block_size", original))
        module.moe_align_block_size = safe_align
    return AlignmentPatch(originals, safe_align)


def _triton_policy(*args, **kwargs):
    raise NotImplementedError("Triton backend is not implemented yet; use backend='torch'")
