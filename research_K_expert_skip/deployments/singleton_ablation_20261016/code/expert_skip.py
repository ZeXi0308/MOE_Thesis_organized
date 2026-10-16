"""Post-selection expert skipping for the vLLM 0.26 Triton MoE path.

The policy accepts the router's actual weights, not an assumed sorted top-k.
Thresholds and the budget are fractions of the *selected* gate mass. Rank is
zero-based, descending by actual weight; ties keep the earlier original slot.
`global` with mass_budget=1 is exact top-(k-drop_j), `gate` ignores batch and
rank, and `marginal` additionally requires tail rank and, by default, singleton
ownership. `require_singleton=False` removes only that marginal count predicate;
its inclusive gate/mass boundaries and all other policy semantics are unchanged.

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
    rank_end: int | None = None
    gate_min: float = 0.0
    require_singleton: bool = True

    def __post_init__(self):
        if self.mode not in {"dense", "global", "gate", "marginal", "rank_gate"}:
            raise ValueError(f"unknown policy mode: {self.mode}")
        if self.backend not in {"torch", "triton", "triton_multi"}:
            raise ValueError(f"unknown backend: {self.backend}")
        if type(self.require_singleton) is not bool:
            raise ValueError("require_singleton must be a bool")
        if self.drop_j < 0 or self.tail_rank < 0:
            raise ValueError("drop_j and tail_rank must be nonnegative")
        if not 0 <= self.gate_threshold <= 1 or not 0 <= self.mass_budget <= 1:
            raise ValueError("gate_threshold and mass_budget must be in [0, 1]")
        if not 0 <= self.gate_min <= self.gate_threshold:
            raise ValueError("gate_min must be in [0, gate_threshold]")
        if self.rank_end is not None and self.rank_end <= self.tail_rank:
            raise ValueError("rank_end must be greater than tail_rank")


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
                     and g <= config.gate_threshold
                     and (not config.require_singleton or counts[e] == 1))
            take |= (config.mode == "rank_gate" and config.tail_rank <= r
                     and r < (config.rank_end if config.rank_end is not None else len(valid))
                     and config.gate_min <= g < config.gate_threshold)
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

    def __init__(self, config: SkipConfig | dict | None = None, collect_stats=True):
        self.config = (SkipConfig(**config) if isinstance(config, dict)
                       else config or SkipConfig())
        self._stats: dict[tuple[str, str], Any] = {}
        self.collect_stats = bool(collect_stats)

    def _counter(self, name, device):
        import torch
        key = (name, str(device))
        if key not in self._stats:
            self._stats[key] = torch.zeros(len(STAT_NAMES), device=device, dtype=torch.float64)
        return self._stats[key]

    def __call__(self, name, weights, ids, num_experts, renormalize,
                 weight_bytes_per_expert=0, active_tokens=None):
        import torch
        if weights.ndim != 2 or ids.shape != weights.shape or ids.shape[1] < 1:
            raise ValueError("expected matching [tokens, topk] weights and IDs")
        if num_experts <= 0 or weight_bytes_per_expert < 0:
            raise ValueError("invalid expert count or weight byte size")
        if ids.device != weights.device:
            raise ValueError("weights and IDs must share a device")
        if active_tokens is not None:
            if (active_tokens.numel()!=1 or active_tokens.device!=weights.device
                    or active_tokens.dtype not in (torch.int32,torch.int64)):
                raise ValueError('active_tokens must be an int32/int64 scalar on the routing device')
        counter = self._counter(name, weights.device) if self.collect_stats else weights
        if self.config.backend in {"triton", "triton_multi"}:
            return _triton_policy(weights, ids, num_experts, self.config,
                                  renormalize, weight_bytes_per_expert, counter,
                                  active_tokens, self.collect_stats)
        valid = (ids >= 0) & (ids < num_experts)
        active_rows = torch.ones(ids.shape[0],dtype=torch.bool,device=ids.device)
        if active_tokens is not None:
            active_rows = torch.arange(ids.shape[0],device=ids.device) < active_tokens
            valid &= active_rows[:,None]
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
                        & (fractions <= cfg.gate_threshold))
            if cfg.require_singleton:
                eligible &= counts[safe_ids] == 1
        elif cfg.mode == "rank_gate":
            eligible = (valid & (rank >= cfg.tail_rank)
                        & (rank < (cfg.rank_end if cfg.rank_end is not None else ids.shape[1]))
                        & (fractions >= cfg.gate_min) & (fractions < cfg.gate_threshold))
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
        if not self.collect_stats:
            return out_weights,out_ids
        after = torch.zeros_like(counts)
        after.scatter_add_(0, safe_ids.flatten(), (valid & ~drop).flatten().to(torch.int32))
        distinct_before = (counts[:num_experts] > 0).sum()
        distinct_after = (after[:num_experts] > 0).sum()
        removed_mass = (fractions * drop).sum(1)
        one = ((active_rows.sum()>0).to(counter.dtype) if active_tokens is not None
               else torch.ones_like(counter[0]))
        values = torch.stack([
            one, active_rows.sum(),
            valid.sum(), (valid & ~drop).sum(), drop.sum(),
            (counts[:num_experts] == 1).sum(), distinct_before, distinct_after,
            removed_mass.sum(), distinct_before * weight_bytes_per_expert,
            distinct_after * weight_bytes_per_expert, drop.any(dim=1).sum(),
        ]).to(counter.dtype)
        # Counters may have been allocated during an inference-mode forward,
        # then reused by a caller outside that context. Keep storage stable.
        with torch.inference_mode():
            counter[:-1].add_(values)
            if ids.shape[0]:
                counter[-1].copy_(torch.maximum(counter[-1], removed_mass.max()))
        # Preserve exact dense/zero-drop values; multiplication by one is exact.
        return out_weights, out_ids

    def reset(self):
        """Zero existing tensors in place (captured graphs retain their pointers)."""
        if not self._stats:
            return
        import torch
        # vLLM forwards create inference tensors; snapshot/reset RPCs need not
        # run in InferenceMode. Replacing or cloning these tensors would leave
        # captured CUDA graphs writing to stale addresses.
        with torch.inference_mode():
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
                "collect_stats": self.collect_stats,
                "byte_semantics": "distinct expert logical weight footprint; not HBM traffic"}


class AlignmentPatch:
    def __init__(self, originals, wrapper, prepare_maps=None):
        self.originals, self.wrapper = originals, wrapper
        self.prepare_maps = prepare_maps

    def restore(self):
        for module, name, original in self.originals:
            if getattr(module, name) is self.wrapper:
                setattr(module, name, original)


def install_safe_alignment(enable_predicate=None):
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

    def prepare_maps(num_experts,device):
        key=(num_experts,str(device))
        if key not in maps:
            maps[key]=torch.cat((
                torch.arange(num_experts,dtype=torch.int32,device=device),
                torch.full((1,),-1,dtype=torch.int32,device=device)))
        return maps[key]

    def safe_align(topk_ids, block_size, num_experts, expert_map=None,
                   pad_sorted_ids=False, ignore_invalid_experts=False):
        if enable_predicate is not None and not enable_predicate():
            return native(topk_ids,block_size,num_experts,expert_map,
                          pad_sorted_ids=pad_sorted_ids,ignore_invalid_experts=ignore_invalid_experts)
        # No tensor->host predicate: this is safe to record in CUDA graphs.
        routed_ids = torch.where(topk_ids < 0, num_experts, topk_ids)
        if expert_map is None:
            extended_map = prepare_maps(num_experts,topk_ids.device)
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
    return AlignmentPatch(originals, safe_align, prepare_maps)


_TRITON_KERNELS = None


def _get_triton_kernels():
    """Lazy import keeps the reference and CPU tests dependency-free."""
    global _TRITON_KERNELS
    if _TRITON_KERNELS is not None:
        return _TRITON_KERNELS
    import triton
    import triton.language as tl

    @triton.jit
    def fused_small_kernel(weights, ids, out_weights, out_ids, counter, active_tokens,
                           M: tl.constexpr, K: tl.constexpr, E: tl.constexpr,
                           MODE: tl.constexpr, DROP_J: tl.constexpr,
                           TAIL: tl.constexpr, TAU: tl.constexpr,
                           RANK_END: tl.constexpr, GATE_MIN: tl.constexpr,
                           BUDGET: tl.constexpr, RENORM: tl.constexpr,
                           EXPERT_BYTES: tl.constexpr, BLOCK_M: tl.constexpr,
                           BLOCK_K: tl.constexpr, BLOCK_E: tl.constexpr,
                           HAS_ACTIVE: tl.constexpr, COLLECT_STATS: tl.constexpr,
                           REQUIRE_SINGLETON: tl.constexpr):
        t = tl.arange(0, BLOCK_M)
        j = tl.arange(0, BLOCK_K)
        offset = t[:, None]*K+j[None, :]
        in_shape = (t[:, None] < M) & (j[None, :] < K)
        expert = tl.load(ids+offset, in_shape, other=-1).to(tl.int32)
        w = tl.load(weights+offset, in_shape, other=0).to(tl.float32)
        actual_m = tl.minimum(M,tl.maximum(0,tl.load(active_tokens))) if HAS_ACTIVE else M
        valid = in_shape & (expert >= 0) & (expert < E) & (t[:,None] < actual_m)
        flat_expert = tl.reshape(expert, (BLOCK_M*BLOCK_K,))
        flat_valid = tl.reshape(valid, (BLOCK_M*BLOCK_K,))
        before = tl.histogram(flat_expert, BLOCK_E, mask=flat_valid)
        count = tl.reshape(tl.gather(before, tl.minimum(tl.maximum(flat_expert, 0), E-1), 0),
                           (BLOCK_M, BLOCK_K))
        total = tl.sum(w, 1)
        g = w/tl.maximum(total[:, None], 1.1754943508222875e-38)
        # Compare i with j within each token. Only selected, valid i count.
        better = ((w[:, :, None] > w[:, None, :]) |
                  ((w[:, :, None] == w[:, None, :]) &
                   (j[None, :, None] < j[None, None, :])))
        rank = tl.sum((better & valid[:, :, None]).to(tl.int32), 1)
        if MODE == 1:
            cutoff = tl.maximum(1, tl.sum(valid.to(tl.int32), 1)-DROP_J)
            eligible = valid & (rank >= cutoff[:, None])
        elif MODE == 2:
            eligible = valid & (g <= TAU)
        elif MODE == 3:
            eligible = valid & (rank >= TAIL) & (g <= TAU)
            if REQUIRE_SINGLETON:
                eligible = eligible & (count == 1)
        elif MODE == 4:
            eligible = valid & (rank >= TAIL) & (rank < RANK_END) & (g >= GATE_MIN) & (g < TAU)
        else:
            eligible = tl.full((BLOCK_M, BLOCK_K), False, tl.int1)
        eligible = eligible & (rank > 0)
        earlier_or_self = ((w[:, :, None] < w[:, None, :]) |
                           ((w[:, :, None] == w[:, None, :]) &
                            (j[None, :, None] >= j[None, None, :])))
        prefix = tl.sum(tl.where(eligible[:, :, None] & earlier_or_self,
                                g[:, :, None], 0.0), 1)
        drop = eligible & (prefix <= BUDGET)
        new_w = tl.where(drop, 0.0, w)
        changed = tl.sum(drop.to(tl.int32), 1) > 0
        if RENORM:
            kept = tl.sum(new_w, 1)
            scale = tl.where(changed & (kept > 0),
                             total/tl.maximum(kept, 1.1754943508222875e-38), 1.0)
            new_w = new_w*scale[:, None]
        tl.store(out_weights+offset, new_w, in_shape)
        tl.store(out_ids+offset, tl.where(drop, -1, expert), in_shape)
        if not COLLECT_STATS:
            return
        flat_keep = tl.reshape(valid & ~drop, (BLOCK_M*BLOCK_K,))
        after = tl.histogram(flat_expert, BLOCK_E, mask=flat_keep)
        slots_before = tl.sum(before, 0).to(tl.float64)
        slots_after = tl.sum(after, 0).to(tl.float64)
        singletons = tl.sum((before == 1).to(tl.int32), 0).to(tl.float64)
        distinct_before = tl.sum((before > 0).to(tl.int32), 0).to(tl.float64)
        distinct_after = tl.sum((after > 0).to(tl.int32), 0).to(tl.float64)
        mass = tl.sum(tl.where(drop, g, 0.0), 1).to(tl.float64)
        mass_sum, mass_max = tl.sum(mass, 0), tl.max(mass, 0)
        changed_sum = tl.sum(changed.to(tl.int32), 0).to(tl.float64)
        index = tl.arange(0, 16)
        value = tl.full((16,), 0, tl.float64)
        value = tl.where(index == 0, (actual_m>0) if HAS_ACTIVE else 1, value)
        value = tl.where(index == 1, actual_m, value)
        value = tl.where(index == 2, slots_before, value)
        value = tl.where(index == 3, slots_after, value)
        value = tl.where(index == 4, slots_before-slots_after, value)
        value = tl.where(index == 5, singletons, value)
        value = tl.where(index == 6, distinct_before, value)
        value = tl.where(index == 7, distinct_after, value)
        value = tl.where(index == 8, mass_sum, value)
        value = tl.where(index == 9, distinct_before*EXPERT_BYTES, value)
        value = tl.where(index == 10, distinct_after*EXPERT_BYTES, value)
        value = tl.where(index == 11, changed_sum, value)
        previous = tl.load(counter+index, index < 13, other=0)
        value = tl.where(index == 12, tl.maximum(previous, mass_max), previous+value)
        tl.store(counter+index, value, index < 13)

    @triton.jit
    def histogram_kernel(ids, counts, active_tokens, SLOTS: tl.constexpr, E: tl.constexpr,
                         K: tl.constexpr, HAS_ACTIVE: tl.constexpr, BLOCK: tl.constexpr):
        x = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
        e = tl.load(ids+x, mask=x < SLOTS, other=-1).to(tl.int32)
        valid = (x < SLOTS) & (e >= 0) & (e < E)
        if HAS_ACTIVE:
            valid = valid & (x < tl.load(active_tokens)*K)
        tl.atomic_add(counts+tl.maximum(e, 0), 1, mask=valid)

    @triton.jit
    def policy_kernel(weights, ids, counts, out_weights, out_ids, row_stats, active_tokens,
                      K: tl.constexpr, E: tl.constexpr, MODE: tl.constexpr,
                      DROP_J: tl.constexpr, TAIL: tl.constexpr,
                      TAU: tl.constexpr, BUDGET: tl.constexpr,
                      RANK_END: tl.constexpr, GATE_MIN: tl.constexpr,
                      RENORM: tl.constexpr, BLOCK_K: tl.constexpr,
                      HAS_ACTIVE: tl.constexpr, COLLECT_STATS: tl.constexpr,
                      REQUIRE_SINGLETON: tl.constexpr):
        t = tl.program_id(0)
        x = tl.arange(0, BLOCK_K)
        e = tl.load(ids+t*K+x, x < K, other=-1).to(tl.int32)
        w = tl.load(weights+t*K+x, x < K, other=0).to(tl.float32)
        valid = (x < K) & (e >= 0) & (e < E)
        if HAS_ACTIVE:
            valid = valid & (t < tl.load(active_tokens))
        total = tl.sum(w, 0)
        g = w / tl.maximum(total, 1.1754943508222875e-38)
        # Rows i precede columns j in descending gate order; tie by slot.
        better = ((w[:, None] > w[None, :]) |
                  ((w[:, None] == w[None, :]) & (x[:, None] < x[None, :])))
        rank = tl.sum((better & valid[:, None]).to(tl.int32), 0)
        if MODE == 1:
            cutoff = tl.maximum(1, tl.sum(valid.to(tl.int32), 0)-DROP_J)
            eligible = valid & (rank >= cutoff)
        elif MODE == 2:
            eligible = valid & (g <= TAU)
        elif MODE == 3:
            eligible = valid & (rank >= TAIL) & (g <= TAU)
            if REQUIRE_SINGLETON:
                count = tl.load(counts+tl.maximum(e, 0), mask=valid, other=0)
                eligible = eligible & (count == 1)
        elif MODE == 4:
            eligible = valid & (rank >= TAIL) & (rank < RANK_END) & (g >= GATE_MIN) & (g < TAU)
        else:
            eligible = tl.full((BLOCK_K,), False, tl.int1)
        eligible = eligible & (rank > 0)
        # Sum eligible mass from weakest through this slot. Pairwise KxK is
        # small for top-k <=32 and avoids sorted-topk assumptions entirely.
        earlier_or_self = ((w[:, None] < w[None, :]) |
                           ((w[:, None] == w[None, :]) & (x[:, None] >= x[None, :])))
        prefix = tl.sum(tl.where(eligible[:, None] & earlier_or_self,
                                g[:, None], 0.0), 0)
        drop = eligible & (prefix <= BUDGET)
        new_w = tl.where(drop, 0.0, w)
        if RENORM:
            kept = tl.sum(new_w, 0)
            scale = tl.where((tl.sum(drop.to(tl.int32), 0) > 0) & (kept > 0),
                             total / tl.maximum(kept, 1.1754943508222875e-38), 1.0)
            new_w = new_w * scale
        tl.store(out_weights+t*K+x, new_w, x < K)
        tl.store(out_ids+t*K+x, tl.where(drop, -1, e), x < K)
        if not COLLECT_STATS:
            return
        tl.atomic_add(counts+(E+1)+tl.maximum(e, 0), 1, mask=valid & ~drop)
        tl.store(row_stats+t*2, tl.sum(tl.where(drop, g, 0.0), 0))
        tl.store(row_stats+t*2+1, tl.sum(drop.to(tl.int32), 0) > 0)

    @triton.jit
    def stats_kernel(counts, row_stats, counter, active_tokens, E: tl.constexpr,
                     M: tl.constexpr, EXPERT_BYTES: tl.constexpr,
                     BLOCK_E: tl.constexpr, BLOCK_M: tl.constexpr, HAS_ACTIVE: tl.constexpr):
        e = tl.arange(0, BLOCK_E)
        before = tl.load(counts+e, e < E, other=0)
        after = tl.load(counts+E+1+e, e < E, other=0)
        slots_before = tl.sum(before, 0).to(tl.float64)
        slots_after = tl.sum(after, 0).to(tl.float64)
        singletons = tl.sum((before == 1).to(tl.int32), 0).to(tl.float64)
        distinct_before = tl.sum((before > 0).to(tl.int32), 0).to(tl.float64)
        distinct_after = tl.sum((after > 0).to(tl.int32), 0).to(tl.float64)
        t = tl.arange(0, BLOCK_M)
        mass = tl.load(row_stats+t*2, t < M, other=0).to(tl.float64)
        changed = tl.load(row_stats+t*2+1, t < M, other=0).to(tl.float64)
        mass_sum = tl.sum(mass, 0)
        mass_max = tl.max(mass, 0)
        changed_sum = tl.sum(changed, 0)
        index = tl.arange(0, 16)
        value = tl.full((16,), 0, tl.float64)
        actual_m = tl.minimum(M,tl.maximum(0,tl.load(active_tokens))) if HAS_ACTIVE else M
        value = tl.where(index == 0, (actual_m>0) if HAS_ACTIVE else 1, value)
        value = tl.where(index == 1, actual_m, value)
        value = tl.where(index == 2, slots_before, value)
        value = tl.where(index == 3, slots_after, value)
        value = tl.where(index == 4, slots_before-slots_after, value)
        value = tl.where(index == 5, singletons, value)
        value = tl.where(index == 6, distinct_before, value)
        value = tl.where(index == 7, distinct_after, value)
        value = tl.where(index == 8, mass_sum, value)
        value = tl.where(index == 9, distinct_before*EXPERT_BYTES, value)
        value = tl.where(index == 10, distinct_after*EXPERT_BYTES, value)
        value = tl.where(index == 11, changed_sum, value)
        previous = tl.load(counter+index, index < 13, other=0)
        value = tl.where(index == 12, tl.maximum(previous, mass_max), previous+value)
        tl.store(counter+index, value, index < 13)

    _TRITON_KERNELS = (triton, histogram_kernel, policy_kernel, stats_kernel, fused_small_kernel)
    return _TRITON_KERNELS


def _triton_policy(weights, ids, num_experts, config, renormalize,
                   weight_bytes_per_expert, counter, active_tokens=None, collect_stats=True):
    """One launch for B<=64, k<=8, E<=256; four launches otherwise.

    No dynamic output shape, item(), Python data predicate, or host sync.
    Kernel compilation and scratch allocation must be warmed before timing.
    Set backend='triton_multi' to force the four-launch reference path.
    This implementation is intended for decode; top-k >32 is rejected.
    """
    import torch
    if not weights.is_cuda or not ids.is_cuda:
        raise ValueError("Triton backend requires CUDA routing tensors")
    if not weights.is_contiguous() or not ids.is_contiguous():
        raise ValueError("Triton backend requires contiguous routing tensors")
    if ids.shape[1] > 32:
        raise ValueError("Triton policy supports top-k <=32")
    triton, histogram, policy, stats, fused_small = _get_triton_kernels()
    m, k = ids.shape
    out_weights, out_ids = torch.empty_like(weights), torch.empty_like(ids)
    mode = {"dense": 0, "global": 1, "gate": 2, "marginal": 3, "rank_gate": 4}[config.mode]
    rank_end = config.rank_end if config.rank_end is not None else k
    active_pointer=active_tokens if active_tokens is not None else ids
    if config.backend == "triton" and m <= 64 and k <= 8 and num_experts <= 256:
        fused_small[(1,)](
            weights, ids, out_weights, out_ids, counter, active_pointer, m, k, num_experts,
            mode, config.drop_j, config.tail_rank, config.gate_threshold, rank_end, config.gate_min,
            config.mass_budget, bool(renormalize), weight_bytes_per_expert,
            BLOCK_M=triton.next_power_of_2(max(m, 1)), BLOCK_K=triton.next_power_of_2(k),
            BLOCK_E=triton.next_power_of_2(num_experts), num_warps=4,
            HAS_ACTIVE=active_tokens is not None, COLLECT_STATS=collect_stats,
            REQUIRE_SINGLETON=config.require_singleton,
            enable_fp_fusion=False)
        return out_weights, out_ids
    counts = torch.zeros((2, num_experts+1), dtype=torch.int32, device=ids.device)
    row_stats = torch.empty((m, 2), dtype=torch.float32, device=ids.device)
    if m:
        histogram[(triton.cdiv(m*k, 256),)](
            ids, counts, active_pointer, m*k, num_experts, k, active_tokens is not None, BLOCK=256)
        policy[(m,)](
            weights, ids, counts, out_weights, out_ids, row_stats, active_pointer,
            k, num_experts, mode,
            config.drop_j, config.tail_rank, config.gate_threshold, config.mass_budget,
            rank_end, config.gate_min,
            bool(renormalize), BLOCK_K=triton.next_power_of_2(k), num_warps=4,
            HAS_ACTIVE=active_tokens is not None, COLLECT_STATS=collect_stats,
            REQUIRE_SINGLETON=config.require_singleton,
            enable_fp_fusion=False)
    if collect_stats:
        stats[(1,)](counts, row_stats, counter, active_pointer, num_experts, m, weight_bytes_per_expert,
                    BLOCK_E=triton.next_power_of_2(num_experts),
                    BLOCK_M=triton.next_power_of_2(max(m, 1)), num_warps=4,
                    HAS_ACTIVE=active_tokens is not None)
    return out_weights, out_ids
