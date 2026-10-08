"""Small, route-visible planner for legal speculative input-row suffix cuts.

This module does not infer row identity, future routes, or acceptance from receipt
chunk sizes. The runtime supplies those inputs and owns sampler/KV rollback.
Input row 0 of each request is mandatory; keeping k drafts retains rows 0..k.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

_bit_count = getattr(int, "bit_count", lambda value: bin(value).count("1"))


def plan_suffix(
    *,
    row_experts: Sequence[Iterable[int]],
    resident_experts: Iterable[int],
    row_starts: Mapping[str, int],
    original_draft_counts: Mapping[str, int],
    current_keep: Mapping[str, int] | None = None,
    expert_bytes: int | Mapping[int, int],
    bandwidth_bytes_per_s: float | None = None,
    acceptance_survival: Mapping[str, Sequence[float]] | Sequence[float] | None = None,
    future_token_price_s: float | None = None,
    overhead_s: float = 0.0,
    mode: str = "conservative",
    remaining_layer_scale: float = 1.0,
    fixed_keep: int | Mapping[str, int] | None = None,
    mandatory_rows: Iterable[int] | None = None,
    include_joint_candidates: bool = True,
) -> dict[str, Any]:
    """Return ``keep_drafts`` and an auditable cost ``receipt``.

    acceptance_survival[r][j-1] is the probability that the first j drafts
    match, estimated from completed, genuinely verified prior proposals.
    A common probability sequence may be supplied for all requests.

    Conservative mode credits only this layer's known cold-expert bytes.
    Horizon multiplies that same local saving by an explicit forecast scale;
    it never reports the scaled amount as actual bytes saved. Neither model
    credits already executed attention/router work or unknown future routes.

    ``bandwidth_bytes_per_s`` must model *exposed* transfer time. Raw DMA
    bandwidth is valid only when transfer is serialized/on the critical path.
    Missing calibration produces a no-cut decision. Full/off bypass policy;
    fixed is a correctness mode and can cut even with no predicted net gain.

    The byte formula assumes each entry-miss expert is loaded once. Runtime
    must validate that assumption or compare actual load-event bytes instead.
    """
    if mode not in {"off", "full", "fixed", "conservative", "horizon"}:
        raise ValueError(f"unknown suffix mode: {mode}")
    if set(row_starts) != set(original_draft_counts):
        raise ValueError("row_starts and original_draft_counts must have identical keys")
    keys = list(row_starts)
    if current_keep is not None and set(current_keep) != set(keys):
        raise ValueError("current_keep must describe every mapped request")
    rows = [frozenset(int(e) for e in experts) for experts in row_experts]
    nrows = len(rows)
    resident = frozenset(int(e) for e in resident_experts)
    expert_ids = sorted(set().union(*rows))
    expert_bits = {e: 1 << i for i, e in enumerate(expert_ids)}
    row_masks = [sum(expert_bits[e] for e in row) for row in rows]
    resident_mask = sum(expert_bits[e] for e in resident if e in expert_bits)
    starts: dict[str, int] = {}
    original: dict[str, int] = {}
    keep: dict[str, int] = {}
    covered: set[int] = set()
    prefix_masks: dict[str, list[int]] = {}
    for r in keys:
        start, count = int(row_starts[r]), int(original_draft_counts[r])
        k = count if current_keep is None else int(current_keep[r])
        if start < 0 or count < 0 or not 0 <= k <= count or start + count >= nrows:
            raise ValueError(f"invalid row range or current keep for request {r}")
        span = set(range(start, start + count + 1))
        if covered.intersection(span):
            raise ValueError("request speculative input ranges overlap")
        covered.update(span)
        starts[r], original[r], keep[r] = start, count, k
        union = 0
        prefixes = []
        for j in range(k + 1):
            union |= row_masks[start + j]
            prefixes.append(union)
        prefix_masks[r] = prefixes

    mandatory = set(range(nrows)).difference(covered)
    if mandatory_rows is not None:
        explicit = set(int(i) for i in mandatory_rows)
        if any(i < 0 or i >= nrows for i in explicit):
            raise ValueError("mandatory row index outside row_experts")
        # Extra mandatory speculative rows would invalidate k+1 sampling.
        if any(i in covered and i not in starts.values() for i in explicit):
            raise ValueError("speculative suffix rows cannot also be mandatory")
        mandatory.update(explicit)
    mandatory_union = frozenset(e for i in mandatory for e in rows[i])
    # Every request's row 0 must survive every action.
    always_union = mandatory_union.union(*(rows[starts[r]] for r in keys))
    mandatory_mask = sum(expert_bits[e] for e in mandatory_union)

    def cold_mask(candidate: Mapping[str, int]) -> int:
        union = mandatory_mask
        for r in keys:
            union |= prefix_masks[r][candidate[r]]
        return union & ~resident_mask

    def mask_experts(mask: int) -> list[int]:
        result = []
        while mask:
            bit = mask & -mask
            result.append(expert_ids[bit.bit_length() - 1])
            mask ^= bit
        return result

    base_mask = cold_mask(keep)
    base_cold = frozenset(mask_experts(base_mask))
    mandatory_covers_all_cold = base_cold.issubset(always_union)

    if isinstance(expert_bytes, Mapping):
        byte_sizes = {e: int(expert_bytes[e]) for e in base_cold}
        if any(size < 0 for size in byte_sizes.values()):
            raise ValueError("expert byte sizes must be nonnegative")

        def mask_bytes(mask: int) -> int:
            return sum(byte_sizes[e] for e in mask_experts(mask))
    else:
        uniform_bytes = int(expert_bytes)
        if uniform_bytes < 0:
            raise ValueError("expert byte sizes must be nonnegative")

        def mask_bytes(mask: int) -> int:
            return _bit_count(mask) * uniform_bytes

    base_bytes = mask_bytes(base_mask)
    if not math.isfinite(overhead_s) or overhead_s < 0:
        raise ValueError("overhead_s must be finite and nonnegative")
    if not math.isfinite(remaining_layer_scale) or remaining_layer_scale < 1:
        raise ValueError("remaining_layer_scale must be finite and at least one")
    scale = float(remaining_layer_scale) if mode == "horizon" else 1.0
    calibrated = (
        bandwidth_bytes_per_s is not None
        and math.isfinite(bandwidth_bytes_per_s)
        and bandwidth_bytes_per_s > 0
        and future_token_price_s is not None
        and math.isfinite(future_token_price_s)
        and future_token_price_s >= 0
    )
    survival: dict[str, tuple[float, ...]] = {}
    if acceptance_survival is None:
        calibrated = False
    else:
        for r in keys:
            values = (acceptance_survival.get(r) if isinstance(acceptance_survival, Mapping)
                      else acceptance_survival)
            if values is None or len(values) < keep[r]:
                calibrated = False
                continue
            p = tuple(float(x) for x in values[:keep[r]])
            if (any(not math.isfinite(x) or x < 0 or x > 1 for x in p)
                    or any(p[j] > p[j - 1] + 1e-12 for j in range(1, len(p)))):
                raise ValueError("acceptance survival must be finite, in [0,1], and nonincreasing")
            survival[r] = p
    lost_by_cut: dict[str, list[float]] = {}
    if calibrated:
        for r in keys:
            losses = [0.0] * (keep[r] + 1)
            for k in range(keep[r] - 1, -1, -1):
                losses[k] = losses[k + 1] + survival[r][k]
            lost_by_cut[r] = losses

    candidates: dict[tuple[int, ...], str] = {}

    def add(candidate: Mapping[str, int], label: str) -> None:
        candidates.setdefault(tuple(candidate[r] for r in keys), label)

    add(keep, "no_cut")
    if mode == "fixed":
        if fixed_keep is None:
            raise ValueError("fixed mode requires fixed_keep")
        chosen = dict(keep)
        for r in keys:
            target = int(fixed_keep.get(r, keep[r]) if isinstance(fixed_keep, Mapping)
                         else fixed_keep)
            if target < 0:
                raise ValueError("fixed_keep must be nonnegative")
            chosen[r] = min(keep[r], target)
        add(chosen, "fixed")
    elif mode not in {"off", "full"} and calibrated and not mandatory_covers_all_cold:
        # Keep the strong short-draft alternatives in the action set. Several
        # requests can jointly eliminate a cold expert only after all shorten;
        # one-request moves alone would miss that shared-set saving.
        for k in range(max(keep.values(), default=0)):
            add({r: min(keep[r], k) for r in keys}, f"uniform_prefix:{k}")
        for r in keys:
            for k in range(keep[r]):
                candidate = dict(keep)
                candidate[r] = k
                add(candidate, f"single:{r}:{k}")
        if include_joint_candidates:
            # Eliminating one cold expert may require cuts in multiple requests.
            # This produces legal joint witnesses, not arbitrary row subsets.
            for e in sorted(base_cold.difference(always_union)):
                candidate = dict(keep)
                for r in keys:
                    for j in range(1, keep[r] + 1):
                        if e in rows[starts[r] + j]:
                            candidate[r] = j - 1
                            break
                add(candidate, f"joint_expert:{e}")

    def evaluate(candidate: Mapping[str, int], label: str) -> dict[str, Any]:
        cold = cold_mask(candidate)
        saved = base_bytes - mask_bytes(cold)
        changed = any(candidate[r] != keep[r] for r in keys)
        lost = (sum(lost_by_cut[r][candidate[r]] for r in keys)
                if calibrated else None)
        transfer = saved / float(bandwidth_bytes_per_s) if calibrated else None
        predicted = transfer * scale if transfer is not None else None
        recovery = lost * float(future_token_price_s) if calibrated else None
        charge = overhead_s if changed else 0.0
        score = predicted - recovery - charge if calibrated else None
        return {
            "candidate": label,
            "entry_cold_experts_before": len(base_cold),
            "entry_cold_experts_after": _bit_count(cold),
            "entry_cold_bytes_before": base_bytes,
            "entry_cold_bytes_after": base_bytes - saved,
            "current_layer_saved_bytes": saved,
            "eliminated_cold_experts": mask_experts(base_mask & ~cold),
            "expected_lost_committed_tokens": lost,
            "estimated_current_layer_saved_transfer_s": transfer,
            "remaining_layer_prediction_scale": scale,
            "predicted_remaining_saved_s": predicted,
            "predicted_future_service_charge_s": recovery,
            "action_overhead_charge_s": charge,
            "predicted_net_gain_s": score,
            "removed_input_rows": sum(keep[r] - candidate[r] for r in keys),
        }

    selected = dict(keep)
    selected_receipt = evaluate(selected, "no_cut")
    if mode == "fixed":
        selected = chosen
        selected_receipt = evaluate(selected, "fixed")
        reason = "fixed_functional_check"
    elif mode in {"off", "full"}:
        reason = "policy_disabled"
    elif not calibrated:
        reason = "missing_calibration"
    else:
        best_score = 0.0
        selected_label = "no_cut"
        for values, label in candidates.items():
            candidate = dict(zip(keys, values))
            saved = base_bytes - mask_bytes(cold_mask(candidate))
            # With transfer-only credit, zero byte saving cannot win. Avoid
            # building detailed receipts or summing losses for these cuts.
            if saved <= 0:
                continue
            lost = sum(lost_by_cut[r][candidate[r]] for r in keys)
            score = (saved / float(bandwidth_bytes_per_s) * scale
                     - lost * float(future_token_price_s) - overhead_s)
            if score > best_score + 1e-15:
                selected, selected_label, best_score = candidate, label, score
        selected_receipt = evaluate(selected, selected_label)
        reason = ("positive_predicted_net_gain" if best_score > 0
                  else "mandatory_rows_cover_all_cold_experts" if mandatory_covers_all_cold
                  else "no_positive_candidate")
    selected_receipt.update({
        "mode": mode,
        "reason": reason,
        "calibrated": calibrated,
        "candidate_count": len(candidates),
        "original_draft_counts": original,
        "previous_keep_drafts": keep,
        "keep_drafts": selected,
        "bandwidth_bytes_per_s": bandwidth_bytes_per_s,
        "future_token_price_s": future_token_price_s,
        "byte_model": "each_entry_miss_loaded_once",
        "actual_runtime_saved_bytes": None,
        "decision_uses_future_routes": False,
    })
    return {"keep_drafts": selected, "receipt": selected_receipt}
