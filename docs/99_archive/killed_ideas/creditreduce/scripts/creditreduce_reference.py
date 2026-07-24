"""Pure numerical reference for CreditReduce combine experiments.

This module deliberately models arithmetic and *logical* record payload only.
It does not model packing, alignment, transport headers, collectives, kernels,
overlap, latency, or actual wire bytes.

The input contract is the backend-visible one used by the P0 experiment:

* expert outputs and routing weights are BF16;
* their product is rounded to one BF16 contribution before reduction;
* contributions are visited in stable global-expert-id order;
* source-domain subtotals and receiver accumulation use FP32;
* the receiver visits source domains in increasing domain-id order and casts
  to the model dtype once at the end.

Home-domain groups never enter the logical wire accounting.  They retain their
FP32 subtotal for every early endpoint.
"""

from __future__ import annotations

# --- shared-lib bootstrap (auto) ---
import sys
from pathlib import Path as _Path

def _ensure_shared_on_path() -> None:
    here = _Path(__file__).resolve().parent
    for p in [here, *here.parents]:
        cand = p / "experiments" / "shared"
        if (cand / "capture_moe.py").exists():
            s = str(cand)
            if s not in sys.path:
                sys.path.insert(0, s)
            return
        if (p / "capture_moe.py").exists():
            s = str(p)
            if s not in sys.path:
                sys.path.insert(0, s)
            return

_ensure_shared_on_path()
del _ensure_shared_on_path, _Path
# --- end bootstrap ---

from dataclasses import dataclass
import math
from typing import Mapping

import torch

try:  # Direct script execution from experiments/idea_a_mac.
    from fake_quant import apply_precision
except ModuleNotFoundError:  # Namespace-package import from the repository root.
    from .fake_quant import apply_precision


ENDPOINTS = (
    "late_bf16",
    "stock_early_bf16",
    "clean_early_bf16",
    "uniform_early_fp32",
    "pd_full",
    "uniform_early_fp8",
    "pd_gated",
)


@dataclass(frozen=True)
class EndpointDiagnostics:
    """Per-token representation diagnostics for one numerical endpoint.

    ``logical_payload_bytes`` counts hidden-vector elements only.  It excludes
    bitmap, scale, header, alignment, and framing bytes.  ``scale_bytes`` makes
    the FP8 helper's one FP32 scale per remote vector explicit, but it is still
    not an actual-wire estimate.
    """

    n32: torch.Tensor  # [tokens], number of remote FP32 group records
    fp32_group_mask: torch.Tensor  # [tokens, source_domains]
    logical_payload_bytes: torch.Tensor  # [tokens], hidden-vector payload only
    minimal_bitmap_bytes: torch.Tensor  # [tokens], dynamic pd_gated only
    scale_bytes: torch.Tensor  # [tokens], uniform FP8's FP32 per-vector scale
    payload_cap_holds: torch.Tensor  # [tokens], relative to flat remote BF16

    @property
    def logical_total_bytes(self) -> torch.Tensor:
        """Payload plus explicitly represented bitmap/scale metadata."""

        return (
            self.logical_payload_bytes
            + self.minimal_bitmap_bytes
            + self.scale_bytes
        )


@dataclass(frozen=True)
class CreditReduceDiagnostics:
    """All-layer-ready per-token/group diagnostics.

    The tensors keep the token dimension instead of reporting only global
    averages, so a caller can aggregate them by layer, phase, request, or
    placement without rerunning this reference.
    """

    home_domains: torch.Tensor  # [tokens]
    group_multiplicities: torch.Tensor  # [tokens, source_domains]
    k_remote: torch.Tensor  # [tokens], routed contributions outside home domain
    d_remote: torch.Tensor  # [tokens], nonempty remote source domains
    c_remote: torch.Tensor  # [tokens], remote groups with multiplicity >= 2
    d_total: torch.Tensor  # [tokens], all nonempty domains, including home
    eligible: torch.Tensor  # [tokens], C_remote >= 1 and D_total >= 2
    recast_residuals: torch.Tensor  # [tokens, source_domains, hidden], p16 - s32
    residual_rms: torch.Tensor  # [tokens, source_domains], raw RMS (no normalization)
    endpoints: Mapping[str, EndpointDiagnostics]

    def aggregate(self) -> dict[str, int | float | bool]:
        """Return a flat, JSON-ready aggregate for one layer invocation."""

        tokens = int(self.k_remote.numel())
        collided_remote = (
            (self.group_multiplicities >= 2)
            & (
                torch.arange(
                    self.group_multiplicities.shape[1],
                    device=self.group_multiplicities.device,
                ).unsqueeze(0)
                != self.home_domains.unsqueeze(1)
            )
        )
        collided_rms = self.residual_rms[collided_remote]
        row: dict[str, int | float | bool] = {
            "tokens": tokens,
            "eligible_tokens": int(self.eligible.sum().item()),
            "eligibility_rate": (
                float(self.eligible.float().mean().item()) if tokens else 0.0
            ),
            "k_remote": int(self.k_remote.sum().item()),
            "d_remote": int(self.d_remote.sum().item()),
            "c_remote": int(self.c_remote.sum().item()),
            "d_total": int(self.d_total.sum().item()),
            "eligible_k_remote": int(self.k_remote[self.eligible].sum().item()),
            "eligible_credit_units": int(
                (self.k_remote - self.d_remote)[self.eligible].sum().item()
            ),
            "remote_credit_units": int(
                (self.k_remote - self.d_remote).sum().item()
            ),
            "remote_collided_residual_rms_mean": (
                float(collided_rms.float().mean().item())
                if collided_rms.numel()
                else 0.0
            ),
        }
        for name, endpoint in self.endpoints.items():
            row[f"{name}_n32"] = int(endpoint.n32.sum().item())
            row[f"{name}_logical_payload_bytes"] = int(
                endpoint.logical_payload_bytes.sum().item()
            )
            row[f"{name}_minimal_bitmap_bytes"] = int(
                endpoint.minimal_bitmap_bytes.sum().item()
            )
            row[f"{name}_scale_bytes"] = int(endpoint.scale_bytes.sum().item())
            row[f"{name}_payload_cap_all"] = bool(
                endpoint.payload_cap_holds.all().item()
            )
        return row

    def token_rows(self) -> list[dict[str, int | bool]]:
        """Return one flat, JSON-ready diagnostics row per token."""

        rows: list[dict[str, int | bool]] = []
        for token in range(self.k_remote.numel()):
            row: dict[str, int | bool] = {
                "token_index": token,
                "home_domain": int(self.home_domains[token].item()),
                "k_remote": int(self.k_remote[token].item()),
                "d_remote": int(self.d_remote[token].item()),
                "c_remote": int(self.c_remote[token].item()),
                "d_total": int(self.d_total[token].item()),
                "eligible": bool(self.eligible[token].item()),
            }
            for name, endpoint in self.endpoints.items():
                row[f"{name}_n32"] = int(endpoint.n32[token].item())
                row[f"{name}_logical_payload_bytes"] = int(
                    endpoint.logical_payload_bytes[token].item()
                )
                row[f"{name}_minimal_bitmap_bytes"] = int(
                    endpoint.minimal_bitmap_bytes[token].item()
                )
                row[f"{name}_scale_bytes"] = int(endpoint.scale_bytes[token].item())
                row[f"{name}_payload_cap_holds"] = bool(
                    endpoint.payload_cap_holds[token].item()
                )
            rows.append(row)
        return rows

    def group_rows(self) -> list[dict[str, int | float | bool]]:
        """Return one JSON-ready row per nonempty token/source-domain group."""

        rows: list[dict[str, int | float | bool]] = []
        pd_full_mask = self.endpoints["pd_full"].fp32_group_mask
        pd_gated_mask = self.endpoints["pd_gated"].fp32_group_mask
        for token in range(self.group_multiplicities.shape[0]):
            home_domain = int(self.home_domains[token].item())
            for domain in range(self.group_multiplicities.shape[1]):
                multiplicity = int(self.group_multiplicities[token, domain].item())
                if multiplicity == 0:
                    continue
                is_home = domain == home_domain
                rows.append(
                    {
                        "token_index": token,
                        "source_domain": domain,
                        "home": is_home,
                        "remote": not is_home,
                        "multiplicity": multiplicity,
                        "collided_remote": (not is_home and multiplicity >= 2),
                        "residual_rms": float(
                            self.residual_rms[token, domain].float().item()
                        ),
                        "pd_full_fp32": bool(pd_full_mask[token, domain].item()),
                        "pd_gated_fp32": bool(pd_gated_mask[token, domain].item()),
                    }
                )
        return rows

    def recorder_records(self) -> dict[str, object]:
        """Return the three structures expected by experiment recorders."""

        return {
            "aggregate": self.aggregate(),
            "token_rows": self.token_rows(),
            "group_rows": self.group_rows(),
        }


@dataclass(frozen=True)
class CreditReduceResult:
    """Outputs plus arithmetic state needed for causal numerical audits."""

    outputs: Mapping[str, torch.Tensor]
    canonical_expert_ids: torch.Tensor  # [tokens, top_k]
    canonical_source_ranks: torch.Tensor  # [tokens, top_k]
    canonical_source_domains: torch.Tensor  # [tokens, top_k]
    canonical_weighted_contributions: torch.Tensor  # [tokens, top_k, hidden], BF16
    late_accumulator_fp32: torch.Tensor  # [tokens, hidden]
    domain_subtotals_fp32: torch.Tensor  # [tokens, source_domains, hidden]
    diagnostics: CreditReduceDiagnostics


def build_expert_to_rank(
    num_experts: int,
    ep_size: int,
    placement: str,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Build a deterministic global-expert -> EP-rank placement table."""

    if num_experts < 1:
        raise ValueError("num_experts must be positive")
    if ep_size < 1:
        raise ValueError("ep_size must be positive")

    expert_ids = torch.arange(num_experts, dtype=torch.int64, device=device)
    if placement == "contiguous":
        # Floor partitioning also gives a defined result when the sizes are not
        # divisible.  Empty ranks are allowed in this pure placement reference.
        return torch.div(
            expert_ids * ep_size, num_experts, rounding_mode="floor"
        ).clamp_max(ep_size - 1)
    if placement in ("round_robin", "mod"):
        return expert_ids.remainder(ep_size)
    raise ValueError(f"unknown expert placement: {placement}")


def expert_rank_ids(
    expert_ids: torch.Tensor,
    num_experts: int,
    ep_size: int,
    placement: str,
) -> torch.Tensor:
    """Map selected global expert ids to their explicit EP ranks."""

    if expert_ids.numel() and (
        int(expert_ids.min().item()) < 0
        or int(expert_ids.max().item()) >= num_experts
    ):
        raise ValueError("expert id outside [0, num_experts)")
    table = build_expert_to_rank(
        num_experts, ep_size, placement, device=expert_ids.device
    )
    return table[expert_ids.long()]


def rank_domain_ids(ranks: torch.Tensor, ep_size: int, ranks_per_domain: int) -> torch.Tensor:
    """Map EP ranks to contiguous source-domain ids."""

    if ep_size < 1:
        raise ValueError("ep_size must be positive")
    if ranks_per_domain < 1:
        raise ValueError("ranks_per_domain must be positive")
    if ranks.numel() and (
        int(ranks.min().item()) < 0 or int(ranks.max().item()) >= ep_size
    ):
        raise ValueError("EP rank outside [0, ep_size)")
    return torch.div(ranks.long(), ranks_per_domain, rounding_mode="floor")


def _validate_inputs(
    raw_outputs: torch.Tensor,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    num_experts: int,
    ep_size: int,
    ranks_per_domain: int,
) -> None:
    if raw_outputs.ndim != 3:
        raise ValueError("raw_outputs must have shape [tokens, top_k, hidden]")
    if raw_outputs.shape[1] < 1 or raw_outputs.shape[2] < 1:
        raise ValueError("top_k and hidden dimensions must be positive")
    if routing_weights.shape != raw_outputs.shape[:2]:
        raise ValueError("routing_weights shape does not match raw_outputs")
    if selected_experts.shape != raw_outputs.shape[:2]:
        raise ValueError("selected_experts shape does not match raw_outputs")
    if raw_outputs.dtype != torch.bfloat16:
        raise TypeError("raw_outputs must be torch.bfloat16")
    if routing_weights.dtype != torch.bfloat16:
        raise TypeError("routing_weights must be torch.bfloat16")
    if selected_experts.dtype == torch.bool or selected_experts.dtype.is_floating_point:
        raise TypeError("selected_experts must use an integer dtype")
    if not (
        raw_outputs.device == routing_weights.device == selected_experts.device
    ):
        raise ValueError("all route tensors must be on the same device")
    if num_experts < 1:
        raise ValueError("num_experts must be positive")
    if ep_size < 1:
        raise ValueError("ep_size must be positive")
    if ranks_per_domain < 1:
        raise ValueError("ranks_per_domain must be positive")


def _resolve_home_domains(
    token_count: int,
    device: torch.device,
    ep_size: int,
    ranks_per_domain: int,
    home_ranks: torch.Tensor | None,
    home_domains: torch.Tensor | None,
) -> torch.Tensor:
    if home_ranks is None and home_domains is None:
        raise ValueError("provide home_ranks, home_domains, or both")

    from_ranks: torch.Tensor | None = None
    if home_ranks is not None:
        if home_ranks.shape != (token_count,):
            raise ValueError("home_ranks must have shape [tokens]")
        if home_ranks.dtype == torch.bool or home_ranks.dtype.is_floating_point:
            raise TypeError("home_ranks must use an integer dtype")
        if home_ranks.device != device:
            raise ValueError("home_ranks must be on the route tensor device")
        from_ranks = rank_domain_ids(home_ranks, ep_size, ranks_per_domain)

    resolved: torch.Tensor
    if home_domains is not None:
        if home_domains.shape != (token_count,):
            raise ValueError("home_domains must have shape [tokens]")
        if home_domains.dtype == torch.bool or home_domains.dtype.is_floating_point:
            raise TypeError("home_domains must use an integer dtype")
        if home_domains.device != device:
            raise ValueError("home_domains must be on the route tensor device")
        num_domains = math.ceil(ep_size / ranks_per_domain)
        if home_domains.numel() and (
            int(home_domains.min().item()) < 0
            or int(home_domains.max().item()) >= num_domains
        ):
            raise ValueError("home domain outside the EP topology")
        resolved = home_domains.long()
        if from_ranks is not None and not torch.equal(resolved, from_ranks):
            raise ValueError("home_ranks and home_domains disagree")
    else:
        assert from_ranks is not None
        resolved = from_ranks
    return resolved


def _receiver_combine(
    domain_vectors: torch.Tensor, present: torch.Tensor, output_dtype: torch.dtype
) -> torch.Tensor:
    """FP32 receiver accumulation in canonical increasing domain-id order."""

    accumulator = torch.zeros(
        domain_vectors.shape[-1], dtype=torch.float32, device=domain_vectors.device
    )
    for domain_id in range(domain_vectors.shape[0]):
        if bool(present[domain_id].item()):
            accumulator.add_(domain_vectors[domain_id].float())
    return accumulator.to(output_dtype)


def _endpoint_diagnostics(
    *,
    n32: torch.Tensor,
    fp32_group_mask: torch.Tensor,
    payload: torch.Tensor,
    bitmap: torch.Tensor,
    scale: torch.Tensor,
    late_payload: torch.Tensor,
) -> EndpointDiagnostics:
    return EndpointDiagnostics(
        n32=n32,
        fp32_group_mask=fp32_group_mask,
        logical_payload_bytes=payload,
        minimal_bitmap_bytes=bitmap,
        scale_bytes=scale,
        payload_cap_holds=payload <= late_payload,
    )


def creditreduce_reference(
    raw_outputs: torch.Tensor,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    *,
    num_experts: int,
    ep_size: int,
    ranks_per_domain: int,
    placement: str,
    home_ranks: torch.Tensor | None = None,
    home_domains: torch.Tensor | None = None,
    residual_rms_threshold: float = math.inf,
) -> CreditReduceResult:
    """Evaluate the locked-route CreditReduce numerical endpoints.

    ``residual_rms_threshold`` is one already-frozen per-layer raw RMS
    threshold.  A collided remote group uses FP32 iff its raw residual RMS is
    strictly greater than the threshold.  This reference intentionally does
    not learn, normalize, or tune the threshold.

    Dynamic CreditReduce is eligible exactly when ``C_remote >= 1`` and at
    least two source domains are nonempty.  Ineligible tokens use the late
    reference unchanged; this makes top-1/top-2 and no-op topology cells
    explicit instead of silently treating them as positive opportunities.
    """

    _validate_inputs(
        raw_outputs,
        routing_weights,
        selected_experts,
        num_experts,
        ep_size,
        ranks_per_domain,
    )
    token_count, top_k, hidden = raw_outputs.shape
    device = raw_outputs.device
    num_domains = math.ceil(ep_size / ranks_per_domain)
    resolved_home = _resolve_home_domains(
        token_count,
        device,
        ep_size,
        ranks_per_domain,
        home_ranks,
        home_domains,
    )

    # Both operands are BF16.  Make the backend-visible rounding boundary
    # explicit before any FP32 subtotal is formed.
    weighted_native = (raw_outputs * routing_weights[..., None]).to(torch.bfloat16)
    canonical_order = torch.argsort(selected_experts.long(), dim=1, stable=True)
    canonical_ids = torch.gather(selected_experts.long(), 1, canonical_order)
    gather_vectors = canonical_order[..., None].expand(-1, -1, hidden)
    weighted = torch.gather(weighted_native, 1, gather_vectors)

    source_ranks_native = expert_rank_ids(
        selected_experts, num_experts, ep_size, placement
    )
    source_ranks = torch.gather(source_ranks_native, 1, canonical_order)
    source_domains = rank_domain_ids(source_ranks, ep_size, ranks_per_domain)

    multiplicities = torch.zeros(
        (token_count, num_domains), dtype=torch.int64, device=device
    )
    subtotals = torch.zeros(
        (token_count, num_domains, hidden), dtype=torch.float32, device=device
    )
    stock_subtotals = torch.zeros(
        (token_count, num_domains, hidden), dtype=torch.bfloat16, device=device
    )
    late_accumulator = torch.zeros(
        (token_count, hidden), dtype=torch.float32, device=device
    )

    # Python loops are intentional: this is the canonical arithmetic oracle,
    # not a performance implementation.  They make every addition boundary
    # and order auditable.
    for token in range(token_count):
        for route_index in range(top_k):
            contribution = weighted[token, route_index]
            domain_id = int(source_domains[token, route_index].item())
            late_accumulator[token].add_(contribution.float())
            subtotals[token, domain_id].add_(contribution.float())
            stock_subtotals[token, domain_id].add_(contribution)
            multiplicities[token, domain_id] += 1

    present = multiplicities > 0
    clean_bf16 = subtotals.to(torch.bfloat16)
    clean_promoted = clean_bf16.float()
    residuals = clean_promoted - subtotals
    residual_rms = torch.sqrt(torch.mean(residuals.square(), dim=-1))

    domain_ids = torch.arange(num_domains, device=device).unsqueeze(0)
    remote = present & (domain_ids != resolved_home.unsqueeze(1))
    collided_remote = remote & (multiplicities >= 2)
    k_remote = (multiplicities * remote.to(torch.int64)).sum(dim=1)
    d_remote = remote.sum(dim=1, dtype=torch.int64)
    c_remote = collided_remote.sum(dim=1, dtype=torch.int64)
    d_total = present.sum(dim=1, dtype=torch.int64)
    eligible = (c_remote >= 1) & (d_total >= 2)

    threshold = torch.as_tensor(
        residual_rms_threshold, dtype=torch.float32, device=device
    )
    if threshold.numel() != 1:
        raise ValueError("residual_rms_threshold must be a scalar")
    gated_fp32 = (
        collided_remote
        & eligible.unsqueeze(1)
        & (residual_rms > threshold.reshape(()))
    )
    pd_full_fp32 = collided_remote

    outputs = {
        name: torch.empty(
            (token_count, hidden), dtype=raw_outputs.dtype, device=device
        )
        for name in ENDPOINTS
    }
    outputs["late_bf16"].copy_(late_accumulator.to(raw_outputs.dtype))

    for token in range(token_count):
        home_domain = int(resolved_home[token].item())
        endpoint_domains = {
            name: torch.zeros(
                (num_domains, hidden), dtype=torch.float32, device=device
            )
            for name in ENDPOINTS
            if name != "late_bf16"
        }

        for domain_id in range(num_domains):
            if not bool(present[token, domain_id].item()):
                continue
            subtotal = subtotals[token, domain_id]
            is_home = domain_id == home_domain
            if is_home:
                # This group is already at the receiver and crosses no
                # representation boundary in any early endpoint.
                for values in endpoint_domains.values():
                    values[domain_id].copy_(subtotal)
                continue

            p16 = clean_promoted[token, domain_id]
            endpoint_domains["stock_early_bf16"][domain_id].copy_(
                stock_subtotals[token, domain_id].float()
            )
            endpoint_domains["clean_early_bf16"][domain_id].copy_(p16)
            endpoint_domains["uniform_early_fp32"][domain_id].copy_(subtotal)
            endpoint_domains["pd_full"][domain_id].copy_(
                subtotal if bool(pd_full_fp32[token, domain_id].item()) else p16
            )
            fp8 = apply_precision(subtotal.unsqueeze(0), "fp8").squeeze(0)
            endpoint_domains["uniform_early_fp8"][domain_id].copy_(fp8.float())
            endpoint_domains["pd_gated"][domain_id].copy_(
                subtotal if bool(gated_fp32[token, domain_id].item()) else p16
            )

        for name, values in endpoint_domains.items():
            outputs[name][token].copy_(
                _receiver_combine(values, present[token], raw_outputs.dtype)
            )

        if not bool(eligible[token].item()):
            # No CreditReduce opportunity: preserve the native late arithmetic,
            # including its canonical expert order rather than domain regrouping.
            outputs["pd_gated"][token].copy_(outputs["late_bf16"][token])

    zeros = torch.zeros(token_count, dtype=torch.int64, device=device)
    false_groups = torch.zeros_like(remote)
    late_payload = 2 * hidden * k_remote
    early16_payload = 2 * hidden * d_remote
    uniform32_payload = 4 * hidden * d_remote
    pd_full_n32 = pd_full_fp32.sum(dim=1, dtype=torch.int64)
    pd_full_payload = 2 * hidden * (d_remote + pd_full_n32)
    gated_n32 = gated_fp32.sum(dim=1, dtype=torch.int64)
    # Ineligible dynamic tokens run the late no-op path rather than claiming a
    # grouped CreditReduce payload.
    gated_credit_payload = 2 * hidden * (d_remote + gated_n32)
    gated_payload = torch.where(eligible, gated_credit_payload, late_payload)
    gated_bitmap = torch.where(
        eligible, torch.div(c_remote + 7, 8, rounding_mode="floor"), zeros
    )
    fp8_payload = hidden * d_remote
    fp8_scale = 4 * d_remote  # fake_quant uses one FP32 scale per vector.

    # Per-token proof obligations for the two PrecisionDividend endpoints:
    # C_remote <= K_remote - D_remote and n32 <= C_remote.
    if not bool((c_remote <= k_remote - d_remote).all().item()):
        raise AssertionError("route multiplicities violated C_remote <= K_remote-D_remote")
    if not bool((pd_full_n32 <= c_remote).all().item()):
        raise AssertionError("PD-Full violated n32 <= C_remote")
    if not bool((gated_n32 <= c_remote).all().item()):
        raise AssertionError("PD-Gated violated n32 <= C_remote")
    if not bool((pd_full_payload <= late_payload).all().item()):
        raise AssertionError("PD-Full payload exceeded flat remote BF16")
    if not bool((gated_payload <= late_payload).all().item()):
        raise AssertionError("PD-Gated payload exceeded flat remote BF16")

    endpoint_diagnostics = {
        "late_bf16": _endpoint_diagnostics(
            n32=zeros,
            fp32_group_mask=false_groups,
            payload=late_payload,
            bitmap=zeros,
            scale=zeros,
            late_payload=late_payload,
        ),
        "stock_early_bf16": _endpoint_diagnostics(
            n32=zeros,
            fp32_group_mask=false_groups,
            payload=early16_payload,
            bitmap=zeros,
            scale=zeros,
            late_payload=late_payload,
        ),
        "clean_early_bf16": _endpoint_diagnostics(
            n32=zeros,
            fp32_group_mask=false_groups,
            payload=early16_payload,
            bitmap=zeros,
            scale=zeros,
            late_payload=late_payload,
        ),
        "uniform_early_fp32": _endpoint_diagnostics(
            n32=d_remote,
            fp32_group_mask=remote,
            payload=uniform32_payload,
            bitmap=zeros,
            scale=zeros,
            late_payload=late_payload,
        ),
        "pd_full": _endpoint_diagnostics(
            n32=pd_full_n32,
            fp32_group_mask=pd_full_fp32,
            payload=pd_full_payload,
            bitmap=zeros,
            scale=zeros,
            late_payload=late_payload,
        ),
        "uniform_early_fp8": _endpoint_diagnostics(
            n32=zeros,
            fp32_group_mask=false_groups,
            payload=fp8_payload,
            bitmap=zeros,
            scale=fp8_scale,
            late_payload=late_payload,
        ),
        "pd_gated": _endpoint_diagnostics(
            n32=gated_n32,
            fp32_group_mask=gated_fp32,
            payload=gated_payload,
            bitmap=gated_bitmap,
            scale=zeros,
            late_payload=late_payload,
        ),
    }

    return CreditReduceResult(
        outputs=outputs,
        canonical_expert_ids=canonical_ids,
        canonical_source_ranks=source_ranks,
        canonical_source_domains=source_domains,
        canonical_weighted_contributions=weighted,
        late_accumulator_fp32=late_accumulator,
        domain_subtotals_fp32=subtotals,
        diagnostics=CreditReduceDiagnostics(
            home_domains=resolved_home,
            group_multiplicities=multiplicities,
            k_remote=k_remote,
            d_remote=d_remote,
            c_remote=c_remote,
            d_total=d_total,
            eligible=eligible,
            recast_residuals=residuals,
            residual_rms=residual_rms,
            endpoints=endpoint_diagnostics,
        ),
    )
