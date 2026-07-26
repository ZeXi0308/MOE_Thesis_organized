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

from dataclasses import dataclass, field
import math
from typing import Iterable

import numpy as np
import torch

from fake_quant import apply_precision
from policies import BYTE_SIZES


REFERENCE_ARM = "bf16_reference"
RECEIVER_ARMS = (
    "uniform_full",
    "uniform_low",
    "calib_static",
    "causal_no_hysteresis",
    "controller",
)


@dataclass(frozen=True)
class ReceiverPolicyConfig:
    arm: str
    ep_size: int = 8
    gpus_per_node: int = 4
    placement: str = "contiguous"
    high_precision: str = "fp8"
    low_precision: str = "int4"
    alpha: float = 0.6
    threshold_high: float = float("inf")
    threshold_low: float = float("inf")
    dwell_min: int = 1
    static_profile: dict[tuple[int, int], float] = field(default_factory=dict)
    static_threshold: float = float("inf")
    high_scale_bytes_per_vector: float = 4.0
    low_scale_bytes_per_vector: float = 4.0
    inter_node_gbps: float = 200.0
    codec_pack_us: float = 0.0
    codec_unpack_us: float = 0.0
    codec_tile_rows: int = 32
    codec_tax_mode: str = "once_per_step"
    # When True, drop all low-lane actions unless analytic wire saving exceeds
    # measured codec tax (pack+unpack, optionally scaled by tiles).
    require_positive_net_saving: bool = False
    # Optional H2D staging tax (µs) folded into the hard-gate codec cost.
    codec_h2d_us: float = 0.0
    # Row count that codec_pack_us/unpack_us/h2d_us were measured at.
    # Used to scale serialized per-tile cost: per_tile = unit * (tile_rows / measured_rows).
    codec_measured_rows: int = 128

    def __post_init__(self) -> None:
        if self.arm not in (REFERENCE_ARM, *RECEIVER_ARMS):
            raise ValueError(f"unknown receiver arm: {self.arm}")
        if self.ep_size < 1:
            raise ValueError("ep_size must be positive")
        if self.gpus_per_node < 1 or self.ep_size % self.gpus_per_node != 0:
            raise ValueError("gpus_per_node must be positive and divide ep_size")
        if self.placement not in ("contiguous", "round_robin"):
            raise ValueError(f"unsupported placement: {self.placement}")
        if self.high_precision not in BYTE_SIZES or self.low_precision not in BYTE_SIZES:
            raise ValueError("unknown high/low precision")
        if BYTE_SIZES[self.low_precision] >= BYTE_SIZES[self.high_precision]:
            raise ValueError("low_precision must use fewer bytes than high_precision")
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        if self.threshold_low > self.threshold_high:
            raise ValueError("threshold_low cannot exceed threshold_high")
        if self.dwell_min < 0:
            raise ValueError("dwell_min cannot be negative")
        if self.high_scale_bytes_per_vector < 0 or self.low_scale_bytes_per_vector < 0:
            raise ValueError("scale metadata bytes cannot be negative")
        if self.inter_node_gbps <= 0:
            raise ValueError("inter_node_gbps must be positive")
        if self.codec_pack_us < 0 or self.codec_unpack_us < 0:
            raise ValueError("codec overhead cannot be negative")
        if self.codec_tile_rows < 1:
            raise ValueError("codec_tile_rows must be positive")
        if self.codec_tax_mode not in ("once_per_step", "serialized_tiles"):
            raise ValueError("codec_tax_mode must be once_per_step or serialized_tiles")
        if self.codec_h2d_us < 0:
            raise ValueError("codec_h2d_us cannot be negative")
        if self.codec_measured_rows < 1:
            raise ValueError("codec_measured_rows must be positive")


def estimate_codec_step_us(
    *,
    low_count: int,
    low_lane_count: int,
    lane_counts: dict[tuple[int, int], int] | None,
    low_lanes: set[tuple[int, int]] | None,
    codec_pack_us: float,
    codec_unpack_us: float,
    codec_h2d_us: float = 0.0,
    codec_tile_rows: int = 32,
    codec_tax_mode: str = "once_per_step",
    codec_measured_rows: int = 128,
) -> tuple[float, float, float, int]:
    """Return (codec_step_us, optimistic_us, serialized_us, codec_tiles).

    ``once_per_step`` / optimistic: one fused pack+copy+unpack paying the measured
    unit once (matches Phase-A single-kernel timing).

    ``serialized_tiles``: pay once per tile, but scale the measured unit to the
    tile width::

        per_tile_us = unit_us * (codec_tile_rows / codec_measured_rows)
        serialized_us = codec_tiles * per_tile_us

    This avoids the bug of multiplying a 128-row measurement by thousands of
    32-row tiles. It is still a host-staging bound, not RDMA.
    """
    codec_unit_us = codec_pack_us + codec_unpack_us + codec_h2d_us
    if low_count <= 0 or codec_unit_us <= 0:
        return 0.0, 0.0, 0.0, 0
    if lane_counts is None or low_lanes is None:
        codec_tiles = max(1, int(math.ceil(low_count / max(codec_tile_rows, 1))))
    else:
        codec_tiles = sum(
            math.ceil(lane_counts[lane] / codec_tile_rows)
            for lane in low_lanes
            if lane in lane_counts
        )
    optimistic = codec_unit_us
    scale = float(codec_tile_rows) / float(max(codec_measured_rows, 1))
    serialized = codec_tiles * codec_unit_us * scale
    selected = optimistic if codec_tax_mode == "once_per_step" else serialized
    return selected, optimistic, serialized, int(codec_tiles)


@dataclass
class ForwardContext:
    receiver_by_batch: torch.Tensor
    attention_mask: torch.Tensor
    request_by_batch: torch.Tensor
    forward_id: int


class ReceiverLaneController:
    """Causal lane controller plus a real combine-output quantization path.

    A routed pair is identified by ``(expert-owner sender, request-origin
    receiver)``. Local pairs stay BF16. Every remote pair uses the configured
    high transport format; pairs on selected low lanes use the low format.
    Decisions for step g use only state observed through g-1. Current active
    lane identities may be intersected with that decision because routing and
    message descriptors are available before transport encoding.
    """

    def __init__(self, config: ReceiverPolicyConfig) -> None:
        self.config = config
        self.context: ForwardContext | None = None
        self.forward_counter = 0
        self.global_step = 0
        self.prev_sender_load = np.zeros(config.ep_size, dtype=np.float64)
        self.prev_receiver_load = np.zeros(config.ep_size, dtype=np.float64)
        self.sender_credit = np.zeros(config.ep_size, dtype=np.float64)
        self.receiver_credit = np.zeros(config.ep_size, dtype=np.float64)
        self.lane_state: dict[tuple[int, int], str] = {}
        self.lane_dwell: dict[tuple[int, int], int] = {}
        self.step_rows: list[dict[str, object]] = []
        self.request_remote_pairs: dict[int, int] = {}
        self.request_low_pairs: dict[int, int] = {}
        self.request_wire_elements: dict[int, float] = {}
        self.observed_lane_pairs: dict[tuple[int, int], int] = {}
        self.observed_receiver_loads: list[float] = []
        self.observed_steps = 0

    def reset_runtime(self, clear_observations: bool = True) -> None:
        if self.context is not None:
            raise RuntimeError("cannot reset while a forward is active")
        self.forward_counter = 0
        self.global_step = 0
        self.prev_sender_load.fill(0.0)
        self.prev_receiver_load.fill(0.0)
        self.sender_credit.fill(0.0)
        self.receiver_credit.fill(0.0)
        self.lane_state.clear()
        self.lane_dwell.clear()
        self.step_rows.clear()
        self.request_remote_pairs.clear()
        self.request_low_pairs.clear()
        self.request_wire_elements.clear()
        if clear_observations:
            self.observed_lane_pairs.clear()
            self.observed_receiver_loads.clear()
            self.observed_steps = 0

    def begin_forward(
        self,
        receiver_by_batch: torch.Tensor,
        attention_mask: torch.Tensor,
        request_by_batch: torch.Tensor,
    ) -> None:
        if self.context is not None:
            raise RuntimeError("begin_forward called before the prior forward ended")
        receivers = torch.as_tensor(receiver_by_batch, dtype=torch.long).detach().cpu()
        requests = torch.as_tensor(request_by_batch, dtype=torch.long).detach().cpu()
        mask = torch.as_tensor(attention_mask, dtype=torch.bool).detach().cpu()
        if mask.ndim != 2:
            raise ValueError("attention_mask must be [batch, sequence]")
        if receivers.ndim != 1 or requests.ndim != 1:
            raise ValueError("receiver_by_batch and request_by_batch must be 1-D")
        if len(receivers) != mask.shape[0] or len(requests) != mask.shape[0]:
            raise ValueError("forward context batch dimensions do not match")
        if bool(((receivers < 0) | (receivers >= self.config.ep_size)).any()):
            raise ValueError("receiver rank outside configured EP size")
        self.context = ForwardContext(receivers, mask, requests, self.forward_counter)
        self.forward_counter += 1

    def end_forward(self) -> None:
        if self.context is None:
            raise RuntimeError("end_forward called without begin_forward")
        self.context = None

    def install_on_model(self, model) -> int:
        count = 0
        for layer_id, layer in enumerate(model.model.layers):
            if hasattr(layer, "block_sparse_moe"):
                moe = layer.block_sparse_moe
            elif hasattr(layer, "mlp") and hasattr(layer.mlp, "experts") and hasattr(layer.mlp, "gate"):
                moe = layer.mlp
            else:
                continue
            moe._idea_policy = LaneMaskedCombinePolicy(self, layer_id)
            count += 1
        if count == 0:
            raise TypeError("model exposes no supported MoE layer")
        return count

    def static_profile(self) -> dict[tuple[int, int], float]:
        denom = max(self.observed_steps, 1)
        return {lane: count / denom for lane, count in self.observed_lane_pairs.items()}

    def fitted_thresholds(self, high_quantile: float, gap_ratio: float) -> tuple[float, float, float]:
        if not 0.0 <= high_quantile <= 1.0:
            raise ValueError("high_quantile must be in [0, 1]")
        if not 0.0 <= gap_ratio <= 1.0:
            raise ValueError("gap_ratio must be in [0, 1]")
        positive = np.asarray([x for x in self.observed_receiver_loads if x > 0], dtype=np.float64)
        high = float(np.quantile(positive, high_quantile)) if len(positive) else float("inf")
        profile = list(self.static_profile().values())
        static = float(np.quantile(profile, high_quantile)) if profile else float("inf")
        return high, high * gap_ratio, static

    def request_exposure(self) -> dict[int, dict[str, float | int]]:
        request_ids = set(self.request_remote_pairs) | set(self.request_low_pairs)
        out: dict[int, dict[str, float | int]] = {}
        for request_id in request_ids:
            remote = self.request_remote_pairs.get(request_id, 0)
            low = self.request_low_pairs.get(request_id, 0)
            out[request_id] = {
                "remote_pairs": remote,
                "low_pairs": low,
                "low_frac": low / max(remote, 1),
                "wire_bytes": self.request_wire_elements.get(request_id, 0.0),
            }
        return out

    def _sender_ranks(self, selected_experts: torch.Tensor, num_experts: int) -> torch.Tensor:
        if self.config.placement == "round_robin":
            return selected_experts.remainder(self.config.ep_size)
        return torch.div(
            selected_experts * self.config.ep_size,
            num_experts,
            rounding_mode="floor",
        ).clamp_max(self.config.ep_size - 1)

    def _layout(
        self, selected_experts: torch.Tensor, num_experts: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.context is None:
            raise RuntimeError("policy apply requires begin_forward context")
        token_count, top_k = selected_experts.shape
        batch, sequence = self.context.attention_mask.shape
        if token_count != batch * sequence:
            raise ValueError(
                f"token count {token_count} != context batch*sequence {batch * sequence}; "
                "call begin_forward with the exact model attention_mask"
            )
        device = selected_experts.device
        valid_token = self.context.attention_mask.reshape(-1).to(device=device)
        receiver_token = self.context.receiver_by_batch.repeat_interleave(sequence).to(device=device)
        request_token = self.context.request_by_batch.repeat_interleave(sequence).to(device=device)
        receivers = receiver_token[:, None].expand(token_count, top_k)
        requests = request_token[:, None].expand(token_count, top_k)
        valid = valid_token[:, None].expand(token_count, top_k)
        senders = self._sender_ranks(selected_experts, num_experts)
        remote = valid & (
            torch.div(senders, self.config.gpus_per_node, rounding_mode="floor")
            != torch.div(receivers, self.config.gpus_per_node, rounding_mode="floor")
        )
        return senders, receivers, requests, valid, remote

    def _load_and_active_lanes(
        self, senders: torch.Tensor, receivers: torch.Tensor, remote: torch.Tensor
    ) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]], dict[tuple[int, int], int]]:
        ep = self.config.ep_size
        if not bool(remote.any()):
            return np.zeros(ep), np.zeros(ep), [], {}
        remote_s = senders[remote]
        remote_r = receivers[remote]
        sender_load = torch.bincount(remote_s, minlength=ep).detach().cpu().numpy().astype(np.float64)
        receiver_load = torch.bincount(remote_r, minlength=ep).detach().cpu().numpy().astype(np.float64)
        lane_ids = remote_s * ep + remote_r
        unique, counts = torch.unique(lane_ids, sorted=True, return_counts=True)
        lane_count = {
            (int(lane // ep), int(lane % ep)): int(count)
            for lane, count in zip(unique.detach().cpu().tolist(), counts.detach().cpu().tolist())
        }
        return sender_load, receiver_load, list(lane_count), lane_count

    def _select_low_lanes(self, active_lanes: Iterable[tuple[int, int]]) -> set[tuple[int, int]]:
        arm = self.config.arm
        active = list(active_lanes)
        if arm in (REFERENCE_ARM, "uniform_full"):
            return set()
        if arm == "uniform_low":
            return set(active)
        if arm == "calib_static":
            return {
                lane for lane in active
                if self.config.static_profile.get(lane, 0.0) >= self.config.static_threshold
            }
        if arm == "causal_no_hysteresis":
            return {
                (s, r) for s, r in active
                if max(self.prev_sender_load[s], self.prev_receiver_load[r]) >= self.config.threshold_high
            }
        if arm != "controller":
            raise AssertionError(arm)

        self.sender_credit = (
            self.config.alpha * self.prev_sender_load
            + (1.0 - self.config.alpha) * self.sender_credit
        )
        self.receiver_credit = (
            self.config.alpha * self.prev_receiver_load
            + (1.0 - self.config.alpha) * self.receiver_credit
        )
        low: set[tuple[int, int]] = set()
        for lane in active:
            s, r = lane
            current = self.lane_state.get(lane, "full")
            dwell = self.lane_dwell.get(lane, self.config.dwell_min)
            score = max(self.sender_credit[s], self.receiver_credit[r])
            if current == "full" and score >= self.config.threshold_high and dwell >= self.config.dwell_min:
                current = "low"
                dwell = 0
            elif current == "low" and score <= self.config.threshold_low and dwell >= self.config.dwell_min:
                current = "full"
                dwell = 0
            else:
                dwell += 1
            self.lane_state[lane] = current
            self.lane_dwell[lane] = dwell
            if current == "low":
                low.add(lane)
        return low

    def apply_layer(
        self,
        layer_id: int,
        raw_outputs: torch.Tensor,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
        num_experts: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        senders, receivers, requests, _valid, remote = self._layout(selected_experts, num_experts)
        sender_load, receiver_load, active_lanes, lane_counts = self._load_and_active_lanes(
            senders, receivers, remote
        )
        # Snapshot before select: hard-gate may reject the action and must not
        # leave dwell/credit advanced as if low ran.
        state_snapshot = {
            "lane_state": dict(self.lane_state),
            "lane_dwell": dict(self.lane_dwell),
            "sender_credit": self.sender_credit.copy(),
            "receiver_credit": self.receiver_credit.copy(),
        }
        low_lanes = self._select_low_lanes(active_lanes)

        low_mask = torch.zeros_like(remote)
        for sender, receiver in low_lanes:
            low_mask |= remote & (senders == sender) & (receivers == receiver)

        hidden_dim = int(raw_outputs.shape[-1])
        high_vector_bytes = (
            hidden_dim * BYTE_SIZES[self.config.high_precision]
            + self.config.high_scale_bytes_per_vector
        )
        low_vector_bytes = (
            hidden_dim * BYTE_SIZES[self.config.low_precision]
            + self.config.low_scale_bytes_per_vector
        )
        remote_count = int(remote.sum().item())
        low_count = int(low_mask.sum().item())
        bytes_per_us = self.config.inter_node_gbps * 1e9 / 8.0 / 1e6
        baseline_sender_bytes = sender_load * high_vector_bytes
        baseline_receiver_bytes = receiver_load * high_vector_bytes
        baseline_bottleneck_bytes = float(max(
            baseline_sender_bytes.max(initial=0.0),
            baseline_receiver_bytes.max(initial=0.0),
        ))
        baseline_step_us = baseline_bottleneck_bytes / bytes_per_us

        blocked_by_codec_tax = False
        if (
            self.config.require_positive_net_saving
            and low_count > 0
            and self.config.arm not in (REFERENCE_ARM, "uniform_full")
        ):
            pair_bytes_probe = torch.full_like(
                senders, high_vector_bytes, dtype=torch.float64
            )
            pair_bytes_probe[low_mask] = low_vector_bytes
            policy_sender_probe = torch.bincount(
                senders[remote],
                weights=pair_bytes_probe[remote],
                minlength=self.config.ep_size,
            ) if remote_count else torch.zeros(
                self.config.ep_size, device=senders.device, dtype=torch.float64
            )
            policy_receiver_probe = torch.bincount(
                receivers[remote],
                weights=pair_bytes_probe[remote],
                minlength=self.config.ep_size,
            ) if remote_count else torch.zeros(
                self.config.ep_size, device=senders.device, dtype=torch.float64
            )
            policy_bottleneck_probe = float(max(
                policy_sender_probe.max().item() if policy_sender_probe.numel() else 0.0,
                policy_receiver_probe.max().item() if policy_receiver_probe.numel() else 0.0,
            ))
            policy_wire_us = policy_bottleneck_probe / bytes_per_us
            codec_step_us, _, _, _ = estimate_codec_step_us(
                low_count=low_count,
                low_lane_count=len(low_lanes),
                lane_counts=lane_counts,
                low_lanes=low_lanes,
                codec_pack_us=self.config.codec_pack_us,
                codec_unpack_us=self.config.codec_unpack_us,
                codec_h2d_us=self.config.codec_h2d_us,
                codec_tile_rows=self.config.codec_tile_rows,
                codec_tax_mode=self.config.codec_tax_mode,
                codec_measured_rows=self.config.codec_measured_rows,
            )
            wire_saving_us = baseline_step_us - policy_wire_us
            if wire_saving_us <= codec_step_us:
                blocked_by_codec_tax = True
                low_lanes = set()
                low_mask = torch.zeros_like(remote)
                low_count = 0
                self.lane_state = state_snapshot["lane_state"]
                self.lane_dwell = state_snapshot["lane_dwell"]
                self.sender_credit = state_snapshot["sender_credit"]
                self.receiver_credit = state_snapshot["receiver_credit"]

        if self.config.arm == REFERENCE_ARM:
            approximated = raw_outputs
        else:
            approximated = raw_outputs.clone()
            if bool(remote.any()):
                approximated[remote] = apply_precision(
                    raw_outputs[remote], self.config.high_precision
                )
            if bool(low_mask.any()):
                approximated[low_mask] = apply_precision(
                    raw_outputs[low_mask], self.config.low_precision
                )

        wire_bytes = (
            (remote_count - low_count) * high_vector_bytes
            + low_count * low_vector_bytes
        )

        pair_bytes = torch.full_like(
            senders, high_vector_bytes, dtype=torch.float64
        )
        pair_bytes[low_mask] = low_vector_bytes
        policy_sender_bytes = torch.bincount(
            senders[remote], weights=pair_bytes[remote], minlength=self.config.ep_size
        ) if remote_count else torch.zeros(self.config.ep_size, device=senders.device, dtype=torch.float64)
        policy_receiver_bytes = torch.bincount(
            receivers[remote], weights=pair_bytes[remote], minlength=self.config.ep_size
        ) if remote_count else torch.zeros(self.config.ep_size, device=senders.device, dtype=torch.float64)
        policy_bottleneck_bytes = float(max(
            policy_sender_bytes.max().item() if policy_sender_bytes.numel() else 0.0,
            policy_receiver_bytes.max().item() if policy_receiver_bytes.numel() else 0.0,
        ))
        codec_step_us, codec_step_us_optimistic, codec_step_us_serialized_tiles, codec_tiles = (
            estimate_codec_step_us(
                low_count=low_count,
                low_lane_count=len(low_lanes),
                lane_counts=lane_counts,
                low_lanes=low_lanes,
                codec_pack_us=self.config.codec_pack_us,
                codec_unpack_us=self.config.codec_unpack_us,
                codec_h2d_us=self.config.codec_h2d_us,
                codec_tile_rows=self.config.codec_tile_rows,
                codec_tax_mode=self.config.codec_tax_mode,
                codec_measured_rows=self.config.codec_measured_rows,
            )
        )
        policy_step_us = policy_bottleneck_bytes / bytes_per_us + codec_step_us

        if remote_count:
            remote_requests, remote_counts = torch.unique(requests[remote], return_counts=True)
            low_requests, low_counts = (
                torch.unique(requests[low_mask], return_counts=True)
                if low_count
                else (torch.empty(0, device=requests.device, dtype=requests.dtype), torch.empty(0, device=requests.device, dtype=torch.long))
            )
            low_by_request = {
                int(req): int(count)
                for req, count in zip(low_requests.detach().cpu().tolist(), low_counts.detach().cpu().tolist())
            }
            for req, count in zip(
                remote_requests.detach().cpu().tolist(), remote_counts.detach().cpu().tolist()
            ):
                request_id = int(req)
                pair_count = int(count)
                request_low = low_by_request.get(request_id, 0)
                self.request_remote_pairs[request_id] = self.request_remote_pairs.get(request_id, 0) + pair_count
                self.request_low_pairs[request_id] = self.request_low_pairs.get(request_id, 0) + request_low
                self.request_wire_elements[request_id] = self.request_wire_elements.get(request_id, 0.0) + (
                    (pair_count - request_low) * high_vector_bytes
                    + request_low * low_vector_bytes
                )

        for lane, count in lane_counts.items():
            self.observed_lane_pairs[lane] = self.observed_lane_pairs.get(lane, 0) + count
        self.observed_receiver_loads.extend(receiver_load[receiver_load > 0].tolist())
        self.observed_steps += 1

        self.step_rows.append({
            "global_step": self.global_step,
            "forward_id": self.context.forward_id if self.context is not None else -1,
            "layer": int(layer_id),
            "arm": self.config.arm,
            "active_lane_count": len(active_lanes),
            "low_lane_count": len(low_lanes),
            "low_lanes": ";".join(f"{s}:{r}" for s, r in sorted(low_lanes)),
            "remote_pairs": remote_count,
            "low_pairs": low_count,
            "low_frac": low_count / max(remote_count, 1),
            "wire_bytes": wire_bytes,
            "high_baseline_wire_bytes": remote_count * high_vector_bytes,
            "baseline_bottleneck_bytes": baseline_bottleneck_bytes,
            "policy_bottleneck_bytes": policy_bottleneck_bytes,
            "baseline_step_us": baseline_step_us,
            "codec_tiles": codec_tiles,
            "codec_step_us_optimistic": codec_step_us_optimistic,
            "codec_step_us_serialized_tiles": codec_step_us_serialized_tiles,
            "codec_step_us": codec_step_us,
            "policy_step_us": policy_step_us,
            "blocked_by_codec_tax": int(blocked_by_codec_tax),
            "max_sender_load": float(sender_load.max(initial=0.0)),
            "max_receiver_load": float(receiver_load.max(initial=0.0)),
        })
        self.prev_sender_load = sender_load
        self.prev_receiver_load = receiver_load
        self.global_step += 1
        return approximated, routing_weights


class LaneMaskedCombinePolicy:
    """Adapter for the existing ``capture_moe._combine_idea_paths`` seam."""

    def __init__(self, controller: ReceiverLaneController, layer_id: int) -> None:
        self.controller = controller
        self.layer_id = layer_id
        self.name = f"receiver_{controller.config.arm}"

    def apply(
        self,
        raw_outputs: torch.Tensor,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor | None = None,
        num_experts: int | None = None,
        num_receiver_groups: int = 1,
        receiver_mapping: str = "contiguous",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del num_receiver_groups, receiver_mapping
        if selected_experts is None or num_experts is None:
            raise ValueError("receiver lane policy requires selected_experts and num_experts")
        return self.controller.apply_layer(
            self.layer_id,
            raw_outputs,
            routing_weights,
            selected_experts,
            num_experts,
        )

    def bytes_per_element_for_selected(
        self,
        selected_experts: torch.Tensor,
        num_experts: int,
        num_receiver_groups: int = 1,
        receiver_mapping: str = "contiguous",
        routing_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del num_experts, num_receiver_groups, receiver_mapping, routing_weights
        return torch.full_like(selected_experts, 2.0, dtype=torch.float32)
