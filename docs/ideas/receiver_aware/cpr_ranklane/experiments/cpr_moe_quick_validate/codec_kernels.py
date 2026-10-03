"""Triton codecs for the CPR-MoE single-GPU necessary-condition gate.

The kernels implement per-row symmetric INT8 and INT4.  They intentionally do
not emulate FP8, NCCL, expert compute, or a fused producer/consumer pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import triton
import triton.language as tl


@triton.jit
def _round_ties_to_even(values):
    """Round FP32 values like torch.round, including exact half-way cases."""
    magnitude = tl.abs(values)
    lower = tl.floor(magnitude)
    fraction = magnitude - lower
    lower_i = lower.to(tl.int32)
    rounded_magnitude = tl.where(
        fraction > 0.5,
        lower + 1.0,
        tl.where(
            fraction < 0.5,
            lower,
            tl.where((lower_i & 1) == 1, lower + 1.0, lower),
        ),
    )
    return tl.where(values < 0, -rounded_magnitude, rounded_magnitude)


@triton.jit
def _pack_int8_kernel(source, packed, scales, hidden: tl.constexpr, block: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, block)
    mask = offsets < hidden
    values = tl.load(source + row * hidden + offsets, mask=mask, other=0.0).to(tl.float32)
    maximum = tl.max(tl.abs(values), axis=0)
    scale = tl.div_rn(tl.maximum(maximum, 1.0e-8), 127.0)
    scaled = tl.div_rn(values, scale)
    quantized = _round_ties_to_even(scaled)
    quantized = tl.maximum(tl.minimum(quantized, 127.0), -127.0).to(tl.int8)
    tl.store(scales + row, scale)
    tl.store(packed + row * hidden + offsets, quantized, mask=mask)


@triton.jit
def _unpack_int8_kernel(packed, scales, output, hidden: tl.constexpr, block: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, block)
    mask = offsets < hidden
    values = tl.load(packed + row * hidden + offsets, mask=mask, other=0).to(tl.float32)
    scale = tl.load(scales + row)
    tl.store(output + row * hidden + offsets, values * scale, mask=mask)


@triton.jit
def _pack_int4_kernel(source, packed, scales, hidden: tl.constexpr, block: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, block)
    values = tl.load(source + row * hidden + offsets, mask=offsets < hidden, other=0.0).to(tl.float32)
    maximum = tl.max(tl.abs(values), axis=0)
    scale = tl.div_rn(tl.maximum(maximum, 1.0e-8), 7.0)
    tl.store(scales + row, scale)

    pairs = tl.arange(0, block // 2)
    even_offsets = 2 * pairs
    odd_offsets = even_offsets + 1
    even = tl.load(source + row * hidden + even_offsets, mask=even_offsets < hidden, other=0.0).to(tl.float32)
    odd = tl.load(source + row * hidden + odd_offsets, mask=odd_offsets < hidden, other=0.0).to(tl.float32)
    even_scaled = tl.div_rn(even, scale)
    odd_scaled = tl.div_rn(odd, scale)
    even_q = _round_ties_to_even(even_scaled)
    odd_q = _round_ties_to_even(odd_scaled)
    even_q = tl.maximum(tl.minimum(even_q, 7.0), -7.0).to(tl.int32)
    odd_q = tl.maximum(tl.minimum(odd_q, 7.0), -7.0).to(tl.int32)
    encoded = (even_q & 15) | ((odd_q & 15) << 4)
    tl.store(
        packed + row * ((hidden + 1) // 2) + pairs,
        encoded.to(tl.uint8),
        mask=pairs < (hidden + 1) // 2,
    )


@triton.jit
def _unpack_int4_kernel(packed, scales, output, hidden: tl.constexpr, block: tl.constexpr):
    row = tl.program_id(0)
    pairs = tl.arange(0, block // 2)
    mask = pairs < (hidden + 1) // 2
    encoded = tl.load(
        packed + row * ((hidden + 1) // 2) + pairs,
        mask=mask,
        other=0,
    ).to(tl.int32)
    low = encoded & 15
    high = (encoded >> 4) & 15
    low = tl.where(low >= 8, low - 16, low).to(tl.float32)
    high = tl.where(high >= 8, high - 16, high).to(tl.float32)
    scale = tl.load(scales + row)
    even_offsets = 2 * pairs
    odd_offsets = even_offsets + 1
    tl.store(output + row * hidden + even_offsets, low * scale, mask=even_offsets < hidden)
    tl.store(output + row * hidden + odd_offsets, high * scale, mask=odd_offsets < hidden)


@dataclass
class CodecCase:
    pack: Callable[[], None]
    unpack: Callable[[], None]
    connected: Callable[[], None]
    packed: torch.Tensor
    scales: torch.Tensor
    output: torch.Tensor
    policy_wire_bytes: int
    scale_bytes: int


def build_codec_case(source: torch.Tensor, mode: str) -> CodecCase:
    if source.device.type != "cuda":
        raise ValueError("codec source must be on CUDA")
    if source.dtype != torch.bfloat16:
        raise ValueError(f"expected BF16 source, got {source.dtype}")
    if source.ndim != 2 or not source.is_contiguous():
        raise ValueError("source must be contiguous [rows, hidden]")
    rows, hidden = (int(source.shape[0]), int(source.shape[1]))
    if hidden <= 0 or hidden % 2:
        raise ValueError("hidden must be positive and even")
    block = triton.next_power_of_2(hidden)
    scales = torch.empty(rows, device=source.device, dtype=torch.float32)
    output = torch.empty_like(source)

    if mode == "int8":
        packed = torch.empty((rows, hidden), device=source.device, dtype=torch.int8)

        def pack() -> None:
            _pack_int8_kernel[(rows,)](
                source, packed, scales, hidden=hidden, block=block, num_warps=8
            )

        def unpack() -> None:
            _unpack_int8_kernel[(rows,)](
                packed, scales, output, hidden=hidden, block=block, num_warps=8
            )

    elif mode == "int4":
        packed = torch.empty((rows, hidden // 2), device=source.device, dtype=torch.uint8)

        def pack() -> None:
            _pack_int4_kernel[(rows,)](
                source, packed, scales, hidden=hidden, block=block, num_warps=8
            )

        def unpack() -> None:
            _unpack_int4_kernel[(rows,)](
                packed, scales, output, hidden=hidden, block=block, num_warps=8
            )

    else:
        raise ValueError(f"unsupported codec mode: {mode}")

    def connected() -> None:
        pack()
        unpack()

    scale_bytes = scales.numel() * scales.element_size()
    policy_wire_bytes = packed.numel() * packed.element_size() + scale_bytes
    return CodecCase(
        pack, unpack, connected, packed, scales, output, policy_wire_bytes, scale_bytes
    )


def reference_codec(
    source: torch.Tensor, mode: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """PyTorch oracle with the same per-row scale and RTNE rounding contract."""
    if source.dtype != torch.bfloat16 or source.ndim != 2:
        raise ValueError("reference source must be a BF16 [rows, hidden] tensor")
    if source.shape[1] % 2:
        raise ValueError("reference hidden size must be even")
    qmax = 127 if mode == "int8" else 7 if mode == "int4" else None
    if qmax is None:
        raise ValueError(f"unsupported codec mode: {mode}")
    values = source.float()
    scales = values.abs().amax(dim=-1).clamp_min(1.0e-8) / float(qmax)
    quantized = torch.round(values / scales[:, None]).clamp(-qmax, qmax).to(torch.int32)
    output = (quantized.float() * scales[:, None]).to(torch.bfloat16)
    if mode == "int8":
        packed = quantized.to(torch.int8)
    else:
        low = quantized[:, 0::2] & 15
        high = (quantized[:, 1::2] & 15) << 4
        packed = (low | high).to(torch.uint8)
    return packed, scales, output


def assert_codec_matches_reference(source: torch.Tensor, mode: str) -> None:
    """Fail closed if the Triton bytes/scale/dequant differ from the oracle."""
    case = build_codec_case(source, mode)
    expected_packed, expected_scales, expected_output = reference_codec(source, mode)
    case.connected()
    torch.cuda.synchronize(source.device)
    if not torch.equal(case.packed, expected_packed):
        raise AssertionError(f"{mode} packed bytes differ from PyTorch reference")
    if not torch.allclose(case.scales, expected_scales, rtol=1.0e-6, atol=1.0e-10):
        raise AssertionError(f"{mode} scales differ from PyTorch reference")
    if not torch.equal(case.output, expected_output):
        raise AssertionError(f"{mode} reconstruction differs from PyTorch reference")
