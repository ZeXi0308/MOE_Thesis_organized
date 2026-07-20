#!/usr/bin/env python3
"""GPU codec and transfer gate for receiver-aware EP representations.

Implements real packed symmetric INT4/INT8 buffers with Triton and measures:
  * sender pack latency;
  * receiver unpack/add latency;
  * pinned-host to GPU transfer latency for the exact packed byte count;
  * serial and optimistic-overlap latency models at configurable link rates.

The equal-payload comparison is:
  * direct_mixed_50: half the vectors INT8, half INT4;
  * progressive_50: every vector gets INT4 base, half get INT4 residual.

Both average six payload bits/element, but progressive has extra residual scales
and a dependency between base and residual packing.  Selection/compaction and
RDMA descriptor costs are deliberately excluded, making this an optimistic
lower bound for progressive transport.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import triton
import triton.language as tl


@triton.jit
def pack_int4_kernel(
    source,
    packed,
    scales,
    hidden: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, block)
    values = tl.load(source + row * hidden + offsets, mask=offsets < hidden, other=0.0)
    maximum = tl.max(tl.abs(values).to(tl.float32), axis=0)
    scale = tl.maximum(maximum / 7.0, 1.0e-8)
    tl.store(scales + row, scale)

    pairs = tl.arange(0, block // 2)
    even_offsets = 2 * pairs
    odd_offsets = even_offsets + 1
    even = tl.load(
        source + row * hidden + even_offsets,
        mask=even_offsets < hidden,
        other=0.0,
    ).to(tl.float32)
    odd = tl.load(
        source + row * hidden + odd_offsets,
        mask=odd_offsets < hidden,
        other=0.0,
    ).to(tl.float32)
    even_scaled = even / scale
    odd_scaled = odd / scale
    even_q = tl.where(
        even_scaled >= 0,
        tl.floor(even_scaled + 0.5),
        tl.ceil(even_scaled - 0.5),
    )
    odd_q = tl.where(
        odd_scaled >= 0,
        tl.floor(odd_scaled + 0.5),
        tl.ceil(odd_scaled - 0.5),
    )
    even_q = tl.maximum(tl.minimum(even_q, 7.0), -7.0).to(tl.int32)
    odd_q = tl.maximum(tl.minimum(odd_q, 7.0), -7.0).to(tl.int32)
    encoded = (even_q & 15) | ((odd_q & 15) << 4)
    tl.store(
        packed + row * ((hidden + 1) // 2) + pairs,
        encoded.to(tl.uint8),
        mask=pairs < (hidden + 1) // 2,
    )


@triton.jit
def unpack_int4_kernel(
    packed,
    scales,
    output,
    hidden: tl.constexpr,
    block: tl.constexpr,
):
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
    tl.store(
        output + row * hidden + even_offsets,
        low * scale,
        mask=even_offsets < hidden,
    )
    tl.store(
        output + row * hidden + odd_offsets,
        high * scale,
        mask=odd_offsets < hidden,
    )


@triton.jit
def add_int4_residual_kernel(
    packed,
    scales,
    output,
    hidden: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0)
    pairs = tl.arange(0, block // 2)
    encoded = tl.load(
        packed + row * ((hidden + 1) // 2) + pairs,
        mask=pairs < (hidden + 1) // 2,
        other=0,
    ).to(tl.int32)
    low = encoded & 15
    high = (encoded >> 4) & 15
    low = tl.where(low >= 8, low - 16, low).to(tl.float32)
    high = tl.where(high >= 8, high - 16, high).to(tl.float32)
    scale = tl.load(scales + row)
    even_offsets = 2 * pairs
    odd_offsets = even_offsets + 1
    old_even = tl.load(
        output + row * hidden + even_offsets,
        mask=even_offsets < hidden,
        other=0.0,
    ).to(tl.float32)
    old_odd = tl.load(
        output + row * hidden + odd_offsets,
        mask=odd_offsets < hidden,
        other=0.0,
    ).to(tl.float32)
    tl.store(
        output + row * hidden + even_offsets,
        old_even + low * scale,
        mask=even_offsets < hidden,
    )
    tl.store(
        output + row * hidden + odd_offsets,
        old_odd + high * scale,
        mask=odd_offsets < hidden,
    )


@triton.jit
def pack_residual_int4_kernel(
    source,
    base_packed,
    base_scales,
    residual_packed,
    residual_scales,
    hidden: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, block)
    values = tl.load(source + row * hidden + offsets, mask=offsets < hidden, other=0.0)
    encoded = tl.load(
        base_packed + row * ((hidden + 1) // 2) + offsets // 2,
        mask=offsets < hidden,
        other=0,
    ).to(tl.int32)
    nibble = tl.where((offsets & 1) == 0, encoded & 15, (encoded >> 4) & 15)
    signed = tl.where(nibble >= 8, nibble - 16, nibble).to(tl.float32)
    base_scale = tl.load(base_scales + row)
    residual = values.to(tl.float32) - signed * base_scale
    maximum = tl.max(tl.abs(residual), axis=0)
    residual_scale = tl.maximum(maximum / 7.0, 1.0e-8)
    tl.store(residual_scales + row, residual_scale)

    pairs = tl.arange(0, block // 2)
    even_offsets = 2 * pairs
    odd_offsets = even_offsets + 1
    original_even = tl.load(
        source + row * hidden + even_offsets,
        mask=even_offsets < hidden,
        other=0.0,
    ).to(tl.float32)
    original_odd = tl.load(
        source + row * hidden + odd_offsets,
        mask=odd_offsets < hidden,
        other=0.0,
    ).to(tl.float32)
    base_encoded = tl.load(
        base_packed + row * ((hidden + 1) // 2) + pairs,
        mask=pairs < (hidden + 1) // 2,
        other=0,
    ).to(tl.int32)
    base_low = base_encoded & 15
    base_high = (base_encoded >> 4) & 15
    base_low = tl.where(base_low >= 8, base_low - 16, base_low).to(tl.float32)
    base_high = tl.where(base_high >= 8, base_high - 16, base_high).to(tl.float32)
    even_residual = original_even - base_low * base_scale
    odd_residual = original_odd - base_high * base_scale
    even_scaled = even_residual / residual_scale
    odd_scaled = odd_residual / residual_scale
    even_q = tl.where(
        even_scaled >= 0,
        tl.floor(even_scaled + 0.5),
        tl.ceil(even_scaled - 0.5),
    )
    odd_q = tl.where(
        odd_scaled >= 0,
        tl.floor(odd_scaled + 0.5),
        tl.ceil(odd_scaled - 0.5),
    )
    even_q = tl.maximum(tl.minimum(even_q, 7.0), -7.0).to(tl.int32)
    odd_q = tl.maximum(tl.minimum(odd_q, 7.0), -7.0).to(tl.int32)
    residual_encoded = (even_q & 15) | ((odd_q & 15) << 4)
    tl.store(
        residual_packed + row * ((hidden + 1) // 2) + pairs,
        residual_encoded.to(tl.uint8),
        mask=pairs < (hidden + 1) // 2,
    )


@triton.jit
def pack_int8_kernel(
    source,
    packed,
    scales,
    hidden: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, block)
    values = tl.load(source + row * hidden + offsets, mask=offsets < hidden, other=0.0)
    maximum = tl.max(tl.abs(values).to(tl.float32), axis=0)
    scale = tl.maximum(maximum / 127.0, 1.0e-8)
    scaled = values.to(tl.float32) / scale
    quantized = tl.where(
        scaled >= 0,
        tl.floor(scaled + 0.5),
        tl.ceil(scaled - 0.5),
    )
    quantized = tl.maximum(tl.minimum(quantized, 127.0), -127.0).to(tl.int8)
    tl.store(packed + row * hidden + offsets, quantized, mask=offsets < hidden)
    tl.store(scales + row, scale)


@triton.jit
def unpack_int8_kernel(
    packed,
    scales,
    output,
    hidden: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, block)
    quantized = tl.load(
        packed + row * hidden + offsets,
        mask=offsets < hidden,
        other=0,
    ).to(tl.float32)
    scale = tl.load(scales + row)
    tl.store(
        output + row * hidden + offsets,
        quantized * scale,
        mask=offsets < hidden,
    )


def time_cuda(function, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        function()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end) * 1000.0 / repeats)


def launch_pack_int4(source, packed, scales, hidden, block) -> None:
    pack_int4_kernel[(len(source),)](
        source,
        packed,
        scales,
        hidden=hidden,
        block=block,
        num_warps=8,
    )


def launch_unpack_int4(packed, scales, output, rows, hidden, block) -> None:
    unpack_int4_kernel[(rows,)](
        packed,
        scales,
        output,
        hidden=hidden,
        block=block,
        num_warps=8,
    )


def launch_pack_int8(source, packed, scales, hidden, block) -> None:
    pack_int8_kernel[(len(source),)](
        source,
        packed,
        scales,
        hidden=hidden,
        block=block,
        num_warps=8,
    )


def launch_unpack_int8(packed, scales, output, rows, hidden, block) -> None:
    unpack_int8_kernel[(rows,)](
        packed,
        scales,
        output,
        hidden=hidden,
        block=block,
        num_warps=8,
    )


def run_shape(
    rows: int,
    hidden: int,
    refined_fraction: float,
    warmup: int,
    repeats: int,
    link_gbps: list[float],
) -> tuple[list[dict], dict]:
    if hidden % 2:
        raise ValueError("hidden size must be even")
    block = triton.next_power_of_2(hidden)
    refined_rows = int(round(rows * refined_fraction))
    high_rows = refined_rows
    low_rows = rows - high_rows
    source = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
    output = torch.empty_like(source)
    int4_packed = torch.empty(
        rows, hidden // 2, device="cuda", dtype=torch.uint8
    )
    int4_scales = torch.empty(rows, device="cuda", dtype=torch.float32)
    residual_packed = torch.empty(
        max(refined_rows, 1), hidden // 2, device="cuda", dtype=torch.uint8
    )
    residual_scales = torch.empty(
        max(refined_rows, 1), device="cuda", dtype=torch.float32
    )
    int8_packed = torch.empty(
        max(high_rows, 1), hidden, device="cuda", dtype=torch.int8
    )
    int8_scales = torch.empty(max(high_rows, 1), device="cuda", dtype=torch.float32)
    low_packed = torch.empty(
        max(low_rows, 1), hidden // 2, device="cuda", dtype=torch.uint8
    )
    low_scales = torch.empty(max(low_rows, 1), device="cuda", dtype=torch.float32)

    def int8_pack() -> None:
        launch_pack_int8(source, int8_packed, int8_scales, hidden, block)

    def int8_unpack() -> None:
        launch_unpack_int8(int8_packed, int8_scales, output, rows, hidden, block)

    # Uniform INT8 needs full-sized storage.
    uniform_int8_packed = torch.empty(rows, hidden, device="cuda", dtype=torch.int8)
    uniform_int8_scales = torch.empty(rows, device="cuda", dtype=torch.float32)

    def uniform_int8_pack() -> None:
        launch_pack_int8(
            source, uniform_int8_packed, uniform_int8_scales, hidden, block
        )

    def uniform_int8_unpack() -> None:
        launch_unpack_int8(
            uniform_int8_packed,
            uniform_int8_scales,
            output,
            rows,
            hidden,
            block,
        )

    def uniform_int4_pack() -> None:
        launch_pack_int4(source, int4_packed, int4_scales, hidden, block)

    def uniform_int4_unpack() -> None:
        launch_unpack_int4(
            int4_packed, int4_scales, output, rows, hidden, block
        )

    def progressive_pack() -> None:
        launch_pack_int4(source, int4_packed, int4_scales, hidden, block)
        if refined_rows:
            pack_residual_int4_kernel[(refined_rows,)](
                source,
                int4_packed,
                int4_scales,
                residual_packed,
                residual_scales,
                hidden=hidden,
                block=block,
                num_warps=8,
            )

    def progressive_unpack() -> None:
        launch_unpack_int4(
            int4_packed, int4_scales, output, rows, hidden, block
        )
        if refined_rows:
            add_int4_residual_kernel[(refined_rows,)](
                residual_packed,
                residual_scales,
                output,
                hidden=hidden,
                block=block,
                num_warps=8,
            )

    def direct_pack() -> None:
        if high_rows:
            launch_pack_int8(
                source[:high_rows],
                int8_packed,
                int8_scales,
                hidden,
                block,
            )
        if low_rows:
            launch_pack_int4(
                source[high_rows:],
                low_packed,
                low_scales,
                hidden,
                block,
            )

    def direct_unpack() -> None:
        if high_rows:
            launch_unpack_int8(
                int8_packed,
                int8_scales,
                output[:high_rows],
                high_rows,
                hidden,
                block,
            )
        if low_rows:
            launch_unpack_int4(
                low_packed,
                low_scales,
                output[high_rows:],
                low_rows,
                hidden,
                block,
            )

    modes = {
        "uniform_int8": (
            uniform_int8_pack,
            uniform_int8_unpack,
            rows * hidden + rows * 4,
        ),
        "uniform_int4": (
            uniform_int4_pack,
            uniform_int4_unpack,
            rows * (hidden // 2) + rows * 4,
        ),
        "direct_mixed_50": (
            direct_pack,
            direct_unpack,
            high_rows * hidden
            + low_rows * (hidden // 2)
            + rows * 4,
        ),
        "progressive_50": (
            progressive_pack,
            progressive_unpack,
            rows * (hidden // 2)
            + refined_rows * (hidden // 2)
            + (rows + refined_rows) * 4,
        ),
    }

    rows_out: list[dict] = []
    correctness: dict[str, float] = {}
    for mode, (pack_function, unpack_function, wire_bytes) in modes.items():
        pack_function()
        unpack_function()
        torch.cuda.synchronize()
        correctness[f"{mode}_reconstruction_mse"] = float(
            (source.float() - output.float()).square().mean().item()
        )
        pack_us = time_cuda(pack_function, warmup, repeats)
        unpack_us = time_cuda(unpack_function, warmup, repeats)
        pinned = torch.empty(wire_bytes, dtype=torch.uint8, pin_memory=True)
        device_buffer = torch.empty(wire_bytes, dtype=torch.uint8, device="cuda")

        def transfer() -> None:
            device_buffer.copy_(pinned, non_blocking=True)

        h2d_us = time_cuda(transfer, warmup, repeats)
        base = {
            "mode": mode,
            "rows": rows,
            "hidden": hidden,
            "refined_fraction": refined_fraction,
            "wire_bytes": wire_bytes,
            "sender_pack_us": pack_us,
            "receiver_unpack_us": unpack_us,
            "measured_pinned_h2d_us": h2d_us,
            "measured_serial_h2d_total_us": pack_us + h2d_us + unpack_us,
        }
        for gbps in link_gbps:
            wire_us = wire_bytes * 8.0 / (gbps * 1e9) * 1e6
            base[f"wire_us_{gbps:g}gbps"] = wire_us
            base[f"modeled_serial_us_{gbps:g}gbps"] = pack_us + wire_us + unpack_us
            base[f"modeled_overlap_us_{gbps:g}gbps"] = max(pack_us, wire_us) + unpack_us
        rows_out.append(base)
    return rows_out, correctness


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden-sizes", default="512,2048")
    parser.add_argument("--rows", default="32,128,512")
    parser.add_argument("--refined-fraction", type=float, default=0.5)
    parser.add_argument("--link-gbps", default="200,400,800")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    hidden_sizes = [int(value) for value in args.hidden_sizes.split(",")]
    row_counts = [int(value) for value in args.rows.split(",")]
    link_gbps = [float(value) for value in args.link_gbps.split(",")]
    all_rows: list[dict] = []
    correctness: dict[str, dict] = {}
    for hidden in hidden_sizes:
        for rows in row_counts:
            print(f"benchmarking rows={rows}, hidden={hidden}", flush=True)
            measurements, checks = run_shape(
                rows,
                hidden,
                args.refined_fraction,
                args.warmup,
                args.repeats,
                link_gbps,
            )
            all_rows.extend(measurements)
            correctness[f"rows{rows}_hidden{hidden}"] = checks
    frame = pd.DataFrame(all_rows)
    frame.to_csv(output / "codec_break_even.csv", index=False)
    metadata = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "hidden_sizes": hidden_sizes,
        "rows": row_counts,
        "refined_fraction": args.refined_fraction,
        "link_gbps": link_gbps,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "correctness": correctness,
        "evidence_boundary": (
            "single-GPU Triton codec plus pinned H2D; no RDMA, selection "
            "compaction, descriptors, receiver credits, or multi-node topology"
        ),
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
