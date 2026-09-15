#!/usr/bin/env python3
"""Homogeneous FP8/INT4 lane codec gate: pack → pinned H2D → unpack.

Measures real Triton pack/unpack plus pinned-host copy for whole-lane
homogeneous codecs, then computes analytic wire-time net savings vs BF16
baseline at configurable link rates.

Evidence boundary (must appear in every report):
  single-GPU codec + pinned H2D; not NCCL/RDMA; no incast/collective headers.
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

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import triton
import triton.language as tl


# OLMoE-1B-7B hidden=2048; LLM-jp MoE hidden=512 (combine vector width).
DEFAULT_HIDDEN = "512,2048"
DEFAULT_ROWS = "32,128,512,2048"
DEFAULT_LINK_GBPS = "100,200,400,800"
# e4m3 finite max used for per-vector symmetric FP8 scaling.
FP8_E4M3_MAX = 448.0


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
def pack_fp8_kernel(
    source,
    packed,
    scales,
    hidden: tl.constexpr,
    block: tl.constexpr,
    fp8_max: tl.constexpr,
):
    """Per-vector absmax scale + round-to-nearest into signed int8 proxy for FP8 wire."""
    row = tl.program_id(0)
    offsets = tl.arange(0, block)
    values = tl.load(source + row * hidden + offsets, mask=offsets < hidden, other=0.0)
    maximum = tl.max(tl.abs(values).to(tl.float32), axis=0)
    scale = tl.maximum(maximum / fp8_max, 1.0e-8)
    tl.store(scales + row, scale)
    scaled = values.to(tl.float32) / scale
    quantized = tl.where(
        scaled >= 0,
        tl.floor(scaled + 0.5),
        tl.ceil(scaled - 0.5),
    )
    quantized = tl.maximum(tl.minimum(quantized, fp8_max), -fp8_max)
    # Store as uint8 bit-pattern proxy (one byte / element), matching FP8 wire width.
    stored = (quantized.to(tl.int32) & 255).to(tl.uint8)
    tl.store(packed + row * hidden + offsets, stored, mask=offsets < hidden)


@triton.jit
def unpack_fp8_kernel(
    packed,
    scales,
    output,
    hidden: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, block)
    encoded = tl.load(
        packed + row * hidden + offsets,
        mask=offsets < hidden,
        other=0,
    ).to(tl.int32)
    # Recover signed proxy in [-128, 127] then scale (fp8_max range clipped at pack).
    signed = tl.where(encoded >= 128, encoded - 256, encoded).to(tl.float32)
    scale = tl.load(scales + row)
    tl.store(
        output + row * hidden + offsets,
        signed * scale,
        mask=offsets < hidden,
    )


def time_cuda_samples(function, warmup: int, repeats: int) -> np.ndarray:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    samples = np.empty(repeats, dtype=np.float64)
    for i in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        end.record()
        end.synchronize()
        samples[i] = float(start.elapsed_time(end) * 1000.0)  # ms → µs
    return samples


def wire_us(bytes_count: float, gbps: float) -> float:
    return bytes_count * 8.0 / (gbps * 1e9) * 1e6


def break_even_gbps(
    baseline_bytes: float,
    policy_bytes: float,
    codec_us: float,
) -> float:
    """Link rate where wire_saving_us == codec_us. inf ⇒ never breaks even."""
    saved = baseline_bytes - policy_bytes
    if saved <= 0:
        return float("inf")
    if codec_us <= 0:
        return 0.0
    # saved_bytes * 8 / (g * 1e9) * 1e6 = codec_us
    # g = saved_bytes * 8 / codec_us * 1e-3
    return saved * 8.0 / codec_us * 1e-3


def run_shape(
    rows: int,
    hidden: int,
    warmup: int,
    repeats: int,
    link_gbps: list[float],
) -> tuple[list[dict], dict]:
    if hidden % 2:
        raise ValueError("hidden size must be even")
    block = triton.next_power_of_2(hidden)
    source = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
    output = torch.empty_like(source)

    # BF16 baseline payload (no pack); optional H2D of raw bytes.
    bf16_wire = rows * hidden * 2
    bf16_pinned = torch.empty(bf16_wire, dtype=torch.uint8, pin_memory=True)
    bf16_device = torch.empty(bf16_wire, dtype=torch.uint8, device="cuda")

    def bf16_h2d() -> None:
        bf16_device.copy_(bf16_pinned, non_blocking=True)

    bf16_h2d_samples = time_cuda_samples(bf16_h2d, warmup, repeats)

    # Homogeneous FP8: 1 byte/elem + FP32 scale/row.
    fp8_packed = torch.empty(rows, hidden, device="cuda", dtype=torch.uint8)
    fp8_scales = torch.empty(rows, device="cuda", dtype=torch.float32)
    fp8_wire = rows * hidden + rows * 4
    fp8_pinned = torch.empty(fp8_wire, dtype=torch.uint8, pin_memory=True)
    fp8_device = torch.empty(fp8_wire, dtype=torch.uint8, device="cuda")

    def fp8_pack() -> None:
        pack_fp8_kernel[(rows,)](
            source,
            fp8_packed,
            fp8_scales,
            hidden=hidden,
            block=block,
            fp8_max=FP8_E4M3_MAX,
            num_warps=8,
        )

    def fp8_unpack() -> None:
        unpack_fp8_kernel[(rows,)](
            fp8_packed,
            fp8_scales,
            output,
            hidden=hidden,
            block=block,
            num_warps=8,
        )

    def fp8_h2d() -> None:
        fp8_device.copy_(fp8_pinned, non_blocking=True)

    # Homogeneous INT4: nibble pack + FP32 scale/row.
    int4_packed = torch.empty(rows, hidden // 2, device="cuda", dtype=torch.uint8)
    int4_scales = torch.empty(rows, device="cuda", dtype=torch.float32)
    int4_wire = rows * (hidden // 2) + rows * 4
    int4_pinned = torch.empty(int4_wire, dtype=torch.uint8, pin_memory=True)
    int4_device = torch.empty(int4_wire, dtype=torch.uint8, device="cuda")

    def int4_pack() -> None:
        pack_int4_kernel[(rows,)](
            source,
            int4_packed,
            int4_scales,
            hidden=hidden,
            block=block,
            num_warps=8,
        )

    def int4_unpack() -> None:
        unpack_int4_kernel[(rows,)](
            int4_packed,
            int4_scales,
            output,
            hidden=hidden,
            block=block,
            num_warps=8,
        )

    def int4_h2d() -> None:
        int4_device.copy_(int4_pinned, non_blocking=True)

    modes: dict[str, tuple] = {
        "baseline_bf16": (None, None, bf16_h2d, bf16_wire, bf16_h2d_samples),
        "homo_fp8": (fp8_pack, fp8_unpack, fp8_h2d, fp8_wire, None),
        "homo_int4": (int4_pack, int4_unpack, int4_h2d, int4_wire, None),
    }

    rows_out: list[dict] = []
    correctness: dict[str, float] = {}
    for mode, (pack_fn, unpack_fn, h2d_fn, policy_bytes, pre_h2d) in modes.items():
        if pack_fn is not None:
            pack_fn()
            unpack_fn()
            torch.cuda.synchronize()
            correctness[f"{mode}_reconstruction_mse"] = float(
                (source.float() - output.float()).square().mean().item()
            )
            pack_samples = time_cuda_samples(pack_fn, warmup, repeats)
            unpack_samples = time_cuda_samples(unpack_fn, warmup, repeats)
            h2d_samples = time_cuda_samples(h2d_fn, warmup, repeats)
            pack_us = float(np.median(pack_samples))
            unpack_us = float(np.median(unpack_samples))
            h2d_us = float(np.median(h2d_samples))
            pack_p95 = float(np.percentile(pack_samples, 95))
            unpack_p95 = float(np.percentile(unpack_samples, 95))
            h2d_p95 = float(np.percentile(h2d_samples, 95))
            codec_us = pack_us + h2d_us + unpack_us
            codec_p95 = pack_p95 + h2d_p95 + unpack_p95
        else:
            pack_us = unpack_us = 0.0
            pack_p95 = unpack_p95 = 0.0
            h2d_us = float(np.median(pre_h2d))
            h2d_p95 = float(np.percentile(pre_h2d, 95))
            codec_us = h2d_us  # optional copy tax only
            codec_p95 = h2d_p95
            correctness[f"{mode}_reconstruction_mse"] = 0.0

        measured_codec = 0.0 if mode == "baseline_bf16" else codec_us
        measured_codec_p95 = 0.0 if mode == "baseline_bf16" else codec_p95
        be = (
            0.0
            if mode == "baseline_bf16"
            else break_even_gbps(bf16_wire, policy_bytes, measured_codec)
        )
        base = {
            "mode": mode,
            "rows": rows,
            "hidden": hidden,
            "baseline_wire_bytes": bf16_wire,
            "policy_wire_bytes": policy_bytes,
            "pack_us_p50": pack_us,
            "pack_us_p95": pack_p95,
            "unpack_us_p50": unpack_us,
            "unpack_us_p95": unpack_p95,
            "h2d_us_p50": h2d_us,
            "h2d_us_p95": h2d_p95,
            "codec_total_us_p50": measured_codec,
            "codec_total_us_p95": measured_codec_p95,
            # Convenience aliases used by hard-gate / replay.
            "pack_us": pack_us,
            "unpack_us": unpack_us,
            "h2d_us": h2d_us,
            "break_even_gbps": be,
        }
        for gbps in link_gbps:
            baseline_wire = wire_us(bf16_wire, gbps)
            policy_wire = wire_us(policy_bytes, gbps)
            # Net vs BF16 wire: wire saving minus measured pack+h2d+unpack.
            net = (baseline_wire - policy_wire) - measured_codec
            net_p95 = (baseline_wire - policy_wire) - measured_codec_p95
            tag = f"{gbps:g}"
            base[f"baseline_wire_us_{tag}gbps"] = baseline_wire
            base[f"policy_wire_us_{tag}gbps"] = policy_wire
            base[f"net_delta_us_{tag}gbps_p50"] = net
            base[f"net_delta_us_{tag}gbps_p95"] = net_p95
        rows_out.append(base)
    return rows_out, correctness


def summarize_gate(frame: pd.DataFrame, serving_rows: list[int], serving_gbps: list[float]) -> dict:
    """Pre-registered serving-point viability for homo_fp8."""
    fp8 = frame[frame["mode"] == "homo_fp8"].copy()
    cells = []
    for _, row in fp8.iterrows():
        if int(row["rows"]) not in serving_rows:
            continue
        for gbps in serving_gbps:
            tag = f"{gbps:g}"
            col = f"net_delta_us_{tag}gbps_p50"
            col_p95 = f"net_delta_us_{tag}gbps_p95"
            if col not in row:
                continue
            cells.append(
                {
                    "rows": int(row["rows"]),
                    "hidden": int(row["hidden"]),
                    "gbps": gbps,
                    "net_p50": float(row[col]),
                    "net_p95": float(row[col_p95]),
                    "viable": bool(row[col] > 0 and row[col_p95] > 0),
                }
            )
    if not cells:
        return {"n_cells": 0, "viable_frac": 0.0, "recommend_hard_gate": True, "cells": []}
    viable = sum(1 for c in cells if c["viable"])
    frac = viable / len(cells)
    return {
        "n_cells": len(cells),
        "viable_count": viable,
        "viable_frac": frac,
        # Hard-gate-on if majority of common serving cells are net-negative.
        "recommend_hard_gate": frac < 0.5,
        "cells": cells,
    }


def write_report(output: Path, frame: pd.DataFrame, gate: dict, metadata: dict) -> None:
    lines = [
        "# Homogeneous Lane Codec Gate",
        "",
        f"GPU: `{metadata.get('gpu', 'unknown')}`",
        "",
        "## Evidence boundary",
        "",
        metadata["evidence_boundary"],
        "",
        "## Pre-registered serving-point summary (homo_fp8)",
        "",
        f"- cells: {gate['n_cells']}",
        f"- viable (net_p50>0 and net_p95>0): {gate.get('viable_count', 0)} "
        f"({100.0 * gate['viable_frac']:.1f}%)",
        f"- recommend `require_positive_net_saving=True`: **{gate['recommend_hard_gate']}**",
        "",
        "### Per-cell",
        "",
        "| rows | hidden | gbps | net_p50_us | net_p95_us | viable |",
        "|---:|---:|---:|---:|---:|:---:|",
    ]
    for c in gate.get("cells", []):
        lines.append(
            f"| {c['rows']} | {c['hidden']} | {c['gbps']} | {c['net_p50']:.3f} | "
            f"{c['net_p95']:.3f} | {'Y' if c['viable'] else 'N'} |"
        )
    lines.extend(
        [
            "",
            "## Codec tax table (p50 pack/unpack/h2d)",
            "",
            "| mode | rows | hidden | pack_us | unpack_us | h2d_us | codec_total | "
            "net@200Gbps | break_even_gbps |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in frame.iterrows():
        be = row.get("break_even_gbps", float("nan"))
        be_s = "inf" if (isinstance(be, float) and math.isinf(be)) else f"{float(be):.3f}"
        net200 = row.get("net_delta_us_200gbps_p50", float("nan"))
        lines.append(
            f"| {row['mode']} | {int(row['rows'])} | {int(row['hidden'])} | "
            f"{row['pack_us']:.3f} | {row['unpack_us']:.3f} | {row['h2d_us']:.3f} | "
            f"{row['codec_total_us_p50']:.3f} | {net200:.3f} | {be_s} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- **Lane-viable**: homo_fp8 net_delta > 0 at 200Gbps with p95 same sign.",
            "- **Hard-gate-on**: if most serving cells (rows in {128,512}, 200–400Gbps) "
            "are net-negative, online low-bit actions should default to blocked.",
            "",
        ]
    )
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden-sizes", default=DEFAULT_HIDDEN)
    parser.add_argument("--rows", default=DEFAULT_ROWS)
    parser.add_argument("--link-gbps", default=DEFAULT_LINK_GBPS)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--serving-rows", default="128,512")
    parser.add_argument("--serving-gbps", default="200,400")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for homogeneous lane codec gate")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    hidden_sizes = [int(v) for v in args.hidden_sizes.split(",")]
    row_counts = [int(v) for v in args.rows.split(",")]
    link_gbps = [float(v) for v in args.link_gbps.split(",")]
    serving_rows = [int(v) for v in args.serving_rows.split(",")]
    serving_gbps = [float(v) for v in args.serving_gbps.split(",")]

    all_rows: list[dict] = []
    correctness: dict[str, dict] = {}
    for hidden in hidden_sizes:
        for rows in row_counts:
            print(f"benchmarking rows={rows}, hidden={hidden}", flush=True)
            measurements, checks = run_shape(
                rows, hidden, args.warmup, args.repeats, link_gbps
            )
            all_rows.extend(measurements)
            correctness[f"rows{rows}_hidden{hidden}"] = checks

    frame = pd.DataFrame(all_rows)
    frame.to_csv(output / "codec_tax_table.csv", index=False)

    # Compact lookup for policy hard-gate (homo codecs + incremental online action).
    lookup = frame[frame["mode"].isin(["homo_fp8", "homo_int4"])][
        [
            "mode",
            "rows",
            "hidden",
            "pack_us",
            "unpack_us",
            "h2d_us",
            "codec_total_us_p50",
            "codec_total_us_p95",
            "break_even_gbps",
            "net_delta_us_200gbps_p50",
            "net_delta_us_200gbps_p95",
        ]
    ].copy()

    # Incremental online question: FP8 wire baseline → INT4 homogeneous lane.
    incr_rows: list[dict] = []
    for (rows_i, hidden_i), grp in frame.groupby(["rows", "hidden"]):
        by_mode = {str(r["mode"]): r for _, r in grp.iterrows()}
        if "homo_fp8" not in by_mode or "homo_int4" not in by_mode:
            continue
        fp8 = by_mode["homo_fp8"]
        int4 = by_mode["homo_int4"]
        fp8_bytes = float(fp8["policy_wire_bytes"])
        int4_bytes = float(int4["policy_wire_bytes"])
        codec = float(int4["pack_us"] + int4["unpack_us"] + int4["h2d_us"])
        be = break_even_gbps(fp8_bytes, int4_bytes, codec)
        rec = {
            "mode": "incr_fp8_to_int4",
            "rows": int(rows_i),
            "hidden": int(hidden_i),
            "pack_us": float(int4["pack_us"]),
            "unpack_us": float(int4["unpack_us"]),
            "h2d_us": float(int4["h2d_us"]),
            "codec_total_us_p50": codec,
            "codec_total_us_p95": float(
                int4["pack_us_p95"] + int4["unpack_us_p95"] + int4["h2d_us_p95"]
            ),
            "break_even_gbps": be,
            "net_delta_us_200gbps_p50": (
                wire_us(fp8_bytes, 200.0) - wire_us(int4_bytes, 200.0) - codec
            ),
            "net_delta_us_200gbps_p95": (
                wire_us(fp8_bytes, 200.0)
                - wire_us(int4_bytes, 200.0)
                - float(int4["pack_us_p95"] + int4["unpack_us_p95"] + int4["h2d_us_p95"])
            ),
        }
        incr_rows.append(rec)
    if incr_rows:
        lookup = pd.concat([lookup, pd.DataFrame(incr_rows)], ignore_index=True)
    lookup.to_csv(output / "codec_tax_lookup.csv", index=False)

    gate = summarize_gate(frame, serving_rows, serving_gbps)
    metadata = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "hidden_sizes": hidden_sizes,
        "rows": row_counts,
        "link_gbps": link_gbps,
        "serving_rows": serving_rows,
        "serving_gbps": serving_gbps,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "fp8_note": (
            "homo_fp8 uses per-vector absmax scale + int8-proxy store at 1 byte/elem "
            f"(fp8_max={FP8_E4M3_MAX}); wire bytes match E4M3+FP32-scale layout"
        ),
        "correctness": correctness,
        "gate_summary": {k: v for k, v in gate.items() if k != "cells"},
        "gate_cells": gate.get("cells", []),
        "evidence_boundary": (
            "single-GPU Triton pack/unpack plus pinned H2D; analytic wire time at "
            "given Gbps; not NCCL/RDMA; no incast, collective headers, or multi-node queueing"
        ),
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    write_report(output, frame, gate, metadata)
    print(frame[["mode", "rows", "hidden", "pack_us", "unpack_us", "h2d_us",
                 "net_delta_us_200gbps_p50", "break_even_gbps"]].to_string(index=False))
    print(f"\nrecommend_hard_gate={gate['recommend_hard_gate']} "
          f"viable_frac={gate['viable_frac']:.3f}")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
