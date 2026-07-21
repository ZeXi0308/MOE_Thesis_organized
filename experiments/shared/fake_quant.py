from __future__ import annotations

import torch


# FP8 E4M3 max finite normal value (exp=1111, mantissa=110 -> 2^8 * 1.75 = 448).
FP8_E4M3_MAX = 448.0
FP4_E2M1_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def symmetric_quant_dequant(x: torch.Tensor, bits: int, eps: float = 1e-8) -> torch.Tensor:
    """Per-token symmetric fake quantization followed by dequantization."""
    if bits not in (4, 8):
        raise ValueError(f"bits must be 4 or 8, got {bits}")
    qmax = (2 ** (bits - 1)) - 1
    x_float = x.float()
    scale = x_float.abs().amax(dim=-1, keepdim=True).clamp_min(eps) / qmax
    q = torch.round(x_float / scale).clamp(-qmax, qmax)
    return (q * scale).to(dtype=x.dtype)


def fp8_e4m3_quant_dequant(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Per-token FP8 E4M3 fake quantization (quant -> dequant).

    Uses a per-row (per-token) scaling factor = absmax / 448, then casts through
    ``torch.float8_e4m3fn`` so the rounding/dynamic range faithfully matches the
    E4M3 format. This mirrors the per-tensor / per-block scaled FP8 used for MoE
    all-to-all communication by MegaScale-MoE and DeepSeek-V3, at per-token
    granularity (same granularity as the INT4/INT8 proxies above, for a fair
    apples-to-apples comparison).
    """
    orig_dtype = x.dtype
    x_float = x.float()
    scale = x_float.abs().amax(dim=-1, keepdim=True).clamp_min(eps) / FP8_E4M3_MAX
    x_scaled = x_float / scale
    x_fp8 = x_scaled.to(torch.float8_e4m3fn)
    x_deq = x_fp8.to(torch.float32) * scale
    return x_deq.to(dtype=orig_dtype)


def _e2m1_round(x: torch.Tensor) -> torch.Tensor:
    """Round a scaled tensor to finite E2M1 values in [-6, 6]."""
    boundaries = torch.tensor(
        (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0),
        dtype=x.dtype,
        device=x.device,
    )
    levels = torch.tensor(FP4_E2M1_LEVELS, dtype=x.dtype, device=x.device)
    indices = torch.bucketize(x.abs(), boundaries)
    return levels[indices] * x.sign()


def _pad_last_dim(x: torch.Tensor, block_size: int) -> tuple[torch.Tensor, int]:
    hidden = x.shape[-1]
    padding = (-hidden) % block_size
    if padding == 0:
        return x, hidden
    padded = torch.nn.functional.pad(x, (0, padding))
    return padded, hidden


def mxfp4_e2m1_quant_dequant(
    x: torch.Tensor, block_size: int = 32, eps: float = 1e-12
) -> torch.Tensor:
    """MXFP4-style E2M1 with one power-of-two scale per 32 elements."""
    orig_dtype = x.dtype
    values, hidden = _pad_last_dim(x.float(), block_size)
    blocks = values.reshape(*values.shape[:-1], -1, block_size)
    block_amax = blocks.abs().amax(dim=-1, keepdim=True)
    raw_scale = (block_amax / 6.0).clamp_min(eps)
    scale = torch.pow(2.0, torch.ceil(torch.log2(raw_scale)))
    scale = torch.where(block_amax > 0, scale, torch.ones_like(scale))
    dequant = _e2m1_round(blocks / scale) * scale
    return dequant.reshape(*values.shape)[..., :hidden].to(dtype=orig_dtype)


def nvfp4_e2m1_quant_dequant(
    x: torch.Tensor, block_size: int = 16, eps: float = 1e-12
) -> torch.Tensor:
    """NVFP4-style E2M1 with E4M3 block scales and a per-row global scale.

    Communication vectors are treated as independent tensors, so each row has
    one FP32 global scale.  Each 16-element micro-block has an E4M3 scale.
    """
    orig_dtype = x.dtype
    values, hidden = _pad_last_dim(x.float(), block_size)
    rows = values.reshape(-1, values.shape[-1])
    blocks = rows.reshape(rows.shape[0], -1, block_size)
    row_amax = rows.abs().amax(dim=-1, keepdim=True).unsqueeze(-1)
    global_scale = (row_amax / (FP8_E4M3_MAX * 6.0)).clamp_min(eps)
    block_amax = blocks.abs().amax(dim=-1, keepdim=True)
    raw_block_scale = (block_amax / 6.0) / global_scale
    block_scale = raw_block_scale.clamp(0, FP8_E4M3_MAX).to(torch.float8_e4m3fn).float()
    combined_scale = (block_scale * global_scale).clamp_min(eps)
    dequant = _e2m1_round(blocks / combined_scale) * combined_scale
    return dequant.reshape(*values.shape)[..., :hidden].to(dtype=orig_dtype)


def apply_precision(x: torch.Tensor, precision: str) -> torch.Tensor:
    if precision in ("bf16", "fp16", "full"):
        return x
    if precision == "fp8":
        return fp8_e4m3_quant_dequant(x)
    if precision == "int8":
        return symmetric_quant_dequant(x, bits=8)
    if precision == "int4":
        return symmetric_quant_dequant(x, bits=4)
    if precision == "mxfp4":
        return mxfp4_e2m1_quant_dequant(x)
    if precision == "nvfp4":
        return nvfp4_e2m1_quant_dequant(x)
    raise ValueError(f"unknown precision: {precision}")
