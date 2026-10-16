"""Numerical wire reference. E2M1 nearest-even, tensor-global NVFP4 scale."""
import torch

LEVELS = (0., .5, 1., 1.5, 2., 3., 4., 6.)


def e2m1_codes(x):
    a = x.float().abs()
    idx = torch.zeros_like(a, dtype=torch.uint8)
    for i, threshold in enumerate((.25, .75, 1.25, 1.75, 2.5, 3.5, 5.)):
        # Adjacent codes alternate parity; choose the even code at a midpoint.
        idx += ((a > threshold) | ((a == threshold) & bool(i % 2))).to(torch.uint8)
    return idx | ((x < 0).to(torch.uint8) << 3)


def decode_e2m1(codes):
    lut = torch.tensor(LEVELS, device=codes.device, dtype=torch.float32)
    return lut[(codes & 7).long()] * torch.where((codes & 8) != 0, -1., 1.)


def fp8_quant(x):
    s = (x.float().abs().amax(-1) / 448.).clamp_min(1e-12)
    return (x.float() / s[:, None]).clamp(-448,448).to(torch.float8_e4m3fn), s


def nvfp4_quant(x, global_scale=None):
    if x.ndim != 2 or x.shape[1] % 16:
        raise ValueError('Expected [rows, d], d divisible by 16')
    v = x.float().reshape(x.shape[0], -1, 16)
    g = (x.float().abs().amax() / 2688.).clamp_min(1e-12).reshape(1) if global_scale is None else global_scale
    s = (v.abs().amax(-1) / (6 * g)).clamp(0,448).to(torch.float8_e4m3fn)
    combined = s.float() * g
    scaled = torch.where(combined[...,None] > 0, v / combined[...,None].clamp_min(1e-38), 0.)
    c = e2m1_codes(scaled).reshape_as(x)
    return c[:, 0::2] | (c[:, 1::2] << 4), s, g


def nvfp4_dequant(packed, scales, global_scale, dtype=torch.bfloat16):
    c = torch.stack((packed & 15, packed >> 4), dim=-1).reshape(packed.shape[0], -1)
    x = decode_e2m1(c).reshape(packed.shape[0], -1, 16)
    return (x * (scales.float() * global_scale)[...,None]).flatten(1).to(dtype)
