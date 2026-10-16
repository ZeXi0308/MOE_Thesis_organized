#!/usr/bin/env python3
"""Numerical diagnostic for one calibrated NVFP4 CUTLASS MoE artifact.

The reference independently decodes packed E2M1, reverses block-scale swizzling,
recovers weight global scales, and computes projections with Torch FP32 matmul.
It does not call vLLM's dequantization/emulation code. W4A16 and two W4A4 stage
rounding variants are reported separately; this is not a model quality test.

  python validate_nvfp4_layer.py --weights runs/layer0.pt --tokens 2 \
    --checkpoint-dir /data/nvfp4 --out runs/layer0_numeric.json
"""
import argparse
from contextlib import ExitStack
from dataclasses import asdict
import hashlib
import json
from pathlib import Path


def reverse_blockscale_swizzle(scales, rows, blocks):
    """Inverse of [E,M/128,4,32,K/4,4] -> [E,M/128,K/4,32,4,4]."""
    e, padded_rows, padded_blocks = scales.shape
    if padded_rows % 128 or padded_blocks % 4:
        raise ValueError('Invalid swizzled scale extents')
    linear = scales.reshape(e, padded_rows // 128, padded_blocks // 4, 32, 4, 4)
    linear = linear.permute(0, 1, 4, 3, 2, 5).contiguous()
    return linear.reshape(e, padded_rows, padded_blocks)[:, :rows, :blocks]


def decode_e2m1(packed):
    import torch
    table = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], device=packed.device, dtype=torch.float32)
    nibbles = torch.stack((packed & 15, packed >> 4), dim=-1).reshape(*packed.shape[:-1], -1)
    magnitudes = table[(nibbles & 7).long()]
    return torch.where((nibbles & 8) != 0, -magnitudes, magnitudes)


def decode_weight(packed, linear_scales, global_scale):
    values = decode_e2m1(packed)
    return values * linear_scales.float().repeat_interleave(16, dim=-1) * global_scale.float()


def round_e2m1_rne(values):
    """Nearest finite E2M1; exact midpoint ties choose an even magnitude code."""
    import torch
    boundaries = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.0],
                              device=values.device, dtype=torch.float32)
    table = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], device=values.device, dtype=torch.float32)
    absolute = values.abs().contiguous()
    indices = torch.bucketize(absolute, boundaries, right=False)
    ties = (absolute == boundaries[indices.clamp(max=6)]) & (indices < 7)
    indices = indices + (ties & ((indices % 2) == 1)).to(indices.dtype)
    return torch.copysign(table[indices], values)


def quant_dequant_activation(values, global_multiplier):
    """Independent block16 NVFP4 fake quantization with E4M3 block scales.

    block_scale=E4M3(clamp(max_abs/6 * global_multiplier, 0, 448));
    q=E2M1_RNE(values * global_multiplier / block_scale).
    """
    import torch
    if values.shape[-1] % 16:
        raise ValueError('NVFP4 activation width must be divisible by 16')
    blocks = values.float().reshape(*values.shape[:-1], -1, 16)
    scales = (blocks.abs().amax(dim=-1, keepdim=True) / 6 * global_multiplier).clamp(0, 448)
    scales = scales.to(torch.float8_e4m3fn).float()
    multiplier = torch.where(scales > 0, global_multiplier / scales, 0)
    quantized = round_e2m1_rne((blocks * multiplier).clamp(-6, 6))
    return (quantized * scales / global_multiplier).reshape_as(values)


def metrics(actual, reference):
    import torch
    a, r = actual.float().reshape(-1), reference.float().reshape(-1)
    delta = a - r
    return {'relative_l2': float(torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(r).clamp_min(1e-30)),
        'max_abs_error': float(delta.abs().max()), 'mean_abs_error': float(delta.abs().mean()),
        'reference_rms': float(r.square().mean().sqrt()),
        'actual_rms': float(a.square().mean().sqrt()),
        'cosine_similarity': float(torch.nn.functional.cosine_similarity(a, r, dim=0)),
        'actual_finite': bool(torch.isfinite(a).all()), 'reference_finite': bool(torch.isfinite(r).all())}


def check_source_checkpoint(path, metadata, shape, tensors, scales1, scales2, expert_ids):
    """Compare selected expert tensors against raw checkpoint, not another exporter."""
    import torch
    from safetensors import safe_open
    path = Path(path)
    index = path / 'model.safetensors.index.json'
    if index.exists():
        mapping = json.loads(index.read_text())['weight_map']
    else:
        mapping = {}
        for shard in path.glob('*.safetensors'):
            with safe_open(shard, framework='pt', device='cpu') as handle:
                mapping.update({key: shard.name for key in handle.keys()})
    rows = []
    with ExitStack() as stack:
        handles = {}
        def tensor(expert, projection, field):
            key = f"model.layers.{metadata['layer']}.mlp.experts.{expert}.{projection}.{field}"
            shard = mapping[key]
            if shard not in handles:
                handles[shard] = stack.enter_context(safe_open(path / shard, framework='pt', device='cpu'))
            return handles[shard].get_tensor(key).to(tensors['w1'].device)
        for expert in expert_ids:
            w1 = torch.cat([tensor(expert, p, 'weight') for p in ('gate_proj', 'up_proj')])
            w2 = tensor(expert, 'down_proj', 'weight')
            s1 = torch.cat([tensor(expert, p, 'weight_scale') for p in ('gate_proj', 'up_proj')])
            s2 = tensor(expert, 'down_proj', 'weight_scale')
            a1 = torch.stack([tensor(expert, p, 'input_scale').reshape(()) for p in ('gate_proj', 'up_proj')]).max()
            a2 = tensor(expert, 'down_proj', 'input_scale').reshape(())
            g1 = tensor(expert, 'gate_proj', 'weight_scale_2').reshape(())
            g2 = tensor(expert, 'down_proj', 'weight_scale_2').reshape(())
            checks = {
                'gate_up_weight_order_exact': bool(torch.equal(w1, tensors['w1'][expert])),
                'down_weight_exact': bool(torch.equal(w2, tensors['w2'][expert])),
                'gate_up_linear_blockscales_exact': bool(torch.equal(s1.view(torch.uint8), scales1[expert].contiguous().view(torch.uint8))),
                'down_linear_blockscales_exact': bool(torch.equal(s2.view(torch.uint8), scales2[expert].contiguous().view(torch.uint8))),
                'a1_gscale_is_inverse_input_scale': bool(torch.allclose(tensors['a1_gscale'][expert], 1 / a1)),
                'a2_gscale_is_inverse_input_scale': bool(torch.allclose(tensors['a2_gscale'][expert], 1 / a2)),
                'g1_alpha_is_weight_times_input_scale': bool(torch.allclose(tensors['g1_alphas'][expert], g1 * a1)),
                'g2_alpha_is_weight_times_input_scale': bool(torch.allclose(tensors['g2_alphas'][expert], g2 * a2)),
            }
            rows.append({'expert': expert, **checks, 'all_pass': all(checks.values())})
    return {'scope': 'selected experts only', 'experts': rows, 'all_pass': all(row['all_pass'] for row in rows)}


def reference_outputs(x, ids, scores, tensors, scales1, scales2, shape):
    import torch
    b, topk = ids.shape
    names = ('w4a16_fp32', 'w4a4_fp32_silu', 'w4a4_bf16_silu',
             'negative_swapped_gate_up', 'negative_omit_unfusing_activation_scale')
    outputs = {name: torch.zeros((b, topk, shape.d), device=x.device, dtype=torch.float32) for name in names}
    # Decode only selected experts and reuse across all diagnostic branches.
    for expert in sorted(set(ids.cpu().flatten().tolist())):
        token_idx, slot_idx = torch.where(ids == expert)
        input_rows = x[token_idx].float()
        wglobal1 = tensors['g1_alphas'][expert] * tensors['a1_gscale'][expert]
        wglobal2 = tensors['g2_alphas'][expert] * tensors['a2_gscale'][expert]
        w1 = decode_weight(tensors['w1'][expert], scales1[expert], wglobal1)
        w2 = decode_weight(tensors['w2'][expert], scales2[expert], wglobal2)
        a1gs, a2gs = tensors['a1_gscale'][expert], tensors['a2_gscale'][expert]
        aq = quant_dequant_activation(input_rows, a1gs)
        for name in names:
            if name == 'negative_omit_unfusing_activation_scale':
                # Intentional wrong interpretation: fused alpha treated as weight scale.
                first = aq @ (w1 / a1gs).T
                second_weight = w2 / a2gs
            else:
                first = (input_rows if name == 'w4a16_fp32' else aq) @ w1.T
                second_weight = w2
            if name != 'w4a16_fp32':
                first = first.to(torch.bfloat16).float()
            gate, up = first.split(shape.intermediate, dim=-1)
            if name == 'negative_swapped_gate_up':
                gate, up = up, gate
            intermediate = torch.nn.functional.silu(gate) * up
            if name == 'w4a4_bf16_silu':
                intermediate = intermediate.to(torch.bfloat16).float()
            if name != 'w4a16_fp32':
                intermediate = quant_dequant_activation(intermediate, a2gs)
            projected = intermediate @ second_weight.T
            if name != 'w4a16_fp32':
                projected = projected.to(torch.bfloat16).float()
            outputs[name][token_idx, slot_idx] = projected
    final = {}
    for name, value in outputs.items():
        if name == 'w4a16_fp32':
            final[name] = (value * scores.unsqueeze(-1)).sum(dim=1)
        else:
            # Matches final run_cutlass_moe_fp4 BF16 multiply and BF16 output sum.
            weighted = value.to(torch.bfloat16) * scores.to(torch.bfloat16).unsqueeze(-1)
            final[name] = weighted.sum(dim=1).float()
    return final


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--checkpoint-dir', type=Path, help='Optional raw checkpoint for independent export-layout checks')
    p.add_argument('--tokens', type=int, default=2)
    p.add_argument('--input-std', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=20261016)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--max-relative-l2', type=float, help='Optional caller-chosen numeric diagnostic limit; no default quality gate')
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.tokens < 1 or args.tokens > 32 or args.input_std <= 0:
        raise ValueError('Use 1..32 tokens and positive input std for this small diagnostic')
    import torch
    from format_backends import ExpertShape, build_backend, _sha256
    torch.cuda.set_device(torch.device(args.device))
    torch.manual_seed(args.seed)
    # Full FP32 reference: disable TF32 and reduced-precision FP32 matmul modes.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    metadata = torch.load(args.weights, map_location='cpu', weights_only=True)['metadata']
    shape = ExpertShape(**metadata['shape'])
    backend = build_backend('nvfp4', shape, device=args.device, weights=args.weights,
                            nvfp4_backend='cutlass', max_tokens=args.tokens)
    tensors = backend.tensors
    ids = ((torch.arange(args.tokens, device=args.device).unsqueeze(1) * (shape.top_k - 1)
            + torch.arange(shape.top_k, device=args.device)) % shape.experts).to(torch.int32)
    unnormalized = torch.arange(1, shape.top_k + 1, device=args.device, dtype=torch.float32)
    scores = (unnormalized / unnormalized.sum()).expand(args.tokens, -1).contiguous()
    x = (torch.randn((args.tokens, shape.d), device=args.device) * args.input_std).to(torch.bfloat16)
    scales1 = reverse_blockscale_swizzle(tensors['w1_scale'], 2 * shape.intermediate, shape.d // 16)
    scales2 = reverse_blockscale_swizzle(tensors['w2_scale'], shape.d, shape.intermediate // 16)
    report = {'schema': 1, 'artifact_sha256': _sha256(args.weights),
        'checkpoint': metadata['checkpoint'], 'checkpoint_revision': metadata['checkpoint_revision'],
        'layer': metadata['layer'], 'shape': asdict(shape), 'tokens': args.tokens,
        'seed': args.seed, 'input_std': args.input_std,
        'topk_ids': ids.cpu().tolist(), 'topk_weights': scores.cpu().tolist(),
        'reference_compute': 'independent E2M1 decode and Torch FP32 matmul; TF32 disabled',
        'independent_repeats': False, 'quality_claim': False,
        'notes': ['W4A16 reference omits activation quantization; its error is expected and is not a kernel correctness gate.',
                  'Two W4A4 references expose whether the fused SiLU+Mul result is rounded to BF16 before FP4 quantization.',
                  'FP32 reference is not bitwise equivalent to CUTLASS reduction/activation arithmetic.',
                  'Synthetic inputs and selected experts do not establish model task quality.']}
    with torch.inference_mode():
        if args.checkpoint_dir:
            report['source_layout_checks'] = check_source_checkpoint(args.checkpoint_dir,
                metadata, shape, tensors, scales1, scales2, sorted(set(ids.cpu().flatten().tolist())))
        first = backend(x, scores, ids)
        second = backend(x, scores, ids)
        torch.cuda.synchronize()
        refs = reference_outputs(x, ids, scores, tensors, scales1, scales2, shape)
        report['reference_errors'] = {name: metrics(first, reference) for name, reference in refs.items()}
        report['same_input_repeat_error'] = metrics(first, second)
        report['output_sample'] = first[0, :16].float().cpu().tolist()
        report['finite_output'] = bool(torch.isfinite(first).all())
        report['activation_reference_difference'] = metrics(refs['w4a4_fp32_silu'], refs['w4a4_bf16_silu'])
    source_ok = report.get('source_layout_checks', {}).get('all_pass', True)
    report['status'] = 'diagnostic_complete' if source_ok and report['finite_output'] else 'failed_structural_or_finite_check'
    if args.max_relative_l2 is not None:
        error = report['reference_errors']['w4a4_fp32_silu']['relative_l2']
        report['caller_relative_l2_limit'] = args.max_relative_l2
        report['caller_numeric_limit_pass'] = error <= args.max_relative_l2
        if not report['caller_numeric_limit_pass']:
            report['status'] = 'failed_caller_numeric_limit'
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'errors': report['reference_errors'], 'out': str(args.out)}))
    if report['status'] != 'diagnostic_complete':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
