#!/usr/bin/env python3
"""Run native FP8/AWQ expert-map mask checks on one GPU.

Synthetic FP8 is allowed ONLY in this standalone mask-semantics diagnostic.
The production DualFormatMoEMethod still requires real checkpoint artifacts.
A compact route reference expands only selected (token, expert) pairs to topk=1;
it does not use expert_map or zero router weights to suppress expert computation.
No task-quality or serving-performance claim follows from this script.

  python validate_expert_split.py --lowbit-weights runs/qwen3_layer0_awq_marlin.pt \
    --tokens 32 --threshold 16 --out runs/expert_split_validation.json
"""
import argparse
from dataclasses import asdict
import inspect
import json
from pathlib import Path
import statistics


def compact_reference(backend, x, scores, ids, enabled, *, fp8=False):
    import torch
    selected = enabled[ids.long()]
    token, slot = torch.where(selected)
    reference_parts = torch.zeros((*ids.shape, x.shape[1]), device=x.device, dtype=x.dtype)
    if not token.numel():
        return reference_parts.sum(dim=1)
    compact_x = x[token].contiguous()
    compact_scores = scores[token, slot].reshape(-1, 1).contiguous()
    compact_ids = ids[token, slot].reshape(-1, 1).contiguous()
    if fp8:
        from vllm.model_executor.layers.fused_moe.config import fp8_w8a8_moe_quant_config
        from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
        from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input
        t = backend.tensors
        # Fix the input scale to the original full input. Removing tokens in
        # compact routing must not change this tensor-level amax. The second
        # scale remains dynamic over only selected expert activations.
        _, original_input_scale = moe_kernel_quantize_input(x, None, torch.float8_e4m3fn, False)
        qc = fp8_w8a8_moe_quant_config(t['w1_scale'], t['w2_scale'], a1_scale=original_input_scale)
        outputs = fused_experts(compact_x, t['w1'], t['w2'], compact_scores, compact_ids,
                                global_num_experts=t['w1'].shape[0], quant_config=qc)
    else:
        outputs = backend(compact_x, compact_scores, compact_ids)
    reference_parts[token, slot] = outputs
    return reference_parts.sum(dim=1)


def alignment_evidence(ids, expert_map, *, ignore_invalid):
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
    block = 16
    _, experts, padded = moe_align_block_size(ids, block, expert_map.numel(), expert_map,
                                             ignore_invalid_experts=ignore_invalid)
    padded_count = int(padded.item())
    blocks = experts[:padded_count // block].cpu().tolist()
    allowed = set(expert_map[expert_map >= 0].cpu().tolist())
    active_blocks = [expert for expert in blocks if expert >= 0]
    return {'inspection_block_size': block, 'num_tokens_post_padded': padded_count,
        'scheduled_blocks': len(blocks), 'masked_zero_write_blocks': blocks.count(-1),
        'blocks_with_gemm': len(active_blocks), 'gemm_experts': sorted(set(active_blocks)),
        'excluded_experts_absent_from_gemm_blocks': set(active_blocks).issubset(allowed),
        'ignore_invalid_experts': ignore_invalid,
        'scope': 'native alignment probe; actual tuned GEMM block size may differ'}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--lowbit-weights', type=Path, required=True)
    p.add_argument('--fp8-weights', type=Path, help='Omit for synthetic mechanism-only FP8')
    p.add_argument('--tokens', type=int, default=32)
    p.add_argument('--threshold', type=int, default=16)
    p.add_argument('--seed', type=int, default=20261016)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--relative-tolerance', type=float, default=.03,
                   help='Diagnostic relative-L2 tolerance for alternate kernel tilings; not a quality criterion')
    p.add_argument('--no-profile', action='store_true')
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.tokens < 1 or args.threshold < 1 or args.relative_tolerance <= 0:
        raise ValueError('tokens, threshold and relative tolerance must be positive')
    import torch
    from format_backends import ExpertShape, build_backend, _sha256
    from dual_format_moe import NativeExpertSplit, _base_identity, _known_revision
    from validate_nvfp4_layer import metrics
    from bench_expert_formats import event_time, profile_call
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig, FusedMoEParallelConfig, RoutingMethodType
    from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
    from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
    torch.cuda.set_device(torch.device(args.device))
    meta = torch.load(args.lowbit_weights, weights_only=True, map_location='cpu')['metadata']
    if meta['format'] != 'awq':
        raise ValueError('Only calibrated AWQ Marlin is supported by this v2 diagnostic')
    shape = ExpertShape(**meta['shape'])
    if shape.top_k < 2 or shape.experts <= shape.top_k:
        raise ValueError('Diagnostic routing requires at least two routes and E > k')
    fp8 = build_backend('fp8', shape, device=args.device, seed=args.seed, weights=args.fp8_weights)
    low = build_backend('awq', shape, device=args.device, weights=args.lowbit_weights)
    moe = FusedMoEConfig(num_experts=shape.experts, experts_per_token=shape.top_k,
        hidden_dim=shape.d, intermediate_size=shape.intermediate, num_local_experts=shape.experts,
        num_logical_experts=shape.experts, activation=MoEActivation.SILU, device=args.device,
        routing_method=RoutingMethodType.Renormalize, moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        in_dtype=torch.bfloat16, max_num_tokens=max(256, args.tokens * shape.top_k))
    split = NativeExpertSplit(fp8, low, moe)
    torch.manual_seed(args.seed)
    x = torch.randn((args.tokens, shape.d), device=args.device, dtype=torch.bfloat16)
    shared = shape.top_k // 2
    head = torch.arange(shared, device=args.device).expand(args.tokens, -1)
    tail_width = shape.top_k - shared
    tail = (torch.arange(args.tokens * tail_width, device=args.device).reshape(args.tokens, tail_width)
            % (shape.experts - shared)) + shared
    ids = torch.cat((head, tail), dim=1).to(torch.int32)
    scores = torch.arange(1, shape.top_k + 1, device=args.device, dtype=torch.float32)
    scores = (scores / scores.sum()).expand(args.tokens, -1).contiguous()
    counts = torch.bincount(ids.flatten().long(), minlength=shape.experts)
    cases = {'all_fp8': torch.ones(shape.experts, device=args.device, dtype=torch.bool),
             'all_awq': torch.zeros(shape.experts, device=args.device, dtype=torch.bool),
             'threshold_split': counts >= args.threshold,
             'alternating_experts': (torch.arange(shape.experts, device=args.device) % 2) == 0}
    addresses = {name: {key: value.data_ptr() for key, value in backend.tensors.items() if torch.is_tensor(value)}
                 for name, backend in [('fp8', fp8), ('awq', low)]}
    report = {'schema': 1, 'shape': asdict(shape), 'tokens': args.tokens, 'threshold': args.threshold,
        'fp8_weight_source': fp8.metadata['weight_source'], 'quality_claim': False,
        'same_base_claim': False, 'scope': 'expert-map masking and numerical decomposition only',
        'lowbit_artifact_sha256': _sha256(args.lowbit_weights),
        'topk_ids': ids.cpu().tolist(), 'topk_weights': scores.cpu().tolist(), 'counts': counts.cpu().tolist(),
        'resident_weight_and_scale_bytes': sum(b.metadata['resident_weight_and_scale_bytes'] for b in (fp8, low)),
        'resident_expert_identity_bytes': split.resident_metadata_bytes,
        'weight_transfer_bytes_per_step': 0, 'maps_bytes_per_split': shape.experts * 4 * 2,
        'relative_l2_diagnostic_tolerance': args.relative_tolerance,
        'mask_contract': 'valid topk_ids + identity/-1 expert_map; router weights remain unchanged',
        'timing_scope': 'two fused calls plus output sum with already-created expert maps; excludes policy count/host decisions',
        'limitations': ['Synthetic FP8 (when used) has no model-quality meaning.',
            'FP8 second activation quantization scale is recalculated on the selected expert subset.',
            'Masking skips GEMM work, but full-shape sorting/activation/quantization buffers remain.',
            'Compact topk=1 reference uses different GEMM tilings and can differ through floating-point rounding.'],
        'source_sha256': {}, 'cases': {}}
    for function in (TritonExperts.apply, fused_marlin_moe, moe_align_block_size):
        path = inspect.getsourcefile(function)
        report['source_sha256'][function.__module__ + '.' + function.__qualname__] = _sha256(path)
    report['unknown_base_revision_checks'] = {
        'unreported_is_unknown': _base_identity({'base_checkpoint': 'same', 'base_checkpoint_revision': 'unreported_by_publisher'}) is None,
        'explicit_false_is_unknown': _base_identity({'base_checkpoint': 'same', 'base_checkpoint_revision': 'a' * 40, 'exact_base_revision_known': False}) is None,
        'checkpoint_main_is_unknown': not _known_revision('main')}
    all_pass = all(report['unknown_base_revision_checks'].values())
    with torch.inference_mode():
        for name, hot in cases.items():
            fp8_map, low_map = split.maps(hot)
            masked_fp8 = split.fp8_masked(x, scores, ids, fp8_map)
            masked_low = split.low_masked(x, scores, ids, low_map)
            ref_fp8 = compact_reference(fp8, x, scores, ids, hot, fp8=True)
            ref_low = compact_reference(low, x, scores, ids, ~hot)
            total = masked_fp8 + masked_low
            expected = ref_fp8 + ref_low
            fp8_error, low_error = metrics(masked_fp8, ref_fp8), metrics(masked_low, ref_low)
            sum_error = metrics(total, expected)
            fp8_align = alignment_evidence(ids, fp8_map, ignore_invalid=False)
            low_align = alignment_evidence(ids, low_map, ignore_invalid=True)
            flags = [fp8_error['actual_finite'], low_error['actual_finite'],
                     sum_error['actual_finite'], sum_error['reference_finite'],
                     sum_error['relative_l2'] <= args.relative_tolerance,
                     fp8_error['relative_l2'] <= args.relative_tolerance,
                     low_error['relative_l2'] <= args.relative_tolerance,
                     fp8_align['excluded_experts_absent_from_gemm_blocks'],
                     low_align['excluded_experts_absent_from_gemm_blocks']]
            if name == 'all_awq':
                flags.append(bool((masked_fp8 == 0).all()))
            if name == 'all_fp8':
                flags.append(bool((masked_low == 0).all()))
            fn = lambda: split.apply_maps(x, scores, ids, fp8_map, low_map)
            fn()
            timing = [event_time(fn, 5)[0] for _ in range(5)]
            row = {'fp8_error': fp8_error, 'awq_error': low_error, 'sum_error': sum_error,
                'fp8_alignment': fp8_align, 'awq_alignment': low_align,
                'hot_experts': int(hot.sum().item()), 'hot_assignments': int(counts[hot].sum().item()),
                'cold_assignments': int(counts[~hot].sum().item()),
                'latency_ms': timing, 'latency_ms_median': statistics.median(timing),
                'pass': all(flags)}
            if name == 'threshold_split' and not args.no_profile:
                trace_path = args.out.with_suffix('.trace.json')
                trace_path.parent.mkdir(parents=True, exist_ok=True)
                row['profile'] = profile_call(fn, trace_path)
            report['cases'][name] = row
            all_pass = all_pass and row['pass']
    after = {name: {key: value.data_ptr() for key, value in backend.tensors.items() if torch.is_tensor(value)}
             for name, backend in [('fp8', fp8), ('awq', low)]}
    report['weight_storage_addresses_unchanged'] = addresses == after
    all_pass = all_pass and addresses == after
    report['status'] = 'mask_diagnostic_passed' if all_pass else 'mask_diagnostic_failed'
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'cases': {name: {'pass': row['pass'],
          'fp8_relative_l2': row['fp8_error']['relative_l2'], 'awq_relative_l2': row['awq_error']['relative_l2']}
          for name, row in report['cases'].items()}, 'out': str(args.out)}))
    if not all_pass:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
