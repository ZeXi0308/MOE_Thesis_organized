#!/usr/bin/env python3
"""Export one calibrated Qwen3 compressed-tensors symmetric AWQ layer to Marlin.

This is a packing/layout conversion, never a quantizer. It reads only selected
expert tensors, checks source shapes and signed-int4 unpacking, then invokes
vLLM's native compressed-tensors WNA16->Marlin conversion. The optional numeric
check compares actual Marlin output with a reference decoded from source
checkpoint tensors; it is not a model quality test.

  python export_awq_artifact.py --checkpoint-dir /data/qwen3_awq \
    --layer 0 --base-revision unreported_by_publisher \
    --out runs/qwen3_layer0_awq_marlin.pt
"""
import argparse
from contextlib import ExitStack
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path

from format_backends import ExpertShape

PROJECTIONS = ('gate_proj', 'up_proj', 'down_proj')
REQUIRED_FIELDS = ('weight_packed', 'weight_scale', 'weight_shape')


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            value.update(chunk)
    return value.hexdigest()


def shape_from_config(config):
    text = config.get('text_config', config)
    shape = ExpertShape(int(text['hidden_size']), int(text['moe_intermediate_size']),
                        int(text['num_experts']), int(text['num_experts_per_tok']))
    shape.validate()
    if shape.d % 128 or shape.intermediate % 128:
        raise ValueError('This group128 exporter requires dimensions divisible by 128')
    return shape


def quantization_from_config(config):
    qc = config['quantization_config']
    if qc.get('quant_method') != 'compressed-tensors' or qc.get('format') != 'pack-quantized':
        raise ValueError('Expected compressed-tensors pack-quantized checkpoint, not legacy AWQ packing')
    groups = qc.get('config_groups', {})
    if len(groups) != 1:
        raise ValueError('This pinned-checkpoint exporter expects exactly one quantization config group')
    group_name, group = next(iter(groups.items()))
    weights = group['weights']
    expected = {'num_bits': 4, 'type': 'int', 'symmetric': True, 'group_size': 128, 'strategy': 'group'}
    for key, value in expected.items():
        if weights.get(key) != value:
            raise ValueError(f'Expected weights.{key}={value!r}, got {weights.get(key)!r}')
    if group.get('input_activations') is not None:
        raise ValueError('AWQ W4A16 arm must have no quantized input activations')
    if weights.get('actorder') not in (None, 'static', 'weight'):
        raise ValueError('Runtime group act-order is not supported by this export path')
    if 'Linear' not in group.get('targets', []):
        raise ValueError('Expected quantization group targeting Linear modules')
    if any('.experts.' in str(pattern) for pattern in qc.get('ignore', [])):
        raise ValueError('Some experts are excluded from quantization; a homogeneous artifact is invalid')
    return group_name, weights


def find_tensors(directory, layer, experts, index=None):
    directory = Path(directory)
    index = Path(index) if index else directory / 'model.safetensors.index.json'
    if index.is_file():
        mapping = json.loads(index.read_text())['weight_map']
    else:
        from safetensors import safe_open
        mapping = {}
        for shard in sorted(directory.glob('*.safetensors')):
            with safe_open(shard, framework='pt', device='cpu') as handle:
                mapping.update({name: shard.name for name in handle.keys()})
    required = [f'model.layers.{layer}.mlp.experts.{expert}.{projection}.{field}'
                for expert in range(experts) for projection in PROJECTIONS for field in REQUIRED_FIELDS]
    missing = [key for key in required if key not in mapping]
    if missing:
        raise ValueError(f'Missing {len(missing)} source tensors; first: {missing[:3]}')
    # Weight order has been fixed by AWQ. Retain any published g_idx as provenance
    # and verify that it does not request an unhandled runtime permutation.
    optional = [key[:-len('weight_packed')] + 'weight_g_idx'
                for key in required if key.endswith('weight_packed')]
    selected = required + [key for key in optional if key in mapping]
    shards = sorted({mapping[key] for key in selected})
    missing_shards = [name for name in shards if not (directory / name).is_file()]
    return mapping, selected, shards, missing_shards


def source_identity(directory, checkpoint, revision):
    manifest = Path(directory) / 'subset_source.json'
    if manifest.is_file():
        data = json.loads(manifest.read_text())
        if data.get('checkpoint') != checkpoint or data.get('revision') != revision:
            raise ValueError('Local subset provenance differs from the pinned AWQ catalog entry')
        return {'evidence': str(manifest), 'manifest_sha256': sha256(manifest),
                'checkpoint_and_revision_match': True,
                'source_download_complete': data.get('source_download_complete'),
                'completed_source_verification': data.get('completed_source_verification', 'unreported')}
    resolved = Path(directory).resolve()
    if resolved.parent.name == 'snapshots':
        if resolved.name != revision:
            raise ValueError('HF snapshot directory revision differs from AWQ catalog')
        return {'evidence': 'HF snapshot path', 'checkpoint_and_revision_match': 'revision_path_only'}
    return {'evidence': 'caller-provided directory and catalog identity; no local provenance manifest',
            'checkpoint_and_revision_match': 'not_independently_verified'}


def unpack_symmetric_int4(packed, logical_input_width):
    """Independent compressed-tensors int4 decode: LSB-first unsigned nibble - 8."""
    import torch
    shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=packed.device)
    values = ((packed.unsqueeze(-1) >> shifts) & 15).reshape(packed.shape[0], -1)
    return (values[:, :logical_input_width] - 8).to(torch.int8)


def load_source(directory, mapping, layer, shape):
    import torch
    from safetensors import safe_open
    from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32
    e, k, n = shape.experts, shape.d, shape.intermediate
    arrays = {'w13': torch.empty((e, k // 8, 2 * n), dtype=torch.int32),
              'w2': torch.empty((e, n // 8, k), dtype=torch.int32),
              's13': torch.empty((e, k // 128, 2 * n), dtype=torch.bfloat16),
              's2': torch.empty((e, n // 128, k), dtype=torch.bfloat16)}
    digest = hashlib.sha256()
    checks = {'all_source_shapes_match': True, 'all_transpose_and_gate_up_copies_exact': True,
              'group128_scales_finite_nonnegative': True, 'source_scale_dtype_counts': {},
              'scale_cast_max_absolute_error': 0.0, 'source_signed_unpack_checks': [],
              'g_idx_tensors_seen': 0, 'weight_shape_tensors_checked': 0}
    with ExitStack() as stack:
        handles = {}
        def read(key):
            shard = mapping[key]
            if shard not in handles:
                handles[shard] = stack.enter_context(safe_open(Path(directory) / shard, framework='pt', device='cpu'))
            value = handles[shard].get_tensor(key)
            digest.update(key.encode())
            digest.update(str((str(value.dtype), list(value.shape))).encode())
            digest.update(value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
            return value
        for expert in range(e):
            for projection_idx, projection in enumerate(PROJECTIONS):
                prefix = f'model.layers.{layer}.mlp.experts.{expert}.{projection}'
                out_dim, in_dim = (k, n) if projection == 'down_proj' else (n, k)
                packed = read(prefix + '.weight_packed')
                scale = read(prefix + '.weight_scale')
                logical_shape = read(prefix + '.weight_shape')
                if packed.dtype != torch.int32 or tuple(packed.shape) != (out_dim, in_dim // 8):
                    raise ValueError(f'{prefix}: expected int32 packed [{out_dim}, {in_dim // 8}], got {packed.dtype} {tuple(packed.shape)}')
                if logical_shape.numel() != 2 or logical_shape.flatten().tolist() != [out_dim, in_dim]:
                    raise ValueError(f'{prefix}: unexpected logical weight_shape {logical_shape.tolist()}')
                checks['weight_shape_tensors_checked'] += 1
                if tuple(scale.shape) != (out_dim, in_dim // 128) or not scale.is_floating_point():
                    raise ValueError(f'{prefix}: invalid group128 scale {scale.dtype} {tuple(scale.shape)}')
                if not torch.isfinite(scale.float()).all() or (scale.float() < 0).any():
                    raise ValueError(f'{prefix}: nonfinite or negative weight scale')
                dtype = str(scale.dtype)
                checks['source_scale_dtype_counts'][dtype] = checks['source_scale_dtype_counts'].get(dtype, 0) + 1
                scale_bf16 = scale.to(torch.bfloat16)
                checks['scale_cast_max_absolute_error'] = max(checks['scale_cast_max_absolute_error'],
                    float((scale_bf16.float() - scale.float()).abs().max()))
                if projection_idx == 2:
                    weight_dest, scale_dest = arrays['w2'][expert], arrays['s2'][expert]
                else:
                    weight_dest = arrays['w13'][expert, :, projection_idx * n:(projection_idx + 1) * n]
                    scale_dest = arrays['s13'][expert, :, projection_idx * n:(projection_idx + 1) * n]
                # This is the transpose performed by RoutedExperts.weight_loader.
                weight_dest.copy_(packed.T)
                scale_dest.copy_(scale_bf16.T)
                if not torch.equal(weight_dest.T, packed) or not torch.equal(scale_dest.T, scale_bf16):
                    raise RuntimeError(f'{prefix}: source-to-vLLM layout copy failed')
                gkey = prefix + '.weight_g_idx'
                if gkey in mapping:
                    g_idx = read(gkey)
                    checks['g_idx_tensors_seen'] += 1
                    canonical = torch.arange(in_dim, dtype=torch.int32) // 128
                    if g_idx.numel() and not torch.equal(g_idx.flatten().to(torch.int32), canonical):
                        raise ValueError(f'{prefix}: unexpected noncanonical source g_idx')
                if expert in (0, e - 1):
                    independent = unpack_symmetric_int4(packed, in_dim)
                    official = unpack_from_int32(packed, num_bits=4,
                                                 shape=torch.Size((out_dim, in_dim)), packed_dim=1)
                    exact = bool(torch.equal(independent, official))
                    checks['source_signed_unpack_checks'].append({'expert': expert,
                        'projection': projection, 'matches_installed_compressed_tensors_unpack': exact,
                        'integer_min': int(independent.min()), 'integer_max': int(independent.max())})
                    if not exact:
                        raise RuntimeError('Independent signed-int4 decoder disagrees with installed checkpoint format')
    checks['selected_tensor_payload_sha256'] = digest.hexdigest()
    checks['hash_order'] = 'expert, gate/up/down, packed/scale/shape/optional-g_idx'
    checks['official_unpack_source_sha256'] = sha256(inspect.getsourcefile(unpack_from_int32))
    return arrays, checks


def convert_to_marlin(arrays, shape, weight_quantization, device):
    import torch
    from compressed_tensors.quantization import QuantizationArgs
    from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import (
        WNA16MoEBackend, convert_to_wna16_moe_kernel_format)
    layer = torch.nn.Module()
    layer.intermediate_size_per_partition = shape.intermediate
    layer.num_groups_w13 = shape.d // 128
    layer.num_groups_w2 = shape.intermediate // 128
    qargs = QuantizationArgs.model_validate(weight_quantization)
    # No act-order at runtime; native conversion resets these to E x 0.
    g13 = torch.zeros((shape.experts, shape.d), device=device, dtype=torch.int32)
    g2 = torch.zeros((shape.experts, shape.intermediate), device=device, dtype=torch.int32)
    gpu = {key: value.to(device) for key, value in arrays.items()}
    values = convert_to_wna16_moe_kernel_format(backend=WNA16MoEBackend.MARLIN,
        layer=layer, quant_config=qargs, input_dtype=None,
        w13=gpu['w13'], w2=gpu['w2'], w13_scale=gpu['s13'], w2_scale=gpu['s2'],
        w13_g_idx=g13, w2_g_idx=g2, w13_qzeros=None, w2_qzeros=None)
    if values is None:
        raise RuntimeError('Native Marlin conversion unexpectedly returned None')
    keys = ('w1', 'w2', 'w1_scale', 'w2_scale', 'g_idx1', 'g_idx2', 'sort_indices1',
            'sort_indices2', 'w1_zeros', 'w2_zeros', 'input_global_scale1',
            'input_global_scale2', 'bias1', 'bias2')
    tensors = {key: value.detach().cpu().contiguous() for key, value in zip(keys, values) if value is not None}
    if any(key in tensors for key in ('w1_zeros', 'w2_zeros', 'input_global_scale1', 'input_global_scale2')):
        raise RuntimeError('Symmetric W4A16 conversion unexpectedly produced zero points or quantized-activation globals')
    expected = {'w1': (shape.experts, shape.d // 16, 4 * shape.intermediate),
                'w2': (shape.experts, shape.intermediate // 16, 2 * shape.d)}
    for key, dims in expected.items():
        if tuple(tensors[key].shape) != dims:
            raise ValueError(f'Unexpected native {key} shape {tuple(tensors[key].shape)}, expected {dims}')
    method_file = inspect.getsourcefile(convert_to_wna16_moe_kernel_format)
    return tensors, {'entrypoint': convert_to_wna16_moe_kernel_format.__module__ + '.' +
                     convert_to_wna16_moe_kernel_format.__name__,
                     'source_file': method_file, 'source_file_sha256': sha256(method_file),
                     'input_dtype': None, 'quant_type': 'uint4b8', 'zero_points': False}


def source_reference_check(artifact, arrays, shape, tokens, seed, device):
    import torch
    from format_backends import build_backend
    from validate_nvfp4_layer import metrics
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    backend = build_backend('awq', shape, device=device, weights=artifact)
    ids = ((torch.arange(tokens, device=device)[:, None] * (shape.top_k - 1)
            + torch.arange(shape.top_k, device=device)) % shape.experts).to(torch.int32)
    scores = torch.arange(1, shape.top_k + 1, device=device, dtype=torch.float32)
    scores = (scores / scores.sum()).expand(tokens, -1).contiguous()
    x = torch.randn((tokens, shape.d), device=device, dtype=torch.bfloat16)
    fp32_parts = torch.zeros((tokens, shape.top_k, shape.d), device=device, dtype=torch.float32)
    bf16_parts = torch.zeros_like(fp32_parts)
    swapped_parts = torch.zeros_like(fp32_parts)
    with torch.inference_mode():
        actual = backend(x, scores, ids)
        repeat = backend(x, scores, ids)
        for expert in sorted(set(ids.cpu().flatten().tolist())):
            token_idx, slot_idx = torch.where(ids == expert)
            source_w13 = arrays['w13'][expert].T.contiguous().to(device)
            source_w2 = arrays['w2'][expert].T.contiguous().to(device)
            w13 = unpack_symmetric_int4(source_w13, shape.d).float()
            w2 = unpack_symmetric_int4(source_w2, shape.intermediate).float()
            w13 *= arrays['s13'][expert].T.to(device).float().repeat_interleave(128, dim=1)
            w2 *= arrays['s2'][expert].T.to(device).float().repeat_interleave(128, dim=1)
            hidden = x[token_idx].float() @ w13.T
            gate, up = hidden.split(shape.intermediate, dim=-1)
            fp32_parts[token_idx, slot_idx] = (torch.nn.functional.silu(gate) * up) @ w2.T
            gate, up = hidden.to(torch.bfloat16).float().split(shape.intermediate, dim=-1)
            intermediate = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16).float()
            result = intermediate @ w2.T
            # Marlin applies router weight within the down-projection GEMM.
            bf16_parts[token_idx, slot_idx] = (result * scores[token_idx, slot_idx, None]).to(torch.bfloat16).float()
            swapped = (torch.nn.functional.silu(up) * gate).to(torch.bfloat16).float() @ w2.T
            swapped_parts[token_idx, slot_idx] = (swapped * scores[token_idx, slot_idx, None]).to(torch.bfloat16).float()
        fp32 = (fp32_parts * scores.unsqueeze(-1)).sum(dim=1)
        bf16 = bf16_parts.to(torch.bfloat16).sum(dim=1)
        swapped = swapped_parts.to(torch.bfloat16).sum(dim=1)
        torch.cuda.synchronize()
    return {'status': 'numerical_diagnostic_complete', 'tokens': tokens, 'seed': seed,
        'topk_ids': ids.cpu().tolist(), 'topk_weights': scores.cpu().tolist(),
        'reference_source': 'independent signed-int4 decode of original checkpoint tensors before Marlin repack',
        'torch_fp32_reference_error': metrics(actual, fp32),
        'torch_bf16_stage_reference_error': metrics(actual, bf16),
        'negative_swapped_gate_up_error': metrics(actual, swapped),
        'same_input_repeat_error': metrics(actual, repeat),
        'finite_output': bool(torch.isfinite(actual).all()), 'quality_claim': False,
        'note': 'No bitwise equivalence or task-quality claim; BF16 stage and GEMM reduction rounding may differ.'}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint-dir', type=Path, required=True)
    p.add_argument('--catalog', type=Path, default=Path(__file__).resolve().parents[1] / 'configs/models.json')
    p.add_argument('--index', type=Path)
    p.add_argument('--layer', type=int, default=0)
    p.add_argument('--base-revision', default='unreported_by_publisher')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--validate-tokens', type=int, default=2, help='0 skips numeric runtime check; default 2')
    p.add_argument('--seed', type=int, default=20261016)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    if args.layer < 0 or not 0 <= args.validate_tokens <= 32:
        raise ValueError('Layer must be nonnegative and validate-tokens must be 0..32')
    if args.base_revision in ('main', 'master'):
        raise ValueError('Do not substitute current base HEAD for publisher provenance')
    catalog = json.loads(args.catalog.read_text())['models']['qwen3']
    entry = catalog['formats']['awq']
    config_path = args.checkpoint_dir / 'config.json'
    config = json.loads(config_path.read_text())
    shape = shape_from_config(config)
    expected_shape = {key: catalog['shape'][key] for key in ('d', 'intermediate', 'experts', 'top_k')}
    if asdict(shape) != expected_shape:
        raise ValueError('Checkpoint shape differs from the pinned Qwen3 catalog entry')
    group_name, weight_quantization = quantization_from_config(config)
    mapping, selected, shards, missing = find_tensors(args.checkpoint_dir, args.layer, shape.experts, args.index)
    plan = {'checkpoint': entry['repo_id'], 'checkpoint_revision': entry['revision'], 'revision': entry['revision'],
        'base_checkpoint': entry['base_model'], 'base_checkpoint_revision': args.base_revision,
        'exact_base_revision_known': args.base_revision != 'unreported_by_publisher',
        'base_revision_evidence': 'publisher provenance required; current base HEAD is not source provenance',
        'shape': asdict(shape), 'layer': args.layer, 'required_tensor_count': len(selected),
        'required_shards': shards, 'missing_shards': missing, 'config_sha256': sha256(config_path),
        'quantization_config_group': group_name, 'source_weight_quantization': weight_quantization,
        'calibration_method': 'AWQ (publisher-documented calibrated checkpoint; no new quantization)',
        'publisher_calibration': entry['calibration'], 'publisher_evidence': entry['sources'],
        'source_identity': source_identity(args.checkpoint_dir, entry['repo_id'], entry['revision'])}
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    if missing:
        raise FileNotFoundError(f'Missing checkpoint shards: {missing}')
    import torch
    import vllm
    torch.cuda.set_device(torch.device(args.device))
    arrays, source_checks = load_source(args.checkpoint_dir, mapping, args.layer, shape)
    with torch.inference_mode():
        tensors, conversion = convert_to_marlin(arrays, shape, weight_quantization, args.device)
    metadata = {**plan, 'schema': 1, 'format': 'awq', 'layout': 'vllm_marlin_processed',
        'quant_type': 'uint4b8', 'symmetric': True, 'group_size': 128,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'torch_version': str(torch.__version__), 'vllm_version': str(vllm.__version__),
        'compressed_tensors_version': importlib.metadata.version('compressed-tensors'),
        'weight_source': 'calibrated_checkpoint', 'quality_claim': False,
        'source_layout_checks': source_checks, 'conversion': conversion,
        'tensor_manifest': {key: {'shape': list(value.shape), 'dtype': str(value.dtype),
            'bytes': value.numel() * value.element_size()} for key, value in tensors.items()}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temp = args.out.with_suffix(args.out.suffix + '.tmp')
    torch.save({'metadata': metadata, 'tensors': tensors}, temp)
    temp.replace(args.out)
    sidecar = args.out.with_suffix(args.out.suffix + '.json')
    summary = {**metadata, 'artifact_sha256': sha256(args.out),
               'numeric_validation': {'status': 'not_run'}}
    sidecar.write_text(json.dumps(summary, indent=2) + '\n')
    if args.validate_tokens:
        diagnostic = source_reference_check(args.out, arrays, shape, args.validate_tokens, args.seed, args.device)
        summary['numeric_validation'] = diagnostic
        sidecar.write_text(json.dumps(summary, indent=2) + '\n')
        if not diagnostic['finite_output']:
            raise RuntimeError('AWQ native Marlin returned nonfinite output; see saved numeric diagnostic')
    print(json.dumps({'artifact': str(args.out), 'metadata': str(sidecar),
                      'numeric_validation': summary['numeric_validation'],
                      'tensor_bytes': sum(value.numel() * value.element_size() for value in tensors.values())}))


if __name__ == '__main__':
    main()
