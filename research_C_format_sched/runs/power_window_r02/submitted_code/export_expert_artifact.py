#!/usr/bin/env python3
"""Export one calibrated ModelOpt NVFP4 MoE layer, without loading a model.

Reads only the requested per-expert safetensors from a pinned local HF snapshot.
Uses installed vLLM's own conversion and CUTLASS scale-fusion methods. This
program does not quantize weights or create calibration data.

Example:
  python export_expert_artifact.py --checkpoint-dir /data/nvfp4 \
    --checkpoint nvidia/Qwen3-30B-A3B-NVFP4 --revision COMMIT \
    --base-checkpoint Qwen/Qwen3-30B-A3B --base-revision COMMIT \
    --layer 0 --backend cutlass --out runs/qwen3_layer0_nvfp4_cutlass.pt
"""
import argparse
from contextlib import ExitStack
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import warnings

from format_backends import ExpertShape


PROJECTIONS = ('gate_proj', 'up_proj', 'down_proj')
FIELDS = ('weight', 'weight_scale', 'weight_scale_2', 'input_scale')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def checkpoint_shape(config):
    inner = config.get('text_config', config)
    return ExpertShape(d=int(inner['hidden_size']),
        intermediate=int(inner.get('moe_intermediate_size', inner.get('intermediate_size', 0))),
        experts=int(inner.get('num_experts', inner.get('num_local_experts', 0))),
        top_k=int(inner.get('num_experts_per_tok', inner.get('num_experts_per_token', 0))))


def expected_keys(shape, layer, prefix='model.layers'):
    return [f'{prefix}.{layer}.mlp.experts.{expert}.{projection}.{field}'
            for expert in range(shape.experts) for projection in PROJECTIONS for field in FIELDS]


def inspect_checkpoint(checkpoint_dir, shape, layer, prefix, index_path=None):
    """Use index or safetensors headers only; no model or GPU is initialized."""
    checkpoint_dir = Path(checkpoint_dir)
    index_path = Path(index_path) if index_path else checkpoint_dir / 'model.safetensors.index.json'
    if index_path.exists():
        weight_map = json.loads(index_path.read_text())['weight_map']
    else:
        from safetensors import safe_open
        weight_map = {}
        for file in checkpoint_dir.glob('*.safetensors'):
            with safe_open(file, framework='pt', device='cpu') as handle:
                weight_map.update({key: file.name for key in handle.keys()})
    keys = expected_keys(shape, layer, prefix)
    missing_keys = [key for key in keys if key not in weight_map]
    if missing_keys:
        raise ValueError(f'{len(missing_keys)} missing expert tensors, first: {missing_keys[:4]}')
    shards = sorted({weight_map[key] for key in keys})
    missing_shards = [name for name in shards if not (checkpoint_dir / name).is_file()]
    return keys, weight_map, shards, missing_shards


def load_modelopt_layer(checkpoint_dir, shape, layer, prefix, weight_map, device):
    import torch
    from safetensors import safe_open
    e, n, d = shape.experts, shape.intermediate, shape.d
    arrays = {
        'w13_weight': torch.empty((e, 2 * n, d // 2), dtype=torch.uint8),
        'w2_weight': torch.empty((e, d, n // 2), dtype=torch.uint8),
        'w13_weight_scale': torch.empty((e, 2 * n, d // 16), dtype=torch.float8_e4m3fn),
        'w2_weight_scale': torch.empty((e, d, n // 16), dtype=torch.float8_e4m3fn),
        'w13_weight_scale_2': torch.empty((e, 2), dtype=torch.float32),
        'w2_weight_scale_2': torch.empty((e,), dtype=torch.float32),
        'w13_input_scale': torch.empty((e, 2), dtype=torch.float32),
        'w2_input_scale': torch.empty((e,), dtype=torch.float32),
    }
    source_hash = hashlib.sha256()
    with ExitStack() as stack:
        handles = {}
        for expert in range(e):
            for projection_idx, projection in enumerate(PROJECTIONS):
                for field in FIELDS:
                    key = f'{prefix}.{layer}.mlp.experts.{expert}.{projection}.{field}'
                    shard = weight_map[key]
                    if shard not in handles:
                        handles[shard] = stack.enter_context(safe_open(
                            Path(checkpoint_dir) / shard, framework='pt', device='cpu'))
                    value = handles[shard].get_tensor(key)
                    source_hash.update(key.encode())
                    source_hash.update(str((str(value.dtype), list(value.shape))).encode())
                    source_hash.update(value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
                    if projection_idx == 2:
                        dest = arrays['w2_' + field][expert]
                    elif field in ('weight', 'weight_scale'):
                        dest = arrays['w13_' + field][expert, projection_idx * n:(projection_idx + 1) * n]
                    else:
                        dest = arrays['w13_' + field][expert, projection_idx]
                    if field in ('weight', 'weight_scale'):
                        if value.shape != dest.shape or value.dtype != dest.dtype:
                            raise ValueError(f'{key}: {tuple(value.shape)} {value.dtype}; '
                                             f'expected {tuple(dest.shape)} {dest.dtype}')
                    elif value.numel() != 1:
                        raise ValueError(f'{key}: expected a scalar scale, got {tuple(value.shape)}')
                    if field != 'weight':
                        values_float = value.float()
                        if not torch.isfinite(values_float).all() or (values_float < 0).any():
                            raise ValueError(f'{key}: invalid/nonfinite scale')
                        if field in ('input_scale', 'weight_scale_2') and (values_float <= 0).any():
                            raise ValueError(f'{key}: global/input scale must be positive')
                    dest.copy_(value.reshape(dest.shape))
    gate_scale, up_scale = arrays['w13_weight_scale_2'].unbind(dim=1)
    close = bool(torch.allclose(gate_scale, up_scale))
    if not close:
        warnings.warn('Gate/up weight_scale_2 differ. Following native ModelOptNvFp4FusedMoE, '
                      'w13 uses gate scale; artifact records this potential accuracy issue.')
    checks = {'gate_up_weight_scale2_allclose': close,
              'gate_up_weight_scale2_max_relative_difference':
                  float(((gate_scale - up_scale).abs() / gate_scale.abs().clamp_min(1e-30)).max()),
              'selected_tensor_payload_sha256': source_hash.hexdigest(),
              'selected_tensor_hash_order': 'expert, gate/up/down, weight/weight_scale/weight_scale_2/input_scale'}
    return {key: value.to(device) for key, value in arrays.items()}, checks


def convert_nvfp4(arrays, shape, backend_name, device):
    """Execute the actual vLLM ModelOpt conversion sequence for this layer."""
    import torch
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig, FusedMoEParallelConfig, RoutingMethodType)
    from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
        NvFp4MoeBackend, convert_to_nvfp4_moe_kernel_format, make_nvfp4_moe_quant_config)
    from vllm.model_executor.layers.fused_moe.experts.cutlass_moe import CutlassExpertsFp4
    backend = NvFp4MoeBackend.VLLM_CUTLASS if backend_name == 'cutlass' else NvFp4MoeBackend.MARLIN
    layer = torch.nn.Module()
    layer.activation = MoEActivation.SILU
    layer.num_experts = shape.experts
    layer.hidden_size = shape.d
    layer.intermediate_size_per_partition = shape.intermediate
    layer.params_dtype = torch.bfloat16
    layer.moe_config = FusedMoEConfig(num_experts=shape.experts, experts_per_token=shape.top_k,
        hidden_dim=shape.d, intermediate_size=shape.intermediate,
        num_local_experts=shape.experts, num_logical_experts=shape.experts,
        activation=MoEActivation.SILU, device=device, routing_method=RoutingMethodType.Renormalize,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(), in_dtype=torch.bfloat16,
        moe_backend=backend_name)
    converted = convert_to_nvfp4_moe_kernel_format(nvfp4_backend=backend, layer=layer,
        w13=arrays['w13_weight'], w13_scale=arrays['w13_weight_scale'],
        # Matches native ModelOptNvFp4FusedMoE.process_weights_after_loading.
        w13_scale_2=arrays['w13_weight_scale_2'][:, 0].contiguous(),
        a13_scale=arrays['w13_input_scale'],
        w2=arrays['w2_weight'], w2_scale=arrays['w2_weight_scale'],
        w2_scale_2=arrays['w2_weight_scale_2'], a2_scale=arrays['w2_input_scale'],
        is_act_and_mul=True)
    converted_names = ['w13_weight', 'w13_weight_scale', 'w13_weight_scale_2', 'w13_input_scale',
                       'w2_weight', 'w2_weight_scale', 'w2_weight_scale_2', 'w2_input_scale']
    for name, tensor in zip(converted_names, converted):
        setattr(layer, name, torch.nn.Parameter(tensor, requires_grad=False) if tensor is not None else None)
    qc = make_nvfp4_moe_quant_config(backend=backend, w13_scale=layer.w13_weight_scale,
        w2_scale=layer.w2_weight_scale, w13_scale_2=layer.w13_weight_scale_2,
        w2_scale_2=layer.w2_weight_scale_2, a13_scale=layer.w13_input_scale,
        a2_scale=layer.w2_input_scale, layer=layer)
    tensors = {'w1': layer.w13_weight, 'w2': layer.w2_weight,
               'w1_scale': layer.w13_weight_scale, 'w2_scale': layer.w2_weight_scale}
    methods = [convert_to_nvfp4_moe_kernel_format, make_nvfp4_moe_quant_config]
    if backend_name == 'cutlass':
        experts = CutlassExpertsFp4(moe_config=layer.moe_config, quant_config=qc)
        # This fuses activation scales into the parameter referenced by g*_alphas.
        experts.process_weights_after_loading(layer)
        tensors.update(g1_alphas=qc.g1_alphas, g2_alphas=qc.g2_alphas,
                       a1_gscale=qc.a1_gscale, a2_gscale=qc.a2_gscale)
        methods.append(CutlassExpertsFp4.process_weights_after_loading)
        layout = 'vllm_cutlass_nvfp4_processed'
    else:
        tensors.update(global_scale1=qc.g1_alphas, global_scale2=qc.g2_alphas)
        layout = 'vllm_marlin_processed'
    provenance = {}
    for method in methods:
        file = inspect.getsourcefile(method)
        provenance[method.__module__ + '.' + method.__qualname__] = {
            'source_file': file, 'source_file_sha256': sha256(file)}
    return {key: value.detach().cpu().contiguous() for key, value in tensors.items()}, layout, provenance


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint-dir', type=Path, required=True)
    p.add_argument('--index', type=Path)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--revision', required=True, help='Pinned quantized checkpoint commit, not main')
    p.add_argument('--base-checkpoint', required=True)
    p.add_argument('--base-revision', required=True, help='Pinned base checkpoint commit')
    p.add_argument('--layer', type=int, default=0)
    p.add_argument('--prefix', default='model.layers')
    p.add_argument('--backend', choices=['cutlass', 'marlin'], default='cutlass')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--calibration-method', default='NVIDIA ModelOpt NVFP4 calibrated checkpoint')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--dry-run', action='store_true', help='Inspect required files only; do not load weights or use GPU')
    args = p.parse_args()
    if args.revision in ('main', 'master') or args.base_revision in ('main', 'master'):
        raise ValueError('Use immutable checkpoint revisions')
    config_path = args.checkpoint_dir / 'config.json'
    config = json.loads(config_path.read_text())
    shape = checkpoint_shape(config)
    shape.validate()
    keys, weight_map, shards, missing = inspect_checkpoint(args.checkpoint_dir, shape,
        args.layer, args.prefix, args.index)
    plan = {'checkpoint': args.checkpoint, 'checkpoint_revision': args.revision,
        'revision': args.revision, 'base_checkpoint': args.base_checkpoint,
        'base_checkpoint_revision': args.base_revision, 'layer': args.layer,
        'shape': asdict(shape), 'required_tensor_count': len(keys),
        'required_shards': shards, 'missing_shards': missing,
        'config_sha256': sha256(config_path), 'backend': args.backend}
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    if missing:
        raise FileNotFoundError(f'Required checkpoint shards are not local: {missing}')
    import torch
    import vllm
    torch.cuda.set_device(torch.device(args.device))
    arrays, checks = load_modelopt_layer(args.checkpoint_dir, shape, args.layer,
                                        args.prefix, weight_map, args.device)
    with torch.inference_mode():
        tensors, layout, conversion = convert_nvfp4(arrays, shape, args.backend, args.device)
    meta = {**plan, **checks, 'schema': 1, 'format': 'nvfp4', 'layout': layout,
        'calibration_method': args.calibration_method,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'torch_version': str(torch.__version__), 'vllm_version': str(vllm.__version__),
        'source_config_quantization': config.get('quantization_config'),
        'weight_source': 'calibrated_checkpoint', 'conversion': conversion,
        'tensor_manifest': {key: {'shape': list(value.shape), 'dtype': str(value.dtype),
            'bytes': value.numel() * value.element_size()} for key, value in tensors.items()},
        'quality_claim': False,
        'note': 'Native conversion from supplied calibrated checkpoint; quality must be measured on the same checkpoint.'}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temp = args.out.with_suffix(args.out.suffix + '.tmp')
    torch.save({'metadata': meta, 'tensors': tensors}, temp)
    temp.replace(args.out)
    sidecar = args.out.with_suffix(args.out.suffix + '.json')
    sidecar.write_text(json.dumps({**meta, 'artifact_sha256': sha256(args.out)}, indent=2) + '\n')
    print(json.dumps({'artifact': str(args.out), 'metadata': str(sidecar),
                      'tensor_bytes': sum(value.numel() * value.element_size() for value in tensors.values())}))


if __name__ == '__main__':
    main()
