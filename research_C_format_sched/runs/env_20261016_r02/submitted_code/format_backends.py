"""Real vLLM 0.26 fused expert backends (imports are lazy for CPU tooling).

BF16/FP8 use synthetic weights for kernel characterization. Four-bit formats
require a calibrated checkpoint artifact, never locally invented RTN weights.
Artifact: torch.save({'metadata': {...}, 'tensors': {...}}, path), weights_only
loadable. metadata must contain format, layout, shape, calibration_method,
checkpoint, checkpoint_revision, layer. Prepared Marlin tensors must be exported
after vLLM process_weights_after_loading, including scales and zero points.
The checkpoint identity must also be used for quality/serving experiments.
"""
from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path


@dataclass(frozen=True)
class ExpertShape:
    d: int
    intermediate: int
    experts: int
    top_k: int

    def validate(self):
        if min(self.d, self.intermediate, self.experts, self.top_k) <= 0:
            raise ValueError('All shape dimensions must be positive')
        if self.top_k > self.experts:
            raise ValueError('top_k exceeds experts')


SHAPES = {
    'olmoe': ExpertShape(2048, 1024, 64, 8),
    'qwen3': ExpertShape(2048, 768, 128, 8),
}


class UnsupportedFormat(RuntimeError):
    pass


class FusedBackend:
    def __init__(self, fn, tensors, metadata):
        self.fn = fn
        self.tensors = tensors
        self.metadata = metadata
        self.metadata['resident_weight_and_scale_bytes'] = sum(
            t.numel() * t.element_size() for t in tensors.values()
            if hasattr(t, 'numel'))

    def __call__(self, hidden_states, topk_weights, topk_ids):
        return self.fn(hidden_states, topk_weights, topk_ids)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _load_calibrated_artifact(path, fmt, shape, device):
    import torch
    if not path:
        raise UnsupportedFormat(
            f'{fmt} requires --weights with calibrated checkpoint tensors; '
            'synthetic RTN is deliberately unsupported')
    data = torch.load(path, map_location='cpu', weights_only=True)
    meta, tensors = data['metadata'], data['tensors']
    required = ['format', 'layout', 'shape', 'calibration_method', 'checkpoint',
                'checkpoint_revision', 'layer']
    missing = [key for key in required if key not in meta or meta[key] in ('', None)]
    if missing:
        raise ValueError(f'Artifact metadata missing {missing}')
    if meta['format'] != fmt or meta['shape'] != asdict(shape):
        raise ValueError('Artifact format/shape differs from requested benchmark')
    if 'rtn' in str(meta['calibration_method']).lower():
        raise ValueError('RTN artifacts are not allowed in this experiment')
    if fmt == 'awq' and 'awq' not in str(meta['calibration_method']).lower():
        raise ValueError('AWQ arm requires an AWQ calibrated artifact')
    meta = {**meta, 'artifact': str(Path(path).resolve()), 'artifact_sha256': _sha256(path),
            'weight_source': 'calibrated_checkpoint'}
    return {key: value.to(device) if torch.is_tensor(value) else value
            for key, value in tensors.items()}, meta


def build_backend(fmt, shape, *, device='cuda:0', seed=20261015,
                  weights=None, nvfp4_backend='marlin', max_tokens=16384):
    import torch
    shape.validate()
    torch.manual_seed(seed)
    if fmt in ('bf16', 'fp8'):
        from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
        from vllm.model_executor.layers.fused_moe.config import fp8_w8a8_moe_quant_config
        weight_shapes = {'w1': (shape.experts, 2 * shape.intermediate, shape.d),
                         'w2': (shape.experts, shape.d, shape.intermediate)}
        tensors = {}
        for key, dims in weight_shapes.items():
            if fmt == 'bf16':
                tensors[key] = torch.randn(dims, device=device, dtype=torch.bfloat16) * 0.02
            else:
                # One expert at a time avoids an additional full FP32 weight copy.
                q = torch.empty(dims, device=device, dtype=torch.float8_e4m3fn)
                scales = torch.empty(shape.experts, device=device, dtype=torch.float32)
                for expert in range(shape.experts):
                    source = torch.randn(dims[1:], device=device, dtype=torch.bfloat16) * 0.02
                    scale = source.float().abs().amax().clamp_min(1e-12) / 448.0
                    q[expert] = (source.float() / scale).clamp(-448, 448).to(q.dtype)
                    scales[expert] = scale
                tensors[key], tensors[key + '_scale'] = q, scales
        qc = None if fmt == 'bf16' else fp8_w8a8_moe_quant_config(
            w1_scale=tensors['w1_scale'], w2_scale=tensors['w2_scale'])
        def run(x, scores, ids):
            return fused_experts(x, tensors['w1'], tensors['w2'], scores, ids,
                                 global_num_experts=shape.experts, quant_config=qc)
        return FusedBackend(run, tensors, {
            'format': fmt, 'requested_backend': 'triton',
            'python_entrypoint': 'vllm.model_executor.layers.fused_moe.fused_moe.fused_experts',
            'weight_source': 'synthetic_normal_std_0.02', 'weight_seed': seed,
            'quantization': 'none' if fmt == 'bf16' else
                'FP8 E4M3 W8A8; static per-expert per-tensor weight scales; '
                'dynamic per-tensor activation scales',
            'auto_backend_selection': False, 'shape': asdict(shape),
            'quality_claim': False,
        })
    if fmt not in ('awq', 'nvfp4'):
        raise UnsupportedFormat(f'Unknown format {fmt}')
    tensors, meta = _load_calibrated_artifact(weights, fmt, shape, device)
    if fmt == 'awq' or nvfp4_backend == 'marlin':
        from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
        from vllm.scalar_type import scalar_types
        if meta['layout'] != 'vllm_marlin_processed':
            raise UnsupportedFormat('Marlin requires layout=vllm_marlin_processed')
        required = ['w1', 'w2', 'w1_scale', 'w2_scale']
        if fmt == 'awq':
            required += ['w1_zeros', 'w2_zeros']
        else:
            required += ['global_scale1', 'global_scale2']
        for key in required:
            if key not in tensors:
                raise ValueError(f'Marlin artifact missing {key}')
        optional = ['g_idx1', 'g_idx2', 'sort_indices1', 'sort_indices2',
                    'w1_zeros', 'w2_zeros', 'global_scale1', 'global_scale2']
        extra = {key: tensors[key] for key in optional if key in tensors}
        quant_type = scalar_types.uint4 if fmt == 'awq' else scalar_types.float4_e2m1f
        def run(x, scores, ids):
            return fused_marlin_moe(x, tensors['w1'], tensors['w2'], None, None,
                tensors['w1_scale'], tensors['w2_scale'], scores, ids,
                quant_type.id, global_num_experts=shape.experts, input_dtype=None, **extra)
        meta.update(requested_backend='marlin', auto_backend_selection=False,
                    python_entrypoint='vllm.model_executor.layers.fused_moe.experts.marlin_moe.fused_marlin_moe',
                    runtime_activation_dtype='bfloat16',
                    quantization='AWQ W4A16' if fmt == 'awq' else 'NVFP4 weights, W4A16 Marlin execution')
        return FusedBackend(run, tensors, meta)
    if nvfp4_backend == 'flashinfer_b12x':
        # This is the same wrapper used by vLLM FlashInferB12xExperts. Keep
        # the lower-level entrypoint explicit: it is opt-in, not auto dispatch.
        from flashinfer.fused_moe import B12xMoEWrapper
        if meta['layout'] != 'vllm_b12x_processed':
            raise UnsupportedFormat('B12X requires vllm_b12x_processed artifact with MMA-layout scales')
        required = ['w1', 'w2', 'w1_sf_mma', 'w2_sf_mma', 'g1_alphas',
                    'g2_alphas', 'fc2_input_scale']
        for key in required:
            if key not in tensors:
                raise ValueError(f'B12X artifact missing {key}')
        wrapper = B12xMoEWrapper(num_experts=shape.experts, top_k=shape.top_k,
            hidden_size=shape.d, intermediate_size=shape.intermediate,
            use_cuda_graph=True, max_num_tokens=max_tokens,
            num_local_experts=shape.experts, activation='silu')
        def run(x, scores, ids):
            return wrapper.run(x=x, w1_weight=tensors['w1'], w1_weight_sf=tensors['w1_sf_mma'],
                w1_alpha=tensors['g1_alphas'], fc2_input_scale=tensors['fc2_input_scale'],
                w2_weight=tensors['w2'], w2_weight_sf=tensors['w2_sf_mma'],
                w2_alpha=tensors['g2_alphas'], token_selected_experts=ids.to(torch.int32),
                token_final_scales=scores)
        meta.update(requested_backend='flashinfer_b12x', auto_backend_selection=False,
                    python_entrypoint='flashinfer.fused_moe.B12xMoEWrapper.run (vLLM B12X expert delegate)',
                    quantization='NVFP4 W4A4, dynamic per-block activation quantization')
        return FusedBackend(run, tensors, meta)
    raise UnsupportedFormat(f'Unsupported NVFP4 backend {nvfp4_backend}')
