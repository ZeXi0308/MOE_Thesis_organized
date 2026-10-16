#!/usr/bin/env python3
"""Record the installed build, idle telemetry and backend eligibility (not execution)."""
import argparse
import dataclasses
import hashlib
import importlib.metadata as metadata
import json
import pathlib
import subprocess
import time
import traceback


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    a = p.parse_args()
    import pynvml as n
    n.nvmlInit()
    h = n.nvmlDeviceGetHandleByIndex(0)
    data = {'timestamp_utc_s': time.time(), 'gpu_uuid': n.nvmlDeviceGetUUID(h),
            'gpu_name': n.nvmlDeviceGetName(h), 'driver': n.nvmlSystemGetDriverVersion(),
            'packages': {}, 'idle_samples': [], 'backend_eligibility': {}}
    for name in ('vllm', 'torch', 'triton', 'flashinfer-python', 'nvidia-ml-py', 'llmcompressor', 'lm-eval'):
        try:
            dist = metadata.distribution(name)
            data['packages'][name] = {'version': dist.version, 'direct_url': dist.read_text('direct_url.json')}
        except metadata.PackageNotFoundError:
            data['packages'][name] = None
    for _ in range(61):
        s = {'monotonic_s': time.perf_counter(), 'power_w': n.nvmlDeviceGetPowerUsage(h) / 1000,
             'utilization_percent': n.nvmlDeviceGetUtilizationRates(h).gpu,
             'memory_used_bytes': n.nvmlDeviceGetMemoryInfo(h).used,
             'compute_pids': [p.pid for p in n.nvmlDeviceGetComputeRunningProcesses(h)]}
        try:
            s['total_energy_mj'] = n.nvmlDeviceGetTotalEnergyConsumption(h)
        except n.NVMLError as e:
            s['total_energy_error'] = str(e)
        data['idle_samples'].append(s)
        time.sleep(.05)
    import torch
    import vllm
    data['torch_cuda_version'] = torch.version.cuda
    data['device_capability'] = torch.cuda.get_device_capability()
    data['torch_arch_list'] = torch.cuda.get_arch_list()
    data['vllm_path'] = vllm.__file__
    source_root = pathlib.Path(vllm.__file__).parent
    data['source_sha256'] = {}
    for rel in ('model_executor/layers/fused_moe/oracle/nvfp4.py',
                'model_executor/layers/fused_moe/experts/flashinfer_b12x_moe.py',
                'model_executor/layers/fused_moe/fused_moe.py',
                'model_executor/layers/quantization/fp8.py',
                'model_executor/layers/quantization/awq_marlin.py'):
        f = source_root / rel
        if f.exists():
            data['source_sha256'][rel] = hashlib.sha256(f.read_bytes()).hexdigest()
    try:
        from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig, FusedMoEParallelConfig, RoutingMethodType
        from vllm.model_executor.layers.fused_moe.activation import MoEActivation
        from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import NvFp4MoeBackend, backend_to_kernel_cls, select_nvfp4_moe_backend
        from vllm.model_executor.layers.quantization.utils.quant_utils import kNvfp4Static, kNvfp4Dynamic
        import vllm.model_executor.layers.fused_moe.modular_kernel as mk
        cfg = FusedMoEConfig(num_experts=64, experts_per_token=8, hidden_dim=2048,
              intermediate_size=2048, num_local_experts=64, num_logical_experts=64,
              activation=MoEActivation.SILU, device='cuda', routing_method=RoutingMethodType.Renormalize,
              moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(), in_dtype=torch.bfloat16)
        for backend in NvFp4MoeBackend:
            rows = []
            try:
                for cls in backend_to_kernel_cls(backend):
                    ok, reason = cls.is_supported_config(cls, cfg, kNvfp4Static, kNvfp4Dynamic,
                                                        mk.FusedMoEActivationFormat.Standard)
                    rows.append({'class': cls.__name__, 'eligible': ok, 'reason': reason})
            except Exception:
                rows.append({'error': traceback.format_exc()})
            data['backend_eligibility'][backend.value] = rows
        backend, cls = select_nvfp4_moe_backend(cfg, kNvfp4Static, kNvfp4Dynamic)
        data['nvfp4_auto_selection'] = {'backend': backend.value, 'class': cls.__name__,
                                       'status': 'eligibility_only_not_executed'}
    except Exception:
        data['backend_probe_error'] = traceback.format_exc()
    for name, cmd in [('nvidia_smi', ['nvidia-smi']), ('disk', ['df', '-h', '/', '/root/autodl-tmp'])]:
        data[name] = subprocess.run(cmd, capture_output=True, text=True).stdout
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, default=str) + '\n')
    print(json.dumps({k: v for k, v in data.items() if k not in ('idle_samples', 'nvidia_smi')}, indent=2, default=str))


if __name__ == '__main__':
    main()
