#!/usr/bin/env python3
"""Bounded same-input BF16 Triton tile probe; not an end-to-end result.

Imports CUDA libraries only after obtaining the shared resource lock. A synthetic
fixture checks the operator, never establishes realistic routing competitiveness.
Actual route bundles are trusted local torch.save dictionaries, weights_only=True.
"""
import argparse
import fcntl
import hashlib
from itertools import combinations
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

CONFIGS = {
    # Resolve with override_config(None), preserving installed tuned/default
    # selection. This is the native Triton reference, not the auto backend.
    "native_triton": None,
    "tile16": dict(BLOCK_SIZE_M=16, BLOCK_SIZE_N=64, BLOCK_SIZE_K=64,
                   GROUP_SIZE_M=1, SPLIT_K=1, num_warps=4, num_stages=3),
    "tile64": dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=64,
                   GROUP_SIZE_M=1, SPLIT_K=1, num_warps=4, num_stages=3),
}
LOCK = "/root/autodl-tmp/moe-research-gpu.lock"


def canonical_kernel_config(config):
    # invoke_fused_moe_triton_kernel copies the selected config and forces
    # SPLIT_K=1 (native_sources/.../fused_moe.py:802-804). A tuned file may omit
    # this key; that alone must not make an alias count as another path.
    return dict(config, SPLIT_K=1)


def save(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("moe_probe.json"))
    p.add_argument("--bundle", type=Path, help="x,w1,w2,topk_ids,topk_weights from one real layer")
    p.add_argument("--synthetic", action="store_true", help="operator check ONLY")
    p.add_argument("--tokens", type=int, choices=[1, 32, 128], default=32)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main():
    a = parse_args()
    result = dict(status="DRY_RUN", configs=CONFIGS, lock=LOCK,
                  tolerance=dict(rtol=0.03, atol=0.01),
                  scope="Same BF16 operands and frozen routes; isolated MoE graph replay latency",
                  warning="No service goodput, no portfolio memory inference, no gate A pass from synthetic routes",
                  synthetic=bool(a.synthetic), gpu_started=False)
    if a.dry_run:
        print(json.dumps(result, indent=2))
        return 0
    if bool(a.bundle) == bool(a.synthetic):
        raise SystemExit("Choose exactly one of --bundle or --synthetic")
    # Use a raw descriptor, not a local file object: main() can return while
    # Torch still owns a live CUDA context. Deliberately leave the acquired fd
    # open until OS process exit, including on exceptions. Do not inherit it
    # into helper processes such as nvidia-smi.
    lock_fd = os.open(LOCK, os.O_CREAT | os.O_RDWR, 0o600)
    os.set_inheritable(lock_fd, False)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(lock_fd)
        result["status"] = "LOCK_BUSY_NO_GPU_INITIALIZED"
        save(a.output, result)
        return 75
    state = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory",
        "--format=csv,noheader"], text=True).strip()
    if state:
        result.update(status="GPU_OCCUPIED_NO_CUDA_INITIALIZED", processes=state)
        save(a.output, result)
        return 75
    os.environ["VLLM_ENABLE_STARTUP_PLAN"] = "0"
    started = time.perf_counter()
    try:
        import torch
        # Reuse only process-local import repair; never edit the shared install.
        from runtime_bootstrap import apply
        result["orphan_inductor_modules_filtered"] = apply()
        # vLLM imports may themselves query CUDA; conservatively mark the
        # boundary before importing them, not after the capability assertion.
        result.update(gpu_started=True, torch=torch.__version__, cuda=torch.version.cuda)
        from vllm.model_executor.layers.fused_moe import fused_experts, override_config
        from vllm.model_executor.layers.fused_moe.fused_moe import try_get_optimal_moe_config
        assert torch.cuda.get_device_capability() == (12, 0), "Re-audit other architectures"
        result.update(gpu=torch.cuda.get_device_name(), synthetic=bool(a.synthetic),
                      backend_path="functional_triton_fused_experts")
        torch.manual_seed(20261008)
        torch.backends.cuda.matmul.allow_tf32 = False
        with torch.inference_mode():
            if a.bundle:
                result["bundle_sha256"] = hashlib.sha256(a.bundle.read_bytes()).hexdigest()
                bundle = torch.load(a.bundle, map_location="cpu", weights_only=True)
                tensors = {k: bundle[k].to("cuda").contiguous() for k in
                           ["x", "w1", "w2", "topk_ids", "topk_weights"]}
            else:
                # One OLMoE layer's exact dimensions, synthetic values explicitly labelled.
                tensors = {
                    "x": torch.randn(a.tokens, 2048, device="cuda", dtype=torch.bfloat16),
                    "w1": torch.randn(64, 2048, 2048, device="cuda", dtype=torch.bfloat16) / 2048**0.5,
                    "w2": torch.randn(64, 2048, 1024, device="cuda", dtype=torch.bfloat16) / 1024**0.5,
                }
                # OLMoE norm_topk_prob=false: do not renormalize selected weights.
                probabilities = torch.randn(a.tokens, 64, device="cuda").softmax(-1)
                weights, ids = probabilities.topk(8, dim=-1)
                tensors.update(topk_weights=weights, topk_ids=ids.int())
                del probabilities, weights, ids
            x, w1, w2, ids, weights = (tensors[k] for k in
                                      ["x", "w1", "w2", "topk_ids", "topk_weights"])
            assert x.ndim == 2 and x.shape[1] == 2048 and x.shape[0] in (1, 32, 128)
            assert tuple(w1.shape) == (64, 2048, 2048) and tuple(w2.shape) == (64, 2048, 1024)
            assert x.dtype == w1.dtype == w2.dtype == torch.bfloat16
            assert ids.shape == weights.shape == (len(x), 8)
            assert ids.dtype in (torch.int32, torch.int64)
            assert weights.dtype == torch.float32 and torch.isfinite(weights).all()
            assert (weights >= 0).all() and (weights.sum(-1) <= 1.00001).all()
            assert ids.min() >= 0 and ids.max() < 64
            assert (ids.sort(-1).values.diff(dim=-1) != 0).all()
            result["tokens"] = len(x)
            result["route_histogram"] = torch.bincount(ids.flatten().long(), minlength=64).cpu().tolist()

            # Independent FP32 dense reference using the very same selected experts.
            # Host inspection is outside timing and never part of a serving dispatcher.
            reference = torch.zeros_like(x, dtype=torch.float32)
            for e in range(64):
                row, slot = torch.where(ids == e)
                if not len(row):
                    continue
                gate, up = torch.nn.functional.linear(x[row].float(), w1[e].float()).chunk(2, dim=-1)
                out = torch.nn.functional.linear(torch.nn.functional.silu(gate) * up, w2[e].float())
                reference.index_add_(0, row, out * weights[row, slot, None])
            reference_cpu = reference.cpu()
            del reference
            # Delete reference temporaries; memory is not inferred from this microprobe.
            del gate, up, out
            outputs = {}
            measurements = {}
            measured_configs = {}
            warmup_stream = torch.cuda.Stream()
            warmup_stream.wait_stream(torch.cuda.current_stream())
            for name, config in CONFIGS.items():
                with override_config(config):
                    selected_config = dict(try_get_optimal_moe_config(
                        tuple(w1.shape), tuple(w2.shape), 8, None, len(x)))
                    if config is not None:
                        assert selected_config == config, "Tile override was not applied"
                    effective_config = canonical_kernel_config(selected_config)
                    signature = json.dumps(effective_config, sort_keys=True)
                    if signature in measured_configs:
                        alias = measured_configs[signature]
                        outputs[name] = outputs[alias]
                        measurements[name] = dict(
                            selected_config=selected_config,
                            effective_config=effective_config, alias_of=alias,
                            independent_measurement=False)
                        continue
                    # Warm up on the same nondefault stream used for capture.
                    with torch.cuda.stream(warmup_stream):
                        for _ in range(3):
                            y = fused_experts(x, w1, w2, weights, ids)
                    torch.cuda.synchronize()
                    actual = y.float().cpu()
                    torch.testing.assert_close(actual, reference_cpu, rtol=.03, atol=.01)
                    outputs[name] = actual
                    graph = torch.cuda.CUDAGraph()
                    t0 = time.perf_counter()
                    with torch.cuda.graph(graph, stream=warmup_stream):
                        graph_out = fused_experts(x, w1, w2, weights, ids)
                    torch.cuda.synchronize()
                    capture_s = time.perf_counter() - t0
                    # A no-op replay must not pass merely because capture left
                    # a correct output. Inputs, routes and weights stay frozen.
                    graph_out.fill_(float("nan"))
                    graph.replay()
                    torch.cuda.synchronize()
                    torch.testing.assert_close(graph_out.float().cpu(), reference_cpu, rtol=.03, atol=.01)
                    for _ in range(5):
                        graph.replay()
                    trials = []
                    for _ in range(5):
                        begin = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        begin.record()
                        for _ in range(100):
                            graph.replay()
                        end.record()
                        end.synchronize()
                        trials.append(begin.elapsed_time(end) / 100)
                    torch.testing.assert_close(graph_out.float().cpu(), reference_cpu, rtol=.03, atol=.01)
                    measurements[name] = dict(replay_ms=trials, median_ms=statistics.median(trials),
                                              selected_config=selected_config,
                                              effective_config=effective_config,
                                              alias_of=None, independent_measurement=True,
                                              capture_s=capture_s,
                                              max_abs_reference_error=(actual-reference_cpu).abs().max().item())
                    measured_configs[signature] = name
                    del graph, graph_out, y
            cross_errors = {}
            for first, second in combinations(outputs, 2):
                torch.testing.assert_close(outputs[first], outputs[second], rtol=.03, atol=.01)
                cross_errors[f"{first}/{second}"] = (outputs[first]-outputs[second]).abs().max().item()
            result.update(status="OPERATOR_CHECK_PASSED", measurements=measurements,
                          distinct_execution_config_count=len(measured_configs),
                          cross_configuration_max_abs_errors=cross_errors,
                          elapsed_s=time.perf_counter()-started,
                          cross_tile_max_abs_error=(outputs["tile16"]-outputs["tile64"]).abs().max().item())
        save(a.output, result)
        return 0
    except Exception as error:
        result.update(status="FAILED", error_type=type(error).__name__, error=str(error),
                      elapsed_s=time.perf_counter()-started)
        save(a.output, result)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
