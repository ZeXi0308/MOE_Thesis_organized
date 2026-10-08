#!/usr/bin/env python3
"""One author-AE Past-Future development cell in the explicit batch4096 regime."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PARENT_MANIFEST_SHA = "acdf36222489773ab1fc3a4b6580adcb5951b08af4fa044cfe2c66fb9e850792"
CONTRACT_SHA = "6896c987f30106deffed0bfce22b98728cf4d78871e41ba45e06b0c1383d72a8"
BOUND_ADAPTER_SHA = "b6a601d1edc05804bee70d7f6ecdb2f00e1dd017b2280163df2db88250fb50ee"
PF_ADAPTER_SHA = "28001321554a72bf1a451d26e15e6d4e2192c26d074891c6efd9bf5091ed3dc7"
PF_PREDICTOR_SHA = "e37d50ee262ae3325521bc95edee14079119451cd96f182c8236e905bc9a7773"
RETIREMENT_ADAPTER_SHA = "261b8f2697042f75130203019e8722c424cca3e71d82ac309c78f29ec1bc8eb6"
GPU_UUID = "GPU-e4434c32-c4a4-2b81-55fa-271af38f3c36"
KV_BYTES = 4097 * 2 * 1024 * 1024
SCHEDULER_SHA = "2ed2a550b6558b2495eda845a97ae38bcf0225027b9e25fbf00fc3880c1d3941"
MANAGER_SHA = "3f4af8d247f3fe9570b0132818b832b66ae6a2ac12942588828f899f6ff77ccf"
EXPECTED_RUNTIME = {'python': '3.12.3 | packaged by Anaconda, Inc. | (main, May  6 2024, 19:46:43) [GCC 11.2.0]', 'torch': '2.11.0+cu130', 'cuda': '13.0', 'vllm': '0.26.0', 'transformers': '5.15.1', 'cpu_threads': 25, 'vllm_source_sha256': {'v1/core/sched/scheduler.py': '2ed2a550b6558b2495eda845a97ae38bcf0225027b9e25fbf00fc3880c1d3941', 'v1/core/kv_cache_manager.py': '3f4af8d247f3fe9570b0132818b832b66ae6a2ac12942588828f899f6ff77ccf', 'v1/core/block_pool.py': '202a13cb129174849d798019aaedc04c59775ec2a4b9dfcc7c1e3c563a43a661', 'v1/worker/gpu_model_runner.py': '81b7627fbe81f7aaa2f77b4bf085faa353c69d03662ebfe369536a9773bb70d0', 'v1/core/kv_cache_coordinator.py': '4c8fbb341f0bd3714eff54ce633f534c1475220decb17c0caed40cbd02b352a2', 'v1/core/single_type_kv_cache_manager.py': 'bcb27e38895332bf6a4c55608f2917eb9fd941ec5a629c14b88358f00aadeba8', 'v1/core/kv_cache_utils.py': '6add6f1d60634b0819833675d86be4adf00c13fe5d0a1605074353b55d3e2d39', 'config/model.py': '7d52e060db54067a21591882bd6e3c2dd2486ac9041dde1117f97023e71bf89f'}}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def dump(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def native_empty_contract(engine) -> dict:
    """Explicit empty request/KV and disabled native-offload observations."""
    scheduler = engine.engine_core.engine_core.scheduler
    pool = scheduler.kv_cache_manager.block_pool
    running, waiting = scheduler.get_request_counts()
    receipt = dict(status="UNQUALIFIED", scheduler_connector_absent=scheduler.connector is None,
        kv_transfer_config_absent=engine.vllm_config.kv_transfer_config is None,
        kv_offloading_size=getattr(engine.vllm_config.cache_config, "kv_offloading_size", None),
        has_unfinished_requests=bool(engine.has_unfinished_requests()),
        scheduler_request_count=len(scheduler.requests), running=int(running), waiting=int(waiting),
        total_blocks=int(pool.num_gpu_blocks), free_blocks=int(pool.get_num_free_blocks()),
        host_scope="No native CPU KV offload connector/storage configured; total host memory is not zero")
    if (receipt["scheduler_connector_absent"] and receipt["kv_transfer_config_absent"]
            and receipt["kv_offloading_size"] is None and not receipt["has_unfinished_requests"]
            and receipt["scheduler_request_count"] == receipt["running"] == receipt["waiting"] == 0
            and receipt["total_blocks"] == 4097 and receipt["free_blocks"] == 4096):
        receipt["status"] = "QUALIFIED"
    return receipt


def gpu_state() -> dict:
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,memory.used",
                                   "--format=csv,noheader,nounits"], text=True).strip()
    procs = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory",
                                     "--format=csv,noheader,nounits"], text=True).strip()
    if not gpu.startswith(GPU_UUID + ", "):
        raise RuntimeError("approved GPU UUID changed")
    if any(int(line.split(",")[0].strip()) != os.getpid()
           for line in procs.splitlines() if line.strip()):
        raise RuntimeError("another GPU compute process was present")
    return dict(gpu=gpu, compute_processes=procs)


def run(parent: Path, inputs: Path, output: Path, arm: str) -> None:
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"immutable output already exists: {output}")
    if sha(parent / "manifest.json") != PARENT_MANIFEST_SHA:
        raise RuntimeError("frozen parent package manifest changed")
    contract_path = Path(__file__).with_name("C_NATIVE_RETIREMENT_FRESH_CONTRACT_V1.py")
    if sha(contract_path) != CONTRACT_SHA:
        raise RuntimeError("fresh input/resource contract source changed")
    adapter_root = Path(__file__).resolve().parent
    if arm == "native_past_future_ae" and sha(
            adapter_root / "C_NATIVE_MAX_BOUND_ADMISSION.py") != BOUND_ADAPTER_SHA:
        raise RuntimeError("frozen max-bound adapter changed")
    if arm == "native_past_future_ae" and sha(
            adapter_root / "C_NATIVE_RETIREMENT_ADMISSION_V1.py") != RETIREMENT_ADAPTER_SHA:
        raise RuntimeError("frozen retirement adapter changed")
    for name, digest in (("C_NATIVE_PAST_FUTURE_AE_ADMISSION_V1.py", PF_ADAPTER_SHA),
                         ("C_PAST_FUTURE_AE_PREDICTOR_V1.py", PF_PREDICTOR_SHA)):
        if sha(adapter_root / name) != digest:
            raise RuntimeError(f"frozen Past-Future source changed: {name}")
    sys.path.insert(0, str(parent))
    from verify_bundle import verify
    verify(parent)
    sys.path.insert(0, str(parent / "pkg"))
    from run_recovery_cadence import load_inputs, RUNTIME_SOURCES, eos_metadata
    from C_NATIVE_RETIREMENT_FRESH_CONTRACT_V1 import (
        ARMS, audit_outputs, configure_measurement, qualify_safe_cap, validate_measurement)
    from native_capture import capture_episode, set_empty_admission_cap
    from memory_telemetry import capture_with_memory, memory_snapshot
    from metrics import summarize_episode_requests

    if arm not in ("native_full", "native_past_future_ae"):
        raise ValueError(f"unsupported fresh policy-test arm: {arm}")
    original, workload = load_inputs(inputs)
    source_receipt = validate_measurement(inputs, original, workload)
    if not (original["requests"] == 128 and original["prompt_tokens"] == 3066
            and original["model"]["revision"] == "6d84c48581ece794365f2b8e9cfb043c68ade9c5"):
        raise RuntimeError("source/model/episode differs")
    warmups = {name: load_inputs(parent / "pkg/warmups" / name)
               for name in ("short", "long")}
    config = configure_measurement(original,
        [len(ids) for ids in workload["actual_prompt_token_ids"]],
        "native_full" if arm == "native_full" else "native_max_bound", source_receipt)
    config.update(variant="native_past_future_ae_batch4096" if arm != "native_full" else "native_full_batch4096",
        preemption_mode="native_recompute" if arm == "native_full" else "native_past_future_ae_recompute",
        reservation_policy=arm, max_num_batched_tokens=4096, policy_seed=20261001,
        metric_role="DEVELOPMENT_AUTHOR_AE_CORE_NATIVE_PORT_BATCH4096",
        comparison_scope="New batch4096 regime; prior batch1024 outcomes are not matched references",
        policy_scope="AE statistical and peak core; joint native token-budget guard keeps admitted prefill/recompute in one iteration; not full LightLLM system reproduction")
    output.mkdir(parents=True, exist_ok=False)
    dump(output / "config.json", config)
    dump(output / "source-receipt.json", source_receipt)
    timing = dict(process_start_perf_s=time.perf_counter(), process_start_unix_s=time.time())
    engine = None
    try:
        before = gpu_state()
        os.environ.update(VLLM_ENABLE_V1_MULTIPROCESSING="0", VLLM_BATCH_INVARIANT="0",
                          VLLM_USE_SIMPLE_KV_OFFLOAD="0")
        import torch
        import vllm
        from importlib.metadata import version
        from vllm.engine.arg_utils import EngineArgs
        from vllm.v1.engine.llm_engine import LLMEngine
        if vllm.__version__ != "0.26.0" or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("pinned vLLM 0.26 single-GPU environment differs")
        scheduler_path = Path(vllm.__file__).parent / "v1/core/sched/scheduler.py"
        manager_path = Path(vllm.__file__).parent / "v1/core/kv_cache_manager.py"
        if sha(scheduler_path) != SCHEDULER_SHA or sha(manager_path) != MANAGER_SHA:
            raise RuntimeError("native scheduler/manager source changed")
        runtime = dict(python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
            vllm=vllm.__version__, transformers=version("transformers"),
            cpu_threads=torch.get_num_threads(),
            vllm_source_sha256={name: sha(Path(vllm.__file__).parent / name) for name in RUNTIME_SOURCES})
        dump(output / "runtime-qualification.json", dict(
            status="QUALIFIED" if runtime == EXPECTED_RUNTIME else "UNQUALIFIED",
            observed=runtime, expected=EXPECTED_RUNTIME))
        if runtime != EXPECTED_RUNTIME:
            raise RuntimeError("software/runtime source identity differs from sustained baseline")
        model = config["model"]
        kwargs = dict(model=model["id"], revision=model["revision"],
                      tokenizer_revision=model["tokenizer_revision"], dtype="bfloat16",
                      seed=config["seed"], max_model_len=4096, max_num_seqs=32,
                      max_num_batched_tokens=4096, long_prefill_token_threshold=0,
                      gpu_memory_utilization=.90,
                      enable_chunked_prefill=True, enable_prefix_caching=False,
                      scheduling_policy="fcfs", async_scheduling=False,
                      kv_cache_memory_bytes=KV_BYTES, scheduler_reserve_full_isl=True,
                      stream_interval=1, enforce_eager=False,
                      enable_return_routed_experts=False)
        dump(output / "engine_args.json", kwargs)
        dump(output / "environment.json", dict(**runtime, gpu_before=before,
             parent_manifest_sha256=PARENT_MANIFEST_SHA,
             pilot_source_sha256=sha(Path(__file__)),
             fresh_contract_sha256=CONTRACT_SHA,
             fresh_input_sha256=source_receipt["input_sha256"],
             bound_adapter_sha256=BOUND_ADAPTER_SHA if arm != "native_full" else None,
             past_future_adapter_sha256=PF_ADAPTER_SHA,
             past_future_predictor_sha256=PF_PREDICTOR_SHA,
             scheduler_sha256=sha(scheduler_path), manager_sha256=sha(manager_path)))
        timing["engine_init_start_perf_s"] = time.perf_counter()
        engine = LLMEngine.from_engine_args(EngineArgs(**kwargs), enable_multiprocessing=False)
        timing["engine_init_end_perf_s"] = time.perf_counter()
        dump(output / "resolved-eos.json", eos_metadata(engine))
        scheduler = engine.engine_core.engine_core.scheduler
        if (scheduler.connector is not None
                or getattr(engine.vllm_config.cache_config, "kv_offloading_size", None) is not None):
            raise RuntimeError("native no-connector arm unexpectedly enabled KV offload")
        memory = memory_snapshot(engine, torch)
        qualification = qualify_safe_cap(engine, config)
        qualification["observed_scheduler_reserve_full_isl"] = scheduler.scheduler_reserve_full_isl
        dump(output / "memory-after-init.json", memory)
        dump(output / "safe-cap-qualification.json", qualification)
        if (qualification["status"] != "QUALIFIED" or qualification["usable_blocks"] != 4096
                or qualification["observed_scheduler_reserve_full_isl"] is not True
                or memory["kv_storage_bytes"] != KV_BYTES):
            raise RuntimeError("physical GPU KV/initial reservation qualification failed")
        timing["warmup_start_perf_s"] = time.perf_counter()
        for index, (domain, count, cap) in enumerate((("short", 32, 16),
                                                       ("short", 32, 32), ("long", 2, 2))):
            warm_config, warm_workload = warmups[domain]
            short_work = dict(warm_workload,
                              source_requests=warm_workload["source_requests"][:count],
                              actual_prompt_token_ids=warm_workload["actual_prompt_token_ids"][:count],
                              arrival_traces_s={"steady": [0.] * count})
            warm_config = dict(warm_config, cap=cap, requests=count,
                               output_tokens=16, output_tokens_by_request={})
            warm_config.pop("output_mode", None)
            set_empty_admission_cap(engine, cap)
            warm = capture_with_memory(engine, capture_episode, short_work, warm_config,
                                       regime="steady", arrival_scale=1.,
                                       run_id=f"warmup{index}", max_seconds=120)
            dump(output / f"warmup-{index}.json", warm)
            if warm["status"] != "COMPLETE":
                raise RuntimeError("application warmup incomplete; measurement unrun")
        timing["warmup_end_perf_s"] = time.perf_counter()
        set_empty_admission_cap(engine, 32)
        if scheduler.connector is not None:
            raise RuntimeError("connector unexpectedly appeared after warmup")
        warm_drain = native_empty_contract(engine)
        dump(output / "warmup-native-drain.json", warm_drain)
        if warm_drain["status"] != "QUALIFIED":
            raise RuntimeError("warmup native request/KV drain qualification failed")
        dump(output / "memory-before.json", memory_snapshot(engine, torch))
        torch.cuda.reset_peak_memory_stats()
        timing["measurement_start_perf_s"] = time.perf_counter()
        if arm == "native_past_future_ae":
            from C_NATIVE_PAST_FUTURE_AE_ADMISSION_V1 import install as install_past_future
            admission, trace_name = install_past_future(engine), "past-future-ae.json"
        else:
            admission, trace_name = nullcontext(None), None
        with admission as gate:
            try:
                raw = capture_with_memory(engine, capture_episode, workload, config,
                                          allow_preemption=True, capture_mode="sparse",
                                          regime="poisson_v1", arrival_scale=1., run_id="measured",
                                          max_seconds=300)
            finally:
                if gate is not None:
                    dump(output / trace_name, gate.receipt())
        timing["measurement_return_perf_s"] = time.perf_counter()
        raw["gpu_before"] = before
        dump(output / "raw.json", raw)
        if raw["status"] != "COMPLETE":
            raise RuntimeError(raw["error"] or "measurement incomplete")
        if gate is not None and gate.receipt()["status"] != "DRAINED":
            raise RuntimeError(f"{arm} admission mechanism did not drain")
        final_drain = native_empty_contract(engine)
        dump(output / "measurement-native-drain.json", final_drain)
        if final_drain["status"] != "QUALIFIED":
            raise RuntimeError("measurement native request/KV drain qualification failed")
        final = audit_outputs(raw, workload)
        dump(output / "baseline-result-audit.json", final)
        metrics = summarize_episode_requests(raw["requests"],
                   observation_end_s=raw["observation_end_s"],
                   ttft_slo_s=5., tpot_slo_s=.2)
        metrics.update(complete_episode_comparison_eligible=True,
                       actual_preemption_count=raw["actual_preemption_count"],
                       role=arm)
        dump(output / "metrics.json", metrics)
        dump(output / "memory-after.json", memory_snapshot(engine, torch))
        dump(output / "gpu-after.json", gpu_state())
        dump(output / "status.json", dict(status="COMPLETE", requests_completed=128,
             finish_reason_counts=final["finish_reason_counts"],
             output_tokens=final["output_tokens"],
             actual_preemption_count=raw["actual_preemption_count"],
             scientific_verdict="DEVELOPMENT_AE_CORE_PORT_BATCH4096_ONLY", arm=arm))
    except Exception as error:
        dump(output / "status.json", dict(status="INCOMPLETE",
             error=f"{type(error).__name__}: {error}", scientific_verdict="UNQUALIFIED"))
        raise
    finally:
        timing["shutdown_start_perf_s"] = time.perf_counter()
        try:
            if engine is not None:
                engine.engine_core.shutdown()
        finally:
            timing["process_end_perf_s"] = time.perf_counter()
            dump(output / "timing.json", timing)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("native_full", "native_past_future_ae"),
                        default="native_past_future_ae")
    parser.add_argument("--parent-package", type=Path, required=True)
    parser.add_argument("--inputs-dir", type=Path,
                        default=Path(__file__).resolve().parent / "20261001_c_retirement_fresh_inputs_v1")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.parent_package, args.inputs_dir, args.output_dir, args.arm)
