#!/usr/bin/env python3
"""Compile expert-skip Triton kernels for SM120 without a CUDA runtime.

This is ONLY a compiler check. It does not launch a kernel, allocate a tensor,
initialize a CUDA driver, verify numerical results, or measure performance.
The inspected API is Triton 3.6.0: compile(ASTSource(...), target=GPUTarget(...)).
CompiledKernel.__init__ reads files; runtime handles are initialized only by
launch-facing APIs, which this script never accesses. Both driver properties
are replaced by exceptions as a process-local guard, in addition to hiding CUDA.

--source is read and copied into --output; the original core is never changed.
Compilation caches are local to --output/cache. --plan-only needs only stdlib.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback


def write_json(path, value):
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temp.replace(path)


def pow2(value):
    return 1 << (max(value, 1)-1).bit_length()


def cases(batches, topologies, modes, renorms, dtypes):
    for batch, (experts, topk), mode, stats, active, renorm, dtype in itertools.product(
            batches, topologies, modes, (False, True), (False, True), renorms, dtypes):
        yield {"batch": batch, "experts": experts, "top_k": topk, "mode": mode,
               "collect_stats": stats, "has_active": active, "renormalize": renorm,
               "routing_weight_dtype": dtype, "id_dtype": "i32", "active_dtype": "i32"}


def kernel_specs(case):
    b, e, k = case["batch"], case["experts"], case["top_k"]
    active, stats = case["has_active"], case["collect_stats"]
    dtype = "*" + case["routing_weight_dtype"]
    common = {"K": k, "E": e, "MODE": {"marginal": 3, "rank_gate": 4}[case["mode"]],
              "DROP_J": 1, "TAIL": 4, "TAU": .1 if case["mode"] == "marginal" else .05,
              "RANK_END": k if case["mode"] == "marginal" else min(5, k),
              "GATE_MIN": 0. if case["mode"] == "marginal" else .025,
              "BUDGET": .2 if case["mode"] == "marginal" else 1.,
              "RENORM": case["renormalize"], "BLOCK_K": pow2(k),
              "HAS_ACTIVE": active, "COLLECT_STATS": stats}
    weights = {"weights": dtype, "ids": "*i32", "out_weights": dtype, "out_ids": "*i32"}
    # The real collect_stats=False call aliases counter to the weight tensor;
    # the pointer is dead in that constexpr branch but its type is retained.
    counter_type = "*fp64" if stats else dtype
    yield ("fused_small_kernel", {**weights, "counter": counter_type, "active_tokens": "*i32"},
           {**common, "M": b, "EXPERT_BYTES": 12582912,
            "BLOCK_M": pow2(b), "BLOCK_E": pow2(e)},
           {"num_warps": 4, "enable_fp_fusion": False})
    yield ("histogram_kernel", {"ids": "*i32", "counts": "*i32", "active_tokens": "*i32"},
           {"SLOTS": b*k, "E": e, "K": k, "HAS_ACTIVE": active, "BLOCK": 256},
           {"num_warps": 4})
    yield ("policy_kernel", {**weights, "counts": "*i32", "row_stats": "*fp32", "active_tokens": "*i32"},
           common, {"num_warps": 4, "enable_fp_fusion": False})
    if stats:
        yield ("stats_kernel", {"counts": "*i32", "row_stats": "*fp32", "counter": "*fp64", "active_tokens": "*i32"},
               {"E": e, "M": b, "EXPERT_BYTES": 12582912,
                "BLOCK_E": pow2(e), "BLOCK_M": pow2(b), "HAS_ACTIVE": active},
               {"num_warps": 4})


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=Path(__file__).with_name("expert_skip.py"))
    p.add_argument("--output", type=Path)
    p.add_argument("--batches", type=int, nargs="+", default=[1, 8, 32, 64])
    p.add_argument("--topologies", nargs="+", default=["64:8", "64:6", "128:8"], help="E:k pairs")
    p.add_argument("--modes", nargs="+", choices=["marginal", "rank_gate"], default=["marginal", "rank_gate"])
    p.add_argument("--renorm", nargs="+", choices=["false", "true"], default=["false"])
    p.add_argument("--weight-dtypes", nargs="+", choices=["fp32", "fp16", "bf16"], default=["fp32"])
    p.add_argument("--plan-only", action="store_true")
    return p


def main():
    args = parser().parse_args()
    topology = [tuple(map(int, text.split(":"))) for text in args.topologies]
    if any(len(t) != 2 or min(t) <= 0 or t[1] > t[0] or t[1] < 5 for t in topology):
        raise ValueError("Use E:k with E >= k >= 5 for the specified tail-rank diagnostics")
    if any(b <= 0 or b > 64 for b in args.batches):
        raise ValueError("This compile matrix covers B in [1,64]")
    matrix = list(cases(args.batches, topology, args.modes, [x == "true" for x in args.renorm], args.weight_dtypes))
    if args.plan_only:
        print(json.dumps({"logical_cases": len(matrix), "compile_requests": sum(len(list(kernel_specs(c))) for c in matrix),
                          "target": "cuda sm120 warp32", "cases": matrix}, indent=2))
        return
    if args.output is None:
        raise ValueError("--output is required for compilation")
    # Set these before importing Triton or any inspected core module.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["NVIDIA_VISIBLE_DEVICES"] = "void"
    os.environ["TRITON_INTERPRET"] = "0"
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.environ["TRITON_CACHE_DIR"] = str(out / "cache")
    os.environ["TRITON_DUMP_DIR"] = str(out / "dump")
    os.environ["TRITON_OVERRIDE_DIR"] = str(out / "override")
    # Reject inherited hooks/override paths that could change the inspected code.
    os.environ["TRITON_KERNEL_DUMP"] = "0"
    os.environ["TRITON_KERNEL_OVERRIDE"] = "0"
    snapshot = out / "expert_skip.py"
    shutil.copyfile(args.source, snapshot)
    source_sha = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    result = {"status": "RUNNING", "scope": "offline compilation only; no execution/correctness/performance evidence",
              "source": str(args.source.resolve()), "source_snapshot": str(snapshot), "source_sha256": source_sha,
              "target": {"backend": "cuda", "arch": 120, "warp_size": 32},
              "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
              "cache_dir": os.environ["TRITON_CACHE_DIR"], "cases": [], "artifacts": {},
              "logical_cases": len(matrix), "initial_driver_state": None}
    write_json(out / "summary.json", result)
    try:
        import triton
        from triton.compiler import ASTSource
        from triton.backends.compiler import GPUTarget
        from triton.runtime import driver
        result["triton_version"] = triton.__version__
        if triton.__version__ != "3.6.0":
            raise RuntimeError("Requires inspected Triton 3.6.0")
        if driver._active is not None or driver._default is not None:
            raise RuntimeError("Triton driver was initialized before offline compilation")
        result["initial_driver_state"] = {"active": None, "default": None}
        def deny_runtime(_self):
            raise RuntimeError("Offline compiler guard: a CUDA runtime/driver was requested")
        type(driver).active = property(deny_runtime)
        type(driver).default = property(deny_runtime)
        result["runtime_guard"] = "Triton active/default driver getters throw before initialization"
        spec = importlib.util.spec_from_file_location("expert_skip_offline_snapshot", snapshot)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _triton, histogram, policy, stats, fused = module._get_triton_kernels()
        kernels = {fn.__name__: fn for fn in (histogram, policy, stats, fused)}
        target = GPUTarget("cuda", 120, 32)
        for index, case in enumerate(matrix):
            entry = {"case": case, "kernels": [], "status": "PASS"}
            for name, pointers, constants, options in kernel_specs(case):
                descriptor = {"kernel": name, "pointers": pointers, "constexprs": constants, "options": options}
                key = hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode()).hexdigest()
                if key not in result["artifacts"]:
                    artifact = {**descriptor, "status": "COMPILING"}
                    start = time.monotonic()
                    try:
                        fn = kernels[name]
                        signature = {**pointers, **{key: "constexpr" for key in constants}}
                        if set(signature) != set(fn.arg_names):
                            raise ValueError(f"Signature mismatch for {name}: {set(signature)^set(fn.arg_names)}")
                        compiled = triton.compile(ASTSource(fn, signature, constexprs=constants), target=target, options=options)
                        # Access only constructor-populated fields. Never access
                        # run, __getitem__, launch_metadata, warmup, or n_regs.
                        if compiled.module is not None or compiled.function is not None or compiled._run is not None:
                            raise RuntimeError("Unexpected runtime handle initialization")
                        cubin = compiled.asm.get("cubin")
                        ptx = compiled.asm.get("ptx", "")
                        if not cubin or "sm_120" not in ptx:
                            raise RuntimeError("Expected an SM120 PTX target and nonempty cubin")
                        artifact.update(status="PASS", binary_bytes=len(cubin),
                            cubin_sha256=hashlib.sha256(cubin).hexdigest(), compiler_cache_key=compiled.hash,
                            compiled_shared_memory_bytes=compiled.metadata.shared,
                            ptx_target_lines=[line.strip() for line in ptx.splitlines() if line.strip().startswith(".target")],
                            metadata_group=compiled.metadata_group,
                            runtime_handles_uninitialized=True)
                    except Exception:
                        artifact.update(status="FAIL", error=traceback.format_exc())
                    artifact["compile_wall_seconds"] = time.monotonic()-start
                    result["artifacts"][key] = artifact
                    print(json.dumps({"case": index, "kernel": name, "status": artifact["status"],
                                      "artifact": key[:12]}), flush=True)
                entry["kernels"].append({"name": name, "artifact": key})
                if result["artifacts"][key]["status"] != "PASS":
                    entry["status"] = "FAIL"
            result["cases"].append(entry)
            write_json(out / "summary.json", result)
        if driver._active is not None or driver._default is not None:
            raise RuntimeError("Unexpected Triton driver initialization")
        result["final_driver_state"] = {"active": None, "default": None}
        result["case_status_counts"] = dict(Counter(row["status"] for row in result["cases"]))
        result["artifact_status_counts"] = dict(Counter(row["status"] for row in result["artifacts"].values()))
        result["status"] = "PASS" if all(row["status"] == "PASS" for row in result["cases"]) else "FAIL"
    except Exception:
        result.update(status="ERROR", error=traceback.format_exc())
        raise
    finally:
        write_json(out / "summary.json", result)
    print(json.dumps({key: result[key] for key in ("status", "logical_cases", "case_status_counts", "artifact_status_counts")}))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
