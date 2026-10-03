#!/usr/bin/env python3
"""Read-only audit of one archived G64 native/eager/LTR T30Q10 pilot session."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
from typing import Any

from evaluate_goodput import pair, summarize

ARMS = ("native_full_native", "eager", "ltr_t30_q10")
PACKAGE_SHA = "2755945122e3c346ca7e7e5ceda0d4668a8e1d3c90ba5c8d9d7b6ab3e89e0eb9"
MODEL_ID = "allenai/OLMoE-1B-7B-0924"
REVISION = "6d84c48581ece794365f2b8e9cfb043c68ade9c5"
KV_BYTES = 8592031744
HOST_BYTES = 17179869184
MODEL_METADATA_SHA = {
    "config.json": "3643aa880d2f1c9b418156269ae791c73e5612d6b6b6fde0724d927cf89b6335",
    "generation_config.json": "d77272ffaa7e62a904e8e130bb25ab11585bd4a5026e388d6d682e4b82892ce2",
    "model.safetensors.index.json": "0e2e1e0d8d357ac7af817cff28410c3dbad398f060c517a433e4076b2aae5579",
    "special_tokens_map.json": "b77491e270c6fcc5b2ecf22370f7318a6a18d3cabea09ba7bab92e9bf12656c2",
    "tokenizer.json": "a094266ac6c4982efba277bc251349a5a6d6ad37efb39a2a90f53d8be2a40a40",
    "tokenizer_config.json": "78a839c7851f14f9fb30e664c2b46166dc0628f2900679e5ec160656f702edff",
}
MODEL_SHARDS = {
    "model-00001-of-00003.safetensors": {"bytes": 4997744872, "sha256": "5e3cff7e367794685c241169072c940d200918617d5e2813f1c387dff52d845e"},
    "model-00002-of-00003.safetensors": {"bytes": 4997235176, "sha256": "15ef5c730ee3cfed7199498788cd2faf337203fc74b529625e7502cdd759f4a7"},
    "model-00003-of-00003.safetensors": {"bytes": 3843741912, "sha256": "a9abac4ac1b55c9adabac721a02fa39971f103eea9a65c310972b1246de76e04"},
}
SHA_RE = re.compile(r"[0-9a-f]{64}\Z")
BASE_SOURCES = (
    "request_measurement.py", "native_capture.py", "memory_telemetry.py", "metrics.py",
    "safe_static.py", "rotation_native.py", "absence_rotation.py",
    "staged_save_contract.py", "native_offload_observer.py", "native_host_snapshot.py",
    "native_store_delta.py", "native_full_store_evidence.py",
)
ARM_SOURCES = {
    "native_full_native": ("run_recovery_cadence.py", "staged_store_rotation.py", *BASE_SOURCES),
    "eager": ("run_recovery_cadence.py", "staged_store_rotation.py", *BASE_SOURCES),
    "ltr_t30_q10": ("run_ltr_style.py", "ltr_style_native.py", "ltr_style_selected.py",
                    "recovery_service_components.py", *BASE_SOURCES),
}
PINNED_RUNTIME_SHA = {
    "v1/core/sched/scheduler.py": "2ed2a550b6558b2495eda845a97ae38bcf0225027b9e25fbf00fc3880c1d3941",
    "v1/core/kv_cache_manager.py": "3f4af8d247f3fe9570b0132818b832b66ae6a2ac12942588828f899f6ff77ccf",
    "v1/core/block_pool.py": "202a13cb129174849d798019aaedc04c59775ec2a4b9dfcc7c1e3c563a43a661",
    "v1/worker/gpu_model_runner.py": "81b7627fbe81f7aaa2f77b4bf085faa353c69d03662ebfe369536a9773bb70d0",
    "v1/core/kv_cache_coordinator.py": "4c8fbb341f0bd3714eff54ce633f534c1475220decb17c0caed40cbd02b352a2",
    "v1/core/single_type_kv_cache_manager.py": "bcb27e38895332bf6a4c55608f2917eb9fd941ec5a629c14b88358f00aadeba8",
    "v1/core/kv_cache_utils.py": "6add6f1d60634b0819833675d86be4adf00c13fe5d0a1605074353b55d3e2d39",
    "config/model.py": "7d52e060db54067a21591882bd6e3c2dd2486ac9041dde1117f97023e71bf89f",
}
REQUIRED_OUTPUTS = {
    "commands.txt", "config.json", "engine_args.json", "environment.json", "raw.json",
    "status.json", "safe-cap-qualification.json", "memory-after-init.json",
    "host-after-init.json", "host-before.json", "host-request-end.json", "host-after.json",
    "warmup-0.json", "warmup-1.json", "warmup-2.json", "warmup-cache-reset.json",
    "warmup-offload-drain.json", "post-request-drain.json", "selective-store.json",
    "offload-events.json", "timing.json", "resolved-scheduler-config.json",
}


class AuditError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def read(path: Path) -> Any:
    require(path.is_file() and not path.is_symlink(), f"missing or symlinked file: {path}")
    return json.loads(path.read_text())


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def archive_hashes(archive: Path) -> dict[str, str]:
    require(archive.is_dir() and not archive.is_symlink(), f"archive missing: {archive}")
    actual: dict[str, str] = {}
    for directory, dirs, files in os.walk(archive, followlinks=False):
        for name in dirs + files:
            path = Path(directory) / name
            require(not path.is_symlink(), f"archive contains symlink: {path}")
            require(path.is_dir() or path.is_file(), f"archive contains special file: {path}")
        for name in files:
            path = Path(directory) / name
            actual[str(path.relative_to(archive))] = sha_file(path)
    return actual


def model_check(session: Path, plan: dict, receipt: dict) -> dict:
    identity = receipt.get("model_identity")
    require(isinstance(identity, dict) and identity.get("status") == "VERIFIED_PRIVATE_MODEL_VIEW",
            "model qualification incomplete")
    names = ("model-identity-receipt.json", "model-private-view-receipt.json",
             "model-offline-resolution-receipt.json")
    require([identity.get(k) for k in ("shared_source_receipt", "private_view_receipt",
                                      "offline_resolution_receipt")] == list(names),
            "model receipt names differ")
    require(identity.get("verifier_sha256") == plan.get("model_verifier_sha256"),
            "model verifier hash differs")
    shared, private, offline = (read(session / name) for name in names)
    require(shared.get("status") == "VERIFIED_READ_ONLY" and
            shared.get("cache") == plan.get("shared_model_source_cache") and
            shared.get("revision") == REVISION, "shared model receipt differs")
    require(private.get("status") == "VERIFIED_PRIVATE_MODEL_VIEW" and
            private.get("private_cache") == plan.get("hf_cache_dir") and
            private.get("source_cache") == plan.get("shared_model_source_cache") and
            private.get("revision") == REVISION, "private model receipt differs")
    require(shared.get("metadata_sha256") == private.get("metadata_sha256") == MODEL_METADATA_SHA and
            shared.get("shards") == private.get("shards") == MODEL_SHARDS,
            "model hashes differ from pinned revision or across caches")
    require(offline.get("status") == "RESOLVED_OFFLINE" and
            offline.get("hf_home") == plan.get("hf_cache_dir") and
            REVISION in offline.get("path", ""), "offline resolution differs")
    return {"shared_model_receipt_sha256": sha_file(session / names[0]),
            "private_model_receipt_sha256": sha_file(session / names[1]),
            "offline_resolution_receipt_sha256": sha_file(session / names[2])}


def cohort_identity(raw: dict, expected: dict[str, tuple]) -> None:
    require(raw.get("status") == "COMPLETE" and raw.get("error") is None,
            "request capture incomplete")
    require(raw.get("regime") == "steady" and raw.get("arrival_scale") == 1.0,
            "arrival regime differs")
    rows = raw.get("requests")
    require(isinstance(rows, list) and len(rows) == 64, "capture request count differs")
    observed: dict[str, tuple] = {}
    for row in rows:
        rid = row.get("request_id")
        require(isinstance(rid, str) and rid not in observed, "duplicate or invalid request ID")
        observed[rid] = (row.get("document_id"), row.get("prompt_token_ids_sha256"),
                         row.get("arrival_s"), row.get("prompt_tokens"),
                         row.get("max_output_tokens"))
        require(row.get("status") == "completed" and row.get("completion_s") is not None,
                f"request not completed: {rid}")
        require(row.get("stop_reason") in ("stop", "length"),
                f"unexpected stop reason: {rid}")
        require(0 <= len(row.get("output_token_ids", [])) <= 1024,
                f"output length differs from natural EOS/cap contract: {rid}")
    require(observed == expected, "capture cohort/prompt/arrival/cap differs from G64 input")


def cell_check(archive: Path, arm: str, plan: dict, manifest: dict,
               workload_config: dict, cohort: dict[str, tuple]) -> tuple[dict, dict]:
    a = lambda name: read(archive / name)
    config, engine, environment = a("config.json"), a("engine_args.json"), a("environment.json")
    status, raw = a("status.json"), a("raw.json")
    require(status.get("status") == status.get("capture_status") == "COMPLETE" and
            status.get("requests_completed") == 64 and status.get("error") is None,
            f"{arm}: status not complete")
    cohort_identity(raw, cohort)
    require(status.get("finish_reason_counts") ==
            dict(Counter(r["stop_reason"] for r in raw["requests"])),
            f"{arm}: finish reason receipt differs from raw requests")
    required_config = {
        "requests": 64, "workload_sha256": workload_config["workload_sha256"],
        "seed": 20260905, "output_tokens": 1024, "output_mode": "eos",
        "ignore_eos": False, "min_tokens": 0, "cap": 32, "engine_max_num_seqs": 32,
        "max_num_batched_tokens": 1024, "max_seconds": 180,
        "fixed_kv_cache_memory_bytes": KV_BYTES, "offload_gib": 16,
        "measurement_mode": "performance_sparse_preemptions",
    }
    for key, value in required_config.items():
        require(config.get(key) == value, f"{arm}: config differs: {key}")
    require(config.get("model") == workload_config["model"], f"{arm}: model config differs")
    scope = "native_full" if arm == "native_full_native" else "selected"
    variant = "ltr_style_selected" if arm.startswith("ltr_") else arm
    require(config.get("variant") == variant and config.get("store_scope") == scope,
            f"{arm}: policy identity differs")
    if arm == "ltr_t30_q10":
        require(config.get("ltr_config") == {"threshold": 30, "quantum": 10},
                "LTR point differs")
    else:
        require(config.get("ltr_config") is None and config.get("commit_recheck") is False,
                f"{arm}: unexpected LTR or recheck action")
    expected_engine = {
        "model": MODEL_ID, "revision": REVISION, "tokenizer_revision": REVISION,
        "dtype": "bfloat16", "seed": 20260905, "max_model_len": 4096,
        "max_num_seqs": 32, "max_num_batched_tokens": 1024,
        "kv_cache_memory_bytes": KV_BYTES, "kv_offloading_size": 16,
        "kv_offloading_backend": "native", "enable_prefix_caching": False,
        "scheduler_reserve_full_isl": True, "scheduling_policy": "fcfs",
        "async_scheduling": False, "stream_interval": 1,
    }
    for key, value in expected_engine.items():
        require(engine.get(key) == value, f"{arm}: engine differs: {key}")
    require(environment.get("vllm") == "0.26.0", f"{arm}: vLLM version differs")
    gpu_before = environment.get("gpu_before", {})
    require(plan["authorized_gpu_uuid"] in gpu_before.get("device", "") and
            not gpu_before.get("compute_processes"), f"{arm}: GPU isolation differs")
    expected_sources = {name: manifest[f"pkg/{name}"] for name in ARM_SOURCES[arm]}
    require(environment.get("source_sha256") == expected_sources,
            f"{arm}: package source hashes differ")
    runtime = environment.get("vllm_source_sha256")
    require(runtime == PINNED_RUNTIME_SHA,
            f"{arm}: pinned runtime sources differ")
    cap = a("safe-cap-qualification.json")
    require(cap.get("status") == "QUALIFIED" and cap.get("usable_blocks") == 4096 and
            cap.get("total_blocks") == 4097 and cap.get("block_size") == 16 and
            cap.get("engine_max_num_seqs") == 32 and
            cap.get("observed_scheduler_reserve_full_isl") is True,
            f"{arm}: physical GPU KV qualification differs")
    require(a("memory-after-init.json").get("kv_storage_bytes") == KV_BYTES,
            f"{arm}: physical KV storage differs")
    for name in ("host-after-init.json", "host-before.json"):
        host = a(name)
        require(host.get("cpu_kv", {}).get("unique_storage_bytes") == HOST_BYTES and
                host.get("manager", {}).get("capacity_blocks") == 8192,
                f"{arm}: host KV allocation differs in {name}")
        host_limit = host.get("parent_cgroup", {}).get("values", {}).get("memory.max")
        require(type(host_limit) is int and 0 < host_limit <= plan["approved_host_bytes"],
                f"{arm}: host cgroup limit differs in {name}")
    for index, count in enumerate((32, 32, 2)):
        warmup = a(f"warmup-{index}.json")
        require(warmup.get("status") == "COMPLETE" and
                len(warmup.get("requests", [])) == count and
                all(r.get("status") == "completed" for r in warmup["requests"]),
                f"{arm}: warmup {index} incomplete")
    require(a("warmup-cache-reset.json").get("success") is True,
            f"{arm}: warmup cache was not reset")
    for name in ("warmup-offload-drain.json", "post-request-drain.json"):
        drain = a(name)
        require(type(drain.get("calls")) is int and drain["calls"] >= 0 and
                isinstance(drain.get("seconds"), (float, int)) and
                math.isfinite(drain["seconds"]) and drain["seconds"] >= 0,
                f"{arm}: {name} invalid")
    host_after = a("host-after.json")
    require(all(host_after.get("pending", {}).get(key) == 0 for key in
                ("scheduler_store_jobs", "scheduler_load_jobs", "pending_worker_acknowledgements")) and
            host_after.get("manager", {}).get("pending_store_entries") == 0 and
            host_after.get("manager", {}).get("active_load_references") == 0,
            f"{arm}: offload drain left pending work")
    installed = a("selective-store.json")
    require(installed.get("store_scope") == scope and
            installed.get("native_calc_overridden") is (arm != "native_full_native") and
            installed.get("status") in ({"NOT_APPLICABLE"} if arm == "native_full_native" else {"DRAINED"}),
            f"{arm}: installed policy or final drain differs")
    require(installed.get("ltr_config") == ({"threshold": 30, "quantum": 10}
                                                if arm == "ltr_t30_q10" else None),
            f"{arm}: installed LTR config differs")
    commands = shlex.split((archive / "commands.txt").read_text())
    runner = "run_ltr_style.py" if arm.startswith("ltr_") else "run_recovery_cadence.py"
    require(any(x.endswith("/" + runner) or x == runner for x in commands) and
            "--output-dir" in commands and "--measurement-mode" in commands and
            commands[commands.index("--measurement-mode") + 1] == "performance",
            f"{arm}: executed runner command differs")
    return raw, {
        "config_sha256": sha_file(archive / "config.json"),
        "raw_sha256": sha_file(archive / "raw.json"),
        "capture_duration_s": raw["observation_end_s"],
        "forced_rotations": status.get("forced_rotations"),
        "stop_reason_counts": dict(Counter(r["stop_reason"] for r in raw["requests"])),
    }


def output_differences(reference: dict, candidate: dict) -> dict:
    left = {r["request_id"]: r for r in reference["requests"]}
    right = {r["request_id"]: r for r in candidate["requests"]}
    require(left.keys() == right.keys(), "output pair request IDs differ")
    changed = [rid for rid in sorted(left) if left[rid]["output_token_ids"] !=
               right[rid]["output_token_ids"]]
    stop_changed = [rid for rid in sorted(left) if left[rid]["stop_reason"] !=
                    right[rid]["stop_reason"]]
    return {"different_output_sequence_count": len(changed),
            "different_output_sequence_request_ids": changed,
            "different_stop_reason_count": len(stop_changed),
            "different_stop_reason_request_ids": stop_changed}


def audit(session: Path, expected_plan_sha256: str) -> dict:
    require(session.is_dir() and not session.is_symlink(), "session directory missing or symlinked")
    require(SHA_RE.fullmatch(expected_plan_sha256) is not None,
            "external frozen plan SHA-256 required")
    plan_path, receipt_path = session / "plan.json", session / "receipt.json"
    require(sha_file(plan_path) == expected_plan_sha256, "plan bytes differ from external SHA-256")
    plan, receipt = read(plan_path), read(receipt_path)
    require(receipt.get("plan_sha256") == expected_plan_sha256 and
            receipt.get("status") == "CELLS_COMPLETE", "session receipt incomplete")
    require(receipt.get("held_lock_device_inode") == plan.get("expected_lock_device_inode"),
            "shared lock inode receipt differs")
    require(plan.get("model_revision") == REVISION and plan.get("schema_version") == 1,
            "plan model or schema differs")
    plan_cells, receipt_cells = plan.get("cells"), receipt.get("cells")
    require(isinstance(plan_cells, list) and isinstance(receipt_cells, list) and
            len(plan_cells) == len(receipt_cells) == 3 and
            tuple(c.get("arm") for c in plan_cells) == ARMS,
            "three-arm pilot plan differs")
    model = model_check(session, plan, receipt)
    package_root = Path(__file__).parent / "candidate_g64_perf_r01"
    require(sha_file(package_root / "manifest.json") == PACKAGE_SHA,
            "local frozen package manifest differs")
    manifest = read(package_root / "manifest.json")
    workload_config = read(package_root / "pkg/inputs/config.json")
    workload = read(package_root / "pkg/inputs/workload.json")
    require(sha_file(package_root / "pkg/inputs/workload.json") ==
            manifest["pkg/inputs/workload.json"], "local G64 input differs")
    source = workload["source_requests"]
    prompts = workload["actual_prompt_token_ids"]
    arrivals = workload["arrival_traces_s"]["steady"]
    require(len(source) == len(prompts) == len(arrivals) == 64,
            "frozen G64 workload shape differs")
    cohort = {s["request_id"]: (s["document_id"], s["prompt_token_ids_sha256"],
               arrivals[i], len(prompts[i]), 1024) for i, s in enumerate(source)}
    require(len(cohort) == 64, "frozen G64 request IDs duplicate")
    raws: dict[str, dict] = {}
    cell_evidence: dict[str, dict] = {}
    engine_reference = None
    runtime_reference = None
    host_limit_reference = None
    for index, (planned, observed) in enumerate(zip(plan_cells, receipt_cells)):
        arm = ARMS[index]
        require(observed.get("arm") == arm and
                observed.get("output_dir") == planned.get("output_dir") and
                observed.get("argv") == [planned["package_dir"] + "/pkg/run.sh",
                                          arm, planned["output_dir"]] and
                observed.get("launch_status") == "FINISHED" and
                observed.get("exit_code") == 0 and observed.get("timed_out") is False and
                observed.get("archive_status") == "VERIFIED" and
                observed.get("gpu_process_state_after") == "EMPTY" and
                observed.get("gpu_process_rows_after") == [],
                f"{arm}: controller cell receipt incomplete")
        require(planned["package_dir"].endswith("/candidate_g64_perf_r01") and
                planned["max_wall_seconds"] > 6, f"{arm}: package or cell budget differs")
        cell_dir = session / f"cell-{index:02d}-{arm}"
        archive = cell_dir / "archive"
        expected_hashes = read(cell_dir / "output_sha256.json")
        require(isinstance(expected_hashes, dict) and REQUIRED_OUTPUTS <= set(expected_hashes),
                f"{arm}: archive file map incomplete")
        require(all(isinstance(v, str) and SHA_RE.fullmatch(v) for v in expected_hashes.values()),
                f"{arm}: invalid archive hash map")
        require(archive_hashes(archive) == expected_hashes,
                f"{arm}: archive hash/readback mismatch")
        raw, evidence = cell_check(archive, arm, plan, manifest, workload_config, cohort)
        engine = read(archive / "engine_args.json")
        runtime = read(archive / "environment.json")
        host_limit = read(archive / "host-after-init.json")["parent_cgroup"]["values"]["memory.max"]
        if engine_reference is None:
            engine_reference, runtime_reference, host_limit_reference = (
                engine, {k: runtime[k] for k in ("python", "torch", "cuda", "vllm", "transformers",
                                                 "vllm_source_sha256")}, host_limit)
        else:
            require(engine == engine_reference and
                    {k: runtime[k] for k in runtime_reference} == runtime_reference and
                    host_limit == host_limit_reference,
                    f"{arm}: engine/runtime/host budget differs from native arm")
        raws[arm] = raw
        cell_evidence[arm] = {**evidence, "archive_file_count": len(expected_hashes),
                              "archive_sha256_map_sha256": sha_file(cell_dir / "output_sha256.json")}
    metrics = {arm: summarize(raws[arm], 64, 180) for arm in ARMS}
    comparisons = {
        "eager_vs_native_full": {"request_metrics": pair(metrics[ARMS[0]], metrics[ARMS[1]]),
                                 "output_differences": output_differences(raws[ARMS[0]], raws[ARMS[1]])},
        "ltr_t30_q10_vs_eager": {"request_metrics": pair(metrics[ARMS[1]], metrics[ARMS[2]]),
                                 "output_differences": output_differences(raws[ARMS[1]], raws[ARMS[2]])},
        "ltr_t30_q10_vs_native_full": {"request_metrics": pair(metrics[ARMS[0]], metrics[ARMS[2]]),
                                       "output_differences": output_differences(raws[ARMS[0]], raws[ARMS[2]])},
    }
    eager, ltr = metrics["eager"], metrics["ltr_t30_q10"]
    budget = (ltr["actual_output_tokens_s"] >= .97 * eager["actual_output_tokens_s"] and
              ltr["mean_completed_flow_s"] <= 1.05 * eager["mean_completed_flow_s"])
    return {
        "schema_version": 1, "status": "PILOT_THREE_ARM_COMPLETE_PARTIAL_TQ_GRID",
        "plan_sha256": expected_plan_sha256, "receipt_sha256": sha_file(receipt_path),
        "package_manifest_sha256": PACKAGE_SHA, "model_receipts": model,
        "evidence_ceiling": "NATIVE_SERVING_INPROCESS_HOST_MEASUREMENT",
        "calibration_status": "NOT_SELECTABLE_PARTIAL_GRID",
        "pilot_t30_q10_within_eager_service_budget": budget,
        "scope": "One same-machine G64 block; no controlled repeat, no statistical claim, no H128 transfer result, no full LTR claim.",
        "work_semantics": "Natural EOS/output lengths and sequences may differ by policy; token-rate and completion differences are not equal-work speedups.",
        "denominator": "Each arm's own observed capture duration; fixed development goodput thresholds, not production SLO.",
        "cells": cell_evidence, "metrics": metrics, "comparisons": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    session = args.session.resolve(strict=True)
    output = args.output.resolve(strict=False)
    require(not output.exists() and output != session and session not in output.parents,
            "new audit JSON must be outside the immutable session")
    result = audit(session, args.expected_plan_sha256)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
