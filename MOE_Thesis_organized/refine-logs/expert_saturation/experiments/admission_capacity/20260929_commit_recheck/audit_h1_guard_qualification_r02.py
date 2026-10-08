#!/usr/bin/env python3
"""Read-only audit of one H128 H1 eager/diagnostic/on native qualification."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import shlex
from typing import Any

import audit_h128_guarded_transfer_r02 as common
from analyze_h1_direct_action_chain import analyze as analyze_action_chain


AuditError = common.AuditError
require = common.require
read = common.read
sha_file = common.sha_file
archive_hashes = common.archive_hashes

SESSION_NAME = "moe-a-h1-guard-qual-session-r02-20260930"
ARM = "eager_diagnostic_on"
PACKAGE_NAME = "candidate_h1_guard_qual_r02"
PACKAGE_SHA = "4231a687db066f113fcc14676f91e8e5825be05b76ef0834112e053726aa13aa"
PLAN_SHA = "48c6fe98eddfca76faec95676184b3ca24a9cf7ce81a24567a825eaddf509a81"
EXPECTED_GPU_UUID = "GPU-3fc910c2-bf65-5273-e6b5-6c0d8b6ce03e"
EXPECTED_LOCK_INODE = "2304:25841682495"
EXPECTED_PYTHON = "/root/autodl-tmp/moe-a-runtime-20260930/venv/bin/python"
EXPECTED_MODEL_CACHE = "/root/moe-a-model-cache-20260930-r02/hf"
EXPECTED_SOURCE_CACHE = "/root/autodl-tmp/c-research-20260930/hf-cache"
EXPECTED_MODEL_VERIFIER = "/root/autodl-tmp/moe-a-shared-assets-launch-20260930-v2/verify_shared_model_20260930.py"
EXPECTED_MODEL_VERIFIER_SHA = "d78c6726a8d010f3a02d50c79220084520c186f9b05f4b4a10c1598ac5fe9a90"
EXPECTED_SESSION = "/root/moe-a-h1-guard-qual-session-r02-20260930"
EXPECTED_OUTPUT = "/root/moe-a-h1-guard-qual-output-eager-on-r02-20260930"
EXPECTED_PACKAGE = "/root/autodl-tmp/moe-a-pkg-20260930/candidate_h1_guard_qual_r02"
REQUIRED_OUTPUTS = common.REQUIRED_OUTPUTS | {
    "metrics.json", "resolved-eos.json", "memory-before.json", "memory-after.json",
    "gpu-after.json",
}
SOURCE_NAMES = common.ARM_SOURCES["eager"]
RAW_STREAM_THRESHOLD = 64 * 1024 * 1024
RAW_RETAINED_ARRAYS = ("requests", "scheduler_steps", "engine_steps")
RAW_COUNTED_ARRAYS = ("memory_trace", "output_events")
RAW_SCALARS = (
    "measurement_origin_perf_counter_s", "status", "error", "regime",
    "arrival_scale", "observation_end_s",
)


def read_raw_projection(path: Path, *, force_stream: bool = False) -> tuple[dict, dict]:
    """Read evidence needed for qualification, without retaining cumulative output events.

    The caller separately hashes the entire original archive byte for byte. The
    returned JSON projection is evidence for predicates, never a substitute hash.
    """
    require(path.is_file() and not path.is_symlink(), f"missing or symlinked file: {path}")
    if path.stat().st_size <= RAW_STREAM_THRESHOLD and not force_stream:
        source = read(path)
        require(isinstance(source, dict), "H1 raw root is not an object")
        projection = {key: source.get(key) for key in RAW_SCALARS + RAW_RETAINED_ARRAYS}
        counts = {key: len(source[key]) if isinstance(source.get(key), list) else -1
                  for key in RAW_RETAINED_ARRAYS + RAW_COUNTED_ARRAYS}
        return projection, {"reader": "stdlib_small_fixture", "counts": counts,
                            "retained_arrays": list(RAW_RETAINED_ARRAYS),
                            "counted_only_arrays": list(RAW_COUNTED_ARRAYS)}

    try:
        import ijson
    except ImportError as error:
        raise AuditError("ijson is required to stream H1 raw.json above 64 MiB") from error
    projection: dict[str, Any] = {key: [] for key in RAW_RETAINED_ARRAYS}
    counts = {key: 0 for key in RAW_RETAINED_ARRAYS + RAW_COUNTED_ARRAYS}
    seen_arrays: set[str] = set()
    seen_scalars: set[str] = set()
    builder = None
    building: str | None = None
    with path.open("rb") as stream:
        for prefix, event, value in ijson.parse(stream, use_float=True):
            if building is not None:
                builder.event(event, value)
                if prefix == f"{building}.item" and event == "end_map":
                    projection[building].append(builder.value)
                    counts[building] += 1
                    builder = None
                    building = None
                continue
            if prefix in RAW_SCALARS and event in ("string", "number", "boolean", "null"):
                require(prefix not in seen_scalars, f"H1 raw scalar duplicate: {prefix}")
                projection[prefix] = value
                seen_scalars.add(prefix)
            elif prefix in RAW_RETAINED_ARRAYS + RAW_COUNTED_ARRAYS and event == "start_array":
                require(prefix not in seen_arrays, f"H1 raw array duplicate: {prefix}")
                seen_arrays.add(prefix)
            elif prefix.endswith(".item") and event == "start_map":
                array_name = prefix[:-5]
                if array_name in RAW_RETAINED_ARRAYS:
                    builder = ijson.common.ObjectBuilder()
                    building = array_name
                    builder.event(event, value)
                elif array_name in RAW_COUNTED_ARRAYS:
                    counts[array_name] += 1
    require(building is None and seen_arrays == set(RAW_RETAINED_ARRAYS + RAW_COUNTED_ARRAYS),
            "H1 raw diagnostic array missing or malformed")
    require(seen_scalars == set(RAW_SCALARS), "H1 raw diagnostic scalar missing")
    return projection, {"reader": "ijson_single_pass", "counts": counts,
                        "retained_arrays": list(RAW_RETAINED_ARRAYS),
                        "counted_only_arrays": list(RAW_COUNTED_ARRAYS)}


def verify_local_package() -> dict[str, str]:
    package = Path(__file__).parent / PACKAGE_NAME
    require(sha_file(package / "manifest.json") == PACKAGE_SHA,
            "local H1 package manifest differs")
    manifest = read(package / "manifest.json")
    require(isinstance(manifest, dict) and len(manifest) == 25,
            "H1 package manifest must list 25 payload files")
    actual: set[str] = set()
    for path in package.rglob("*"):
        if path.is_symlink():
            require(False, f"H1 package symlink: {path}")
        if path.is_file() and (path.is_relative_to(package / "pkg") or
                               path.name == "verify_package.py"):
            actual.add(path.relative_to(package).as_posix())
    require(actual == set(manifest), "H1 package payload file set differs")
    for name, expected in manifest.items():
        require(common.SHA_RE.fullmatch(expected) is not None and
                sha_file(package / name) == expected,
                f"H1 package payload differs: {name}")
    return manifest


def check_plan_and_receipt(session: Path, external_plan_sha: str) -> tuple[dict, dict, dict]:
    require(session.is_dir() and not session.is_symlink() and session.name == SESSION_NAME,
            "wrong H1 diagnostic session identity")
    require(external_plan_sha == PLAN_SHA, "H1 plan differs from frozen SHA-256")
    plan_path, receipt_path = session / "plan.json", session / "receipt.json"
    require(sha_file(plan_path) == external_plan_sha,
            "H1 session plan bytes differ from external SHA-256")
    plan, receipt = read(plan_path), read(receipt_path)
    require(receipt.get("plan_sha256") == external_plan_sha and
            receipt.get("status") == "CELLS_COMPLETE" and
            receipt.get("held_lock_device_inode") == plan.get("expected_lock_device_inode"),
            "H1 controller receipt incomplete")
    required_plan = {
        "schema_version": 1,
        "authorized_gpu_uuid": EXPECTED_GPU_UUID,
        "approved_host_bytes": 96636764160,
        "approved_total_wall_seconds": 2100,
        "lock_path": "/root/autodl-tmp/moe-research-gpu.lock",
        "expected_lock_device_inode": EXPECTED_LOCK_INODE,
        "session_dir": EXPECTED_SESSION,
        "python": EXPECTED_PYTHON,
        "hf_cache_dir": EXPECTED_MODEL_CACHE,
        "shared_model_source_cache": EXPECTED_SOURCE_CACHE,
        "model_revision": common.REVISION,
        "model_verifier_path": EXPECTED_MODEL_VERIFIER,
        "model_verifier_sha256": EXPECTED_MODEL_VERIFIER_SHA,
        "cgroup_memory_max_file": "/sys/fs/cgroup/memory.max",
    }
    for key, expected in required_plan.items():
        require(plan.get(key) == expected, f"H1 plan differs: {key}")
    cells, observed = plan.get("cells"), receipt.get("cells")
    require(isinstance(cells, list) and isinstance(observed, list) and
            len(cells) == len(observed) == 1,
            "H1 qualification must be one controller cell")
    cell, result = cells[0], observed[0]
    require(cell.get("arm") == result.get("arm") == ARM and
            cell.get("package_dir") == EXPECTED_PACKAGE and
            cell.get("output_dir") == result.get("output_dir") == EXPECTED_OUTPUT and
            cell.get("max_wall_seconds") == 900,
            "H1 diagnostic cell identity differs")
    expected_argv = [EXPECTED_PACKAGE + "/pkg/run.sh", "eager", "diagnostic", "on",
                     EXPECTED_OUTPUT]
    require(result.get("argv") == expected_argv and
            result.get("launch_status") == "FINISHED" and
            result.get("exit_code") == 0 and result.get("timed_out") is False and
            result.get("archive_status") == "VERIFIED" and
            result.get("gpu_process_state_after") == "EMPTY" and
            result.get("gpu_process_rows_after") == [],
            "H1 controller cell receipt incomplete")
    return plan, receipt, result


def check_cell(archive: Path, plan: dict, manifest: dict[str, str]) -> tuple[dict, dict, dict]:
    a = lambda name: read(archive / name)
    package = Path(__file__).parent / PACKAGE_NAME
    workload_config = read(package / "pkg/inputs/config.json")
    workload = read(package / "pkg/inputs/workload.json")
    require(sha_file(package / "pkg/inputs/workload.json") ==
            manifest["pkg/inputs/workload.json"], "H1 frozen workload differs")
    sources = workload.get("source_requests")
    prompts = workload.get("actual_prompt_token_ids")
    arrivals = workload.get("arrival_traces_s", {}).get("steady")
    require(all(isinstance(x, list) and len(x) == 128 for x in
                (sources, prompts, arrivals)), "H1 input shape differs")
    cohort = {source["request_id"]: (
        source["document_id"], source["prompt_token_ids_sha256"],
        arrivals[index], len(prompts[index]), 1024,
    ) for index, source in enumerate(sources)}
    require(len(cohort) == 128, "H1 frozen request IDs duplicate")

    status, config = a("status.json"), a("config.json")
    raw, raw_projection = read_raw_projection(archive / "raw.json")
    require(status.get("status") == status.get("capture_status") == "COMPLETE" and
            status.get("requests_completed") == 128 and status.get("error") is None,
            "H1 diagnostic status incomplete")
    common.cohort_identity(raw, cohort)
    require(status.get("finish_reason_counts") ==
            dict(Counter(row["stop_reason"] for row in raw["requests"])),
            "H1 stop-reason receipt differs from raw")
    required_config = {
        "requests": 128, "workload_sha256": workload_config["workload_sha256"],
        "seed": 20260905, "output_tokens": 1024, "output_mode": "eos",
        "ignore_eos": False, "min_tokens": 0, "cap": 32, "engine_max_num_seqs": 32,
        "max_num_batched_tokens": 1024, "max_seconds": 180,
        "fixed_kv_cache_memory_bytes": common.KV_BYTES, "offload_gib": 16,
        "measurement_mode": "diagnostic", "variant": "eager", "store_scope": "selected",
        "commit_recheck": True, "population_mode": "open",
        "global_cooldown_steps": 0, "rotation_victim_order": "most_output",
    }
    for key, expected in required_config.items():
        require(config.get(key) == expected, f"H1 diagnostic config differs: {key}")
    require(config.get("model") == workload_config["model"] and
            config.get("rotation_config", {}).get("min_steps_between_swaps") == 0,
            "H1 diagnostic model or simple eager rule differs")

    engine, environment = a("engine_args.json"), a("environment.json")
    expected_engine = {
        "model": common.MODEL_ID, "revision": common.REVISION,
        "tokenizer_revision": common.REVISION, "dtype": "bfloat16",
        "seed": 20260905, "max_model_len": 4096, "max_num_seqs": 32,
        "max_num_batched_tokens": 1024, "kv_cache_memory_bytes": common.KV_BYTES,
        "kv_offloading_size": 16, "kv_offloading_backend": "native",
        "enable_prefix_caching": False, "scheduler_reserve_full_isl": True,
        "scheduling_policy": "fcfs", "async_scheduling": False,
        "stream_interval": 1,
    }
    for key, expected in expected_engine.items():
        require(engine.get(key) == expected, f"H1 engine differs: {key}")
    require(environment.get("vllm") == "0.26.0" and
            plan["authorized_gpu_uuid"] in environment.get("gpu_before", {}).get("device", "") and
            not environment.get("gpu_before", {}).get("compute_processes"),
            "H1 runtime or GPU isolation differs")
    require(environment.get("source_sha256") ==
            {name: manifest[f"pkg/{name}"] for name in SOURCE_NAMES} and
            environment.get("vllm_source_sha256") == common.PINNED_RUNTIME_SHA,
            "H1 executed source or pinned runtime differs")

    cap = a("safe-cap-qualification.json")
    require(cap.get("status") == "QUALIFIED" and cap.get("usable_blocks") == 4096 and
            cap.get("total_blocks") == 4097 and cap.get("block_size") == 16 and
            cap.get("engine_max_num_seqs") == 32 and
            cap.get("observed_scheduler_reserve_full_isl") is True and
            a("memory-after-init.json").get("kv_storage_bytes") == common.KV_BYTES,
            "H1 physical GPU KV qualification differs")
    for name in ("host-after-init.json", "host-before.json"):
        host = a(name)
        host_limit = host.get("parent_cgroup", {}).get("values", {}).get("memory.max")
        require(host.get("cpu_kv", {}).get("unique_storage_bytes") == common.HOST_BYTES and
                host.get("manager", {}).get("capacity_blocks") == 8192 and
                type(host_limit) is int and 0 < host_limit <= plan["approved_host_bytes"],
                f"H1 host KV or cgroup differs in {name}")
    for index, count in enumerate((32, 32, 2)):
        warmup = a(f"warmup-{index}.json")
        require(warmup.get("status") == "COMPLETE" and
                len(warmup.get("requests", [])) == count and
                all(row.get("status") == "completed" for row in warmup["requests"]),
                f"H1 warmup {index} incomplete")
    require(a("warmup-cache-reset.json").get("success") is True,
            "H1 warmup connector cache was not reset")
    for name in ("warmup-offload-drain.json", "post-request-drain.json"):
        drain = a(name)
        require(type(drain.get("calls")) is int and drain["calls"] >= 0 and
                isinstance(drain.get("seconds"), (float, int)) and
                math.isfinite(drain["seconds"]) and drain["seconds"] >= 0,
                f"H1 {name} invalid")
    host_after = a("host-after.json")
    require(all(host_after.get("pending", {}).get(key) == 0 for key in
                ("scheduler_store_jobs", "scheduler_load_jobs",
                 "pending_worker_acknowledgements")) and
            host_after.get("manager", {}).get("pending_store_entries") == 0 and
            host_after.get("manager", {}).get("active_load_references") == 0,
            "H1 offload drain left pending work")
    installed, offload = a("selective-store.json"), a("offload-events.json")
    require(installed.get("status") == "DRAINED" and
            installed.get("store_scope") == "selected" and
            installed.get("native_calc_overridden") is True and
            installed.get("commit_recheck") is True and
            installed.get("diagnostic") is True and
            installed.get("population_mode") == "open" and
            installed.get("rotation_config") == config.get("rotation_config"),
            "H1 on adapter not installed or drained as requested")
    require(offload.get("diagnostic") is True and
            offload.get("detailed_logging") == "ENABLED" and
            offload.get("worker_wrapped") is True and
            all(isinstance(offload.get(key), list) for key in
                ("lookup", "transfers", "completed_jobs", "dispatch", "host_snapshots")) and
            not any("error" in row for row in offload["host_snapshots"]),
            "H1 diagnostic offload observer incomplete")
    require(status.get("direct_commits") == installed.get("direct_commits") and
            a("metrics.json").get("direct_commits") == installed.get("direct_commits") and
            a("metrics.json").get("complete_episode_comparison_eligible") is True,
            "H1 direct count or request completion receipt differs")
    trace_counts = raw_projection["counts"]
    require(all(isinstance(raw.get(key), list) and raw[key] and
                len(raw[key]) == trace_counts[key] for key in
                ("scheduler_steps", "engine_steps")) and
            trace_counts["memory_trace"] > 0 and
            isinstance(raw.get("measurement_origin_perf_counter_s"), (float, int)) and
            math.isfinite(raw["measurement_origin_perf_counter_s"]),
            "H1 diagnostic scheduler/engine trace or memory trace count missing")
    steps = [row.get("step") for row in raw["scheduler_steps"]]
    require(all(type(step) is int for step in steps) and len(steps) == len(set(steps)),
            "H1 diagnostic scheduler step identities duplicate or invalid")
    commands = shlex.split((archive / "commands.txt").read_text())
    require(any(token.endswith("/run_recovery_cadence.py") or
                token == "run_recovery_cadence.py" for token in commands) and
            all(flag in commands for flag in ("--variant", "--measurement-mode",
                                             "--commit-recheck", "--output-dir")) and
            commands[commands.index("--variant") + 1] == "eager" and
            commands[commands.index("--measurement-mode") + 1] == "diagnostic" and
            commands[commands.index("--output-dir") + 1] == EXPECTED_OUTPUT,
            "H1 executed diagnostic-on command differs")
    try:
        action = analyze_action_chain(raw, installed, offload)
    except (KeyError, TypeError, ValueError) as error:
        raise AuditError(f"H1 direct action chain invalid: {error}") from error
    require(action["status"] in ("NO_ACTION", "OBSERVED_DIRECT_ACTION_CHAIN"),
            "H1 action result unknown")
    return raw, action, raw_projection


def audit(session: Path, expected_plan_sha256: str) -> dict[str, Any]:
    plan, receipt, _ = check_plan_and_receipt(session, expected_plan_sha256)
    require({path.name for path in session.glob("cell-*")} == {f"cell-00-{ARM}"},
            "H1 session contains unexpected controller cells")
    model = common.model_check(session, plan, receipt)
    manifest = verify_local_package()
    cell_dir = session / f"cell-00-{ARM}"
    archive = cell_dir / "archive"
    expected_hashes = read(cell_dir / "output_sha256.json")
    require(isinstance(expected_hashes, dict) and REQUIRED_OUTPUTS <= set(expected_hashes) and
            all(isinstance(value, str) and common.SHA_RE.fullmatch(value) for value in expected_hashes.values()),
            "H1 archive hash map incomplete")
    require(archive_hashes(archive) == expected_hashes,
            "H1 archive file hash/readback mismatch")
    launch_log = cell_dir / "launch.log"
    require(launch_log.is_file() and not launch_log.is_symlink() and
            "Candidate payload verified: 25 files" in launch_log.read_text(errors="replace"),
            "H1 runner did not report 25/25 manifest verification")
    raw, action, raw_projection = check_cell(archive, plan, manifest)
    return {
        "schema_version": 1,
        "status": action["status"],
        "qualification": "NO_ACTION_STOP" if action["status"] == "NO_ACTION" else
                         "OBSERVED_NATIVE_DIRECT_AND_LATER_OUTPUT",
        "plan_sha256": expected_plan_sha256,
        "receipt_sha256": sha_file(session / "receipt.json"),
        "package_manifest_sha256": PACKAGE_SHA,
        "model_receipts": model,
        "archive_sha256_map_sha256": sha_file(cell_dir / "output_sha256.json"),
        "archive_file_count": len(expected_hashes),
        "raw_sha256": expected_hashes["raw.json"],
        "raw_projection": raw_projection,
        "requests_completed": len(raw["requests"]),
        "direct_action": action,
        "evidence_ceiling": "NATIVE_SERVING_INPROCESS_HOST_MEASUREMENT",
        "scope": "One seen-H128-input eager/diagnostic/on action qualification. No off/on causal comparison, performance ranking, blind test, statistical claim, or H1 method benefit.",
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
