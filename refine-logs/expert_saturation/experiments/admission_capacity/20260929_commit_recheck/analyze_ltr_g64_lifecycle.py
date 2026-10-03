#!/usr/bin/env python3
"""Read-only audit of one frozen G64/T30/Q10 LTR native diagnostic.

OBSERVED_CHAIN means that one recovery episode has joined native store and
load receipts, a returned new output, and positive service after that output.
It is not a performance comparison, a tensor-value check, or full LTR.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
PACKAGE = HERE / "candidate_ltr_r02_env"
MANIFEST_SHA256 = "f58340cd5228ea7339f0d4081176aab265583cb824a26026d19b8f7ae93cc8a7"
KV_BYTES = 8_592_031_744
HOST_BYTES = 16 * 1024**3
RUNTIME_OBSERVED_SOURCES = (
    "v1/core/sched/scheduler.py", "v1/core/kv_cache_manager.py",
    "v1/core/block_pool.py", "v1/worker/gpu_model_runner.py",
    "v1/core/kv_cache_coordinator.py", "v1/core/single_type_kv_cache_manager.py",
    "v1/core/kv_cache_utils.py", "config/model.py",
)
# The runner records this core-source set. Its separate preflight checks the
# offloading-source set in pkg/runtime_source_hashes.json before GPU loading.
RUNTIME_PINNED_CORE = {
    "v1/core/sched/scheduler.py": "2ed2a550b6558b2495eda845a97ae38bcf0225027b9e25fbf00fc3880c1d3941",
    "v1/core/kv_cache_manager.py": "3f4af8d247f3fe9570b0132818b832b66ae6a2ac12942588828f899f6ff77ccf",
}
REQUIRED_JSON = (
    "config.json", "status.json", "raw.json", "selective-store.json",
    "offload-events.json", "environment.json", "engine_args.json",
    "safe-cap-qualification.json", "memory-after-init.json", "memory-before.json",
    "memory-after.json", "host-after-init.json", "host-before.json",
    "host-request-end.json", "host-after.json", "post-request-drain.json",
    "warmup-0.json", "warmup-1.json", "warmup-2.json",
    "warmup-offload-drain.json", "warmup-cache-reset.json",
    "resolved-eos.json", "resolved-scheduler-config.json", "timing.json",
    "metrics.json", "gpu-after.json",
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def get(obj: Any, *keys: Any) -> Any:
    for key in keys:
        if not isinstance(obj, dict):
            return None
        obj = obj.get(key)
    return obj


def number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _audit(directory: Path) -> dict[str, Any]:
    directory = directory.resolve()
    issues: list[str] = []
    unknown: list[str] = [
        "Independent KV tensor-value equality is not captured by these receipts.",
        "A diagnostic cell cannot establish a request-level performance benefit.",
        "No external archive/readback manifest was supplied to authenticate result-file hashes.",
    ]
    files: dict[str, dict[str, Any]] = {}
    docs: dict[str, Any] = {}
    report: dict[str, Any] = dict(
        schema_version=1, result_dir=str(directory), verdict="INCOMPLETE",
        evidence_ceiling="NATIVE_SERVING_DIAGNOSTIC",
        files=files, checks={}, counts={}, observed_chains=[],
        coverage={}, issues=issues, unknown=unknown,
    )

    def require(condition: Any, message: str) -> bool:
        if not condition:
            issues.append(message)
            return False
        return True

    # Read every expected result file without changing the capture directory.
    for name in (*REQUIRED_JSON, "commands.txt"):
        path = directory / name
        row: dict[str, Any] = dict(present=path.is_file())
        files[name] = row
        if not row["present"]:
            issues.append(f"missing result file: {name}")
            continue
        try:
            row.update(bytes=path.stat().st_size, sha256=digest(path))
            if name.endswith(".json"):
                docs[name] = json.loads(path.read_text())
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            row["parse_error"] = f"{type(exc).__name__}: {exc}"
            issues.append(f"unreadable result file: {name}")
    if any(name not in docs for name in REQUIRED_JSON) or not files["commands.txt"]["present"]:
        return report

    # The output receipts must identify the frozen package and native runtime.
    try:
        manifest_path = PACKAGE / "manifest.json"
        require(digest(manifest_path) == MANIFEST_SHA256,
                "local frozen candidate manifest identity is unavailable or changed")
        manifest = json.loads(manifest_path.read_text())
        frozen_input = json.loads((PACKAGE / "pkg/inputs/config.json").read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        issues.append(f"cannot read frozen package reference: {type(exc).__name__}: {exc}")
        return report

    config, status, raw = (docs[name] for name in ("config.json", "status.json", "raw.json"))
    selective = docs["selective-store.json"]
    offload = docs["offload-events.json"]
    env, args = docs["environment.json"], docs["engine_args.json"]
    safety = docs["safe-cap-qualification.json"]
    if not all(isinstance(row, dict) for row in (
            config, status, raw, selective, offload, env, args, safety)):
        issues.append("one or more top-level result JSON values are not objects")
        return report
    checks = report["checks"]

    identity_fields = {
        "workload_sha256": frozen_input["workload_sha256"],
        "requests": 64, "arrival_gap_s": 0.2, "output_tokens": 1024,
        "output_mode": "eos", "ignore_eos": False, "min_tokens": 0,
        "cap": 32, "engine_max_num_seqs": 32,
        "measurement_mode": "diagnostic", "variant": "ltr_style_selected",
        "store_scope": "selected", "selective_save": "on", "offload_gib": 16,
        "fixed_kv_cache_memory_bytes": KV_BYTES,
        "ltr_config": {"threshold": 30, "quantum": 10},
    }
    checks["identity"] = all(config.get(k) == v for k, v in identity_fields.items()) if isinstance(config, dict) else False
    require(checks["identity"], "G64/T30/Q10 input or diagnostic configuration differs")
    require(get(config, "model") == frozen_input["model"], "model/revision identity differs")
    require(get(args, "model") == get(config, "model", "id")
            and get(args, "revision") == get(config, "model", "revision")
            and get(args, "tokenizer_revision") == get(config, "model", "tokenizer_revision")
            and get(args, "max_num_seqs") == 32 and get(args, "max_num_batched_tokens") == 1024
            and get(args, "kv_cache_memory_bytes") == KV_BYTES
            and get(args, "kv_offloading_size") == 16
            and get(args, "enable_prefix_caching") is False
            and get(args, "async_scheduling") is False,
            "engine arguments differ from the frozen native resource contract")
    runtime_sources = get(env, "vllm_source_sha256")
    checks["runtime_source"] = (
        get(env, "vllm") == "0.26.0"
        and isinstance(runtime_sources, dict)
        and set(runtime_sources) == set(RUNTIME_OBSERVED_SOURCES)
        and all(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)
                for value in runtime_sources.values())
        and all(runtime_sources.get(name) == value
                for name, value in RUNTIME_PINNED_CORE.items())
    )
    require(checks["runtime_source"], "recorded core vLLM runtime hashes/version differ")
    source_hashes = get(env, "source_sha256")
    require(isinstance(source_hashes, dict) and bool(source_hashes)
            and all(manifest.get(f"pkg/{name}") == sha for name, sha in source_hashes.items())
            and all(name in source_hashes for name in (
                "run_ltr_style.py", "ltr_style_native.py", "ltr_style_selected.py",
                "recovery_service_components.py", "native_capture.py",
                "native_offload_observer.py", "native_store_delta.py")),
            "scientific source receipts differ from frozen package")
    def host_kv_snapshot_usable(name: str) -> bool:
        snapshot = docs[name]
        status = get(snapshot, "status")
        errors = get(snapshot, "errors")
        peak_error_only = (
            status == "PARTIAL" and isinstance(errors, dict)
            and set(errors) == {"memory.peak"}
            and isinstance(errors["memory.peak"], str)
            and "FileNotFoundError" in errors["memory.peak"]
            and get(snapshot, "parent_cgroup", "values", "memory.peak") is None
        )
        memory_current = get(snapshot, "parent_cgroup", "values", "memory.current")
        return (
            (status == "COMPLETE" or peak_error_only)
            and get(snapshot, "parent_cgroup", "values", "memory.max") == 96636764160
            and get(snapshot, "parent_cgroup", "values", "memory.swap.max") == 0
            and number(memory_current) and 0 <= memory_current <= 96636764160
            and get(snapshot, "cpu_kv", "unique_storage_bytes") == HOST_BYTES
            and get(snapshot, "manager", "capacity_blocks") == 8192
        )

    checks["physical_resources"] = (
        get(safety, "status") == "QUALIFIED"
        and get(safety, "usable_blocks") == 4096
        and get(safety, "block_size") == 16
        and get(safety, "observed_scheduler_reserve_full_isl") is True
        and get(docs["memory-after-init.json"], "kv_storage_bytes") == KV_BYTES
        and host_kv_snapshot_usable("host-after-init.json")
        and get(docs["resolved-scheduler-config.json"], "max_num_running_reqs") == 32
    )
    require(checks["physical_resources"], "actual GPU/host KV, block size, or scheduler cap differs")
    for name in ("warmup-0.json", "warmup-1.json", "warmup-2.json"):
        require(get(docs[name], "status") == "COMPLETE", f"warmup incomplete: {name}")
    require(get(docs["warmup-cache-reset.json"], "success") is True,
            "warmup connector cache reset missing")
    require(get(docs["resolved-eos.json"], "status") == "READ",
            "native EOS metadata unavailable")
    for name in ("host-before.json", "host-request-end.json", "host-after.json"):
        require(host_kv_snapshot_usable(name), f"host KV snapshot lacks required evidence: {name}")
    if any(get(docs[name], "status") == "PARTIAL" for name in (
            "host-after-init.json", "host-before.json", "host-request-end.json", "host-after.json")):
        unknown.append("Cgroup memory.peak is absent in the container; host peak memory cannot be certified from these snapshots.")
    after = docs["host-after.json"]
    require(all(get(after, "pending", key) == 0 for key in (
                "scheduler_store_jobs", "scheduler_load_jobs", "pending_worker_acknowledgements"))
            and all(get(after, "manager", key) == 0 for key in (
                "pending_store_entries", "recorded_pending_store_blocks"))
            and all(get(after, "cpu_kv", key) == 0 for key in (
                "worker_store_pending_events", "worker_load_pending_events",
                "worker_unsubmitted_store_jobs", "worker_load_jobs")),
            "native host store/load/ack work remains after drain")
    require(number(get(docs["post-request-drain.json"], "seconds")),
            "post-request drain receipt missing")
    timing = docs["timing.json"]
    require(all(number(timing.get(k)) for k in (
                "measurement_start_perf_s", "measurement_return_perf_s",
                "post_request_drain_end_perf_s", "process_end_perf_s"))
            and timing["measurement_start_perf_s"] <= timing["measurement_return_perf_s"]
            <= timing["post_request_drain_end_perf_s"] <= timing["process_end_perf_s"],
            "measurement/drain timing receipt is incomplete or unordered")

    requests = get(raw, "requests")
    output_events = get(raw, "output_events")
    steps = get(raw, "scheduler_steps")
    engine_steps = get(raw, "engine_steps")
    events = get(selective, "events")
    require(get(status, "status") == "COMPLETE" and get(status, "capture_status") == "COMPLETE"
            and get(status, "requests_completed") == 64 and get(raw, "status") == "COMPLETE"
            and get(raw, "error") is None, "capture/status is not a complete 64-request episode")
    require(isinstance(requests, list) and len(requests) == 64
            and all(isinstance(r, dict) and r.get("status") == "completed" for r in requests),
            "raw requests are missing, failed, or unfinished")
    require(isinstance(output_events, list) and isinstance(steps, list)
            and isinstance(engine_steps, list) and isinstance(events, list),
            "native output/scheduler/policy events missing")
    require(isinstance(get(raw, "memory_trace"), list)
            and isinstance(get(raw, "preemption_events"), list),
            "native pool/preemption diagnostic capture missing")
    require(get(selective, "status") == "DRAINED" and get(selective, "store_scope") == "selected"
            and get(selective, "native_calc_overridden") is True
            and get(selective, "ltr_config") == {"threshold": 30, "quantum": 10}
            and get(selective, "active_target") is None and get(selective, "pending_plan") is None,
            "LTR selected adapter did not drain cleanly")
    require(get(offload, "diagnostic") is True and get(offload, "worker_wrapped") is True
            and get(offload, "detailed_logging") == "ENABLED"
            and all(isinstance(offload.get(k), list) for k in (
                "lookup", "dispatch", "completed_jobs", "transfers")),
            "native diagnostic worker observer is missing")
    if not all(isinstance(x, list) for x in (requests, output_events, steps, engine_steps, events)):
        return report
    require(get(selective, "schedule_calls") == len(steps)
            and len(engine_steps) == len(steps)
            and all(get(s, "step") == i for i, s in enumerate(steps))
            and all(get(s, "call_index") == i and get(s, "completed") is True
                    and get(s, "scheduler_step_start") == i
                    and get(s, "scheduler_step_end") == i + 1
                    for i, s in enumerate(engine_steps)),
            "adapter, native engine calls and raw scheduler steps do not align")

    request_by_internal = {}
    stop_counts = Counter()
    returned_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for e in output_events:
        if isinstance(e, dict) and isinstance(e.get("request_id"), str):
            returned_by_source[e["request_id"]].append(e)
    for r in requests:
        internal = r.get("internal_request_id")
        source = r.get("request_id")
        if not isinstance(internal, str) or not isinstance(source, str) or internal in request_by_internal:
            issues.append("raw internal/source request identity missing or duplicated")
            continue
        request_by_internal[internal] = r
        ids, times = r.get("output_token_ids"), r.get("token_times_s")
        if not isinstance(ids, list) or not isinstance(times, list) or not ids or len(ids) != len(times):
            issues.append(f"returned output/token times incomplete: {source}")
        receipts = returned_by_source[source]
        if (not receipts or not any(type(e.get("chunk_size")) is int and e["chunk_size"] > 0 for e in receipts)
                or receipts[-1].get("cumulative_token_ids") != ids):
            issues.append(f"returned output event history incomplete: {source}")
        reason = r.get("stop_reason")
        stop_counts[reason] += 1
        if reason not in ("stop", "length") or reason == "length" and len(ids if isinstance(ids, list) else []) != 1024:
            issues.append(f"invalid natural-EOS/length completion: {source}")
    require(len(request_by_internal) == 64 and get(raw, "internal_to_source") == {
                k: v["request_id"] for k, v in request_by_internal.items()},
            "raw internal-to-source mapping differs")
    require(sum(1 for e in output_events if isinstance(e, dict) and e.get("chunk_size", 0) > 0) > 0
            and all(isinstance(e, dict) and e.get("prefix_valid") is True for e in output_events),
            "returned output receipts absent or cumulative prefix invalid")
    require(all(isinstance(s, dict) for s in steps), "malformed raw scheduler step")
    if not all(isinstance(s, dict) for s in steps):
        return report
    report["counts"].update(requests=len(requests), natural_eos=stop_counts["stop"],
                            length_stops=stop_counts["length"], scheduler_steps=len(steps),
                            returned_output_events=len(output_events))

    by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for e in events:
        if (not isinstance(e, dict) or not isinstance(e.get("event"), str)
                or type(e.get("step")) is not int or not number(e.get("host_perf_counter_s"))):
            issues.append("malformed selective-store event")
            continue
        by_kind[e["event"]].append(e)
    ready = [e for e in by_kind["commit_check"] if e.get("reason") == "READY"]
    accepts = by_kind["accept"]
    report["counts"].update(accepted_intents=len(accepts), prepared=len(by_kind["prepare"]),
                            ready_commits=len(ready), applied_rotations=get(selective, "applied_rotations"),
                            native_preemption_events=len(by_kind["native_preemption"]),
                            actual_preemptions=get(raw, "actual_preemption_count"))
    require(get(selective, "applied_rotations") == len(ready),
            "READY commit count differs from applied custom rotations")
    require(get(raw, "actual_preemption_count") == sum(
                e.get("original_preemption_returned") is True
                for e in (get(raw, "preemption_events") or []) if isinstance(e, dict)),
            "native preemption receipt count differs")
    require(all(any(a.get("step") == p["step"] and a.get("action") == "PREPARE_SELECTED"
                    and a.get("target_id") == p.get("target") and a.get("victim_id") == p.get("victim")
                    for a in accepts) for p in by_kind["prepare"]),
            "staged prepare lacks accepted LTR intent")
    for e in ready:
        step = e["step"]
        victim = e.get("victim")
        if not any(p.get("step") == step - 1 and p.get("victim") == victim
                   and p.get("target") == e.get("target") for p in by_kind["prepare"]):
            issues.append(f"READY commit at step {step} lacks prior matching prepare")
        source_victim = get(request_by_internal.get(victim), "request_id")
        if not (0 <= step < len(steps)) or source_victim not in steps[step].get("preempted_request_ids", []):
            issues.append(f"READY commit at step {step} lacks native victim preemption")

    # Job identity must agree across adapter metadata, worker dispatch, and the
    # worker completion callback. A registration or dispatch is never completion.
    metadata = by_kind["metadata"]
    meta_job: dict[tuple[int, int, str, bool], dict[str, Any]] = {}
    for e in metadata:
        for j in e.get("jobs", []):
            if isinstance(j, dict):
                meta_job[(e["step"], j.get("job_id"), j.get("request"), j.get("is_store"))] = e
    dispatches: dict[tuple[int, str, bool], list[dict[str, Any]]] = defaultdict(list)
    completions: dict[tuple[int, str, bool], list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for row in offload.get("dispatch", []):
        if isinstance(row, dict):
            dispatches[(row.get("job_id"), row.get("request"), row.get("is_store"))].append(row)
    for row in offload.get("completed_jobs", []):
        if isinstance(row, dict):
            for j in row.get("jobs", []):
                if isinstance(j, dict):
                    completions[(j.get("job_id"), j.get("request"), j.get("is_store"))].append((row, j))

    def receipt(job: int, request: str, is_store: bool) -> tuple[float, float] | None:
        key = (job, request, is_store)
        candidates = [d for d in dispatches[key] if d.get("accepted") is True
                      and "error" not in d and number(d.get("after_perf_s"))]
        finished = [(c, j) for c, j in completions[key]
                    if number(c.get("time_s")) and type(j.get("count")) is int and j["count"] > 0]
        for d in candidates:
            for c, _ in finished:
                if c["time_s"] >= d["after_perf_s"]:
                    return d["after_perf_s"], c["time_s"]
        return None

    for key in dispatches:
        if receipt(*key) is None:
            issues.append(f"worker dispatch lacks accepted, ordered completion: job {key[0]}")

    store_jobs = set()
    load_jobs = set()
    for e in by_kind["store_delta"]:
        if e.get("status") == "NEW_STORES_VALID":
            for j in e.get("jobs", []):
                if not isinstance(j, dict) or type(j.get("job")) is not int:
                    issues.append(f"malformed selected store at step {e['step']}")
                    continue
                matches = [(key, meta) for key, meta in meta_job.items()
                           if key[0] == e["step"] and key[1] == j["job"] and key[3] is True
                           and j["job"] in meta.get("stores", [])]
                if len(matches) != 1:
                    issues.append(f"selected store {j['job']} lacks unique native metadata identity")
                    continue
                _, request, _ = matches[0][0][1:]
                store_jobs.add((j["job"], request))
                if receipt(j["job"], request, True) is None:
                    issues.append(f"selected store {j['job']} lacks accepted worker dispatch and completion")
    for e in metadata:
        for kind, is_store in (("stores", True), ("loads", False)):
            for job_id in e.get(kind, []):
                matches = [j for j in e.get("jobs", []) if isinstance(j, dict)
                           and j.get("job_id") == job_id and j.get("is_store") is is_store]
                if len(matches) != 1:
                    issues.append(f"native {kind} job {job_id} lacks unique metadata identity")
        for j in e.get("jobs", []):
            if not isinstance(j, dict) or j.get("is_store") is not False or j.get("job_id") not in e.get("loads", []):
                continue
            key = (j["job_id"], j.get("request"))
            load_jobs.add(key)
            if receipt(j["job_id"], j.get("request"), False) is None:
                issues.append(f"native load {j['job_id']} lacks accepted worker dispatch and completion")
    report["counts"].update(registered_selected_store_jobs=len(store_jobs),
                            registered_native_load_jobs=len(load_jobs),
                            completed_selected_store_jobs=sum(receipt(j, r, True) is not None for j, r in store_jobs),
                            completed_native_load_jobs=sum(receipt(j, r, False) is not None for j, r in load_jobs))

    origin = get(raw, "measurement_origin_perf_counter_s")
    require(number(origin), "raw measurement clock origin is missing")
    if not number(origin):
        return report
    for prepare in by_kind["prepare"]:
        step, target, victim = prepare["step"], prepare.get("target"), prepare.get("victim")
        accept = next((e for e in accepts if e.get("step") == step
                       and e.get("target_id") == target and e.get("victim_id") == victim
                       and e.get("action") == "PREPARE_SELECTED"), None)
        commit = next((e for e in ready if e.get("step") == step + 1
                       and e.get("target") == target and e.get("victim") == victim), None)
        delta = next((e for e in by_kind["store_delta"] if e.get("step") == step
                      and e.get("status") == "NEW_STORES_VALID"), None)
        if not (accept and commit and delta and target in request_by_internal):
            continue
        selected_store_receipts = []
        for j in delta.get("jobs", []):
            if isinstance(j, dict) and (j.get("job"), victim) in store_jobs:
                value = receipt(j["job"], victim, True)
                if value is not None:
                    selected_store_receipts.append((j["job"], value))
        if not selected_store_receipts:
            continue
        flush = next((e for e in metadata if e.get("step") == commit["step"]), None)
        if not flush or not all(j in flush.get("flush", []) or done <= commit.get("host_perf_counter_s", float("-inf"))
                                for j, (_, done) in selected_store_receipts):
            continue
        next_release = next((e for e in by_kind["release"] if e.get("target") == target
                             and e.get("step", -1) >= step), None)
        end_step = next_release["step"] if next_release else len(steps)
        target_loads = []
        for e in metadata:
            if not (commit["step"] <= e["step"] < end_step):
                continue
            for j in e.get("jobs", []):
                if isinstance(j, dict) and j.get("is_store") is False and j.get("request") == target \
                        and j.get("job_id") in e.get("loads", []):
                    value = receipt(j["job_id"], target, False)
                    if value is not None:
                        target_loads.append((j["job_id"], value, e["step"]))
        if not target_loads or not any(row.get("request") == target and
                                       ((type(row.get("matched")) is int and row["matched"] > 0)
                                        or row.get("asynchronous") is True)
                                       for row in offload.get("lookup", [])):
            continue
        first = min((e for e in output_events if e.get("request_id") == request_by_internal[target]["request_id"]
                     and type(e.get("chunk_size")) is int and e["chunk_size"] > 0
                     and number(e.get("received_s"))
                     and origin + e["received_s"] > commit["host_perf_counter_s"]),
                    key=lambda e: e["received_s"], default=None)
        marker = next((e for e in by_kind["target_new_output"] if e.get("target", e.get("request")) == target
                       and commit["step"] <= e["step"] < end_step), None)
        if not first or not marker \
                or marker["host_perf_counter_s"] < origin + first["received_s"] \
                or type(marker.get("output_tokens")) is not int \
                or marker["output_tokens"] != first.get("cumulative_tokens"):
            continue
        initial_loads = {job: value for job, value, load_step in target_loads
                         if load_step <= marker["step"]}
        if len(initial_loads) != 1:
            continue
        load_started, load_done = next(iter(initial_loads.values()))
        if not (commit["host_perf_counter_s"] <= load_started <= load_done
                <= origin + first["received_s"] <= marker["host_perf_counter_s"]
                and all(done <= load_done for _, (_, done) in selected_store_receipts)):
            continue
        before_output = [e for e in by_kind["allocation"] if commit["step"] <= e["step"] <= marker["step"]
                         and type(get(e, "scheduled", target)) is int and get(e, "scheduled", target) > 0
                         and e.get("host_perf_counter_s", float("inf")) <= origin + first["received_s"]]
        if not before_output or not all(0 <= e["step"] < len(steps) and any(
                x.get("internal_request_id") == target and x.get("scheduled_tokens", 0) > 0
                for x in steps[e["step"]].get("scheduled", [])) for e in before_output):
            continue
        # Verify every observed boosted scheduling turn in this active epoch.
        q = accept.get("quantum_remaining")
        quantum_ok = type(q) is int and q > 0
        post_output_positive = []
        for e in sorted((e for e in by_kind["allocation"] if step <= e["step"] < end_step
                         and e.get("active_target") == target), key=lambda e: e["step"]):
            amount = get(e, "scheduled", target)
            amount = amount if type(amount) is int else 0
            q -= int(amount > 0) if type(q) is int else 0
            if get(e, "boosted", target, "quantum_remaining") != q:
                quantum_ok = False
            if e["step"] >= marker["step"] and e["host_perf_counter_s"] > marker["host_perf_counter_s"] and amount > 0:
                post_output_positive.append(e["step"])
        if not quantum_ok or not post_output_positive or not next_release \
                or next_release.get("reason") not in ("quantum_expired", "terminal"):
            continue
        if next_release["reason"] == "quantum_expired" and q != 0:
            continue
        external_id = f"measured/{request_by_internal[target]['request_id']}"
        if not all(0 <= s < len(steps) and any(
                x.get("internal_request_id") == target and x.get("scheduled_tokens", 0) > 0
                for x in steps[s].get("scheduled", []))
                and external_id in engine_steps[s].get("output_request_ids", [])
                for s in post_output_positive):
            continue
        report["observed_chains"].append(dict(
            target=target, victim=victim, prepare_step=step, commit_step=commit["step"],
            selected_store_jobs=[j for j, _ in selected_store_receipts],
            selected_store_completed_perf_s=[done for _, (_, done) in selected_store_receipts],
            target_load_jobs=list(initial_loads),
            target_load_completed_perf_s=[done for _, done in initial_loads.values()],
            later_target_load_jobs=[j for j, _, load_step in target_loads
                                    if load_step > marker["step"]],
            first_returned_output_s=first["received_s"], first_output_marker_step=marker["step"],
            post_output_positive_steps=post_output_positive,
            release_reason=next_release["reason"], release_step=next_release["step"],
        ))

    report["counts"]["observed_chains"] = len(report["observed_chains"])
    report["counts"]["terminal_releases"] = sum(e.get("reason") == "terminal" for e in by_kind["release"])
    report["counts"]["quantum_expirations"] = sum(e.get("reason") == "quantum_expired" for e in by_kind["release"])
    recovered_eos = [e for e in by_kind["release"] if e.get("reason") == "terminal"
                     and get(request_by_internal.get(e.get("target")), "stop_reason") == "stop"]
    report["counts"]["recovered_target_natural_eos"] = len(recovered_eos)
    wait_load = [e for e in by_kind["intent"] if e.get("action") == "WAIT_LOAD"]
    wait_load_valid = bool(wait_load)
    wait_load_zero_scheduled = 0
    wait_load_positive_scheduled = 0
    for e in wait_load:
        allocation = next((a for a in by_kind["allocation"] if a.get("step") == e["step"]), None)
        target = e.get("target_id")
        amount = get(allocation, "scheduled", target)
        amount = 0 if amount is None else amount
        if type(amount) is int and amount == 0:
            wait_load_zero_scheduled += 1
        elif type(amount) is int and amount > 0:
            wait_load_positive_scheduled += 1
        expected_q = e.get("quantum_remaining")
        if type(expected_q) is int and type(amount) is int:
            expected_q -= int(amount > 0)
        if (allocation is None or type(amount) is not int or amount < 0
                or type(e.get("quantum_remaining")) is not int
                or get(allocation, "boosted", target, "quantum_remaining") != expected_q
                or amount > 0 and not (0 <= e["step"] < len(steps) and any(
                    x.get("internal_request_id") == target and x.get("scheduled_tokens", 0) > 0
                    for x in steps[e["step"]].get("scheduled", [])))):
            wait_load_valid = False
    report["counts"].update(wait_load_intents=len(wait_load),
                            wait_load_zero_scheduled=wait_load_zero_scheduled,
                            wait_load_positive_scheduled=wait_load_positive_scheduled)
    report["coverage"].update(
        selected_store_load_output_post_quantum="OBSERVED" if report["observed_chains"] else "UNOBSERVED",
        pending_load_zero_quantum="OBSERVED" if wait_load_valid and wait_load_zero_scheduled else
            "UNOBSERVED" if not wait_load or wait_load_valid else "CONTRADICTED",
        terminal_release="OBSERVED" if report["counts"]["terminal_releases"] else "UNOBSERVED",
        recovery_phase_natural_eos="OBSERVED" if recovered_eos else "UNOBSERVED",
    )
    if wait_load and not wait_load_valid:
        issues.append("WAIT_LOAD quantum change differs from actual native scheduled tokens")
    if not wait_load:
        unknown.append("No WAIT_LOAD scheduling call was observed; pending-load zero-decrement behavior is untested.")
    elif wait_load_valid and not wait_load_zero_scheduled:
        unknown.append("WAIT_LOAD was observed only with positive native scheduling; zero-service load waiting is untested.")
    if not recovered_eos:
        unknown.append("Recovery-phase natural EOS was not observed; other requests' EOS does not prove it.")
    unknown.append("Transfer statistics are aggregate; worker job IDs prove dispatch/completion, not per-job bytes.")
    if not issues:
        if report["observed_chains"]:
            report["verdict"] = "OBSERVED_CHAIN"
        elif not accepts and not by_kind["prepare"] and not ready and get(selective, "applied_rotations") == 0:
            report["verdict"] = "NO_ACTION"
        else:
            issues.append("accepted action exists, but no joined store/load/output/post-output-quantum chain is proven")
    return report


def audit(directory: Path) -> dict[str, Any]:
    """Always return a JSON-serializable fail-closed result for corrupt input."""
    try:
        return _audit(directory)
    except Exception as exc:
        return dict(schema_version=1, result_dir=str(directory.resolve()),
                    verdict="INCOMPLETE", evidence_ceiling="NATIVE_SERVING_DIAGNOSTIC",
                    files={}, checks={}, counts={}, coverage={}, observed_chains=[],
                    issues=[f"audit could not establish evidence: {type(exc).__name__}: {exc}"],
                    unknown=["Malformed or incomplete result data require manual inspection."])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path, help="one diagnostic-ltr-t30-q10 output directory")
    parser.add_argument("--output", type=Path, help="optional audit JSON path; default: stdout")
    args = parser.parse_args()
    report = audit(args.results_dir)
    payload = json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        if args.output.resolve().is_relative_to(args.results_dir.resolve()):
            parser.error("audit output must be outside the immutable result directory")
        with args.output.open("x") as stream:
            stream.write(payload)


if __name__ == "__main__":
    main()
