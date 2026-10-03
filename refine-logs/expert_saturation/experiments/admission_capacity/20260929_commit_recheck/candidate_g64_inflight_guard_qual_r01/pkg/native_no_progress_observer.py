"""Bounded, read-only post-LTR native no-progress observer.

Install *after* the LTR adapter, and remove *before* uninstalling that adapter.
This is diagnostic instrumentation, never a performance measurement arm. It
forwards every native method call unchanged until a bounded empty-schedule
condition captures a snapshot and raises NoProgressSnapshot.
"""

from __future__ import annotations

import time
from typing import Any


EMPTY_SCHEDULE_CALLS = 32
EMPTY_SCHEDULE_SECONDS = 1.0
MAX_EVENTS = 64
MAX_REQUEST_ROWS = 32
MAX_JOB_ROWS = 16


class NoProgressSnapshot(RuntimeError):
    """The diagnostic captured a stable native no-progress state."""

    def __init__(self, snapshot: dict[str, Any]):
        self.snapshot = snapshot
        super().__init__(
            "diagnostic_no_progress: repeated empty native schedules with "
            "the same target and no pending connector work"
        )


def install(scheduler: Any) -> tuple[dict[str, Any], Any]:
    """Wrap only existing object methods; return mutable data and an uninstaller.

    The observed scheduler must already have the LTR instance hooks. A failed
    prerequisite leaves every object untouched. The wrappers never call the
    underlying method twice and preserve the exact args, kwargs and return.
    """
    manager = scheduler.kv_cache_manager
    coordinator = manager.coordinator
    connector = scheduler.connector
    if type(connector).__name__ != "OffloadingConnector":
        raise ValueError("native OffloadingConnector required")
    connector_scheduler = connector.connector_scheduler
    if connector_scheduler is None:
        raise ValueError("native connector scheduler required")
    if not all(name in vars(scheduler) for name in
               ("schedule", "_rotation_begin", "_rotation_hold", "_rotation_target")):
        raise ValueError("install after the LTR adapter and before capture")

    targets = (
        (manager, "allocate_slots"),
        (coordinator, "get_num_blocks_to_allocate"),
        (connector, "get_num_new_matched_tokens"),
        (scheduler, "_rotation_begin"),
        (scheduler, "_rotation_hold"),
        (scheduler, "schedule"),
    )
    originals: dict[str, tuple[Any, bool, Any]] = {}
    for obj, name in targets:
        original = getattr(obj, name, None)
        if not callable(original):
            raise ValueError(f"missing native method: {name}")
        originals[name] = (obj, name in vars(obj), original)

    data: dict[str, Any] = {
        "status": "OBSERVING",
        "scope": "DIAGNOSTIC_ONLY_NOT_PERFORMANCE",
        "trigger": {
            "minimum_consecutive_empty_schedules": EMPTY_SCHEDULE_CALLS,
            "minimum_empty_duration_seconds": EMPTY_SCHEDULE_SECONDS,
            "same_candidate_required": True,
            "zero_scheduler_jobs_and_transfers_required": True,
        },
        "schedule_calls": 0,
        "allocation_calls": 0,
        "allocation_refusals": 0,
        "block_requirement_calls": 0,
        "outside_block_requirement_calls": 0,
        "lookup_calls": 0,
        "consecutive_empty_schedules": 0,
        "recent_allocations": [],
        "recent_block_requirements": [],
        "recent_lookups": [],
        "recent_begins": [],
        "recent_holds": [],
        "recent_schedules": [],
        "observer_errors": [],
        "snapshot": None,
    }
    current_holds: dict[str, bool] = {}
    active_allocation_calls: list[int] = []
    empty_candidate: tuple[str, str] | None = None
    empty_started_s: float | None = None
    installed = False

    def ring(name: str, row: dict[str, Any]) -> None:
        rows = data[name]
        rows.append(row)
        if len(rows) > MAX_EVENTS:
            del rows[:len(rows) - MAX_EVENTS]

    def problem(label: str, error: Exception) -> None:
        ring("observer_errors", {"where": label,
                                 "error": f"{type(error).__name__}: {error}"})

    def read(label: str, fn: Any) -> Any:
        try:
            return fn()
        except Exception as error:
            problem(label, error)
            return None

    def request_id(request: Any) -> str | None:
        value = getattr(request, "request_id", None)
        return None if value is None else str(value)

    def request_row(request: Any) -> dict[str, Any]:
        result: dict[str, Any] = {"request_id": request_id(request)}
        status = getattr(request, "status", None)
        result["status"] = getattr(status, "name", None)
        for name in ("num_computed_tokens", "num_prompt_tokens", "num_output_tokens",
                     "num_tokens", "max_tokens", "num_in_flight_tokens"):
            value = read(f"request.{name}", lambda name=name: getattr(request, name, None))
            result[name] = value if type(value) in (int, float, bool, str) or value is None else None
        return result

    def queue_rows(queue: Any) -> dict[str, Any]:
        rows = read("queue_iteration", lambda: list(queue))
        if rows is None:
            return {"count": None, "requests": []}
        return {"count": len(rows),
                "requests": [request_row(item) for item in rows[:MAX_REQUEST_ROWS]]}

    def safe_scalar(value: Any) -> Any:
        if value is None or type(value) in (bool, int, float, str):
            return value
        return {"object_type": type(value).__name__}

    def pool_free() -> int | None:
        return read("pool_free", lambda: int(manager.block_pool.get_num_free_blocks()))

    def inflight_reserved() -> int | None:
        method = getattr(scheduler, "_inflight_prefill_reserved_blocks", None)
        return read("inflight_reserved", lambda: int(method())) if callable(method) else None

    def pending_state(target: str | None) -> dict[str, Any]:
        jobs = read("connector_jobs", lambda: dict(connector_scheduler._jobs))
        req_status = read("connector_request_status", lambda: dict(connector_scheduler._req_status))
        if jobs is None or req_status is None:
            return {"known": False, "scheduler_job_count": None,
                    "transfer_job_count": None, "target_transfer_jobs": None,
                    "pending_push_work": None, "clear": False, "jobs": []}
        transfers: dict[str, list[Any]] = {}
        for rid, row in req_status.items():
            job_ids = read("request_transfer_jobs", lambda row=row: list(row.transfer_jobs))
            if job_ids is None:
                return {"known": False, "scheduler_job_count": len(jobs),
                        "transfer_job_count": None, "target_transfer_jobs": None,
                        "pending_push_work": None, "clear": False, "jobs": []}
            if job_ids:
                transfers[str(rid)] = sorted(str(job) for job in job_ids)
        pending_push = read("pending_push_work", connector.has_pending_push_work)
        job_rows = []
        for job_id, job in list(jobs.items())[:MAX_JOB_ROWS]:
            job_rows.append({"job_id": str(job_id), "request_id": str(getattr(job, "req_id", "")),
                             "is_store": bool(getattr(job, "is_store", False)),
                             "pending_count": safe_scalar(getattr(job, "pending_count", None))})
        transfer_count = sum(len(items) for items in transfers.values())
        target_jobs = transfers.get(target, []) if target is not None else []
        return {"known": type(pending_push) is bool,
                "scheduler_job_count": len(jobs),
                "transfer_job_count": transfer_count,
                "target_transfer_jobs": target_jobs,
                "pending_push_work": pending_push,
                "clear": type(pending_push) is bool and not pending_push and
                         not jobs and transfer_count == 0,
                "jobs": job_rows}

    def candidate() -> tuple[str, str] | None:
        active = getattr(scheduler, "_rotation_target", None)
        if active is not None:
            return "rotation_target", str(active)
        for label in ("waiting", "skipped_waiting", "running"):
            queue = getattr(scheduler, label, None)
            if queue is None:
                continue
            first = read(f"{label}_head", lambda queue=queue: next(iter(queue), None))
            rid = request_id(first) if first is not None else None
            if rid is not None:
                return label + "_head", rid
        return None

    def state(candidate_id: str | None) -> dict[str, Any]:
        running = queue_rows(getattr(scheduler, "running", ()))
        for row in running["requests"]:
            row["held_this_schedule"] = current_holds.get(row["request_id"])
            req = read("running_request_lookup", lambda rid=row["request_id"]:
                       scheduler.requests.get(rid))
            remaining = getattr(scheduler, "_request_remaining_blocks", None)
            row["remaining_blocks"] = (
                read("remaining_blocks", lambda req=req: int(remaining(req)))
                if req is not None and callable(remaining) else None
            )
        inflight = read("inflight_prefills", lambda: list(getattr(scheduler, "_inflight_prefills", ())))
        return {"free_blocks": pool_free(),
                "inflight_prefill_reserved_blocks": inflight_reserved(),
                "inflight_prefill_request_ids":
                    [request_id(req) for req in inflight[:MAX_REQUEST_ROWS]] if inflight is not None else None,
                "running": running,
                "waiting": queue_rows(getattr(scheduler, "waiting", ())),
                "skipped_waiting": queue_rows(getattr(scheduler, "skipped_waiting", ())),
                "candidate_request": request_row(scheduler.requests[candidate_id])
                    if candidate_id is not None and candidate_id in scheduler.requests else None,
                "pending": pending_state(candidate_id)}

    def observed_allocate(*args: Any, **kwargs: Any) -> Any:
        data["allocation_calls"] += 1
        rid = request_id(args[0]) if args else request_id(kwargs.get("request"))
        row = {"call": data["allocation_calls"], "schedule_call": data["schedule_calls"] + 1,
               "request_id": rid, "num_new_tokens": safe_scalar(args[1] if len(args) > 1
                                                         else kwargs.get("num_new_tokens")),
               "extra_positional": [safe_scalar(item) for item in args[2:]],
               "actual_kwargs": {key: safe_scalar(value) for key, value in kwargs.items()},
               "free_blocks_before": pool_free(),
               "inflight_prefill_reserved_blocks_before": inflight_reserved()}
        active_allocation_calls.append(data["allocation_calls"])
        try:
            try:
                result = originals["allocate_slots"][2](*args, **kwargs)
            except BaseException as error:
                row["native_error"] = f"{type(error).__name__}: {error}"
                row["free_blocks_after"] = pool_free()
                ring("recent_allocations", row)
                raise
        finally:
            active_allocation_calls.pop()
        row["returned_none"] = result is None
        row["free_blocks_after"] = pool_free()
        if result is None:
            data["allocation_refusals"] += 1
            row["refusal_stage"] = "UNKNOWN_NATIVE_RETURN_NONE"
        ring("recent_allocations", row)
        return result

    def observed_block_requirement(*args: Any, **kwargs: Any) -> Any:
        if not active_allocation_calls:
            # _request_remaining_blocks and other read-only snapshots can use
            # this same coordinator method. Forward them, but do not let those
            # auxiliary calls evict the native allocate_slots evidence ring.
            data["outside_block_requirement_calls"] += 1
            return originals["get_num_blocks_to_allocate"][2](*args, **kwargs)
        data["block_requirement_calls"] += 1
        row = {"call": data["block_requirement_calls"],
               "allocation_call": active_allocation_calls[-1],
               "schedule_call": data["schedule_calls"] + 1,
               "actual_args": [safe_scalar(item) for item in args],
               "actual_kwargs": {key: safe_scalar(value) for key, value in kwargs.items()},
               "free_blocks_at_call": pool_free(),
               "native_stage_hint": "full_sequence_precheck" if kwargs.get("apply_admission_cap") is True
                                    else "incremental_or_other"}
        try:
            result = originals["get_num_blocks_to_allocate"][2](*args, **kwargs)
        except BaseException as error:
            row["native_error"] = f"{type(error).__name__}: {error}"
            ring("recent_block_requirements", row)
            raise
        row["required_blocks_returned"] = safe_scalar(result)
        ring("recent_block_requirements", row)
        return result

    def observed_lookup(*args: Any, **kwargs: Any) -> Any:
        data["lookup_calls"] += 1
        rid = request_id(args[0]) if args else request_id(kwargs.get("request"))
        row = {"call": data["lookup_calls"], "schedule_call": data["schedule_calls"] + 1,
               "request_id": rid,
               "num_computed_tokens": safe_scalar(args[1] if len(args) > 1
                                                  else kwargs.get("num_computed_tokens"))}
        try:
            result = originals["get_num_new_matched_tokens"][2](*args, **kwargs)
        except BaseException as error:
            row["native_error"] = f"{type(error).__name__}: {error}"
            ring("recent_lookups", row)
            raise
        row["matched_tokens"] = safe_scalar(result[0])
        row["load_kv_async"] = safe_scalar(result[1])
        ring("recent_lookups", row)
        return result

    def observed_begin(*args: Any, **kwargs: Any) -> Any:
        row = {"schedule_call": data["schedule_calls"] + 1,
               "running_before": [request_id(req) for req in list(scheduler.running)[:MAX_REQUEST_ROWS]],
               "target_before": safe_scalar(getattr(scheduler, "_rotation_target", None)),
               "free_blocks_before": pool_free()}
        try:
            result = originals["_rotation_begin"][2](*args, **kwargs)
        except BaseException as error:
            row["native_error"] = f"{type(error).__name__}: {error}"
            ring("recent_begins", row)
            raise
        row["target_after"] = safe_scalar(getattr(scheduler, "_rotation_target", None))
        row["running_after"] = [request_id(req) for req in list(scheduler.running)[:MAX_REQUEST_ROWS]]
        row["free_blocks_after"] = pool_free()
        ring("recent_begins", row)
        return result

    def observed_hold(*args: Any, **kwargs: Any) -> Any:
        rid = request_id(args[0]) if args else request_id(kwargs.get("req"))
        row = {"schedule_call": data["schedule_calls"] + 1, "request_id": rid,
               "target": safe_scalar(getattr(scheduler, "_rotation_target", None)),
               "free_blocks": pool_free()}
        try:
            result = originals["_rotation_hold"][2](*args, **kwargs)
        except BaseException as error:
            row["native_error"] = f"{type(error).__name__}: {error}"
            ring("recent_holds", row)
            raise
        row["held"] = bool(result)
        if rid is not None:
            current_holds[rid] = bool(result)
        ring("recent_holds", row)
        return result

    def observed_schedule(*args: Any, **kwargs: Any) -> Any:
        nonlocal empty_candidate, empty_started_s
        current_holds.clear()
        try:
            result = originals["schedule"][2](*args, **kwargs)
        except BaseException:
            raise
        data["schedule_calls"] += 1
        scheduled = getattr(result, "num_scheduled_tokens", None)
        if not isinstance(scheduled, dict):
            # The observer does not reinterpret a scheduler result it cannot read.
            problem("scheduled_map", TypeError("num_scheduled_tokens is not a dict"))
            return result
        pair = candidate()
        rid = pair[1] if pair is not None else None
        pending = pending_state(rid)
        now = time.perf_counter()
        row = {"call": data["schedule_calls"], "monotonic_s": now,
               "scheduled": {str(key): int(value) for key, value in list(scheduled.items())[:MAX_REQUEST_ROWS]},
               "candidate_source": pair[0] if pair else None, "candidate_request_id": rid,
               "running_count": len(scheduler.running), "waiting_count": len(scheduler.waiting),
               "free_blocks": pool_free(), "inflight_prefill_reserved_blocks": inflight_reserved(),
               "pending": pending,
               "held_request_ids": sorted(key for key, held in current_holds.items() if held)}
        ring("recent_schedules", row)
        eligible = (not scheduled and pair is not None and pending["clear"] and
                    bool(scheduler.requests) and rid in scheduler.requests)
        if eligible:
            if pair == empty_candidate:
                data["consecutive_empty_schedules"] += 1
            else:
                empty_candidate = pair
                empty_started_s = now
                data["consecutive_empty_schedules"] = 1
        else:
            empty_candidate = None
            empty_started_s = None
            data["consecutive_empty_schedules"] = 0
        elapsed = now - empty_started_s if empty_started_s is not None else 0.0
        if eligible and data["consecutive_empty_schedules"] >= EMPTY_SCHEDULE_CALLS and elapsed >= EMPTY_SCHEDULE_SECONDS:
            snapshot = {"reason": "GLOBAL_EMPTY_SCHEDULE_NO_PENDING_NATIVE_WORK",
                        "schedule_call": data["schedule_calls"],
                        "consecutive_empty_schedules": data["consecutive_empty_schedules"],
                        "empty_duration_seconds": elapsed,
                        "candidate_source": pair[0], "candidate_request_id": rid,
                        "state": state(rid),
                        "recent_allocations": list(data["recent_allocations"]),
                        "recent_block_requirements": list(data["recent_block_requirements"]),
                        "recent_lookups": list(data["recent_lookups"]),
                        "recent_begins": list(data["recent_begins"]),
                        "recent_holds": list(data["recent_holds"]),
                        "recent_schedules": list(data["recent_schedules"])}
            data["snapshot"] = snapshot
            data["status"] = "SNAPSHOT_CAPTURED"
            raise NoProgressSnapshot(snapshot)
        return result

    wrappers = {
        "allocate_slots": observed_allocate,
        "get_num_blocks_to_allocate": observed_block_requirement,
        "get_num_new_matched_tokens": observed_lookup,
        "_rotation_begin": observed_begin,
        "_rotation_hold": observed_hold,
        "schedule": observed_schedule,
    }
    try:
        for name, (obj, _, _) in originals.items():
            setattr(obj, name, wrappers[name])
        installed = True
    except BaseException:
        for name, (obj, had_instance, original) in originals.items():
            if name in vars(obj) and getattr(obj, name) is wrappers[name]:
                if had_instance:
                    setattr(obj, name, original)
                else:
                    delattr(obj, name)
        raise

    def uninstall() -> dict[str, Any]:
        nonlocal installed
        if installed:
            for name, (obj, had_instance, original) in originals.items():
                if getattr(obj, name) is not wrappers[name]:
                    raise RuntimeError(f"observer method changed before uninstall: {name}")
                if had_instance:
                    setattr(obj, name, original)
                else:
                    delattr(obj, name)
            installed = False
        if data["status"] == "OBSERVING":
            data["status"] = "UNINSTALLED_WITHOUT_SNAPSHOT"
        return data

    return data, uninstall
