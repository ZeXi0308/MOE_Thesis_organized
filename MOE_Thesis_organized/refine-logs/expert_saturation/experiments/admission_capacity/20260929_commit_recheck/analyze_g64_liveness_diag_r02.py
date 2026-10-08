#!/usr/bin/env python3
"""Read-only source localization for the one G64 T30/Q1 liveness diagnostic.

This analyzes only the new r02 diagnostic archive. It does not rank performance
or alter the failed r02 performance raw/session. Native source hashes fix the
allocator conditions used below; absent observations yield UNKNOWN.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any


SESSION_NAME = "moe-a-g64-liveness-diag-session-r02-20260930"
CELL_NAME = "cell-00-ltr_t30_q1"
PLAN_SHA256 = "730245220658765f553ef282c9394524a0c66b75446e46be8d75275cd5465e14"
PACKAGE_MANIFEST_SHA256 = "2fa25606a2566bc8c18e1e11995478ee629c39b8ee7f00fa87f8561c17933d01"
OBSERVER_SHA256 = "01e8e4497b60e31d339e99be9115d1d724a4af392a3669ddcb8a03372a0fcacb"
SCHEDULER_SHA256 = "2ed2a550b6558b2495eda845a97ae38bcf0225027b9e25fbf00fc3880c1d3941"
KV_MANAGER_SHA256 = "3f4af8d247f3fe9570b0132818b832b66ae6a2ac12942588828f899f6ff77ccf"
SHA_RE = re.compile(r"[0-9a-f]{64}\Z")


class AuditError(ValueError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise AuditError(message)


def read_json(path: Path) -> Any:
    require(path.is_file() and not path.is_symlink(), f"missing or linked JSON: {path}")
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_archive(cell: Path) -> tuple[Path, dict[str, str], str]:
    archive = cell / "archive"
    require(archive.is_dir() and not archive.is_symlink(), "archived output missing")
    manifest_path = cell / "output_sha256.json"
    manifest = read_json(manifest_path)
    require(isinstance(manifest, dict) and manifest, "output_sha256.json is empty or invalid")
    actual: set[str] = set()
    for path in archive.rglob("*"):
        require(not path.is_symlink(), f"archive symlink: {path}")
        if path.is_file():
            actual.add(path.relative_to(archive).as_posix())
        else:
            require(path.is_dir(), f"archive special file: {path}")
    require(set(manifest) == actual, "archive file set differs from output_sha256.json")
    for name, expected in manifest.items():
        pure = PurePosixPath(name)
        require(not pure.is_absolute() and ".." not in pure.parts and str(pure) == name,
                f"unsafe archive path: {name}")
        require(isinstance(expected, str) and SHA_RE.fullmatch(expected) is not None,
                f"invalid archive digest: {name}")
        require(sha256(archive / name) == expected, f"archive SHA mismatch: {name}")
    return archive, manifest, sha256(manifest_path)


def integer(value: Any) -> int | None:
    return value if type(value) is int else None


def infer_native_refusal(allocation: dict[str, Any],
                         requirements: list[dict[str, Any]],
                         watermark_blocks: int) -> dict[str, Any]:
    """Classify only branches entailed by pinned KVCacheManager.allocate_slots.

    With scheduler_reserve_full_isl, native first compares full required+
    watermark to free. If that succeeds it compares incremental required+
    watermark to free-reserved. A coordinator row is an *actual* nested native
    call, not a synthetic repeat of the allocation calculation.
    """
    result: dict[str, Any] = {"branch": "UNKNOWN", "reason": "insufficient observed native inputs",
                              "allocation": allocation, "requirements": requirements}
    if allocation.get("returned_none") is not True:
        result["reason"] = "allocation did not return None"
        return result
    kwargs = allocation.get("actual_kwargs") or {}
    full_enabled = kwargs.get("full_sequence_must_fit") is True
    reserved = integer(kwargs.get("reserved_blocks"))
    if reserved is None or reserved < 0 or watermark_blocks < 0:
        result["reason"] = "reserved or watermark blocks unavailable"
        return result
    full_rows = [r for r in requirements if r.get("native_stage_hint") == "full_sequence_precheck"]
    incremental_rows = [r for r in requirements if r.get("native_stage_hint") == "incremental_or_other"]
    if full_enabled and len(full_rows) != 1:
        result["reason"] = "missing or ambiguous full-sequence precheck"
        return result
    if full_rows:
        full = full_rows[0]
        required = integer(full.get("required_blocks_returned"))
        free = integer(full.get("free_blocks_at_call"))
        if required is None or free is None:
            result["reason"] = "full-precheck required/free values unavailable"
            return result
        result["full_sequence_precheck"] = {"required_plus_watermark": required + watermark_blocks,
                                            "free_blocks": free,
                                            "passed": required + watermark_blocks <= free}
        if required + watermark_blocks > free:
            if incremental_rows:
                result["reason"] = "incremental calculation followed a failed full precheck"
                return result
            result.update(branch="FULL_SEQUENCE_FIT_REFUSAL",
                          reason="native full-sequence requirement exceeds observed free blocks")
            return result
    if len(incremental_rows) != 1:
        result["reason"] = "missing or ambiguous incremental requirement"
        return result
    incremental = incremental_rows[0]
    required = integer(incremental.get("required_blocks_returned"))
    free = integer(incremental.get("free_blocks_at_call"))
    if required is None or free is None:
        result["reason"] = "incremental required/free values unavailable"
        return result
    budget = free - reserved
    result["incremental_check"] = {"required_plus_watermark": required + watermark_blocks,
                                   "free_blocks": free, "reserved_blocks": reserved,
                                   "free_minus_reserved": budget,
                                   "raw_free_would_fit": required + watermark_blocks <= free,
                                   "reserved_is_decisive":
                                       required + watermark_blocks <= free and
                                       required + watermark_blocks > budget}
    if required + watermark_blocks > budget:
        result.update(branch="INCREMENTAL_FREE_MINUS_RESERVED_REFUSAL",
                      reason="native incremental requirement exceeds free minus reserved blocks")
    else:
        result["reason"] = "observed required blocks fit both native gates despite None result"
    return result


def analyze_snapshot(observer: dict[str, Any], watermark: int) -> dict[str, Any]:
    snapshot = observer.get("snapshot")
    if observer.get("status") != "SNAPSHOT_CAPTURED" or not isinstance(snapshot, dict):
        return {"status": "NO_SNAPSHOT", "branch": "UNKNOWN", "circular_wait_support": "UNASSESSABLE"}
    target = snapshot.get("candidate_request_id")
    schedules = snapshot.get("recent_schedules") or []
    trigger = observer.get("trigger") or {}
    minimum_calls = integer(trigger.get("minimum_consecutive_empty_schedules"))
    minimum_seconds = trigger.get("minimum_empty_duration_seconds")
    require(isinstance(target, str) and target, "snapshot target absent")
    require(minimum_calls == 32 and minimum_seconds == 1.0, "observer trigger contract drift")
    require(integer(snapshot.get("consecutive_empty_schedules")) is not None and
            snapshot["consecutive_empty_schedules"] >= minimum_calls and
            type(snapshot.get("empty_duration_seconds")) in (int, float) and
            snapshot["empty_duration_seconds"] >= minimum_seconds,
            "snapshot does not meet fixed no-progress bound")
    require(len(schedules) >= minimum_calls, "bounded schedule window missing")
    window = schedules[-minimum_calls:]
    require(all(row.get("scheduled") == {} and row.get("candidate_request_id") == target and
                (row.get("pending") or {}).get("clear") is True for row in window),
            "last bounded schedules do not show the same empty candidate without pending work")
    state = snapshot.get("state") or {}
    pending = state.get("pending") or {}
    require(pending.get("clear") is True and pending.get("scheduler_job_count") == 0 and
            pending.get("transfer_job_count") == 0 and pending.get("pending_push_work") is False,
            "snapshot has native pending work")
    allocations = snapshot.get("recent_allocations") or []
    requirements = snapshot.get("recent_block_requirements") or []
    refusals = [row for row in allocations if row.get("request_id") == target and
                row.get("returned_none") is True]
    selected = refusals[-1] if refusals else None
    linked = [row for row in requirements if selected is not None and
              row.get("allocation_call") == selected.get("call")]
    branch = (infer_native_refusal(selected, linked, watermark)
              if selected is not None else
              {"branch": "UNKNOWN", "reason": "no target allocate_slots(None) in bounded ring"})
    matching_lookup = [row for row in snapshot.get("recent_lookups") or []
                       if row.get("request_id") == target and
                       selected is not None and row.get("schedule_call") == selected.get("schedule_call")]
    lookup = matching_lookup[-1] if matching_lookup else None
    held = [row["request_id"] for row in (state.get("running") or {}).get("requests", [])
            if row.get("held_this_schedule") is True and row.get("request_id") != target]
    inflight_ids = set(state.get("inflight_prefill_request_ids") or [])
    inflight_rows = [row for row in (state.get("running") or {}).get("requests", [])
                     if row.get("request_id") in inflight_ids]
    args = (selected or {}).get("actual_kwargs") or {}
    async_target = (lookup is not None and lookup.get("load_kv_async") is True and
                    args.get("delay_cache_blocks") is True)
    observer_errors = observer.get("observer_errors")
    evidence_complete = (bool(held) and async_target and branch["branch"] != "UNKNOWN" and
                         observer_errors == [])
    if evidence_complete and branch["branch"] == "INCREMENTAL_FREE_MINUS_RESERVED_REFUSAL" and \
            branch.get("incremental_check", {}).get("reserved_is_decisive") is True:
        support = "SOURCE_LOCAL_CIRCULAR_WAIT_SUPPORTED_RESERVED_GATE_DECISIVE"
    elif evidence_complete:
        support = "SOURCE_LOCAL_CIRCULAR_WAIT_CONSISTENT_OTHER_NATIVE_CAPACITY_GATE"
    else:
        support = "INSUFFICIENT_FOR_CIRCULAR_WAIT_LINK"
    return {
        "status": "SNAPSHOT_VALIDATED",
        "candidate_request_id": target,
        "candidate_source": snapshot.get("candidate_source"),
        "empty_schedule_calls": snapshot["consecutive_empty_schedules"],
        "empty_duration_seconds": snapshot["empty_duration_seconds"],
        "snapshot_free_blocks": state.get("free_blocks"),
        "snapshot_inflight_prefill_reserved_blocks": state.get("inflight_prefill_reserved_blocks"),
        "snapshot_inflight_prefill_request_ids": state.get("inflight_prefill_request_ids"),
        "inflight_prefill_running_rows": inflight_rows,
        "snapshot_pending": pending,
        "running": (state.get("running") or {}).get("requests", []),
        "waiting": state.get("waiting"),
        "skipped_waiting": state.get("skipped_waiting"),
        "held_peer_request_ids": held,
        "held_peer_count": len(held),
        "observer_errors": observer_errors,
        "target_refusals_in_ring": len(refusals),
        "last_target_refusal": selected,
        "linked_coordinator_requirements": linked,
        "same_schedule_connector_lookup": lookup,
        "native_refusal_inference": branch,
        "circular_wait_support": support,
        "support_condition": "The same live target has >=32 empty schedules over >=1 s with zero visible scheduler jobs/transfers; a same-window async target allocation returns None at an identified native gate while at least one running peer is held. This supports a source-local wait cycle in the observed diagnostic window, not a full-service outcome or intervention effect.",
        "scope": "Bounded in-process diagnostic; refusal-stage inference uses pinned synchronous KVCacheManager source and observed coordinator returns. Missing linked calls remain UNKNOWN. No performance ranking.",
    }


def analyze_session(session: Path) -> dict[str, Any]:
    require(session.name == SESSION_NAME and session.is_dir() and not session.is_symlink(),
            "wrong or missing r02 diagnostic session")
    plan_path = session / "plan.json"
    require(sha256(plan_path) == PLAN_SHA256, "r02 plan SHA differs")
    plan = read_json(plan_path)
    receipt_path = session / "receipt.json"
    receipt = read_json(receipt_path)
    require(receipt.get("plan_sha256") == PLAN_SHA256 and
            receipt.get("status") == "CELLS_COMPLETE" and
            receipt.get("held_lock_device_inode") == plan["expected_lock_device_inode"],
            "diagnostic group receipt incomplete or inconsistent")
    cells = receipt.get("cells")
    require(isinstance(cells, list) and len(cells) == 1, "expected one diagnostic cell")
    cell = cells[0]
    require(cell.get("arm") == "ltr_t30_q1" and cell.get("exit_code") == 0 and
            cell.get("timed_out") is False and cell.get("archive_status") == "VERIFIED" and
            cell.get("gpu_process_state_after") == "EMPTY" and
            cell.get("gpu_process_rows_after") == [],
            "cell did not exit cleanly with a verified archive and empty GPU")
    archive, hashes, hashes_sha = verify_archive(session / CELL_NAME)
    for name in ("environment.json", "safe-cap-qualification.json", "engine_args.json",
                 "config.json", "status.json", "raw.json", "no-progress-observer.json"):
        require(name in hashes, f"missing required archive file: {name}")
    env = read_json(archive / "environment.json")
    sources = env.get("vllm_source_sha256") or {}
    require(env.get("vllm") == "0.26.0" and
            sources.get("v1/core/sched/scheduler.py") == SCHEDULER_SHA256 and
            sources.get("v1/core/kv_cache_manager.py") == KV_MANAGER_SHA256 and
            (env.get("source_sha256") or {}).get("native_no_progress_observer.py") == OBSERVER_SHA256,
            "pinned native source or observer identity differs")
    require(sha256(Path(__file__).with_name("candidate_g64_liveness_diag_r02") / "manifest.json") ==
            PACKAGE_MANIFEST_SHA256, "local frozen r02 package manifest differs")
    qualification = read_json(archive / "safe-cap-qualification.json")
    args = read_json(archive / "engine_args.json")
    config = read_json(archive / "config.json")
    require(qualification.get("status") == "QUALIFIED" and
            qualification.get("watermark_blocks") == 0 and
            qualification.get("usable_blocks") == 4096 and
            args.get("scheduler_reserve_full_isl") is True and
            config.get("ltr_config") == {"threshold": 30, "quantum": 1} and
            config.get("measurement_mode") == "admission_refusal_diagnostic",
            "fixed native allocation or diagnostic config differs")
    status = read_json(archive / "status.json")
    raw = read_json(archive / "raw.json")
    observer = read_json(archive / "no-progress-observer.json")
    if observer.get("status") == "SNAPSHOT_CAPTURED":
        require(status.get("status") == "DIAGNOSTIC_CAPTURED" and
                status.get("expected_diagnostic_stop") is True and
                raw.get("status") == "INCOMPLETE" and
                "NoProgressSnapshot" in str(raw.get("error")),
                "snapshot and cell/raw diagnostic stop disagree")
    else:
        require(status.get("status") == "DIAGNOSTIC_NOT_REPRODUCED" and
                raw.get("status") == "COMPLETE", "non-snapshot diagnostic status disagrees")
    result = analyze_snapshot(observer, qualification["watermark_blocks"])
    return {
        "schema_version": 1,
        "status": "SOURCE_LOCALIZATION_ONLY" if result["status"] == "SNAPSHOT_VALIDATED" else
                  "DIAGNOSTIC_NOT_REPRODUCED",
        "session_name": SESSION_NAME,
        "plan_sha256": PLAN_SHA256,
        "receipt_sha256": sha256(receipt_path),
        "package_manifest_sha256": PACKAGE_MANIFEST_SHA256,
        "archive_manifest_sha256": hashes_sha,
        "archive_files_verified": len(hashes),
        "observer_sha256": OBSERVER_SHA256,
        "native_source_sha256": {"scheduler": SCHEDULER_SHA256,
                                 "kv_cache_manager": KV_MANAGER_SHA256},
        "capture_status": raw.get("status"),
        "capture_error": raw.get("error"),
        "requests_completed": status.get("requests_completed"),
        "planned_requests": config.get("requests"),
        "diagnostic": result,
        "relationship_to_prior_performance_run": "The separate r02 performance run observed a long repeat/no-output episode but did not record actual allocate_slots/coordinator/lookup inputs. Its exact native refusal branch remains UNKNOWN. This r02 liveness diagnostic directly observes the reservation-gate branch in a new run of the same frozen G64/T30/Q1 setting; observer overhead and the deliberate early stop bar performance ranking or an equality claim between executions.",
        "prohibited_claims": ["performance ranking", "equal-work speedup", "general deadlock proof",
                              "statistical stability", "method benefit"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, default=Path(__file__).with_name(SESSION_NAME))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze_session(args.session)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"],
                      "branch": result["diagnostic"].get("native_refusal_inference", {}).get("branch", "UNKNOWN"),
                      "circular_wait_support": result["diagnostic"]["circular_wait_support"],
                      "archive_files_verified": result["archive_files_verified"]}, sort_keys=True))


if __name__ == "__main__":
    main()
