#!/usr/bin/env python3
"""CPU counterexample for the frozen G64 T30/Q1 adapter; never runs vLLM/GPU.

This reproduces a possible accepted-intent/native-admission gap using the exact
pinned guard formulas. It does not reconstruct the unlogged r02 allocator state.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

# Import frozen package modules without creating __pycache__ inside the package.
sys.dont_write_bytecode = True

BASE = Path(__file__).resolve().parent
PACKAGE = BASE / "candidate_g64_perf_r01"
PINNED = BASE / "liveness_pinned_sources_20260930"
ARCHIVE = (BASE / "moe-a-g64-perf-remaining-session-r02-20260930" /
           "cell-00-ltr_t30_q1" / "archive")
EXPECTED_PACKAGE_MANIFEST = "2755945122e3c346ca7e7e5ceda0d4668a8e1d3c90ba5c8d9d7b6ab3e89e0eb9"
EXPECTED_SCHEDULER = "2ed2a550b6558b2495eda845a97ae38bcf0225027b9e25fbf00fc3880c1d3941"
EXPECTED_KV_MANAGER = "3f4af8d247f3fe9570b0132818b832b66ae6a2ac12942588828f899f6ff77ccf"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def check_frozen_sources() -> dict[str, str]:
    identities = {
        "package_manifest": (PACKAGE / "manifest.json", EXPECTED_PACKAGE_MANIFEST),
        "scheduler": (PINNED / "scheduler.py", EXPECTED_SCHEDULER),
        "kv_cache_manager": (PINNED / "kv_cache_manager.py", EXPECTED_KV_MANAGER),
    }
    actual = {key: file_sha256(path) for key, (path, _) in identities.items()}
    for key, (_, expected) in identities.items():
        assert actual[key] == expected, f"frozen {key} identity differs"
    selector_text = (PACKAGE / "pkg" / "ltr_style_selected.py").read_text()
    adapter_text = (PACKAGE / "pkg" / "ltr_style_native.py").read_text()
    scheduler_text = (PINNED / "scheduler.py").read_text()
    kv_text = (PINNED / "kv_cache_manager.py").read_text()
    for body, needle in (
        (selector_text, "free_blocks >= target.remaining_blocks"),
        (adapter_text, "if scheduler.skipped_waiting:return 'unrelated blocked native queue'"),
        (adapter_text, "pool.get_num_free_blocks()>=view(target).remaining_blocks"),
        (adapter_text, "release('native admission censored; rescan without resetting counters')"),
        (scheduler_text, "reserved_blocks = self._inflight_prefill_reserved_blocks()"),
        (scheduler_text, "reserved_blocks=reserved_blocks"),
        (kv_text, "available_blocks = self.block_pool.get_num_free_blocks() - reserved_blocks"),
        (kv_text, "if required_blocks > available_blocks:"),
    ):
        assert needle in body, f"frozen code guard changed: {needle}"
    return actual


def cpu_counterexample() -> dict:
    sys.path.insert(0, str(PACKAGE / "pkg"))
    from ltr_style_selected import LTRStyleSelected, Request
    from recovery_service_components import LTRCounters
    from staged_save_contract import RequestState, recovery_guard

    # One positive native prefill/recompute scheduling turn consumes Q1 even
    # when the application has received no new output token.
    counters = LTRCounters(threshold=30, quantum=1)
    for _ in range(30):
        counters.begin_schedule(["recovery"])
        counters.after_schedule([])
    assert counters.begin_schedule(["recovery"])["recovery"] == -1
    counters.after_schedule(["recovery"])
    after_non_output_compute = counters.states["recovery"].quantum_remaining
    next_priority = counters.begin_schedule(["recovery"])["recovery"]
    counters.after_schedule([])
    assert (after_non_output_compute, next_priority) == (0, 0)

    # A target with 96 offloaded history tokens needs six 16-token KV blocks.
    # Another in-flight prefill still reserves one block. These are synthetic
    # values chosen to demonstrate the strict inequality, not r02 measurements.
    free_blocks = 6
    target = RequestState("target", computed=0, prompt=80, output=16,
                          max_output=1024, status="PREEMPTED", blocks=())
    target_required_blocks = target.remaining_blocks
    inflight_prefill_reserved_blocks = 1
    other_growth_blocks = 1
    assert target_required_blocks == 6
    rows = [
        Request("target", arrival=0.0, status="PREEMPTED",
                remaining_blocks=target_required_blocks, output_tokens=16),
        Request("inflight-prefill", arrival=1.0, status="RUNNING",
                remaining_blocks=inflight_prefill_reserved_blocks),
    ]
    selector = LTRStyleSelected(threshold=30, quantum=1)
    for _ in range(30):
        intent = selector.begin_step(rows, free_blocks, free_slots=31)
        assert intent.action == "NOOP"
        selector.after_step({})

    cycles = []
    for _ in range(5):
        # This mirrors the adapter's skipped-queue and sequence-slot checks;
        # neither incorporates the native in-flight prefill reservation.
        intent = selector.begin_step(
            rows, free_blocks, free_slots=31,
            executable=lambda proposal: None if proposal.action == "PRIORITIZE_WAITING" else "not needed",
        )
        assert intent.action == "PRIORITIZE_WAITING"
        selector.accept(intent)
        adapter_reserves_target = free_blocks >= target_required_blocks
        peer_growth_held = not recovery_guard(
            target, free_blocks, other_growth_blocks)["other_growth_allowed"]
        # Pinned KVCacheManager.allocate_slots first checks full fit against raw
        # free, then checks required <= free - reserved for the async load.
        native_full_fit = target_required_blocks <= free_blocks
        native_async_admit = (target_required_blocks <=
                              free_blocks - inflight_prefill_reserved_blocks)
        assert adapter_reserves_target and peer_growth_held and native_full_fit
        assert not native_async_admit
        selector.after_step({})  # Native scheduled no tokens in this example.
        selector.release_active()  # Frozen adapter's backend-censored branch.
        cycles.append({
            "intent": intent.action,
            "adapter_reserves_target": adapter_reserves_target,
            "peer_growth_held": peer_growth_held,
            "native_full_fit": native_full_fit,
            "native_async_admit": native_async_admit,
            "target_priority": selector.counters.states["target"].priority,
            "target_quantum_remaining": selector.counters.states["target"].quantum_remaining,
        })
    assert all(c["target_priority"] == -1 and c["target_quantum_remaining"] == 1
               for c in cycles)
    return {
        "q1_non_output_compute": {
            "quantum_after_one_positive_schedule": after_non_output_compute,
            "priority_at_next_begin": next_priority,
        },
        "synthetic_allocator_state": {
            "free_blocks": free_blocks,
            "target_required_blocks": target_required_blocks,
            "inflight_prefill_reserved_blocks": inflight_prefill_reserved_blocks,
            "other_growth_blocks": other_growth_blocks,
            "free_slots": 31,
            "async_load_assumed": True,
        },
        "repeat_cycles": cycles,
    }


def archived_observations() -> dict:
    raw = json.loads((ARCHIVE / "raw.json").read_text())
    selected = json.loads((ARCHIVE / "selective-store.json").read_text())
    status = json.loads((ARCHIVE / "status.json").read_text())
    origin = raw["measurement_origin_perf_counter_s"]
    target = "measured/memory-train-article-0000464-bc945a3d"
    previous_target = "measured/memory-train-article-0005122-a44f994f"
    events = selected["events"]
    first = next(e for e in events if e["step"] == 833 and e["event"] == "accept"
                 and e.get("target_id") == target)
    expiry = next(e for e in events if e["step"] == 833 and e["event"] == "release"
                  and e.get("target") == previous_target and e.get("reason") == "quantum_expired")
    commit = next(e for e in events if e["step"] == 832 and e["event"] == "commit_check"
                  and e.get("target") == previous_target)
    loop = [e for e in events if 833 <= e["step"] <= 105050 and
            ((e["event"] == "accept" and e.get("target_id") == target and
              e.get("action") == "PRIORITIZE_WAITING") or
             (e["event"] == "backend_censored" and e.get("target") == target and
              e.get("reason") == "PREEMPTED") or
             (e["event"] == "release" and e.get("target") == target and
              e.get("reason", "").startswith("native admission censored")))]
    counts = Counter(e["event"] for e in loop)
    assert counts == {"accept": 104218, "backend_censored": 104218, "release": 104218}
    last_prior_output = max(e["received_s"] for e in raw["output_events"]
                            if e["external_request_id"] == "measured/memory-train-article-0005122")
    commit_time = commit["host_perf_counter_s"] - origin
    assert last_prior_output < commit_time
    return {
        "cell_status": status["status"],
        "completed_of_planned": [status["requests_completed"], len(raw["requests"])],
        "runtime_error": status["error"],
        "schedule_calls": selected["schedule_calls"],
        "backend_censored_repeat_steps": [833, 105050],
        "backend_censored_repeat_count": counts["backend_censored"],
        "repeat_target": target,
        "first_repeat_action": first["action"],
        "previous_target_q1_expired_at_s": expiry["host_perf_counter_s"] - origin,
        "previous_target_commit_at_s": commit_time,
        "previous_target_last_returned_output_at_s": last_prior_output,
        "last_any_returned_output_at_s": max(e["received_s"] for e in raw["output_events"]),
        "engine_calls": raw["engine_call_count"],
        "engine_returns": raw["engine_return_count"],
        "unmeasured_runtime_fields": [
            "per-step free KV blocks", "in-flight prefill reservation blocks",
            "whether target load_kv_async was true", "native allocation failure branch",
            "waiting and skipped queue head at rejection",
        ],
    }


def main() -> None:
    report = {
        "schema_version": 1,
        "scope": "CPU causal counterexample plus archived observations; not an observed exact native rejection branch",
        "frozen_source_sha256": check_frozen_sources(),
        "archived_observations": archived_observations(),
        "cpu_counterexample": cpu_counterexample(),
        "inference": (
            "The accepted PRIORITIZE_WAITING intent need not imply native async KV admission: "
            "the adapter compares required blocks with raw free blocks, while the pinned native "
            "allocator can subtract other in-flight prefill reservations. With Q1, a positive "
            "recompute/prefill turn can also expire the quantum before any new output. The archive "
            "confirms the repeated accept/censor/release loop but does not record the exact native "
            "allocator or queue branch that caused its first rejection."
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
