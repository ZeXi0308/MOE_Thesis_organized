"""One CPU maintenance counterexample; no model, vLLM install, or GPU.

Run: python3 -B cpu_preparation/check_store_replay_eviction.py
Uses unchanged native AST function bodies plus the official v0.26.0 LRU/base
classes. Request/config/spec objects and transfer completion are CPU fixtures.
This executes the STORE builder, manager and prefix lookup, not inference.
"""
from __future__ import annotations

import ast
import ctypes
import hashlib
import json
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Iterable
from pathlib import Path
from types import SimpleNamespace as NS

from check_recompute_store_semantics import ROOT, MANAGER, SCHED, load_native_definitions


POLICY_DIR = ROOT / "related_sources/vllm_v0_26_0_cpu_policy"
OUTPUT = ROOT / "cpu_preparation/store_replay_eviction_result.json"
OFFICIAL = "https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/vllm/v1/kv_offload/cpu/policies/"
EXPECTED_SHA = {
    "base.py": "ca9d2fa873813e2887b1aae1d4667835a64ef85371deb4f1c88702f08d32ef9f",
    "lru.py": "fd7ddee1f288f19197f5519fe7cd73d4e92a2460e9354892cfedcba4664b7f32",
}
A = [f"A{i}" for i in range(8)]
B = [f"B{i}" for i in range(8)]


def compile_nodes(nodes, path, ns):
    tree = ast.Module(body=[ast.ImportFrom(module="__future__",
        names=[ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), ns)


def load_sources():
    ns, references = load_native_definitions()
    ns.update(ctypes=ctypes, ABC=ABC, abstractmethod=abstractmethod,
              OrderedDict=OrderedDict, Iterable=Iterable, MEDIUM_CPU="CPU")
    for name, expected in EXPECTED_SHA.items():
        path = POLICY_DIR / name
        if not path.exists():
            raise SystemExit(f"BLOCKED_MISSING_OFFICIAL_SOURCE: fetch {OFFICIAL + name} to {path}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise SystemExit(f"BLOCKED_SOURCE_HASH_MISMATCH: {path}")
        classes = [n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef)]
        compile_nodes(classes, path, ns)
    ns["_CACHE_POLICIES"] = {"lru": ns["LRUCachePolicy"]}
    # Reuse the helper's unchanged manager methods and also its real constructor.
    manager_ast = next(n for n in ast.parse(MANAGER.read_text()).body
                       if isinstance(n, ast.ClassDef) and n.name == "CPUOffloadingManager")
    init = next(n for n in manager_ast.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    compile_nodes([init], MANAGER, ns)
    ns["CPUOffloadingManager"].__init__ = ns.pop("__init__")
    return ns, references


def fixture(ns):
    manager = ns["CPUOffloadingManager"](num_blocks=15, cache_policy="lru")
    context = NS(req_id="fixture")
    # Normal completed stores produce the chosen legal LRU order; no edits to
    # LRU containers, victim selection, refcounts or free-list contents.
    for key in B + [k for k in A if k != "A2"]:
        prepared = manager.prepare_store([key], context)
        assert prepared is not None and prepared.keys_to_store == [key]
        assert prepared.evicted_keys == []
        manager.complete_store([key], context)
    group = NS(tokens_per_block=16, tokens_per_chunk=16, hashes_per_chunk=1,
               group_idx=0, is_eagle_group=False, sliding_window_size_in_chunks=None,
               alignment_chunk_count=None)
    config = NS(kv_group_configs=[group], blocks_per_chunk=1,
                num_workers=1, offload_prompt_only=False)
    request = NS(request_id="A", num_tokens=128, num_prompt_tokens=128,
                 num_output_tokens=0, num_computed_tokens=64,
                 kv_transfer_params=None, is_finished=lambda: False)
    state = ns["RequestOffloadState"](config, request, NS(req_id="A"), NS(policy="block"))
    state.group_states[0].offload_keys = list(A)
    state.group_states[0].block_ids = list(range(1, 9))
    state.group_states[0].next_stored_chunk_idx = 8
    scheduler = ns["OffloadingConnectorScheduler"]()
    scheduler.__dict__.update(config=config, manager=manager, _req_status={"A": state},
        _sliding_window_groups=(), _lookup_groups=(0,), _jobs={}, _job_counter=0,
        _current_batch_allocated_block_ids=set(), _block_id_to_pending_jobs={},
        _current_batch_jobs_to_flush=set(),
        _connector_stats=NS(increase_counter=lambda *args: None),
        _events_tracker=NS(record_store=lambda *args: None))
    return scheduler, state


def snapshot(scheduler, state):
    manager = scheduler.manager
    residents = {k: {"block_id": v.block_id, "ref_cnt": v.ref_cnt,
                     "ready": v.is_ready} for k, v in manager._policy.blocks.items()}
    assert len(residents) <= 15 and manager._num_allocated_blocks <= 15
    return dict(residents=residents, lru_oldest_first=list(manager._policy.evictable_blocks),
        prefix_chunks={r: scheduler._maximal_prefix_lookup(keys, NS(req_id=r))
                       for r, keys in (("A", A), ("B", B))},
        cursor=state.group_states[0].next_stored_chunk_idx,
        free_blocks=manager._get_num_free_blocks(),
        allocated_blocks=manager._num_allocated_blocks,
        evictable_blocks=manager._num_evictable_cache_blocks,
        write_pending_blocks=manager._num_write_pending_blocks)


def run_arm(ns, replay):
    scheduler, state = fixture(ns)
    initial = snapshot(scheduler, state)
    assert initial["prefix_chunks"] == {"A": 2, "B": 8}
    assert initial["lru_oldest_first"][0] == "B0"
    assert initial["residents"]["B0"]["ref_cnt"] == 0
    if replay:
        state.group_states[0].next_stored_chunk_idx = 2
    # Same supplied GPU work/progress in both arms: 64 + 64 = 128 known tokens.
    jobs = scheduler._build_store_jobs(NS(num_scheduled_tokens={"A": 64}, finished_req_ids=set()))
    pending = snapshot(scheduler, state)
    transfers = []
    for jid, transfer in jobs.items():
        job = scheduler._jobs[jid]
        assert job.is_store
        transfers.append(dict(job_id=jid, keys=sorted(job.keys),
                              src_gpu_ids=transfer.src_spec.block_ids,
                              dst_cpu_ids=transfer.dst_spec.block_ids))
        # Synchronous fixture acknowledgement, not a worker/CUDA execution.
        scheduler.manager.complete_store(job.keys, state.req_context)
        state.transfer_jobs.remove(jid)
        del scheduler._jobs[jid]
    final = snapshot(scheduler, state)
    return dict(initial=initial, cursor_after_intervention=2 if replay else 8,
                pending=pending, submitted_transfers=transfers, final=final,
                removed_keys=sorted(set(initial["residents"]) - set(final["residents"])),
                added_keys=sorted(set(final["residents"]) - set(initial["residents"])))


def main():
    ns, references = load_sources()
    off, on = run_arm(ns, False), run_arm(ns, True)
    assert off["initial"] == on["initial"]
    assert off["final"] == off["initial"]
    assert off["submitted_transfers"] == []
    assert on["added_keys"] == ["A2"] and on["removed_keys"] == ["B0"]
    # Native prefix lookup defers (None) while A2 has a pending STORE.
    assert on["pending"]["prefix_chunks"] == {"A": None, "B": 0}
    assert on["pending"]["write_pending_blocks"] == 1
    assert on["final"]["prefix_chunks"] == {"A": 8, "B": 0}
    assert on["final"]["write_pending_blocks"] == 0
    assert len(on["submitted_transfers"]) == 1
    assert on["submitted_transfers"][0]["keys"] == ["A2"]
    files = [POLICY_DIR / name for name in EXPECTED_SHA] + [MANAGER, SCHED,
             Path(__file__).with_name("check_recompute_store_semantics.py"), Path(__file__)]
    provenance = [dict(path=str(p.relative_to(ROOT)), sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                  **({"url": OFFICIAL + p.name, "version": "v0.26.0"} if p.parent == POLICY_DIR else {}))
                  for p in files]
    result = dict(status="PASS_NATIVE_CPU_MAINTENANCE_COUNTEREXAMPLE",
        hypothesis="Re-admitting one missing target block can evict another request's prefix head, reducing its immediately loadable prefix.",
        fixture=dict(host_capacity_blocks=15, tokens_per_chunk=16, A_known_chunks=8,
                     A_initial_ready_chunks=7, B_initial_ready_chunks=8,
                     initial_store_cursor=8, cache_policy="official v0.26.0 LRU",
                     precondition="No in-flight transfers; B0 is an unprotected ref_cnt=0 oldest block."),
        arms={"off": off, "on": on},
        findings=dict(stored_chunks=1, evicted_chunks=1, A_prefix_change_chunks=6,
                      B_prefix_change_chunks=-8,
                      interpretation="A gains a complete reusable prefix while B loses contiguous prefix reuse. B1..B7 remain resident; this is not loss of eight physical blocks."),
        scope=dict(executed="Unchanged native STORE builder, prefix lookup, manager constructor/prepare_store/complete_store/lookup; official complete BlockStatus, CachePolicy and LRU classes.",
                   fixture_only="Opaque string keys, request/config/spec DTOs, supplied GPU block IDs and compute progress, synchronous successful STORE acknowledgement, no model values.",
                   limitations=["No GPU, tensor payload, actual transfer, worker completion path, concurrent scheduler or full inference engine.",
                                "One deterministic constructed LRU state, not evidence of occurrence in E246 or a workload frequency estimate.",
                                "No future requests/resumes are executed; prefix losses are not measured recomputation or wall-clock costs.",
                                "Does not establish incorrect outputs, a universal repair, a useful policy, or a paper contribution."]),
        native_ast_references=references, sources=provenance)
    result["verification_note"] = (
        "The first test assertion incorrectly expected A's pending prefix to be 2. "
        "Inspection showed native lookup correctly returns None (defer) while A2 "
        "is not ready; the assertion was corrected. No native body or final prediction changed."
    )
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "off_prefix": off["final"]["prefix_chunks"],
                      "on_prefix": on["final"]["prefix_chunks"],
                      "stored": on["added_keys"], "evicted": on["removed_keys"],
                      "result": str(OUTPUT.relative_to(ROOT))}))


if __name__ == "__main__":
    main()
