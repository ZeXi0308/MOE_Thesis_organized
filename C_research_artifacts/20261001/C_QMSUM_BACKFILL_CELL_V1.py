#!/usr/bin/env python3
"""QMSum200 density-order native control or exact-allocator backfill probe.

Both arms use the same model/engine/generation lifecycle and source order.
Only the measured backfill arm changes the pinned scheduler's failed-WAITING
branch and restores deferred requests at the end of that schedule call.
"""
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
import textwrap
import time
from types import MethodType

import C_LONG_DOCUMENT_QA_DENSITY_CELL_V1 as base
from C_LONG_DOCUMENT_QA_DENSITY_ORDER_V1 import density_order
from C_SPARE_CELL_HELPERS_V1 import dump, sha

N, CAP = 200, 512
SCHEDULER_SHA = "2ed2a550b6558b2495eda845a97ae38bcf0225027b9e25fbf00fc3880c1d3941"


def pinned_scheduler_source(native_schedule):
    func = getattr(native_schedule, "__func__", None)
    path = inspect.getsourcefile(func) if func is not None else None
    if func is None or func.__name__ != "schedule" or path is None or sha(Path(path)) != SCHEDULER_SHA:
        raise RuntimeError("native pinned scheduler source differs")
    return func, Path(path), textwrap.dedent(inspect.getsource(func))


def make_backfill_schedule(native_schedule, probe, restore):
    """Compile exactly two bounded edits into the pinned schedule method."""
    func, path, source = pinned_scheduler_source(native_schedule)
    failed_head = (
        "                if new_blocks is None:\n"
        "                    # The request cannot be scheduled.\n\n"
        "                    # NOTE: we need to untouch the request from the encode cache\n"
        "                    # manager\n"
        "                    if request.has_encoder_inputs:\n"
        "                        self.encoder_cache_manager.free(request)\n"
        "                    break\n"
    )
    failed_backfill = failed_head.replace(
        "                    break\n",
        "                    if _c_qmsum_backfill_probe(self, request_queue, request):\n"
        "                        continue\n"
        "                    break\n",
    )
    restore_at_end = (
        "            if step_skipped_waiting:\n"
        "                self.skipped_waiting.prepend_requests(step_skipped_waiting)\n"
    )
    restore_patched = restore_at_end + "            _c_qmsum_backfill_restore(self)\n"
    if source.count(failed_head) != 1 or source.count(restore_at_end) != 1:
        raise RuntimeError("pinned scheduler waiting branch anchors differ")
    patched = source.replace(failed_head, failed_backfill).replace(
        restore_at_end, restore_patched)
    scope = dict(func.__globals__)
    scope.update(_c_qmsum_backfill_probe=probe, _c_qmsum_backfill_restore=restore)
    namespace = {}
    exec(compile(patched, str(path), "exec"), scope, namespace)
    return MethodType(namespace["schedule"], native_schedule.__self__)


def restore_waiting_order(waiting, deferred):
    """Restore skipped FCFS heads, retaining un-restored entries on error."""
    original_order = [r.request_id for r in deferred]
    while deferred:
        request = deferred[-1]
        waiting.prepend_request(request)
        deferred.pop()
    return original_order


def validate_inputs(inputs):
    config = json.loads((inputs / "config.json").read_text())
    workload = json.loads((inputs / "workload.json").read_text())
    expected_model = dict(id=base.MODEL_ID, revision=base.MODEL_REVISION,
                          tokenizer_revision=base.MODEL_REVISION)
    if (config.get("schema") != "c-longbench-qmsum-full-config-v1"
            or workload.get("schema") != "c-longbench-qmsum-full-workload-v1"
            or config.get("task") != "qmsum" or workload.get("task") != "qmsum"
            or config.get("workload_sha256") != sha(inputs / "workload.json")
            or config.get("requests") != N or config.get("output_tokens") != CAP
            or config.get("max_model_len") != 4096 or config.get("model") != expected_model
            or workload.get("arrival_traces_s") != [0.0] * N
            or workload.get("sampling") != dict(temperature=0.0, max_tokens=CAP,
                                                min_tokens=0, ignore_eos=False, stop=[])):
        raise RuntimeError("official QMSum input/model/sampling contract differs")
    runtime = config.get("proposed_runtime", {})
    if any(runtime.get(k) != v for k, v in dict(max_num_seqs=128,
            max_num_batched_tokens=1024, usable_kv_blocks=4096, prefix_caching=True).items()):
        raise RuntimeError("unchanged native resource contract differs")
    rows = workload.get("requests", [])
    if (len(rows) != N or len({r["request_id"] for r in rows}) != N
            or [r["source_index"] for r in rows] != list(range(N))
            or [r["example_index"] for r in rows] != list(range(N))):
        raise RuntimeError("all200 source-order inventory differs")
    for row in rows:
        ids = row.get("prompt_token_ids")
        if (not isinstance(ids, list) or not ids or len(ids) + CAP > 4096
                or any(type(t) is not int or t < 0 for t in ids)
                or any(not isinstance(row.get(k), str) or not row[k] for k in
                       ("prompt", "raw_prompt", "chat_user_content", "question"))
                or not row.get("answers") or any(not isinstance(a, str) for a in row["answers"])):
            raise RuntimeError("QMSum request fields or context bound differs")
    return config, workload, rows


def run(inputs, output, parent, model_dir, model_manifest):
    mode = output.parent.name
    if mode not in ("control", "backfill"):
        raise RuntimeError("QMSum backfill cell output parent must be control/backfill")
    config, _, rows = validate_inputs(inputs)
    receipt = json.loads((Path(__file__).parent / "qmsum_density_order_v1.json").read_text())
    if receipt["workload_sha256"] != config["workload_sha256"] or receipt["request_count"] != N:
        raise RuntimeError("frozen QMSum order receipt differs")
    expected_order = receipt["source_indices_in_submission_order"]
    old_validate, old_generate, old_reset = base.validate_inputs, base.generate, base.reset_warmup_cache
    old_count, old_cap = base.REQUEST_COUNT, base.OUTPUT_CAP

    def generate(engine, requests, max_tokens, *, run_id, output_dir, max_seconds=900):
        # The legacy run() initially reads its old order receipt. Every call
        # replaces that unused value here before generate() can enqueue work.
        base.DENSITY_REORDER, base.EXPECTED_ORDER = density_order, expected_order
        scheduler = engine.engine_core.engine_core.scheduler
        manager = scheduler.kv_cache_manager
        native_schedule, raw_allocate = scheduler.schedule, manager.allocate_slots
        pinned_scheduler_source(native_schedule)
        call, allocations, detailed, first, seen = -1, {}, [], [], set()
        deferred, scans, restored = [], [], []
        scan = None
        expected_external = {run_id + "/" + row["request_id"] for row in requests}

        def external(rid):
            state = engine.output_processor.request_states[rid]
            value = state.external_req_id
            if state.request_id != rid or value not in expected_external:
                raise RuntimeError("native internal/external request mapping differs")
            return value

        def close_scan(accepted_id=None):
            nonlocal scan
            if scan is None:
                return
            scan["accepted_external_request_id"] = (
                external(accepted_id) if accepted_id is not None else None)
            scan["scan_host_s"] = time.perf_counter() - scan.pop("started_host_s")
            scan["scan_process_cpu_s"] = time.process_time() - scan.pop("started_cpu_s")
            scans.append(scan)
            scan = None

        def restore_deferred(s):
            if s is not scheduler or not deferred:
                return
            original_order = [external(r.request_id) for r in deferred]
            restore_waiting_order(s.waiting, deferred)
            restored.append(dict(call=call, external_request_ids=original_order))

        def probe_failed_head(s, request_queue, request):
            nonlocal scan
            if (s is not scheduler or request_queue is not s.waiting
                    or s.skipped_waiting or request.status.name != "WAITING"
                    or request.num_preemptions != 0 or request.has_encoder_inputs
                    or any(r.status.name == "PREEMPTED" for r in s.waiting)):
                return False
            popped = request_queue.pop_request()
            if popped is not request:
                request_queue.prepend_request(popped)
                raise RuntimeError("native waiting head changed during backfill")
            deferred.append(request)
            if scan is None:
                scan = dict(call=call, started_host_s=time.perf_counter(),
                            started_cpu_s=time.process_time(),
                            skipped_external_request_ids=[],
                            candidate_allocation_attempts=[])
            scan["skipped_external_request_ids"].append(external(request.request_id))
            return True

        raw_schedule = (make_backfill_schedule(native_schedule, probe_failed_head,
                                               restore_deferred)
                        if run_id == "measured" and mode == "backfill"
                        else native_schedule)

        def allocate(request, num_new_tokens, *args, **kwargs):
            if scan is not None:
                scan["candidate_allocation_attempts"].append(
                    external(request.request_id))
            previous = int(request.num_computed_tokens)
            hit = int(kwargs.get("num_new_computed_tokens", 0))
            result = raw_allocate(request, num_new_tokens, *args, **kwargs)
            if result is not None:
                rid = request.request_id
                if scan is not None:
                    close_scan(rid)
                event = dict(external_request_id=external(rid), internal_request_id=rid,
                    previous_computed_tokens=previous, new_prefix_cached_tokens=hit,
                    prompt_length_tokens=len(request.prompt_token_ids),
                    allocated_compute_tokens=int(num_new_tokens))
                allocations[rid] = event
                if rid not in seen:
                    first.append(dict(event, schedule_call=call))
                    seen.add(rid)
            return result

        def schedule(*args, **kwargs):
            nonlocal call
            call += 1
            allocations.clear()
            try:
                result = raw_schedule(*args, **kwargs)
            finally:
                # The source-level end hook handles successful calls. This
                # also restores popped heads if native scheduling raises.
                if scan is not None:
                    close_scan()
                restore_deferred(scheduler)
            served = []
            for rid, tokens in result.num_scheduled_tokens.items():
                event = dict(allocations[rid])
                before = event["previous_computed_tokens"] + event["new_prefix_cached_tokens"]
                prefill = min(tokens, max(0, event["prompt_length_tokens"] - before))
                served.append(dict(event, scheduled_tokens=tokens,
                    computed_before_execution=before, scheduled_prefill_tokens=prefill,
                    scheduled_decode_tokens=tokens-prefill))
            detailed.append(dict(call=call, requests=served,
                scheduled_tokens=sum(v["scheduled_tokens"] for v in served),
                prefill_tokens=sum(v["scheduled_prefill_tokens"] for v in served),
                decode_tokens=sum(v["scheduled_decode_tokens"] for v in served)))
            return result

        scheduler.schedule, manager.allocate_slots = schedule, allocate
        try:
            return old_generate(engine, requests, max_tokens, run_id=run_id,
                                output_dir=output_dir, max_seconds=max_seconds)
        finally:
            scheduler.schedule, manager.allocate_slots = native_schedule, raw_allocate
            dump(output_dir / (run_id + "-service-mix.json"), dict(
                schema="c-qmsum-native-service-mix-v1", first_successful_allocation=first,
                scheduler_calls=detailed,
                scope="Native successful allocation state and schedule actions; prefill/decode tokens classified by prompt boundary after the current local cache hit. Host observer cost is inside service. No future EOS, inferred eviction events, or GPU-kernel timing."))
            if run_id == "measured":
                dump(output_dir / "measured-backfill.json", dict(
                    schema="c-qmsum-backfill-action-v1", mode=mode,
                    scheduler_source_sha256=SCHEDULER_SHA,
                    scans=scans, restored=restored,
                    skipped_request_count=sum(len(x["skipped_external_request_ids"]) for x in scans),
                    accepted_after_skip_count=sum(x["accepted_external_request_id"] is not None for x in scans),
                    scan_host_s=sum(x["scan_host_s"] for x in scans),
                    scan_process_cpu_s=sum(x["scan_process_cpu_s"] for x in scans),
                    scope="Only measured backfill alters failed WAITING head handling; native running order/cache allocator unchanged. Scan timers cover only intervals after a skipped head until the next successful allocator result or end of call; full policy CPU cost is inside episode host wall time, not additive."))

    def reset_with_shape(engine):
        started = time.perf_counter()
        pre = old_reset(engine)
        dump(output / "warmup-large-odd-pre-reset.json", pre)
        synthetic = dict(request_id="synthetic-source0-prefix1023", source_index=0,
                         prompt_token_ids=rows[0]["prompt_token_ids"][:1023])
        if len(synthetic["prompt_token_ids"]) != 1023:
            raise RuntimeError("source0 lacks warmup prefix")
        generate(engine, [synthetic], 1, run_id="warmup-large-odd", output_dir=output)
        calls = json.loads((output / "warmup-large-odd-steps.json").read_text())["scheduler_calls"]
        if len(calls) != 1 or calls[0]["scheduled_tokens_total"] != 1023:
            raise RuntimeError("common shape warmup did not execute1023 tokens")
        post = old_reset(engine)
        dump(output / "shape-warmup-source.json", dict(policy="whole", source_index=0,
            prompt_tokens=1023, max_tokens=1, input_workload_sha256=config["workload_sha256"],
            total_shape_control_wall_s=time.perf_counter()-started,
            measurement_prefix_reset_file="prefix-cache-reset.json"))
        return post

    base.validate_inputs, base.generate, base.reset_warmup_cache = validate_inputs, generate, reset_with_shape
    base.REQUEST_COUNT, base.OUTPUT_CAP = N, CAP
    try:
        base.run(inputs, output, parent, model_dir, model_manifest)
    finally:
        base.validate_inputs, base.generate, base.reset_warmup_cache = old_validate, old_generate, old_reset
        base.REQUEST_COUNT, base.OUTPUT_CAP = old_count, old_cap
        if output.is_dir():
            dump(output / "qmsum-adapter-source.json", dict(
                task="qmsum", policy_mode=mode, requests=N, output_cap=CAP,
                adapter_sha256=sha(Path(__file__)), base_cell_sha256=sha(Path(base.__file__)),
                order_receipt_sha256=sha(Path(__file__).parent / "qmsum_density_order_v1.json"),
                scope="Same native128/1024/4096 QMSum200 density order and common shape warmup; development control versus source-pinned allocator-driven backfill of one native failed WAITING head; no method or performance claim"))
            path = output / "status.json"
            if path.exists():
                status = json.loads(path.read_text())
                status["scientific_scope"] = ("Official QMSum200 all-arrival-zero development "
                    f"{mode} cell: same density order, EOS/512 ceiling and native resources; "
                    "backfill arm only probes later waiting requests after exact native allocator "
                    "failure of an unstarted head. No held-out or benefit claim.")
                dump(path, status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs", "output", "parent", "model-dir", "model-manifest"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    run(args.inputs, args.output, args.parent, args.model_dir, args.model_manifest)
