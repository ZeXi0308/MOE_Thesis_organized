#!/usr/bin/env python3
"""Native QMSum200 offline-default-budget8192 versus512 reference pair.

Keep density order, workload, model, KV, sequence cap and EOS sampling fixed.
Both arms receive identical first/last, 511/1023 and three-request8190 warmups.
"""
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
import textwrap
import time

import C_LONG_DOCUMENT_QA_DENSITY_CELL_V1 as base
from C_LONG_DOCUMENT_QA_DENSITY_ORDER_V1 import density_order
from C_SPARE_CELL_HELPERS_V1 import dump, sha

N, CAP = 200, 512
BASE_CELL_SHA = "1773491af46e11045c55c027d44bfd5a8eba0d3789befe7864cffda0e8591a9f"


def native_run_for_budget(budget):
    """Compile the pinned original run with only its two runtime budget literals changed."""
    path = Path(base.__file__)
    if sha(path) != BASE_CELL_SHA or inspect.getsourcefile(base.run) != str(path):
        raise RuntimeError("pinned native base run source differs")
    source = textwrap.dedent(inspect.getsource(base.run))
    requested = "max_num_batched_tokens=1024,"
    qualified = 'resolved["max_num_batched_tokens"] != 1024'
    if source.count(requested) != 1 or source.count(qualified) != 1:
        raise RuntimeError("pinned native budget source anchors differ")
    patched = source.replace(requested, f"max_num_batched_tokens={budget},")
    patched = patched.replace(qualified,
                              f'resolved["max_num_batched_tokens"] != {budget}')
    namespace = {}
    # Preserve the original module globals so the QMSum validation/generation
    # hooks installed below are read by the compiled run at call time.
    exec(compile(patched, str(path), "exec"), base.__dict__, namespace)
    return namespace["run"]


def qualified_shape_tokens(summary, calls, prompt_tokens, budget):
    scheduled = [int(call["scheduled_tokens_total"]) for call in calls]
    if (summary["status"] != "COMPLETE" or summary["request_count"] != 1
            or summary["output_tokens"] != 1 or not scheduled
            or sum(scheduled) != prompt_tokens
            or any(count < 1 or count > budget for count in scheduled)):
        raise RuntimeError(f"common {prompt_tokens}-token warmup differs")
    return scheduled


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
    if output.parent.name not in ("8192", "512"):
        raise RuntimeError("offline batch reference output parent must be 8192 or 512")
    budget = int(output.parent.name)
    native_run = native_run_for_budget(budget)
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
        raw_schedule, raw_allocate = scheduler.schedule, manager.allocate_slots
        call, allocations, detailed, first, seen = -1, {}, [], [], set()
        expected_external = {run_id + "/" + row["request_id"] for row in requests}

        def external(rid):
            state = engine.output_processor.request_states[rid]
            value = state.external_req_id
            if state.request_id != rid or value not in expected_external:
                raise RuntimeError("native internal/external request mapping differs")
            return value

        def allocate(request, num_new_tokens, *args, **kwargs):
            previous = int(request.num_computed_tokens)
            hit = int(kwargs.get("num_new_computed_tokens", 0))
            result = raw_allocate(request, num_new_tokens, *args, **kwargs)
            if result is not None:
                rid = request.request_id
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
            result = raw_schedule(*args, **kwargs)
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
            scheduler.schedule, manager.allocate_slots = raw_schedule, raw_allocate
            dump(output_dir / (run_id + "-service-mix.json"), dict(
                schema="c-qmsum-native-service-mix-v1", first_successful_allocation=first,
                scheduler_calls=detailed,
                scope="Native successful allocation state and schedule actions; prefill/decode tokens classified by prompt boundary after the current local cache hit. Host observer cost is inside service. No future EOS, inferred eviction events, or GPU-kernel timing."))

    def reset_with_shape(engine):
        started = time.perf_counter()
        initial_reset = old_reset(engine)
        dump(output / "warmup-shape-initial-reset.json", initial_reset)
        shapes = []
        post = None
        for prompt_tokens in (511, 1023):
            run_id = f"warmup-prefix{prompt_tokens}"
            synthetic = dict(request_id=f"synthetic-source0-prefix{prompt_tokens}",
                             source_index=0,
                             prompt_token_ids=rows[0]["prompt_token_ids"][:prompt_tokens])
            if len(synthetic["prompt_token_ids"]) != prompt_tokens:
                raise RuntimeError("source0 lacks common shape warmup prefix")
            wall_start = time.perf_counter()
            summary = generate(engine, [synthetic], 1, run_id=run_id, output_dir=output)
            calls = json.loads((output / (run_id + "-steps.json")).read_text())["scheduler_calls"]
            scheduled = qualified_shape_tokens(summary, calls, prompt_tokens, budget)
            post = old_reset(engine)
            dump(output / (run_id + "-reset.json"), post)
            shapes.append(dict(run_id=run_id, source_index=0,
                prompt_tokens=prompt_tokens, max_tokens=1,
                schedule_calls=len(calls), scheduled_tokens_per_call=scheduled,
                warmup_and_reset_wall_s=time.perf_counter() - wall_start,
                reset_status=post["drained_after"]["status"]))
        batch_id = "warmup-batch8190"
        batch_rows = [dict(request_id=f"synthetic-source{index}-prefix2730",
            source_index=index, prompt_token_ids=rows[index]["prompt_token_ids"][:2730])
            for index in (0, 1, 2)]
        if any(len(row["prompt_token_ids"]) != 2730 for row in batch_rows):
            raise RuntimeError("source0/1/2 lack the common 2730-token prefixes")
        batch_summary = generate(engine, batch_rows, 1, run_id=batch_id, output_dir=output)
        batch_mix = json.loads((output / (batch_id + "-service-mix.json")).read_text())
        scheduled = [call["scheduled_tokens"] for call in batch_mix["scheduler_calls"]]
        cached = sum(row["new_prefix_cached_tokens"]
                     for row in batch_mix["first_successful_allocation"])
        if (batch_summary["status"] != "COMPLETE" or batch_summary["request_count"] != 3
                or batch_summary["output_tokens"] != 3 or batch_summary["preemptions"] != 0
                or len(batch_mix["first_successful_allocation"]) != 3
                or sum(scheduled) + cached != 8190
                or not scheduled or any(not 1 <= tokens <= budget for tokens in scheduled)
                or any(call["decode_tokens"] for call in batch_mix["scheduler_calls"])):
            raise RuntimeError("common three-request8190 shape warmup differs")
        post = old_reset(engine)
        dump(output / (batch_id + "-reset.json"), post)
        batch_shape = dict(run_id=batch_id, source_indices=[0, 1, 2],
            prompt_tokens_per_request=[2730, 2730, 2730], request_count=3, max_tokens=1,
            prompt_tokens_total=8190, scheduled_tokens_per_call=scheduled,
            first_allocation_cached_tokens=cached, reset_status=post["drained_after"]["status"])
        dump(output / "shape-warmup-source.json", dict(policy="whole", source_index=0,
            shapes=shapes, batch_shape=batch_shape, native_batch_budget=budget,
            input_workload_sha256=config["workload_sha256"],
            total_shape_control_wall_s=time.perf_counter()-started,
            measurement_prefix_reset_file="prefix-cache-reset.json"))
        return post

    base.validate_inputs, base.generate, base.reset_warmup_cache = validate_inputs, generate, reset_with_shape
    base.REQUEST_COUNT, base.OUTPUT_CAP = N, CAP
    try:
        native_run(inputs, output, parent, model_dir, model_manifest)
    finally:
        base.validate_inputs, base.generate, base.reset_warmup_cache = old_validate, old_generate, old_reset
        base.REQUEST_COUNT, base.OUTPUT_CAP = old_count, old_cap
        if output.is_dir():
            dump(output / "qmsum-adapter-source.json", dict(
                task="qmsum", requests=N, output_cap=CAP,
                executed_batch_budget=budget,
                input_proposed_batch_budget=config["proposed_runtime"]["max_num_batched_tokens"],
                adapter_sha256=sha(Path(__file__)), base_cell_sha256=sha(Path(base.__file__)),
                order_receipt_sha256=sha(Path(__file__).parent / "qmsum_density_order_v1.json"),
                scope="Native QMSum200 density-order batch-budget reference at 8192 or 512; same inputs, 128 sequence slots, 4096 usable KV blocks, official 512 output cap and common 511/1023 and three-request8190 shape warmups; offline LLM_CLASS budget reference with max_num_seqs held at128, not a full-default configuration; no new controller"))
            path = output / "status.json"
            if path.exists():
                status = json.loads(path.read_text())
                status["scientific_scope"] = ("Complete official QMSum200 native "
                    f"{budget}-batched-token budget reference with density order, EOS/512 "
                    "ceiling and all arrivals zero; no new method or held-out claim")
                dump(path, status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs", "output", "parent", "model-dir", "model-manifest"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    run(args.inputs, args.output, args.parent, args.model_dir, args.model_manifest)
