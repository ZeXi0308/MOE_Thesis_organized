#!/usr/bin/env python3
"""Full official QMSum200 density baseline with native service-mix observations.

Reuse the frozen model/engine/generation lifecycle. Only the official task,
request count and its official 512-token ceiling change. This is a domain
measurement, not a new scheduling policy or a fixed-grain rescue.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import C_LONG_DOCUMENT_QA_DENSITY_CELL_V1 as base
from C_LONG_DOCUMENT_QA_DENSITY_ORDER_V1 import density_order
from C_SPARE_CELL_HELPERS_V1 import dump, sha

N, CAP = 200, 512


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
                task="qmsum", requests=N, output_cap=CAP,
                adapter_sha256=sha(Path(__file__)), base_cell_sha256=sha(Path(base.__file__)),
                order_receipt_sha256=sha(Path(__file__).parent / "qmsum_density_order_v1.json"),
                scope="Same native128/1024/4096 density baseline, full official QMSum200, common shape warmup; no policy comparison or new mechanism"))
            path = output / "status.json"
            if path.exists():
                status = json.loads(path.read_text())
                status["scientific_scope"] = "Complete official QMSum200 density baseline with EOS/512 ceiling; all arrivals zero; a new natural task domain, not a fixed-grain success or a held-out method confirmation"
                dump(path, status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs", "output", "parent", "model-dir", "model-manifest"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    run(args.inputs, args.output, args.parent, args.model_dir, args.model_manifest)
