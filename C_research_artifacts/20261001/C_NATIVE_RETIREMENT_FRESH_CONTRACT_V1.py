"""Input, resource, and output contracts for the fixed fresh 128-request cohort."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


INPUT_SHA256 = {
    "config.json": "634e7715618879775daf312fc98d77c1b25f4cd8e97007c101604ded317738a3",
    "workload.json": "9576395c9540c71be86dd182df03ab6d39cf1c961a6a24dc851d40991cf0a985",
    "INPUT_STATS.json": "797ccd408a60ee01bed2ac370f51f7145fa4f9ea3bcaecc5a1a4535afd034ac1",
}
ARRIVAL_SHA256 = "a6e1d41c88b40a336f287b77353332b5b051771de485626ac7c94ba1e1bf9fef"
SOURCE_SHA256 = "74da360f23826045b3e6ac6375411fdb15f003030aa74f2596ed08b857cb9212"
PLAN_SHA256 = "984e507fa0cdc27a3da8ba68cddc8fcbe051b5f268fa9a510ac270e900074b1d"
GENERATOR_SHA256 = "5d8e481f1092721b869058a8e715a2412a59f250a1defa8cdf465876af4e56c2"
MODEL_REVISION = "6d84c48581ece794365f2b8e9cfb043c68ade9c5"
KV_BYTES = 4097 * 2 * 1024 * 1024
ARMS = ("native_full", "native_max_bound", "native_retirement")


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def sha_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compact_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def validate_measurement(folder: Path, config: dict, workload: dict) -> dict:
    """Check the immutable fresh bytes and the identities consumed by capture."""
    folder = Path(folder)
    for name, digest in INPUT_SHA256.items():
        require(sha_file(folder / name) == digest, f"fresh input bytes changed: {name}")
    stats = json.loads((folder / "INPUT_STATS.json").read_text())
    require(stats["files_sha256"] == {name: INPUT_SHA256[name] for name in
                                   ("config.json", "workload.json")},
            "fresh input file receipt changed")
    require(config.get("schema") == workload.get("schema") == stats.get("schema")
            == "c-native-retirement-fresh-inputs-v1"
            and config.get("status") == stats.get("status") == "FRESH_INPUTS_GPU_UNRUN"
            and config.get("preregistered_plan_sha256") == PLAN_SHA256
            and stats.get("generator_sha256") == GENERATOR_SHA256,
            "fresh scope, plan, or generator identity changed")
    require(config.get("model") == {
        "dtype": "bfloat16", "id": "allenai/OLMoE-1B-7B-0924", "key": "olmoe",
        "revision": MODEL_REVISION, "tokenizer_revision": MODEL_REVISION,
    } and config.get("seed") == 20260905,
            "model or generation seed changed")
    source = config["source"]
    require(source.get("active_source_kind") == "parquet"
            and source.get("active_source_sha256") == SOURCE_SHA256
            and source.get("dataset_revision") == "b08601e04326c79dfdd32d625aee71d232d685c3"
            and source.get("hf_commit_attested") is False
            and source.get("arrow_equivalence_established") is False
            and source.get("historic_receipt_lineage_complete") is False
            and source.get("previous_sustained_all_128_row_text_token_ids_reproduced") is True
            and source.get("previous_eligible_identity_sha256")
                == stats.get("previous_eligible_identity_sha256")
            and stats.get("old_sustained_all_128_row_text_token_ids_checked") is True
            and stats.get("gh_anchor_count") == 192
            and stats.get("source_byte_checks", {}).get("status")
                == "LOCAL_BYTES_AND_ANCHORS_CHECKED"
            and stats["source_byte_checks"].get("parquet_sha256") == SOURCE_SHA256,
            "fresh source or prior-token compatibility declaration changed")
    require(config.get("requests") == 128 and config.get("prompt_tokens") == 3066
            and config.get("max_output_tokens") == 1024
            and config.get("ignore_eos") is False and config.get("min_tokens") == 0
            and workload.get("output_contract") == {
                "max_output_tokens": 1024, "ignore_eos": False, "min_tokens": 0,
                "actual_output_lengths": "UNKNOWN_UNTIL_EXECUTION",
            }, "request count or natural EOS contract changed")
    require(stats.get("selected_eligible_ranks") == [832, 959]
            and stats.get("previous_eligible_ranks") == [0, 831]
            and source.get("previous_eligible_ranks") == [0, 831],
            "fresh selected source ranks changed")
    arrivals = workload["arrival_traces_s"].get("poisson_v1")
    require(set(workload["arrival_traces_s"]) == {"poisson_v1"}
            and isinstance(arrivals, list) and len(arrivals) == 128
            and all(type(t) in (int, float) and math.isfinite(t) and t >= 0 for t in arrivals)
            and all(a <= b for a, b in zip(arrivals, arrivals[1:]))
            and arrivals[0] == 0 and arrivals[-1] == config.get("arrival_span_s") == 22.682329
            and compact_sha(arrivals) == stats.get("arrival_trace_compact_json_sha256")
                == ARRIVAL_SHA256
            and config.get("arrival_regime") == "poisson_v1"
            and config.get("arrival_seed") == 20261001
            and config.get("arrival_rate_per_s") == 5.0,
            "single frozen Poisson arrival trace changed")
    rows, prompts = workload["source_requests"], workload["actual_prompt_token_ids"]
    require(len(rows) == len(prompts) == 128, "fresh request count changed")
    lengths = [len(ids) for ids in prompts]
    require(min(lengths) == stats["prompt_tokens_min"] == 414
            and max(lengths) == stats["prompt_tokens_max"] == config["prompt_tokens"] == 3066
            and lengths == config["prompt_tokens_by_request"]
            and all(256 <= n <= 3072 and n + 1024 <= 4096 for n in lengths),
            "fresh prompt or model-length bound changed")
    for index, (row, ids) in enumerate(zip(rows, prompts)):
        text_sha = hashlib.sha256(row["prompt"].encode()).hexdigest()
        require(row["source_index"] == index
                and row["eligible_source_rank"] == 832 + index
                and row["document_id"] == row["request_id"]
                and row["request_id"] == f"memory-train-article-{row['dataset_row_index']:07d}"
                and row["prompt_token_count"] == lengths[index]
                and row["original_document_token_count"] == lengths[index]
                and row["max_output_tokens"] == 1024
                and row["document_sha256"] == row["prompt_sha256"] == text_sha
                and row["prompt_token_ids_sha256"] == compact_sha(ids),
                f"fresh article or token identity changed at rank {832 + index}")
    require(len({r["request_id"] for r in rows}) == 128
            and len({r["document_sha256"] for r in rows}) == 128
            and len({r["prompt_token_ids_sha256"] for r in rows}) == 128
            and all(a["dataset_row_end_exclusive"] <= b["dataset_row_index"]
                    for a, b in zip(rows, rows[1:])),
            "fresh article identities overlap")
    require(hashlib.sha256(json.dumps(workload, sort_keys=True).encode()).hexdigest()
            == config["workload_sha256"], "fresh logical workload changed")
    return dict(status="READY_FRESH_POLICY_TEST_INPUT", requests=128,
        input_sha256=dict(INPUT_SHA256), active_source_sha256=SOURCE_SHA256,
        selected_eligible_ranks=[832, 959], arrival_sha256=ARRIVAL_SHA256,
        preregistered_plan_sha256=PLAN_SHA256,
        scope="New relative to recorded C policy selection; historical source gaps remain")


def configure_measurement(base: dict, prompt_lengths: list[int], arm: str,
                          input_receipt: dict) -> dict:
    if arm not in ARMS:
        raise ValueError("unsupported fresh policy-test arm")
    require(len(prompt_lengths) == 128 and all(256 <= n <= 3072 for n in prompt_lengths)
            and max(prompt_lengths) == base["prompt_tokens"] == 3066,
            "fresh prompt lengths changed")
    variant = {"native_full": "native_recompute_native",
               "native_max_bound": "native_max_bound_fifo",
               "native_retirement": "native_retirement_fifo"}[arm]
    preemption = {"native_full": "native_recompute",
                  "native_max_bound": "native_max_bound_no_offload",
                  "native_retirement": "native_retirement_no_offload"}[arm]
    config = dict(base)
    config.update(requests=128, prompt_tokens=3072, output_tokens=1024,
        output_mode="eos", output_tokens_by_request={}, cap=32,
        engine_max_num_seqs=32, policy="static", max_seconds=300,
        variant=variant, commit_recheck=False, gate_arm="none",
        gate_scope="No selected-KV/offload gate; native scheduler plus named admission arm",
        global_cooldown_steps=None, population_mode="native",
        preemption_mode=preemption, measurement_mode="diagnostic",
        capture_mode="sparse", selective_save="off", store_scope="none",
        offload_gib=0, ignore_eos=False, min_tokens=0,
        fixed_kv_cache_memory_bytes=KV_BYTES, gpu_memory_utilization=.90,
        reservation_policy=arm, rotation_config=None, rotation_victim_order=None,
        prompt_tokens_role="fixed conservative eligibility ceiling 3072; observed maximum 3066; output_tokens is requested cap",
        evidence_ceiling="NATIVE_SERVING_INPROCESS_HOST_MEASUREMENT",
        metric_role="C fresh matched-input policy-test sparse measurement; not held-out confirmation",
        development_input_receipt=input_receipt)
    return config


def qualify_safe_cap(engine, config: dict) -> dict:
    """Preserve the old physical checks with the fixed 3072-token ceiling."""
    result = dict(status="QUALIFICATION_FAILED", measurement_status="UNRUN",
        formula="min(engine_max_num_seqs, usable_blocks // ceil((prompt_tokens + output_tokens) / block_size))")

    def check(condition, message):
        if not condition:
            raise ValueError(message)

    try:
        from vllm.v1.kv_cache_interface import FullAttentionSpec
        scheduler = engine.engine_core.engine_core.scheduler
        manager = scheduler.kv_cache_manager
        layout, pool, coordinator = scheduler.kv_cache_config, manager.block_pool, manager.coordinator
        groups = layout.kv_cache_groups
        result.update(group_count=len(groups), manager_group_count=manager.num_kv_cache_groups,
            coordinator_type=type(coordinator).__name__, watermark_blocks=manager.watermark_blocks,
            prefix_caching=manager.enable_caching, use_eagle=manager.use_eagle,
            num_lookahead_tokens=scheduler.num_lookahead_tokens,
            num_spec_tokens=scheduler.num_spec_tokens,
            dcp_world_size=scheduler.dcp_world_size, pcp_world_size=scheduler.pcp_world_size)
        check(len(groups) == manager.num_kv_cache_groups == 1, "requires one KV cache group")
        check(type(coordinator).__name__ == "KVCacheCoordinatorNoPrefixCache",
              "unverified KV coordinator")
        group, spec = groups[0], groups[0].kv_cache_spec
        result.update(spec_type=type(spec).__name__, layer_names=list(group.layer_names),
            is_eagle_group=group.is_eagle_group)
        check(type(spec) is FullAttentionSpec and not group.is_eagle_group,
              "requires plain FullAttentionSpec")
        check(spec.sliding_window is None and spec.attention_chunk_size is None,
              "window/chunk attention unsupported")
        check(manager.watermark_blocks == 0, "nonzero watermark unsupported")
        check(not manager.enable_caching and not engine.vllm_config.cache_config.enable_prefix_caching,
              "prefix sharing unsupported")
        check(not manager.use_eagle and engine.vllm_config.speculative_config is None
              and scheduler.num_spec_tokens == scheduler.num_lookahead_tokens == 0,
              "speculative/lookahead unsupported")
        check(scheduler.dcp_world_size == scheduler.pcp_world_size == 1,
              "context parallel unsupported")
        single = coordinator.single_type_managers
        check(len(single) == 1 and single[0].block_pool is pool,
              "group does not share verified block pool")
        block_size = spec.block_size
        result.update(single_type_block_size=single[0].block_size,
                      scheduler_block_size=coordinator.scheduler_block_size)
        check(type(block_size) is int and block_size == 16
              and single[0].block_size == coordinator.scheduler_block_size == block_size,
              "unverified logical block size")
        total, free = int(pool.num_gpu_blocks), int(pool.get_num_free_blocks())
        usable = total - 1
        result.update(block_size=block_size, total_blocks=total, usable_blocks=usable,
            free_blocks=free, null_block_id=pool.null_block.block_id,
            null_block_is_null=pool.null_block.is_null)
        check(pool.null_block.block_id == 0 and pool.null_block.is_null,
              "unverified null block reservation")
        check(layout.num_blocks == manager.kv_cache_config.num_blocks == total
              and total == 4097 and usable == free == 4096,
              "pool layout mismatch or pool not empty")
        check(not scheduler.requests and scheduler.get_request_counts() == (0, 0)
              and not engine.has_unfinished_requests(), "engine not drained")
        check((config.get("requests"), config.get("prompt_tokens"),
               config.get("output_tokens")) == (128, 3072, 1024),
              "fresh episode bounds differ")
        maximum = engine.vllm_config.scheduler_config.max_num_seqs
        check(maximum == 32 and engine.vllm_config.model_config.max_model_len == 4096,
              "engine bounds differ")
        per_request = (config["prompt_tokens"] + config["output_tokens"]
                       + block_size - 1) // block_size
        safe = min(maximum, usable // per_request)
        check(per_request == 256 and safe == 16 and safe * per_request == 4096,
              "fresh maximum request reservation differs")
        result.update(per_request_reserved_blocks=per_request,
            maximum_request_tokens=config["prompt_tokens"] + config["output_tokens"],
            engine_max_num_seqs=maximum, safe_cap=safe,
            reserved_blocks_at_safe_cap=safe * per_request,
            remaining_blocks_at_full_reservation=usable - safe * per_request)
        result.update(status="QUALIFIED", measurement_status="NOT_YET_RUN",
            bound_semantics="Conservative 3072+1024 eligibility bound, not actual EOS or future physical reservation")
    except (AttributeError, ImportError, IndexError, KeyError, TypeError, ValueError) as error:
        result["error"] = f"{type(error).__name__}: {error}"
    return result


def audit_outputs(raw: dict, workload: dict) -> dict:
    """The sustained sparse final-output audit with the Poisson regime name."""
    rows = raw.get("requests")
    sources = workload["source_requests"]
    arrivals = workload["arrival_traces_s"]["poisson_v1"]
    if (raw.get("status") != "COMPLETE" or raw.get("regime") != "poisson_v1"
            or raw.get("output_mode") != "eos" or raw.get("capture_mode") != "sparse"
            or raw.get("output_events_omitted") is not True
            or raw.get("memory_request_maps_omitted") is not True
            or raw.get("output_events") != []
            or raw.get("target_cap") != 32
            or len(rows or ()) != 128 or len(sources) != 128
            or len(raw.get("scheduler_steps", ())) == 0
            or len(raw.get("engine_steps", ())) == 0):
        raise ValueError("incomplete or non-sparse full-cohort capture")
    if [row["request_id"] for row in rows] != [s["request_id"] for s in sources]:
        raise ValueError("captured request order or identity changed")
    completed = output_tokens = 0
    reasons = {"stop": 0, "length": 0}
    end = raw["observation_end_s"]
    if not isinstance(end, (int, float)) or not math.isfinite(end):
        raise ValueError("invalid observation end")
    for row, source, arrival in zip(rows, sources, arrivals):
        tokens, times = row["output_token_ids"], row["token_times_s"]
        if (row["status"] != "completed" or row["document_id"] != source["document_id"]
                or row["arrival_s"] != arrival
                or row["prompt_token_ids_sha256"] != source["prompt_token_ids_sha256"]
                or row["max_output_tokens"] != 1024
                or len(tokens) != len(times) or len(tokens) > 1024
                or row.get("stop_reason") not in reasons
                or (row["stop_reason"] == "length" and len(tokens) != 1024)
                or row.get("completion_s") is None
                or row["completion_s"] < arrival or row["completion_s"] > end
                or any(not isinstance(t, (int, float)) or not math.isfinite(t)
                       for t in times)
                or any(t < arrival or t > row["completion_s"] for t in times)
                or any(later < earlier for earlier, later in zip(times, times[1:]))):
            raise ValueError("final output, EOS, or host timing contract changed")
        digest = hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()
        if row.get("output_token_ids_sha256") != digest:
            raise ValueError("final output token digest changed")
        completed += 1
        output_tokens += len(tokens)
        reasons[row["stop_reason"]] += 1
    return dict(status="PASS", requests=completed, output_tokens=output_tokens,
        finish_reason_counts=reasons,
        scope="Final IDs, host receipt times, fresh source cohort and natural EOS/length closure; "
              "sparse capture omits cumulative output events and per-request KV maps.")
