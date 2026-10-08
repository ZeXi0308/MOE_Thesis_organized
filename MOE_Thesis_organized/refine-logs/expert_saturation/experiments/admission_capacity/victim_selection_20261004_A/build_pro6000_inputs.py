#!/usr/bin/env python3
"""Compose nested, length-balanced natural-article inputs without using outcomes.

Run with Python's standard library only. Existing inputs are verified, never
retokenized or truncated. Output sets are exploration inputs reusing previously
executed source pools, not unseen documents or statistically independent sets.
"""
import collections
import copy
import hashlib
import json
from pathlib import Path
import statistics


D = Path(__file__).resolve().parent
SOURCE_ROOT = D.parent / "20260929_commit_recheck"
PACKAGE = D / "candidate_pro6000_r01" / "pkg"
OUT = PACKAGE / "inputs"
SOURCES = (
    "candidate_native_residency_victim_r01",
    "candidate_native_oldest_strong_r01",
    "candidate_native_current_guard_fresh_r01",
)
MODEL = {
    "dtype": "bfloat16", "id": "allenai/OLMoE-1B-7B-0924", "key": "olmoe",
    "revision": "6d84c48581ece794365f2b8e9cfb043c68ade9c5",
    "tokenizer_revision": "6d84c48581ece794365f2b8e9cfb043c68ade9c5",
}
ORDER_RULE = (
    "Sort all 384 unique requests by (prompt length, request_id); partition into "
    "64 consecutive rank strata of six. Interleave one item from each stratum "
    "per 64-request phase, visiting strata in six-bit-reversed order. Within "
    "even strata use rank order [1,4,0,5,2,3], within odd strata [4,1,5,0,3,2]. "
    "Thus prefixes 128,256,320 take exactly 2,4,5 requests from every stratum. "
    "Only prompt lengths and stable identities determine selection and order."
)


def sha_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def workload_hash(workload):
    # This exact convention is used by the existing runner's load_inputs().
    return hashlib.sha256(json.dumps(workload, sort_keys=True).encode()).hexdigest()


def token_hash(tokens):
    return hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert path.read_text() == text, f"Refusing to replace different input: {path}"
    else:
        path.write_text(text)


def validate(workload, config=None):
    rows, prompts = workload["source_requests"], workload["actual_prompt_token_ids"]
    assert len(rows) == len(prompts) == len({r["request_id"] for r in rows})
    assert all(len(trace) == len(rows) for trace in workload["arrival_traces_s"].values())
    for row, tokens in zip(rows, prompts):
        assert tokens and all(type(t) is int and 0 <= t < 50304 for t in tokens)
        assert token_hash(tokens) == row["prompt_token_ids_sha256"]
        assert len(tokens) == row["prompt_token_count"]
    if config is not None:
        assert config["model"] == MODEL
        assert workload_hash(workload) == config["workload_sha256"]
        assert config["requests"] == len(rows)
        assert config["min_tokens"] == 0 and config["ignore_eos"] is False
        assert config["output_tokens"] == config["max_output_tokens"] == 1024
        assert config["max_model_len"] == 4096
        assert all(len(tokens) + 1024 <= 4096 for tokens in prompts)


def main():
    records, provenance = [], []
    template = None
    tokenizer_hashes = None
    for name in SOURCES:
        directory = SOURCE_ROOT / name / "pkg" / "inputs"
        config, workload = read(directory / "config.json"), read(directory / "workload.json")
        validate(workload, config)
        assert config["requests"] == 128
        if template is None:
            template = config
            tokenizer_hashes = config["source"]["tokenizer_files_sha256"]
        assert config["source"]["tokenizer_files_sha256"] == tokenizer_hashes
        provenance.append({
            "input_directory": f"../20260929_commit_recheck/{name}/pkg/inputs",
            "config_file_sha256": sha_file(directory / "config.json"),
            "workload_file_sha256": sha_file(directory / "workload.json"),
            "workload_sha256": config["workload_sha256"],
            "source": config["source"],
        })
        for row, tokens in zip(workload["source_requests"], workload["actual_prompt_token_ids"]):
            assert row["original_document_token_count"] == len(tokens)
            assert row["max_output_tokens"] == 1024
            assert hashlib.sha256(row["prompt"].encode()).hexdigest() == row["prompt_sha256"]
            assert row["prompt_sha256"] == row["document_sha256"]
            records.append((copy.deepcopy(row), list(tokens), name))
    assert len(records) == 384
    for key in ("request_id", "document_id", "document_sha256", "prompt_token_ids_sha256"):
        assert len({r[0][key] for r in records}) == 384, f"Duplicate {key}"
    records.sort(key=lambda r: (len(r[1]), r[0]["request_id"]))
    strata = [records[i * 6:(i + 1) * 6] for i in range(64)]
    visit = [int(f"{i:06b}"[::-1], 2) for i in range(64)]
    ordered = []
    for phase in range(6):
        for stratum in visit:
            ranks = [1, 4, 0, 5, 2, 3] if stratum % 2 == 0 else [4, 1, 5, 0, 3, 2]
            row, tokens, source = strata[stratum][ranks[phase]]
            row.update(source_pool=source, source_pool_index=row["source_index"],
                       source_index=len(ordered), prompt_length_stratum=stratum)
            ordered.append((row, tokens, source))

    summaries = []
    for case, count in (("low", 128), ("knee", 256), ("high", 320)):
        selected = ordered[:count]
        rows, prompts = [r[0] for r in selected], [r[1] for r in selected]
        lengths = list(map(len, prompts))
        assert set(collections.Counter(r["prompt_length_stratum"] for r in rows).values()) == {count // 64}
        workload = {
            "schema": "olmoe-natural-cadence-holdout-v1", "source_requests": rows,
            "actual_prompt_token_ids": prompts,
            "arrival_traces_s": {"steady": [i / 100 for i in range(count)]},
            "arrival_rule": "External arrival at i*0.01 seconds in deterministic length-stratified order.",
            "output_contract": {"max_output_tokens": 1024, "ignore_eos": False,
                                "min_tokens": 0, "actual_output_lengths": "UNKNOWN_UNTIL_EXECUTION"},
        }
        config = copy.deepcopy(template)
        for old in ("exclusion_inventory_sha256", "output_tokens_by_request"):
            config.pop(old, None)
        config.update(
            status="PRO6000_NORMAL_CAPACITY_EXPLORATION_CPU_PREPARED_GPU_UNRUN",
            requests=count, prompt_tokens=max(lengths), prompt_tokens_by_request=lengths,
            arrival_gap_s=0.01, arrival_process="deterministic_length_stratified_interleaving",
            arrival_span_s=(count - 1) / 100, workload_sha256=workload_hash(workload),
            cap=384, engine_max_num_seqs=384, max_seconds=600, max_num_batched_tokens=1024,
            gpu_memory_utilization=0.9, fixed_kv_cache_memory_bytes=None,
            target_usable_kv_blocks=None, intended_usable_kv_bytes=None,
            resource_state="Normal vLLM 0.9 memory profiling requested; actual KV pages unknown until GPU initialization.",
            input_preparation=ORDER_RULE,
            prompt_tokens_semantics=f"Full natural articles, {min(lengths)}..{max(lengths)} tokens; no padding or truncation.",
            input_independence="Previously used source pools; not an unseen holdout. Cases are nested prefixes, not independent repetitions.",
            source={"dataset_id": "wikitext", "dataset_config": "wikitext-103-raw-v1",
                    "split": "train", "dataset_revision": template["source"]["dataset_revision"],
                    "tokenizer_files_sha256": tokenizer_hashes, "model_max_position_embeddings": 4096,
                    "selection": ORDER_RULE, "pools": provenance},
        )
        stats = {
            "case": case, "requests": count, "unique_request_ids": count,
            "unique_document_ids": count, "prompt_tokens_total": sum(lengths),
            "prompt_tokens_min": min(lengths), "prompt_tokens_mean": statistics.mean(lengths),
            "prompt_tokens_median": statistics.median(lengths), "prompt_tokens_max": max(lengths),
            "prompt_tokens_quartiles": statistics.quantiles(lengths, n=4, method="inclusive"),
            "source_request_counts": dict(sorted(collections.Counter(r[2] for r in selected).items())),
            "requests_per_length_stratum": count // 64, "length_strata": 64,
            "prompt_only_kv_gib_page_rounded": sum((n + 15) // 16 for n in lengths) / 512,
            "maximum_potential_kv_gib_page_rounded": sum((n + 1024 + 15) // 16 for n in lengths) / 512,
            "kv_bound_semantics": "All requests simultaneously resident through maximum output, 16-token pages at 2 MiB/page; upper bound, not measured occupancy. Excludes one null page.",
            "arrival_span_s": (count - 1) / 100, "workload_sha256": workload_hash(workload),
            "no_outcome_information_used": True,
        }
        validate(workload, config)
        directory = OUT / f"pro_{case}"
        write(directory / "workload.json", workload)
        write(directory / "config.json", config)
        write(directory / "stats.json", stats)
        summaries.append(stats)

    warm_path = PACKAGE / "warmups" / "short" / "workload.json"
    warm_source = read(warm_path)
    validate(warm_source)
    assert len(warm_source["source_requests"]) == 32
    assert set(map(len, warm_source["actual_prompt_token_ids"])) == {128}
    warm_rows, warm_tokens = [], []
    for i in range(384):
        source = warm_source["source_requests"][i % 32]
        row = {k: copy.deepcopy(v) for k, v in source.items() if k != "prompt"}
        row.update(request_id=f"warmup-short384-{i:04d}", max_output_tokens=16,
                   source_request_id=source["request_id"], source_index=i,
                   warmup_repeat_index=i // 32)
        warm_rows.append(row)
        warm_tokens.append(list(warm_source["actual_prompt_token_ids"][i % 32]))
    warm = {
        "schema": "olmoe-admission-inputs-v1", "source_requests": warm_rows,
        "actual_prompt_token_ids": warm_tokens, "arrival_traces_s": {"steady": [0.0] * 384},
        "arrival_rule": "All warmup requests arrive at zero; these are not measurement requests.",
        "output_contract": {"max_output_tokens": 16, "ignore_eos": True, "min_tokens": 16},
        "provenance": {"source": "warmups/short/workload.json", "source_sha256": sha_file(warm_path),
                       "model": MODEL, "unique_requests": 384, "unique_documents": 32,
                       "semantics": "Repeat the existing 32 warmup token prefixes twelve times with unique request IDs. Repeated prompts are allowed for shape warmup; they are not 384 independent documents. Runner resets GPU prefix cache and drains/reset host offload state before measurement. Full source prompt text omitted; actual 128 token prefixes preserved."},
    }
    validate(warm)
    assert not ({r["document_id"] for r in warm_rows} & {r[0]["document_id"] for r in records})
    write(OUT / "warmup_short.json", warm)
    print(json.dumps({"cases": summaries, "warmup_requests": 384, "warmup_unique_documents": 32}, indent=2))


if __name__ == "__main__":
    main()
