#!/usr/bin/env python3
"""Prepare five fixed 128-request mixes from an existing 256-request pool.

CPU/stdlib only. No tokenizer/model loading, generation, download, truncation,
replacement, arrival-rate scan or server launch. The source 256-request files
remain unchanged. Long-request fraction means the LongBench SOURCE CLASS,
not a strict length split: the ShareGPT pool can also contain long prompts.
It is NOT a GPU prefill-time fraction. Output-token shares require measured
generation outcomes, including failures/unfinished requests.

  python code/prepare_serving_mixes.py --source-dir datasets/serving_olmoe_r01 \
    --output datasets/serving_olmoe_mixes_r01 --seed 20261016
  python code/prepare_serving_mixes.py --source-dir datasets/serving_qwen3_r01 \
    --output datasets/serving_qwen3_mixes_r01 --seed 20261016

Each long000/025/050/075/100 mix has a natural and forced128 JSONL. The two
modes preserve exactly the same source request IDs, prompts and order. Within
each class, seeded hash ranks define nested sampling without replacement;
a separate seeded global rank preserves relative order across mixtures.
The same source pools are deliberately reused across mixture conditions.

For run_serving_group.py, put ONE desired trace in an existing group's runs:
  {"id":"long050_natural_r1", "trace":"../datasets/serving_olmoe_mixes_r01/long050_natural.jsonl",
   "request_rate":1, "repeat_id":1, "seed":20261016}
The trace path is relative to that group config, and request_rate=1 is only an
example. Use the identical trace/schedule seed for every arm at a chosen rate.
Inspect with run_serving_group.py --config GROUP.json --output RUN/group
--lock-status RUN/status.json; --run remains exclusively under run_locked.py.
No group configs or experimental arrival-rate choices are generated here.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
CLASS_SIZE = 128
REQUESTS = 128
LONG_COUNTS = (0, 32, 64, 96, 128)
CLASSES = ("sharegpt", "longbench")
NATURAL = {"temperature": 0.0, "ignore_eos": False, "min_tokens": 0}
FORCED = {"temperature": 0.0, "ignore_eos": True, "min_tokens": 128}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def save_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path, rows):
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(canonical(row) + "\n")
    return {"bytes": path.stat().st_size, "sha256": sha256(path)}


def read_fixed_pool(directory):
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("state") != "complete" or manifest.get("selected_total") != 256 or
            manifest.get("per_class") != CLASS_SIZE or manifest.get("max_tokens") != 128 or
            manifest.get("truncate") is not False):
        raise ValueError("Expected a completed untruncated 256-request source with 128 per class and cap=128")
    inputs = {}
    # These are existing small prepared assets, not upstream multi-GB data or weights.
    required = ("serving_natural.jsonl", "serving_fixed_workload.jsonl", "tokenizer/config.json")
    for name in required:
        path = directory / name
        expected = manifest["output_files"][name]
        actual = {"bytes": path.stat().st_size, "sha256": sha256(path)}
        if actual != expected:
            raise ValueError(f"Prepared input differs from the fixed source manifest: {name}")
        inputs[name] = actual
    model_config = json.loads((directory / "tokenizer/config.json").read_text())
    if model_config.get("model_type") not in ("olmoe", "qwen3_moe"):
        raise ValueError("Expected the saved OLMoE or Qwen3-MoE tokenizer config")
    natural = read_jsonl(directory / "serving_natural.jsonl")
    forced = read_jsonl(directory / "serving_fixed_workload.jsonl")
    if len(natural) != 256 or len(forced) != 256:
        raise ValueError("Both fixed source modes must contain all 256 requests")
    pools, by_id, forced_by_id, seen_prompts = {name: [] for name in CLASSES}, {}, {}, set()
    for index, (row, replay) in enumerate(zip(natural, forced)):
        rid = row.get("request_id")
        if not isinstance(rid, str) or not rid or rid in by_id:
            raise ValueError("Source request IDs must be nonempty and unique")
        if (row.get("source") not in CLASSES or not isinstance(row.get("prompt"), str) or not row["prompt"] or
                row.get("max_tokens") != 128 or row.get("sampling") != NATURAL or replay.get("sampling") != FORCED or
                "arrival_s" in row or "arrival_s" in replay):
            raise ValueError(f"Unexpected fixed source request/sampling contract: {rid}")
        if {k: v for k, v in row.items() if k != "sampling"} != {k: v for k, v in replay.items() if k != "sampling"}:
            raise ValueError("Source natural and forced128 modes differ in prompt/order/metadata")
        prompt_hash = hashlib.sha256(row["prompt"].encode()).hexdigest()
        count = row.get("prompt_tokens")
        if (prompt_hash != row.get("prompt_sha256") or prompt_hash in seen_prompts or
                type(count) is not int or count <= 0 or count + 128 > manifest["context_limit"]):
            raise ValueError(f"Invalid duplicate, prompt hash, token count or context budget: {rid}")
        if row["source"] == "longbench" and count < manifest["long_min_tokens"]:
            raise ValueError("LongBench source pool violates its frozen minimum prompt length")
        seen_prompts.add(prompt_hash)
        pools[row["source"]].append(rid)
        by_id[rid] = {"row": row, "source_trace_row": index}
        forced_by_id[rid] = replay
    if any(len(pool) != CLASS_SIZE for pool in pools.values()):
        raise ValueError("Expected exactly 128 prepared requests in each source class")
    return manifest, inputs, model_config["model_type"], pools, by_id, forced_by_id


def ranked_ids(ids, seed, role):
    return sorted(ids, key=lambda rid: (digest({"seed": seed, "role": role, "request_id": rid}), rid))


def consumer_check(natural_path, forced_path, rows, seed):
    # Both modules have CPU/stdlib-safe imports; neither model/GPU entrypoint is called.
    from bench_serving import load_trace
    from capture_model_routes import read_prompts
    natural = load_trace(natural_path, rate=1.0, seed=seed)
    forced = load_trace(forced_path, rate=1.0, seed=seed)
    expected = [(row["request_id"], row["prompt"]) for row in rows]
    if (len(natural) != REQUESTS or len(forced) != REQUESTS or
            [(row["request_id"], row["prompt"]) for row in natural] != expected or
            [(row["request_id"], row["prompt"]) for row in forced] != expected or
            [row["arrival_s"] for row in natural] != [row["arrival_s"] for row in forced] or
            any(row["sampling"] != NATURAL for row in natural) or any(row["sampling"] != FORCED for row in forced)):
        raise ValueError("Serving consumer changed request population/order/sampling/schedule")
    selected, available = read_prompts(natural_path, limit=32)
    if available != REQUESTS or [(row["request_id"], row["prompt"]) for row in selected] != expected[:32]:
        raise ValueError("Natural route consumer cannot read the unmodified prompt prefix")
    return {"bench_serving": {"rows_per_mode": REQUESTS, "same_prompt_order": True,
        "same_poisson_schedule_for_same_seed_and_rate": True, "validation_request_rate_only": 1.0,
        "natural_and_forced_sampling_preserved": True},
        "capture_model_routes": {"available_prompts": available, "validated_natural_prefix_rows": len(selected),
            "scope": "read_prompts only, matching its CLI's maximum 32; not a 128-request capture or forced128 semantics check"},
        "no_generation_performed": True, "torch_imported": "torch" in sys.modules}


def prepare(source, output, seed):
    source, output = source.resolve(), output.resolve()
    if output == source or output.is_relative_to(source):
        raise ValueError("Mix outputs must be separate from the frozen source dataset")
    manifest, inputs, model_type, pools, by_id, forced_by_id = read_fixed_pool(source)
    ranked = {name: ranked_ids(pools[name], seed, "class_selection:" + name) for name in CLASSES}
    all_order = ranked_ids(by_id, seed, "global_prompt_order")
    order_index = {rid: index for index, rid in enumerate(all_order)}
    threshold = manifest["long_min_tokens"]
    state = {"schema_version": 1, "state": "preparing", "stage": "prepare_serving_mixes",
        "created_utc": datetime.now(timezone.utc).isoformat(), "script_sha256": sha256(__file__),
        "source_directory": str(source), "source_manifest_sha256": sha256(source / "manifest.json"),
        "source_files_verified": inputs, "source_model_type": model_type,
        "source_template": manifest["template"], "source_tokenizer_files": {k: v for k, v in manifest["output_files"].items() if k.startswith("tokenizer/")},
        "source_dataset_identities": manifest["sources"], "source_context_limit": manifest["context_limit"],
        "source_256_request_main_workload_preserved": True, "seed": seed, "requests_per_mix": REQUESTS,
        "long_request_fractions": [n / REQUESTS for n in LONG_COUNTS], "long_class": "longbench", "other_class": "sharegpt",
        "long_request_fraction_definition": "count(source == longbench) / 128; ShareGPT can overlap the long prompt-length range",
        "not_gpu_prefill_time_fraction": True,
        "output_token_fraction": None, "output_token_fraction_status": "must be calculated from actual generation/termination outcomes; do not infer from prompt mix or cap",
        "selection": "seeded SHA-256 rank per class, take prefix without replacement; seeded global rank orders each selected union",
        "rank_digest_encoding": "UTF-8 JSON with sort_keys=true,separators=(',',':'),ensure_ascii=false; keys seed/role/request_id",
        "cross_mix_reuse": "nested source-class subsets; same fixed pool reused across conditions, not independent workload repetitions",
        "prompt_token_counts_source": "hash-verified prepared trace counts; no retokenization or prompt modification by this tool",
        "arrival_schedule": "no arrival_s stored; consumer generates Poisson arrivals from the same trace order/rate/seed for each arm",
        "generation": {"max_tokens": 128, "natural": NATURAL, "forced128": FORCED,
            "forced128_scope": "forced-length generated workload; not recorded-token replay or quality-equivalent output"},
        "length_overlap": {"long_min_tokens": threshold, "sharegpt_at_or_above_long_min": sum(by_id[r]["row"]["prompt_tokens"] >= threshold for r in pools["sharegpt"]),
            "sharegpt_prompt_tokens_max": max(by_id[r]["row"]["prompt_tokens"] for r in pools["sharegpt"]),
            "longbench_prompt_tokens_min": min(by_id[r]["row"]["prompt_tokens"] for r in pools["longbench"])},
        "pool_request_ids_in_selection_order": ranked, "mixes": [], "output_files": {},
        "cpu_only": True, "model_loaded": False, "no_generation_performed": True, "truncate": False,
        "sampling_replacement": False, "consumer_source_sha256": {name: sha256(Path(__file__).parent / name)
            for name in ("bench_serving.py", "capture_model_routes.py")}}
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "manifest.json", state)
    try:
        for long_count in LONG_COUNTS:
            mix_id = f"long{100 * long_count // REQUESTS:03d}"
            ids = sorted(ranked["sharegpt"][:REQUESTS - long_count] + ranked["longbench"][:long_count], key=order_index.__getitem__)
            if len(ids) != REQUESTS or len(set(ids)) != REQUESTS:
                raise ValueError("Mixture violated exact size or sampling without replacement")
            rows = [by_id[rid]["row"] for rid in ids]
            replay = [forced_by_id[rid] for rid in ids]
            counts = Counter(row["source"] for row in rows)
            if counts["longbench"] != long_count or counts["sharegpt"] != REQUESTS - long_count:
                raise ValueError("Selected class fraction differs from configured fraction")
            total_tokens = sum(row["prompt_tokens"] for row in rows)
            long_tokens = sum(row["prompt_tokens"] for row in rows if row["source"] == "longbench")
            paths = {"natural": output / (mix_id + "_natural.jsonl"), "forced128": output / (mix_id + "_forced128.jsonl")}
            for mode, selected in (("natural", rows), ("forced128", replay)):
                state["output_files"][paths[mode].name] = write_jsonl(paths[mode], selected)
            row = {"mix_id": mix_id, "request_count": REQUESTS, "configured_long_request_fraction": long_count / REQUESTS,
                "observed_long_class_requests": long_count, "observed_other_class_requests": REQUESTS - long_count,
                "source_request_ids_in_prompt_order": ids,
                "prompt_order_sha256": digest([{k: r[k] for k in ("request_id", "prompt_sha256", "prompt_tokens")} for r in rows]),
                "total_prompt_tokens": total_tokens, "prompt_tokens_by_source_class": {name: sum(r["prompt_tokens"] for r in rows if r["source"] == name) for name in CLASSES},
                "long_class_prompt_token_fraction": long_tokens / total_tokens,
                "requests_at_or_above_long_min_tokens": sum(r["prompt_tokens"] >= threshold for r in rows),
                "sharegpt_requests_at_or_above_long_min_tokens": sum(r["source"] == "sharegpt" and r["prompt_tokens"] >= threshold for r in rows),
                "prompt_tokens_min": min(r["prompt_tokens"] for r in rows), "prompt_tokens_max": max(r["prompt_tokens"] for r in rows),
                "selected_by_task": dict(Counter(r["task"] for r in rows)),
                "requests": [{k: r[k] for k in ("request_id", "source", "task", "source_id", "source_row", "prompt_sha256", "prompt_tokens")}
                             | {"source_trace_row": by_id[r["request_id"]]["source_trace_row"]} for r in rows],
                "traces": {mode: {"path": path.name, **state["output_files"][path.name]} for mode, path in paths.items()},
                "consumer_validation": consumer_check(paths["natural"], paths["forced128"], rows, seed),
                "gpu_prefill_time_fraction": None, "actual_output_token_fraction": None}
            state["mixes"].append(row)
            save_json(output / "manifest.json", state)
        # Re-check both original main traces: generating mixes must not alter them.
        if any(sha256(source / name) != spec["sha256"] for name, spec in inputs.items()):
            raise RuntimeError("Frozen source inputs changed during preparation")
        if "torch" in sys.modules:
            raise RuntimeError("CPU-only preparation unexpectedly imported torch")
        state.update(state="complete", trace_count=10, source_files_unchanged_verified=True, torch_imported=False)
    except BaseException as error:
        state.update(state="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        state["ended_utc"] = datetime.now(timezone.utc).isoformat()
        save_json(output / "manifest.json", state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261016)
    args = parser.parse_args()
    result = prepare(args.source_dir, args.output, args.seed)
    print(json.dumps({"state": result["state"], "model": result["source_model_type"], "output": str(args.output.resolve()),
        "trace_count": result["trace_count"], "seed": result["seed"], "length_overlap": result["length_overlap"],
        "mixes": [{key: row[key] for key in ("mix_id", "configured_long_request_fraction", "total_prompt_tokens", "long_class_prompt_token_fraction")} for row in result["mixes"]]}, indent=2))


if __name__ == "__main__":
    main()
