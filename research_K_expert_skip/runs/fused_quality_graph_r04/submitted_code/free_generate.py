#!/usr/bin/env python3
"""Natural-termination paired generation, not a throughput benchmark.

Default target: 200 distinct prompts for every arm. EOS is enabled; no forced
tokens, minimum output length, or repetition penalty. Cohorts drain naturally,
so --batch is an admission ceiling, not a constant decode batch size. Use the
main session's external GPU lock. --self-test is dependency-free and CPU-only.

vLLM 0.26 interfaces: https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/v1/engine/llm_engine.py
and https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/outputs.py
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import time
import traceback

from quality_decode import atomic_json, load_arms_file

ROOT = Path(__file__).resolve().parents[1]


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def append_json(path, value):
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def resolve_arms(path):
    from expert_skip import SkipConfig
    arms, provenance = load_arms_file(Path(path))
    resolved = []
    for name, spec in arms:
        if spec is not None and spec.get("mode") == "global_native":
            if set(spec) - {"mode", "drop_j", "count_routes", "backend"}:
                raise ValueError("Unknown native reduced-k fields")
            if type(spec.get("drop_j")) is not int or spec["drop_j"] < 0:
                raise ValueError("global_native needs a nonnegative integer drop_j")
            if not isinstance(spec.get("count_routes", True), bool):
                raise ValueError("count_routes must be boolean")
            SkipConfig(backend=spec.get("backend", "torch"))
        elif spec is not None:
            spec = asdict(SkipConfig(**spec))
        resolved.append((name, spec))
    return resolved, provenance


def offline_model(model):
    """Require an existing local artifact and forbid model/network downloads."""
    path = Path(model).expanduser().resolve()
    if not path.is_dir() or not (path / "config.json").is_file():
        raise ValueError("--model must be an existing local model directory with config.json")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    return str(path)


def model_kwargs(args):
    return dict(model=offline_model(args.model), dtype=args.dtype,
                quantization=args.quantization, trust_remote_code=args.trust_remote_code,
                tensor_parallel_size=1, enforce_eager=True,
                compilation_config={"mode": 0, "cudagraph_mode": "NONE"},
                worker_cls="skip_worker.ExpertSkipWorker", moe_backend="triton",
                enable_prefix_caching=False, enable_chunked_prefill=False,
                async_scheduling=False, max_model_len=args.max_model_len,
                max_num_seqs=args.batch,
                max_num_batched_tokens=args.batch * args.max_model_len,
                gpu_memory_utilization=args.gpu_memory_utilization,
                seed=args.seed, disable_log_stats=True)


def check_versions(require_lm_eval=False):
    versions = {"vllm": importlib.metadata.version("vllm")}
    if versions["vllm"].split("+")[0] != "0.26.0":
        raise RuntimeError("This entry point requires inspected vLLM 0.26.0")
    if require_lm_eval:
        versions["lm_eval"] = importlib.metadata.version("lm_eval")
        if versions["lm_eval"] != "0.4.13":
            raise RuntimeError("This entry point requires inspected lm_eval 0.4.13")
    return versions


def one_worker(llm, method, *args):
    result = llm.collective_rpc(method, args=args)
    if len(result) != 1:
        raise RuntimeError("Expected one GPU worker")
    return result[0]


def finish_category(finish_reason, stop_reason, token_ids, eos_ids):
    if finish_reason == "length":
        return "limit"
    if finish_reason == "stop":
        if stop_reason is None or stop_reason in eos_ids or (token_ids and token_ids[-1] in eos_ids):
            return "eos"
        return "stop"
    return "fail"


def repetition_metrics(token_ids):
    def ratio(n):
        grams = [tuple(token_ids[i:i+n]) for i in range(max(0, len(token_ids)-n+1))]
        return 1 - len(set(grams))/len(grams) if grams else None
    longest = current = 0
    previous = None
    for token in token_ids:
        current = current + 1 if token == previous else 1
        longest = max(longest, current)
        previous = token
    suffix = {"block_tokens": None, "repeats": 1, "covered_tokens": 0}
    for size in range(1, min(64, len(token_ids)//2) + 1):
        block = token_ids[-size:]
        count = 1
        while (count+1)*size <= len(token_ids) and token_ids[-(count+1)*size:-count*size] == block:
            count += 1
        if count >= 2 and count*size > suffix["covered_tokens"]:
            suffix = {"block_tokens": size, "repeats": count, "covered_tokens": count*size}
    r4 = ratio(4)
    return {"repeated_4gram_ratio": r4, "repeated_8gram_ratio": ratio(8),
            "longest_identical_token_run": longest, "repeated_suffix": suffix,
            "review_flag": len(token_ids) >= 32 and r4 is not None and r4 > .5}


def read_prompts(path, limit=200):
    rows, seen = [], set()
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            source = json.loads(line)
            source = {"prompt": source} if isinstance(source, str) else source
            prompt = source.get("prompt", source.get("text"))
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"Line {line_number} needs a nonempty prompt/text string")
            key = hashlib.sha256(prompt.encode()).hexdigest()
            if key in seen:
                raise ValueError(f"Duplicate prompt at line {line_number}; supply distinct records")
            seen.add(key)
            rows.append({**source, "id": source.get("id", f"line{line_number}"),
                         "prompt": prompt, "prompt_sha256": key, "source_line": line_number})
            if limit is not None and len(rows) == limit:
                break
    if not rows or (limit is not None and len(rows) != limit):
        raise ValueError(f"Requested {limit} prompts, found {len(rows)}; use an explicit smaller --limit for exploration")
    if len({str(row["id"]) for row in rows}) != len(rows):
        raise ValueError("Prompt IDs must be unique")
    return rows


def tokenize_prompts(rows, tokenizer, max_model_len, max_tokens, prompt_tokens=None, require_distinct=False):
    result, seen = [], set()
    for row in rows:
        ids = list(tokenizer.encode(row["prompt"], add_special_tokens=True))
        original_count = len(ids)
        if prompt_tokens is not None:
            ids = ids[:prompt_tokens]
        if not ids or len(ids) + max_tokens > max_model_len:
            raise ValueError(f"Prompt {row['id']} needs {len(ids)} + {max_tokens} tokens; no silent truncation")
        token_hash = digest(ids)
        if require_distinct and token_hash in seen:
            raise ValueError(f"Duplicate actual tokenized prompt at {row['id']}; select distinct prefixes")
        seen.add(token_hash)
        result.append({**row, "prompt_token_ids": ids, "prompt_tokens": len(ids),
                       "source_token_count": original_count,
                       "explicit_prefix_selection": prompt_tokens,
                       "prompt_token_sha256": token_hash})
    return result


def natural_cohort(engine, rows, sampling_factory, prefix, eos_ids):
    """Drain a fixed admission cohort; preserve partial rows if execution fails."""
    if engine.has_unfinished_requests():
        raise RuntimeError("Engine has outstanding requests before cohort")
    states, steps = {}, []
    error = None
    try:
        for index, row in enumerate(rows):
            rid = f"{prefix}-{index}"
            states[rid] = {"id": row["id"], "request_id": rid, "prompt_sha256": row["prompt_sha256"],
                           "prompt_token_sha256": row["prompt_token_sha256"],
                           "prompt_tokens": row["prompt_tokens"], "text": "", "token_ids": [],
                           "finished": False, "finish_reason": None, "stop_reason": None}
            engine.add_request(rid, {"prompt_token_ids": row["prompt_token_ids"]}, sampling_factory(index))
        while engine.has_unfinished_requests():
            active = sum(not s["finished"] for s in states.values())
            outputs = engine.step()
            emitted = 0
            for output in outputs:
                state = states[output.request_id]
                if len(output.outputs) != 1:
                    raise RuntimeError("Expected one completion per request")
                comp = output.outputs[0]
                emitted += len(comp.token_ids) - len(state["token_ids"])
                state.update(text=comp.text, token_ids=list(comp.token_ids), finished=output.finished,
                             finish_reason=comp.finish_reason, stop_reason=comp.stop_reason)
            steps.append({"step": len(steps), "active_requests_before": active,
                          "emitted_tokens": emitted, "requests_returned": len(outputs)})
    except Exception:
        error = traceback.format_exc()
        for rid, state in states.items():
            if not state["finished"]:
                try:
                    engine.abort_request(rid)
                except Exception:
                    pass
    result = []
    for row in rows:
        state = next((s for s in states.values() if s["id"] == row["id"]), None)
        state = state or {"id": row["id"], "prompt_sha256": row["prompt_sha256"],
                          "token_ids": [], "text": "", "finished": False,
                          "finish_reason": None, "stop_reason": None}
        ids = state["token_ids"]
        category = finish_category(state["finish_reason"], state["stop_reason"], ids, eos_ids)
        result.append({**state, "termination": category, "output_tokens": len(ids),
                       "decode_predictions_emitted": max(0, len(ids)-1),
                       "first_prediction_only": len(ids) <= 1,
                       "repetition": repetition_metrics(ids),
                       "error": error if category == "fail" else None})
    return result, steps, error


def summarize(rows, planned):
    completed = [r for r in rows if r["termination"] != "fail"]
    lengths = [r["output_tokens"] for r in completed]
    metrics = [r["repetition"]["repeated_4gram_ratio"] for r in completed
               if r["repetition"]["repeated_4gram_ratio"] is not None]
    return {"planned": planned, "attempted": len(rows), "not_run": planned-len(rows),
            "termination_counts": dict(Counter(r["termination"] for r in rows)),
            "output_tokens_total_including_partial": sum(r["output_tokens"] for r in rows),
            "completed_output_tokens_mean": statistics.mean(lengths) if lengths else None,
            "completed_output_tokens_median": statistics.median(lengths) if lengths else None,
            "completed_output_tokens_min": min(lengths) if lengths else None,
            "completed_output_tokens_max": max(lengths) if lengths else None,
            "first_prediction_only_count": sum(r["first_prediction_only"] for r in completed),
            "mean_repeated_4gram_ratio": statistics.mean(metrics) if metrics else None,
            "review_flag_count": sum(r["repetition"]["review_flag"] for r in completed),
            "repetition_scope": "descriptive token repetition; review_flag means >=32 tokens and repeated-4gram ratio >0.5, not a correctness judgment"}


def run_pairs(llm, rows, arms, args, outdir, scorer=None):
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind
    tokenizer = llm.get_tokenizer()
    eos = getattr(tokenizer, "eos_token_id", None)
    eos_ids = set(eos if isinstance(eos, list) else [eos]) - {None}
    results = {name: [] for name, _ in arms}
    manifest = {"status": "RUNNING", "arms": {}, "cohorts": [], "planned_per_arm": len(rows),
                "scope": "natural free generation; fixed admission cohorts, variable actual decode batch; no timing or equal-work claim"}
    atomic_json(outdir / "summary.json", manifest)
    try:
        for group, start in enumerate(range(0, len(rows), args.batch)):
            cohort = rows[start:start+args.batch]
            baseline = None
            for name, spec in arms:
                installed = one_worker(llm, "skip_set_policy", spec, "decode", 0, False)
                def sampling(index):
                    return SamplingParams(temperature=args.temperature, top_p=args.top_p,
                        max_tokens=args.max_tokens, min_tokens=0, ignore_eos=False,
                        seed=args.seed + start + index, repetition_penalty=1.,
                        output_kind=RequestOutputKind.CUMULATIVE, detokenize=True)
                records, steps, error = natural_cohort(llm.llm_engine, cohort, sampling,
                                                       f"g{group}-{name}", eos_ids)
                try:
                    snapshot = one_worker(llm, "skip_snapshot")
                except Exception:
                    snapshot = {"snapshot_error": traceback.format_exc()}
                if snapshot.get("snapshot_error") and error is None:
                    error = snapshot["snapshot_error"]
                if snapshot.get("phase_steps", {}).get("mixed", 0) and error is None:
                    error = "Unexpected mixed scheduling; the fixed admission cohort must finish prefill before decode"
                tag = f"g{group:05d}.{name}"
                atomic_json(outdir / f"{tag}.worker.json", {"installation": installed, "snapshot": snapshot,
                                                             "steps": steps, "error": error})
                for record, source in zip(records, cohort):
                    record.update(arm=name, cohort=group, cohort_admitted=len(cohort))
                    if scorer:
                        record.update(scorer(source, record))
                    if baseline is not None:
                        before = baseline[str(record["id"])]
                        record["same_tokens_as_native"] = record["token_ids"] == before["token_ids"]
                        record["output_token_delta_from_native"] = record["output_tokens"] - before["output_tokens"]
                    append_json(outdir / "outputs.jsonl", record)
                results[name].extend(records)
                if name == "native":
                    baseline = {str(r["id"]): r for r in records}
                manifest["cohorts"].append({"cohort": group, "arm": name, "worker_file": f"{tag}.worker.json",
                                            "phase_steps": snapshot.get("phase_steps"),
                                            "decode_exercised": snapshot.get("phase_steps", {}).get("decode", 0) > 0,
                                            "policy_counters": snapshot.get("layers", {}).get("totals"),
                                            "error": error})
                manifest["arms"] = {n: summarize(rs, len(rows)) for n, rs in results.items()}
                atomic_json(outdir / "summary.json", manifest)
                if error:
                    raise RuntimeError(f"Generation failed in {tag}; partial outputs retained")
                if spec is not None and spec.get("mode") == "dense" and any(not r["same_tokens_as_native"] for r in records):
                    raise RuntimeError("Dense identity arm changed generated tokens")
        manifest["status"] = "COMPLETE"
    except Exception:
        manifest["status"] = "ERROR"
        manifest["error"] = traceback.format_exc()
        raise
    finally:
        atomic_json(outdir / "summary.json", manifest)
    return results, manifest


def add_runtime_args(parser):
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arms-file", type=Path, default=ROOT / "configs/quality_eval_arms.json")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--quantization")
    parser.add_argument("--gpu-memory-utilization", type=float, default=.5)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--trust-remote-code", action="store_true")


def validate_runtime(args):
    if min(args.batch, args.max_tokens, args.max_model_len) <= 0 or args.max_tokens >= args.max_model_len:
        raise ValueError("Invalid batch/context/generation bounds")
    if not 0 < args.gpu_memory_utilization < 1:
        raise ValueError("gpu-memory-utilization must be between 0 and 1")


def self_test():
    assert repetition_metrics([1, 2, 3, 4]*10)["review_flag"]
    assert repetition_metrics(list(range(40)))["repeated_4gram_ratio"] == 0
    assert repetition_metrics([])["repeated_4gram_ratio"] is None
    assert finish_category("length", None, [1], {2}) == "limit"
    assert finish_category("stop", None, [1, 2], {2}) == "eos"
    assert finish_category("stop", "Question:", [1], {2}) == "stop"
    assert finish_category(None, None, [], {2}) == "fail"
    from types import SimpleNamespace as N
    class Engine:
        def __init__(self): self.ids = []; self.i = 0
        def has_unfinished_requests(self): return bool(self.ids) and self.i < 3
        def add_request(self, rid, prompt, params): self.ids.append(rid)
        def step(self):
            self.i += 1
            return [N(request_id=rid, finished=True, outputs=[N(token_ids=[1]*self.i,
                     text="x"*self.i, finish_reason="stop", stop_reason=None)])
                    for j, rid in enumerate(self.ids) if self.i == (1 if j == 0 else 3)]
    rows = [{"id": str(i), "prompt_sha256": str(i), "prompt_token_sha256": str(i),
             "prompt_tokens": 2, "prompt_token_ids": [1, 2]} for i in range(2)]
    records, steps, error = natural_cohort(Engine(), rows, lambda i: None, "cpu", {2})
    assert error is None and [r["output_tokens"] for r in records] == [1, 3]
    assert [s["active_requests_before"] for s in steps] == [2, 1, 1]
    assert summarize(records, 3)["not_run"] == 1
    class FailedEngine(Engine):
        def step(self):
            if self.i:
                raise RuntimeError("CPU injected execution failure")
            return super().step()
        def abort_request(self, rid): pass
    partial, _, error = natural_cohort(FailedEngine(), rows, lambda i: None, "cpu-fail", {2})
    assert error and [r["termination"] for r in partial] == ["eos", "fail"]
    print("CPU natural generation bookkeeping, shrinking cohort, termination and repetition checks passed")


def main():
    import sys
    if sys.argv[1:] == ["--self-test"]:
        self_test()
        return
    parser = argparse.ArgumentParser(description=__doc__)
    add_runtime_args(parser)
    parser.add_argument("--prompts", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=200, help="Target 200; smaller explicit values are exploratory")
    parser.add_argument("--prompt-tokens", type=int, help="Explicitly select this many initial tokens from each source text; otherwise preserve the complete prompt")
    parser.add_argument("--temperature", type=float, default=0.)
    parser.add_argument("--top-p", type=float, default=1.)
    args = parser.parse_args()
    validate_runtime(args)
    if args.limit <= 0 or args.temperature < 0 or not 0 < args.top_p <= 1 or (args.prompt_tokens is not None and args.prompt_tokens <= 0):
        parser.error("Invalid sample/sampling settings")
    rows = read_prompts(args.prompts, args.limit)
    arms, source = resolve_arms(args.arms_file)
    kwargs = model_kwargs(args)
    versions = check_versions()
    args.output.mkdir(parents=True, exist_ok=False)
    config = {"args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "model_kwargs": kwargs, "versions": versions, "arm_source": source,
              "arms": arms, "target_prompts": 200, "exploratory_subset": args.limit < 200,
              "prompt_file_sha256": hashlib.sha256(args.prompts.read_bytes()).hexdigest(),
              "semantics": "Raw prompt completion, no implicit chat template; EOS enabled; first predicted token is prefill. Natural variable output length and decode batch.",
              "normalization": "preserve model router defaults; logical route bytes are not measured DRAM traffic"}
    atomic_json(args.output / "config.json", config)
    from vllm import LLM
    llm = LLM(**kwargs)
    rows = tokenize_prompts(rows, llm.get_tokenizer(), args.max_model_len, args.max_tokens,
                            args.prompt_tokens, require_distinct=True)
    atomic_json(args.output / "prompts.json", rows)
    run_pairs(llm, rows, arms, args, args.output)


if __name__ == "__main__":
    main()
