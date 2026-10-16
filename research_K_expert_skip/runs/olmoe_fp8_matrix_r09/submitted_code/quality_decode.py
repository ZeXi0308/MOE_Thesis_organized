"""Controlled vLLM decode teacher-forced quality, not a speed benchmark.

Example (GPU execution belongs to the main session's lock wrapper):
  python quality_decode.py --model /path/to/OLMoE --articles validation.jsonl \
    --output quality_b8.json --batch 8 --tokens 32 --prefix-tokens 128 \
    --limit-docs 16 --arms dense global1 global2 global4 gate05 marginal05

--tokens counts genuine decode predictions. One extra forced token is emitted
from the prefill prediction and retained separately, excluded from summaries.
PPL describes the selected fixed-length WikiText decode windows; it is not the
standard whole-corpus WikiText PPL or a free-generation task-quality result.
Alternatively --tokenized-windows takes prepare_wikitext_rolling.py's manifest
and evaluates all windows by default, including the short tail/incomplete batch.
Its prefix includes one excluded prefill bridge: a context of P tokens uses
an engine prompt of P-1, then scores every source position [P,N) in true decode.
This is rolling PPL under the manifest's declared context resets, not independent
article samples or an equivalence claim to other standard context protocols.
The loaded router's normalization is preserved. In the inspected OLMoE vLLM
path renormalize=False: dropped weights become zero and surviving weights are
unchanged; normalized gate fractions are used only for eligibility/budgeting.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
import traceback


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def arm_specs(names, threshold=.15, tail_rank=4, backend="torch"):
    names = [name for group in names for name in group.split(",") if name and name != "native"]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate policy names")
    arms = [("native", None)]
    for name in names:
        if name == "dense":
            spec = {"mode": "dense"}
        elif name in {"global1", "global2", "global4"}:
            spec = {"mode": "global", "drop_j": int(name[-1]), "mass_budget": 1.}
        elif name in {f"{mode}{mass}" for mode in ("gate", "marginal") for mass in ("05", "10", "20")}:
            mode = "marginal" if name.startswith("marginal") else "gate"
            spec = {"mode": mode, "gate_threshold": threshold,
                    "mass_budget": int(name[-2:]) / 100, "tail_rank": tail_rank}
        else:
            raise ValueError(f"Unknown arm {name}; use dense, global1/2/4, gate05/10/20, marginal05/10/20")
        spec["backend"] = backend
        arms.append((name, spec))
    return arms


def load_arms_file(path):
    """Read exact policy specs; file values are never replaced by CLI defaults.

    Accept a list of {name, spec} objects, or an object with an `arms` list and
    optional descriptive metadata. The untouched file and SHA-256 are retained.
    Native is always first and has spec=None; all other arms require a dict.
    """
    source = path.read_bytes()
    payload = json.loads(source)
    entries = payload.get("arms") if isinstance(payload, dict) else payload
    if not isinstance(entries, list) or not entries:
        raise ValueError("arms-file must contain a nonempty arms list")
    names, candidates = set(), []
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"name", "spec"}:
            raise ValueError("Each arms-file entry must contain exactly name and spec")
        name, spec = entry["name"], entry["spec"]
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", name):
            raise ValueError("Arm names must be safe filename components (letters, digits, _, ., -)")
        if name in names:
            raise ValueError(f"Duplicate arm {name}")
        names.add(name)
        if name == "native":
            if spec is not None:
                raise ValueError("Reserved native arm must have null spec")
        else:
            if not isinstance(spec, dict) or not isinstance(spec.get("mode"), str):
                raise ValueError(f"Arm {name} requires an explicit spec object with mode")
            candidates.append((name, dict(spec)))
    return [("native", None), *candidates], {
        "path": str(path.resolve()), "sha256": hashlib.sha256(source).hexdigest(),
        "payload": payload,
    }


def resolve_arm_specs(arms):
    """Quality uses counted native reduced-k; latency is measured separately."""
    from dataclasses import asdict
    from expert_skip import SkipConfig
    resolved = []
    for name, spec in arms:
        if spec is None:
            resolved.append((name, None))
        elif spec.get("mode") == "global_native":
            extra = set(spec) - {"mode", "drop_j", "backend", "count_routes"}
            drop = spec.get("drop_j", 1)
            backend = spec.get("backend", "triton")
            if extra or not isinstance(drop, int) or isinstance(drop, bool) or drop < 0:
                raise ValueError(f"Invalid native reduced-k arm {name}: {spec}")
            SkipConfig(mode="dense", backend=backend)
            resolved.append((name, {"mode": "global_native", "drop_j": drop,
                                    "backend": backend, "count_routes": True}))
        else:
            resolved.append((name, asdict(SkipConfig(**spec))))
    return resolved


def prepare_documents(path, tokenizer, prefix, decode_tokens, limit, batch):
    documents, seen_keys = [], set()
    counts = {"lines_seen": 0, "short_documents": 0, "duplicate_documents": 0,
              "eligible_documents_not_in_full_batch": 0}
    needed = prefix + decode_tokens + 1
    target_docs = limit - limit % batch
    if target_docs < batch:
        raise ValueError("limit-docs must permit at least one full requested batch")
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            counts["lines_seen"] += 1
            row = json.loads(line)
            text = row["text"]
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if row.get("text_sha256") and row["text_sha256"] != digest:
                raise ValueError(f"Article text hash mismatch at line {number}")
            if digest in seen_keys:
                counts["duplicate_documents"] += 1
                continue
            tokens = list(tokenizer.encode(text, add_special_tokens=True))
            if len(tokens) < needed:
                counts["short_documents"] += 1
                continue
            seen_keys.add(digest)
            documents.append({"doc_key": f"line{number}:{digest}", "id": row.get("id", number),
                              "source_line": number, "text_sha256": digest,
                              "split": row.get("split", "unspecified"), "source_token_count": len(tokens),
                              "window_start_token": 0, "prompt_token_ids": tokens[:prefix],
                              "forced_tokens": tokens[prefix:needed]})
            if len(documents) == target_docs:
                break
    complete_count = len(documents) - len(documents) % batch
    counts["eligible_documents_not_in_full_batch"] = len(documents) - complete_count
    documents = documents[:complete_count]
    if not documents:
        raise ValueError("Insufficient distinct articles of required length for one complete batch")
    return documents, counts


def read_and_validate_metrics(path, group, docs, arm, batch, tokens):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    expected = {(d["doc_key"], position): d["forced_tokens"][position]
                for d in docs for position in range(tokens + 1)}
    observed = {}
    prompt_lengths = {d["doc_key"]: len(d["prompt_token_ids"]) for d in docs}
    for row in rows:
        key = (row["doc_key"], row["output_position"])
        if key in observed or key not in expected:
            raise RuntimeError(f"Duplicate or unexpected token metric: {key}")
        if row["group"] != group or row["arm"] != arm or row["batch_rows"] != batch:
            raise RuntimeError("Metric group, arm or observed batch mismatch")
        if row["target_token"] != expected[key]:
            raise RuntimeError("Metric target does not match article")
        if (row["prediction_phase"] != ("prefill_next_token" if key[1] == 0 else "decode")
                or row["context_tokens"] != prompt_lengths[key[0]] + key[1]):
            raise RuntimeError("Metric prediction phase/context does not match the true-decode contract")
        observed[key] = row
    if observed.keys() != expected.keys():
        raise RuntimeError(f"Missing metrics: observed {len(observed)} of {len(expected)} predictions")
    return rows


def document_groups(documents, batch):
    """Keep fixed-length cohorts; retain short final cohorts and scoring tails."""
    current, shape = [], None
    for doc in documents:
        next_shape = (len(doc["prompt_token_ids"]), len(doc["forced_tokens"]))
        if current and (len(current) == batch or next_shape != shape):
            yield current
            current = []
        current.append(doc)
        shape = next_shape
    if current:
        yield current


def merge_quality_summaries(summaries):
    """Pool token-weighted sufficient statistics without retaining a split's rows.

    Groups have disjoint doc/window keys. 'documents' is a compatibility field;
    for rolling input it counts windows, never independent article replicates.
    """
    count = sum(row["tokens"] for row in summaries)
    if not count:
        return {"tokens": 0, "mean_nll": None, "ppl": None, "mean_kl": None}
    mean_nll = math.fsum(row["mean_nll"] * row["tokens"] for row in summaries) / count
    return {"tokens": count, "documents": sum(row["documents"] for row in summaries),
            "mean_nll": mean_nll, "ppl": math.exp(mean_nll) if mean_nll < 709 else None,
            "mean_kl": math.fsum(row["mean_kl"] * row["tokens"] for row in summaries) / count,
            "mean_nll_delta": math.fsum(row["mean_nll_delta"] * row["tokens"] for row in summaries) / count,
            "max_kl": max(row["max_kl"] for row in summaries), "min_kl": min(row["min_kl"] for row in summaries)}


def validate_worker_snapshot(snapshots, installation, batch, tokens, has_policy):
    if len(snapshots) != 1 or len(installation) != 1:
        raise RuntimeError("This runner requires exactly one GPU worker")
    snapshot = snapshots[0]
    phases = snapshot["phase_steps"]
    if phases.get("decode", 0) != tokens or phases.get("prefill", 0) != 1 or phases.get("mixed", 0):
        raise RuntimeError(f"Actual scheduler phase coverage differs from 1 prefill + {tokens} decode steps: {phases}")
    if has_policy:
        layers = snapshot["layers"]["layers"]
        if len(layers) != len(installation[0]["layers"]):
            raise RuntimeError("Not all installed MoE layers executed the policy")
        if any(v["tokens"] != batch * tokens or v["events"] != tokens for v in layers.values()):
            raise RuntimeError("Per-layer policy counters do not match actual controlled decode batch")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    input_source = parser.add_mutually_exclusive_group(required=True)
    input_source.add_argument("--articles", type=Path, help="Original article-text JSONL workflow")
    input_source.add_argument("--tokenized-windows", type=Path,
                              help="Rolling manifest.json; exact token IDs, tokenizer identity checked; all windows by default")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=32, help="Text input only: scored decode tokens per article")
    parser.add_argument("--prefix-tokens", type=int, default=128, help="Text input only: engine prompt length")
    parser.add_argument("--limit-docs", type=int, default=32, help="Text input only: article selection limit")
    parser.add_argument("--limit-windows", type=int, help="Rolling input only: explicit exploratory prefix; default evaluates complete split coverage")
    arm_source = parser.add_mutually_exclusive_group()
    arm_source.add_argument("--arms", nargs="+", default=None)
    arm_source.add_argument("--arms-file", type=Path,
                            help="JSON arms list with arbitrary explicit SkipConfig specs; native added automatically")
    parser.add_argument("--gate-threshold", type=float, default=.15)
    parser.add_argument("--tail-rank", type=int, default=4, help="Zero-based actual-weight rank threshold")
    parser.add_argument("--policy-backend", choices=["torch", "triton", "triton_multi"], default="torch")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--quantization", default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=.5)
    parser.add_argument("--reference-max-mib", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--identity-tolerance", type=float, default=1e-6)
    return parser


def main():
    args = build_parser().parse_args()
    if min(args.batch, args.tokens, args.prefix_tokens, args.limit_docs, args.reference_max_mib) <= 0:
        raise ValueError("Batch, token counts, document limit and cache budget must be positive")
    if not 0 <= args.gate_threshold <= 1 or args.tail_rank < 0:
        raise ValueError("Invalid gate threshold or rank")
    if args.limit_windows is not None and (args.tokenized_windows is None or args.limit_windows < 1):
        raise ValueError("A positive --limit-windows requires --tokenized-windows")
    if args.arms_file:
        arms, arm_source = load_arms_file(args.arms_file)
    else:
        names = args.arms if args.arms is not None else ["dense", "global1", "global2", "global4",
                "gate05", "gate10", "gate20", "marginal05", "marginal10", "marginal20"]
        arms = arm_specs(names, args.gate_threshold, args.tail_rank, args.policy_backend)
        arm_source = {"kind": "named CLI arms"}
    # Validate arbitrary fields before loading weights and record resolved
    # defaults, so a typo or stale core version fails before GPU allocation.
    arms = resolve_arm_specs(arms)
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    output = args.output.resolve()
    if output.suffix == ".json":
        summary_path, artifact_dir = output, output.with_name(output.stem + "_artifacts")
    else:
        artifact_dir, summary_path = output, output / "summary.json"
    if summary_path.exists() or artifact_dir.exists():
        raise FileExistsError("Output already exists; choose a new run path")
    artifact_dir.mkdir(parents=True)
    manifest = {"status": "RUNNING", "started_unix": time.time(), "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "arms": [{"name": n, "spec": s} for n, s in arms], "batches": [],
                "arm_source": arm_source,
                "scope": "teacher-forced selected decode windows; no timing or task-accuracy claim",
                "aggregation": "exp(sum token NLL / scored decode tokens); position zero excluded",
                "kl_direction": "KL(native || policy), full vocabulary, before forcing mask",
                "normalization": "preserve each installed router's renormalize flag and routing scale; decision gate fractions are normalized separately",
                "runtime_contract": {"vllm": "0.26.0", "VLLM_ENABLE_V1_MULTIPROCESSING": "0", "async_scheduling": False}}
    atomic_json(summary_path, manifest)
    accumulated = defaultdict(list)
    try:
        rolling = None
        engine_prefix, max_decode_tokens = args.prefix_tokens, args.tokens
        if args.tokenized_windows:
            from prepare_wikitext_rolling import load_rolling_documents, validate_tokenizer_identity
            from transformers import AutoTokenizer
            docs, selection, rolling = load_rolling_documents(args.tokenized_windows, args.limit_windows)
            # Fail identity mismatches before allocating a GPU/model; recheck the
            # actual vLLM tokenizer after engine construction as well.
            preflight_tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True,
                trust_remote_code=args.trust_remote_code, use_fast=True)
            manifest["preflight_tokenizer_identity"] = validate_tokenizer_identity(rolling["tokenizer"], preflight_tokenizer)
            del preflight_tokenizer
            engine_prefix = rolling["engine_prompt_tokens"]
            max_decode_tokens = rolling["max_scored_tokens"]
            manifest.update(scope=rolling["scope"], rolling_input=rolling,
                            statistical_unit=rolling["statistical_unit"],
                            per_document_field_unit="rolling window, not independent article",
                            context_definition="P-1 engine prompt + one excluded prefill bridge; scored context lengths P..P+S-1")
        from vllm import LLM, SamplingParams
        from teacher_force import EXTRA_KEY, quality_summary
        llm = LLM(model=args.model, dtype=args.dtype, quantization=args.quantization,
                  trust_remote_code=args.trust_remote_code, tensor_parallel_size=1,
                  enforce_eager=True, compilation_config={"mode": 0, "cudagraph_mode": "NONE"},
                  worker_cls="skip_worker.ExpertSkipWorker", moe_backend="triton",
                  logits_processors=["teacher_force:TeacherForceLogitsProcessor"],
                  enable_prefix_caching=False, enable_chunked_prefill=False, async_scheduling=False,
                  max_model_len=engine_prefix + max_decode_tokens + 8, max_num_seqs=args.batch,
                  max_num_batched_tokens=max(args.batch * engine_prefix, engine_prefix + max_decode_tokens + 8),
                  gpu_memory_utilization=args.gpu_memory_utilization, seed=args.seed)
        if rolling:
            manifest["runtime_tokenizer_identity"] = validate_tokenizer_identity(rolling["tokenizer"], llm.get_tokenizer())
        else:
            docs, selection = prepare_documents(args.articles, llm.get_tokenizer(), args.prefix_tokens,
                                                 args.tokens, args.limit_docs, args.batch)
        manifest["selection"] = selection
        manifest["selected_documents"] = len(docs)
        manifest["selected_splits"] = sorted({d["split"] for d in docs})
        atomic_json(artifact_dir / "documents.json", docs)
        for group_index, group_docs in enumerate(document_groups(docs, args.batch)):
            actual_batch = len(group_docs)
            actual_tokens = len(group_docs[0]["forced_tokens"]) - 1
            actual_prefix = len(group_docs[0]["prompt_token_ids"])
            group = f"batch{group_index:04d}"
            for arm_index, (arm, spec) in enumerate(arms):
                installed = llm.collective_rpc("skip_set_policy", args=(spec, "decode", 0, False))
                token_path = artifact_dir / f"{group}.{arm}.tokens.jsonl"
                token_path.touch(exist_ok=False)
                params = [SamplingParams(temperature=0., top_p=1., top_k=-1,
                          max_tokens=actual_tokens + 1, ignore_eos=True, detokenize=False,
                          extra_args={EXTRA_KEY: {"group": group, "doc_key": d["doc_key"],
                              "arm": arm, "role": "baseline" if arm == "native" else "candidate",
                              "forced_tokens": d["forced_tokens"], "metrics_path": str(token_path),
                              "expected_batch": actual_batch, "prefix_tokens": actual_prefix,
                              "reference_max_mib": args.reference_max_mib,
                              "release_reference": arm_index == len(arms) - 1}}) for d in group_docs]
                generated = llm.generate([{"prompt_token_ids": d["prompt_token_ids"]} for d in group_docs],
                                         params, use_tqdm=False)
                if len(generated) != actual_batch:
                    raise RuntimeError("Not all requested documents returned")
                for result, doc in zip(generated, group_docs):
                    if len(result.outputs) != 1 or list(result.outputs[0].token_ids) != doc["forced_tokens"]:
                        raise RuntimeError("Returned output differs from forced token trajectory")
                snapshot = llm.collective_rpc("skip_snapshot")
                raw_path = artifact_dir / f"{group}.{arm}.worker.json"
                atomic_json(raw_path, {"installation": installed, "snapshot": snapshot,
                                      "outputs": [{"request_id": r.request_id,
                                                   "finish_reason": r.outputs[0].finish_reason,
                                                   "tokens": list(r.outputs[0].token_ids)} for r in generated]})
                validate_worker_snapshot(snapshot, installed, actual_batch, actual_tokens, spec is not None)
                rows = read_and_validate_metrics(token_path, group, group_docs, arm, actual_batch, actual_tokens)
                decode_rows = [r for r in rows if r["prediction_phase"] == "decode"]
                if arm == "dense" and max(abs(r["kl_base_to_policy"]) for r in rows) > args.identity_tolerance:
                    raise RuntimeError("Dense identity KL exceeds tolerance; inspect implementation/measurement")
                per_doc = {d["doc_key"]: quality_summary([r for r in decode_rows if r["doc_key"] == d["doc_key"]])
                           for d in group_docs}
                group_quality = quality_summary(decode_rows)
                if rolling:
                    accumulated[arm].append(group_quality)
                else:
                    accumulated[arm].extend(decode_rows)
                result_row = {"group": group, "arm": arm, "spec": spec, "valid": True,
                              "actual_batch": actual_batch, "scored_tokens_per_window": actual_tokens,
                              "engine_prompt_tokens": actual_prefix,
                              "quality": group_quality, "per_document": per_doc,
                              "prefill_prediction": quality_summary([r for r in rows if r["output_position"] == 0]),
                              "policy_counter_file": str(raw_path), "token_file": str(token_path)}
                counters = snapshot[0]["layers"].get("totals")
                result_row["policy_counters"] = counters
                if spec is not None and spec["mode"] == "rank_gate":
                    result_row["diagnostic_intervention"] = {
                        "rank_start_zero_based": spec["tail_rank"], "rank_end_exclusive": spec["rank_end"],
                        "gate_min_inclusive": spec["gate_min"], "gate_max_exclusive": spec["gate_threshold"],
                        "layer_scope": "all routed decode layers",
                        "observed_skipped_slots": counters["skipped_slots"],
                        "event_state": "observed" if counters["skipped_slots"] else "no_selected_events",
                        "estimand": "whole-model consequence of this repeatedly applied bin-specific policy; not an additive per-event causal cost",
                    }
                manifest["batches"].append(result_row)
                summarize = merge_quality_summaries if rolling else quality_summary
                manifest["quality_by_arm"] = {name: summarize(values) for name, values in accumulated.items()}
                atomic_json(summary_path, manifest)
                print(json.dumps({"group": group, "arm": arm, **result_row["quality"]}), flush=True)
        if rolling:
            expected_tokens = selection["selected_scored_tokens"]
            if any(manifest["quality_by_arm"][arm]["tokens"] != expected_tokens for arm, _ in arms):
                raise RuntimeError("At least one arm did not score every selected source token exactly once")
            manifest["rolling_coverage_verified"] = {"selected_scored_tokens_per_arm": expected_tokens,
                "selected_source_span": selection["selected_scored_span"],
                "complete_split_after_initial_context": selection["complete_split_after_initial_context"],
                "all_scored_predictions_decode": True,
                "actual_batch_distribution": dict(Counter(
                    row["actual_batch"] for row in manifest["batches"] if row["arm"] == "native"))}
        manifest["status"] = "PASS"
        manifest["completed_unix"] = time.time()
        atomic_json(summary_path, manifest)
    except BaseException as error:
        manifest["status"] = "ERROR"
        manifest["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        manifest["completed_unix"] = time.time()
        atomic_json(summary_path, manifest)
        raise


if __name__ == "__main__":
    main()
