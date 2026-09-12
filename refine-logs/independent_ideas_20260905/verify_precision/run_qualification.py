#!/usr/bin/env python3
"""Offline-prepared, CUDA-only real QDQ selector qualification; no CPU timing fallback."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

for key in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
    os.environ[key] = "1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from budget_adapter import ROOT, POLICIES, STEPS, run_policy
from triage_runtime import PreparedInt4ExpertBackend, cache_sequence_length, clone_cache

MODEL, REVISION = "allenai/OLMoE-1B-7B-0924", "6d84c48581ece794365f2b8e9cfb043c68ade9c5"


def write(path, value):
    with path.open("x") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_prepared(path):
    seal = json.loads((path / "COMPLETE.json").read_text())
    if seal.get("status") != "PREPARED":
        raise ValueError("prepared directory has no completed input preparation")
    for name in ("config.json", "documents.json"):
        if file_sha(path / name) != seal.get("input_sha256", {}).get(name):
            raise ValueError(f"prepared input changed: {name}")
    config = json.loads((path / "config.json").read_text())
    if config["model"] != MODEL or config["revision"] != REVISION:
        raise ValueError("prepared model/revision changed")
    for name, digest in config["source_sha256"].items():
        if file_sha(ROOT / name) != digest:
            raise ValueError(f"implementation differs from prepared inputs: {name}")
    return config, json.loads((path / "documents.json").read_text())


def require_idle_gpu():
    rows = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name",
                                    "--format=csv,noheader,nounits"], text=True).splitlines()
    if any(int(row.split(",", 1)[0]) != os.getpid() for row in rows if row.strip()):
        raise RuntimeError("foreign GPU process; do not run the qualification")


def prepare_documents(tokenizer, documents, excluded, prompt_tokens):
    prepared, seen = [], set()
    for text in documents:
        digest = hashlib.sha256(text.encode()).hexdigest()
        if digest in excluded or digest in seen:
            raise ValueError("duplicate or previously evaluated document; do not silently replace it")
        seen.add(digest)
        ids = tokenizer(text, add_special_tokens=True)["input_ids"]
        if len(ids) < prompt_tokens + STEPS:
            raise ValueError("frozen document is too short; no outcome-dependent replacement")
        prepared.append(dict(document_sha256=digest, text=text,
                             prompt_ids=ids[:prompt_tokens], decode_ids=ids[prompt_tokens:prompt_tokens + STEPS]))
    return prepared


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--prepared-dir", type=Path, help="reuse sealed input IDs/config without retokenizing")
    parser.add_argument("--documents", type=int, default=8)
    parser.add_argument("--offset", type=int, default=1000)
    parser.add_argument("--prompt-tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
    args = parser.parse_args()
    if args.prepared_dir:
        allowed = {"--output-dir", "--prepared-dir", "--prepare-only"}
        if any(value.split("=")[0] not in allowed for value in sys.argv[1:] if value.startswith("--")):
            parser.error("prepared inputs freeze workload arguments; create a new preparation to change them")
    if min(args.documents, args.prompt_tokens, args.repeats) < 1 or args.offset < 0:
        parser.error("positive sizes and nonnegative offset required")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = args.output_dir
    completed = 0
    policy_started = False
    try:
        import torch
        import transformers
        write(output / "environment.json", dict(python=sys.version, torch=torch.__version__,
            transformers=transformers.__version__, cuda=torch.version.cuda,
            cuda_available=torch.cuda.is_available(), cwd=str(ROOT)))
        sys.path.insert(0, str(ROOT / "experiments/shared"))
        from prompts import get_prompts
        lock_path = Path(__file__).with_name("source_lock.json")
        lock = json.loads(lock_path.read_text())
        sources = [Path(__file__), Path(__file__).with_name("budget_adapter.py"), lock_path,
                   ROOT / "docs/ideas/B_verify_precision/confidenceguard_v3/experiments/triage_runtime.py",
                   ROOT / "experiments/shared/prompts.py"]
        config = dict(model=MODEL, revision=REVISION, threshold=lock["threshold"],
            steps=32, high_budget=4, dataset="wikitext103_docs", split=args.split, offset=args.offset,
            documents=args.documents, prompt_tokens=args.prompt_tokens, repeats=args.repeats,
            seed=20260905, policies=POLICIES, source_lock=lock,
            source_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
            torch=torch.__version__, transformers=transformers.__version__, cuda=torch.version.cuda,
            evidence_ceiling="DOUBLE_SHADOW_TEACHER_FORCED_QDQ_SELECTOR_QUALIFICATION")
        if args.prepared_dir:
            config, document_bundle = load_prepared(args.prepared_dir)
            args.repeats, args.prompt_tokens = config["repeats"], config["prompt_tokens"]
            lock = config["source_lock"]
        else:
            tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
            texts = get_prompts("wikitext103_docs", args.documents, offset=args.offset, split=args.split, seed=20260905)
            documents = prepare_documents(tokenizer, texts, set(lock["old_document_sha256s"]), args.prompt_tokens)
            document_bundle = dict(documents=documents, freshness_scope=lock["freshness_scope"])
        documents = document_bundle["documents"]
        write(output / "config.json", config)
        write(output / "documents.json", document_bundle)
        if args.prepare_only:
            write(output / "COMPLETE.json", dict(status="PREPARED", scientific_status="UNRUN",
                input_sha256={name: file_sha(output / name) for name in ("config.json", "documents.json")}))
            return 0
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("exactly one CUDA GPU required; scientific experiment UNRUN")
        require_idle_gpu()
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 GPU required")
        torch.manual_seed(20260905)
        model = transformers.AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION,
            local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation="eager").cuda().eval()
        if args.prompt_tokens + STEPS > model.config.max_position_embeddings:
            raise ValueError("prompt plus continuation exceeds model context")
        backend = PreparedInt4ExpertBackend(model, expected_linears=3072)

        def forward(token, cache, low=False):
            kwargs = dict(input_ids=token, past_key_values=cache, use_cache=True,
                          cache_position=torch.tensor([cache_sequence_length(cache)], device=token.device))
            with torch.inference_mode():
                if low:
                    with backend:
                        return model(**kwargs)
                return model(**kwargs)

        for repeat in range(args.repeats):
            for index, doc in enumerate(documents):
                require_idle_gpu()
                prompt = torch.tensor([doc["prompt_ids"]], device="cuda")
                tokens = torch.tensor([doc["decode_ids"]], device="cuda")
                torch.cuda.synchronize()
                preparation_start = time.perf_counter()
                with torch.inference_mode():
                    initial = model(input_ids=prompt, use_cache=True).past_key_values
                reference_cache, references = clone_cache(initial), []
                for step in range(STEPS):
                    ref = forward(tokens[:, step:step + 1], reference_cache)
                    reference_cache = ref.past_key_values
                    references.append(ref.logits[:, -1, :].detach())
                torch.cuda.synchronize()
                reference_s = time.perf_counter() - preparation_start
                order = POLICIES if (repeat + index) % 2 == 0 else tuple(reversed(POLICIES))
                for policy in order:
                    stem = f"r{repeat:02d}-d{index:03d}-{policy}"
                    with (output / f"{stem}.raw.jsonl").open("x") as raw:
                        def journal(row):
                            raw.write(json.dumps(row, allow_nan=False) + "\n")
                            raw.flush()
                        policy_started = True
                        result = run_policy(initial, tokens, references,
                            high_forward=lambda token, state: forward(token, state),
                            low_forward=lambda token, state: forward(token, state, True),
                            policy=policy, threshold=lock["threshold"], seed=20260905 + index,
                            sync=torch.cuda.synchronize, journal=journal)
                    result.update(document_sha256=doc["document_sha256"], repeat=repeat,
                                  reference_and_common_prefill_s=reference_s,
                                  low_backend="persistent INT4 QDQ weights executed by BF16 F.linear")
                    write(output / f"{stem}.json", result)
                    completed += 1
                    require_idle_gpu()
                del initial, reference_cache, references, ref
        write(output / "COMPLETE.json", dict(status="COMPLETE", policy_runs=completed,
            scientific_status="MEASUREMENT_ONLY_PENDING_MATCHED_BUDGET_ANALYSIS"))
        return 0
    except Exception as exc:
        write(output / "FAILURE.json", dict(error=f"{type(exc).__name__}: {exc}",
            completed_policy_runs=completed, scientific_status="PARTIAL" if policy_started else "UNRUN"))
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
