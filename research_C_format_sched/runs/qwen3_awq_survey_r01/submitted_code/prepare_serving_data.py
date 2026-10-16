#!/usr/bin/env python3
"""Fetch two pinned public sources, then prepare CPU-only rendered requests.

  python code/prepare_serving_data.py fetch
  python code/prepare_serving_data.py prepare --tokenizer /local/model \
      --output datasets/serving_olmoe_r01

Preparation never truncates prompts or loads model weights. The two serving
files share prompts but explicitly distinguish natural EOS from forced-length
workload replay. LongBench-derived prompts are not a LongBench quality score.
"""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import shlex
import shutil
import sys
import time
import traceback
import zipfile

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "sharegpt": {
        "repo": "anon8231489123/ShareGPT_Vicuna_unfiltered",
        "revision": "192ab2185289094fc556ec8ce5ce1e8e587154ca",
        "filename": "ShareGPT_V3_unfiltered_cleaned_split.json", "bytes": 672837942,
        "sha256": "35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4"},
    "longbench": {
        "repo": "zai-org/LongBench", "revision": "5e628be450b7e67fb7ae6e201bd6d8f7056f7672",
        "filename": "data.zip", "bytes": 113932529,
        "sha256": "cb45b11a4133c6bc1d6a44b0f8e701335ff1e543195db1103472e575857f7f64"},
}
TASK_TEMPLATES = {
    "qasper": "Read the scientific article and answer the question using the article.\n\nArticle:\n{context}\n\nQuestion: {input}\n\nAnswer:",
    "multi_news": "Write a concise summary of the following news articles.\n\nArticles:\n{context}\n\nSummary:",
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def save(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(encoded(value) + "\n")
    temporary.replace(path)


def process_record(stage):
    return {"stage": stage, "state": "running", "pid": os.getpid(), "parent_pid": os.getppid(),
            "command": shlex.join([sys.executable, *sys.argv]), "start_unix_s": time.time(),
            "script_sha256": sha256(__file__), "cpu_only": True, "model_loaded": False}


def check_source(dest, name):
    spec = SOURCES[name]
    path = dest / name / spec["filename"]
    actual = {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path)}
    if actual["bytes"] != spec["bytes"] or actual["sha256"] != spec["sha256"]:
        raise ValueError(f"source byte/hash mismatch: {path}")
    return {**spec, **actual, "verified": True}


def fetch(args):
    from huggingface_hub import hf_hub_download
    dest = args.sources.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    # Independent CPU downloader lock: never takes or inspects the GPU lock.
    with (dest / ".fetch.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("a serving-source downloader already owns this directory") from None
        status_path = dest / "fetch_status.json"
        previous = json.loads(status_path.read_text()) if status_path.exists() else None
        state = process_record("fetch")
        state.update(endpoint=os.environ["HF_ENDPOINT"], files={},
                     expected_sources=SOURCES, previous_attempt=previous)
        save(status_path, state)
        try:
            for name, spec in SOURCES.items():
                state["current_source"] = name
                save(status_path, state)
                path = dest / name / spec["filename"]
                if not path.exists():
                    if shutil.disk_usage(dest).free < spec["bytes"] + 2 * 1024**3:
                        raise RuntimeError("less than source size plus 2 GiB disk reserve")
                    print(f"fetch {name} {spec['revision']} {spec['bytes']} bytes", flush=True)
                    hf_hub_download(repo_id=spec["repo"], repo_type="dataset", revision=spec["revision"],
                                    filename=spec["filename"], local_dir=dest / name)
                state["files"][name] = check_source(dest, name)
                save(status_path, state)
                print(f"verified {name}: {state['files'][name]['sha256']}", flush=True)
            state["state"] = "complete"
        except BaseException as error:
            state.update(state="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
            raise
        finally:
            state["end_unix_s"] = time.time()
            save(status_path, state)


def json_array_rows(path):
    """Stream the large ShareGPT top-level array using only the stdlib."""
    decoder = json.JSONDecoder()
    with path.open(encoding="utf-8") as stream:
        buf, ended, first = "", False, True
        while True:
            if not ended and len(buf) < 1024 * 1024:
                chunk = stream.read(1024 * 1024)
                buf += chunk
                ended = not chunk
            buf = buf.lstrip()
            if first:
                if not buf.startswith("["):
                    raise ValueError("ShareGPT source is not a JSON array")
                buf, first = buf[1:], False
                continue
            if buf.startswith("]"):
                if buf[1:].strip() or stream.read().strip():
                    raise ValueError("unexpected content after ShareGPT array")
                return
            if buf.startswith(","):
                buf = buf[1:].lstrip()
            try:
                row, end = decoder.raw_decode(buf)
            except json.JSONDecodeError:
                if ended:
                    raise
                chunk = stream.read(1024 * 1024)
                buf += chunk
                ended = not chunk
                continue
            yield row
            buf = buf[end:]


def candidates(sources):
    path = sources / "sharegpt" / SOURCES["sharegpt"]["filename"]
    for index, row in enumerate(json_array_rows(path)):
        yield "sharegpt", "first_human_turn", index, row
    path = sources / "longbench" / SOURCES["longbench"]["filename"]
    with zipfile.ZipFile(path) as archive:
        for task in TASK_TEMPLATES:
            matches = [n for n in archive.namelist() if Path(n).name == task + ".jsonl"]
            if len(matches) != 1:
                raise ValueError(f"expected exactly one {task}.jsonl member, got {matches}")
            with archive.open(matches[0]) as stream:
                for index, line in enumerate(stream):
                    if line.strip():
                        yield "longbench", task, index, json.loads(line)


def content_for(source, task, row):
    if not isinstance(row, dict):
        return None, "malformed_record"
    if source == "sharegpt":
        turns = row.get("conversations")
        if not isinstance(turns, list):
            return None, "missing_conversations"
        human = next((x for x in turns if isinstance(x, dict) and x.get("from") in ("human", "user")), None)
        if human is None:
            return None, "missing_human_turn"
        value = human.get("value")
        return (value, None) if isinstance(value, str) and value.strip() else (None, "empty_or_nontext_first_human")
    context, question = row.get("context"), row.get("input", "")
    if not isinstance(context, str) or not context.strip():
        return None, "empty_or_nontext_context"
    if task == "qasper" and (not isinstance(question, str) or not question.strip()):
        return None, "empty_or_nontext_question"
    return TASK_TEMPLATES[task].format(context=context, input=question), None


def jsonl(path, rows):
    with path.open("x") as out:
        for row in rows:
            out.write(encoded(row) + "\n")


def prepare(args):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoTokenizer
    sources, output, tokenizer_path = args.sources.resolve(), args.output.resolve(), args.tokenizer.resolve()
    output.mkdir(parents=True, exist_ok=False)
    state = process_record("prepare")
    manifest = output / "manifest.json"
    save(manifest, state)
    try:
        state["sources"] = {name: check_source(sources, name) for name in SOURCES}
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, trust_remote_code=False)
        if not tokenizer.chat_template:
            raise ValueError("local tokenizer has no chat template; no guessed chat format is allowed")
        config = json.loads((tokenizer_path / "config.json").read_text())
        limit = config.get("max_position_embeddings")
        if type(limit) is not int or limit <= 0:
            raise ValueError("local config must declare max_position_embeddings")
        context_limit = min(args.context_limit, limit)
        if args.max_tokens >= context_limit or args.long_min_tokens > context_limit - args.max_tokens:
            raise ValueError("generation cap/long-prompt lower bound exceeds context budget")
        tokenizer.save_pretrained(output / "tokenizer")
        shutil.copy2(tokenizer_path / "config.json", output / "tokenizer" / "config.json")
        state.update(tokenizer_path=str(tokenizer_path), transformers=importlib.metadata.version("transformers"),
                     tokenizer_class=type(tokenizer).__name__, seed=args.seed,
                     per_class=args.per_class, context_limit=context_limit, max_tokens=args.max_tokens,
                     long_min_tokens=args.long_min_tokens, truncate=False, add_special_tokens=True,
                     template={"chat_template": tokenizer.chat_template, "add_generation_prompt": True,
                               "longbench_content_templates": TASK_TEMPLATES,
                               "scope": "study-specific task prompts, not official LongBench quality evaluation"},
                     selection="independent reservoir sampling per source class, complete population scan; then seeded request shuffle",
                     natural_generation={"temperature": 0.0, "ignore_eos": False, "min_tokens": 0},
                     workload_replay={"temperature": 0.0, "ignore_eos": True, "min_tokens": args.max_tokens,
                                      "scope": "forced-length generated workload, not recorded-token or quality-equivalent replay"},
                     population={}, no_generation_performed=True)
        save(manifest, state)
        population, selected = [], {name: [] for name in SOURCES}
        eligible = Counter()
        reasons = {name: Counter() for name in SOURCES}
        task_counts = {}
        rng = {name: random.Random(f"{args.seed}:{name}") for name in SOURCES}
        seen = set()
        for source, task, index, row in candidates(sources):
            rid = f"{source}:{task}:{index}"
            source_id = row.get("id", row.get("_id")) if isinstance(row, dict) else None
            item = {"request_id": rid, "source": source, "task": task, "source_row": index, "source_id": source_id}
            text, reason = content_for(source, task, row)
            prompt, token_count = None, None
            if reason is None:
                prompt = tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)
                tokens = tokenizer(prompt, add_special_tokens=True, truncation=False)["input_ids"]
                token_count = len(tokens)
                prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
                if token_count == 0:
                    reason = "zero_prompt_tokens"
                elif token_count + args.max_tokens > context_limit:
                    reason = "prompt_plus_cap_exceeds_context"
                elif source == "longbench" and token_count < args.long_min_tokens:
                    reason = "below_long_prompt_minimum"
                elif prompt_hash in seen:
                    reason = "duplicate_eligible_prompt"
                else:
                    seen.add(prompt_hash)
                    item["prompt_sha256"] = prompt_hash
            item["prompt_tokens"] = token_count
            item["eligibility"] = reason or "eligible"
            population.append(item)
            reasons[source][item["eligibility"]] += 1
            task_counts.setdefault(task, Counter())[item["eligibility"]] += 1
            if reason is None:
                eligible[source] += 1
                candidate = {**item, "prompt": prompt, "source_record": row}
                if len(selected[source]) < args.per_class:
                    selected[source].append(candidate)
                else:
                    slot = rng[source].randrange(eligible[source])
                    if slot < args.per_class:
                        selected[source][slot] = candidate
            if len(population) % 2000 == 0:
                state["population"] = {name: dict(counts) for name, counts in reasons.items()}
                state["rows_scanned"] = len(population)
                save(manifest, state)
                print(f"scanned={len(population)} eligible={dict(eligible)}", flush=True)
        state["population"] = {name: dict(counts) for name, counts in reasons.items()}
        state["population_by_task"] = {task: dict(counts) for task, counts in task_counts.items()}
        chosen_ids = {r["request_id"] for rows in selected.values() for r in rows}
        for row in population:
            row["selected"] = row["request_id"] in chosen_ids
        jsonl(output / "population.jsonl", population)
        if any(len(rows) != args.per_class for rows in selected.values()):
            raise ValueError(f"insufficient eligible population for fixed per-class count: {dict(eligible)}")
        chosen = selected["sharegpt"] + selected["longbench"]
        random.Random(args.seed).shuffle(chosen)
        base_rows = [{k: v for k, v in r.items() if k not in ("source_record", "eligibility")} for r in chosen]
        jsonl(output / "selected_source_records.jsonl", [{"request_id": r["request_id"], "source_record": r["source_record"]} for r in chosen])
        for filename, sampling in (("serving_natural.jsonl", state["natural_generation"]),
                                   ("serving_fixed_workload.jsonl", {k: v for k, v in state["workload_replay"].items() if k != "scope"})):
            jsonl(output / filename, [{**r, "max_tokens": args.max_tokens, "sampling": sampling} for r in base_rows])
        route_rows = []
        for source in SOURCES:
            rows = [r for r in base_rows if r["source"] == source]
            random.Random(f"{args.seed}:routes:{source}").shuffle(rows)
            route_rows.extend(rows[:16])
        random.Random(f"{args.seed}:route_order").shuffle(route_rows)
        jsonl(output / "routes_natural_32.jsonl", route_rows)
        state.update(state="complete", rows_scanned=len(population), selected_total=len(chosen),
                     selected_by_task=dict(Counter(r["task"] for r in chosen)),
                     route_prompt_count=len(route_rows), route_generation_cap=32,
                     route_command_options="--max-prompts 32 --max-new-tokens 32 --add-special-tokens",
                     request_schedule="no arrival_s embedded; bench_serving sets fixed-seed Poisson arrivals per request rate",
                     termination_accounting="data preparation does not generate; consumers must retain EOS/cap/error/timeout/unfinished outcomes",
                     prompt_token_statistics={name: {"min": min(r["prompt_tokens"] for r in rows),
                         "max": max(r["prompt_tokens"] for r in rows),
                         "mean": sum(r["prompt_tokens"] for r in rows) / len(rows)} for name, rows in selected.items()})
        torch = sys.modules.get("torch")
        state["torch_cuda_initialized"] = bool(torch and torch.cuda.is_initialized())
        if state["torch_cuda_initialized"]:
            raise RuntimeError("CUDA unexpectedly initialized")
        state["output_files"] = {str(p.relative_to(output)): {"bytes": p.stat().st_size, "sha256": sha256(p)}
                                 for p in sorted(output.rglob("*")) if p.is_file() and p != manifest}
    except BaseException as error:
        state.update(state="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        raise
    finally:
        state["end_unix_s"] = time.time()
        save(manifest, state)
    print(encoded({"state": state["state"], "selected_total": state["selected_total"], "population": state["population"], "output": str(output)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)
    p = sub.add_parser("fetch")
    p.add_argument("--sources", type=Path, default=ROOT / "datasets/serving_sources")
    p = sub.add_parser("prepare")
    p.add_argument("--sources", type=Path, default=ROOT / "datasets/serving_sources")
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20261016)
    p.add_argument("--per-class", type=int, default=128)
    p.add_argument("--context-limit", type=int, default=4096)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--long-min-tokens", type=int, default=1024)
    args = parser.parse_args()
    if args.stage == "prepare" and (args.per_class < 16 or args.max_tokens < 32 or args.context_limit <= 0 or args.long_min_tokens < 1):
        parser.error("per-class >=16, max-tokens >=32 and positive length bounds are required")
    fetch(args) if args.stage == "fetch" else prepare(args)


if __name__ == "__main__":
    main()
