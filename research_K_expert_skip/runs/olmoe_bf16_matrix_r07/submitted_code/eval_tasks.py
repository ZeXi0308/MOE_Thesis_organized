#!/usr/bin/env python3
"""Task quality entry points for lm-eval 0.4.13 and vLLM 0.26.0.

prepare: CPU-only, pin the four official task dataset repos to immutable HF
commits, cache them under --data-dir, export explicit generation adaptations.
standard: official MMLU 5-shot, GSM8K 5-shot, HellaSwag 0-shot and ARC-c 0-shot.
MMLU/HellaSwag/ARC-c use prompt loglikelihood and are identity controls for a
decode-only intervention, NOT evidence that generated task quality is safe.
GSM8K's official generate_until evaluation does exercise decode.
adapted: separately labelled MC generation adaptation (brief reasoning then
"Final answer: A"). First-token-only answers still do not exercise decode.

All inference uses an existing local model, offline datasets, identical samples
and prompt hashes across arms, and the main session's external GPU lock. No
forced length or ignore_eos. --limit is per leaf task (including each MMLU
subject); the default is the full task. --self-test uses CPU stdlib only.

Official APIs checked against installed lm_eval 0.4.13 and these sources:
https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/evaluator.py
https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/models/vllm_causallms.py
https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/mmlu
https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/gsm8k
https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/hellaswag
https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/arc
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import statistics
import traceback

from quality_decode import atomic_json
from free_generate import (ROOT, add_runtime_args, append_json, check_versions,
    digest, finish_category, model_kwargs, one_worker, resolve_arms, run_pairs,
    tokenize_prompts, validate_runtime)

REPOS = {"mmlu": "cais/mmlu", "gsm8k": "openai/gsm8k",
         "hellaswag": "Rowan/hellaswag", "arc_challenge": "allenai/ai2_arc"}
SHOTS = {"mmlu": 5, "gsm8k": 5, "hellaswag": 0, "arc_challenge": 0}


def family(task):
    return "mmlu" if task.startswith("mmlu_") else task


def plain(value):
    """JSON-compatible conversion retaining numeric values, not repr strings."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if hasattr(value, "tolist"):
        return plain(value.tolist())
    return str(value)


def storage_bytes(path):
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())


def data_environment(path, offline):
    path = Path(path).resolve()
    os.environ["HF_DATASETS_CACHE"] = str(path / "cache")
    os.environ["HF_HUB_CACHE"] = str(path / "hub")
    os.environ["HF_HUB_OFFLINE"] = "1" if offline else "0"
    os.environ["HF_DATASETS_OFFLINE"] = "1" if offline else "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    return path


@contextmanager
def pinned_datasets(source, data_dir, budget_bytes=None):
    """Scope every harness download to the four fixed official repo commits."""
    import datasets
    original = datasets.load_dataset
    def load(*args, **kwargs):
        repo = kwargs.get("path", args[0] if args else None)
        if repo not in source["repositories"]:
            raise ValueError(f"Unexpected dataset repository: {repo}")
        kwargs["revision"] = source["repositories"][repo]["revision"]
        kwargs["cache_dir"] = str(data_dir / "cache")
        kwargs.pop("trust_remote_code", None)
        result = original(*args, **kwargs)
        if budget_bytes is not None and storage_bytes(data_dir) > budget_bytes:
            raise RuntimeError("Dataset directory exceeded the configured byte budget; stopping further downloads")
        return result
    datasets.load_dataset = load
    try:
        yield
    finally:
        datasets.load_dataset = original


def definition_fingerprint():
    import lm_eval
    root = Path(lm_eval.__file__).parent / "tasks"
    files = {}
    for subdir in ("mmlu", "gsm8k", "hellaswag", "arc"):
        for path in sorted((root / subdir).rglob("*")):
            if path.is_file() and path.suffix in {".yaml", ".py", ""}:
                files[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"sha256": digest(files), "files": files}


def load_tasks(tasks, source, data_dir, budget_bytes=None):
    from lm_eval.tasks import TaskManager
    with pinned_datasets(source, data_dir, budget_bytes):
        loaded = TaskManager().load(tasks)
    for name, task in loaded["tasks"].items():
        if "task_info" in source:
            actual = {k: getattr(v, "_fingerprint", None) for k, v in task.dataset.items()}
            if actual != source["task_info"][name]["dataset_fingerprints"]:
                raise RuntimeError(f"Offline dataset fingerprint changed for {name}")
        task.set_config(key="num_fewshot", value=SHOTS[family(name)])
        task.set_fewshot_seed(seed=source["seed"])
    return loaded


def mc_question(task_name, task, doc):
    if family(task_name) == "mmlu":
        question, choices = doc["question"].strip(), list(doc["choices"])
    else:
        question = str(task.doc_to_text(doc))
        question = re.sub(r"\nAnswer:\s*$", "", question)
        choices = list(task.doc_to_choice(doc))
    if not 2 <= len(choices) <= 26:
        raise ValueError("Unsupported number of MC options")
    target = int(task.doc_to_target(doc))
    if not 0 <= target < len(choices):
        raise ValueError("MC target outside choice range")
    rendered = question + "\n" + "\n".join(f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices))
    return rendered, chr(65+target), len(choices)


def adaptation_rows(name, task, limit):
    """Explicit adaptation; these scores are not the official MC benchmark."""
    count = 5 if family(name) == "mmlu" else 0
    demonstrations = []
    if count:
        from itertools import islice
        # MMLU official first_n sampler uses the five dev examples.
        demonstrations = list(islice(task.fewshot_docs(), count))
        if len(demonstrations) != count:
            raise ValueError(f"{name} lacks five development demonstrations")
    intro = ("Choose the best answer to the multiple-choice question. Briefly explain your reasoning, "
             "then write a final line exactly as 'Final answer: X', replacing X with one option letter.\n\n")
    prefix = intro
    for example in demonstrations:
        question, target, _ = mc_question(name, task, example)
        prefix += question + f"\nFinal answer: {target}\n\n"
    for index, doc in enumerate(task.eval_docs):
        if limit is not None and index >= limit:
            break
        question, target, options = mc_question(name, task, doc)
        prompt = prefix + question + "\nExplanation:"
        yield {"id": f"{name}:{index}", "task": name, "family": family(name), "doc_id": index,
               "prompt": prompt, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
               "doc_sha256": digest(plain(doc)), "target": target, "num_options": options,
               "num_fewshot": count, "fewshot_doc_sha256": [digest(plain(x)) for x in demonstrations],
               "mode": "generation_adaptation", "first_answer_token_is_prefill": True}


def prepare(args):
    # This subcommand cannot see CUDA even if the user has an idle GPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    path = data_environment(args.data_dir, offline=False)
    path.mkdir(parents=True, exist_ok=False)
    if importlib.metadata.version("lm_eval") != "0.4.13":
        raise RuntimeError("Preparation requires inspected lm_eval 0.4.13")
    from huggingface_hub import HfApi
    source = {"status": "PREPARING", "lm_eval": "0.4.13", "seed": args.seed,
              "tasks": args.tasks, "prepare_limit_per_leaf": args.limit,
              "transport_endpoint": os.environ.get("HF_ENDPOINT", "https://huggingface.co"),
              "byte_budget": args.max_mib*1024**2, "repositories": {},
              "task_definitions": definition_fingerprint(), "fewshot": SHOTS,
              "scope": "official task cache plus separate generation adaptations; no model weights downloaded"}
    atomic_json(path / "source.json", source)
    try:
        api = HfApi()
        for repo in sorted({REPOS[name] for name in args.tasks}):
            info = api.dataset_info(repo, revision="main", timeout=30)
            if not re.fullmatch(r"[0-9a-f]{40}", info.sha):
                raise ValueError("Dataset revision must resolve to a full immutable commit")
            source["repositories"][repo] = {"revision": info.sha,
                "url": f"https://huggingface.co/datasets/{repo}/tree/{info.sha}"}
        atomic_json(path / "source.json", source)
        loaded = load_tasks(args.tasks, source, path, source["byte_budget"])
        task_info, adaptation_counts = {}, {}
        export = path / "generation_adaptation.jsonl"
        export.touch(exist_ok=False)
        for name, task in sorted(loaded["tasks"].items()):
            task_info[name] = {"dataset_path": task.DATASET_PATH, "dataset_name": task.DATASET_NAME,
                "output_type": task.OUTPUT_TYPE, "evaluation_documents": len(task.eval_docs),
                "num_fewshot": SHOTS[family(name)], "config": plain(task.dump_config()),
                "dataset_fingerprints": {k: getattr(v, "_fingerprint", None) for k, v in task.dataset.items()}}
            if task.OUTPUT_TYPE == "multiple_choice":
                total = 0
                for row in adaptation_rows(name, task, args.limit):
                    append_json(export, row)
                    total += 1
                adaptation_counts[name] = total
        source.update(status="READY", task_info=task_info, adaptation_counts=adaptation_counts,
                      adaptation_sha256=hashlib.sha256(export.read_bytes()).hexdigest(),
                      storage_bytes=storage_bytes(path))
        if source["storage_bytes"] > source["byte_budget"]:
            raise RuntimeError("Dataset preparation exceeded its byte budget")
    except Exception:
        source.update(status="ERROR", error=traceback.format_exc(), storage_bytes=storage_bytes(path))
        raise
    finally:
        atomic_json(path / "source.json", source)
        (path / "source.sha256").write_text(
            hashlib.sha256((path / "source.json").read_bytes()).hexdigest() + "  source.json\n")
    print(json.dumps({"status": source["status"], "data_dir": str(path),
                      "bytes": source["storage_bytes"], "adaptation_counts": adaptation_counts}))


def read_source(data_dir):
    raw = (data_dir / "source.json").read_bytes()
    checksum = data_dir / "source.sha256"
    if checksum.exists() and checksum.read_text().split()[0] != hashlib.sha256(raw).hexdigest():
        raise ValueError("source.json checksum mismatch")
    source = json.loads(raw)
    if source["status"] != "READY":
        raise ValueError("Dataset preparation is not READY")
    return source


def score_adaptation(source, output):
    matches = list(re.finditer(r"(?im)^\s*Final answer:\s*([A-Z])\s*[.!]?\s*$", output["text"]))
    answer = matches[-1].group(1).upper() if matches else None
    valid = answer is not None and len(answer) == 1 and 0 <= ord(answer)-65 < source["num_options"]
    correct = output["termination"] != "fail" and valid and answer == source["target"]
    return {"task": source["task"], "family": source["family"], "doc_id": source["doc_id"],
            "mode": "generation_adaptation", "target": source["target"], "parsed_answer": answer,
            "answer_parse_valid": valid, "correct": bool(correct),
            "parse_rule": "last complete Final answer: LETTER line; no fallback to arbitrary letters",
            "answer_char_start": matches[-1].start(1) if matches else None}


def adapted(args):
    validate_runtime(args)
    data_dir = data_environment(args.data_dir, offline=True)
    source = read_source(data_dir)
    if not set(args.tasks) <= set(source["tasks"]):
        raise ValueError("Requested tasks were not prepared")
    raw = (data_dir / "generation_adaptation.jsonl").read_bytes()
    if hashlib.sha256(raw).hexdigest() != source["adaptation_sha256"]:
        raise ValueError("Generation adaptation data hash mismatch")
    rows = []
    counts = defaultdict(int)
    for line in raw.decode().splitlines():
        row = json.loads(line)
        if row["family"] not in args.tasks:
            continue
        if args.limit is None or counts[row["task"]] < args.limit:
            rows.append(row)
            counts[row["task"]] += 1
    if not rows or "gsm8k" in args.tasks:
        raise ValueError("Adapted mode supports MMLU/HellaSwag/ARC-c; use standard for GSM8K")
    arms, provenance = resolve_arms(args.arms_file)
    kwargs = model_kwargs(args)
    versions = check_versions()
    args.output.mkdir(parents=True, exist_ok=False)
    atomic_json(args.output / "config.json", {"args": plain(vars(args)), "versions": versions,
        "model_kwargs": kwargs, "arm_source": provenance, "arms": arms,
        "dataset_source": source, "selected_per_leaf": dict(counts),
        "scope": "nonstandard generation adaptation; do not label as official MC acc/acc_norm",
        "first_token_caveat": "first prediction is prefill; a one-token answer cannot test decode intervention",
        "actual_sample_scope": "prepared subset" if source.get("prepare_limit_per_leaf") else "full prepared tasks"})
    from vllm import LLM
    llm = LLM(**kwargs)
    rows = tokenize_prompts(rows, llm.get_tokenizer(), args.max_model_len, args.max_tokens)
    atomic_json(args.output / "prompts.json", rows)
    args.temperature, args.top_p = 0., 1.
    results, summary = run_pairs(llm, rows, arms, args, args.output, scorer=score_adaptation)
    scores = {}
    for arm, outputs in results.items():
        scores[arm] = {}
        for task_family in sorted({r["family"] for r in rows}):
            selected = [r for r in outputs if r["family"] == task_family]
            scores[arm][task_family] = {"samples": len(selected),
                "micro_accuracy": statistics.mean(r["correct"] for r in selected),
                "unparseable": sum(not r["answer_parse_valid"] for r in selected),
                "first_prediction_only": sum(r["first_prediction_only"] for r in selected),
                "scope": "generation adaptation; includes length-limit completions; failures incorrect"}
    atomic_json(args.output / "task_scores.json", scores)


def sample_signature(result):
    return {task: [(row.get("doc_id"), row.get("doc_hash"), row.get("prompt_hash"), row.get("target_hash"))
                    for row in rows] for task, rows in result["samples"].items()}


def task_micro_metrics(result):
    """Sample-weighted aggregate across subjects, keeping extraction filters apart."""
    values = defaultdict(list)
    for rows in result["samples"].values():
        for row in rows:
            for metric in row.get("metrics", []):
                value = row.get(metric)
                if isinstance(value, (int, float, bool)):
                    values[f"{metric},{row.get('filter', 'none')}"].append(value)
    return {key: {"mean": statistics.mean(vals), "sample_rows": len(vals)}
            for key, vals in values.items()}


def close_nested(left, right, atol=1e-6):
    if isinstance(left, (float, int)) and isinstance(right, (float, int)):
        return math.isclose(left, right, rel_tol=0., abs_tol=atol)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(close_nested(a, b, atol) for a, b in zip(left, right))
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(close_nested(left[k], right[k], atol) for k in left)
    return left == right


def standard(args):
    validate_runtime(args)
    data_dir = data_environment(args.data_dir, offline=True)
    source = read_source(data_dir)
    if not set(args.tasks) <= set(source["tasks"]):
        raise ValueError("Requested tasks were not prepared")
    kwargs = model_kwargs(args)
    versions = check_versions(require_lm_eval=True)
    if definition_fingerprint()["sha256"] != source["task_definitions"]["sha256"]:
        raise RuntimeError("Installed task definitions differ from the pinned preparation")
    arms, provenance = resolve_arms(args.arms_file)
    loaded = load_tasks(args.tasks, source, data_dir)
    # Freeze the exact task objects and data in memory; recreate evaluator
    # requests with the same seeds for each arm and verify prompt/sample hashes.
    from lm_eval import evaluator
    from lm_eval.models.vllm_causallms import VLLM
    args.output.mkdir(parents=True, exist_ok=False)
    atomic_json(args.output / "config.json", {"args": plain(vars(args)), "versions": versions,
        "model_kwargs": kwargs, "arms": arms, "arm_source": provenance,
        "dataset_source": source,
        "semantics": {"mmlu": "official 5-shot prompt-loglikelihood identity control",
                      "hellaswag": "official 0-shot prompt-loglikelihood identity control",
                      "arc_challenge": "official 0-shot prompt-loglikelihood identity control",
                      "gsm8k": "official 5-shot generation task; max output tokens explicitly recorded"},
        "sample_limit": "per leaf task; applies separately to each MMLU subject",
        "normalization": "preserve model router default; decode-only policy; no model downloads"})

    class ObservedVLLM(VLLM):
        arm = None
        calls = 0
        def loglikelihood(self, requests, **kw):
            for request in requests:
                if len(self.tok_encode(request.args[0] + request.args[1])) > self.max_length-1:
                    raise ValueError("Standard task request exceeds context; no silent truncation allowed")
            return super().loglikelihood(requests, **kw)
        def generate_until(self, requests, **kw):
            for request in requests:
                limit = request.args[1].get("max_gen_toks", self.max_gen_toks)
                if len(self.tok_encode(request.args[0])) + limit > self.max_length:
                    raise ValueError("Standard generation request exceeds context; no silent truncation allowed")
            return super().generate_until(requests, **kw)
        def _model_generate(self, requests, generate=False, sampling_params=None):
            if generate:
                params = sampling_params if isinstance(sampling_params, list) else [sampling_params]
                if any(p is None or p.ignore_eos or p.min_tokens > 0 for p in params):
                    raise RuntimeError("Task generation must permit natural EOS without minimum length")
            one_worker(self.model, "skip_snapshot", True)
            outputs = super()._model_generate(requests, generate, sampling_params)
            snap = one_worker(self.model, "skip_snapshot")
            tag = f"{self.arm}.call{self.calls:06d}"
            self.calls += 1
            atomic_json(args.output / f"{tag}.worker.json", snap)
            for tokens, output in zip(requests, outputs, strict=True):
                comp = output.outputs[0]
                record = {"arm": self.arm, "call": tag, "generate": generate,
                    "prompt_token_sha256": digest(tokens), "prompt_tokens": len(tokens),
                    "text": comp.text, "token_ids": list(comp.token_ids),
                    "output_tokens": len(comp.token_ids), "finish_reason": comp.finish_reason,
                    "first_prediction_only": len(comp.token_ids) <= 1,
                    "stop_reason": comp.stop_reason,
                    "termination": finish_category(comp.finish_reason, comp.stop_reason,
                                                    list(comp.token_ids), {self.eot_token_id}),
                    "scope": "generation" if generate else "discarded one-token prompt-logprob request"}
                append_json(args.output / "raw_outputs.jsonl", record)
            if not generate and (snap["phase_steps"].get("decode", 0) or snap["phase_steps"].get("mixed", 0)):
                raise RuntimeError("Loglikelihood identity control unexpectedly used decode/mixed execution")
            return outputs

    adapter_kwargs = dict(kwargs)
    pretrained = adapter_kwargs.pop("model")
    model = ObservedVLLM(pretrained=pretrained, batch_size=args.batch, data_parallel_size=1,
                        max_gen_toks=args.max_tokens, **adapter_kwargs)
    baseline, results = {}, {}
    manifest = {"status": "RUNNING", "evaluations": [], "task_request_semantics": "MC loglikelihood is a zero-decode identity control; GSM8K is generation"}
    atomic_json(args.output / "summary.json", manifest)
    try:
        for task_family in args.tasks:
            task_objects = [task for name, task in loaded["tasks"].items() if family(name) == task_family]
            for arm, spec in arms:
                model.arm = f"{task_family}.{arm}"
                installation = one_worker(model.model, "skip_set_policy", spec, "decode", 0, False)
                for task in task_objects:
                    task.set_fewshot_seed(seed=args.seed)
                result = evaluator.simple_evaluate(model=model, tasks=task_objects,
                    num_fewshot=SHOTS[task_family], batch_size=args.batch, limit=args.limit,
                    log_samples=True, bootstrap_iters=0, use_cache=None, cache_requests=False,
                    apply_chat_template=False, gen_kwargs={"max_gen_toks": args.max_tokens, "do_sample": False, "temperature": 0.},
                    random_seed=args.seed, numpy_random_seed=args.seed,
                    torch_random_seed=args.seed, fewshot_random_seed=args.seed)
                if result is None or not result.get("samples"):
                    raise RuntimeError("Harness returned no per-sample outputs")
                result = plain(result)
                tag = f"{task_family}.{arm}"
                atomic_json(args.output / f"{tag}.results.json", result)
                atomic_json(args.output / f"{tag}.installation.json", installation)
                if arm == "native":
                    baseline[task_family] = result
                else:
                    before = baseline[task_family]
                    if sample_signature(result) != sample_signature(before):
                        raise RuntimeError("Native/candidate task samples or prompt hashes differ")
                    identity_expected = task_family != "gsm8k" or (spec or {}).get("mode") == "dense"
                    if identity_expected:
                        for task_name, rows in result["samples"].items():
                            for row, old in zip(rows, before["samples"][task_name], strict=True):
                                if not close_nested(row["resps"], old["resps"]):
                                    raise RuntimeError("Expected task identity control changed responses/loglikelihood")
                entry = {"task": task_family, "arm": arm, "file": f"{tag}.results.json",
                         "results": result["results"], "sample_weighted_metrics": task_micro_metrics(result),
                         "control_only": task_family != "gsm8k"}
                manifest["evaluations"].append(entry)
                atomic_json(args.output / "summary.json", manifest)
        manifest["status"] = "COMPLETE"
    except Exception:
        manifest.update(status="ERROR", error=traceback.format_exc())
        raise
    finally:
        atomic_json(args.output / "summary.json", manifest)


def self_test():
    from types import SimpleNamespace as N
    task = N(doc_to_target=lambda d: d["answer"])
    doc = {"question": "2+2?", "choices": ["3", "4", "5", "6"], "answer": 1}
    question, target, count = mc_question("mmlu_test", task, doc)
    assert "B. 4" in question and target == "B" and count == 4
    source = {"task": "mmlu_test", "family": "mmlu", "doc_id": 0, "target": "B", "num_options": 4}
    assert score_adaptation(source, {"text": "2+2 is 4.\nFinal answer: B", "termination": "eos"})["correct"]
    assert score_adaptation(source, {"text": "Final answer: b", "termination": "eos"})["correct"]
    assert not score_adaptation(source, {"text": "B", "termination": "eos"})["correct"]
    assert not score_adaptation(source, {"text": "Final answer: Z", "termination": "limit"})["answer_parse_valid"]
    assert close_nested([[.1, True]], [[.1+1e-8, True]])
    assert not close_nested([[.1]], [[.2]])
    assert sample_signature({"samples": {"x": [{"doc_id": 0, "prompt_hash": "a"}]}}) != sample_signature({"samples": {"x": [{"doc_id": 0, "prompt_hash": "b"}]}})
    print("CPU adaptation rendering/scoring, prompt identity and numerical identity checks passed")


def main():
    import sys
    if sys.argv[1:] == ["--self-test"]:
        self_test()
        return
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    prep = modes.add_parser("prepare", help="CPU-only task dataset preparation, no model loading")
    prep.add_argument("--data-dir", type=Path, required=True)
    prep.add_argument("--tasks", nargs="+", choices=list(REPOS), default=list(REPOS))
    prep.add_argument("--limit", type=int, help="Export only N adaptation docs per leaf; official cache remains full")
    prep.add_argument("--seed", type=int, default=1701)
    prep.add_argument("--max-mib", type=int, default=2048)
    for mode in ("standard", "adapted"):
        sub = modes.add_parser(mode)
        add_runtime_args(sub)
        sub.add_argument("--data-dir", type=Path, required=True)
        sub.add_argument("--tasks", nargs="+", choices=list(REPOS),
                         default=list(REPOS) if mode == "standard" else ["mmlu", "hellaswag", "arc_challenge"])
        sub.add_argument("--limit", type=int, help="Exploratory cap per leaf task; unset means all prepared task samples")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.mode == "prepare":
        if not 0 < args.max_mib <= 2048:
            parser.error("Preparation budget must be in (0, 2048] MiB")
        prepare(args)
    elif args.mode == "standard":
        standard(args)
    else:
        adapted(args)


if __name__ == "__main__":
    main()
