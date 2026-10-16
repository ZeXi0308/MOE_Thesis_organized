#!/usr/bin/env python3
"""Prepare the full fixed lm-eval suite using CPU only, without loading a model.

Run in the existing tools_venv (lm-eval 0.4.13, datasets 5.x):
  HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 \
    python code/prepare_quality_tasks.py --output runs/quality_data_setup/preload_r01

Each task runs separately, preserving failures and continuing the other tasks.
No dataset substitutions, package changes, sampling, or model downloads occur.
The task YAML, resolved configurations, full split hashes, cache files, and all
evaluation prompt/target/choice hashes are recorded. This prepares data only;
it does not measure model accuracy or promise offline availability of every Hub
metadata request. Use the same tasks/configuration in eval_quality.py.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

# Set before any harness/torch import, also inherited by every worker process.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
TASKS = ("hellaswag", "arc_easy", "arc_challenge", "winogrande", "piqa")
REGISTRY = Path(__file__).resolve().parents[1] / "configs" / "models.json"


def json_default(value):
    if callable(value):
        return {"callable": f"{value.__module__}.{value.__qualname__}"}
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (set, tuple)):
        return list(value)
    raise TypeError(f"unserializable value: {type(value)}")


def encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=json_default,
                      allow_nan=False, separators=(",", ":")).encode()


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    default=json_default, allow_nan=False) + "\n")
    temporary.replace(path)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def versions():
    result = {}
    for name in ("lm_eval", "datasets", "huggingface-hub", "pyarrow", "torch",
                 "transformers"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def describe_dataset(dataset):
    """Hash every row in order, independently of datasets' cached fingerprint."""
    digest = hashlib.sha256()
    count = 0
    for row in dataset:
        digest.update(encoded(row) + b"\n")
        count += 1
    if count != len(dataset):
        raise RuntimeError("dataset iteration did not cover the full split")
    cache = []
    for entry in dataset.cache_files:
        path = Path(entry["filename"]).resolve()
        cache.append({"path": str(path), "bytes": path.stat().st_size,
                      "sha256": file_hash(path)})
    info = dataset.info
    return {"rows": count, "datasets_fingerprint": dataset._fingerprint,
            "ordered_rows_sha256": digest.hexdigest(),
            "hash_encoding": "ordered UTF-8 canonical JSON rows, LF terminated",
            "features": dataset.features.to_dict(), "cache_files": cache,
            "builder_name": info.builder_name, "config_name": info.config_name,
            "dataset_version": str(info.version),
            "download_checksums": info.download_checksums,
            "download_size": info.download_size, "dataset_size": info.dataset_size}


def source_snapshot(manager, name, folder):
    entry = manager.task_index[name]
    yaml_path = Path(entry.yaml_path)
    sources = {}
    # Snapshot the task directory's small YAML/Python sources, including include
    # targets (ARC-Challenge inherits ARC-Easy) and formatting helper functions.
    for path in sorted(yaml_path.parent.iterdir()):
        if path.suffix in (".yaml", ".yml", ".py") and path.is_file():
            content = path.read_bytes()
            (folder / path.name).write_bytes(content)
            sources[path.name] = {"installed_path": str(path), "bytes": len(content),
                                  "sha256": hashlib.sha256(content).hexdigest()}
    return {"entry_yaml_path": str(yaml_path), "sources": sources,
            "resolved_yaml_config": entry.cfg,
            "yaml_num_fewshot": entry.cfg.get("num_fewshot"),
            "yaml_num_fewshot_explicit": "num_fewshot" in entry.cfg}


def worker(name, output):
    folder = Path(output)
    record = {"task": name, "status": "running", "start_unix_s": time.time(),
              "versions": versions(), "cpu_only": True, "model_loaded": False,
              "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"]}
    record_path = folder / "task.json"
    save(record_path, record)
    try:
        if record["versions"]["lm_eval"] != "0.4.13":
            raise RuntimeError("this task API/configuration is pinned to lm-eval 0.4.13")
        if not (record["versions"]["datasets"] or "").startswith("5."):
            raise RuntimeError("this compatibility preparation requires the existing datasets 5.x environment")
        from lm_eval.tasks import TaskManager
        import datasets

        record["dataset_cache_root"] = str(datasets.config.HF_DATASETS_CACHE)
        manager = TaskManager()
        record["task_sources"] = source_snapshot(manager, name, folder)
        save(record_path, record)
        # Actual 0.4.13 API: .load returns {'tasks': {name: Task}, ...}.
        loaded = manager.load(name)
        if set(loaded["tasks"]) != {name}:
            raise RuntimeError(f"unexpected task expansion: {list(loaded['tasks'])}")
        task = loaded["tasks"][name]
        record["harness_default_config"] = task.dump_config()
        task.set_config("num_fewshot", 0)
        record["effective_task_config"] = task.dump_config()
        record["effective_config_sha256"] = hashlib.sha256(encoded(record["effective_task_config"])).hexdigest()
        record["splits"] = {}
        for split, dataset in task.dataset.items():
            record["splits"][split] = describe_dataset(dataset)
            save(record_path, record)
        if task.has_test_docs():
            evaluation_split = task.get_config("test_split")
            docs = task.test_docs()
        elif task.has_validation_docs():
            evaluation_split = task.get_config("validation_split")
            docs = task.validation_docs()
        else:
            raise RuntimeError("task has no evaluation split")
        record["evaluation_split"] = evaluation_split
        record["evaluation_docs"] = describe_dataset(docs)
        formatted = hashlib.sha256()
        choices_histogram = {}
        for row in docs:
            choices = task.doc_to_choice(row)
            target = task.doc_to_target(row)
            text = task.doc_to_text(row)
            if not choices or not isinstance(target, int) or not 0 <= target < len(choices):
                raise RuntimeError(f"invalid multiple-choice target {target!r} for {len(choices)} choices")
            key = str(len(choices))
            choices_histogram[key] = choices_histogram.get(key, 0) + 1
            formatted.update(encoded({"text": text, "choices": choices, "target": target}) + b"\n")
        record["formatted_evaluation"] = {"rows": len(docs), "ordered_sha256": formatted.hexdigest(),
                                           "choices_count_histogram": choices_histogram,
                                           "all_targets_valid": True,
                                           "num_fewshot": 0, "apply_chat_template": False}
        torch = sys.modules.get("torch")
        initialized = bool(torch and torch.cuda.is_initialized())
        record["torch_cuda_initialized"] = initialized
        if initialized:
            raise RuntimeError("unexpected CUDA initialization during CPU data preparation")
        record["status"] = "complete"
    except Exception as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc())
    finally:
        record["end_unix_s"] = time.time()
        save(record_path, record)
    return 0 if record["status"] == "complete" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task-timeout-s", type=int, default=600,
                        help="per-task CPU/download deadline; timeout is preserved as failure")
    parser.add_argument("--worker", choices=TASKS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return worker(args.worker, args.output)
    if args.task_timeout_s <= 0:
        parser.error("--task-timeout-s must be positive")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    defaults = json.loads(REGISTRY.read_text())["quality_defaults"]
    if tuple(defaults["tasks"]) != TASKS or defaults["num_fewshot"] != 0 or defaults["apply_chat_template"]:
        raise RuntimeError("registry differs from the fixed five-task zero-shot raw-prompt suite")
    configuration = {"tasks": list(TASKS), "num_fewshot": 0, "limit": None,
                     "apply_chat_template": False, "fewshot_as_multiturn": False,
                     "random_seed": defaults["seed"], "numpy_random_seed": defaults["seed"],
                     "torch_random_seed": defaults["seed"], "fewshot_random_seed": defaults["seed"],
                     "batch_size": defaults["batch_size"], "bootstrap_iters": 1000,
                     "log_samples": True}
    save(output / "evaluation_config.json", configuration)
    receipt = {"schema_version": 1, "status": "running", "start_unix_s": time.time(),
               "purpose": "full evaluation-data preparation only; no model/quality measurement",
               "versions": versions(), "script_sha256": file_hash(__file__),
               "evaluation_config_sha256": hashlib.sha256(encoded(configuration)).hexdigest(),
               "evaluation_config": configuration, "registry_sha256": file_hash(REGISTRY),
               "task_timeout_s": args.task_timeout_s,
               "environment": {key: os.environ.get(key) for key in (
                   "HF_ENDPOINT", "HF_HUB_DISABLE_XET", "HF_HOME", "HF_DATASETS_CACHE",
                   "HF_HUB_CACHE", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES")}, "tasks": {}}
    save(output / "receipt.json", receipt)
    for name in TASKS:
        folder = output / name
        folder.mkdir()
        with (folder / "prepare.log").open("w") as log:
            try:
                completed = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                    "--output", str(folder), "--worker", name], stdout=log, stderr=subprocess.STDOUT,
                    timeout=args.task_timeout_s, check=False)
                exit_code = completed.returncode
                timeout = False
            except subprocess.TimeoutExpired:
                exit_code, timeout = None, True
        record_path = folder / "task.json"
        record = json.loads(record_path.read_text()) if record_path.exists() else {"task": name}
        if timeout or record.get("status") not in ("complete", "failed"):
            record.update(status="failed", error_type="TimeoutExpired" if timeout else "WorkerExit",
                          error=f"worker deadline {args.task_timeout_s}s exceeded" if timeout else f"worker exit {exit_code}",
                          end_unix_s=time.time())
            save(record_path, record)
        receipt["tasks"][name] = {"status": record["status"], "worker_exit_code": exit_code,
                                  "record": str(record_path), "record_sha256": file_hash(record_path),
                                  "evaluation_split": record.get("evaluation_split"),
                                  "evaluation_rows": record.get("evaluation_docs", {}).get("rows"),
                                  "error_type": record.get("error_type"), "error": record.get("error")}
        save(output / "receipt.json", receipt)
        print(json.dumps({"task": name, **receipt["tasks"][name]}, ensure_ascii=False), flush=True)
    receipt["end_unix_s"] = time.time()
    receipt["status"] = "complete" if all(r["status"] == "complete" for r in receipt["tasks"].values()) else "partial_failure"
    save(output / "receipt.json", receipt)
    print(json.dumps({"status": receipt["status"], "receipt": str(output / "receipt.json")}), flush=True)
    return 0 if receipt["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
