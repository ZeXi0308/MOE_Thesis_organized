#!/usr/bin/env python3
"""Run lm-eval against the exact checkpoint used for serving/performance.

Default is a CPU-only plan. Add --run inside the shared GPU-lock launcher.
  python eval_quality.py --checkpoint-manifest /data/olmoe-identity.json --format bf16 --output runs/quality/olmoe-bf16
  python eval_quality.py --checkpoint-manifest /data/qwen-awq/checkpoint_manifest.json --output runs/quality/qwen-awq --run

Uses the official lm_eval.simple_evaluate API, avoiding changes between the old
``lm_eval --model ...`` and new ``lm-eval run --model ...`` command interfaces.
Interface references are in configs/models.json. Requires lm-eval's vLLM backend
already installed; this runner does not install or download model weights.
Task datasets may be downloaded by lm-eval at execution time. ``--limit`` means
smoke-only and is explicitly recorded, not a complete quality-table result.

--worker-config installs real per-layer sidecars with DualFormatWorker. It forces
eager execution and requires --lock-status for --run. The host checkpoint keeps
its own nonexpert precision; selecting FP8 sidecars does not quantize that host.
Single policies infer bf16/fp8/awq/nvfp4; dual policies infer dual_fp8_awq or
dual_fp8_nvfp4. Example JSONs in configs/quality_worker_*.example.json are layer
templates: supply real artifacts/basis IDs and enumerate EVERY MoE layer.
The default five tasks score prompt loglikelihood (prefill), not mixed-decode
generation quality. No model/worker is imported by the read-only plan path.

--generation-config configs/quality_generation_gsm8k_128.json selects a separate
128-example held-out generate_until diagnostic, with official gold scoring and
full termination records. --prepare-generation-data prepares that pinned task
on CPU before a locked run; it may download only its task dataset. No generation
occurs in preparation. Do not combine this protocol with --tasks or --limit.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import re
import signal
import time
from collections import Counter, defaultdict, deque

from prepare_quantization import REGISTRY, read_json, sha256, validate_config, write_json


WORKER_CLASS = "dual_format_worker.DualFormatWorker"
MC_TASKS = {"hellaswag", "arc_easy", "arc_challenge", "winogrande", "piqa"}


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def generation_plan(path):
    """One pinned, gold-scored held-out task; no dataset/model access here."""
    config = read_json(path)
    required = {"schema_version", "task", "task_yaml_sha256", "dataset_revision",
                "expected_test_size", "sample_count", "selection_seed", "max_gen_toks",
                "batch_size", "wall_budget_s", "apply_chat_template", "enable_thinking"}
    if set(config) != required or config["schema_version"] != 1:
        raise ValueError(f"generation config requires exactly {sorted(required)}")
    if config["task"] != "gsm8k_cot_zeroshot" or config["expected_test_size"] != 1319:
        raise ValueError("this bounded generation entry supports the GSM8K zero-shot test task only")
    for key, high in (("sample_count", 1319), ("max_gen_toks", 1024), ("batch_size", 128), ("wall_budget_s", 7200)):
        if type(config[key]) is not int or not 1 <= config[key] <= high:
            raise ValueError(f"invalid generation {key}")
    if type(config["selection_seed"]) is not int or config["enable_thinking"] is not False or config["apply_chat_template"] is not True:
        raise ValueError("generation requires an integer selection seed, native chat template and thinking disabled")
    if not re.fullmatch(r"[0-9a-f]{40}", config["dataset_revision"]) or not re.fullmatch(r"[0-9a-f]{64}", config["task_yaml_sha256"]):
        raise ValueError("generation requires pinned dataset revision and installed task YAML hash")
    config["sample_ids"] = sorted(random.Random(config["selection_seed"]).sample(
        range(config["expected_test_size"]), config["sample_count"]))
    config["maximum_generated_tokens_per_arm"] = config["sample_count"] * config["max_gen_toks"]
    config["source_config"] = str(Path(path).resolve())
    config["source_config_sha256"] = sha256(path)
    config["scope"] = "fixed held-out generation diagnostic, not the full GSM8K benchmark; held out from this project's calibration/selection, not certified absent from pretraining"
    return config


class GenerationLedger:
    """Persist every selected example before model construction, including failures."""
    def __init__(self, output, spec):
        self.output, self.spec = Path(output), spec
        self.rows = {i: {"task": spec["task"], "doc_id": i, "status": "not_started"}
                     for i in spec["sample_ids"]}
        self.state, self.error, self.metrics = "preparing", None, None
        self.save()

    def save(self):
        records = list(self.rows.values())
        counts = dict(Counter(row["status"] for row in records))
        completed = sum(counts.get(key, 0) for key in ("eos", "stop_string", "stop_token", "stop_unspecified", "cap"))
        summary = {"status": self.state, "selected_population": len(records),
            "termination_counts": counts, "completed_population": completed,
            "uncompleted_population": len(records) - completed,
            "output_tokens_observed": sum(len(row.get("output_token_ids", [])) for row in records),
            "multi_token_outputs": sum(len(row.get("output_token_ids", [])) > 1 for row in records),
            "sample_ids": self.spec["sample_ids"], "maximum_generated_tokens_per_arm": self.spec["maximum_generated_tokens_per_arm"],
            "metrics": self.metrics, "error": self.error,
            "scoring_denominator": len(records),
            "failure_rule": "Never report a success-only score. Engine/timeout/unfinished failures invalidate the aggregate; all selected records remain. Capped and unparseable completed outputs stay in official task scoring.",
            "scope": self.spec["scope"]}
        # Atomic snapshots also leave a known full population after SIGKILL.
        for name, value in (("generation_records.json", records), ("generation_summary.json", summary)):
            path = self.output / name
            temporary = path.with_suffix(".tmp")
            write_json(temporary, value)
            temporary.replace(path)

    def fail(self, error):
        self.state, self.error = "failed", f"{type(error).__name__}: {error}"
        self.metrics = None
        for row in self.rows.values():
            if row["status"] == "in_flight":
                row.update(status="timeout" if isinstance(error, TimeoutError) else "error", error=self.error)
            elif row["status"] == "not_started":
                row.update(status="unfinished", error="not executed before run failure")
        self.save()


def prepare_generation_task(spec, ledger):
    """Called only under --run; lm-eval may fetch this one pinned dataset."""
    import importlib.util
    import yaml
    from lm_eval.tasks import TaskManager
    if importlib.metadata.version("lm_eval") != "0.4.13":
        raise RuntimeError("generation adapter targets inspected lm_eval 0.4.13")
    task_path = Path(importlib.util.find_spec("lm_eval").origin).parent / "tasks/gsm8k/gsm8k-cot-zeroshot.yaml"
    if sha256(task_path) != spec["task_yaml_sha256"]:
        raise ValueError("installed generation task YAML differs from the pinned scoring protocol")
    config = yaml.safe_load(task_path.read_text())
    config["dataset_kwargs"] = {**config.get("dataset_kwargs", {}), "revision": spec["dataset_revision"]}
    config["generation_kwargs"] = {**config["generation_kwargs"], "max_gen_toks": spec["max_gen_toks"],
                                    "do_sample": False, "temperature": 0.0}
    manager = TaskManager()
    task = manager.load([config])["tasks"][spec["task"]]
    docs = task.eval_docs
    if len(docs) != spec["expected_test_size"] or task.config.output_type != "generate_until":
        raise ValueError("test population or task type changed")
    selected = []
    for index in spec["sample_ids"]:
        doc = dict(docs[index])
        record = {"doc_id": index, "doc": doc, "doc_sha256": digest_json(doc),
                  "gold": task.doc_to_target(doc)}
        ledger.rows[index].update(record)
        selected.append(record)
    metadata = {"dataset": "openai/gsm8k", "config": "main", "split": "test",
                "revision": spec["dataset_revision"], "dataset_fingerprint": getattr(docs, "_fingerprint", None),
                "task_yaml_sha256": sha256(task_path), "task_config": config,
                "selection": spec, "selected_records_sha256": digest_json(selected), "records": selected}
    write_json(ledger.output / "generation_population.json", metadata)
    ledger.state = "model_initialization"
    ledger.save()
    return manager, task


def recording_vllm(base_class, ledger):
    """Use the installed harness tokenization/stops/scorer; observe real outputs."""
    class RecordedVLLM(base_class):
        def generate_until(self, requests, disable_tqdm=False):
            requests = list(requests)
            by_doc = defaultdict(deque)
            for row in ledger.rows.values():
                by_doc[row["doc_sha256"]].append(row)
            self._quality_pending_tokens = defaultdict(deque)
            seen = set()
            encodings = self.tok_encode([req.args[0] for req in requests])
            for req, tokens in zip(requests, encodings, strict=True):
                key = digest_json(req.doc)
                if not by_doc[key] or req.task_name != ledger.spec["task"]:
                    raise ValueError("harness request does not match the frozen generation population")
                row = by_doc[key].popleft()
                seen.add(row["doc_id"])
                row.update(prompt=req.args[0], generation_kwargs=req.args[1],
                           harness_doc_id=req.doc_id, prompt_token_ids=list(tokens), prompt_tokens=len(tokens))
                if len(tokens) + ledger.spec["max_gen_toks"] > self.max_length:
                    row.update(status="error", error="context budget exceeded; no truncation, filtering or resampling permitted")
                    ledger.save()
                    raise ValueError("selected generation prompt exceeds context budget; retain failed population")
                self._quality_pending_tokens[digest_json(list(tokens))].append(row)
            if seen != set(ledger.rows):
                raise ValueError("harness omitted or duplicated selected generation documents")
            ledger.state = "generating"
            ledger.save()
            responses = super().generate_until(requests, disable_tqdm=disable_tqdm)
            # Store the exact postprocessed text handed to the official filters.
            queues = defaultdict(deque)
            for row in ledger.rows.values():
                queues[row["doc_sha256"]].append(row)
            for req, text in zip(requests, responses, strict=True):
                queues[digest_json(req.doc)].popleft()["scored_text"] = text
            ledger.save()
            return responses

        def _model_generate(self, requests, generate=False, sampling_params=None):
            if not generate:
                raise ValueError("generation-only run unexpectedly requested prefill loglikelihood")
            rows = []
            params = sampling_params if isinstance(sampling_params, list) else [sampling_params] * len(requests)
            for tokens, parameter in zip(requests, params, strict=True):
                queue = self._quality_pending_tokens[digest_json(list(tokens))]
                if not queue:
                    raise ValueError("harness truncated/changed an input or generated it twice")
                row = queue.popleft()
                if parameter.max_tokens != ledger.spec["max_gen_toks"] or parameter.ignore_eos:
                    raise ValueError("actual generation cap/EOS differs from the frozen protocol")
                row.update(status="in_flight", started_unix_s=time.time(),
                           actual_sampling={"max_tokens": parameter.max_tokens, "temperature": parameter.temperature,
                                            "ignore_eos": parameter.ignore_eos, "stop": parameter.stop,
                                            "stop_token_ids": parameter.stop_token_ids})
                rows.append(row)
            ledger.save()
            outputs = super()._model_generate(requests, generate=True, sampling_params=sampling_params)
            for row, output in zip(rows, outputs, strict=True):
                if list(output.prompt_token_ids) != row["prompt_token_ids"] or len(output.outputs) != 1:
                    raise ValueError("vLLM returned a different prompt or more than one completion")
                completion = output.outputs[0]
                tokens = list(completion.token_ids)
                finish, stop = completion.finish_reason, completion.stop_reason
                if not output.finished or finish is None:
                    status = "unfinished"
                elif finish == "length":
                    status = "cap"
                elif finish == "stop":
                    status = ("stop_string" if isinstance(stop, str) else "stop_token" if isinstance(stop, int)
                              else "eos" if tokens and tokens[-1] == self.eot_token_id else "stop_unspecified")
                else:
                    status = "error"
                row.update(status=status, completed_unix_s=time.time(), request_id=output.request_id,
                           output_token_ids=tokens, raw_text=completion.text, finish_reason=finish, stop_reason=stop)
            ledger.save()
            if any(row["status"] in ("error", "unfinished") for row in rows):
                raise RuntimeError("vLLM returned aborted/unfinished generation; aggregate quality is invalid")
            return outputs
    return RecordedVLLM


def worker_plan(path, output, *, hash_weights):
    """Freeze resolved inputs without importing torch/vLLM or loading a model.

    Content hashes identify the actual sidecar files, not just their filenames or
    the host checkpoint. Basis/shape/source validation remains in the real worker;
    its installation report is retained even when model initialization fails.
    """
    path = Path(path).resolve()
    config = read_json(path)
    if config.get("schema_version") != 1:
        raise ValueError("worker config requires schema_version=1")
    if config.get("mechanism_only", False) is not False or config.get("allow_unverified_source", False) is not False:
        raise ValueError("quality evaluation rejects mechanism_only and unverified-source sidecars")
    policy = config.get("policy")
    if policy == "single":
        fmt = config.get("single_format")
        if fmt not in ("bf16", "fp8", "awq", "nvfp4"):
            raise ValueError("single worker requires single_format=bf16/fp8/awq/nvfp4")
        if any(key in config for key in ("threshold", "lowbit_format")):
            raise ValueError("single worker must not carry dual-policy fields")
        if config.get("release_original_weights", True) is not True:
            raise ValueError("single-residency worker must release the original expert weights")
        weights = {"weights": fmt}
        label = fmt
    elif policy in ("threshold", "phase", "expert_split"):
        fmt = config.get("lowbit_format")
        if fmt not in ("awq", "nvfp4") or (policy == "expert_split" and fmt != "awq"):
            raise ValueError("dual lowbit_format must be awq/nvfp4; expert_split supports awq only")
        if type(config.get("threshold")) is not int or config["threshold"] < 1:
            raise ValueError("dual worker requires a positive integer threshold")
        if "single_format" in config:
            raise ValueError("dual worker must not carry single_format")
        weights = {"fp8_weights": "fp8", "lowbit_weights": fmt}
        label = f"dual_fp8_{fmt}"
    else:
        raise ValueError("worker policy must be single, threshold, phase or expert_split")
    layers = config.get("layers")
    if not isinstance(layers, list) or not layers:
        raise ValueError("worker config must enumerate every actual MoE layer")
    records, seen = [], set()
    required = {"layer", "module", "worker_activation_basis_id", *weights}
    for layer in layers:
        if not isinstance(layer, dict) or set(layer) != required:
            raise ValueError(f"worker layer requires exactly {sorted(required)}")
        index = layer["layer"]
        if type(index) is not int or index < 0 or index in seen:
            raise ValueError("worker layer indices must be distinct nonnegative integers")
        seen.add(index)
        if layer["module"] != f"model.layers.{index}.mlp.experts":
            raise ValueError("unsupported worker module path")
        basis = layer["worker_activation_basis_id"]
        if not isinstance(basis, str) or not re.fullmatch(r"rmsnorm_input_basis_v1:sha256:[0-9a-f]{64}", basis):
            raise ValueError("worker layer requires its actual expected canonical RMSNorm basis ID")
        for key, artifact_format in weights.items():
            if not isinstance(layer[key], str) or not layer[key]:
                raise ValueError("sidecar path must be a nonempty string")
            artifact = (path.parent / layer[key]).resolve()
            if not artifact.is_file():
                raise FileNotFoundError(artifact)
            layer[key] = str(artifact)
            records.append({"layer": index, "role": key, "format": artifact_format,
                            "path": str(artifact), "bytes": artifact.stat().st_size,
                            "sha256": sha256(artifact) if hash_weights else None,
                            "worker_activation_basis_id": basis})
    output = Path(output).resolve()
    # Every run gets a new report, and relative sidecars retain source-JSON meaning.
    config["report_path"] = str(output / "worker_installation.json")
    return {"worker_class": WORKER_CLASS, "policy": policy, "format": label,
            "worker_model_kwargs": {"worker_cls": WORKER_CLASS, "enforce_eager": True,
                                    "compilation_config": {"mode": 0, "cudagraph_mode": "NONE"}},
            "source_config": str(path), "source_config_sha256": sha256(path),
            "effective_config_path": str(output / "worker_config.json"),
            "effective_config": config, "artifacts": records,
            "sidecar_hashes_verified": hash_weights,
            "basis_validation": "pending actual loaded model and independently checked sidecar norm tensors",
            "precision_scope": "expert sidecars only; router, attention and other nonexpert parameters follow the host checkpoint",
            "installation_verified": False}


def verify_checkpoint(manifest, *, hash_weights):
    root = Path(manifest["checkpoint_path"]).resolve()
    if sha256(root / "config.json") != manifest["identity"]["config_sha256"]:
        raise ValueError("checkpoint config changed since registration")
    for record in manifest["identity"]["weight_files"]:
        path = (root / record["path"]).resolve()
        if not path.is_relative_to(root) or not path.is_file() or path.stat().st_size != record["bytes"]:
            raise ValueError(f"missing/changed checkpoint shard {record['path']}")
        if hash_weights and sha256(path) != record["sha256"]:
            raise ValueError(f"weight hash changed for {record['path']}; quality/performance identity is lost")
    validate_config(read_json(root / "config.json"), manifest["model_key"], manifest["format"])
    return root


def build_plan(args):
    generation = generation_plan(args.generation_config) if getattr(args, "generation_config", None) else None
    if generation and (args.tasks or args.limit is not None or args.num_fewshot != 0):
        raise ValueError("generation config fixes its task/sample IDs/zero-shot protocol; do not combine --tasks, --limit or fewshot")
    manifest = read_json(args.checkpoint_manifest)
    worker = worker_plan(args.worker_config, args.output, hash_weights=args.run) if args.worker_config else None
    fmt = args.format or (worker["format"] if worker else manifest["format"])
    if worker:
        if fmt != worker["format"]:
            raise ValueError(f"--format must be {worker['format']} for this worker config; dual is not ordinary FP8")
    elif fmt != manifest["format"] and not (manifest["format"] == "bf16" and fmt == "fp8"):
        raise ValueError("format must match checkpoint; only BF16 -> runtime FP8 is an allowed override")
    defaults = read_json(REGISTRY)["quality_defaults"]
    root = verify_checkpoint(manifest, hash_weights=args.run)
    quantization = "fp8" if fmt == "fp8" and not worker else manifest["runtime_quantization"]
    model_args = {"pretrained": str(root), "tokenizer": str(root),
                  "dtype": "bfloat16", "tensor_parallel_size": 1,
                  "gpu_memory_utilization": args.gpu_memory_utilization,
                  "max_model_len": args.max_model_len, "kv_cache_dtype": "auto",
                  "trust_remote_code": False, "seed": args.seed}
    if quantization:
        model_args["quantization"] = quantization
    if worker:
        model_args.update(worker["worker_model_kwargs"])
    if generation:
        model_args.update(enable_thinking=False, chat_template_args={"enable_thinking": False})
    if args.enforce_eager or worker or generation:
        model_args["enforce_eager"] = True
        model_args["compilation_config"] = {"mode": 0, "cudagraph_mode": "NONE"}
    tasks = [generation["task"]] if generation else args.tasks.split(",") if args.tasks else defaults["tasks"]
    mc_only = bool(tasks) and set(tasks) <= MC_TASKS
    evaluated_identity = {"host_checkpoint_fingerprint": manifest["checkpoint_fingerprint"],
                          "host_runtime_quantization": quantization, "format": fmt}
    if worker:
        evaluated_identity["worker_policy"] = {key: value for key, value in worker["effective_config"].items()
                                               if key not in ("layers", "report_path")}
        evaluated_identity["sidecars"] = [{key: value for key, value in record.items() if key != "path"}
                                          for record in worker["artifacts"]]
    evaluated_fingerprint = (hashlib.sha256(json.dumps(evaluated_identity, sort_keys=True,
                              separators=(",", ":")).encode()).hexdigest() if args.run else None)
    return {"schema_version": 2, "status": "planned", "model_key": manifest["model_key"],
            "format": fmt, "checkpoint_fingerprint": manifest["checkpoint_fingerprint"],
            "host_checkpoint_format": manifest["format"], "worker": worker,
            "generation": generation,
            "evaluated_model_identity": evaluated_identity,
            "evaluated_model_fingerprint": evaluated_fingerprint,
            "checkpoint_manifest": str(Path(args.checkpoint_manifest).resolve()),
            "identity": manifest["identity"], "calibration": manifest["calibration"],
            "weights_hashes_verified": args.run, "smoke_only": args.limit is not None,
            "call": {"model": "vllm", "model_args": model_args, "tasks": tasks,
                     "num_fewshot": args.num_fewshot, "batch_size": generation["batch_size"] if generation else args.batch_size,
                     "limit": args.limit, "log_samples": True,
                     "apply_chat_template": True if generation else args.apply_chat_template,
                     "random_seed": args.seed, "numpy_random_seed": args.seed,
                     "torch_random_seed": args.seed, "fewshot_random_seed": args.seed,
                     "bootstrap_iters": args.bootstrap_iters},
            "scope": generation["scope"] if generation else ("multiple-choice prompt-loglikelihood quality (prefill); not mixed-decode generation quality"
                      if mc_only else "task-specific quality; inspect actual task output types before interpreting generation"),
            "evaluation_contract": {
                "default_five_tasks_unchanged": tasks == defaults["tasks"],
                "full_task_population_requested": args.limit is None and not generation,
                "known_multiple_choice_only": mc_only,
                "scoring_path": "lm-eval vLLM prompt_logprobs=1, max_tokens=1; extra sampled token is not the scored continuation" if mc_only else "task-dependent",
                "mixed_decode_quality_established": False,
                "phase_policy_limit": "generation executes the actual worker; selection counts must demonstrate both formats" if generation else "default MC scoring exercises prompt/prefill choices, not a multi-step decode policy",
                "length_limit": "generation rejects over-budget prompts without filtering/resampling/truncation" if generation else "installed lm-eval truncates context+continuation to max_model_len-1; this is not the serving no-truncation protocol"},
            "comparison_note": "Use identical tasks, prompt/template, fewshot, KV dtype and seeds; sidecar arms must share the same nonexpert host checkpoint. Never substitute a different base release."}


def json_default(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (set, tuple)):
        return list(value)
    raise TypeError(f"cannot serialize {type(value)}")


def main():
    defaults = read_json(REGISTRY)["quality_defaults"]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint-manifest")
    parser.add_argument("--format", choices=("bf16", "fp8", "awq", "nvfp4", "dual_fp8_awq", "dual_fp8_nvfp4"))
    parser.add_argument("--worker-config", help="Local DualFormatWorker JSON; infer format, force eager and retain sidecar identities")
    parser.add_argument("--lock-status", help="RUNNING run_locked status.json; required for every --run")
    parser.add_argument("--generation-config", help="Pinned held-out generate_until protocol; separate from the unchanged default five MC tasks")
    parser.add_argument("--prepare-generation-data", action="store_true", help="CPU-only task/population preparation, no model; requires --generation-config")
    parser.add_argument("--output", required=True)
    parser.add_argument("--tasks", help="comma-separated lm-eval task names")
    parser.add_argument("--num-fewshot", type=int, default=defaults["num_fewshot"])
    parser.add_argument("--seed", type=int, default=defaults["seed"])
    parser.add_argument("--batch-size", default=defaults["batch_size"])
    parser.add_argument("--max-model-len", type=int, default=defaults["max_model_len"])
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--bootstrap-iters", type=int, default=1000)
    parser.add_argument("--apply-chat-template", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true",
                        help="Disable compilation and CUDA graphs for bounded quality checks; recorded in model_args")
    parser.add_argument("--limit", type=float, help="smoke-only per-task count or fraction")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if not 0 < args.gpu_memory_utilization < 1:
        parser.error("--gpu-memory-utilization must be between zero and one")
    if args.prepare_generation_data:
        if not args.generation_config or args.run or args.worker_config or args.tasks or args.limit is not None:
            parser.error("CPU data preparation requires only --generation-config and --output, without --run/worker/tasks/limit")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=False)
        ledger = GenerationLedger(output, generation_plan(args.generation_config))
        try:
            prepare_generation_task(ledger.spec, ledger)
            ledger.state = "prepared_no_generation"
            ledger.save()
        except BaseException as error:
            ledger.fail(error)
            raise
        print(json.dumps({"status": ledger.state, "output": str(output.resolve()), "selected": len(ledger.rows)}))
        return
    if not args.checkpoint_manifest:
        parser.error("--checkpoint-manifest is required outside CPU generation-data preparation")
    if args.run:
        if not args.lock_status:
            parser.error("--run requires --lock-status and a real run_locked ancestor")
        from bench_serving import verify_lock
        verify_lock(args.lock_status)
    plan = build_plan(args)
    if not args.run:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return
    # Weights are local and already validated. Prevent an unexpected alternate
    # model download while leaving dataset access available to the harness.
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    if plan["worker"]:
        worker = plan["worker"]
        write_json(worker["effective_config_path"], worker["effective_config"])
        worker["effective_config_sha256"] = sha256(worker["effective_config_path"])
        os.environ["C_FORMAT_DUAL_CONFIG"] = worker["effective_config_path"]
        code_path = str(Path(__file__).resolve().parent)
        os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, (code_path, os.environ.get("PYTHONPATH"))))
        worker["environment"] = {"C_FORMAT_DUAL_CONFIG": os.environ["C_FORMAT_DUAL_CONFIG"],
                                 "PYTHONPATH_code_prefix": code_path}
        worker["implementation_sha256"] = {name: sha256(Path(code_path) / name) for name in
            ("dual_format_worker.py", "dual_format_moe.py", "format_backends.py", "export_bf16_fp8_artifact.py")}
    plan["versions"] = {}
    for package in ("lm_eval", "vllm", "torch", "transformers", "compressed-tensors"):
        try:
            plan["versions"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            plan["versions"][package] = None
    plan.update(status="running", start_unix_s=time.time())
    write_json(output / "run.json", plan)
    ledger, lm, previous_handlers = None, None, {}
    try:
        call = dict(plan["call"])
        if plan["generation"]:
            ledger = GenerationLedger(output, plan["generation"])
            def interrupt(signum, frame):
                if signum == signal.SIGALRM:
                    raise TimeoutError("generation wall budget exhausted")
                raise InterruptedError(f"generation interrupted by signal {signum}")
            for signum in (signal.SIGTERM, signal.SIGALRM):
                previous_handlers[signum] = signal.signal(signum, interrupt)
            signal.setitimer(signal.ITIMER_REAL, plan["generation"]["wall_budget_s"])
        import lm_eval
        if ledger:
            manager, task = prepare_generation_task(plan["generation"], ledger)
            from lm_eval.models.vllm_causallms import VLLM
            recorded = recording_vllm(VLLM, ledger)
            lm = recorded(**call["model_args"], batch_size=plan["generation"]["batch_size"])
            if plan["worker"]:
                startup = lm.model.collective_rpc("dual_format_reset_statistics", timeout=5)
                write_json(output / "generation_worker_startup.json", startup)
            call.update(model=lm, tasks=[task], task_manager=manager,
                        samples={plan["generation"]["task"]: plan["generation"]["sample_ids"]},
                        cache_requests=False, limit=None)
        results = lm_eval.simple_evaluate(**call)
        if results is None:
            raise RuntimeError("lm-eval returned no results")
        (output / "results.json").write_text(json.dumps(results, indent=2,
                           ensure_ascii=False, default=json_default, allow_nan=False) + "\n")
        plan["actual_task_output_types"] = {name: config.get("output_type")
            for name, config in results.get("configs", {}).items() if isinstance(config, dict)}
        plan["actual_task_sample_counts"] = results.get("n-samples", {})
        if ledger:
            # Only a complete fixed population receives an official aggregate.
            if any(row["status"] not in ("eos", "stop_string", "stop_token", "stop_unspecified", "cap")
                   for row in ledger.rows.values()):
                raise RuntimeError("generation population has missing/failed outcomes")
            scored = results.get("samples", {}).get(plan["generation"]["task"], [])
            populations = defaultdict(set)
            for row in scored:
                if row["doc_id"] not in ledger.rows:
                    raise RuntimeError("scored sample ID is outside the frozen population")
                populations[row["filter"]].add(row["doc_id"])
                ledger.rows[row["doc_id"]].setdefault("official_scores", {})[row["filter"]] = {
                    "exact_match": row.get("exact_match"), "filtered_resps": row.get("filtered_resps")}
            if set(populations) != {"strict-match", "flexible-extract"} or any(
                    ids != set(ledger.rows) for ids in populations.values()):
                raise RuntimeError("official gold scoring omitted part of the fixed population")
            ledger.metrics = results.get("results", {})
            ledger.state = "complete"
            ledger.save()
            plan["generation_summary"] = str((output / "generation_summary.json").resolve())
        if plan["worker"]:
            report = read_json(plan["worker"]["effective_config"]["report_path"])
            if report.get("status") != "installed_before_kv_profile" or report.get("config_sha256") != plan["worker"]["effective_config_sha256"]:
                raise RuntimeError("worker installation evidence is missing, failed or from a different config")
            plan["worker"]["installation_verified"] = True
            plan["worker"]["basis_validation"] = "passed real worker installation checks; see installation_report"
        plan.update(status="complete", end_unix_s=time.time())
    except BaseException as error:
        if ledger:
            ledger.fail(error)
        plan.update(status="failed", end_unix_s=time.time(), error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if previous_handlers:
            signal.setitimer(signal.ITIMER_REAL, 0)
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)
        if ledger and lm is not None and plan["worker"]:
            try:
                statistics = lm.model.collective_rpc("dual_format_summary", timeout=5)
                write_json(output / "generation_worker_statistics.json", statistics)
                plan["generation_worker_statistics"] = str((output / "generation_worker_statistics.json").resolve())
            except Exception as error:
                plan["generation_worker_statistics_error"] = f"{type(error).__name__}: {error}"
        if plan["worker"]:
            report_path = Path(plan["worker"]["effective_config"]["report_path"])
            if report_path.is_file():
                plan["worker"]["installation_report"] = str(report_path)
                plan["worker"]["installation_report_sha256"] = sha256(report_path)
            else:
                plan["worker"]["installation_report_missing"] = True
        write_json(output / "run.json", plan)
    print(json.dumps({"status": plan["status"], "output": str(output.resolve()),
                      "format": plan["format"],
                      "checkpoint_fingerprint": plan["checkpoint_fingerprint"],
                      "evaluated_model_fingerprint": plan["evaluated_model_fingerprint"],
                      "smoke_only": plan["smoke_only"]}, indent=2))


if __name__ == "__main__":
    main()
