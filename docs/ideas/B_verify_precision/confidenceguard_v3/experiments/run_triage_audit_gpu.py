#!/usr/bin/env python3
"""Formal single-GPU runner for TriageAudit v2 / ConfidenceGuard v3.

``smoke`` is the pre-approval OLMoE memory check. ``calibration`` and
``sealed`` refuse to start without a source-bound Code Review approval file.
The runner measures quality only; prepared W4A16 RTN weights are not a native
INT4 kernel and wall-clock values must not be cited as speed evidence.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch


def _find_workspace_root() -> Path:
    here = Path(__file__).resolve().parent
    for candidate in (here, *here.parents):
        if (candidate / "experiments" / "shared" / "capture_moe.py").is_file():
            return candidate
    raise RuntimeError("cannot locate experiments/shared/capture_moe.py")


WORKSPACE_ROOT = _find_workspace_root()
SHARED = WORKSPACE_ROOT / "experiments" / "shared"
if str(SHARED) not in sys.path:
    sys.path.insert(0, str(SHARED))

from capture_moe import patch_mixtral_moe  # noqa: E402
from triage_artifacts import (  # noqa: E402
    ArtifactError,
    JsonlJournal,
    environment_snapshot,
    sha256_file,
    source_manifest,
    write_json_no_overwrite,
)
from triage_executor import execute_policy_trajectory  # noqa: E402
from triage_features import extract_full_route_features, prefill_mean_nll  # noqa: E402
from triage_manifest import text_sha256  # noqa: E402
from triage_policy import (  # noqa: E402
    FrozenConfidenceGuard,
    FrozenRidgeTriage,
    budget_matched_hash_periods,
    common_audit_phase,
)
from triage_runtime import PreparedInt4ExpertBackend  # noqa: E402
from triage_runtime import per_step_kl  # noqa: E402


class RunnerError(RuntimeError):
    pass


class MemoryGateError(RunnerError):
    pass


PRIMARY_SOURCE_NAMES = (
    "triage_policy.py",
    "triage_runtime.py",
    "triage_executor.py",
    "triage_features.py",
    "triage_manifest.py",
    "triage_statistics.py",
    "triage_artifacts.py",
    "finalize_calibration.py",
    "finalize_confidence_guard.py",
    "analyze_triage_results.py",
    "prepare_triage_data.py",
    "run_triage_audit_gpu.py",
)


def load_manifest(
    path: Path,
    *,
    expected_split: str,
    expected_count: int,
    allow_prefix: bool = False,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise RunnerError(f"manifest line {line_number} is not an object")
        text = value.get("text")
        digest = value.get("text_sha256")
        if value.get("schema_version") != "triage-document-v2" or value.get("split") != expected_split:
            raise RunnerError(f"manifest line {line_number} has wrong schema/split")
        if not isinstance(text, str) or digest != text_sha256(text):
            raise RunnerError(f"manifest line {line_number} text hash mismatch")
        rows.append(value)
    if (not allow_prefix and len(rows) != expected_count) or (allow_prefix and len(rows) < expected_count):
        raise RunnerError(f"manifest requires {expected_count} rows, got {len(rows)}")
    rows = rows[:expected_count]
    hashes = [str(row["text_sha256"]) for row in rows]
    if len(set(hashes)) != len(hashes):
        raise RunnerError("manifest contains duplicate document hashes")
    return rows


def build_period_plan(
    documents: Sequence[Mapping[str, object]],
    features: Mapping[str, Mapping[str, float]],
    predictor: FrozenRidgeTriage | FrozenConfidenceGuard,
    *,
    model_key: str,
    split: str,
) -> dict[str, dict[str, int]]:
    hashes = [str(row["text_sha256"]) for row in documents]
    if set(features) != set(hashes):
        raise RunnerError("feature/document sets do not close")
    triage = {digest: predictor.period(features[digest]) for digest in hashes}
    hashed = budget_matched_hash_periods(
        hashes,
        [triage[digest] for digest in hashes],
        model_key=model_key,
        split=split,
    )
    if sorted(triage.values()) != sorted(hashed.values()):
        raise RunnerError("exact period multiset closure failed")
    return {"triage_2_4_8": triage, "hash_budget_matched_2_4_8": hashed}


def _canonical_json(value: Mapping[str, object]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _write_or_verify(path: Path, value: Mapping[str, object], *, resume: bool) -> None:
    payload = _canonical_json(value)
    if path.exists():
        if not resume or path.read_bytes() != payload:
            raise RunnerError(f"existing artifact differs or resume is disabled: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _protocol_path(config_path: Path) -> Path:
    protocol_name = (
        "ConfidenceGuard_v3_冻结实验设计_2026-07-23.md"
        if config_path.name == "config_v3.json"
        else "TriageAudit_Phase2_v2_冻结实验设计_2026-07-23.md"
    )
    return Path(__file__).resolve().parent.parent / protocol_name


def _source_manifest(config_path: Path) -> dict[str, object]:
    here = Path(__file__).resolve().parent
    sources = [here / name for name in PRIMARY_SOURCE_NAMES]
    sources.extend(
        [
            config_path,
            _protocol_path(config_path),
            WORKSPACE_ROOT / "experiments" / "shared" / "capture_moe.py",
        ]
    )
    return source_manifest(sources, root=WORKSPACE_ROOT)


def initialize_artifacts(
    output_dir: Path,
    config_path: Path,
    config: Mapping[str, object],
    manifest_path: Path,
    documents: Sequence[Mapping[str, object]],
    *,
    mode: str,
    resume: bool,
    calibration_lock: Mapping[str, object] | None,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    source = _source_manifest(config_path)
    _write_or_verify(output_dir / "config.json", dict(config), resume=resume)
    _write_or_verify(output_dir / "source_manifest.json", source, resume=resume)
    environment_path = output_dir / "environment.json"
    if not environment_path.exists():
        _write_or_verify(environment_path, environment_snapshot(), resume=False)
    elif not resume:
        raise RunnerError("environment.json exists without --resume")
    _write_or_verify(
        output_dir / "data_manifest.json",
        {
            "schema_version": "triage-data-manifest-reference-v2",
            "mode": mode,
            "input_sha256": sha256_file(manifest_path),
            "document_count": len(documents),
            "document_sha256s": [row["text_sha256"] for row in documents],
        },
        resume=resume,
    )
    protocol_path = _protocol_path(config_path)
    _write_or_verify(
        output_dir / "protocol_sha256.json",
        {"path": str(protocol_path.relative_to(WORKSPACE_ROOT)), "sha256": sha256_file(protocol_path)},
        resume=resume,
    )
    if calibration_lock is not None:
        _write_or_verify(output_dir / "calibration_lock.json", dict(calibration_lock), resume=resume)
    log_path = output_dir / "stdout.log"
    if log_path.exists() and not resume:
        raise RunnerError("stdout.log exists without --resume")
    if not log_path.exists():
        log_path.touch(exist_ok=False)
    return source


def validate_approval(
    path: Path,
    *,
    source: Mapping[str, object],
    config_path: Path,
) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": "triage-code-review-approval-v2",
        "gpu_run_approved": "MECHANISM PROBE ONLY",
        "source_aggregate_sha256": source["aggregate_sha256"],
        "config_sha256": sha256_file(config_path),
    }
    if not isinstance(value, Mapping) or any(value.get(key) != item for key, item in expected.items()):
        raise RunnerError("Code Review approval is missing, stale, or not source-bound")


def validate_memory_certificate(
    path: Path,
    *,
    config_path: Path,
    config: Mapping[str, object],
    source: Mapping[str, object],
) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        value.get("schema_version") != "triage-memory-certificate-v2"
        or not value.get("pass")
        or value.get("config_sha256") != sha256_file(config_path)
        or value.get("model_revision") != config["models"]["olmoe"]["revision"]
        or value.get("source_aggregate_sha256") != source["aggregate_sha256"]
        or value.get("gpu_name") != config["memory_gate"]["gpu_exact_name"]
    ):
        raise RunnerError("OLMoE memory certificate is missing, failed, or stale")


def _set_reproducibility(seed: int) -> None:
    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if workspace is None:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    elif workspace not in {":4096:8", ":16:8"}:
        raise RunnerError("CUBLAS_WORKSPACE_CONFIG is incompatible with deterministic execution")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def _patch_full(model: torch.nn.Module, *, record_routes: bool):
    recorder = patch_mixtral_moe(
        model,
        "full",
        num_receiver_groups=1,
        record_routes=record_routes,
        record_diagnostics=False,
    )
    recorder.update_contrib = lambda *args, **kwargs: None
    recorder.update_receiver = lambda *args, **kwargs: None
    recorder.update_error = lambda *args, **kwargs: None
    recorder.update_pair_audit = lambda *args, **kwargs: None
    return recorder


def _num_experts(model: torch.nn.Module) -> int:
    value = getattr(model.config, "num_experts", None)
    if value is None:
        value = getattr(model.config, "num_local_experts", None)
    if value is None:
        raise RunnerError("model config has no expert count")
    return int(value)


def _moe_layer_count(model: torch.nn.Module) -> int:
    count = 0
    for layer in model.model.layers:
        if hasattr(layer, "block_sparse_moe") or (hasattr(layer, "mlp") and hasattr(layer.mlp, "experts")):
            count += 1
    return count


def _load_model_and_tokenizer(model_config: Mapping[str, object], *, offline: bool):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    repo = str(model_config["repo_id"])
    revision = str(model_config["revision"])
    tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision, local_files_only=offline)
    model = AutoModelForCausalLM.from_pretrained(
        repo,
        revision=revision,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=offline,
        trust_remote_code=False,
    )
    model.eval().to("cuda")
    if model.training or next(model.parameters()).device.type != "cuda":
        raise RunnerError("model is not in CUDA eval mode")
    return model, tokenizer


def _tokenize(tokenizer, text: str, required: int, device: torch.device) -> torch.Tensor:
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=required)["input_ids"]
    if encoded.shape != (1, required):
        raise RunnerError(f"document has {encoded.shape[1]} tokens, requires exactly {required}")
    return encoded.to(device)


def _prefill_with_features(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    *,
    num_experts: int,
) -> tuple[Any, dict[str, float]]:
    recorder = _patch_full(model, record_routes=True)
    with torch.inference_mode():
        output = model(input_ids=prompt_ids, use_cache=True)
    if len(recorder.route_batches) != _moe_layer_count(model):
        raise RunnerError(
            f"route hook hit {len(recorder.route_batches)} batches, expected {_moe_layer_count(model)}"
        )
    features = extract_full_route_features(recorder.route_batches, num_experts)
    features["full_mean_nll"] = prefill_mean_nll(output.logits, prompt_ids)
    recorder.route_batches.clear()
    recorder.routing_weight_batches.clear()
    return output.past_key_values, features


def _prefill(model: torch.nn.Module, prompt_ids: torch.Tensor) -> Any:
    _patch_full(model, record_routes=False)
    with torch.inference_mode():
        return model(input_ids=prompt_ids, use_cache=True).past_key_values


def _decode_reference(model: torch.nn.Module, cache: Any, decode_ids: torch.Tensor) -> torch.Tensor:
    _patch_full(model, record_routes=False)
    logits: list[torch.Tensor] = []
    current = cache
    with torch.inference_mode():
        for step in range(decode_ids.shape[1]):
            output = model(
                input_ids=decode_ids[:, step : step + 1],
                past_key_values=current,
                use_cache=True,
            )
            current = output.past_key_values
            logits.append(output.logits[:, -1, :].detach())
    return torch.cat(logits, dim=0)


def _check_original_vs_patched_full(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    limits: Mapping[str, object],
) -> dict[str, float]:
    """Verify that the shared capture hook does not change the BF16 baseline."""
    with torch.inference_mode():
        original = model(input_ids=prompt_ids, use_cache=False).logits[:, -1, :].detach()
    _patch_full(model, record_routes=False)
    with torch.inference_mode():
        patched = model(input_ids=prompt_ids, use_cache=False).logits[:, -1, :].detach()
    max_abs = float((original.float() - patched.float()).abs().max().cpu().item())
    kl = per_step_kl(original, patched)
    if max_abs > float(limits["patched_full_max_abs_logit_error"]) or kl > float(limits["patched_full_max_token_kl"]):
        raise RunnerError(f"patched-full equivalence failed: max_abs={max_abs}, kl={kl}")
    return {"patched_full_max_abs_logit_error": max_abs, "patched_full_token_kl": kl}


def _run_candidate(
    model: torch.nn.Module,
    backend: PreparedInt4ExpertBackend,
    *,
    policy: str,
    initial_cache: Any,
    decode_ids: torch.Tensor,
    reference_logits: torch.Tensor,
    threshold: float,
    period: int | None,
    phase: int | None,
    controller: Mapping[str, object],
    collect_diagnostics: bool = True,
    fingerprint_final_cache: bool = False,
) -> tuple[dict[str, object], list[dict[str, object]], int]:
    def high_forward(token: torch.Tensor, cache: Any):
        with torch.inference_mode():
            return model(input_ids=token, past_key_values=cache, use_cache=True)

    def low_forward(token: torch.Tensor, cache: Any):
        with backend, torch.inference_mode():
            return model(input_ids=token, past_key_values=cache, use_cache=True)

    before = backend.low_model_forwards
    before_expert_calls = backend.expert_linear_calls
    summary, steps = execute_policy_trajectory(
        policy=policy,
        initial_cache=initial_cache,
        decode_tokens=decode_ids,
        reference_logits=reference_logits,
        high_forward=high_forward,
        low_forward=low_forward,
        discrepancy_threshold=threshold,
        period=period,
        phase=phase,
        max_unaudited_steps=int(controller["max_unaudited_steps"]),
        lockout_following_steps=0 if policy == "full_shadow" else int(controller["lockout_following_steps"]),
        collect_diagnostics=collect_diagnostics,
        fingerprint_final_cache=fingerprint_final_cache,
    )
    low_forwards = backend.low_model_forwards - before
    expert_calls = backend.expert_linear_calls - before_expert_calls
    if collect_diagnostics and low_forwards != decode_ids.shape[1]:
        raise RunnerError("every candidate step must execute exactly one low branch including diagnostics")
    if low_forwards != summary["physical_low_forward_calls"]:
        raise RunnerError("physical low-forward counter does not match backend activation count")
    if low_forwards > 0 and expert_calls <= 0:
        raise RunnerError("INT4 backend activated but no expert linear proxy was called")
    return summary, steps, low_forwards


def _always_bf16_summary(decode_steps: int) -> dict[str, object]:
    return {
        "policy": "always_bf16",
        "decode_steps": decode_steps,
        "document_mean_kl": 0.0,
        "document_cvar90_kl": 0.0,
        "document_p95_kl": 0.0,
        "dangerous_steps": 0,
        "dangerous_step_recall": 1.0,
        "threshold_violation_fraction": 0.0,
        "diagnostic_forward_calls": 0,
        "diagnostic_high_forward_calls": 0,
        "diagnostic_low_forward_calls": 0,
        "diagnostic_clone_events": 0,
        "physical_high_forward_calls": decode_steps,
        "physical_low_forward_calls": 0,
        "audit_events": 0,
        "high_forward_calls": decode_steps,
        "low_forward_calls": 0,
        "total_candidate_forward_calls": decode_steps,
        "cache_clone_events": 0,
        "served_high_steps": decode_steps,
        "served_low_steps": 0,
        "lockout_steps": 0,
    }


def _memory_snapshot(config: Mapping[str, object]) -> dict[str, object]:
    total = int(torch.cuda.get_device_properties(0).total_memory)
    max_bytes = min(
        float(config["memory_gate"]["peak_gib_max_cap"]) * 2**30,
        total - float(config["memory_gate"]["guard_gib_min"]) * 2**30,
    )
    peak_allocated = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())
    return {
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_total_bytes": total,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "allowed_peak_bytes": int(max_bytes),
        "pass": max(peak_allocated, peak_reserved) <= max_bytes,
    }


def _load_feature_journal(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    result: dict[tuple[str, str], dict[str, float]] = {}
    if not path.exists():
        return result
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            key = (str(row["model_key"]), str(row["text_sha256"]))
            if key in result:
                raise RunnerError("duplicate feature journal key")
            result[key] = {name: float(value) for name, value in row["features"].items()}
    return result


def _log(output_dir: Path, message: str) -> None:
    print(message, flush=True)
    with (output_dir / "stdout.log").open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def run_model(
    *,
    model_key: str,
    model_config: Mapping[str, object],
    documents: Sequence[Mapping[str, object]],
    mode: str,
    config: Mapping[str, object],
    output_dir: Path,
    result_journal: JsonlJournal,
    feature_journal: JsonlJournal,
    existing_features: dict[tuple[str, str], dict[str, float]],
    calibration_lock: Mapping[str, object] | None,
    offline: bool,
) -> None:
    _log(output_dir, f"[{model_key}] loading pinned revision {model_config['revision']}")
    model, tokenizer = _load_model_and_tokenizer(model_config, offline=offline)
    backend = PreparedInt4ExpertBackend(
        model,
        expected_linears=int(model_config["expected_expert_linears"]),
        scope=str(config["action"]["expert_scope"]),
    )
    num_experts = _num_experts(model)
    prompt_len = 8 if mode == "smoke" else int(config["dataset"]["prompt_len"])
    decode_steps = 2 if mode == "smoke" else int(config["dataset"]["decode_steps"])
    required = prompt_len + decode_steps
    split = "calibration" if mode in {"smoke", "calibration"} else "sealed"
    hook_equivalence: dict[str, float] | None = None

    if mode == "sealed":
        for index, document in enumerate(documents):
            digest = str(document["text_sha256"])
            resume_key = f"feature|{model_key}|{digest}"
            if resume_key in feature_journal.completed_keys:
                continue
            ids = _tokenize(tokenizer, str(document["text"]), required, next(model.parameters()).device)
            cache, features = _prefill_with_features(model, ids[:, :prompt_len], num_experts=num_experts)
            del cache, ids
            feature_journal.append({
                "resume_key": resume_key,
                "model_key": model_key,
                "split": split,
                "text_sha256": digest,
                "features": features,
            })
            existing_features[(model_key, digest)] = features
            _log(output_dir, f"[{model_key}] sealed feature {index + 1}/{len(documents)}")

    plan: dict[str, dict[str, int]] | None = None
    predictor: FrozenRidgeTriage | None = None
    confidence_guard: FrozenConfidenceGuard | None = None
    threshold = 0.0
    if calibration_lock is not None:
        locked = calibration_lock["models"][model_key]
        if calibration_lock.get("schema_version") == "confidence-guard-calibration-lock-v3":
            confidence_guard = FrozenConfidenceGuard.from_dict(
                locked["confidence_guard"]["frozen_guard"]
            )
            predictor = confidence_guard.point_model
        else:
            predictor = FrozenRidgeTriage.from_dict(locked["stability"]["frozen_model"])
        threshold = float(locked["audit_threshold"])
    if mode == "sealed":
        assert predictor is not None
        feature_map = {str(row["text_sha256"]): existing_features[(model_key, str(row["text_sha256"]))] for row in documents}
        period_predictor = confidence_guard if confidence_guard is not None else predictor
        plan = build_period_plan(documents, feature_map, period_predictor, model_key=model_key, split=split)

    for index, document in enumerate(documents):
        digest = str(document["text_sha256"])
        ids = _tokenize(tokenizer, str(document["text"]), required, next(model.parameters()).device)
        prompt_ids = ids[:, :prompt_len]
        decode_ids = ids[:, prompt_len:required]
        if mode == "smoke" and hook_equivalence is None:
            hook_equivalence = _check_original_vs_patched_full(
                model,
                prompt_ids,
                config["hook_equivalence"],
            )
        reference_cache = _prefill(model, prompt_ids)
        reference_logits = _decode_reference(model, reference_cache, decode_ids)
        del reference_cache
        if mode in {"smoke", "calibration"}:
            resume_key = f"{mode}|{model_key}|{digest}"
            if resume_key in result_journal.completed_keys:
                del ids, reference_logits
                continue
            candidate_cache, features = _prefill_with_features(model, prompt_ids, num_experts=num_experts)
            summary, steps, low_forwards = _run_candidate(
                model,
                backend,
                policy="always_low" if mode == "calibration" else "fixed_2",
                initial_cache=candidate_cache,
                decode_ids=decode_ids,
                reference_logits=reference_logits,
                threshold=0.0,
                period=None if mode == "calibration" else 2,
                phase=None if mode == "calibration" else 0,
                controller=config["controller"],
                fingerprint_final_cache=mode == "smoke",
            )
            result_row: dict[str, object] = {
                "resume_key": resume_key,
                "schema_version": "triage-raw-document-v2",
                "model_key": model_key,
                "split": split,
                "text_sha256": digest,
                "features": features,
                "same_state_discrepancies": [row["same_state_discrepancy"] for row in steps],
                "physical_low_model_forwards": low_forwards,
                "summary": summary,
                "steps": steps,
            }
            if mode == "smoke":
                if max(float(row["same_state_discrepancy"]) for row in steps) <= float(
                    config["hook_equivalence"]["int4_discrepancy_min_exclusive"]
                ):
                    raise RunnerError("INT4 intervention produced no measurable discrepancy")
                invariant_cache = _prefill(model, prompt_ids)
                invariant_summary, invariant_steps, _ = _run_candidate(
                    model,
                    backend,
                    policy="fixed_2",
                    initial_cache=invariant_cache,
                    decode_ids=decode_ids,
                    reference_logits=reference_logits,
                    threshold=0.0,
                    period=2,
                    phase=0,
                    controller=config["controller"],
                    collect_diagnostics=False,
                    fingerprint_final_cache=True,
                )
                if (
                    [row["served_action"] for row in steps]
                    != [row["served_action"] for row in invariant_steps]
                    or [row["served_logits_sha256"] for row in steps]
                    != [row["served_logits_sha256"] for row in invariant_steps]
                    or summary["final_cache_sha256"] != invariant_summary["final_cache_sha256"]
                ):
                    raise RunnerError("diagnostic-off action/logit/cache invariance failed")
                result_row["hook_equivalence"] = hook_equivalence
                result_row["diagnostic_off_invariance"] = True
            result_journal.append(result_row)
            _log(output_dir, f"[{model_key}] {mode} document {index + 1}/{len(documents)}")
        else:
            assert plan is not None and predictor is not None
            features = existing_features[(model_key, digest)]
            arms: list[tuple[str, int | None]] = [
                ("always_bf16", None),
                ("always_low", None),
                ("triage_2_4_8", plan["triage_2_4_8"][digest]),
                ("hash_budget_matched_2_4_8", plan["hash_budget_matched_2_4_8"][digest]),
                ("fixed_2", 2),
                ("fixed_4", 4),
                ("fixed_8", 8),
                ("full_shadow", 1),
            ]
            for policy, period in arms:
                resume_key = f"sealed|{model_key}|{digest}|{policy}"
                if resume_key in result_journal.completed_keys:
                    continue
                if policy == "always_bf16":
                    summary = _always_bf16_summary(decode_steps)
                    steps: list[dict[str, object]] = []
                    low_forwards = 0
                    phase = None
                else:
                    phase = common_audit_phase(digest, period) if period is not None else None
                    candidate_cache = _prefill(model, prompt_ids)
                    summary, steps, low_forwards = _run_candidate(
                        model,
                        backend,
                        policy=policy,
                        initial_cache=candidate_cache,
                        decode_ids=decode_ids,
                        reference_logits=reference_logits,
                        threshold=threshold,
                        period=period,
                        phase=phase,
                        controller=config["controller"],
                    )
                result_journal.append({
                    "resume_key": resume_key,
                    "schema_version": "triage-raw-document-v2",
                    "model_key": model_key,
                    "split": split,
                    "text_sha256": digest,
                    "policy": policy,
                    "period": period,
                    "phase": phase,
                    "features": features,
                    "risk_score": predictor.score(features),
                    "safe_probability": (
                        confidence_guard.safe_probability(features)
                        if confidence_guard is not None
                        else None
                    ),
                    "physical_low_model_forwards": low_forwards,
                    **summary,
                    "steps": steps,
                })
            _log(output_dir, f"[{model_key}] sealed document {index + 1}/{len(documents)} all arms")
        del ids, reference_logits
    del backend, model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "calibration", "sealed"), required=True)
    parser.add_argument("--model-key", choices=("all", "olmoe", "llmjp"), default="all")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calibration-lock", type=Path)
    parser.add_argument("--approval-file", type=Path)
    parser.add_argument("--memory-certificate", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RunnerError("runner requires exactly one visible CUDA GPU")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("status") != "EXPERIMENT_DESIGN_FROZEN_GPU_NOT_APPROVED":
        raise RunnerError("unexpected config status")
    if torch.cuda.get_device_name(0) != config["memory_gate"]["gpu_exact_name"]:
        raise RunnerError(f"unexpected GPU: {torch.cuda.get_device_name(0)}")
    if args.mode == "smoke" and args.model_key != "olmoe":
        raise RunnerError("memory smoke must run OLMoE only")
    if args.mode == "sealed" and args.calibration_lock is None:
        raise RunnerError("sealed mode requires calibration lock")
    if args.mode != "sealed" and args.calibration_lock is not None:
        raise RunnerError("calibration lock may only be supplied to sealed mode")
    expected_split = "sealed" if args.mode == "sealed" else "calibration"
    expected_count = 1 if args.mode == "smoke" else int(config["dataset"][f"{expected_split}_documents"])
    documents = load_manifest(
        args.manifest,
        expected_split=expected_split,
        expected_count=expected_count,
        allow_prefix=args.mode == "smoke",
    )
    lock = json.loads(args.calibration_lock.read_text(encoding="utf-8")) if args.calibration_lock else None
    if lock is not None:
        config_canonical_sha256 = hashlib.sha256(
            json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        schema = lock.get("schema_version")
        lock_pass = (
            lock.get("reformulation_gate_all_models_pass")
            if schema == "confidence-guard-calibration-lock-v3"
            else lock.get("h0_all_models_pass")
        )
        expected_schema = (
            "confidence-guard-calibration-lock-v3"
            if config.get("schema_version") == "confidence-guard-v3-design"
            else "triage-calibration-lock-v2"
        )
        if (
            schema != expected_schema
            or not lock_pass
            or lock.get("config_canonical_sha256") != config_canonical_sha256
            or set(lock.get("models", {})) != set(config["models"])
        ):
            raise RunnerError("sealed mode forbidden by missing, failed, or stale calibration lock")
        calibration_sets = {
            tuple(value.get("document_sha256s", [])) for value in lock["models"].values()
        }
        if len(calibration_sets) != 1:
            raise RunnerError("calibration document sets differ across models")
    source = _source_manifest(args.config)
    if args.mode in {"calibration", "sealed"}:
        if args.approval_file is None or args.memory_certificate is None:
            raise RunnerError("formal run requires approval file and OLMoE memory certificate")
        validate_approval(args.approval_file, source=source, config_path=args.config)
        validate_memory_certificate(
            args.memory_certificate,
            config_path=args.config,
            config=config,
            source=source,
        )
    initialized_source = initialize_artifacts(
        args.output_dir,
        args.config,
        config,
        args.manifest,
        documents,
        mode=args.mode,
        resume=args.resume,
        calibration_lock=lock,
    )
    if initialized_source != source:
        raise RunnerError("source manifest changed during preflight")
    _set_reproducibility(int(config["seed"]))
    torch.cuda.reset_peak_memory_stats()
    results = JsonlJournal(args.output_dir / "raw_results.jsonl", resume=args.resume)
    features = JsonlJournal(args.output_dir / "sealed_features.jsonl", resume=args.resume)
    existing_features = _load_feature_journal(args.output_dir / "sealed_features.jsonl")
    model_keys = ("olmoe", "llmjp") if args.model_key == "all" else (args.model_key,)
    started = time.time()
    try:
        for model_key in model_keys:
            run_model(
                model_key=model_key,
                model_config=config["models"][model_key],
                documents=documents,
                mode=args.mode,
                config=config,
                output_dir=args.output_dir,
                result_journal=results,
                feature_journal=features,
                existing_features=existing_features,
                calibration_lock=lock,
                offline=args.offline,
            )
        memory = _memory_snapshot(config)
        if not memory["pass"]:
            raise MemoryGateError("peak GPU memory exceeded frozen guard")
        if args.mode == "smoke":
            certificate = {
                "schema_version": "triage-memory-certificate-v2",
                "config_sha256": sha256_file(args.config),
                "model_revision": config["models"]["olmoe"]["revision"],
                "source_aggregate_sha256": source["aggregate_sha256"],
                **memory,
            }
            write_json_no_overwrite(args.output_dir / "memory_certificate.json", certificate)
        status = {
            "schema_version": "triage-run-status-v2",
            "status": "ENGINEERING_PASS",
            "mode": args.mode,
            "model_keys": list(model_keys),
            "elapsed_seconds_diagnostic_only": time.time() - started,
            "memory": memory,
            "scientific_result": None,
        }
        write_json_no_overwrite(args.output_dir / "status.json", status)
        summary = (
            f"# TriageAudit {args.mode} run\n\n"
            f"Engineering status: PASS\n\n"
            f"Scientific result: not decided by this runner.\n\n"
            f"Evidence boundary: {config['evidence_boundary']}\n"
        )
        summary_path = args.output_dir / "summary.md"
        with summary_path.open("x", encoding="utf-8") as handle:
            handle.write(summary)
    except (torch.cuda.OutOfMemoryError, MemoryGateError):
        if not (args.output_dir / "status.json").exists():
            write_json_no_overwrite(args.output_dir / "status.json", {
                "schema_version": "triage-run-status-v2",
                "status": "BLOCKED_MEMORY",
                "mode": args.mode,
                "scientific_result": None,
            })
        raise
    except Exception as exc:
        if not (args.output_dir / "status.json").exists():
            write_json_no_overwrite(args.output_dir / "status.json", {
                "schema_version": "triage-run-status-v2",
                "status": "INVALID_RUN",
                "mode": args.mode,
                "error_type": type(exc).__name__,
                "scientific_result": None,
            })
        raise


if __name__ == "__main__":
    main()
