#!/usr/bin/env python3
"""Execute the frozen RouteGuard-KV R0-A decode-only mechanism probe."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping

# Must be set before the first cuBLAS handle is created; PyTorch deterministic
# mode otherwise rejects CUDA matmul on CUDA >= 10.2.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from kv_quant_cache import (
    QuantizedDynamicCache,
    assert_no_storage_aliases,
    cache_structure_fingerprint,
    clone_dynamic_cache,
)
from r0a_artifacts import (
    ArtifactError,
    append_jsonl_fsync,
    assert_5090_environment,
    assert_formal_approval,
    build_frozen_bindings,
    canonical_json_sha256,
    environment_snapshot,
    load_config,
    load_json,
    load_jsonl,
    ordered_hash_of_hashes,
    sha256_file,
    verify_frozen_bindings,
    write_json_no_overwrite,
)
from route_lock import (
    RouteController,
    RouteReference,
    exact_set_equal,
    jaccard_overlap,
    patch_olmoe_routes,
)


class RunError(RuntimeError):
    pass


SOURCE_NAMES = (
    "run_r0a_5090.py",
    "kv_quant_cache.py",
    "route_lock.py",
    "r0a_artifacts.py",
    "prepare_r0a_data.py",
    "analyze_r0a.py",
    "requirements-r0a-5090.txt",
    "run_cpu_tests.py",
    "test_kv_quant_cache.py",
    "test_route_lock.py",
    "test_r0a_artifacts.py",
    "test_r0a_analysis.py",
    "test_prepare_r0a_data.py",
    "test_run_gates.py",
    "test_olmoe_integration.py",
    "prepare_r0a_approval.py",
)


@dataclass
class ReferenceRun:
    logits: list[torch.Tensor]
    references: dict[tuple[int, int], RouteReference]
    trajectory: dict[str, Any]


def _tensor_kl(reference: torch.Tensor, treatment: torch.Tensor) -> float:
    if reference.shape != treatment.shape:
        raise RunError(f"logit shape mismatch: {reference.shape} != {treatment.shape}")
    log_p = F.log_softmax(reference.float(), dim=-1)
    log_q = F.log_softmax(treatment.float(), dim=-1)
    value = (log_p.exp() * (log_p - log_q)).sum()
    result = float(value.item())
    if not math.isfinite(result):
        raise RunError("non-finite token KL")
    return result


def _token_nll(logits: torch.Tensor, target: torch.Tensor) -> float:
    value = F.cross_entropy(logits.float(), target.reshape(-1), reduction="mean")
    result = float(value.item())
    if not math.isfinite(result):
        raise RunError("non-finite token NLL")
    return result


def _reference_routes(
    router_logits: Iterable[torch.Tensor], *, top_k: int, norm_topk_prob: bool, dtype: torch.dtype
) -> dict[int, RouteReference]:
    result: dict[int, RouteReference] = {}
    for layer_idx, logits in enumerate(router_logits):
        flat = logits.reshape(-1, logits.shape[-1])
        probabilities = F.softmax(flat, dim=-1, dtype=torch.float)
        weights, experts = torch.topk(probabilities, top_k, dim=-1)
        if norm_topk_prob:
            weights = weights / weights.sum(dim=-1, keepdim=True)
        weights = weights.to(dtype)
        boundary = torch.topk(flat.float(), top_k + 1, dim=-1).values
        margin = boundary[:, top_k - 1] - boundary[:, top_k]
        result[layer_idx] = RouteReference(
            selected_experts=experts.detach().cpu().clone(),
            native_weights=weights.detach().cpu().clone(),
            boundary_margin=margin.detach().cpu().clone(),
        )
    return result


def _base_trajectory(
    *,
    document: Mapping[str, Any],
    prompt_length: int,
    target: str,
    arm: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "routeguard-kv-trajectory-v1",
        "split": document["split"],
        "document_index": int(document["document_index"]),
        "text_sha256": document["text_sha256"],
        "prompt_length": int(prompt_length),
        "target": target,
        "arm": arm,
        "teacher_forced": True,
        "decode_steps_expected": int(config["dataset"]["decode_steps"]),
    }


def _run_reference(
    model: torch.nn.Module,
    prompt_cache: DynamicCache,
    token_ids: torch.Tensor,
    *,
    document: Mapping[str, Any],
    prompt_length: int,
    config: Mapping[str, Any],
) -> ReferenceRun:
    cache = clone_dynamic_cache(prompt_cache, config=model.config)
    references: dict[tuple[int, int], RouteReference] = {}
    logits_cpu: list[torch.Tensor] = []
    steps: list[dict[str, Any]] = []
    expected_layers = int(config["model"]["expected"]["num_hidden_layers"])
    top_k = int(config["model"]["expected"]["num_experts_per_tok"])
    norm_topk_prob = bool(config["model"]["expected"]["norm_topk_prob"])
    decode_steps = int(config["dataset"]["decode_steps"])
    started = time.perf_counter()
    for step in range(decode_steps):
        input_token = token_ids[:, prompt_length + step : prompt_length + step + 1]
        target_token = token_ids[:, prompt_length + step + 1]
        output = model(
            input_ids=input_token,
            past_key_values=cache,
            use_cache=True,
            output_router_logits=True,
            return_dict=True,
            logits_to_keep=1,
        )
        cache = output.past_key_values
        logits = output.logits[:, -1, :]
        logits_cpu.append(logits.detach().float().cpu())
        per_layer = _reference_routes(
            output.router_logits,
            top_k=top_k,
            norm_topk_prob=norm_topk_prob,
            dtype=logits.dtype,
        )
        if len(per_layer) != expected_layers:
            raise RunError(f"reference step {step} returned {len(per_layer)} router layers")
        for layer_idx, reference in per_layer.items():
            references[(step, layer_idx)] = reference
        sequence_length = cache.get_seq_length()
        if sequence_length != prompt_length + step + 1:
            raise RunError(
                f"reference cache length {sequence_length} != {prompt_length + step + 1}"
            )
        steps.append(
            {
                "step": step,
                "input_token_id": int(input_token.item()),
                "target_token_id": int(target_token.item()),
                "nll": _token_nll(logits, target_token),
                "kl": 0.0,
                "cache_sequence_length": sequence_length,
            }
        )
    trajectory = _base_trajectory(
        document=document,
        prompt_length=prompt_length,
        target="bf16",
        arm="bf16_reference",
        config=config,
    )
    trajectory.update(
        {
            "completed_steps": len(steps),
            "mean_kl": 0.0,
            "max_token_kl": 0.0,
            "mean_nll": sum(row["nll"] for row in steps) / len(steps),
            "elapsed_seconds": time.perf_counter() - started,
            "steps": steps,
            "reference_route_records": len(references),
            "cache_fingerprint": cache_structure_fingerprint(cache),
        }
    )
    del cache
    return ReferenceRun(logits=logits_cpu, references=references, trajectory=trajectory)


def _route_metrics(
    controller: RouteController,
    references: Mapping[tuple[int, int], RouteReference],
    *,
    tie_margin: float,
) -> dict[str, Any]:
    set_flips = 0
    non_tie_flips = 0
    non_tie_cells = 0
    executed_mismatches = 0
    jaccards: list[float] = []
    for observation in controller.observations:
        key = (observation.step, observation.layer)
        reference = references[key]
        equal = bool(
            exact_set_equal(observation.natural_experts, reference.selected_experts).all().item()
        )
        executed_equal = bool(
            exact_set_equal(observation.executed_experts, reference.selected_experts).all().item()
        )
        is_tie = bool(
            (reference.boundary_margin.abs() <= tie_margin).any().item()
            or (observation.boundary_margin.abs() <= tie_margin).any().item()
        )
        if not equal:
            set_flips += 1
            if not is_tie:
                non_tie_flips += 1
        if not is_tie:
            non_tie_cells += 1
        if not executed_equal:
            executed_mismatches += 1
        jaccards.extend(
            float(value) for value in jaccard_overlap(
                observation.natural_experts, reference.selected_experts
            ).tolist()
        )
    return {
        "route_cell_count": len(controller.observations),
        "set_flip_count": set_flips,
        "non_tie_set_flip_count": non_tie_flips,
        "non_tie_cell_count": non_tie_cells,
        "executed_reference_set_mismatch_count": executed_mismatches,
        "mean_jaccard": sum(jaccards) / len(jaccards) if jaccards else 0.0,
    }


def _run_treatment(
    model: torch.nn.Module,
    cache: DynamicCache,
    token_ids: torch.Tensor,
    reference: ReferenceRun,
    *,
    document: Mapping[str, Any],
    prompt_length: int,
    target: str,
    arm: str,
    controller_mode: str,
    config: Mapping[str, Any],
    quantizer_ledger: Any | None,
) -> dict[str, Any]:
    expected_layers = int(config["model"]["expected"]["num_hidden_layers"])
    controller = RouteController(
        mode=controller_mode,
        text_sha256=str(document["text_sha256"]),
        prompt_length=prompt_length,
        expected_layers=expected_layers,
        references=reference.references,
    )
    steps: list[dict[str, Any]] = []
    decode_steps = int(config["dataset"]["decode_steps"])
    started = time.perf_counter()
    with patch_olmoe_routes(model, controller):
        for step in range(decode_steps):
            controller.begin_step(step)
            input_token = token_ids[:, prompt_length + step : prompt_length + step + 1]
            target_token = token_ids[:, prompt_length + step + 1]
            output = model(
                input_ids=input_token,
                past_key_values=cache,
                use_cache=True,
                output_router_logits=False,
                return_dict=True,
                logits_to_keep=1,
            )
            controller.end_step()
            cache = output.past_key_values
            logits = output.logits[:, -1, :]
            reference_logits = reference.logits[step].to(logits.device)
            sequence_length = cache.get_seq_length()
            if sequence_length != prompt_length + step + 1:
                raise RunError(
                    f"{target}/{arm} cache length {sequence_length} != {prompt_length + step + 1}"
                )
            steps.append(
                {
                    "step": step,
                    "input_token_id": int(input_token.item()),
                    "target_token_id": int(target_token.item()),
                    "nll": _token_nll(logits, target_token),
                    "kl": _tensor_kl(reference_logits, logits),
                    "max_abs_logit_error": float(
                        (reference_logits.float() - logits.float()).abs().max().item()
                    ),
                    "cache_sequence_length": sequence_length,
                }
            )
    if len(controller.observations) != decode_steps * expected_layers:
        raise RunError(
            f"route ledger has {len(controller.observations)} rows, "
            f"expected {decode_steps * expected_layers}"
        )
    route_metrics = _route_metrics(
        controller,
        reference.references,
        tie_margin=float(config["route_lock"]["tie_boundary_abs_margin_max"]),
    )
    trajectory = _base_trajectory(
        document=document,
        prompt_length=prompt_length,
        target=target,
        arm=arm,
        config=config,
    )
    trajectory.update(
        {
            "completed_steps": len(steps),
            "mean_kl": sum(row["kl"] for row in steps) / len(steps),
            "max_token_kl": max(row["kl"] for row in steps),
            "max_abs_logit_error": max(row["max_abs_logit_error"] for row in steps),
            "mean_nll": sum(row["nll"] for row in steps) / len(steps),
            "elapsed_seconds": time.perf_counter() - started,
            "steps": steps,
            "route_metrics": route_metrics,
            "cache_fingerprint": cache_structure_fingerprint(cache),
        }
    )
    if quantizer_ledger is not None:
        snapshot_events = [event for event in quantizer_ledger.events if event.phase == "snapshot"]
        update_events = [event for event in quantizer_ledger.events if event.phase == "update"]
        trajectory["quantizer_ledger"] = {
            "snapshot_events": len(snapshot_events),
            "update_events": len(update_events),
            "expected_snapshot_events": expected_layers,
            "expected_update_events": expected_layers * decode_steps,
            "pass": len(snapshot_events) == expected_layers
            and len(update_events) == expected_layers * decode_steps,
        }
    del cache
    return trajectory


def _trajectory_filename(row: Mapping[str, Any]) -> str:
    key = "|".join(
        str(row[name]) for name in ("text_sha256", "prompt_length", "target", "arm")
    )
    return hashlib.sha256(key.encode("utf-8")).hexdigest() + ".json"


def _store_trajectory(output_dir: Path, row: Mapping[str, Any]) -> bool:
    path = output_dir / "trajectories" / _trajectory_filename(row)
    if path.exists():
        existing = load_json(path)
        fields = ("text_sha256", "prompt_length", "target", "arm")
        if any(existing.get(field) != row.get(field) for field in fields):
            raise RunError(f"trajectory filename collision at {path}")
        return False
    write_json_no_overwrite(path, dict(row), mode=0o600)
    append_jsonl_fsync(
        output_dir / "journal.jsonl",
        {
            "event": "trajectory_complete",
            "text_sha256": row["text_sha256"],
            "prompt_length": row["prompt_length"],
            "target": row["target"],
            "arm": row["arm"],
            "trajectory_sha256": sha256_file(path),
        },
    )
    return True


def _core_source_binding(experiment_dir: Path, config: Mapping[str, Any], repo_root: Path) -> dict[str, Any]:
    files = {name: sha256_file(experiment_dir / name) for name in SOURCE_NAMES}
    protocol = repo_root / str(config["protocol"])
    files[str(protocol.relative_to(repo_root))] = sha256_file(protocol)
    config_path = experiment_dir / "configs/r0a_5090_v1.json"
    files[str(config_path.relative_to(repo_root))] = sha256_file(config_path)
    return {"files": dict(sorted(files.items())), "canonical_sha256": canonical_json_sha256(files)}


def _configure_runtime(config: Mapping[str, Any]) -> None:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != config["software"]["cublas_workspace_config"]:
        raise RunError("CUBLAS_WORKSPACE_CONFIG differs from the frozen deterministic contract")
    if not torch.cuda.is_available():
        raise RunError("CUDA is unavailable")
    total_bytes = torch.cuda.get_device_properties(0).total_memory
    cap_bytes = float(config["hardware"]["peak_allocated_gib_max"]) * (1024**3)
    torch.cuda.set_per_process_memory_fraction(min(1.0, cap_bytes / total_bytes), device=0)
    torch.backends.cuda.matmul.allow_tf32 = bool(config["software"]["allow_tf32"])
    torch.backends.cudnn.allow_tf32 = bool(config["software"]["allow_tf32"])
    torch.use_deterministic_algorithms(bool(config["software"]["deterministic_algorithms"]))
    torch.manual_seed(int(config["seed"]))
    torch.cuda.manual_seed_all(int(config["seed"]))


def _assert_model_identity(model: torch.nn.Module, config: Mapping[str, Any]) -> None:
    expected = config["model"]["expected"]
    actual = {
        "num_hidden_layers": model.config.num_hidden_layers,
        "hidden_size": model.config.hidden_size,
        "num_attention_heads": model.config.num_attention_heads,
        "num_key_value_heads": model.config.num_key_value_heads,
        "num_experts": model.config.num_experts,
        "num_experts_per_tok": model.config.num_experts_per_tok,
        "norm_topk_prob": model.config.norm_topk_prob,
        "max_position_embeddings": model.config.max_position_embeddings,
    }
    if actual != expected:
        raise RunError(f"model config identity mismatch: {actual} != {expected}")
    commit = getattr(model.config, "_commit_hash", None)
    if commit is not None and commit != config["model"]["revision"]:
        raise RunError(f"loaded model commit mismatch: {commit}")


def _load_model_and_tokenizer(config: Mapping[str, Any], *, local_files_only: bool):
    model_config = config["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["repo_id"],
        revision=model_config["revision"],
        local_files_only=local_files_only,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_config["repo_id"],
        revision=model_config["revision"],
        dtype=torch.bfloat16,
        attn_implementation=config["software"]["attention_implementation"],
        low_cpu_mem_usage=True,
        local_files_only=local_files_only,
    )
    model.eval().to("cuda")
    _assert_model_identity(model, config)
    return model, tokenizer


def _load_and_validate_manifest(path: Path, config: Mapping[str, Any], phase: str) -> list[dict[str, Any]]:
    rows = load_jsonl(path)
    expected_count = {
        "smoke": int(config["dataset"]["smoke_documents"]),
        "calibration": int(config["dataset"]["calibration_documents"]),
        "formal": int(config["dataset"]["sealed_documents"]),
    }[phase]
    expected_split = {"smoke": "smoke", "calibration": "calibration", "formal": "sealed"}[phase]
    if len(rows) != expected_count:
        raise RunError(f"manifest has {len(rows)} rows, expected {expected_count}")
    hashes: set[str] = set()
    for index, row in enumerate(rows):
        if row.get("split") != expected_split or int(row.get("document_index", -1)) != index:
            raise RunError(f"manifest split/index mismatch at row {index}")
        text = row.get("text")
        if not isinstance(text, str):
            raise RunError(f"manifest row {index} has no text")
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest != row.get("text_sha256") or digest in hashes:
            raise RunError(f"manifest hash mismatch/duplicate at row {index}")
        hashes.add(digest)
    return rows


def _validate_data_provenance(
    provenance_path: Path,
    registry_path: Path,
    *,
    config_path: Path,
    config: Mapping[str, Any],
    manifest: list[Mapping[str, Any]],
    phase: str,
) -> dict[str, Any]:
    provenance = load_json(provenance_path)
    registry = load_json(registry_path)
    expected_split = {"smoke": "smoke", "calibration": "calibration", "formal": "sealed"}[phase]
    data = config["dataset"]
    checks = {
        "schema_version": provenance.get("schema_version") == "routeguard-kv-data-provenance-v1",
        "config_sha256": provenance.get("config_sha256") == sha256_file(config_path),
        "dataset_repo_id": provenance.get("dataset_repo_id") == data["repo_id"],
        "dataset_config": provenance.get("dataset_config") == data["config"],
        "dataset_revision": provenance.get("dataset_revision") == data["revision"],
        "dataset_split": provenance.get("dataset_split") == data["split"],
        "required_tokens": int(provenance.get("required_tokens", -1)) == int(data["required_tokens"]),
        "registry_sha256": provenance.get("historical_hash_registry_sha256")
        == sha256_file(registry_path),
        "registry_schema": registry.get("schema_version") == "routeguard-kv-historical-hashes-v1",
    }
    manifest_hashes = [str(row["text_sha256"]) for row in manifest]
    checks["manifest_ordered_hash"] = provenance.get("ordered_hash_of_hashes", {}).get(
        expected_split
    ) == ordered_hash_of_hashes(manifest_hashes)
    historical = set(str(value) for value in registry.get("hashes", []))
    checks["historical_disjoint"] = not (historical & set(manifest_hashes))
    checks["selected_lengths_present"] = all(
        digest in provenance.get("selected_token_lengths", {}) for digest in manifest_hashes
    )
    checks["dataset_fingerprint_present"] = bool(provenance.get("dataset_fingerprint"))
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RunError(f"data provenance validation failed: {failed}")
    return {
        "path": str(provenance_path.resolve()),
        "sha256": sha256_file(provenance_path),
        "historical_registry_path": str(registry_path.resolve()),
        "historical_registry_sha256": sha256_file(registry_path),
        "dataset_fingerprint": provenance["dataset_fingerprint"],
        "ordered_hash_of_hashes": provenance["ordered_hash_of_hashes"][expected_split],
        "historical_unique_count": provenance["historical_unique_count"],
        "eligible_count": provenance["eligible_count"],
    }


def _treatment_plan(config: Mapping[str, Any], phase: str, digest: str, prompt_length: int):
    plan: list[tuple[str, str, str]] = []
    if phase in {"smoke", "calibration"}:
        plan.append(("patched_bf16", "patched_free", "free"))
        plan.extend(
            [
                ("identity", "identity_free", "free"),
                ("identity", "identity_set_locked", "set_locked"),
                ("identity", "identity_fully_locked", "fully_locked"),
            ]
        )
    else:
        plan.append(("identity", "identity_free", "free"))
    for target in config["quantization"]["targets"]:
        if prompt_length not in [int(value) for value in target["prompt_lengths"]]:
            continue
        for arm in ("free", "set_locked", "fully_locked"):
            plan.append((str(target["name"]), arm, arm))
    return sorted(
        plan,
        key=lambda item: hashlib.sha256(
            f"{digest}|{prompt_length}|{item[0]}|{item[1]}".encode("utf-8")
        ).hexdigest(),
    )


def _prepare_output(
    *,
    output_dir: Path,
    repo_root: Path,
    experiment_dir: Path,
    config_path: Path,
    manifest_path: Path,
    provenance_path: Path,
    registry_path: Path,
    data_provenance_summary: Mapping[str, Any],
    extra_binding_paths: Iterable[Path],
    phase: str,
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], int]:
    source_paths = [
        *(experiment_dir / name for name in SOURCE_NAMES),
        provenance_path,
        registry_path,
        *extra_binding_paths,
    ]
    current = build_frozen_bindings(
        repo_root=repo_root,
        config_path=config_path,
        manifest_path=manifest_path,
        source_paths=source_paths,
    )
    binding_digest = canonical_json_sha256(current)
    new = not output_dir.exists()
    if new:
        output_dir.mkdir(parents=True, exist_ok=False)
        (output_dir / "trajectories").mkdir()
        write_json_no_overwrite(output_dir / "frozen_bindings.json", current, mode=0o600)
        write_json_no_overwrite(output_dir / "environment.json", environment_snapshot())
        write_json_no_overwrite(
            output_dir / "metadata.json",
            {
                "schema_version": "routeguard-kv-run-metadata-v1",
                "phase": phase,
                "manifest": str(manifest_path.resolve()),
                "manifest_sha256": sha256_file(manifest_path),
                "frozen_bindings_sha256": binding_digest,
                "core_source_binding": _core_source_binding(experiment_dir, config, repo_root),
                "data_provenance": dict(data_provenance_summary),
                "evidence_boundary": config["evidence_boundary"],
            },
        )
        resume_count = 0
    else:
        if (output_dir / "integrity.json").exists():
            raise RunError("completed output is immutable; choose a new run id")
        existing = load_json(output_dir / "frozen_bindings.json")
        verify_frozen_bindings(existing, repo_root)
        if canonical_json_sha256(existing) != binding_digest:
            raise RunError("resume bindings differ from the current source/config/manifest")
        metadata = load_json(output_dir / "metadata.json")
        if metadata.get("phase") != phase:
            raise RunError("resume phase differs from output metadata")
        if metadata.get("frozen_bindings_sha256") != binding_digest:
            raise RunError("resume metadata binding digest mismatch")
        if metadata.get("core_source_binding") != _core_source_binding(
            experiment_dir, config, repo_root
        ):
            raise RunError("resume metadata core source binding mismatch")
        if metadata.get("data_provenance") != dict(data_provenance_summary):
            raise RunError("resume metadata data provenance mismatch")
        prior_starts = 0
        journal = output_dir / "journal.jsonl"
        if journal.is_file():
            prior_starts = sum(
                1
                for line in journal.read_text(encoding="utf-8").splitlines()
                if line.strip() and json.loads(line).get("event") == "run_start"
            )
        resume_count = prior_starts
    append_jsonl_fsync(
        output_dir / "journal.jsonl",
        {"event": "run_start", "phase": phase, "resume_count": resume_count},
    )
    return current, resume_count


def _run_cpu_qualification(experiment_dir: Path, output_dir: Path, resume_count: int) -> None:
    result = subprocess.run(
        [sys.executable, str(experiment_dir / "run_cpu_tests.py")],
        cwd=str(experiment_dir),
        check=False,
        capture_output=True,
        text=True,
    )
    record = {
        "schema_version": "routeguard-kv-cpu-tests-v1",
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
    write_json_no_overwrite(
        output_dir / f"cpu_tests_session_{resume_count:03d}.json", record, mode=0o600
    )
    if result.returncode != 0 or "CPU_TESTS_OK" not in result.stdout:
        raise RunError("CPU qualification tests failed; refusing GPU execution")


def qualification_artifact_paths(path: Path) -> list[Path]:
    required = [
        path / "metadata.json",
        path / "environment.json",
        path / "frozen_bindings.json",
        path / "integrity.json",
        path / "journal.jsonl",
    ]
    required.extend(sorted(path.glob("cpu_tests_session_*.json")))
    required.extend(sorted((path / "trajectories").glob("*.json")))
    missing = [value for value in required[:5] if not value.is_file()]
    if missing or not any(value.name.startswith("cpu_tests_session_") for value in required):
        raise RunError(f"qualification artifacts are incomplete: {missing}")
    if not any("trajectories" in value.parts for value in required):
        raise RunError("qualification has no trajectory artifacts")
    return required


def _load_qualification(
    path: Path,
    *,
    expected_phase: str,
    expected_core_binding: Mapping[str, Any],
    current_environment: Mapping[str, Any],
) -> dict[str, Any]:
    integrity = load_json(path / "integrity.json")
    metadata = load_json(path / "metadata.json")
    environment = load_json(path / "environment.json")
    if integrity.get("status") != "PASS":
        raise RunError("formal run requires a PASS calibration qualification")
    if metadata.get("phase") != expected_phase:
        raise RunError(f"qualification must come from {expected_phase}, got {metadata.get('phase')}")
    if metadata.get("core_source_binding") != expected_core_binding:
        raise RunError("qualification source/config binding differs from formal run")
    for field in (
        "python",
        "platform",
        "torch",
        "torch_cuda",
        "transformers",
        "datasets",
        "huggingface_hub",
        "numpy",
        "cuda_available",
        "gpu_name",
        "compute_capability",
        "cuda_driver",
    ):
        if environment.get(field) != current_environment.get(field):
            raise RunError(f"qualification environment differs at {field}")
    return {"path": str(path.resolve()), "integrity": integrity, "environment": environment}


def _integrity_summary(
    *,
    phase: str,
    trajectories: list[Mapping[str, Any]],
    config: Mapping[str, Any],
    resume_count: int,
    peak_allocated_gib: float,
    peak_reserved_gib: float,
    qualification: Mapping[str, Any] | None,
) -> dict[str, Any]:
    gates = config["integrity_gates"]
    failures: list[str] = []
    document_count = {
        "smoke": int(config["dataset"]["smoke_documents"]),
        "calibration": int(config["dataset"]["calibration_documents"]),
        "formal": int(config["dataset"]["sealed_documents"]),
    }[phase]
    expected_trajectory_count = document_count * (25 if phase in {"smoke", "calibration"} else 19)
    if len(trajectories) != expected_trajectory_count:
        failures.append("trajectory_matrix")
    quant_failures = [
        row for row in trajectories if "quantizer_ledger" in row and not row["quantizer_ledger"]["pass"]
    ]
    incomplete = [
        row for row in trajectories if int(row.get("completed_steps", -1)) != gates["required_steps_per_trajectory"]
    ]
    if quant_failures:
        failures.append("quantizer_ledger")
    if incomplete:
        failures.append("trajectory_steps")
    if peak_allocated_gib > float(config["hardware"]["peak_allocated_gib_max"]):
        failures.append("memory_cap")

    qualification_rows = trajectories if phase in {"smoke", "calibration"} else []
    patched = [row for row in qualification_rows if row["target"] == "patched_bf16"]
    identity = [row for row in trajectories if row["target"] == "identity"]
    patched_route_mismatch = max(
        (row["route_metrics"]["set_flip_count"] for row in patched), default=0
    )
    patched_kl = max((row["max_token_kl"] for row in patched), default=0.0)
    patched_logit = max((row["max_abs_logit_error"] for row in patched), default=0.0)
    identity_route_flip = max(
        (row["route_metrics"]["set_flip_count"] for row in identity), default=0
    )
    identity_kl = max((row["max_token_kl"] for row in identity), default=0.0)
    locked_executed_mismatch = max(
        (
            row["route_metrics"]["executed_reference_set_mismatch_count"]
            for row in trajectories
            if "locked" in str(row["arm"])
        ),
        default=0,
    )
    if locked_executed_mismatch > 0:
        failures.append("locked_executed_route_mismatch")
    if phase in {"smoke", "calibration"}:
        if len(patched) != document_count * 2 or len(identity) != document_count * 6:
            failures.append("missing_qualification_controls")
        if patched_route_mismatch > int(gates["patched_full_route_mismatch_max"]):
            failures.append("patched_route_mismatch")
        if patched_kl > float(gates["patched_full_max_token_kl"]):
            failures.append("patched_kl")
        if patched_logit > float(gates["patched_full_max_abs_logit_error"]):
            failures.append("patched_logit")
    elif qualification is None:
        failures.append("missing_calibration_qualification")
    if identity_route_flip > int(gates["identity_route_flip_max"]):
        failures.append("identity_route_flip")
    if identity_kl > float(gates["identity_max_token_kl"]):
        failures.append("identity_kl")
    if phase == "calibration" and qualification is None:
        failures.append("missing_smoke_qualification")

    if "quantizer_ledger" in failures:
        invalid_code = "INVALID_QUANTIZATION_PATH"
    elif "trajectory_steps" in failures or "trajectory_matrix" in failures:
        invalid_code = "INVALID_INCOMPLETE_RUN"
    elif failures:
        invalid_code = "INVALID_STATE_OR_NUMERICAL_CONTROL"
    else:
        invalid_code = None
    return {
        "schema_version": "routeguard-kv-integrity-v1",
        "status": "PASS" if not failures else "FAIL",
        "decision_code": invalid_code,
        "failures": failures,
        "trajectory_count": len(trajectories),
        "expected_trajectory_count": expected_trajectory_count,
        "quantizer_ledger_failure_count": len(quant_failures),
        "incomplete_trajectory_count": len(incomplete),
        "patched_route_mismatch_max": patched_route_mismatch,
        "patched_max_token_kl": patched_kl,
        "patched_max_abs_logit_error": patched_logit,
        "identity_route_flip_max": identity_route_flip,
        "identity_max_token_kl": identity_kl,
        "locked_executed_route_mismatch_max": locked_executed_mismatch,
        "cache_storage_alias_count": 0,
        "peak_allocated_gib": peak_allocated_gib,
        "peak_reserved_gib": peak_reserved_gib,
        "resume_count": resume_count,
        "qualification": qualification,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-provenance", type=Path, required=True)
    parser.add_argument("--historical-registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("smoke", "calibration", "formal"), required=True)
    parser.add_argument("--approval", type=Path)
    parser.add_argument("--qualification", type=Path)
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()

    experiment_dir = Path(__file__).resolve().parent
    repo_root = Path(__file__).resolve().parents[4]
    canonical_config_path = experiment_dir / "configs/r0a_5090_v1.json"
    if args.config.resolve() != canonical_config_path.resolve():
        raise RunError("runner only accepts the canonical frozen config path")
    config = load_config(args.config)
    if args.phase == "calibration" and args.qualification is None:
        raise RunError("calibration phase requires --qualification from smoke")
    if args.phase == "formal":
        if args.approval is None:
            raise RunError("formal phase requires --approval")
        if args.qualification is None:
            raise RunError("formal phase requires --qualification from calibration")
    # Hash-only binding and approval happen before opening sealed manifest text
    # or creating a formal output directory.
    qualification_paths = (
        qualification_artifact_paths(args.qualification)
        if args.phase in {"calibration", "formal"}
        else []
    )
    preflight_bindings = build_frozen_bindings(
        repo_root=repo_root,
        config_path=args.config,
        manifest_path=args.manifest,
        source_paths=[
            *(experiment_dir / name for name in SOURCE_NAMES),
            args.data_provenance,
            args.historical_registry,
            *qualification_paths,
        ],
    )
    preflight_binding_digest = canonical_json_sha256(preflight_bindings)
    if args.phase == "formal":
        assert_formal_approval(args.approval, config, preflight_binding_digest)
    manifest = _load_and_validate_manifest(args.manifest, config, args.phase)
    data_provenance = _validate_data_provenance(
        args.data_provenance,
        args.historical_registry,
        config_path=args.config,
        config=config,
        manifest=manifest,
        phase=args.phase,
    )
    bindings, resume_count = _prepare_output(
        output_dir=args.output_dir,
        repo_root=repo_root,
        experiment_dir=experiment_dir,
        config_path=args.config,
        manifest_path=args.manifest,
        provenance_path=args.data_provenance,
        registry_path=args.historical_registry,
        data_provenance_summary=data_provenance,
        extra_binding_paths=qualification_paths,
        phase=args.phase,
        config=config,
    )
    if canonical_json_sha256(bindings) != preflight_binding_digest:
        raise RunError("preflight and output binding calculation diverged")

    _run_cpu_qualification(experiment_dir, args.output_dir, resume_count)

    _configure_runtime(config)
    assert_5090_environment(config)
    current_environment = environment_snapshot()
    core_binding = _core_source_binding(experiment_dir, config, repo_root)
    qualification = None
    if args.phase in {"calibration", "formal"}:
        qualification = _load_qualification(
            args.qualification,
            expected_phase="smoke" if args.phase == "calibration" else "calibration",
            expected_core_binding=core_binding,
            current_environment=current_environment,
        )

    torch.cuda.reset_peak_memory_stats()
    model, tokenizer = _load_model_and_tokenizer(config, local_files_only=not args.allow_download)
    for document in manifest:
        encoded = tokenizer(
            document["text"],
            add_special_tokens=True,
            return_tensors="pt",
            truncation=False,
        )["input_ids"]
        if encoded.shape[1] < int(config["dataset"]["required_tokens"]):
            raise RunError(f"manifest document became too short: {document['text_sha256']}")
        token_ids = encoded.to("cuda")
        for prompt_length in [int(value) for value in config["dataset"]["prompt_lengths"]]:
            with torch.inference_mode():
                prefill = model(
                    input_ids=token_ids[:, :prompt_length],
                    use_cache=True,
                    output_router_logits=False,
                    return_dict=True,
                    logits_to_keep=1,
                )
                prompt_cache = prefill.past_key_values
                if prompt_cache.get_seq_length() != prompt_length:
                    raise RunError("prefill cache length mismatch")
                del prefill
                reference = _run_reference(
                    model,
                    prompt_cache,
                    token_ids,
                    document=document,
                    prompt_length=prompt_length,
                    config=config,
                )
                _store_trajectory(args.output_dir, reference.trajectory)

                for target, arm, controller_mode in _treatment_plan(
                    config, args.phase, str(document["text_sha256"]), prompt_length
                ):
                    if target == "patched_bf16":
                        cache = clone_dynamic_cache(prompt_cache, config=model.config)
                        ledger = None
                    else:
                        cache = QuantizedDynamicCache.from_prompt_cache(
                            prompt_cache,
                            target=target if target in {"k_only", "v_only", "kv"} else "identity",
                            config=model.config,
                        )
                        ledger = cache.ledger
                    assert_no_storage_aliases([prompt_cache, cache])
                    trajectory = _run_treatment(
                        model,
                        cache,
                        token_ids,
                        reference,
                        document=document,
                        prompt_length=prompt_length,
                        target=target,
                        arm=arm,
                        controller_mode=controller_mode,
                        config=config,
                        quantizer_ledger=ledger,
                    )
                    _store_trajectory(args.output_dir, trajectory)
                    del cache, trajectory
                    gc.collect()
                    torch.cuda.empty_cache()
                    peak = torch.cuda.max_memory_allocated() / (1024**3)
                    if peak > float(config["hardware"]["peak_allocated_gib_max"]):
                        raise RunError(f"peak allocated memory exceeded frozen cap: {peak:.3f} GiB")
                del reference, prompt_cache
        del token_ids
        gc.collect()
        torch.cuda.empty_cache()

    # Include resumed trajectory files in the final integrity replay.
    all_trajectories = [
        load_json(path) for path in sorted((args.output_dir / "trajectories").glob("*.json"))
    ]
    peak_allocated_gib = torch.cuda.max_memory_allocated() / (1024**3)
    peak_reserved_gib = torch.cuda.max_memory_reserved() / (1024**3)
    integrity = _integrity_summary(
        phase=args.phase,
        trajectories=all_trajectories,
        config=config,
        resume_count=resume_count,
        peak_allocated_gib=peak_allocated_gib,
        peak_reserved_gib=peak_reserved_gib,
        qualification=qualification,
    )
    if (args.output_dir / "integrity.json").exists():
        raise RunError("integrity.json already exists; completed runs are immutable")
    write_json_no_overwrite(args.output_dir / "integrity.json", integrity, mode=0o600)
    append_jsonl_fsync(
        args.output_dir / "journal.jsonl",
        {"event": "run_complete", "integrity_status": integrity["status"]},
    )
    print(
        f"R0A_RUN_COMPLETE phase={args.phase} trajectories={len(all_trajectories)} "
        f"integrity={integrity['status']} peak_allocated_gib={peak_allocated_gib:.3f} "
        f"peak_reserved_gib={peak_reserved_gib:.3f}"
    )
    if integrity["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    try:
        main()
    except (ArtifactError, RunError, KeyError, ValueError) as exc:
        raise SystemExit(f"R0A_RUN_FAILED: {exc}") from exc
