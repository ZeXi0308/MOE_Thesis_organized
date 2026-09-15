#!/usr/bin/env python3
"""Run the N0d matched-prestate serial-vs-batch router-logit Gate.

The diagnostic uses a fresh, sealed custom continuous-decode capture only to
bind request and generated-token identity.  At every measured decode step it
clones one canonical per-request KV state into three independent branches:
serial-A, fixed batch width four, and serial-B.  Serial-B is a negative control;
the batched branch is never used to construct a later canonical state.

This is an execution-conformance source-localization experiment.  It does not
measure serving capacity, run a scheduling action, or authorize a Controller.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Any, Mapping, Sequence


SCHEMA = "n0d-matched-prestate-router-gate-v1"
CLAIM_CEILING = "CUSTOM_TRANSFORMERS_MATCHED_PRESTATE_CONFORMANCE_ONLY"
EXPECTED_REPO_HEAD = "b141c1d587fe2c918643c3c7c3a8f5f5157d4c8a"
EXPECTED_WORKLOAD_SHA256 = (
    "47babe9d8f875fda3457a68ca83ee7d1274866ebc47013622691d1fc1b556a6d"
)
REQUESTS = 4
DECODE_STEPS = 8
BATCH_SIZE = 4
PROCESS_REPEATS = 3
GIT_BIN = "/usr/bin/git"
ARM_ORDERS = (
    ("serial_a", "batch_4", "serial_b"),
    ("batch_4", "serial_b", "serial_a"),
    ("serial_b", "serial_a", "batch_4"),
)
ROUTER_RAW_DTYPES = frozenset(
    {"torch.float16", "torch.bfloat16", "torch.float32", "torch.float64"}
)

REQUIRED_REPOSITORY_FILES = {
    "docs/ideas/bcrd/experiments/capture_continuous_decode.py": (
        "564d9fb6734462789eaca9bf0cf5cfd1ff8a04271a923cacf021015c6893b2db"
    ),
    "docs/ideas/bcrd/experiments/configs/gate0_continuous_decode_v1.json": (
        "5664e1e457548b6564a1bf3d24af5c3d2d98c1d1ddbd6510a93556ea49042de4"
    ),
    "docs/ideas/bcrd/experiments/configs/workloads/olmoe.formal.json": (
        "2bf4b4897c15b165fea90d730ed9136d0777535daab7f6807336c09a7c70cdbe"
    ),
    "docs/ideas/bcrd/experiments/core.py": (
        "9115acf75ab60eeb9145521e1de7fb8be14455c85b7d1d85a4ad6ac7ab8be575"
    ),
    "docs/ideas/route_shape_slo/v2_capacity_envelope/experiments/"
    "capture_dev_continuous_decode.py": (
        "5cda4159c94662e07efc07ff02ba42df31f1c9c5268b37d6757ff378df490f86"
    ),
    "docs/ideas/route_shape_slo/v2_capacity_envelope/experiments/"
    "compare_serial_batched_router_logits.py": (
        "ab4cb2e1f3091d55f8f1952b00de4a45673f52465211d8d70a6c19e0a816cd1b"
    ),
    "docs/ideas/route_shape_slo/v2_capacity_envelope/experiments/"
    "olmoe_dev_workload.json": (
        "0fc7fccd168b231d62812907cf1df6d352a130798fa4b7efbe521b9c62be60f6"
    ),
    "docs/ideas/route_shape_slo/v2_capacity_envelope/experiments/"
    "prepare_dev_workloads.py": (
        "1d3f753fa837222cfe892e9ce2cc9cbe3c3d14aff2466503cfbe76a711a01d88"
    ),
    "experiments/shared/modeling.py": (
        "f98269bd3084988cc952a272c8d6eec97f50e189d7689b9c581c2c170c4a623e"
    ),
}


class GateError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_source_identity(source_root: Path) -> dict[str, Any]:
    """Bind every runtime-relevant tracked file, not just repository HEAD."""

    root = source_root.resolve()
    head = subprocess.run(
        [GIT_BIN, "rev-parse", "HEAD"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    if head.returncode != 0 or head.stdout.strip() != EXPECTED_REPO_HEAD:
        raise GateError("N0d source checkout is not the frozen repository HEAD")
    observed: dict[str, str] = {}
    for relative, expected in REQUIRED_REPOSITORY_FILES.items():
        path = root / relative
        if not path.is_file():
            raise GateError(f"required N0d source is missing: {relative}")
        value = sha256_file(path)
        if value != expected:
            raise GateError(f"required N0d source hash drifted: {relative}")
        observed[relative] = value
    status = subprocess.run(
        [
            GIT_BIN,
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--",
            *sorted(REQUIRED_REPOSITORY_FILES),
        ],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    if status.returncode != 0 or status.stdout.strip():
        raise GateError("runtime-relevant N0d source paths are not clean")
    return {
        "repo_head": EXPECTED_REPO_HEAD,
        "relevant_paths_clean": True,
        "files_sha256": observed,
    }


def load_comparator(source_root: Path) -> Any:
    path = (
        source_root
        / "docs/ideas/route_shape_slo/v2_capacity_envelope/experiments/"
        "compare_serial_batched_router_logits.py"
    ).resolve()
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("n0d_frozen_comparator", path)
    if spec is None or spec.loader is None:
        raise GateError(f"cannot import frozen comparator: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def validate_import_binding(
    comparator: Any, source_root: Path, *, require_modeling: bool
) -> dict[str, str]:
    root = source_root.resolve()
    expected = {
        "comparator": (
            root
            / "docs/ideas/route_shape_slo/v2_capacity_envelope/experiments/"
            "compare_serial_batched_router_logits.py"
        ),
        "development_wrapper": (
            root
            / "docs/ideas/route_shape_slo/v2_capacity_envelope/experiments/"
            "capture_dev_continuous_decode.py"
        ),
        "producer": root / "docs/ideas/bcrd/experiments/capture_continuous_decode.py",
        "core": root / "docs/ideas/bcrd/experiments/core.py",
    }
    modules = {
        "comparator": comparator,
        "development_wrapper": sys.modules.get("capture_dev_continuous_decode"),
        "producer": comparator.PRODUCER,
        "core": sys.modules.get("core"),
    }
    if require_modeling:
        expected["modeling"] = root / "experiments/shared/modeling.py"
        modules["modeling"] = sys.modules.get("modeling")
    observed: dict[str, str] = {}
    for role, expected_path in expected.items():
        module = modules.get(role)
        actual = Path(str(getattr(module, "__file__", ""))).resolve()
        if module is None or actual != expected_path.resolve():
            raise GateError(
                f"N0d import binding failed for {role}: {actual} != {expected_path}"
            )
        relative = actual.relative_to(root).as_posix()
        expected_hash = REQUIRED_REPOSITORY_FILES.get(relative)
        if expected_hash is not None and sha256_file(actual) != expected_hash:
            raise GateError(f"N0d imported source hash drifted for {role}")
        observed[role] = str(actual)
    return observed


def _clone_cache(torch: Any, producer: Any, cache: Any) -> Any:
    layers = []
    for key, value in producer._legacy_cache(cache):
        layers.append((key.detach().clone(), value.detach().clone()))
    return producer._dynamic_cache(tuple(layers))


def clone_states(torch: Any, producer: Any, states: Sequence[Any]) -> list[Any]:
    clones: list[Any] = []
    for state in states:
        clones.append(
            SimpleNamespace(
                spec=state.spec,
                cache=_clone_cache(torch, producer, state.cache),
                attention_mask=state.attention_mask.detach().clone(),
                next_token=state.next_token.detach().clone(),
                prompt_length=int(state.prompt_length),
                decode_step=int(state.decode_step),
            )
        )
    return clones


def validate_forks(
    torch: Any,
    producer: Any,
    canonical: Sequence[Any],
    forks: Mapping[str, Sequence[Any]],
) -> dict[str, Any]:
    total_tensors = 0
    total_elements = 0
    for arm, states in forks.items():
        if len(states) != len(canonical):
            raise GateError(f"state fork width differs for {arm}")
        for source, branch in zip(canonical, states):
            if source.spec.request_id != branch.spec.request_id:
                raise GateError(f"state fork request identity differs for {arm}")
            if not torch.equal(source.attention_mask, branch.attention_mask):
                raise GateError(f"state fork attention mask differs for {arm}")
            if not torch.equal(source.next_token, branch.next_token):
                raise GateError(f"state fork next token differs for {arm}")
            if source.attention_mask.data_ptr() == branch.attention_mask.data_ptr():
                raise GateError(f"state fork attention storage aliases for {arm}")
            source_layers = producer._legacy_cache(source.cache)
            branch_layers = producer._legacy_cache(branch.cache)
            if len(source_layers) != len(branch_layers):
                raise GateError(f"state fork cache layer count differs for {arm}")
            for (source_key, source_value), (branch_key, branch_value) in zip(
                source_layers, branch_layers
            ):
                for left, right in (
                    (source_key, branch_key),
                    (source_value, branch_value),
                ):
                    if not torch.equal(left, right):
                        raise GateError(f"state fork KV value differs for {arm}")
                    if left.data_ptr() == right.data_ptr():
                        raise GateError(f"state fork KV storage aliases for {arm}")
                    total_tensors += 1
                    total_elements += int(left.numel())
    return {
        "arms": sorted(forks),
        "requests": len(canonical),
        "independent_equal_tensors_checked": total_tensors,
        "independent_equal_elements_checked": total_elements,
    }


def _empty_trace() -> dict[str, list[dict[str, Any]]]:
    return {"tokens": [], "router": []}


def _serial_step(
    torch: Any,
    comparator: Any,
    model: Any,
    states: Sequence[Any],
    decode_step: int,
) -> tuple[dict[str, Any], list[Any]]:
    producer = comparator.PRODUCER
    trace = _empty_trace()
    outputs: list[Any] = []
    for state in states:
        input_token = int(state.next_token.item())
        prior_length = producer._cache_length(state.cache)
        attention_mask = torch.cat(
            (state.attention_mask, state.attention_mask.new_ones((1, 1))), dim=1
        )
        position_ids = attention_mask.long().cumsum(-1)[:, -1:] - 1
        with torch.inference_mode():
            output, _ = producer._timed_call(
                model,
                "n0d_serial_decode",
                1,
                None,
                input_ids=state.next_token,
                attention_mask=attention_mask,
                position_ids=position_ids,
                cache_position=torch.tensor(
                    [prior_length], dtype=torch.long, device=state.next_token.device
                ),
                past_key_values=state.cache,
                use_cache=True,
                output_router_logits=True,
                return_dict=True,
            )
        cache = getattr(output, "past_key_values", None)
        logits = getattr(output, "logits", None)
        if cache is None or logits is None or producer._cache_length(cache) != prior_length + 1:
            raise GateError("N0d serial decode did not append exactly one KV position")
        predicted = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        predicted_token = int(predicted.item())
        trace["tokens"].append(
            {
                "request_id": state.spec.request_id,
                "decode_step": decode_step,
                "input_token_id": input_token,
                "predicted_next_token_id": predicted_token,
            }
        )
        trace["router"].extend(
            comparator._router_rows(
                torch,
                output,
                model=model,
                request_ids=[state.spec.request_id],
                decode_step=decode_step,
                input_tokens=[input_token],
                predicted_tokens=[predicted_token],
            )
        )
        outputs.append(
            SimpleNamespace(
                spec=state.spec,
                cache=cache,
                attention_mask=attention_mask,
                next_token=predicted,
                prompt_length=state.prompt_length,
                decode_step=decode_step + 1,
            )
        )
    return trace, outputs


def _batched_step(
    torch: Any,
    comparator: Any,
    model: Any,
    states: Sequence[Any],
    decode_step: int,
) -> tuple[dict[str, Any], list[Any]]:
    producer = comparator.PRODUCER
    (
        input_ids,
        attention_mask,
        position_ids,
        cache,
        prior_lengths,
        prior_max,
    ) = producer._pad_decode_inputs(states)
    if int(input_ids.shape[0]) != BATCH_SIZE:
        raise GateError("N0d batched branch did not preserve width four")
    with torch.inference_mode():
        output, _ = producer._timed_call(
            model,
            "n0d_batched_decode",
            BATCH_SIZE,
            None,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=torch.tensor(
                [prior_max], dtype=torch.long, device=input_ids.device
            ),
            past_key_values=cache,
            use_cache=True,
            output_router_logits=True,
            return_dict=True,
        )
    logits = getattr(output, "logits", None)
    output_cache = getattr(output, "past_key_values", None)
    if logits is None or output_cache is None:
        raise GateError("N0d batched decode returned no logits or KV")
    split_caches = producer.split_left_padded_cache(
        output_cache,
        prior_lengths=prior_lengths,
        prior_max_length=prior_max,
    )
    predicted = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
    request_ids = [state.spec.request_id for state in states]
    input_tokens = [int(input_ids[index].item()) for index in range(BATCH_SIZE)]
    predicted_tokens = [int(predicted[index].item()) for index in range(BATCH_SIZE)]
    trace = _empty_trace()
    trace["router"].extend(
        comparator._router_rows(
            torch,
            output,
            model=model,
            request_ids=request_ids,
            decode_step=decode_step,
            input_tokens=input_tokens,
            predicted_tokens=predicted_tokens,
        )
    )
    outputs: list[Any] = []
    for index, state in enumerate(states):
        trace["tokens"].append(
            {
                "request_id": state.spec.request_id,
                "decode_step": decode_step,
                "input_token_id": input_tokens[index],
                "predicted_next_token_id": predicted_tokens[index],
            }
        )
        outputs.append(
            SimpleNamespace(
                spec=state.spec,
                cache=split_caches[index],
                attention_mask=torch.cat(
                    (
                        state.attention_mask,
                        state.attention_mask.new_ones((1, 1)),
                    ),
                    dim=1,
                ),
                next_token=predicted[index : index + 1],
                prompt_length=state.prompt_length,
                decode_step=decode_step + 1,
            )
        )
    return trace, outputs


def run_matched_trajectory(
    torch: Any,
    comparator: Any,
    model: Any,
    requests: Sequence[Any],
    *,
    arm_order: Sequence[str],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    producer = comparator.PRODUCER
    canonical = comparator._prefill_states(torch, model, requests)
    traces = {arm: _empty_trace() for arm in ("serial_a", "batch_4", "serial_b")}
    fork_checks: list[dict[str, Any]] = []
    for decode_step in range(DECODE_STEPS):
        forks = {
            arm: clone_states(torch, producer, canonical)
            for arm in ("serial_a", "batch_4", "serial_b")
        }
        fork_checks.append(validate_forks(torch, producer, canonical, forks))
        results: dict[str, tuple[dict[str, Any], list[Any]]] = {}
        for arm in arm_order:
            results[arm] = (
                _batched_step(torch, comparator, model, forks[arm], decode_step)
                if arm == "batch_4"
                else _serial_step(torch, comparator, model, forks[arm], decode_step)
            )
            traces[arm]["tokens"].extend(results[arm][0]["tokens"])
            traces[arm]["router"].extend(results[arm][0]["router"])
        canonical = results["serial_a"][1]
    return traces, fork_checks


def first_divergences(
    serial_trace: Mapping[str, Any], batched_trace: Mapping[str, Any]
) -> dict[str, Any]:
    """Build request-step-local causal frontiers and the first global set.

    A router-logit value difference can explain an assignment difference only
    inside the same ``(request_id, decode_step)`` cell and at a layer no later
    than the assignment layer.  In particular, a difference from a prior
    decode step is state history, not a pre-top-k source for a later matched
    intervention.
    """

    fields = ("request_id", "decode_step", "layer")

    def index(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[Any, ...], Mapping[str, Any]]:
        result: dict[tuple[Any, ...], Mapping[str, Any]] = {}
        for row in rows:
            key = tuple(row[field] for field in fields)
            if key in result:
                raise GateError(f"duplicate router identity: {key}")
            result[key] = row
        return result

    left = index(list(serial_trace.get("router", [])))
    right = index(list(batched_trace.get("router", [])))
    if left.keys() != right.keys():
        raise GateError("serial/batched router identities do not align")
    request_steps = sorted(
        {(str(key[0]), int(key[1])) for key in left},
        key=lambda item: (item[1], item[0]),
    )
    per_request_step: list[dict[str, Any]] = []
    for request_id, decode_step in request_steps:
        keys = sorted(
            (
                key
                for key in left
                if str(key[0]) == request_id and int(key[1]) == decode_step
            ),
            key=lambda key: int(key[2]),
        )
        first_logit = None
        first_assignment = None
        for key in keys:
            serial_logits = [float(value) for value in left[key]["router_logits"]]
            batch_logits = [float(value) for value in right[key]["router_logits"]]
            if len(serial_logits) != len(batch_logits) or not serial_logits:
                raise GateError("router-logit vector width differs at causal frontier")
            serial_dtype = left[key].get("router_logits_dtype_before_float32_copy")
            batch_dtype = right[key].get("router_logits_dtype_before_float32_copy")
            if (
                serial_dtype not in ROUTER_RAW_DTYPES
                or batch_dtype not in ROUTER_RAW_DTYPES
                or serial_dtype != batch_dtype
            ):
                raise GateError("router-logit source dtype differs at causal frontier")
            deltas = [
                abs(serial_logits[index] - batch_logits[index])
                for index in range(len(serial_logits))
            ]
            common = {
                "request_id": request_id,
                "decode_step": int(key[1]),
                "layer": int(key[2]),
            }
            if first_logit is None and any(delta != 0.0 for delta in deltas):
                first_logit = {
                    **common,
                    "max_abs_logit_delta": max(deltas),
                    "source_dtype": serial_dtype,
                }
            serial_experts = sorted(
                int(value) for value in left[key]["selected_experts"]
            )
            batch_experts = sorted(
                int(value) for value in right[key]["selected_experts"]
            )
            if first_assignment is None and serial_experts != batch_experts:
                first_assignment = {
                    **common,
                    "serial_experts": serial_experts,
                    "batched_experts": batch_experts,
                    "max_abs_logit_delta_at_assignment": max(deltas),
                    "serial_selection_boundary": left[key]["topk_margins"][
                        "selection_boundary"
                    ],
                    "batched_selection_boundary": right[key]["topk_margins"][
                        "selection_boundary"
                    ],
                }
        category = None
        if first_assignment is not None:
            category = (
                "PRE_TOPK_NUMERICAL_DIVERGENCE"
                if first_logit is not None
                and int(first_logit["layer"]) <= int(first_assignment["layer"])
                else "RECONSTRUCTED_TOPK_INCONSISTENCY"
            )
        per_request_step.append(
            {
                "request_id": request_id,
                "decode_step": decode_step,
                "first_router_logit_value_difference": first_logit,
                "first_expert_assignment_difference": first_assignment,
                "assignment_source_category": category,
            }
        )

    logit_rows = [
        row["first_router_logit_value_difference"]
        for row in per_request_step
        if row["first_router_logit_value_difference"] is not None
    ]
    assignment_rows = [
        row
        for row in per_request_step
        if row["first_expert_assignment_difference"] is not None
    ]
    logit_frontier: list[dict[str, Any]] = []
    if logit_rows:
        position = min((int(row["decode_step"]), int(row["layer"])) for row in logit_rows)
        logit_frontier = sorted(
            [
                row
                for row in logit_rows
                if (int(row["decode_step"]), int(row["layer"])) == position
            ],
            key=lambda row: str(row["request_id"]),
        )
    assignment_frontier: list[dict[str, Any]] = []
    signature = None
    if assignment_rows:
        position = min(
            (
                int(row["first_expert_assignment_difference"]["decode_step"]),
                int(row["first_expert_assignment_difference"]["layer"]),
            )
            for row in assignment_rows
        )
        frontier_rows = sorted(
            [
                row
                for row in assignment_rows
                if (
                    int(row["first_expert_assignment_difference"]["decode_step"]),
                    int(row["first_expert_assignment_difference"]["layer"]),
                )
                == position
            ],
            key=lambda row: str(row["request_id"]),
        )
        assignment_frontier = [
            {
                **row["first_expert_assignment_difference"],
                "assignment_source_category": row["assignment_source_category"],
                "first_router_logit_value_difference": row[
                    "first_router_logit_value_difference"
                ],
            }
            for row in frontier_rows
        ]
        categories = {
            str(row["assignment_source_category"]) for row in frontier_rows
        }
        global_category = (
            next(iter(categories)) if len(categories) == 1 else "MIXED_CAUSAL_FRONTIER"
        )
        signature = {
            "decode_step": position[0],
            "layer": position[1],
            "category": global_category,
            "records": [
                {
                    "request_id": row["request_id"],
                    "serial_experts": row["serial_experts"],
                    "batched_experts": row["batched_experts"],
                    "assignment_source_category": row[
                        "assignment_source_category"
                    ],
                    "first_router_logit_position": (
                        {
                            "decode_step": int(
                                row["first_router_logit_value_difference"][
                                    "decode_step"
                                ]
                            ),
                            "layer": int(
                                row["first_router_logit_value_difference"]["layer"]
                            ),
                        }
                        if row["assignment_source_category"]
                        == "PRE_TOPK_NUMERICAL_DIVERGENCE"
                        else None
                    ),
                }
                for row in assignment_frontier
            ],
        }
    return {
        "per_request_step": per_request_step,
        "first_router_logit_value_difference_frontier": logit_frontier,
        "first_expert_assignment_frontier": assignment_frontier,
        "causal_frontier_signature": signature,
    }


def first_divergence_signature(value: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return only the causal-frontier identity/category used by the Gate."""

    signature = value.get("causal_frontier_signature")
    if not isinstance(signature, Mapping):
        return None
    return dict(signature)


def classify_gate(
    *,
    source_batch_dependence: bool,
    serial_negative_control_exact: bool,
    token_parity: bool,
    assignment_changed: bool,
    double_sided_signature_match: bool,
    first_category: str | None,
) -> str:
    if not source_batch_dependence:
        return "STOP_FRESH_CAPTURE_NO_BATCH_DEPENDENT_ROUTE"
    if not serial_negative_control_exact:
        return "INVALID_SERIAL_NEGATIVE_CONTROL"
    if not token_parity:
        return "STOP_TOKEN_PARITY_FAILED"
    if not assignment_changed:
        return "STOP_MATCHED_PRESTATE_NO_ASSIGNMENT_DIVERGENCE"
    if not double_sided_signature_match:
        return "STOP_BATCH_CONTRAST_NOT_DOUBLE_SIDED"
    if first_category == "PRE_TOPK_NUMERICAL_DIVERGENCE":
        return "PROCESS_CANDIDATE_PRETOPK_NUMERICAL_DIVERGENCE_ASSOCIATION"
    return "INCONCLUSIVE_RECONSTRUCTED_TOPK_INCONSISTENCY"


def _write_json_once(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(
                value,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except FileExistsError as exc:
        raise GateError(f"refusing to overwrite N0d output: {path}") from exc
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def process_identity() -> dict[str, Any]:
    pid = os.getpid()
    try:
        fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(
            encoding="utf-8"
        ).strip()
        start_ticks = int(fields[21])
    except (OSError, IndexError, ValueError) as exc:
        raise GateError(f"cannot bind N0d fresh-process identity: {exc}") from exc
    return {"pid": pid, "start_time_ticks": start_ticks, "boot_id": boot_id}


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_root = Path(args.source_root).resolve()
    source_identity = validate_source_identity(source_root)
    runner_sha256 = sha256_file(Path(__file__).resolve())
    workload_path = Path(args.workload_manifest).resolve()
    if sha256_file(workload_path) != EXPECTED_WORKLOAD_SHA256:
        raise GateError("prepared N0d steady workload hash drifted")
    comparator = load_comparator(source_root)
    import_binding = validate_import_binding(
        comparator, source_root, require_modeling=False
    )
    producer = comparator.PRODUCER
    capture_dir = Path(args.capture_dir).resolve()
    manifest, complete, captured_manifest_path = comparator.load_capture_contract(capture_dir)
    if comparator._sha256_file(captured_manifest_path) != EXPECTED_WORKLOAD_SHA256:
        raise GateError("fresh capture did not bind the frozen N0d steady workload")
    captured_environment = comparator.load_captured_environment(capture_dir, complete)
    reference_ids = comparator.select_request_ids(manifest, REQUESTS)
    reference_tokens = comparator.load_reference_tokens(
        capture_dir, complete, reference_ids, DECODE_STEPS
    )
    if comparator.query_gpu_compute_processes():
        raise GateError("GPU is not idle before N0d model load")
    torch, transformers, tokenizer, model, load_seconds = comparator.load_exact_model(manifest)
    import_binding = validate_import_binding(
        comparator, source_root, require_modeling=True
    )
    current_environment = comparator._environment(torch, transformers)
    runtime_validation = comparator.validate_runtime_environment(
        captured_environment, current_environment
    )
    processes = comparator.query_gpu_compute_processes()
    if len(processes) != 1:
        raise GateError("N0d model load did not create one isolated GPU process")
    prepared = producer._prepare_requests(manifest, tokenizer, model.device)
    by_id = {request.request_id: request for request in prepared}
    requests = [by_id[request_id] for request_id in reference_ids]
    token_identity = comparator.validate_tokenizer_and_prompt_identity(
        manifest, tokenizer, prepared, reference_ids
    )
    seed = int(manifest["seed"])
    process_repeat = int(args.process_repeat)
    if process_repeat < 0 or process_repeat >= PROCESS_REPEATS:
        raise GateError("--process-repeat must be 0, 1, or 2")
    arm_order = ARM_ORDERS[process_repeat]
    monitor = comparator.GpuIsolationMonitor(processes)
    monitor.start()
    try:
        comparator._reset_rng(torch, seed)
        run_matched_trajectory(
            torch,
            comparator,
            model,
            requests,
            arm_order=("serial_a", "batch_4", "serial_b"),
        )
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)
        started = time.monotonic()
        monitor.check("before_measured_trajectory")
        monitor.require_clean()
        comparator._reset_rng(torch, seed)
        measured, fork_checks = run_matched_trajectory(
            torch, comparator, model, requests, arm_order=arm_order
        )
        monitor.check("after_measured_trajectory")
        monitor.require_clean()
        torch.cuda.synchronize(0)
        elapsed_seconds = time.monotonic() - started
    finally:
        monitor.stop()
    monitor.require_clean()

    serial_a = measured["serial_a"]
    batch_4 = measured["batch_4"]
    serial_b = measured["serial_b"]
    control = comparator.summarize_trace_pair(serial_a, serial_b)
    serial_negative_control_exact = bool(
        control["tokens"]["full_token_parity"]
        and control["router_logits"]["record_exact_match_fraction"] == 1.0
        and control["expert_assignment"]["ordered_match_fraction"] == 1.0
    )
    cross_a = comparator.summarize_trace_pair(serial_a, batch_4)
    cross_b = comparator.summarize_trace_pair(serial_b, batch_4)
    assignment_changed = bool(
        cross_a["expert_assignment"]["different_multiset_records"] > 0
        and cross_b["expert_assignment"]["different_multiset_records"] > 0
    )
    first_a = first_divergences(serial_a, batch_4)
    first_b = first_divergences(serial_b, batch_4)
    signature_a = first_divergence_signature(first_a)
    signature_b = first_divergence_signature(first_b)
    double_sided_signature_match = bool(
        signature_a is not None and signature_a == signature_b
    )
    reference = {
        "serial_a": comparator._reference_summary([serial_a], reference_tokens),
        "batch_4": comparator._reference_summary([batch_4], reference_tokens),
        "serial_b": comparator._reference_summary([serial_b], reference_tokens),
    }
    token_parity = bool(
        cross_a["tokens"]["full_token_parity"]
        and cross_b["tokens"]["full_token_parity"]
        and all(value["all_repeats_match"] for value in reference.values())
    )
    source_batch_dependence = bool(
        complete["serial_audit"].get("batch_dependent_route_observed", False)
    )
    status = classify_gate(
        source_batch_dependence=source_batch_dependence,
        serial_negative_control_exact=serial_negative_control_exact,
        token_parity=token_parity,
        assignment_changed=assignment_changed,
        double_sided_signature_match=double_sided_signature_match,
        first_category=(signature_a or {}).get("category"),
    )
    payload = {
        "schema": SCHEMA,
        "status": status,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "claim_ceiling": CLAIM_CEILING,
        "capacity_claim_authorized": False,
        "action_oracle_authorized": False,
        "controller_authorized": False,
        "source_identity": {
            **source_identity,
            "runner": str(Path(__file__).resolve()),
            "runner_sha256": runner_sha256,
            "actual_import_paths": import_binding,
        },
        "fresh_capture": {
            "capture_dir": str(capture_dir),
            "capture_complete_sha256": sha256_file(capture_dir / "CAPTURE_COMPLETE.json"),
            "serial_audit": complete["serial_audit"],
            "source_batch_dependence": source_batch_dependence,
            "reference_tokens": reference_tokens,
        },
        "execution": {
            "process_identity": process_identity(),
            "requests": REQUESTS,
            "request_ids": reference_ids,
            "decode_steps": DECODE_STEPS,
            "batch_width": BATCH_SIZE,
            "fresh_process_repeat": process_repeat,
            "planned_fresh_process_repeats": PROCESS_REPEATS,
            "arm_order": list(arm_order),
            "canonical_state_advance": "serial_a_only",
            "batch_state_propagated_to_next_step": False,
            "matched_prestate_fork_checks": fork_checks,
            "seed_reset_before_process_trajectory": seed,
            "warmup_trajectory_discarded": True,
            "elapsed_seconds_excluding_model_load": elapsed_seconds,
            "model_load_seconds": load_seconds,
            "peak_cuda_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
            "runtime_validation": runtime_validation,
            "environment": current_environment,
            "token_identity": token_identity,
            "gpu_isolation": monitor.summary(),
        },
        "serial_negative_control": {
            "exact": serial_negative_control_exact,
            "comparison": control,
        },
        "reference_token_parity": {"passed": token_parity, **reference},
        "serial_vs_batch_4": {
            "serial_a_comparison": cross_a,
            "serial_b_comparison": cross_b,
            "first_divergence_vs_serial_a": first_a,
            "first_divergence_vs_serial_b": first_b,
            "first_divergence_signature_vs_serial_a": signature_a,
            "first_divergence_signature_vs_serial_b": signature_b,
            "double_sided_signature_match": double_sided_signature_match,
        },
        "traces": measured,
    }
    _write_json_once(Path(args.output).resolve(), payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--capture-dir", required=True)
    parser.add_argument("--workload-manifest", required=True)
    parser.add_argument("--process-repeat", required=True, type=int)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run(args)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "output": str(Path(args.output).resolve()),
                "claim_ceiling": CLAIM_CEILING,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
