#!/usr/bin/env python3
"""Replay real OLMoE route-divergence events from one canonical pre-step state.

This development-only harness isolates execution conditions.  It does not run a
capacity action and cannot establish safe capacity, native-serving behaviour,
or production latency.  Every measured arm is rebuilt from independently cloned
KV storage and teacher-forced captured tokens.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import shlex
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence


ALLCLOSE_ATOL = 1e-6
ALLCLOSE_RTOL = 1e-5
# Inherited from the checked-in pre-top-k diagnostic before this run.
NEAR_TIE_MARGIN = 1e-2
EVENT_COORDINATES = {
    "steady": [
        ("olmoe-dev-steady-000", 1),
        ("olmoe-dev-steady-002", 5),
        ("olmoe-dev-steady-003", 4),
    ],
    "bursty": [
        ("olmoe-dev-bursty-000", 0),
        ("olmoe-dev-bursty-001", 0),
        ("olmoe-dev-bursty-002", 0),
    ],
}
PROPAGATION_EVENTS = {
    "steady:olmoe-dev-steady-002:5",
    "steady:olmoe-dev-steady-003:4",
    "bursty:olmoe-dev-bursty-000:0",
    "bursty:olmoe-dev-bursty-002:0",
}


class ExperimentError(RuntimeError):
    def __init__(self, message: str, category: str = "INVALID_EXPERIMENT") -> None:
        super().__init__(message)
        self.category = category


def classify_failure(error: BaseException) -> str:
    text = str(error).lower()
    if any(
        value in text
        for value in ("gpu", "nvidia", "cuda", "isolation", "compute process")
    ):
        return "INCONCLUSIVE_ENVIRONMENT_OR_GPU_ISOLATION"
    if any(
        value in text
        for value in (
            "reconstruct",
            "teacher-forced",
            "canonical replay",
            "prefill",
            "state step",
        )
    ):
        return "BLOCKED_STATE_RECONSTRUCTION"
    if any(value in text for value in ("arm c", "source event", "clone")):
        return "INVALID_SOURCE_REPRODUCTION_OR_CLONE_CONTROL"
    if isinstance(error, ExperimentError):
        return error.category
    return "HARNESS_RUNTIME_ERROR"


@dataclass
class ArmInputs:
    name: str
    input_ids: Any
    attention_mask: Any
    position_ids: Any
    cache: Any
    cache_position: Any
    prior_lengths: list[int]
    prior_max: int
    target_row: int
    request_ids: list[str]
    state_storage_ptrs: set[int]
    cache_storage_ptrs: set[int]
    target_logical_cache_sha256: str


def repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / ".git").exists():
            return parent
    raise ExperimentError("cannot locate repository root")


ROOT = repo_root()
V2 = ROOT / "docs/ideas/route_shape_slo/v2_capacity_envelope/experiments"
BCRD = ROOT / "docs/ideas/bcrd/experiments"
sys.path.insert(0, str(V2))
sys.path.insert(0, str(BCRD))
import capture_continuous_decode as CAPTURE  # noqa: E402
import compare_serial_batched_router_logits as BASE  # noqa: E402


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ExperimentError(f"{path} is not a JSON object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ExperimentError(f"{path}:{line_number} is not an object")
            rows.append(value)
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_text_exclusive(path: Path, value: str, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(value)
    if mode is not None:
        path.chmod(mode)


def tensor_sha256(tensor: Any) -> str:
    import torch

    raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def write_json_exclusive(path: Path, value: Any) -> None:
    if path.exists():
        raise ExperimentError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, text=True, capture_output=True, check=False
    )
    if result.returncode:
        raise ExperimentError(result.stderr.strip() or "git command failed")
    return result.stdout.strip()


def load_capture(capture_dir: Path, regime: str) -> dict[str, Any]:
    if not capture_dir.is_dir():
        raise ExperimentError(f"missing capture directory: {capture_dir}")
    status = read_json(capture_dir / "RUN_STATUS.json")
    if status != {"required_sentinel": "CAPTURE_COMPLETE.json", "status": "COMPLETE"}:
        raise ExperimentError(f"capture is not COMPLETE: {capture_dir}")
    complete = read_json(capture_dir / "CAPTURE_COMPLETE.json")
    if complete.get("status") != "CAPTURE_COMPLETE":
        raise ExperimentError("capture completion sentinel is invalid")
    bound = complete.get("files")
    if not isinstance(bound, dict):
        raise ExperimentError("capture sentinel has no file hashes")
    for name, expected in bound.items():
        path = capture_dir / str(name)
        if not path.is_file() or sha256_file(path) != expected:
            raise ExperimentError(f"capture hash mismatch: {path}")
    manifest = CAPTURE.load_workload_manifest(capture_dir / "workload_manifest.json")
    marker = manifest.get("route_capacity_envelope", {})
    if marker.get("arrival_regime") != regime:
        raise ExperimentError("capture regime does not match requested regime")
    ledger_rows = read_jsonl(capture_dir / "request_ledger.jsonl")
    ledger = {str(row["request_id"]): row for row in ledger_rows}
    if len(ledger) != len(ledger_rows):
        raise ExperimentError("request ledger contains duplicate IDs")
    batches = read_jsonl(capture_dir / "decode_batches.jsonl")
    audit = read_json(capture_dir / "serial_audit.json")
    return {
        "dir": capture_dir,
        "complete": complete,
        "manifest": manifest,
        "ledger": ledger,
        "batches": batches,
        "audit": audit,
        "hashes": {
            name: sha256_file(capture_dir / name)
            for name in sorted(bound)
        },
    }


def batch_for_step(
    batches: Sequence[Mapping[str, Any]], request_ids: Sequence[str], step: int
) -> Mapping[str, Any]:
    matches = []
    for row in batches:
        ids = [str(value) for value in row["request_ids"]]
        steps = [int(value) for value in row["decode_steps"]]
        if ids == list(request_ids) and steps == [step] * len(ids):
            matches.append(row)
    if len(matches) != 1:
        raise ExperimentError(
            f"cannot uniquely bind batch for step={step}, ids={list(request_ids)}"
        )
    return matches[0]


def build_selection(captures: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    selected: list[dict[str, Any]] = []
    for regime, coordinates in EVENT_COORDINATES.items():
        capture = captures[regime]
        audit_differences = list(capture["audit"].get("difference_examples", []))
        for request_id, decode_step in coordinates:
            differences = [
                row
                for row in audit_differences
                if str(row.get("request_id")) == request_id
                and int(row.get("decode_step", -1)) == decode_step
            ]
            if not differences:
                raise ExperimentError(
                    f"selected event is not a retained real difference: {request_id}/{decode_step}"
                )
            first = min(differences, key=lambda row: int(row["layer"]))
            target_ledger = capture["ledger"][request_id]
            source_step = target_ledger["steps"][decode_step]
            batch_index = int(source_step["batch_index"])
            batch = next(
                row for row in capture["batches"]
                if int(row["batch_index"]) == batch_index
            )
            request_ids = [str(value) for value in batch["request_ids"]]
            target_row = request_ids.index(request_id)
            if int(batch["decode_steps"][target_row]) != decode_step:
                raise ExperimentError("event/batch decode-step identity does not close")
            if [int(value) for value in batch["decode_steps"]] != [
                decode_step
            ] * len(request_ids):
                raise ExperimentError(
                    "selected event is outside the frozen synchronous-batch subset"
                )
            expected_lengths = [
                int(capture["ledger"][value]["prompt_tokens"]) + decode_step
                for value in request_ids
            ]
            if [int(value) for value in batch["prior_cache_lengths"]] != expected_lengths:
                raise ExperimentError("selected event logical KV identity does not close")
            selected.append(
                {
                    "event_id": f"{regime}:{request_id}:{decode_step}",
                    "episode": str(
                        capture["manifest"]["route_capacity_envelope"]["episode_id"]
                    ),
                    "regime": regime,
                    "request_id": request_id,
                    "document_id": str(target_ledger["document_id"]),
                    "decode_step": decode_step,
                    "original_batch_index": batch_index,
                    "target_row_index": target_row,
                    "original_companion_ids": [
                        value for value in request_ids if value != request_id
                    ],
                    "original_request_ids": request_ids,
                    "original_document_ids": [
                        str(capture["ledger"][value]["document_id"])
                        for value in request_ids
                    ],
                    "logical_kv_lengths": [
                        int(value) for value in batch["prior_cache_lengths"]
                    ],
                    "physical_padded_length": max(
                        int(value) for value in batch["prior_cache_lengths"]
                    ),
                    "left_padding": [int(value) for value in batch["left_padding"]],
                    "forced_token_ids": [
                        int(capture["ledger"][value]["steps"][decode_step]["input_token_id"])
                        for value in request_ids
                    ],
                    "serial_experts": [int(value) for value in first["serial_experts"]],
                    "batched_experts": [int(value) for value in first["batched_experts"]],
                    "first_known_different_layer": int(first["layer"]),
                    "retained_difference_layers": sorted(
                        int(row["layer"]) for row in differences
                    ),
                    "source_artifact_hashes": dict(capture["hashes"]),
                    "selection_limit": (
                        "serial_audit retains at most 16 examples per episode; "
                        "selected events are known real mismatches, not claimed maxima"
                    ),
                }
            )
    return {
        "schema": "longrun-a-event-selection-v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "events": selected,
        "event_count": len(selected),
    }


def plan_matched_alternatives(
    captures: Mapping[str, Mapping[str, Any]], selection: Mapping[str, Any]
) -> dict[str, Any]:
    plans: dict[str, Any] = {}
    for event in selection["events"]:
        original_ids = {str(value) for value in event["original_request_ids"]}
        original_documents = {str(value) for value in event["original_document_ids"]}
        used = set(original_ids)
        used_documents = set(original_documents)
        rows = []
        target_row = int(event["target_row_index"])
        for row_index, logical_length in enumerate(event["logical_kv_lengths"]):
            if row_index == target_row:
                rows.append(
                    {
                        "row": row_index,
                        "logical_kv_length": int(logical_length),
                        "role": "target",
                        "regime": event["regime"],
                        "request_id": event["request_id"],
                        "document_id": event["document_id"],
                        "decode_step": int(event["decode_step"]),
                        "replaced": False,
                    }
                )
                continue
            candidates = []
            for regime in ("steady", "bursty"):
                for request_id, ledger in sorted(captures[regime]["ledger"].items()):
                    if request_id in used:
                        continue
                    document_id = str(ledger["document_id"])
                    if document_id in used_documents:
                        continue
                    prompt_tokens = int(ledger["prompt_tokens"])
                    for decode_step in range(len(ledger["steps"])):
                        if prompt_tokens + decode_step == int(logical_length):
                            candidates.append((regime, request_id, decode_step))
            if candidates:
                regime, request_id, decode_step = candidates[0]
                used.add(request_id)
                document_id = str(captures[regime]["ledger"][request_id]["document_id"])
                used_documents.add(document_id)
                rows.append(
                    {
                        "row": row_index,
                        "logical_kv_length": int(logical_length),
                        "role": "matched_alternative_companion",
                        "regime": regime,
                        "request_id": request_id,
                        "document_id": document_id,
                        "decode_step": decode_step,
                        "replaced": True,
                    }
                )
            else:
                request_id = str(event["original_request_ids"][row_index])
                rows.append(
                    {
                        "row": row_index,
                        "logical_kv_length": int(logical_length),
                        "role": "unmatched_fallback_original_companion",
                        "regime": event["regime"],
                        "request_id": request_id,
                        "document_id": str(
                            captures[str(event["regime"])]["ledger"][request_id]["document_id"]
                        ),
                        "decode_step": int(event["decode_step"]),
                        "replaced": False,
                    }
                )
        plans[str(event["event_id"])] = {
            "rows": rows,
            "companion_rows": len(rows) - 1,
            "replaced_companion_rows": sum(
                int(row["replaced"]) for row in rows if row["role"] != "target"
            ),
            "exact_logical_length_vector": [
                int(value) for value in event["logical_kv_lengths"]
            ],
            "selection_rule": (
                "lexicographically first different request with exact prompt_tokens+decode_step; "
                "freeze before GPU metrics"
            ),
        }
    return {
        "schema": "longrun-a-matched-alternatives-v1",
        "plans": plans,
    }


def cache_storage_ptrs(cache: Any) -> set[int]:
    pointers: set[int] = set()
    for key, value in CAPTURE._legacy_cache(cache):
        pointers.add(int(key.untyped_storage().data_ptr()))
        pointers.add(int(value.untyped_storage().data_ptr()))
    return pointers


def clone_cache(cache: Any) -> Any:
    return CAPTURE._dynamic_cache(
        tuple(
            (key.detach().clone(), value.detach().clone())
            for key, value in CAPTURE._legacy_cache(cache)
        )
    )


def clone_state(state: Any) -> Any:
    return SimpleNamespace(
        spec=state.spec,
        cache=clone_cache(state.cache),
        attention_mask=state.attention_mask.detach().clone(),
        next_token=state.next_token.detach().clone(),
        prompt_length=int(state.prompt_length),
        decode_step=int(state.decode_step),
    )


def logical_cache_sha256(cache: Any, row: int, logical_length: int) -> str:
    import torch

    digest = hashlib.sha256()
    for layer, (key, value) in enumerate(CAPTURE._legacy_cache(cache)):
        for kind, tensor in (("k", key), ("v", value)):
            logical = tensor[row : row + 1, :, -logical_length:, :]
            digest.update(f"{layer}:{kind}:{tuple(logical.shape)}:".encode("ascii"))
            digest.update(
                logical.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
            )
    return digest.hexdigest()


def prefill_state(model: Any, spec: Any, expected_token: int) -> Any:
    import torch

    with torch.inference_mode():
        output, _ = CAPTURE._timed_call(
            model,
            "longrun_a_prefill",
            1,
            None,
            input_ids=spec.input_ids,
            attention_mask=spec.attention_mask,
            use_cache=True,
            output_router_logits=False,
            return_dict=True,
        )
    cache = output.past_key_values
    predicted = torch.argmax(output.logits[:, -1, :], dim=-1, keepdim=True)
    if cache is None or CAPTURE._cache_length(cache) != int(spec.input_ids.shape[1]):
        raise ExperimentError(f"prefill cache closure failed for {spec.request_id}")
    if int(predicted.item()) != expected_token:
        raise ExperimentError(f"prefill token drifted for {spec.request_id}")
    return SimpleNamespace(
        spec=spec,
        cache=cache,
        attention_mask=spec.attention_mask.detach().clone(),
        next_token=predicted,
        prompt_length=int(spec.input_ids.shape[1]),
        decode_step=0,
    )


def route_lists(output: Any, model: Any, rows: int) -> list[list[list[int]]]:
    batches = CAPTURE._native_route_batches(
        output, expected_rows=rows, config=model.config
    )
    return [
        [
            [int(value) for value in batch["selected_experts"][row].tolist()]
            for row in range(rows)
        ]
        for batch in batches
    ]


def reconstruct_state(
    torch: Any,
    model: Any,
    prepared: Mapping[str, Any],
    capture: Mapping[str, Any],
    event: Mapping[str, Any],
) -> tuple[list[Any], dict[str, Any]]:
    request_ids = [str(value) for value in event["original_request_ids"]]
    states = [
        prefill_state(
            model,
            prepared[request_id],
            int(capture["ledger"][request_id]["steps"][0]["input_token_id"]),
        )
        for request_id in request_ids
    ]
    closures: list[dict[str, Any]] = []
    for step in range(int(event["decode_step"])):
        frozen = batch_for_step(capture["batches"], request_ids, step)
        forced = [
            int(capture["ledger"][request_id]["steps"][step]["input_token_id"])
            for request_id in request_ids
        ]
        for state, token in zip(states, forced):
            if state.decode_step != step:
                raise ExperimentError("teacher-forced state step drifted")
            state.next_token = torch.tensor(
                [[token]], dtype=torch.long, device=model.device
            )
        (
            input_ids,
            attention_mask,
            position_ids,
            cache_value,
            prior_lengths,
            prior_max,
        ) = CAPTURE._pad_decode_inputs(states)
        if prior_lengths != [int(value) for value in frozen["prior_cache_lengths"]]:
            raise ExperimentError("teacher-forced logical KV lengths drifted")
        if [int(value) for value in input_ids[:, 0].tolist()] != forced:
            raise ExperimentError("teacher-forced input IDs drifted")
        with torch.inference_mode():
            output, _ = CAPTURE._timed_call(
                model,
                "longrun_a_teacher_force",
                len(states),
                None,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                cache_position=torch.tensor(
                    [prior_max], dtype=torch.long, device=model.device
                ),
                past_key_values=cache_value,
                use_cache=True,
                output_router_logits=True,
                return_dict=True,
            )
        observed = route_lists(output, model, len(states))
        predicted = torch.argmax(output.logits[:, -1, :], dim=-1)
        expected_predicted = [
            int(capture["ledger"][request_id]["steps"][step]["predicted_next_token_id"])
            for request_id in request_ids
        ]
        expected_routes = [
            [
                [int(value) for value in layer["experts"]]
                for layer in capture["ledger"][request_id]["steps"][step]["route_signature"]
            ]
            for request_id in request_ids
        ]
        route_match = all(
            observed[layer][row] == expected_routes[row][layer]
            for layer in range(len(observed))
            for row in range(len(states))
        )
        token_match = [int(value) for value in predicted.tolist()] == expected_predicted
        if not route_match or not token_match:
            raise ExperimentError(
                f"canonical replay failed source closure at step {step}: "
                f"route={route_match}, token={token_match}"
            )
        split = CAPTURE.split_left_padded_cache(
            output.past_key_values,
            prior_lengths=prior_lengths,
            prior_max_length=prior_max,
        )
        for index, state in enumerate(states):
            state.cache = clone_cache(split[index])
            state.attention_mask = torch.cat(
                (state.attention_mask, state.attention_mask.new_ones((1, 1))), dim=1
            )
            state.decode_step += 1
        closures.append(
            {"decode_step": step, "route_match": route_match, "token_match": token_match}
        )
    target_step = int(event["decode_step"])
    for state in states:
        token = int(
            capture["ledger"][state.spec.request_id]["steps"][target_step]["input_token_id"]
        )
        state.next_token = torch.tensor([[token]], dtype=torch.long, device=model.device)
    return states, {
        "status": "EXACT_PRE_STEP_RECONSTRUCTED",
        "prior_steps_closed": closures,
        "target_step": target_step,
    }


def reconstruct_matched_alternative_states(
    torch: Any,
    model: Any,
    prepared_by_regime: Mapping[str, Mapping[str, Any]],
    captures: Mapping[str, Mapping[str, Any]],
    event: Mapping[str, Any],
    plan: Mapping[str, Any],
    original_states: Sequence[Any],
) -> tuple[list[Any], dict[str, Any]]:
    output: list[Any] = list(original_states)
    closures: list[dict[str, Any]] = []
    for row in plan["rows"]:
        row_index = int(row["row"])
        if row["role"] == "target":
            output[row_index] = original_states[row_index]
            continue
        if not bool(row["replaced"]):
            output[row_index] = original_states[row_index]
            closures.append(
                {
                    "row": row_index,
                    "status": "UNMATCHED_FALLBACK_ORIGINAL_COMPANION",
                    "request_id": row["request_id"],
                }
            )
            continue
        regime = str(row["regime"])
        request_id = str(row["request_id"])
        decode_step = int(row["decode_step"])
        capture = captures[regime]
        matching_batches = [
            batch
            for batch in capture["batches"]
            if request_id in [str(value) for value in batch["request_ids"]]
            and int(
                batch["decode_steps"][[str(value) for value in batch["request_ids"]].index(request_id)]
            )
            == decode_step
        ]
        if len(matching_batches) != 1:
            raise ExperimentError("matched alternative does not bind one source batch")
        source_batch = matching_batches[0]
        request_ids = [str(value) for value in source_batch["request_ids"]]
        if [int(value) for value in source_batch["decode_steps"]] != [
            decode_step
        ] * len(request_ids):
            raise ExperimentError(
                "matched alternative source is not in the frozen synchronous replay subset"
            )
        synthetic_event = {
            "original_request_ids": request_ids,
            "decode_step": decode_step,
        }
        candidate_states, closure = reconstruct_state(
            torch,
            model,
            prepared_by_regime[regime],
            capture,
            synthetic_event,
        )
        candidate = candidate_states[request_ids.index(request_id)]
        observed_length = int(CAPTURE._cache_length(candidate.cache))
        if observed_length != int(row["logical_kv_length"]):
            raise ExperimentError("matched alternative logical KV length drifted")
        if str(candidate.spec.document_id) != str(row["document_id"]):
            raise ExperimentError("matched alternative document identity drifted")
        if str(candidate.spec.document_id) in {
            str(value) for value in event["original_document_ids"]
        }:
            raise ExperimentError("matched alternative reused an original document")
        output[row_index] = candidate
        closures.append(
            {
                "row": row_index,
                "status": "MATCHED_ALTERNATIVE_RECONSTRUCTED",
                "regime": regime,
                "request_id": request_id,
                "decode_step": decode_step,
                "logical_kv_length": observed_length,
                "source_closure": closure,
            }
        )
    if int(CAPTURE._cache_length(output[int(event["target_row_index"])].cache)) != int(
        event["logical_kv_lengths"][int(event["target_row_index"])]
    ):
        raise ExperimentError("matched D target state changed")
    return output, {
        "status": "MATCHED_ALTERNATIVE_STATES_READY",
        "rows": closures,
        "replaced_companion_rows": int(plan["replaced_companion_rows"]),
        "companion_rows": int(plan["companion_rows"]),
    }


def build_arm(
    name: str, source_states: Sequence[Any], target_row: int, request_ids: Sequence[str]
) -> ArmInputs:
    clones = [clone_state(state) for state in source_states]
    source_ptrs = set().union(*(cache_storage_ptrs(state.cache) for state in source_states))
    state_ptrs: set[int] = set()
    for state in clones:
        pointers = cache_storage_ptrs(state.cache)
        if source_ptrs.intersection(pointers):
            raise ExperimentError(f"{name} cache clone aliases canonical source")
        if state_ptrs.intersection(pointers):
            raise ExperimentError(f"{name} per-row cache clones alias")
        state_ptrs.update(pointers)
    (
        input_ids,
        attention_mask,
        position_ids,
        cache_value,
        prior_lengths,
        prior_max,
    ) = CAPTURE._pad_decode_inputs(clones)
    cache_ptrs = cache_storage_ptrs(cache_value)
    if cache_ptrs.intersection(state_ptrs):
        raise ExperimentError(f"{name} stacked cache aliases source rows")
    logical = int(prior_lengths[target_row])
    return ArmInputs(
        name=name,
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        cache=cache_value,
        cache_position=input_ids.new_tensor([prior_max]),
        prior_lengths=[int(value) for value in prior_lengths],
        prior_max=int(prior_max),
        target_row=int(target_row),
        request_ids=[str(value) for value in request_ids],
        state_storage_ptrs=state_ptrs,
        cache_storage_ptrs=cache_ptrs,
        target_logical_cache_sha256=logical_cache_sha256(
            cache_value, int(target_row), logical
        ),
    )


def build_arms(
    states: Sequence[Any],
    alternative_states: Sequence[Any],
    event: Mapping[str, Any],
) -> dict[str, ArmInputs]:
    target_row = int(event["target_row_index"])
    target = states[target_row]
    width = len(states)
    original_ids = [state.spec.request_id for state in states]
    a = build_arm("A", [target], 0, [target.spec.request_id])
    b_states = [target for _ in range(width)]
    b_ids = [f"{target.spec.request_id}:clone:{index}" for index in range(width)]
    b = build_arm("B", b_states, target_row, b_ids)
    c = build_arm("C", states, target_row, original_ids)
    d_ids = [state.spec.request_id for state in alternative_states]
    d = build_arm("D", alternative_states, target_row, d_ids)
    arms = {arm.name: arm for arm in (a, b, c, d)}
    seen_cache: set[int] = set()
    for arm in arms.values():
        if seen_cache.intersection(arm.cache_storage_ptrs):
            raise ExperimentError("cache storage aliases across execution arms")
        seen_cache.update(arm.cache_storage_ptrs)
    hashes = {arm.target_logical_cache_sha256 for arm in arms.values()}
    tokens = {int(arm.input_ids[arm.target_row, 0].item()) for arm in arms.values()}
    positions = {int(arm.position_ids[arm.target_row, 0].item()) for arm in arms.values()}
    if len(hashes) != 1 or len(tokens) != 1 or len(positions) != 1:
        raise ExperimentError("target causal input equality failed across arms")
    if b.prior_max != a.prior_max or c.prior_max != d.prior_max:
        raise ExperimentError("arm physical-shape controls drifted")
    if c.prior_lengths != d.prior_lengths:
        raise ExperimentError("C/D per-row logical KV-length vector drifted")
    return arms


def vector_metrics(reference: Any, value: Any) -> dict[str, Any]:
    import torch

    left = reference.float().reshape(-1)
    right = value.float().reshape(-1)
    delta = right - left
    left_norm = float(torch.linalg.vector_norm(left).item())
    right_norm = float(torch.linalg.vector_norm(right).item())
    denominator = max(left_norm * right_norm, 1e-30)
    return {
        "exact": bool(torch.equal(reference, value)),
        "allclose": bool(
            torch.allclose(reference, value, atol=ALLCLOSE_ATOL, rtol=ALLCLOSE_RTOL)
        ),
        "differing_elements": int(torch.count_nonzero(reference != value).item()),
        "max_abs_delta": float(delta.abs().max().item()),
        "relative_l2": float(torch.linalg.vector_norm(delta).item()) / max(left_norm, 1e-30),
        "cosine_similarity": float(torch.dot(left, right).item()) / denominator,
    }


def public_state_boundary(state: Any) -> dict[str, Any]:
    mask = state.attention_mask.detach()
    cache_length = int(CAPTURE._cache_length(state.cache))
    return {
        "cache_length": cache_length,
        "cache_sha256": logical_cache_sha256(state.cache, 0, cache_length),
        "attention_mask_shape": [int(value) for value in mask.shape],
        "attention_mask_sum": int(mask.sum().item()),
        "attention_mask_unique_values": [
            int(value) for value in sorted(set(mask.reshape(-1).tolist()))
        ],
        "attention_mask_sha256": tensor_sha256(mask),
    }


def public_arm_record(record: Mapping[str, Any]) -> dict[str, Any]:
    layers = []
    for layer in range(len(record["routes"])):
        logits = record["vectors"]["router_logits"][layer]
        top_k = len(record["routes"][layer])
        boundary_values = logits.float().topk(k=top_k + 1).values
        layers.append(
            {
                "layer": layer,
                "selected_experts": list(record["routes"][layer]),
                "selection_boundary_margin": float(
                    boundary_values[top_k - 1].item() - boundary_values[top_k].item()
                ),
                "residual_input_sha256": tensor_sha256(
                    record["vectors"]["residual_input"][layer]
                ),
                "attention_output_sha256": tensor_sha256(
                    record["vectors"]["attention_output"][layer]
                ),
                "pre_router_hidden_sha256": tensor_sha256(
                    record["vectors"]["pre_router_hidden"][layer]
                ),
                "router_logits_sha256": tensor_sha256(logits),
                "moe_output_sha256": tensor_sha256(
                    record["vectors"]["moe_output"][layer]
                ),
            }
        )
    return {
        "elapsed_model_call_ms": float(record["elapsed_us"]) / 1000.0,
        "gpu_memory": dict(record["peak_memory"]),
        "predicted_next_token_id": int(record["predicted_token"]),
        "final_logits_sha256": tensor_sha256(record["vectors"]["final_logits"]),
        "final_top5": [
            int(value)
            for value in record["vectors"]["final_logits"].float().topk(k=5).indices.tolist()
        ],
        "output_state_boundary": public_state_boundary(record["output_state"]),
        "layers": layers,
    }


def clone_row_consistency(record: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    routes = record["all_routes"]
    rows = int(record["all_final_logits"].shape[0])
    route_equal = all(
        routes[layer][row] == routes[layer][0]
        for layer in range(len(routes))
        for row in range(1, rows)
    )
    router_exact = all(
        torch.equal(record["all_router_logits"][layer][row], record["all_router_logits"][layer][0])
        for layer in range(len(routes))
        for row in range(1, rows)
    )
    final_exact = all(
        torch.equal(record["all_final_logits"][row], record["all_final_logits"][0])
        for row in range(1, rows)
    )
    output_cache_exact = len(set(record["all_output_cache_sha256"])) == 1
    output_cache_length_equal = len(set(record["all_output_cache_lengths"])) == 1
    return {
        "rows": rows,
        "route_exact": route_equal,
        "router_logits_exact": router_exact,
        "final_logits_exact": final_exact,
        "post_step_cache_exact": output_cache_exact,
        "post_step_cache_length_equal": output_cache_length_equal,
        "post_step_cache_sha256_by_row": list(record["all_output_cache_sha256"]),
        "passed": (
            route_equal
            and router_exact
            and final_exact
            and output_cache_exact
            and output_cache_length_equal
        ),
    }


def run_arm(torch: Any, model: Any, arm: ArmInputs) -> dict[str, Any]:
    captured = {
        "residual_input": {},
        "attention_output": {},
        "pre_router_hidden": {},
        "moe_output": {},
    }
    handles = []
    for layer_index, layer in enumerate(model.model.layers):
        def layer_pre(_module: Any, inputs: tuple[Any, ...], bound: int = layer_index) -> None:
            captured["residual_input"][bound] = (
                inputs[0][arm.target_row, -1, :].detach().clone()
            )

        def mlp_pre(_module: Any, inputs: tuple[Any, ...], bound: int = layer_index) -> None:
            captured["pre_router_hidden"][bound] = (
                inputs[0][arm.target_row, -1, :].detach().clone()
            )

        def attention_post(
            _module: Any, _inputs: tuple[Any, ...], output: Any, bound: int = layer_index
        ) -> None:
            captured["attention_output"][bound] = (
                output[0][arm.target_row, -1, :].detach().clone()
            )

        def mlp_post(
            _module: Any, _inputs: tuple[Any, ...], output: Any, bound: int = layer_index
        ) -> None:
            captured["moe_output"][bound] = (
                output[0][arm.target_row, -1, :].detach().clone()
            )

        handles.append(layer.register_forward_pre_hook(layer_pre))
        handles.append(layer.self_attn.register_forward_hook(attention_post))
        handles.append(layer.mlp.register_forward_pre_hook(mlp_pre))
        handles.append(layer.mlp.register_forward_hook(mlp_post))
    try:
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            output, elapsed_us = CAPTURE._timed_call(
                model,
                f"longrun_a_arm_{arm.name}",
                int(arm.input_ids.shape[0]),
                None,
                input_ids=arm.input_ids,
                attention_mask=arm.attention_mask,
                position_ids=arm.position_ids,
                cache_position=arm.cache_position,
                past_key_values=arm.cache,
                use_cache=True,
                output_router_logits=True,
                return_dict=True,
            )
    finally:
        for handle in handles:
            handle.remove()
    peak_memory = {
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "allocated_after_call_bytes": int(torch.cuda.memory_allocated()),
        "reserved_after_call_bytes": int(torch.cuda.memory_reserved()),
    }
    layer_count = int(model.config.num_hidden_layers)
    if any(set(captured[name]) != set(range(layer_count)) for name in captured):
        raise ExperimentError("target-row hook ownership did not close")
    captured = {
        name: {layer: tensor.cpu() for layer, tensor in rows.items()}
        for name, rows in captured.items()
    }
    all_routes = route_lists(output, model, int(arm.input_ids.shape[0]))
    all_router_logits = {
        layer: output.router_logits[layer].detach().cpu().clone()
        for layer in range(layer_count)
    }
    all_final_logits = output.logits[:, -1, :].detach().cpu().clone()
    all_predicted_tokens = [
        int(value) for value in torch.argmax(output.logits[:, -1, :], dim=-1).tolist()
    ]
    target_routes = [layer[arm.target_row] for layer in all_routes]
    router_logits = {
        layer: all_router_logits[layer][arm.target_row].clone()
        for layer in range(layer_count)
    }
    final_logits = all_final_logits[arm.target_row].clone()
    split = CAPTURE.split_left_padded_cache(
        output.past_key_values,
        prior_lengths=arm.prior_lengths,
        prior_max_length=arm.prior_max,
    )
    output_lengths = [int(value) + 1 for value in arm.prior_lengths]
    all_output_cache_sha256 = [
        logical_cache_sha256(cache, 0, logical_length)
        for cache, logical_length in zip(split, output_lengths)
    ]
    output_state = SimpleNamespace(
        spec=SimpleNamespace(request_id=arm.request_ids[arm.target_row]),
        cache=clone_cache(split[arm.target_row]),
        attention_mask=arm.attention_mask[
            arm.target_row : arm.target_row + 1,
            -(arm.prior_lengths[arm.target_row] + 1) :,
        ].detach().clone(),
        next_token=None,
        prompt_length=0,
        decode_step=0,
    )
    record = {
        "elapsed_us": float(elapsed_us),
        "peak_memory": peak_memory,
        "predicted_token": int(torch.argmax(output.logits[arm.target_row, -1, :]).item()),
        "routes": target_routes,
        "all_routes": all_routes,
        "all_router_logits": all_router_logits,
        "all_final_logits": all_final_logits,
        "all_predicted_tokens": all_predicted_tokens,
        "all_output_cache_sha256": all_output_cache_sha256,
        "all_output_cache_lengths": output_lengths,
        "vectors": {
            **captured,
            "router_logits": router_logits,
            "final_logits": final_logits,
        },
        "output_state": output_state,
    }
    record["public"] = public_arm_record(record)
    return record


def compare_records(reference: Mapping[str, Any], value: Mapping[str, Any]) -> dict[str, Any]:
    layer_rows = []
    first_any = None
    first_exact_stage = None
    first_material_stage = None
    first_pre_router = None
    first_router = None
    first_route = None
    near_boundary_flips = 0
    disputed_gap_crossing_flips = 0
    route_flips = 0
    for layer, (left_route, right_route) in enumerate(
        zip(reference["routes"], value["routes"])
    ):
        metrics = {
            name: vector_metrics(
                reference["vectors"][name][layer], value["vectors"][name][layer]
            )
            for name in (
                "residual_input",
                "attention_output",
                "pre_router_hidden",
                "router_logits",
                "moe_output",
            )
        }
        route_changed = sorted(left_route) != sorted(right_route)
        left_logits = reference["vectors"]["router_logits"][layer].float()
        right_logits = value["vectors"]["router_logits"][layer].float()
        top_k = len(left_route)
        left_values = left_logits.topk(k=top_k + 1).values
        right_values = right_logits.topk(k=top_k + 1).values
        margins = [
            float(left_values[top_k - 1] - left_values[top_k]),
            float(right_values[top_k - 1] - right_values[top_k]),
        ]
        near_boundary = min(abs(value) for value in margins) <= NEAR_TIE_MARGIN
        lost = sorted(set(int(item) for item in left_route) - set(int(item) for item in right_route))
        gained = sorted(set(int(item) for item in right_route) - set(int(item) for item in left_route))
        disputed_pairs = []
        for lost_expert in lost:
            for gained_expert in gained:
                left_gap = float(left_logits[lost_expert] - left_logits[gained_expert])
                right_gap = float(right_logits[lost_expert] - right_logits[gained_expert])
                disputed_pairs.append(
                    {
                        "lost_expert": lost_expert,
                        "gained_expert": gained_expert,
                        "reference_lost_minus_gained_gap": left_gap,
                        "comparison_lost_minus_gained_gap": right_gap,
                        "sign_crossing": bool(
                            (left_gap > 0.0 and right_gap <= 0.0)
                            or (left_gap >= 0.0 and right_gap < 0.0)
                        ),
                        "near_zero_association": min(abs(left_gap), abs(right_gap))
                        <= NEAR_TIE_MARGIN,
                    }
                )
        disputed_gap_crossing = bool(
            route_changed and any(item["sign_crossing"] for item in disputed_pairs)
        )
        route_flips += int(route_changed)
        near_boundary_flips += int(route_changed and near_boundary)
        disputed_gap_crossing_flips += int(disputed_gap_crossing)
        if first_any is None and any(not item["exact"] for item in metrics.values()):
            first_any = layer
        for stage in (
            "residual_input",
            "attention_output",
            "pre_router_hidden",
            "router_logits",
            "route_membership",
            "moe_output",
        ):
            exact_changed = (
                route_changed if stage == "route_membership" else not metrics[stage]["exact"]
            )
            material_changed = (
                route_changed
                if stage == "route_membership"
                else not metrics[stage]["allclose"]
            )
            if first_exact_stage is None and exact_changed:
                first_exact_stage = {"layer": layer, "stage": stage}
            if first_material_stage is None and material_changed:
                first_material_stage = {"layer": layer, "stage": stage}
        if first_pre_router is None and not metrics["pre_router_hidden"]["allclose"]:
            first_pre_router = layer
        if first_router is None and not metrics["router_logits"]["allclose"]:
            first_router = layer
        if first_route is None and route_changed:
            first_route = layer
        layer_rows.append(
            {
                "layer": layer,
                **metrics,
                "route_multiset_changed": route_changed,
                "reference_boundary_margin": margins[0],
                "comparison_boundary_margin": margins[1],
                "near_boundary_association_by_preregistered_0_01_margin": bool(
                    route_changed and near_boundary
                ),
                "disputed_expert_pairs": disputed_pairs,
                "disputed_pair_gap_sign_crossing": disputed_gap_crossing,
            }
        )
    final_metrics = vector_metrics(
        reference["vectors"]["final_logits"], value["vectors"]["final_logits"]
    )
    reference_state = public_state_boundary(reference["output_state"])
    comparison_state = public_state_boundary(value["output_state"])
    output_cache_exact = (
        reference_state["cache_sha256"] == comparison_state["cache_sha256"]
    )
    output_mask_exact = (
        reference_state["attention_mask_sha256"]
        == comparison_state["attention_mask_sha256"]
    )
    if first_exact_stage is None and not final_metrics["exact"]:
        first_exact_stage = {"layer": None, "stage": "final_norm_or_lm_head"}
    if first_exact_stage is None and not output_cache_exact:
        first_exact_stage = {"layer": None, "stage": "kv_cache_write"}
    if first_exact_stage is None and not output_mask_exact:
        first_exact_stage = {"layer": None, "stage": "attention_mask_state"}
    return {
        "first_any_tensor_difference_layer": first_any,
        "first_exact_causal_stage": first_exact_stage,
        "first_non_allclose_or_route_causal_stage": first_material_stage,
        "first_pre_router_allclose_difference_layer": first_pre_router,
        "first_router_allclose_difference_layer": first_router,
        "first_route_membership_difference_layer": first_route,
        "route_flip_layers": route_flips,
        "near_boundary_associated_route_flip_layers": near_boundary_flips,
        "near_boundary_associated_route_flip_fraction": near_boundary_flips
        / max(1, route_flips),
        "disputed_pair_gap_sign_crossing_route_flip_layers": disputed_gap_crossing_flips,
        "disputed_pair_gap_sign_crossing_route_flip_fraction": disputed_gap_crossing_flips
        / max(1, route_flips),
        "final_logits": final_metrics,
        "post_step_state": {
            "output_cache_exact": output_cache_exact,
            "attention_mask_exact": output_mask_exact,
            "reference": reference_state,
            "comparison": comparison_state,
        },
        "predicted_token_changed": (
            int(reference["predicted_token"]) != int(value["predicted_token"])
        ),
        "layers": layer_rows,
    }


def source_target_route(capture: Mapping[str, Any], event: Mapping[str, Any]) -> list[list[int]]:
    row = capture["ledger"][event["request_id"]]["steps"][event["decode_step"]]
    return [[int(value) for value in layer["experts"]] for layer in row["route_signature"]]


def save_target_tensors(
    torch: Any,
    path: Path,
    records: Mapping[str, Mapping[str, Any]],
    first_layer: int | None,
) -> None:
    if path.exists():
        raise ExperimentError(f"refusing to overwrite tensor bundle: {path}")
    layer_count = len(next(iter(records.values()))["routes"])
    center = int(first_layer or 0)
    layers = sorted({max(0, center - 1), center, min(layer_count - 1, center + 1)})
    payload = {"layers": layers, "arms": {}}
    for arm, record in records.items():
        payload["arms"][arm] = {
            name: {layer: record["vectors"][name][layer] for layer in layers}
            for name in (
                "residual_input",
                "attention_output",
                "pre_router_hidden",
                "router_logits",
                "moe_output",
            )
        }
        payload["arms"][arm]["final_logits"] = record["vectors"]["final_logits"]
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def validate_propagation_states(
    torch: Any,
    states: Mapping[str, Any],
    expected_length: int,
) -> dict[str, Any]:
    boundaries = {name: public_state_boundary(state) for name, state in states.items()}
    for name, boundary in boundaries.items():
        if int(boundary["cache_length"]) != expected_length:
            raise ExperimentError(
                f"{name} propagation cache length {boundary['cache_length']} != {expected_length}"
            )
        if boundary["attention_mask_shape"] != [1, expected_length]:
            raise ExperimentError(f"{name} propagation attention-mask shape drifted")
        if int(boundary["attention_mask_sum"]) != expected_length:
            raise ExperimentError(f"{name} propagation attention-mask sum drifted")
        if boundary["attention_mask_unique_values"] != [1]:
            raise ExperimentError(f"{name} propagation attention mask is not all one")
    mask_hashes = {row["attention_mask_sha256"] for row in boundaries.values()}
    if len(mask_hashes) != 1:
        raise ExperimentError("propagation logical attention masks differ across arms")
    seen: set[int] = set()
    for name, state in states.items():
        pointers = cache_storage_ptrs(state.cache)
        if seen.intersection(pointers):
            raise ExperimentError(f"{name} propagation cache aliases another arm")
        seen.update(pointers)
    return {
        "passed": True,
        "expected_logical_length": expected_length,
        "cache_storage_non_alias_across_arms": True,
        "logical_attention_masks_exact_across_arms": True,
        "arms": boundaries,
    }


def propagation_step(
    torch: Any,
    model: Any,
    states: Mapping[str, Any],
    forced_token: int,
    order: Sequence[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    inputs: dict[str, ArmInputs] = {}
    for arm_name, state in states.items():
        state.next_token = torch.tensor(
            [[forced_token]], dtype=torch.long, device=model.device
        )
        inputs[arm_name] = build_arm(
            arm_name, [state], 0, [state.spec.request_id]
        )
    if sorted(order) != ["A", "B", "C", "D"]:
        raise ExperimentError(f"invalid propagation arm order: {order}")
    records = {name: run_arm(torch, model, inputs[name]) for name in order}
    records = {name: records[name] for name in ("A", "B", "C", "D")}
    comparisons = {
        name: compare_records(records["A"], records[name])
        for name in ("B", "C", "D")
    }
    return records, {
        "forced_token_id": forced_token,
        "arm_order": list(order),
        "arms": {name: record["public"] for name, record in records.items()},
        "A_vs": comparisons,
    }


def arm_stability_fingerprint(arm: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {
            "predicted_next_token_id": arm["predicted_next_token_id"],
            "final_logits_sha256": arm["final_logits_sha256"],
            "final_top5": arm["final_top5"],
            "output_state_boundary": arm["output_state_boundary"],
            "layers": arm["layers"],
        }
    )


def comparison_has_exact_effect(comparison: Mapping[str, Any]) -> bool:
    return bool(
        comparison["first_exact_causal_stage"] is not None
        or comparison["predicted_token_changed"]
    )


def classify_event(repeats: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    arm_hashes: dict[str, list[str]] = {name: [] for name in ("A", "B", "C", "D")}
    for repeat in repeats:
        for arm in arm_hashes:
            arm_hashes[arm].append(arm_stability_fingerprint(repeat["arms"][arm]))
    stable = all(len(set(values)) == 1 for values in arm_hashes.values())
    alternatives_complete = all(
        int(repeat["matched_alternative_reconstruction"]["replaced_companion_rows"])
        == int(repeat["matched_alternative_reconstruction"]["companion_rows"])
        for repeat in repeats
    )
    if not stable:
        primary = "RUNTIME_NONDETERMINISM"
        stability_status = "INCONCLUSIVE_UNSTABLE"
        source_subtype = None
        near_boundary_fractions = None
        disputed_crossing_fractions = None
        ab_changed = None
        cd_changed = None
        first_sources = None
    else:
        stability_status = "STABLE_WITHIN_ARM"
        source_subtype = None
        comparison = repeats[0]["A_vs"]
        a_c_flip = comparison["C"]["route_flip_layers"] > 0
        ab_changed = comparison_has_exact_effect(comparison["B"])
        cd_changed = comparison_has_exact_effect(repeats[0]["C_vs_D"])
        first_sources = {
            "A_vs_B": comparison["B"]["first_exact_causal_stage"],
            "A_vs_C": comparison["C"]["first_exact_causal_stage"],
            "A_vs_D": comparison["D"]["first_exact_causal_stage"],
            "C_vs_D": repeats[0]["C_vs_D"]["first_exact_causal_stage"],
        }
        if not a_c_flip:
            primary = "NOT_REPRODUCED"
        elif ab_changed and cd_changed:
            primary = "MIXED_SOURCE_IDENTIFIED"
            source_subtype = "BATCH_SHAPE_AND_COMPANION_IDENTITY"
        elif cd_changed:
            primary = "COMPANION_IDENTITY_EXTERNALITY"
        elif ab_changed:
            first_ab = comparison["B"]["first_exact_causal_stage"]
            if first_ab is None:
                primary = "MIXED_SOURCE_IDENTIFIED"
                source_subtype = "FINAL_OUTPUT_EFFECT_UNLOCALIZED"
                first_ab_stage = None
            else:
                first_ab_stage = first_ab["stage"]
            if first_ab_stage is not None:
                primary = {
                    "residual_input": "UPSTREAM_BATCH_CONTEXT_EFFECT",
                    "attention_output": "UPSTREAM_BATCH_CONTEXT_EFFECT",
                    "pre_router_hidden": "UPSTREAM_BATCH_CONTEXT_EFFECT",
                    "router_logits": "ROUTER_KERNEL_SHAPE_EFFECT",
                    "route_membership": "PHYSICAL_SHAPE_EFFECT",
                    "moe_output": "PHYSICAL_SHAPE_EFFECT",
                    "final_norm_or_lm_head": "PHYSICAL_SHAPE_EFFECT",
                    "kv_cache_write": "PHYSICAL_SHAPE_EFFECT",
                    "attention_mask_state": "PHYSICAL_SHAPE_EFFECT",
                }[first_ab_stage]
                source_subtype = f"WIDTH_ONLY_FIRST_AT_{first_ab_stage.upper()}"
        elif alternatives_complete and not cd_changed:
            primary = "PHYSICAL_SHAPE_EFFECT"
            source_subtype = "HETEROGENEOUS_LENGTH_OR_PADDING"
        else:
            primary = "MIXED_SOURCE_IDENTIFIED"
        near_boundary_fractions = [
            float(
                repeat["A_vs"]["C"][
                    "near_boundary_associated_route_flip_fraction"
                ]
            )
            for repeat in repeats
        ]
        disputed_crossing_fractions = [
            float(
                repeat["A_vs"]["C"][
                    "disputed_pair_gap_sign_crossing_route_flip_fraction"
                ]
            )
            for repeat in repeats
        ]
    return {
        "primary": primary,
        "stability_status": stability_status,
        "source_subtype": source_subtype,
        "within_arm_repeat_stable": stable,
        "arm_public_record_hashes": arm_hashes,
        "near_boundary_association_secondary": (
            None
            if near_boundary_fractions is None
            else bool(near_boundary_fractions and min(near_boundary_fractions) >= 0.5)
        ),
        "near_tie_amplification_supported_secondary": (
            None
            if near_boundary_fractions is None
            else bool(
                near_boundary_fractions
                and disputed_crossing_fractions
                and min(near_boundary_fractions) >= 0.5
                and min(disputed_crossing_fractions) >= 0.5
            )
        ),
        "near_boundary_route_flip_fraction_by_repeat": near_boundary_fractions,
        "disputed_pair_gap_sign_crossing_fraction_by_repeat": disputed_crossing_fractions,
        "arm_b_exact_effect_present": ab_changed,
        "arm_c_vs_d_exact_effect_present": cd_changed,
        "first_exact_source_by_comparison": first_sources,
        "matched_alternative_companions_complete": alternatives_complete,
    }


def summarize_propagation(
    propagation: Mapping[str, Sequence[Mapping[str, Any]]], repeats_required: int
) -> dict[str, Any]:
    events: dict[str, Any] = {}
    all_stable = True
    for event_id, repeat_rows in propagation.items():
        if len(repeat_rows) != repeats_required:
            raise ExperimentError(f"propagation repeat count did not close for {event_id}")
        step_ids = [
            int(step["decode_step"])
            for step in repeat_rows[0]["teacher_forced_serial_steps"]
        ]
        per_step: dict[str, Any] = {}
        event_stable = True
        for step_id in step_ids:
            hashes: dict[str, list[str]] = {name: [] for name in ("A", "B", "C", "D")}
            for repeat_row in repeat_rows:
                matches = [
                    step
                    for step in repeat_row["teacher_forced_serial_steps"]
                    if int(step["decode_step"]) == step_id
                ]
                if len(matches) != 1:
                    raise ExperimentError(
                        f"propagation step closure failed for {event_id} step {step_id}"
                    )
                for arm, public in matches[0]["arms"].items():
                    hashes[arm].append(arm_stability_fingerprint(public))
            stable_by_arm = {
                arm: len(set(values)) == 1 for arm, values in hashes.items()
            }
            step_stable = all(stable_by_arm.values())
            event_stable = event_stable and step_stable
            per_step[str(step_id)] = {
                "within_arm_repeat_stable": step_stable,
                "stable_by_arm": stable_by_arm,
                "arm_public_record_hashes": hashes,
            }
        all_stable = all_stable and event_stable
        events[event_id] = {
            "classification": (
                "STABLE_PROPAGATION_MEASUREMENT_ONLY"
                if event_stable
                else "INCONCLUSIVE_UNSTABLE_PROPAGATION"
            ),
            "within_arm_repeat_stable": event_stable,
            "steps": per_step,
        }
    return {
        "all_events_within_arm_repeat_stable": all_stable,
        "latency_claim_allowed": False,
        "latency_claim_reason": "single calls are diagnostic and not a benchmark",
        "events": events,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steady-capture", type=Path, required=True)
    parser.add_argument("--bursty-capture", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--select-only", action="store_true")
    parser.add_argument("--offline", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    captures = {
        "steady": load_capture(args.steady_capture.resolve(), "steady"),
        "bursty": load_capture(args.bursty_capture.resolve(), "bursty"),
    }
    manifest_contracts = {}
    for regime, capture in captures.items():
        model_spec = capture["manifest"]["model"]
        generation = capture["manifest"]["generation"]
        manifest_contracts[regime] = {
            "model": {
                key: model_spec.get(key)
                for key in ("id", "revision", "tokenizer_revision", "dtype")
            },
            "generation": {
                key: generation.get(key)
                for key in ("mode", "do_sample", "max_decode_steps")
            },
        }
    if canonical_sha256(manifest_contracts["steady"]) != canonical_sha256(
        manifest_contracts["bursty"]
    ):
        raise ExperimentError("steady/bursty frozen model or generation contract differs")
    for key, value in BASE.EXPECTED_MODEL.items():
        if manifest_contracts["steady"]["model"].get(key) != value:
            raise ExperimentError(f"capture model contract differs from expected {key}")
    selection = build_selection(captures)
    if args.select_only:
        write_json_exclusive(args.selection.resolve(), selection)
        print(json.dumps({"status": "EVENT_SELECTION_COMPLETE", "events": 6}))
        return
    if not args.offline:
        raise ExperimentError("pass --offline; this harness never downloads models")
    if args.repeats < 3:
        raise ExperimentError("at least three repeats are required")
    if args.output_dir is None:
        raise ExperimentError("--output-dir is required for GPU execution")
    selected_file = read_json(args.selection.resolve())
    if canonical_sha256(selected_file["events"]) != canonical_sha256(selection["events"]):
        raise ExperimentError("EVENT_SELECTION.json drifted from hash-bound captures")
    alternative_plan = plan_matched_alternatives(captures, selection)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "config.json").exists():
        raise ExperimentError("output directory already contains a run")
    write_json_exclusive(
        output_dir / "RUN_STARTED.json",
        {
            "status": "RUN_STARTED",
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )
    write_text_exclusive(
        output_dir / "commands.sh",
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        + shlex.join([sys.executable, *sys.argv])
        + "\n",
        mode=0o755,
    )

    if BASE.query_gpu_compute_processes():
        raise ExperimentError("GPU is not idle before model load")
    torch, transformers, tokenizer, model, load_seconds = BASE.load_exact_model(
        captures["steady"]["manifest"]
    )
    own_process = BASE.query_gpu_compute_processes()
    if len(own_process) != 1:
        raise ExperimentError("model load did not create one isolated GPU process")
    monitor = BASE.GpuIsolationMonitor(own_process)
    prepared_by_regime = {}
    for regime, capture in captures.items():
        prepared = CAPTURE._prepare_requests(capture["manifest"], tokenizer, model.device)
        prepared_by_regime[regime] = {state.request_id: state for state in prepared}
        BASE.validate_tokenizer_and_prompt_identity(
            capture["manifest"],
            tokenizer,
            prepared,
            [str(row["request_id"]) for row in capture["manifest"]["requests"]],
        )
    config = {
        "schema": "longrun-a-exact-event-config-v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_head": git_value("rev-parse", "HEAD"),
        "git_branch": git_value("branch", "--show-current"),
        "git_status_short": git_value("status", "--short"),
        "source_files": {
            str(Path(__file__).resolve().relative_to(ROOT)): sha256_file(
                Path(__file__).resolve()
            ),
            str(Path(CAPTURE.__file__).resolve().relative_to(ROOT)): sha256_file(
                Path(CAPTURE.__file__).resolve()
            ),
            str(Path(BASE.__file__).resolve().relative_to(ROOT)): sha256_file(
                Path(BASE.__file__).resolve()
            ),
        },
        "model": BASE.EXPECTED_MODEL,
        "runtime": BASE._environment(torch, transformers),
        "model_load_seconds": load_seconds,
        "repeats": args.repeats,
        "arms": {
            "A": "target serial, width one",
            "B": "target clones at original width, equal logical/physical length",
            "C": "original companion batch and padding distribution",
            "D": "different-document exact-length matched companions with target row fixed",
        },
        "allclose": {"atol": ALLCLOSE_ATOL, "rtol": ALLCLOSE_RTOL},
        "near_tie_margin_inherited_pre_run": NEAR_TIE_MARGIN,
        "capture_hashes": {
            regime: capture["hashes"] for regime, capture in captures.items()
        },
        "cross_capture_model_generation_contract": manifest_contracts,
        "matched_alternatives_sha256": canonical_sha256(alternative_plan),
        "event_selection_file_sha256": sha256_file(args.selection.resolve()),
        "claim_ceiling": (
            "single-OLMoE single-RTX5090 custom-runtime execution-conformance "
            "diagnostic; no capacity action or native-serving claim"
        ),
    }
    write_json_exclusive(output_dir / "config.json", config)
    write_json_exclusive(output_dir / "selected_events.json", selection)
    write_json_exclusive(output_dir / "matched_alternatives.json", alternative_plan)

    all_event_repeats: dict[str, list[dict[str, Any]]] = {
        event["event_id"]: [] for event in selection["events"]
    }
    propagation: dict[str, list[dict[str, Any]]] = {
        event_id: [] for event_id in PROPAGATION_EVENTS
    }
    monitor.start()
    try:
        for event in selection["events"]:
            event_id = str(event["event_id"])
            regime = str(event["regime"])
            capture = captures[regime]
            for repeat in range(args.repeats):
                monitor.check(f"before_{event_id}_repeat_{repeat}")
                monitor.require_clean()
                seed = int(capture["manifest"]["seed"])
                random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                states, reconstruction = reconstruct_state(
                    torch,
                    model,
                    prepared_by_regime[regime],
                    capture,
                    event,
                )
                alternative_states, alternative_closure = reconstruct_matched_alternative_states(
                    torch,
                    model,
                    prepared_by_regime,
                    captures,
                    event,
                    alternative_plan["plans"][event_id],
                    states,
                )
                arms = build_arms(states, alternative_states, event)
                order = (
                    ["A", "B", "C", "D"]
                    if repeat % 2 == 0
                    else ["D", "C", "B", "A"]
                )
                records = {name: run_arm(torch, model, arms[name]) for name in order}
                records = {name: records[name] for name in ("A", "B", "C", "D")}
                clone_consistency = clone_row_consistency(records["B"])
                if not clone_consistency["passed"]:
                    raise ExperimentError(
                        f"Arm B identical target clones are inconsistent: {clone_consistency}"
                    )
                source_route = source_target_route(capture, event)
                source_token = int(
                    capture["ledger"][event["request_id"]]["steps"][event["decode_step"]][
                        "predicted_next_token_id"
                    ]
                )
                c_source_route_match = records["C"]["routes"] == source_route
                c_source_token_match = records["C"]["predicted_token"] == source_token
                original_ids = [str(value) for value in event["original_request_ids"]]
                expected_all_routes = [
                    [
                        [int(value) for value in layer["experts"]]
                        for layer in capture["ledger"][request_id]["steps"][
                            event["decode_step"]
                        ]["route_signature"]
                    ]
                    for request_id in original_ids
                ]
                c_all_rows_route_match = all(
                    records["C"]["all_routes"][layer][row]
                    == expected_all_routes[row][layer]
                    for layer in range(len(records["C"]["all_routes"]))
                    for row in range(len(original_ids))
                )
                expected_all_tokens = [
                    int(
                        capture["ledger"][request_id]["steps"][event["decode_step"]][
                            "predicted_next_token_id"
                        ]
                    )
                    for request_id in original_ids
                ]
                c_all_rows_token_match = (
                    records["C"]["all_predicted_tokens"] == expected_all_tokens
                )
                expected_inputs = [
                    int(
                        capture["ledger"][request_id]["steps"][event["decode_step"]][
                            "input_token_id"
                        ]
                    )
                    for request_id in original_ids
                ]
                c_all_rows_input_match = [
                    int(value) for value in arms["C"].input_ids[:, 0].tolist()
                ] == expected_inputs
                if not (
                    c_source_route_match
                    and c_source_token_match
                    and c_all_rows_route_match
                    and c_all_rows_token_match
                    and c_all_rows_input_match
                ):
                    raise ExperimentError(
                        f"Arm C did not reproduce source event {event_id}: "
                        f"target_route={c_source_route_match}, target_token={c_source_token_match}, "
                        f"all_routes={c_all_rows_route_match}, all_tokens={c_all_rows_token_match}, "
                        f"all_inputs={c_all_rows_input_match}"
                    )
                comparisons = {
                    name: compare_records(records["A"], records[name])
                    for name in ("B", "C", "D")
                }
                c_vs_d = compare_records(records["C"], records["D"])
                repeat_public = {
                    "event_id": event_id,
                    "repeat": repeat,
                    "run_id": f"{output_dir.name}:{event_id}:repeat-{repeat}",
                    "timestamp_utc": time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                    ),
                    "arm_order": order,
                    "environment_ref": {
                        "git_commit": config["git_head"],
                        "python": config["runtime"]["python"],
                        "torch": config["runtime"]["torch"],
                        "transformers": config["runtime"]["transformers"],
                        "cuda": config["runtime"]["cuda_version"],
                        "gpu_model": config["runtime"]["gpu"]["name"],
                        "gpu_compute_processes": [
                            list(row) for row in BASE.query_gpu_compute_processes()
                        ],
                    },
                    "event_inputs": {
                        "request_ids": original_ids,
                        "logical_kv_lengths": list(event["logical_kv_lengths"]),
                        "physical_kv_extent": int(event["physical_padded_length"]),
                        "left_padding": list(event["left_padding"]),
                        "forced_token_ids": list(event["forced_token_ids"]),
                    },
                    "reconstruction": reconstruction,
                    "matched_alternative_reconstruction": alternative_closure,
                    "causal_input_equality": {
                        "passed": True,
                        "same_target_logical_cache_sha256": arms["A"].target_logical_cache_sha256,
                        "same_target_token_id": int(arms["A"].input_ids[0, 0].item()),
                        "same_target_position_id": int(arms["A"].position_ids[0, 0].item()),
                        "cache_storage_non_alias_within_and_across_arms": True,
                        "arm_batch_widths": {
                            name: int(arm.input_ids.shape[0]) for name, arm in arms.items()
                        },
                        "arm_physical_kv_extents": {
                            name: int(arm.prior_max) for name, arm in arms.items()
                        },
                        "arm_logical_kv_lengths": {
                            name: list(arm.prior_lengths) for name, arm in arms.items()
                        },
                        "arm_b_target_clone_consistency": clone_consistency,
                    },
                    "source_reproduction": {
                        "arm_c_route_match": c_source_route_match,
                        "arm_c_predicted_token_match": c_source_token_match,
                        "arm_c_all_rows_route_match": c_all_rows_route_match,
                        "arm_c_all_rows_predicted_token_match": c_all_rows_token_match,
                        "arm_c_all_rows_input_token_match": c_all_rows_input_match,
                        "historical_serial_first_known_layer_match": (
                            records["A"]["routes"][
                                int(event["first_known_different_layer"])
                            ]
                            == list(event["serial_experts"])
                        ),
                        "historical_serial_is_selection_metadata_not_truth": True,
                    },
                    "arms": {name: record["public"] for name, record in records.items()},
                    "A_vs": comparisons,
                    "C_vs_D": c_vs_d,
                }
                repeat_path = output_dir / "repeats" / event_id.replace(":", "_") / f"repeat_{repeat}.json"
                write_json_exclusive(repeat_path, repeat_public)
                all_event_repeats[event_id].append(repeat_public)
                if repeat == 0:
                    first_layer = comparisons["C"]["first_any_tensor_difference_layer"]
                    save_target_tensors(
                        torch,
                        output_dir / "target_tensors" / f"{event_id.replace(':', '_')}.pt",
                        records,
                        first_layer,
                    )
                if event_id in PROPAGATION_EVENTS:
                    propagated_states = {
                        name: record["output_state"] for name, record in records.items()
                    }
                    current_expected_length = int(
                        arms["A"].prior_lengths[arms["A"].target_row]
                    ) + 1
                    initial_boundary = validate_propagation_states(
                        torch, propagated_states, current_expected_length
                    )
                    steps = []
                    max_steps = len(capture["ledger"][event["request_id"]]["steps"])
                    for follow_step in range(
                        int(event["decode_step"]) + 1,
                        min(int(event["decode_step"]) + 3, max_steps),
                    ):
                        forced = int(
                            capture["ledger"][event["request_id"]]["steps"][follow_step][
                                "input_token_id"
                            ]
                        )
                        input_boundary = validate_propagation_states(
                            torch, propagated_states, current_expected_length
                        )
                        follow_records, follow_public = propagation_step(
                            torch, model, propagated_states, forced, order
                        )
                        follow_public["decode_step"] = follow_step
                        propagated_states = {
                            name: record["output_state"]
                            for name, record in follow_records.items()
                        }
                        current_expected_length += 1
                        output_boundary = validate_propagation_states(
                            torch, propagated_states, current_expected_length
                        )
                        follow_public["input_state_boundary"] = input_boundary
                        follow_public["output_state_boundary"] = output_boundary
                        steps.append(follow_public)
                    if len(steps) != 2:
                        raise ExperimentError(
                            f"propagation requires exactly two follow steps for {event_id}"
                        )
                    propagation[event_id].append(
                        {
                            "repeat": repeat,
                            "arm_order": order,
                            "post_target_state_boundary": initial_boundary,
                            "teacher_forced_serial_steps": steps,
                        }
                    )
                monitor.check(f"after_{event_id}_repeat_{repeat}")
                monitor.require_clean()
                print(
                    json.dumps(
                        {
                            "event": event_id,
                            "repeat": repeat,
                            "A_vs_C_route_flip_layers": comparisons["C"]["route_flip_layers"],
                            "first_pre_router": comparisons["C"][
                                "first_pre_router_allclose_difference_layer"
                            ],
                            "first_route": comparisons["C"][
                                "first_route_membership_difference_layer"
                            ],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                del states, alternative_states, arms, records
                torch.cuda.empty_cache()
    finally:
        monitor.stop()
    monitor.require_clean()

    classifications = {
        event_id: classify_event(repeats)
        for event_id, repeats in all_event_repeats.items()
    }
    arm_metrics = {
        "schema": "longrun-a-arm-metrics-v1",
        "events": all_event_repeats,
        "classifications": classifications,
        "gpu_isolation": monitor.summary(),
    }
    first_divergence = {
        "schema": "longrun-a-first-divergence-v1",
        "events": {
            event_id: {
                "classification": classifications[event_id],
                "by_repeat": [
                    {
                        "repeat": row["repeat"],
                        "A_vs_B": {
                            key: value
                            for key, value in row["A_vs"]["B"].items()
                            if key != "layers"
                        },
                        "A_vs_C": {
                            key: value
                            for key, value in row["A_vs"]["C"].items()
                            if key != "layers"
                        },
                        "A_vs_D": {
                            key: value
                            for key, value in row["A_vs"]["D"].items()
                            if key != "layers"
                        },
                        "C_vs_D": {
                            key: value
                            for key, value in row["C_vs_D"].items()
                            if key != "layers"
                        },
                    }
                    for row in all_event_repeats[event_id]
                ],
            }
            for event_id in all_event_repeats
        },
    }
    write_json_exclusive(output_dir / "arm_metrics.json", arm_metrics)
    write_json_exclusive(output_dir / "first_divergence.json", first_divergence)
    propagation_stability = summarize_propagation(propagation, args.repeats)
    write_json_exclusive(
        output_dir / "propagation.json",
        {
            "schema": "longrun-a-propagation-v2",
            "events": propagation,
            "stability": propagation_stability,
        },
    )
    sealed_paths = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file()
        and path.name not in {"RUN_COMPLETE.json", "RUN_FAILED.json", "run.log"}
    )
    write_json_exclusive(
        output_dir / "RUN_COMPLETE.json",
        {
            "status": "MAIN_EXPERIMENT_COMPLETE",
            "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "target_event_classifications": {
                key: value["primary"] for key, value in classifications.items()
            },
            "propagation_all_events_stable": propagation_stability[
                "all_events_within_arm_repeat_stable"
            ],
            "files": {
                str(path.relative_to(output_dir)): {
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                }
                for path in sealed_paths
            },
            "unsealed_stream_files": {
                "run.log": "tee stream remains open until process exit"
            },
        },
    )
    print(
        json.dumps(
            {
                "status": "MAIN_EXPERIMENT_COMPLETE",
                "output_dir": str(output_dir),
                "classifications": {
                    key: value["primary"] for key, value in classifications.items()
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        try:
            if "--output-dir" in sys.argv:
                value = sys.argv[sys.argv.index("--output-dir") + 1]
                failed_output = Path(value).resolve()
                complete = failed_output / "RUN_COMPLETE.json"
                failed = failed_output / "RUN_FAILED.json"
                if failed_output.exists() and not complete.exists() and not failed.exists():
                    write_json_exclusive(
                        failed,
                        {
                            "status": "RUN_FAILED",
                            "failed_utc": time.strftime(
                                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                            ),
                            "error_type": type(error).__name__,
                            "error": str(error),
                            "failure_category": classify_failure(error),
                        },
                    )
        except BaseException:
            pass
        raise
