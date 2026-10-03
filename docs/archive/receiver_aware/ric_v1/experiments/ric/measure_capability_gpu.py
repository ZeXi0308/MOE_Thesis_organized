#!/usr/bin/env python3
"""Measure the executable RIC-v1 sender-order/early-release capability on CUDA.

The payloads are reconstructed by executing the frozen model's native experts
on hidden states captured from an unpatched forward.  The experiment changes
only the service order of two ready result blocks.  Streaming release and a
full-layer-barrier negative fixture share the same payloads and canonical
reduction.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys
import tempfile
from typing import Any, Mapping, Sequence


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from capture_routes_gpu import (  # noqa: E402
    NATIVE_TOPK_SELECTION_RULE,
    RouteCaptureError,
    _gpu_compute_apps,
    _gpu_environment,
    _load_config,
    _load_data_manifest,
    _load_model_and_tokenizer,
    _normalizes_topk,
    _producer_source_sha256 as _capture_routes_source_sha256,
    _routes_from_logits,
    assigned_layer,
    discover_moe_modules,
    expert_sender,
    model_load_reference,
    origin_lpt,
    selected_layers,
)
from prepare_data import (  # noqa: E402
    DataPreparationError,
    _producer_source_sha256 as _prepare_data_source_sha256,
    add_self_hash,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)
from formal_provenance import (  # noqa: E402
    FormalProvenanceError,
    is_sha256,
    loads_json_mapping_strict,
    materialize_verified_signoff,
    verify_phase4_signoff,
)
from capability_contract import (  # noqa: E402
    EXECUTION_ORDER_RULE,
    CapabilityContractError,
    capability_execution_order,
)


REPO_ROOT = next(candidate for candidate in HERE.parents if (candidate / "experiments/shared").is_dir())
IDEA_ROOT = HERE.parents[1]
DEFAULT_CONFIG = IDEA_ROOT / "configs" / "ric_v1.json"
DEFAULT_PROTOCOL = IDEA_ROOT / "RIC_Phase2_冻结实验协议_2026-07-22.md"
FORMAL_BLOCK_ROWS = 32
FORMAL_WARMUPS = 10
FORMAL_TRIALS = 30


class CapabilityError(RuntimeError):
    """A capability/provenance invariant failed."""


def _producer_source_sha256() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__),
        HERE / "capture_routes_gpu.py",
        HERE / "prepare_data.py",
        HERE / "formal_provenance.py",
        HERE / "capability_contract.py",
    ):
        digest.update(str(path.resolve().relative_to(REPO_ROOT.resolve())).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _require_formal_signoff(
    path: Path | None,
    *,
    protocol_sha256: str,
    config_sha256: str,
    source_sha256: str,
    data_manifest_sha256: str,
    data_producer_signoff_sha256: str,
    model_key: str,
    model_tree_manifest_sha256: str,
) -> Mapping[str, Any]:
    try:
        return verify_phase4_signoff(
            path,
            repo_root=REPO_ROOT,
            expected_fields={
                "stage": "measure_capability",
                "protocol_sha256": protocol_sha256,
                "config_sha256": config_sha256,
                "measure_capability_source_sha256": source_sha256,
                "capture_routes_source_sha256": _capture_routes_source_sha256(),
                "prepare_data_source_sha256": _prepare_data_source_sha256(),
                "data_manifest_sha256": data_manifest_sha256,
                "data_producer_signoff_sha256": data_producer_signoff_sha256,
                "model_key": model_key,
                "model_tree_manifest_sha256": model_tree_manifest_sha256,
            },
            required_source_paths=(
                Path(__file__),
                HERE / "capture_routes_gpu.py",
                HERE / "prepare_data.py",
                HERE / "formal_provenance.py",
                HERE / "capability_contract.py",
            ),
        )
    except (FormalProvenanceError, DataPreparationError) as exc:
        raise CapabilityError(str(exc)) from exc


def _first_tensor(value: Any) -> Any:
    if hasattr(value, "shape"):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            if hasattr(item, "shape"):
                return item
    raise CapabilityError("expert output contains no tensor")


def tensor_sha256(tensor: Any) -> str:
    """Hash the exact tensor representation, without dtype conversion."""

    import torch

    original = tensor.detach().cpu()
    contiguous = original.contiguous()
    raw_bytes = contiguous.view(torch.uint8).numpy().tobytes(order="C")
    digest = hashlib.sha256()
    digest.update(str(tuple(original.shape)).encode())
    digest.update(b"\0")
    digest.update(str(original.dtype).encode())
    digest.update(b"\0")
    digest.update(str(tuple(original.stride())).encode())
    digest.update(b"\0")
    digest.update(raw_bytes)
    return digest.hexdigest()


def profiler_gpu_stream_ordinal(path: Path) -> int:
    """Require all profiler-visible GPU activities to use one stream ordinal."""

    try:
        payload = loads_json_mapping_strict(
            path.read_text(encoding="utf-8"), label="capability profiler trace"
        )
    except (OSError, UnicodeError, FormalProvenanceError) as exc:
        raise CapabilityError("capability profiler trace is invalid JSON") from exc
    events = payload.get("traceEvents")
    if not isinstance(events, list):
        raise CapabilityError("capability profiler trace lacks traceEvents")
    streams: set[int] = set()
    gpu_event_count = 0
    for event in events:
        if (
            not isinstance(event, Mapping)
            or str(event.get("cat", "")).lower()
            not in {"kernel", "gpu_memcpy", "gpu_memset"}
        ):
            continue
        gpu_event_count += 1
        args = event.get("args")
        stream = args.get("stream") if isinstance(args, Mapping) else None
        if type(stream) is not int or stream < 0:
            raise CapabilityError("capability GPU activity lacks a stream ordinal")
        streams.add(stream)
    if gpu_event_count < 1 or len(streams) != 1:
        raise CapabilityError(
            "capability profiler observed mixed or missing GPU stream activity"
        )
    return next(iter(streams))


def canonical_reduce(siblings: Any) -> Any:
    """Reduce [rows, top_k, hidden] in frozen top-k-slot order."""

    if siblings.ndim != 3 or siblings.shape[1] < 2:
        raise CapabilityError("canonical siblings must be [rows, top_k, hidden]")
    accumulator = siblings[:, 0, :].clone()
    for topk_slot in range(1, siblings.shape[1]):
        accumulator = accumulator + siblings[:, topk_slot, :]
    return accumulator


def _sender_pack(block: Any) -> Any:
    """Pack contribution rows into the frozen reverse-row wire order."""

    import torch

    permutation = torch.arange(
        block.shape[0] - 1, -1, -1, device=block.device, dtype=torch.long
    )
    return torch.index_select(block, 0, permutation)


def _receiver_unpack(packed: Any) -> Any:
    """Unpack frozen reverse-row wire order back to canonical row order."""

    import torch

    permutation = torch.arange(
        packed.shape[0] - 1, -1, -1, device=packed.device, dtype=torch.long
    )
    unpacked = torch.empty_like(packed)
    unpacked.index_copy_(0, permutation, packed)
    return unpacked


def _pack_roundtrip(block: Any) -> Any:
    """Capability fixture composition; LUT times pack and unpack separately."""

    return _receiver_unpack(_sender_pack(block))


@contextmanager
def _nvtx_range(torch: Any, label: str):
    """Emit both a real NVTX range and a torch-profiler-visible label."""

    with torch.profiler.record_function(label):
        torch.cuda.nvtx.range_push(label)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()


def _fixture_id(identity: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(identity))


def select_sender_local_blocks(
    selected_experts: Sequence[Sequence[int]],
    *,
    request_id: str,
    layer_id: int,
    model_revision: str,
    selection_seed: int,
    num_experts: int,
    ep_size: int,
    block_rows: int,
) -> dict[str, Any]:
    """Select two disjoint route-derived blocks owned by one sender.

    The rule reads only frozen route identities and source constants.  It does
    not inspect service time, policy order, closure, or any measured outcome.
    Each block contains at most one contribution from a token join.
    """

    if block_rows < 1 or ep_size < 1 or num_experts < ep_size:
        raise CapabilityError("invalid sender-local fixture geometry")
    if not selected_experts:
        raise CapabilityError("sender-local fixture route matrix is empty")
    top_k = len(selected_experts[0])
    if top_k < 2 or any(len(row) != top_k for row in selected_experts):
        raise CapabilityError("sender-local fixture route matrix is ragged")

    by_sender: dict[int, dict[int, list[dict[str, int]]]] = {}
    for token_index, row in enumerate(selected_experts):
        if len(set(int(value) for value in row)) != top_k:
            raise CapabilityError("one route row contains a duplicate expert")
        for topk_slot, raw_expert in enumerate(row):
            expert_id = int(raw_expert)
            sender_rank = expert_sender(expert_id, num_experts, ep_size)
            by_sender.setdefault(sender_rank, {}).setdefault(token_index, []).append(
                {
                    "token_index": token_index,
                    "topk_slot": topk_slot,
                    "expert_id": expert_id,
                    "sender_rank": sender_rank,
                }
            )

    needed = 2 * block_rows
    one_per_token: dict[int, list[dict[str, int]]] = {}
    for sender_rank, token_candidates in by_sender.items():
        chosen = []
        for token_index, candidates in token_candidates.items():
            chosen.append(
                min(
                    candidates,
                    key=lambda row: sha256_bytes(
                        canonical_json_bytes(
                            {
                                "rule": "ric-g1-sender-local-slot-v1",
                                "selection_seed": selection_seed,
                                "model_revision": model_revision,
                                "request_id": request_id,
                                "layer_id": layer_id,
                                "sender_rank": sender_rank,
                                "token_index": token_index,
                                "topk_slot": row["topk_slot"],
                                "expert_id": row["expert_id"],
                            }
                        )
                    ),
                )
            )
        one_per_token[sender_rank] = chosen
    eligible = [
        sender_rank
        for sender_rank, candidates in one_per_token.items()
        if len(candidates) >= needed
    ]
    if not eligible:
        support = {sender: len(rows) for sender, rows in sorted(one_per_token.items())}
        raise CapabilityError(
            "BLOCKED_G1_SENDER_LOCAL_SUPPORT: "
            f"no sender owns {needed} distinct-token contributions; support={support}"
        )
    sender_rank = min(
        eligible,
        key=lambda sender: sha256_bytes(
            canonical_json_bytes(
                {
                    "rule": "ric-g1-sender-choice-v1",
                    "selection_seed": selection_seed,
                    "model_revision": model_revision,
                    "request_id": request_id,
                    "layer_id": layer_id,
                    "sender_rank": sender,
                }
            )
        ),
    )
    ordered = sorted(
        one_per_token[sender_rank],
        key=lambda row: sha256_bytes(
            canonical_json_bytes(
                {
                    "rule": "ric-g1-contribution-order-v1",
                    "selection_seed": selection_seed,
                    "model_revision": model_revision,
                    "request_id": request_id,
                    "layer_id": layer_id,
                    **row,
                }
            )
        ),
    )[:needed]
    if len({row["token_index"] for row in ordered}) != needed:
        raise CapabilityError("sender-local fixture reused a token join")
    return {
        "selection_rule": "route_identity_hash_sender_local_distinct_token_v1",
        "sender_rank": sender_rank,
        "x_closing": ordered[:block_rows],
        "y_nonclosing": ordered[block_rows:],
        "support_by_sender": {
            str(sender): len(rows) for sender, rows in sorted(one_per_token.items())
        },
    }


def build_task_fixture(
    *,
    task_name: str,
    block: Any,
    model_key: str,
    model_revision: str,
    sender_rank: int,
    receiver_rank: int,
    ranks_per_node: int,
    contribution_identities: Sequence[Mapping[str, Any]],
    queue_id: str,
) -> dict[str, Any]:
    """Build a stable, route-derived identity for one ready result block."""

    row_count = int(block.shape[0])
    if row_count < 1 or len(contribution_identities) != row_count:
        raise CapabilityError("capability fixture row/identity mismatch")
    required = {
        "request_id",
        "forward_id",
        "batch_id",
        "phase",
        "decode_step",
        "layer_id",
        "token_id",
        "token_block_id",
        "topk_slot",
        "expert_id",
        "sender_rank",
        "receiver_rank",
        "epoch",
    }
    rows = [dict(row) for row in contribution_identities]
    if any(set(row) != required for row in rows):
        raise CapabilityError("capability contribution identity schema mismatch")
    if any(
        int(row["sender_rank"]) != sender_rank
        or int(row["receiver_rank"]) != receiver_rank
        for row in rows
    ):
        raise CapabilityError("capability block mixes sender/receiver ownership")
    if len({str(row["token_block_id"]) for row in rows}) != row_count:
        raise CapabilityError("capability block reuses a token join")
    identity = {
        "schema_version": "ric-capability-task-v1",
        "task_name": task_name,
        "model_key": model_key,
        "model_revision": model_revision,
        "sender_rank": int(sender_rank),
        "receiver_rank": int(receiver_rank),
        "sender_local_queue_id": queue_id,
        "shared_cut_path": (
            f"node{sender_rank // ranks_per_node}->"
            f"node{receiver_rank // ranks_per_node}"
        ),
        "receiver_combine_resource": f"receiver:{receiver_rank}:combine",
        "contribution_identities": rows,
        "payload_shape": [int(value) for value in block.shape],
        "payload_stride": [int(value) for value in block.stride()],
        "payload_dtype": str(block.dtype),
        "payload_bytes": int(block.numel() * block.element_size()),
        "payload_sha256": tensor_sha256(block),
    }
    return {**identity, "fixture_identity_sha256": _fixture_id(identity)}


def _reconstruct_real_contributions(
    *,
    moe: Any,
    hidden_states: Any,
    router_logits: Any,
    top_k: int,
) -> tuple[Any, Any, Any]:
    """Execute native experts and return weighted per-token contributions."""

    import torch

    hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
    experts, weights = _routes_from_logits(
        router_logits.reshape(-1, router_logits.shape[-1]),
        top_k=top_k,
        normalize_topk=_normalizes_topk(moe),
        selection_rule=NATIVE_TOPK_SELECTION_RULE,
        output_dtype=hidden.dtype,
    )
    raw = torch.zeros(
        (hidden.shape[0], top_k, hidden.shape[-1]),
        device=hidden.device,
        dtype=hidden.dtype,
    )
    with torch.inference_mode():
        for expert_id in range(len(moe.experts)):
            token_idx, slot_idx = torch.where(experts == expert_id)
            if token_idx.numel() == 0:
                continue
            output = _first_tensor(moe.experts[expert_id](hidden[token_idx]))
            if output.shape != hidden[token_idx].shape:
                raise CapabilityError("native expert output shape mismatch")
            raw[token_idx, slot_idx, :] = output
    weighted = raw * weights.to(dtype=raw.dtype).unsqueeze(-1)
    return weighted, experts, weights


def _run_trial(
    *,
    policy: str,
    release_mode: str,
    a_siblings: Any,
    x_closing: Any,
    x_closing_slots: Any,
    y_nonclosing: Any,
    task_fixtures: Mapping[str, Mapping[str, Any]],
    trial: int,
    execution_order_index: int,
    stream: Any,
) -> tuple[dict[str, float], str, list[dict[str, Any]]]:
    import torch

    if policy == "candidate_closing_first":
        order = ("x_closing", "y_nonclosing")
    elif policy == "baseline_nonclosing_first":
        order = ("y_nonclosing", "x_closing")
    else:
        raise CapabilityError(f"unknown policy {policy}")
    if release_mode not in {"streaming", "full_layer_barrier"}:
        raise CapabilityError(f"unknown release mode {release_mode}")
    if set(task_fixtures) != {"x_closing", "y_nonclosing"}:
        raise CapabilityError("capability queue requires exactly two frozen tasks")

    start = torch.cuda.Event(enable_timing=True)
    queue_ready = torch.cuda.Event(enable_timing=True)
    physical_frontier = torch.cuda.Event(enable_timing=True)
    application_release = torch.cuda.Event(enable_timing=True)
    downstream_start = torch.cuda.Event(enable_timing=True)
    downstream_end = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    task_events = {
        task_name: {
            name: torch.cuda.Event(enable_timing=True)
            for name in ("enqueue", "selected", "service_start", "service_end")
        }
        for task_name in task_fixtures
    }
    ready_task_ids = [
        str(task_fixtures[name]["fixture_identity_sha256"])
        for name in ("x_closing", "y_nonclosing")
    ]
    queue_snapshot = {
        "ready_count": 2,
        "ready_task_ids": ready_task_ids,
        "all_ready_before_selection": True,
    }
    queue_snapshot_sha = sha256_bytes(canonical_json_bytes(queue_snapshot))
    group_id = sha256_bytes(
        canonical_json_bytes(
            {
                "trial": int(trial),
                "execution_order_index": int(execution_order_index),
                "policy": policy,
                "release_mode": release_mode,
                "ready_task_ids": ready_task_ids,
            }
        )
    )
    joined = None
    trial_label = f"ric_capability/trial={trial}/policy={policy}/release={release_mode}"
    with _nvtx_range(torch, trial_label), torch.cuda.stream(stream):
        start.record(stream)
        # Both payloads already exist on device.  Enqueue both before the first
        # selection so this is an explicit two-ready-task sender-local queue.
        for task_name in ("x_closing", "y_nonclosing"):
            with _nvtx_range(torch, f"{trial_label}/enqueue={task_name}"):
                task_events[task_name]["enqueue"].record(stream)
        with _nvtx_range(torch, f"{trial_label}/queue_snapshot=both_ready"):
            queue_ready.record(stream)
        for task_name in order:
            events = task_events[task_name]
            with _nvtx_range(torch, f"{trial_label}/select={task_name}"):
                events["selected"].record(stream)
            events["service_start"].record(stream)
            task_label = f"{trial_label}/service={task_name}"
            with _nvtx_range(torch, task_label):
                if task_name == "x_closing":
                    serviced_x = _pack_roundtrip(x_closing)
                    completed = a_siblings.clone()
                    completed[
                        torch.arange(
                            completed.shape[0], device=completed.device, dtype=torch.long
                        ),
                        x_closing_slots,
                        :,
                    ] = serviced_x
                    joined = canonical_reduce(completed)
                    physical_frontier.record(stream)
                    if release_mode == "streaming":
                        application_release.record(stream)
                        downstream_start.record(stream)
                        downstream = torch.tanh(joined.float())
                        downstream_end.record(stream)
                else:
                    serviced_y = _pack_roundtrip(y_nonclosing)
                    # A real CUDA consumer prevents the non-closing service
                    # from becoming a dead timing-only Python object.
                    _nonclosing_sink = serviced_y.float().square().sum()
            events["service_end"].record(stream)
        if joined is None:
            raise CapabilityError("closing task was never serviced")
        if release_mode == "full_layer_barrier":
            application_release.record(stream)
            downstream_start.record(stream)
            downstream = torch.tanh(joined.float())
            downstream_end.record(stream)
        end.record(stream)
        # Diagnostics are excluded from ``end`` timing but must remain on the
        # same logical sender-local stream.  Performing either operation after
        # leaving this context silently schedules CUDA reduction/D2H work on
        # the restored default stream and contaminates the next arm.
        downstream_checksum_value = float(downstream.sum().item())
        output_hash = tensor_sha256(joined)
    end.synchronize()
    metrics = {
        "physical_frontier_us": float(start.elapsed_time(physical_frontier) * 1000.0),
        "application_release_us": float(start.elapsed_time(application_release) * 1000.0),
        "downstream_start_us": float(start.elapsed_time(downstream_start) * 1000.0),
        "downstream_end_us": float(start.elapsed_time(downstream_end) * 1000.0),
        "total_us": float(start.elapsed_time(end) * 1000.0),
        "downstream_checksum": downstream_checksum_value,
    }
    stream_id = int(stream.cuda_stream)
    action_rows: list[dict[str, Any]] = []
    for order_index, task_name in enumerate(order):
        events = task_events[task_name]
        identity = dict(task_fixtures[task_name])
        action_rows.append(
            {
                "schema_version": "ric-capability-action-v1",
                "action_trace_group_id": group_id,
                "trial": int(trial),
                "execution_order_index": int(execution_order_index),
                "policy": policy,
                "release_mode": release_mode,
                "service_order_index": order_index,
                "task_name": task_name,
                "fixture_identity_sha256": identity["fixture_identity_sha256"],
                "task_identity": identity,
                "payload_bytes": int(identity["payload_bytes"]),
                "stream_id": stream_id,
                "queue_snapshot": queue_snapshot,
                "queue_snapshot_sha256": queue_snapshot_sha,
                "enqueue_ts_us": float(start.elapsed_time(events["enqueue"]) * 1000.0),
                "queue_ready_ts_us": float(start.elapsed_time(queue_ready) * 1000.0),
                "selected_ts_us": float(start.elapsed_time(events["selected"]) * 1000.0),
                "service_start_ts_us": float(
                    start.elapsed_time(events["service_start"]) * 1000.0
                ),
                "service_end_ts_us": float(
                    start.elapsed_time(events["service_end"]) * 1000.0
                ),
                "nvtx_range_label": f"{trial_label}/service={task_name}",
                "nvtx_labels": {
                    "enqueue": f"{trial_label}/enqueue={task_name}",
                    "queue_snapshot": f"{trial_label}/queue_snapshot=both_ready",
                    "select": f"{trial_label}/select={task_name}",
                    "service": f"{trial_label}/service={task_name}",
                },
                "source": "measured_5090_cuda",
            }
        )
    return metrics, output_hash, action_rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise CapabilityError("refusing to write empty capability trials")
    fields = list(rows[0])
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            if list(row) != fields:
                raise CapabilityError("capability trial schema drift")
            writer.writerow(row)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise CapabilityError("refusing to write empty capability action trace")
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def _median(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    return float(statistics.median(float(row[field]) for row in rows))


def paired_effect_lcbs(
    effects: Mapping[str, Sequence[float]],
    *,
    replicates: int,
    order_statistic_one_based: int,
    seed: int,
) -> dict[str, float]:
    """Paired bootstrap LCBs using common within-trial resample indices."""

    if (
        not effects
        or type(replicates) is not int
        or replicates < 1
        or type(order_statistic_one_based) is not int
        or not 1 <= order_statistic_one_based <= replicates
    ):
        raise CapabilityError("invalid paired capability bootstrap")
    lengths = {len(values) for values in effects.values()}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) < 1:
        raise CapabilityError("paired capability effects are incomplete")
    count = next(iter(lengths))
    rng = random.Random(seed)
    distributions = {name: [] for name in effects}
    for _ in range(replicates):
        indexes = [rng.randrange(count) for _row in range(count)]
        for name, values in effects.items():
            distributions[name].append(
                math.fsum(float(values[index]) for index in indexes) / count
            )
    rank = order_statistic_one_based - 1
    return {
        name: sorted(values)[rank] for name, values in distributions.items()
    }


def event_precedence_failures(
    raw_rows: Sequence[Mapping[str, Any]],
    action_rows: Sequence[Mapping[str, Any]],
    *,
    trials: int,
) -> list[str]:
    """Recompute Amendment-O event semantics from raw CUDA-event records."""

    policies = ("baseline_nonclosing_first", "candidate_closing_first")
    releases = ("streaming", "full_layer_barrier")
    raw_by_key = {
        (int(row["trial"]), str(row["policy"]), str(row["release_mode"])): row
        for row in raw_rows
    }
    actions_by_key: dict[tuple[int, str, str], list[Mapping[str, Any]]] = {}
    for row in action_rows:
        key = (int(row["trial"]), str(row["policy"]), str(row["release_mode"]))
        actions_by_key.setdefault(key, []).append(row)
    expected = {
        (trial, policy, release)
        for trial in range(trials)
        for policy in policies
        for release in releases
    }
    failures: list[str] = []
    if set(raw_by_key) != expected or set(actions_by_key) != expected:
        return ["event_grid_mismatch"]

    def ordered_actions(key: tuple[int, str, str]) -> list[Mapping[str, Any]]:
        return sorted(
            actions_by_key[key], key=lambda row: int(row["service_order_index"])
        )

    def action_by_name(
        key: tuple[int, str, str], task_name: str
    ) -> Mapping[str, Any]:
        matches = [row for row in actions_by_key[key] if row["task_name"] == task_name]
        if len(matches) != 1:
            raise CapabilityError("capability event trace task coverage is invalid")
        return matches[0]

    for trial in range(trials):
        for policy in policies:
            streaming_key = (trial, policy, "streaming")
            barrier_key = (trial, policy, "full_layer_barrier")
            streaming_ordered = ordered_actions(streaming_key)
            barrier_ordered = ordered_actions(barrier_key)
            cross_release_fields = (
                "task_name",
                "fixture_identity_sha256",
                "task_identity",
                "payload_bytes",
                "stream_id",
                "queue_snapshot",
                "queue_snapshot_sha256",
                "service_order_index",
            )
            if len(streaming_ordered) != 2 or len(barrier_ordered) != 2 or any(
                streaming[field] != barrier[field]
                for streaming, barrier in zip(streaming_ordered, barrier_ordered)
                for field in cross_release_fields
            ):
                failures.append(f"trial={trial}/policy={policy}/cross_release_identity")

            barrier_raw = raw_by_key[barrier_key]
            barrier_release = float(barrier_raw["application_release_us"])
            barrier_downstream = float(barrier_raw["downstream_start_us"])
            barrier_ends = [
                float(row["service_end_ts_us"]) for row in barrier_ordered
            ]
            if not all(end <= barrier_release for end in barrier_ends):
                failures.append(f"trial={trial}/policy={policy}/barrier_early_release")
            if barrier_release > barrier_downstream:
                failures.append(
                    f"trial={trial}/policy={policy}/barrier_downstream_before_release"
                )

            streaming_raw = raw_by_key[streaming_key]
            frontier = float(streaming_raw["physical_frontier_us"])
            release = float(streaming_raw["application_release_us"])
            downstream = float(streaming_raw["downstream_start_us"])
            if frontier > release:
                failures.append(f"trial={trial}/policy={policy}/release_before_frontier")
            if release > downstream:
                failures.append(
                    f"trial={trial}/policy={policy}/streaming_downstream_before_release"
                )

            y = action_by_name(streaming_key, "y_nonclosing")
            x = action_by_name(streaming_key, "x_closing")
            if not (
                float(x["service_start_ts_us"])
                <= frontier
                <= float(x["service_end_ts_us"])
            ):
                failures.append(
                    f"trial={trial}/policy={policy}/streaming_frontier_outside_closing"
                )
            barrier_x = action_by_name(barrier_key, "x_closing")
            barrier_frontier = float(barrier_raw["physical_frontier_us"])
            if not (
                float(barrier_x["service_start_ts_us"])
                <= barrier_frontier
                <= float(barrier_x["service_end_ts_us"])
            ):
                failures.append(
                    f"trial={trial}/policy={policy}/barrier_frontier_outside_closing"
                )
            if policy == "candidate_closing_first":
                y_start = float(y["service_start_ts_us"])
                if not (release < y_start and downstream < y_start):
                    failures.append(
                        f"trial={trial}/policy={policy}/nonclosing_before_early_use"
                    )
                barrier_y_start = float(
                    action_by_name(barrier_key, "y_nonclosing")["service_start_ts_us"]
                )
                if not barrier_frontier < barrier_y_start:
                    failures.append(
                        f"trial={trial}/policy={policy}/barrier_frontier_order"
                    )
            else:
                if float(y["service_end_ts_us"]) > float(x["service_start_ts_us"]):
                    failures.append(f"trial={trial}/policy={policy}/baseline_order")
                barrier_y = action_by_name(barrier_key, "y_nonclosing")
                if float(barrier_y["service_end_ts_us"]) > float(
                    barrier_x["service_start_ts_us"]
                ):
                    failures.append(
                        f"trial={trial}/policy={policy}/barrier_baseline_order"
                    )
    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-key", choices=("olmoe", "llmjp"), required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("dev", "formal"), default="dev")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--signoff", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--block-rows", type=int, default=FORMAL_BLOCK_ROWS)
    parser.add_argument("--warmups", type=int, default=FORMAL_WARMUPS)
    parser.add_argument("--trials", type=int, default=FORMAL_TRIALS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment capability
        raise CapabilityError("CUDA PyTorch is required") from exc
    if not torch.cuda.is_available():
        raise CapabilityError("CUDA is required; proxy capability fallback is forbidden")
    if args.output_dir.exists():
        raise CapabilityError("refusing to overwrite capability output directory")
    try:
        compute_apps_before = _gpu_compute_apps()
    except RouteCaptureError as exc:
        raise CapabilityError(str(exc)) from exc
    if min(args.block_rows, args.warmups, args.trials) < 1:
        raise CapabilityError("block rows, warmups and trials must be positive")
    if args.mode == "formal" and (
        args.block_rows != FORMAL_BLOCK_ROWS
        or args.warmups != FORMAL_WARMUPS
        or args.trials != FORMAL_TRIALS
    ):
        raise CapabilityError("formal capability parameters are source-frozen")
    config = _load_config(args.config)
    capability_cfg = config.get("capability_probes")
    if not isinstance(capability_cfg, Mapping):
        raise CapabilityError("config lacks frozen capability probe contract")
    event_gate = capability_cfg.get("event_precedence_gate")
    event_fields = {
        "same_policy_cross_release_task_identity_queue_payload_and_order_exact",
        "barrier_release_after_both_task_service_end",
        "barrier_downstream_not_before_release",
        "streaming_release_at_or_after_physical_frontier",
        "streaming_downstream_not_before_release",
        "physical_frontier_within_closing_service",
        "candidate_streaming_release_and_downstream_before_nonclosing_service",
        "candidate_barrier_frontier_before_nonclosing_service",
        "baseline_nonclosing_service_before_closing_service",
        "all_trials_and_policies_required",
    }
    if (
        not isinstance(event_gate, Mapping)
        or set(event_gate) != event_fields
        or any(event_gate[field] is not True for field in event_fields)
        or capability_cfg.get("barrier_cross_policy_timing_is_diagnostic_only")
        is not True
        or capability_cfg.get("profiler_diagnostics_required_release_modes")
        != ["streaming", "full_layer_barrier"]
        or capability_cfg.get("formal_block_rows") != FORMAL_BLOCK_ROWS
        or capability_cfg.get("formal_warmups") != FORMAL_WARMUPS
        or capability_cfg.get("measured_trials") != FORMAL_TRIALS
    ):
        raise CapabilityError("config lacks frozen Amendment-O capability semantics")
    existence_gate = capability_cfg.get("paired_existence_gate")
    expected_effect_definitions = {
        "frontier_advance": (
            "baseline_frontier_release_us-candidate_frontier_release_us"
        ),
        "downstream_start_advance": (
            "baseline_downstream_start_us-candidate_downstream_start_us"
        ),
        "release_interaction": (
            "(baseline-candidate)_streaming_release-"
            "(baseline-candidate)_barrier_release"
        ),
        "downstream_interaction": (
            "(baseline-candidate)_streaming_downstream-"
            "(baseline-candidate)_barrier_downstream"
        ),
    }
    if (
        not isinstance(existence_gate, Mapping)
        or existence_gate.get("effect_definitions_us")
        != expected_effect_definitions
        or existence_gate.get("estimator")
        != "mean_of_30_within_trial_paired_differences"
        or existence_gate.get("bootstrap_unit") != "within_trial_pair"
        or existence_gate.get("bootstrap_replicates") != 10000
        or existence_gate.get("bootstrap_seed") != 2026072226
        or type(existence_gate.get("one_sided_confidence")) is not float
        or existence_gate.get("one_sided_confidence") != 0.95
        or existence_gate.get("quantile") != "type1_nearest_rank"
        or existence_gate.get("lcb_order_statistic_one_based") != 500
        or type(
            existence_gate.get("each_effect_lcb_must_be_strictly_greater_than_us")
        )
        not in {int, float}
        or existence_gate.get(
            "each_effect_lcb_must_be_strictly_greater_than_us"
        )
        != 0.0
        or existence_gate.get("logical_operator") != "AND"
    ):
        raise CapabilityError("config lacks frozen Amendment-O paired estimand")
    protocol_sha = sha256_file(args.protocol)
    config_sha = sha256_file(args.config)
    try:
        manifest = _load_data_manifest(
            args.data_manifest,
            mode=args.mode,
            model_key=args.model_key,
            config=config,
            protocol_sha256=protocol_sha,
            config_sha256=config_sha,
        )
    except RouteCaptureError as exc:
        raise CapabilityError(str(exc)) from exc
    if manifest.get("role") != "calibration":
        raise CapabilityError("G1 capability measurement consumes calibration only")

    source_sha = _producer_source_sha256()
    manifest_sha = str(manifest["manifest_sha256"])
    data_producer_signoff_sha = manifest.get("signoff_sha256")
    spec = config["models"][args.model_key]
    if args.mode == "formal" and args.model_path is None:
        raise CapabilityError("formal capability requires an explicit hashed --model-path")
    model_reference = model_load_reference(spec, args.model_path)
    model_tree_sha = model_reference[2].get("tree_manifest_sha256")
    signoff = None
    if args.mode == "formal":
        if not is_sha256(data_producer_signoff_sha):
            raise CapabilityError("formal data producer signoff hash is missing")
        if not isinstance(model_tree_sha, str):
            raise CapabilityError("formal local model tree has no manifest hash")
        signoff = _require_formal_signoff(
            args.signoff,
            protocol_sha256=protocol_sha,
            config_sha256=config_sha,
            source_sha256=source_sha,
            data_manifest_sha256=manifest_sha,
            data_producer_signoff_sha256=str(data_producer_signoff_sha),
            model_key=args.model_key,
            model_tree_manifest_sha256=model_tree_sha,
        )

    model_revision = f"{spec['repo_id']}@{spec['revision']}"
    model, tokenizer, transformers_version, model_source = _load_model_and_tokenizer(
        spec,
        cache_dir=args.cache_dir,
        allow_download=args.allow_download,
        model_path=args.model_path,
        model_reference=model_reference,
    )
    modules = discover_moe_modules(model)
    layer_ids = [row[0] for row in modules]
    frozen_layers = selected_layers(
        layer_ids,
        selection_seed=int(config["data"]["selection_seed"]),
        model_revision=model_revision,
        count=int(config["route_capture"]["selected_layer_count_per_model"]),
    )
    request = manifest["requests"][0]
    request_id = str(request["request_id"])
    target_layer = assigned_layer(request_id, frozen_layers)
    moe = {layer: module for layer, _name, module in modules}[target_layer]
    captured: dict[str, Any] = {}

    def input_hook(_module: Any, inputs: tuple[Any, ...]) -> None:
        if "hidden" in captured:
            raise CapabilityError("target MoE module called more than once")
        captured["hidden"] = _first_tensor(inputs).detach()

    def gate_hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
        if "router_logits" in captured:
            raise CapabilityError("target gate called more than once")
        captured["router_logits"] = _first_tensor(output).detach()

    input_handle = moe.register_forward_pre_hook(input_hook)
    gate_handle = moe.gate.register_forward_hook(gate_hook)
    try:
        text = str(request["text"])
        if sha256_bytes(text.encode("utf-8")) != request["text_sha256"]:
            raise CapabilityError("calibration request text hash mismatch")
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=int(config["data"]["sequence_length"]),
            return_tensors="pt",
        )
        if tuple(encoded["input_ids"].shape) != (1, 128):
            raise CapabilityError("capability request is not exactly 128 tokens")
        with torch.inference_mode():
            model(
                input_ids=encoded["input_ids"].to("cuda:0"),
                use_cache=False,
                output_router_logits=True,
                return_dict=True,
            )
        torch.cuda.synchronize()
    finally:
        input_handle.remove()
        gate_handle.remove()
    if set(captured) != {"hidden", "router_logits"}:
        raise CapabilityError("native hidden/router capture incomplete")
    weighted, selected_experts, route_weights = _reconstruct_real_contributions(
        moe=moe,
        hidden_states=captured["hidden"],
        router_logits=captured["router_logits"],
        top_k=int(spec["top_k"]),
    )
    ep_size = int(config["topology_proxy"]["ep_size"])
    ranks_per_node = int(config["topology_proxy"]["ranks_per_node"])
    receiver_rank = origin_lpt(manifest["requests"], ep_size)[request_id]
    plan = select_sender_local_blocks(
        selected_experts.detach().cpu().tolist(),
        request_id=request_id,
        layer_id=target_layer,
        model_revision=model_revision,
        selection_seed=int(config["data"]["selection_seed"]),
        num_experts=int(spec["num_experts"]),
        ep_size=ep_size,
        block_rows=args.block_rows,
    )
    sender_rank = int(plan["sender_rank"])
    x_rows = list(plan["x_closing"])
    y_rows = list(plan["y_nonclosing"])
    x_token_indices = torch.tensor(
        [row["token_index"] for row in x_rows], device=weighted.device, dtype=torch.long
    )
    y_token_indices = torch.tensor(
        [row["token_index"] for row in y_rows], device=weighted.device, dtype=torch.long
    )
    x_closing_slots = torch.tensor(
        [row["topk_slot"] for row in x_rows], device=weighted.device, dtype=torch.long
    )
    y_nonclosing_slots = torch.tensor(
        [row["topk_slot"] for row in y_rows], device=weighted.device, dtype=torch.long
    )
    a_siblings = weighted.index_select(0, x_token_indices).contiguous()
    b_siblings = weighted.index_select(0, y_token_indices).contiguous()
    row_index = torch.arange(args.block_rows, device=weighted.device, dtype=torch.long)
    x_closing = a_siblings[row_index, x_closing_slots, :].contiguous()
    y_nonclosing = b_siblings[row_index, y_nonclosing_slots, :].contiguous()
    reference = canonical_reduce(a_siblings)
    reference_hash = tensor_sha256(reference)

    def contribution_identities(rows: Sequence[Mapping[str, int]]) -> list[dict[str, Any]]:
        result = []
        for row in rows:
            token_index = int(row["token_index"])
            result.append(
                {
                    "request_id": request_id,
                    "forward_id": f"{request_id}:capability-forward",
                    "batch_id": f"{request_id}:capability-batch",
                    "phase": "prefill",
                    "decode_step": 0,
                    "layer_id": int(target_layer),
                    "token_id": f"{request_id}:token:{token_index}",
                    "token_block_id": f"{request_id}:token-block:{token_index}",
                    "topk_slot": int(row["topk_slot"]),
                    "expert_id": int(row["expert_id"]),
                    "sender_rank": sender_rank,
                    "receiver_rank": receiver_rank,
                    "epoch": 1,
                }
            )
        return result

    queue_id = f"cuda:0/ric-capability/sender:{sender_rank}:local-return-queue"
    task_fixtures = {
        "x_closing": build_task_fixture(
            task_name="x_closing",
            block=x_closing,
            model_key=args.model_key,
            model_revision=model_revision,
            sender_rank=sender_rank,
            receiver_rank=receiver_rank,
            ranks_per_node=ranks_per_node,
            contribution_identities=contribution_identities(x_rows),
            queue_id=queue_id,
        ),
        "y_nonclosing": build_task_fixture(
            task_name="y_nonclosing",
            block=y_nonclosing,
            model_key=args.model_key,
            model_revision=model_revision,
            sender_rank=sender_rank,
            receiver_rank=receiver_rank,
            ranks_per_node=ranks_per_node,
            contribution_identities=contribution_identities(y_rows),
            queue_id=queue_id,
        ),
    }
    torch.cuda.synchronize()

    policies = ("baseline_nonclosing_first", "candidate_closing_first")
    release_modes = ("streaming", "full_layer_barrier")
    if (
        capability_cfg.get("persistent_sender_local_cuda_stream_across_all_arms")
        is not True
    ):
        raise CapabilityError("persistent sender-local CUDA stream is not frozen")
    sender_local_stream = torch.cuda.Stream(device=a_siblings.device)
    sender_local_stream_id = int(sender_local_stream.cuda_stream)
    if sender_local_stream_id <= 0:
        raise CapabilityError("sender-local CUDA stream must be non-default")
    for warmup in range(args.warmups):
        for execution_index, (release_mode, policy) in enumerate(
            capability_execution_order(config, warmup)
        ):
            _run_trial(
                policy=policy,
                release_mode=release_mode,
                a_siblings=a_siblings,
                x_closing=x_closing,
                x_closing_slots=x_closing_slots,
                y_nonclosing=y_nonclosing,
                task_fixtures=task_fixtures,
                trial=-(warmup + 1),
                execution_order_index=execution_index,
                stream=sender_local_stream,
            )
    rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    for trial in range(args.trials):
        for execution_index, (release_mode, policy) in enumerate(
            capability_execution_order(config, trial)
        ):
            metrics, output_hash, trial_actions = _run_trial(
                policy=policy,
                release_mode=release_mode,
                a_siblings=a_siblings,
                x_closing=x_closing,
                x_closing_slots=x_closing_slots,
                y_nonclosing=y_nonclosing,
                task_fixtures=task_fixtures,
                trial=trial,
                execution_order_index=execution_index,
                stream=sender_local_stream,
            )
            action_rows.extend(trial_actions)
            group_ids = {row["action_trace_group_id"] for row in trial_actions}
            snapshots = {row["queue_snapshot_sha256"] for row in trial_actions}
            stream_ids = {row["stream_id"] for row in trial_actions}
            if len(group_ids) != 1 or len(snapshots) != 1 or len(stream_ids) != 1:
                raise CapabilityError("per-trial capability action trace drift")
            rows.append(
                {
                        "trial": trial,
                        "model_key": args.model_key,
                        "model_revision": model_revision,
                        "request_id": request_id,
                        "layer_id": target_layer,
                        "sender_rank": sender_rank,
                        "receiver_rank": receiver_rank,
                        "block_rows": args.block_rows,
                        "top_k": int(spec["top_k"]),
                        "policy": policy,
                        "release_mode": release_mode,
                        "execution_order_index": execution_index,
                        "service_order": (
                            "x_closing,y_nonclosing"
                            if policy == "candidate_closing_first"
                            else "y_nonclosing,x_closing"
                        ),
                        "action_trace_group_id": next(iter(group_ids)),
                        "queue_snapshot_sha256": next(iter(snapshots)),
                        "stream_id": next(iter(stream_ids)),
                        **metrics,
                        "canonical_output_sha256": output_hash,
                        "canonical_reference_sha256": reference_hash,
                        "canonical_equal": output_hash == reference_hash,
                        "source": "measured_5090_cuda",
                }
            )
    if not all(bool(row["canonical_equal"]) for row in rows):
        raise CapabilityError("canonical output changed with service order")
    if {int(row["stream_id"]) for row in rows} != {sender_local_stream_id}:
        raise CapabilityError("capability arms did not reuse one sender-local CUDA stream")

    def selected(policy: str, release_mode: str) -> list[dict[str, Any]]:
        return [
            row
            for row in rows
            if row["policy"] == policy and row["release_mode"] == release_mode
        ]

    streaming_baseline = selected("baseline_nonclosing_first", "streaming")
    streaming_candidate = selected("candidate_closing_first", "streaming")
    barrier_baseline = selected("baseline_nonclosing_first", "full_layer_barrier")
    barrier_candidate = selected("candidate_closing_first", "full_layer_barrier")
    by_key = {
        (int(row["trial"]), str(row["policy"]), str(row["release_mode"])): row
        for row in rows
    }
    frontier_effects = [
        float(by_key[(trial, "baseline_nonclosing_first", "streaming")]["application_release_us"])
        - float(by_key[(trial, "candidate_closing_first", "streaming")]["application_release_us"])
        for trial in range(args.trials)
    ]
    downstream_effects = [
        float(by_key[(trial, "baseline_nonclosing_first", "streaming")]["downstream_start_us"])
        - float(by_key[(trial, "candidate_closing_first", "streaming")]["downstream_start_us"])
        for trial in range(args.trials)
    ]
    barrier_frontier_effects = [
        float(
            by_key[
                (trial, "baseline_nonclosing_first", "full_layer_barrier")
            ]["application_release_us"]
        )
        - float(
            by_key[
                (trial, "candidate_closing_first", "full_layer_barrier")
            ]["application_release_us"]
        )
        for trial in range(args.trials)
    ]
    barrier_downstream_effects = [
        float(
            by_key[
                (trial, "baseline_nonclosing_first", "full_layer_barrier")
            ]["downstream_start_us"]
        )
        - float(
            by_key[
                (trial, "candidate_closing_first", "full_layer_barrier")
            ]["downstream_start_us"]
        )
        for trial in range(args.trials)
    ]
    release_interactions = [
        streaming - barrier
        for streaming, barrier in zip(frontier_effects, barrier_frontier_effects)
    ]
    downstream_interactions = [
        streaming - barrier
        for streaming, barrier in zip(
            downstream_effects, barrier_downstream_effects
        )
    ]
    existence = capability_cfg["paired_existence_gate"]
    lcbs = paired_effect_lcbs(
        {
            "frontier": frontier_effects,
            "downstream": downstream_effects,
            "release_interaction": release_interactions,
            "downstream_interaction": downstream_interactions,
        },
        replicates=int(existence["bootstrap_replicates"]),
        order_statistic_one_based=int(
            existence["lcb_order_statistic_one_based"]
        ),
        seed=int(existence["bootstrap_seed"]),
    )
    precedence_failures = event_precedence_failures(
        rows, action_rows, trials=args.trials
    )
    summary = {
        "streaming_frontier_advance_us": _median(
            streaming_baseline, "application_release_us"
        )
        - _median(streaming_candidate, "application_release_us"),
        "streaming_downstream_advance_us": _median(
            streaming_baseline, "downstream_start_us"
        )
        - _median(streaming_candidate, "downstream_start_us"),
        "barrier_application_release_difference_us": _median(
            barrier_baseline, "application_release_us"
        )
        - _median(barrier_candidate, "application_release_us"),
        "barrier_downstream_start_difference_us": _median(
            barrier_baseline, "downstream_start_us"
        )
        - _median(barrier_candidate, "downstream_start_us"),
        "baseline_streaming_release_median_us": _median(
            streaming_baseline, "application_release_us"
        ),
        "candidate_streaming_release_median_us": _median(
            streaming_candidate, "application_release_us"
        ),
        "baseline_barrier_release_median_us": _median(
            barrier_baseline, "application_release_us"
        ),
        "candidate_barrier_release_median_us": _median(
            barrier_candidate, "application_release_us"
        ),
        "baseline_barrier_total_median_us": _median(barrier_baseline, "total_us"),
        "streaming_frontier_paired_mean_us": statistics.fmean(frontier_effects),
        "streaming_frontier_paired_lcb_us": lcbs["frontier"],
        "streaming_downstream_paired_mean_us": statistics.fmean(downstream_effects),
        "streaming_downstream_paired_lcb_us": lcbs["downstream"],
        "release_interaction_paired_mean_us": statistics.fmean(
            release_interactions
        ),
        "release_interaction_paired_lcb_us": lcbs["release_interaction"],
        "downstream_interaction_paired_mean_us": statistics.fmean(
            downstream_interactions
        ),
        "downstream_interaction_paired_lcb_us": lcbs[
            "downstream_interaction"
        ],
        "paired_bootstrap_replicates": int(existence["bootstrap_replicates"]),
        "paired_bootstrap_seed": int(existence["bootstrap_seed"]),
        "paired_one_sided_confidence": float(existence["one_sided_confidence"]),
        "paired_lcb_order_statistic_one_based": int(
            existence["lcb_order_statistic_one_based"]
        ),
        "barrier_cross_policy_timing_diagnostic_only": True,
        "barrier_application_release_paired_mean_us_diagnostic": statistics.fmean(
            barrier_frontier_effects
        ),
        "barrier_application_release_max_abs_paired_difference_us_diagnostic": max(
            abs(value) for value in barrier_frontier_effects
        ),
        "barrier_downstream_paired_mean_us_diagnostic": statistics.fmean(
            barrier_downstream_effects
        ),
        "barrier_downstream_max_abs_paired_difference_us_diagnostic": max(
            abs(value) for value in barrier_downstream_effects
        ),
        "event_precedence_all_trials_pass": not precedence_failures,
        "event_precedence_failure_count": len(precedence_failures),
        "event_precedence_failures": precedence_failures,
    }

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{args.output_dir.name}.partial-", dir=args.output_dir.parent
    ) as temporary_directory:
        temporary = Path(temporary_directory)
        raw_path = temporary / "capability_raw.csv"
        _write_csv(raw_path, rows)
        action_trace_path = temporary / "capability_action_trace.jsonl"
        _write_jsonl(action_trace_path, action_rows)
        profiler_policy = "candidate_closing_first"
        profiler_diagnostics: dict[str, dict[str, Any]] = {}
        for profiler_release, profiler_trial in (
            ("streaming", -1_000_000),
            ("full_layer_barrier", -1_000_001),
        ):
            profiler_prefix = (
                f"ric_capability/trial={profiler_trial}/policy={profiler_policy}/"
                f"release={profiler_release}"
            )
            profiler_required_labels = [
                profiler_prefix,
                *(
                    f"{profiler_prefix}/enqueue={name}"
                    for name in ("x_closing", "y_nonclosing")
                ),
                f"{profiler_prefix}/queue_snapshot=both_ready",
                *(
                    f"{profiler_prefix}/select={name}"
                    for name in ("x_closing", "y_nonclosing")
                ),
                *(
                    f"{profiler_prefix}/service={name}"
                    for name in ("x_closing", "y_nonclosing")
                ),
            ]
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
                profile_memory=False,
                with_stack=False,
            ) as profiler:
                _metrics, profiler_output_hash, profiler_actions = _run_trial(
                    policy=profiler_policy,
                    release_mode=profiler_release,
                    a_siblings=a_siblings,
                    x_closing=x_closing,
                    x_closing_slots=x_closing_slots,
                    y_nonclosing=y_nonclosing,
                    task_fixtures=task_fixtures,
                    trial=profiler_trial,
                    execution_order_index=0,
                    stream=sender_local_stream,
                )
                torch.cuda.synchronize()
            if (
                profiler_output_hash != reference_hash
                or {int(row["stream_id"]) for row in profiler_actions}
                != {sender_local_stream_id}
            ):
                raise CapabilityError("profiler diagnostic drifted from reviewed path")
            profiler_file = f"capability_cuda_trace_{profiler_release}.json"
            profiler_path = temporary / profiler_file
            profiler.export_chrome_trace(str(profiler_path))
            gpu_stream_ordinal = profiler_gpu_stream_ordinal(profiler_path)
            profiler_diagnostics[profiler_release] = {
                "trial": profiler_trial,
                "policy": profiler_policy,
                "release_mode": profiler_release,
                "trace_file": profiler_file,
                "required_labels": profiler_required_labels,
                "trace_sha256": sha256_file(profiler_path),
                "sender_local_stream_id": sender_local_stream_id,
                "gpu_activity_stream_ordinal": gpu_stream_ordinal,
                "canonical_output_sha256": profiler_output_hash,
            }
        if (
            len(
                {
                    diagnostic["gpu_activity_stream_ordinal"]
                    for diagnostic in profiler_diagnostics.values()
                }
            )
            != 1
        ):
            raise CapabilityError("profiler release modes used different GPU streams")
        payload_path = temporary / "expert_contributions.pt"
        torch.save(
            {
                "a_siblings": a_siblings.detach().cpu(),
                "b_siblings": b_siblings.detach().cpu(),
                "selected_experts": torch.cat(
                    (
                        selected_experts.index_select(0, x_token_indices),
                        selected_experts.index_select(0, y_token_indices),
                    ),
                    dim=0,
                )
                .detach()
                .cpu(),
                "route_weights": torch.cat(
                    (
                        route_weights.index_select(0, x_token_indices),
                        route_weights.index_select(0, y_token_indices),
                    ),
                    dim=0,
                )
                .detach()
                .cpu(),
                "sender_local_block_plan": plan,
                "request_id": request_id,
                "layer_id": target_layer,
                "model_revision": model_revision,
            },
            payload_path,
        )
        embedded_signoff_sha256 = (
            materialize_verified_signoff(args.signoff, temporary)
            if signoff is not None
            else None
        )
        artifact = add_self_hash(
            {
                "schema_version": "ric-capability-v1",
                "status": "CAPABILITY_ONLY" if args.mode == "formal" else "NOT_TESTED",
                "scientific_result": False,
                "evidence_boundary": (
                    "REAL_5090_EXPERT_OUTPUT_AND_LOCAL_CUDA_STREAM / NOT_NCCL / NOT_RDMA"
                ),
                "mode": args.mode,
                "model_key": args.model_key,
                "model_revision": model_revision,
                "transformers_version": transformers_version,
                "model_source": model_source,
                "model_tree_manifest_sha256": model_tree_sha,
                "request_id": request_id,
                "target_layer": target_layer,
                "frozen_selected_layers": frozen_layers,
                "block_rows": args.block_rows,
                "top_k": int(spec["top_k"]),
                "warmups": args.warmups,
                "trials": args.trials,
                "ready_result_orders": {
                    "baseline": ["y_nonclosing", "x_closing"],
                    "candidate": ["x_closing", "y_nonclosing"],
                },
                "execution_order_rule": EXECUTION_ORDER_RULE,
                "persistent_sender_local_cuda_stream_across_all_arms": True,
                "sender_local_stream_id": sender_local_stream_id,
                "sender_local_queue_id": queue_id,
                "sender_rank": sender_rank,
                "receiver_rank": receiver_rank,
                "ep_size": ep_size,
                "ranks_per_node": ranks_per_node,
                "num_experts": int(spec["num_experts"]),
                "sender_local_selection": plan,
                "task_fixtures": task_fixtures,
                "queue_snapshot_ready_count": 2,
                "action_trace_schema_version": "ric-capability-action-v1",
                "action_trace_row_count": len(action_rows),
                "capability_action_trace_sha256": sha256_file(action_trace_path),
                "nvtx_ranges_emitted": True,
                "action_trace_evidence_boundary": (
                    "CUDA_EVENT_ACTION_TRACE_WITH_EMITTED_NVTX_LABELS"
                ),
                "profiler_diagnostic_not_in_timing_trials": True,
                "profiler_trace_kind": "torch_profiler_chrome_trace_cpu_cuda",
                "profiler_diagnostics": profiler_diagnostics,
                "release_modes": list(release_modes),
                "canonical_reference_sha256": reference_hash,
                "all_canonical_hashes_equal": True,
                "summary": summary,
                "raw_trials_sha256": sha256_file(raw_path),
                "expert_contributions_sha256": sha256_file(payload_path),
                "expert_contributions_source": "native_unpatched_model_expert_execution",
                "protocol_sha256": protocol_sha,
                "config_sha256": config_sha,
                "data_manifest_sha256": manifest_sha,
                "data_producer_signoff_sha256": data_producer_signoff_sha,
                "measure_capability_source_sha256": source_sha,
                "gpu_environment": _gpu_environment(
                    compute_apps_before=compute_apps_before
                ),
                "signoff_sha256": embedded_signoff_sha256,
            }
        )
        (temporary / "capability_probe.json").write_text(
            json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.rename(args.output_dir)
    print(json.dumps(artifact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
