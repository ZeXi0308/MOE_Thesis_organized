#!/usr/bin/env python3
"""Run the frozen RR-credit multi-rank receiver census under torchrun.

The default process group carries CUDA payloads with NCCL.  A separate Gloo
group carries one-word receiver credits.  The runtime emits rank-local event
halves and a merged logical trace, but deliberately marks the result as
awaiting Nsight/CUPTI binding.  Rank-local timestamps alone are not accepted as
the formal cross-rank physical timeline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    from .smoke_multirank_trace_gpu import _object_sha256, validate_plan
except ImportError:  # Direct script execution under torchrun.
    from smoke_multirank_trace_gpu import _object_sha256, validate_plan


SCHEMA_VERSION = "multirank-rr-census-rank-trace-v1"
MERGED_SCHEMA_VERSION = "multirank-rr-census-merged-trace-v1"
FORMAL_WORLD_SIZES = {4, 8}


class MultiRankRuntimeError(RuntimeError):
    pass


def physical_rank(virtual_rank: int, world_size: int) -> int:
    if world_size not in FORMAL_WORLD_SIZES:
        raise MultiRankRuntimeError("physical runtime requires world_size 4 or 8")
    return int(virtual_rank) % world_size


def rr_order(contributions: Sequence[Mapping[str, Any]], world_size: int) -> list[dict[str, Any]]:
    queues: dict[int, deque[dict[str, Any]]] = defaultdict(deque)
    for contribution in sorted(contributions, key=lambda row: int(row["topk_slot"])):
        value = dict(contribution)
        value["physical_sender_rank"] = physical_rank(value["sender_rank"], world_size)
        value["physical_receiver_rank"] = physical_rank(value["receiver_rank"], world_size)
        queues[value["physical_sender_rank"]].append(value)
    order: list[dict[str, Any]] = []
    active = sorted(queues)
    while active:
        next_active = []
        for sender in active:
            order.append(queues[sender].popleft())
            if queues[sender]:
                next_active.append(sender)
        active = next_active
    return order


def execution_message_id(
    route_message_id: str, *, credit_b: int, trial_kind: str, trial_index: int
) -> str:
    return _object_sha256(
        {
            "route_message_id": route_message_id,
            "credit_b": credit_b,
            "trial_kind": trial_kind,
            "trial_index": trial_index,
        }
    )


def merge_event_halves(
    plan: Mapping[str, Any], rank_records: Iterable[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    expected_routes = {
        row["message_id"]: row
        for wave in plan["waves"]
        for row in wave["contributions"]
    }
    halves: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for value in rank_records:
        message_id = value.get("message_id")
        side = value.get("event_side")
        if not isinstance(message_id, str) or side not in {"sender", "receiver", "local"}:
            raise MultiRankRuntimeError("invalid rank event half")
        if side in halves[message_id]:
            raise MultiRankRuntimeError(f"duplicate {side} event for {message_id}")
        halves[message_id][side] = value
    merged = []
    for message_id in sorted(halves):
        sides = halves[message_id]
        if "local" in sides:
            if len(sides) != 1:
                raise MultiRankRuntimeError("local event cannot also have sender/receiver halves")
            row = dict(sides["local"])
        else:
            if set(sides) != {"sender", "receiver"}:
                raise MultiRankRuntimeError(f"unmatched message halves for {message_id}")
            sender = sides["sender"]
            receiver = sides["receiver"]
            identity_fields = (
                "run_id",
                "wave_id",
                "route_message_id",
                "message_id",
                "credit_b",
                "trial_kind",
                "trial_index",
                "payload_bytes",
                "physical_sender_rank",
                "physical_receiver_rank",
            )
            if any(sender.get(name) != receiver.get(name) for name in identity_fields):
                raise MultiRankRuntimeError("sender/receiver identity mismatch")
            row = {**sender, **receiver}
        route = expected_routes.get(row.get("route_message_id"))
        if route is None or int(row["payload_bytes"]) != int(route["payload_bytes"]):
            raise MultiRankRuntimeError("merged event does not bind to frozen payload")
        wave = next(value for value in plan["waves"] if value["wave_id"] == row["wave_id"])
        for name in ("request_id", "layer_id", "token_position"):
            row.setdefault(name, wave[name])
        for name in ("topk_slot", "expert_id", "sender_rank", "receiver_rank"):
            row.setdefault(name, route[name])
        row.pop("event_side", None)
        merged.append(row)
    grouped: dict[tuple[str, int, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in merged:
        required = (
            "expert_start_ns",
            "expert_ready_ns",
            "credit_recv_ns",
            "send_start_ns",
            "send_end_ns",
            "recv_visible_ns",
            "unpack_start_ns",
            "unpack_end_ns",
            "join_close_ns",
        )
        timestamps = {name: int(row[name]) for name in required}
        sender_order = (
            timestamps["expert_start_ns"]
            <= timestamps["expert_ready_ns"]
            <= timestamps["credit_recv_ns"]
            <= timestamps["send_start_ns"]
            <= timestamps["send_end_ns"]
        )
        receiver_order = (
            timestamps["send_start_ns"]
            <= timestamps["recv_visible_ns"]
            <= timestamps["unpack_start_ns"]
            <= timestamps["unpack_end_ns"]
            <= timestamps["join_close_ns"]
        )
        if not sender_order or not receiver_order:
            raise MultiRankRuntimeError("merged event chronology violation")
        grouped[
            (row["wave_id"], int(row["credit_b"]), row["trial_kind"], int(row["trial_index"]))
        ].append(row)
    wave_by_id = {wave["wave_id"]: wave for wave in plan["waves"]}
    for (wave_id, _credit_b, _kind, _index), rows in grouped.items():
        wave = wave_by_id[wave_id]
        if len(rows) != int(wave["top_k"]):
            raise MultiRankRuntimeError("join contribution conservation failed")
        close = max(int(row["unpack_end_ns"]) for row in rows)
        if any(int(row["join_close_ns"]) != close for row in rows):
            raise MultiRankRuntimeError("join_close does not equal last unpack")
    return merged


def _write_jsonl_once(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    if path.exists() or path.is_symlink():
        raise MultiRankRuntimeError(f"refusing to overwrite {path}")
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    path.chmod(0o444)


def _write_json_once(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise MultiRankRuntimeError(f"refusing to overwrite {path}")
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, allow_nan=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    path.chmod(0o444)


def _topology() -> str:
    try:
        return subprocess.run(
            ["nvidia-smi", "topo", "-m"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return "UNAVAILABLE"


def _run(args: argparse.Namespace) -> int:
    import torch
    import torch.distributed as dist
    import torch.nn.functional as functional

    if not dist.is_available() or not dist.is_nccl_available() or not dist.is_gloo_available():
        raise MultiRankRuntimeError("NCCL and Gloo process groups are required")
    rank = int(os.environ.get("RANK", "-1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    world_size = int(os.environ.get("WORLD_SIZE", "-1"))
    if world_size not in FORMAL_WORLD_SIZES:
        raise MultiRankRuntimeError("torchrun world_size must be 4 or 8")
    if torch.cuda.device_count() < world_size:
        raise MultiRankRuntimeError("one distinct visible CUDA device per rank is required")
    if not (0 <= local_rank < torch.cuda.device_count()):
        raise MultiRankRuntimeError("invalid LOCAL_RANK")
    plan = json.loads(args.plan.read_text())
    validate_plan(plan)
    if len(plan["waves"]) != 64:
        raise MultiRankRuntimeError("formal census requires exactly 64 frozen waves")
    output = args.output.resolve()
    if rank in {-1} or local_rank == -1:
        raise MultiRankRuntimeError("must be launched with torchrun")
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    control = dist.new_group(backend="gloo")
    if rank == 0:
        if output.exists() or output.is_symlink():
            raise MultiRankRuntimeError(f"refusing to overwrite {output}")
        output.mkdir(parents=True)
    dist.barrier(group=control)
    device = torch.device(f"cuda:{local_rank}")
    hidden = int(plan["waves"][0]["hidden"])
    intermediate = int(plan["waves"][0]["intermediate"])
    generator = torch.Generator(device=device).manual_seed(20260723 + rank)
    input_row = torch.randn((1, hidden), dtype=torch.bfloat16, device=device, generator=generator)
    weight_in = torch.randn(
        (hidden, intermediate), dtype=torch.bfloat16, device=device, generator=generator
    )
    weight_out = torch.randn(
        (intermediate, hidden), dtype=torch.bfloat16, device=device, generator=generator
    )
    torch.cuda.synchronize()
    run_id = _object_sha256(
        {
            "plan": plan["artifact_sha256"],
            "world_size": world_size,
            "run_nonce": args.run_nonce,
        }
    )
    records: list[dict[str, Any]] = []
    stream_id = int(torch.cuda.current_stream(device).cuda_stream)
    cells = [("warmup", index) for index in range(args.warmups)] + [
        ("measured", index) for index in range(args.measured)
    ]
    for credit_b in (1, 2, 4, 8):
        for trial_kind, trial_index in cells:
            wave = plan["waves"][trial_index % len(plan["waves"])]
            order = rr_order(wave["contributions"], world_size)
            receiver = physical_rank(wave["receiver_rank"], world_size)
            # Trial reset only.  It prevents cross-trial residue; it is part of
            # the controlled microbenchmark and must not be interpreted as a
            # production arrival process.
            dist.barrier(group=control)
            owned: dict[str, tuple[dict[str, Any], Any, int, int]] = {}
            for contribution in order:
                if contribution["physical_sender_rank"] != rank:
                    continue
                label = contribution["message_id"][:16]
                torch.cuda.nvtx.range_push(f"EXPERT:{plan['model']}:{label}")
                expert_start = time.perf_counter_ns()
                payload = torch.matmul(
                    functional.silu(torch.matmul(input_row, weight_in)), weight_out
                )
                torch.cuda.synchronize()
                expert_ready = time.perf_counter_ns()
                torch.cuda.nvtx.range_pop()
                owned[contribution["message_id"]] = (
                    contribution,
                    payload,
                    expert_start,
                    expert_ready,
                )
            if rank == receiver:
                receiver_halves: list[dict[str, Any]] = []
                # Local and remote contributions share the same RR order and
                # consume the same B-sized admission window.  Pulling local
                # rows ahead would silently change the baseline.
                for start in range(0, len(order), credit_b):
                    window = order[start : start + credit_b]
                    posted: dict[str, tuple[Any, Any]] = {}
                    for contribution in window:
                        if contribution["physical_sender_rank"] == receiver:
                            continue
                        buffer = torch.empty((1, hidden), dtype=torch.bfloat16, device=device)
                        work = dist.irecv(buffer, src=contribution["physical_sender_rank"])
                        posted[contribution["message_id"]] = (buffer, work)
                    for contribution in window:
                        if contribution["physical_sender_rank"] == receiver:
                            continue
                        token = torch.tensor([1], dtype=torch.int64)
                        dist.send(token, dst=contribution["physical_sender_rank"], group=control)
                    for contribution in window:
                        if contribution["physical_sender_rank"] == receiver:
                            value, payload, expert_start, expert_ready = owned[
                                contribution["message_id"]
                            ]
                            message_id = execution_message_id(
                                value["message_id"],
                                credit_b=credit_b,
                                trial_kind=trial_kind,
                                trial_index=trial_index,
                            )
                            send_start = time.perf_counter_ns()
                            received = payload.clone()
                            torch.cuda.synchronize()
                            send_end = time.perf_counter_ns()
                            unpack_start = time.perf_counter_ns()
                            _ = received.mul(float(value["route_weight"])).sum()
                            torch.cuda.synchronize()
                            unpack_end = time.perf_counter_ns()
                            receiver_halves.append(
                                {
                                    "event_side": "local",
                                    "run_id": run_id,
                                    "wave_id": wave["wave_id"],
                                    "request_id": wave["request_id"],
                                    "layer_id": wave["layer_id"],
                                    "token_position": wave["token_position"],
                                    "topk_slot": value["topk_slot"],
                                    "expert_id": value["expert_id"],
                                    "sender_rank": value["sender_rank"],
                                    "receiver_rank": value["receiver_rank"],
                                    "physical_sender_rank": receiver,
                                    "physical_receiver_rank": receiver,
                                    "payload_bytes": value["payload_bytes"],
                                    "route_message_id": value["message_id"],
                                    "message_id": message_id,
                                    "credit_b": credit_b,
                                    "trial_kind": trial_kind,
                                    "trial_index": trial_index,
                                    "expert_start_ns": expert_start,
                                    "expert_ready_ns": expert_ready,
                                    "credit_recv_ns": expert_ready,
                                    "send_start_ns": send_start,
                                    "send_end_ns": send_end,
                                    "recv_visible_ns": send_end,
                                    "unpack_start_ns": unpack_start,
                                    "unpack_end_ns": unpack_end,
                                    "join_close_ns": 0,
                                    "stream_id": stream_id,
                                    "transport": "LOCAL_SAME_RANK",
                                }
                            )
                            continue
                        buffer, work = posted[contribution["message_id"]]
                        work.wait()
                        torch.cuda.synchronize()
                        recv_visible = time.perf_counter_ns()
                        label = contribution["message_id"][:16]
                        torch.cuda.nvtx.range_push(f"UNPACK:{plan['model']}:{label}")
                        unpack_start = time.perf_counter_ns()
                        _ = buffer.mul(float(contribution["route_weight"])).sum()
                        torch.cuda.synchronize()
                        unpack_end = time.perf_counter_ns()
                        torch.cuda.nvtx.range_pop()
                        receiver_halves.append(
                            {
                                "event_side": "receiver",
                                "run_id": run_id,
                                "wave_id": wave["wave_id"],
                                "route_message_id": contribution["message_id"],
                                "message_id": execution_message_id(
                                    contribution["message_id"],
                                    credit_b=credit_b,
                                    trial_kind=trial_kind,
                                    trial_index=trial_index,
                                ),
                                "credit_b": credit_b,
                                "trial_kind": trial_kind,
                                "trial_index": trial_index,
                                "payload_bytes": contribution["payload_bytes"],
                                "physical_sender_rank": contribution["physical_sender_rank"],
                                "physical_receiver_rank": receiver,
                                "recv_visible_ns": recv_visible,
                                "unpack_start_ns": unpack_start,
                                "unpack_end_ns": unpack_end,
                                "join_close_ns": 0,
                            }
                        )
                join_close = max(int(value["unpack_end_ns"]) for value in receiver_halves)
                for value in receiver_halves:
                    value["join_close_ns"] = join_close
                if trial_kind == "measured":
                    records.extend(receiver_halves)
            else:
                for contribution in order:
                    if contribution["physical_sender_rank"] != rank:
                        continue
                    value, payload, expert_start, expert_ready = owned[contribution["message_id"]]
                    token = torch.empty((1,), dtype=torch.int64)
                    dist.recv(token, src=receiver, group=control)
                    credit_recv = time.perf_counter_ns()
                    label = value["message_id"][:16]
                    torch.cuda.nvtx.range_push(f"SEND:{plan['model']}:{label}")
                    send_start = time.perf_counter_ns()
                    work = dist.isend(payload, dst=receiver)
                    work.wait()
                    torch.cuda.synchronize()
                    send_end = time.perf_counter_ns()
                    torch.cuda.nvtx.range_pop()
                    if trial_kind == "measured":
                        records.append(
                            {
                                "event_side": "sender",
                                "run_id": run_id,
                                "wave_id": wave["wave_id"],
                                "route_message_id": value["message_id"],
                                "message_id": execution_message_id(
                                    value["message_id"],
                                    credit_b=credit_b,
                                    trial_kind=trial_kind,
                                    trial_index=trial_index,
                                ),
                                "credit_b": credit_b,
                                "trial_kind": trial_kind,
                                "trial_index": trial_index,
                                "payload_bytes": value["payload_bytes"],
                                "physical_sender_rank": rank,
                                "physical_receiver_rank": receiver,
                                "expert_start_ns": expert_start,
                                "expert_ready_ns": expert_ready,
                                "credit_recv_ns": credit_recv,
                                "send_start_ns": send_start,
                                "send_end_ns": send_end,
                                "stream_id": stream_id,
                                "transport": "NCCL_P2P",
                            }
                        )
            dist.barrier(group=control)
    rank_path = output / f"rank_{rank:02d}.jsonl"
    _write_jsonl_once(rank_path, records)
    dist.barrier(group=control)
    if rank == 0:
        all_records = []
        for value in range(world_size):
            with (output / f"rank_{value:02d}.jsonl").open() as handle:
                all_records.extend(json.loads(line) for line in handle if line.strip())
        merged = merge_event_halves(plan, all_records)
        _write_jsonl_once(output / "merged_trace.jsonl", merged)
        manifest = {
            "schema_version": MERGED_SCHEMA_VERSION,
            "status": "RAW_TRACE_AWAITING_NSYS_CUPTI_BINDING_NOT_HEADROOM_RESULT",
            "scientific_result": False,
            "evidence_boundary": "REAL_MULTI_GPU_NCCL_P2P_RAW_EVENTS_RANK_LOCAL_CLOCKS_NOT_YET_CROSS_RANK_GPU_TIMELINE",
            "run_id": run_id,
            "plan_artifact_sha256": plan["artifact_sha256"],
            "model": plan["model"],
            "world_size": world_size,
            "virtual_rank_mapping": "identity" if world_size == 8 else "virtual_rank_mod_4",
            "credit_b_values": [1, 2, 4, 8],
            "warmups_per_cell": args.warmups,
            "measured_per_cell": args.measured,
            "merged_record_count": len(merged),
            "backend_payload": "nccl",
            "backend_credit": "gloo",
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "nccl_version": torch.cuda.nccl.version(),
            "platform": platform.platform(),
            "topology": _topology(),
            "nsys_report_bound": False,
            "physical_headroom_gate_evaluated": False,
        }
        manifest["artifact_sha256"] = _object_sha256(manifest)
        _write_json_once(output / "manifest.json", manifest)
    dist.barrier(group=control)
    dist.destroy_process_group(control)
    dist.destroy_process_group()
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-nonce", required=True)
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--measured", type=int, default=100)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.warmups < 1 or args.measured < 1:
        print(json.dumps({"status": "BLOCKED", "error": "counts must be positive"}))
        return 2
    try:
        return _run(args)
    except (OSError, ValueError, RuntimeError, MultiRankRuntimeError) as exc:
        print(json.dumps({"status": "BLOCKED", "error": str(exc)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
